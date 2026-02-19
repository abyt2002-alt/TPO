import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
import os
import sqlite3
import json
import uuid
import copy
import re
from datetime import datetime, date
import urllib.request
import urllib.error

from models.rfm_models import (
    RFMRequest, RFMResponse, OutletRFM, 
    SegmentSummary, ClusterSummary,
    BaseDepthRequest, BaseDepthResponse, BaseDepthPoint,
    DiscountOptionsRequest, DiscountOptionsResponse,
    ModelingRequest, ModelingResponse, ModelingSlabResult, ModelingPoint,
    PlannerRequest, PlannerResponse, PlannerMonthPoint
)


class CustomConstrainedRidge:
    """Ridge regression with sign-constrained coefficients."""

    def __init__(self, l2_penalty=1.0, non_negative_indices=None, non_positive_indices=None, maxiter=2000):
        self.l2_penalty = float(l2_penalty)
        self.non_negative_indices = tuple(non_negative_indices or [])
        self.non_positive_indices = tuple(non_positive_indices or [])
        self.maxiter = int(maxiter)
        self.intercept_ = 0.0
        self.coef_ = None
        self.n_features_in_ = 0
        self.success_ = False
        self.message_ = ""

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, p = X.shape
        self.n_features_in_ = p

        x_mean = X.mean(axis=0)
        x_std = X.std(axis=0)
        x_std = np.where(x_std <= 1e-12, 1.0, x_std)
        Xs = (X - x_mean) / x_std

        y_mean = float(np.mean(y))
        theta0 = np.zeros(p + 1, dtype=float)
        theta0[0] = y_mean

        bounds = [(None, None)] * (p + 1)
        for idx in self.non_negative_indices:
            if 0 <= idx < p:
                bounds[idx + 1] = (0.0, None)
        for idx in self.non_positive_indices:
            if 0 <= idx < p:
                bounds[idx + 1] = (None, 0.0)

        lam = self.l2_penalty

        def obj(theta):
            b0 = theta[0]
            w = theta[1:]
            resid = y - (Xs @ w + b0)
            return float(np.mean(resid * resid) + lam * np.sum(w * w))

        def grad(theta):
            b0 = theta[0]
            w = theta[1:]
            resid = (Xs @ w + b0) - y
            g_b0 = 2.0 * np.mean(resid)
            g_w = 2.0 * (Xs.T @ resid) / n + 2.0 * lam * w
            return np.concatenate([[g_b0], g_w])

        res = minimize(
            obj,
            theta0,
            jac=grad,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': self.maxiter}
        )
        self.success_ = bool(res.success)
        self.message_ = str(res.message)

        theta = np.asarray(res.x, dtype=float)
        b0_s = float(theta[0])
        w_s = theta[1:]

        coef = w_s / x_std
        intercept = b0_s - float(np.sum((w_s * x_mean) / x_std))

        if self.non_negative_indices:
            coef[list(self.non_negative_indices)] = np.maximum(coef[list(self.non_negative_indices)], 0.0)
        if self.non_positive_indices:
            coef[list(self.non_positive_indices)] = np.minimum(coef[list(self.non_positive_indices)], 0.0)

        self.intercept_ = float(intercept)
        self.coef_ = coef.astype(float)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercept_

class RFMService:
    def __init__(self):
        self.data_cache = None
        self.rfm_result_cache = {}
        self.max_cache_entries = 5
        self.state_db_path = Path(__file__).resolve().parent.parent / "analysis_state.db"
        self._init_state_db()
        self.load_data()

    def _init_state_db(self):
        try:
            with sqlite3.connect(self.state_db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS analysis_runs (
                        run_id TEXT PRIMARY KEY,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS analysis_state (
                        run_id TEXT PRIMARY KEY,
                        state_json TEXT NOT NULL,
                        step1_result_json TEXT,
                        step2_result_json TEXT,
                        updated_at TEXT NOT NULL,
                        FOREIGN KEY(run_id) REFERENCES analysis_runs(run_id) ON DELETE CASCADE
                    )
                """)
                conn.commit()
        except Exception as e:
            print(f"Error initializing run-state DB: {e}")

    def _utc_now_iso(self) -> str:
        return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    def _default_run_state(self) -> Dict:
        return {
            "active_step": "step1",
            "filters": {
                "states": [],
                "categories": [],
                "subcategories": [],
                "brands": [],
                "sizes": [],
                "recency_threshold": 90,
                "frequency_threshold": 20,
            },
            "table_query": {
                "page": 1,
                "page_size": 20,
                "search": "",
                "sort_key": "total_net_amt",
                "sort_direction": "desc",
            },
            "step2_filters": {
                "rfm_segments": [],
                "outlet_classifications": [],
                "slabs": [],
            },
            "base_depth_config": {
                "time_aggregation": "D",
                "rolling_window_periods": 10,
                "quantile": 0.5,
                "round_step": 0.5,
                "min_upward_jump_pp": 1.0,
                "min_downward_drop_pp": 1.0,
            },
            "last_calculated_filters": None,
            "ui_state": {
                "is_base_depth_config_expanded": True,
            },
        }

    def _json_default(self, value):
        if isinstance(value, (datetime, date, pd.Timestamp)):
            return value.isoformat()
        if isinstance(value, np.generic):
            return value.item()
        return str(value)

    def _safe_json_dumps(self, value) -> str:
        return json.dumps(value, default=self._json_default)

    def _safe_json_loads(self, raw, default_value):
        if not raw:
            return copy.deepcopy(default_value)
        try:
            return json.loads(raw)
        except Exception:
            return copy.deepcopy(default_value)

    def _deep_merge_dict(self, base: Dict, updates: Dict) -> Dict:
        merged = copy.deepcopy(base)
        for key, value in (updates or {}).items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = self._deep_merge_dict(merged[key], value)
            else:
                merged[key] = value
        return merged

    def create_run(self, run_id: str = None) -> str:
        candidate = (run_id or "").strip()
        final_run_id = candidate if candidate else str(uuid.uuid4())
        now_iso = self._utc_now_iso()
        default_state_json = self._safe_json_dumps(self._default_run_state())

        with sqlite3.connect(self.state_db_path) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO analysis_runs (run_id, created_at, updated_at) VALUES (?, ?, ?)",
                (final_run_id, now_iso, now_iso),
            )
            conn.execute(
                "INSERT OR IGNORE INTO analysis_state (run_id, state_json, step1_result_json, step2_result_json, updated_at) VALUES (?, ?, ?, ?, ?)",
                (final_run_id, default_state_json, None, None, now_iso),
            )
            conn.execute(
                "UPDATE analysis_runs SET updated_at = ? WHERE run_id = ?",
                (now_iso, final_run_id),
            )
            conn.commit()

        return final_run_id

    def get_run_state(self, run_id: str):
        if not run_id:
            return None

        with sqlite3.connect(self.state_db_path) as conn:
            row = conn.execute(
                "SELECT state_json, step1_result_json, step2_result_json FROM analysis_state WHERE run_id = ?",
                (run_id,),
            ).fetchone()

        if row is None:
            return None

        default_state = self._default_run_state()
        loaded_state = self._safe_json_loads(row[0], default_state)
        if not isinstance(loaded_state, dict):
            loaded_state = copy.deepcopy(default_state)
        normalized_state = self._deep_merge_dict(default_state, loaded_state)

        return {
            "run_id": run_id,
            "state": normalized_state,
            "step1_result": self._safe_json_loads(row[1], None),
            "step2_result": self._safe_json_loads(row[2], None),
        }

    def save_run_state(self, run_id: str, state_update: Dict = None, step1_result=None, step2_result=None):
        if not run_id:
            return

        final_run_id = self.create_run(run_id)
        existing = self.get_run_state(final_run_id) or {
            "state": self._default_run_state(),
            "step1_result": None,
            "step2_result": None,
        }

        merged_state = self._deep_merge_dict(existing.get("state", self._default_run_state()), state_update or {})
        merged_step1 = existing.get("step1_result") if step1_result is None else step1_result
        merged_step2 = existing.get("step2_result") if step2_result is None else step2_result
        now_iso = self._utc_now_iso()

        with sqlite3.connect(self.state_db_path) as conn:
            conn.execute(
                """
                UPDATE analysis_state
                SET state_json = ?, step1_result_json = ?, step2_result_json = ?, updated_at = ?
                WHERE run_id = ?
                """,
                (
                    self._safe_json_dumps(merged_state),
                    self._safe_json_dumps(merged_step1) if merged_step1 is not None else None,
                    self._safe_json_dumps(merged_step2) if merged_step2 is not None else None,
                    now_iso,
                    final_run_id,
                ),
            )
            conn.execute(
                "UPDATE analysis_runs SET updated_at = ? WHERE run_id = ?",
                (now_iso, final_run_id),
            )
            conn.commit()
    
    def load_data(self):
        """Load parquet files from DATA folder"""
        try:
            script_dir = Path(__file__).resolve().parent.parent
            candidate_paths = [
                script_dir / "DATA",
                Path.cwd() / "DATA",
                script_dir.parent / "DATA"
            ]
            
            folder_path = next((p for p in candidate_paths if p.exists()), None)
            
            if folder_path is None:
                print("Warning: Could not find DATA folder")
                return None
            
            parquet_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.parquet')])
            
            if not parquet_files:
                print("Warning: No parquet files found")
                return None
            
            dfs = []
            for file in parquet_files:
                df = pd.read_parquet(folder_path / file)
                dfs.append(df)
            
            combined_df = pd.concat(dfs, ignore_index=True)
            combined_df['Date'] = pd.to_datetime(combined_df['Date'])
            
            self.data_cache = combined_df
            print(f"Loaded {len(combined_df):,} rows from {len(parquet_files)} files")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            self.data_cache = None

    def _apply_base_filters(self, df: pd.DataFrame, request: RFMRequest) -> pd.DataFrame:
        if request.states:
            df = df[df['Final_State'].isin(request.states)]
        if request.categories:
            df = df[df['Category'].isin(request.categories)]
        if request.subcategories:
            df = df[df['Subcategory'].isin(request.subcategories)]
        if request.brands:
            df = df[df['Brand'].isin(request.brands)]
        if request.sizes:
            df = df[df['Sizes'].isin(request.sizes)]
        return df

    def _base_filter_cache_key(self, request: RFMRequest):
        return (
            tuple(sorted(request.states or [])),
            tuple(sorted(request.categories or [])),
            tuple(sorted(request.subcategories or [])),
            tuple(sorted(request.brands or [])),
            tuple(sorted(request.sizes or [])),
            int(request.recency_threshold),
            int(request.frequency_threshold),
        )

    def _segment_summaries(self, rfm: pd.DataFrame, df: pd.DataFrame) -> List[SegmentSummary]:
        total_sales_value = df['SalesValue_atBasicRate'].sum()
        all_segments = [
            'Recent-High-High', 'Recent-High-Low',
            'Recent-Low-High', 'Recent-Low-Low',
            'Stale-High-High', 'Stale-High-Low',
            'Stale-Low-High', 'Stale-Low-Low'
        ]
        segment_summaries = []

        for seg in all_segments:
            seg_data = rfm[rfm['RFM_Segment'] == seg]
            total_outlets = len(seg_data)
            pct_total = (total_outlets / len(rfm) * 100) if len(rfm) > 0 else 0

            state_breakdown = seg_data['Final_State'].value_counts().to_dict()
            mah_count = state_breakdown.get('MAH', 0)
            up_count = state_breakdown.get('UP', 0)

            avg_order_days = seg_data['unique_order_days'].mean() if len(seg_data) > 0 else 0
            avg_aov = seg_data['AOV'].mean() if len(seg_data) > 0 else 0
            avg_recency = seg_data['Recency_days'].mean() if len(seg_data) > 0 else 0

            seg_outlets = seg_data['Outlet_ID'].tolist()
            seg_sales_value = df[df['Outlet_ID'].isin(seg_outlets)]['SalesValue_atBasicRate'].sum()
            market_share_pct = (seg_sales_value / total_sales_value * 100) if total_sales_value > 0 else 0

            segment_summaries.append(SegmentSummary(
                segment=seg,
                total_outlets=total_outlets,
                percentage=round(pct_total, 2),
                mah_count=mah_count,
                up_count=up_count,
                avg_order_days=round(avg_order_days, 2),
                avg_aov=round(avg_aov, 2),
                avg_recency=round(avg_recency, 2),
                market_share=round(market_share_pct, 2)
            ))

        return segment_summaries

    def _apply_outlet_query(self, rfm: pd.DataFrame, request: RFMRequest):
        queried = rfm.copy()
        search = (request.search or '').strip().lower()
        if search:
            queried = queried[
                queried['Outlet_ID'].astype(str).str.lower().str.contains(search, na=False) |
                queried['Final_State'].astype(str).str.lower().str.contains(search, na=False) |
                queried['RFM_Segment'].astype(str).str.lower().str.contains(search, na=False)
            ]

        sort_key_map = {
            'outlet_id': 'Outlet_ID',
            'final_state': 'Final_State',
            'rfm_segment': 'RFM_Segment',
            'total_net_amt': 'Total_Net_Amt',
            'orders_count': 'orders_count',
            'aov': 'AOV',
            'recency_days': 'Recency_days',
            'unique_order_days': 'unique_order_days',
            'orders_per_day': 'orders_per_day',
        }
        sort_col = sort_key_map.get((request.sort_key or 'total_net_amt').lower(), 'Total_Net_Amt')
        ascending = (request.sort_direction or 'desc').lower() == 'asc'
        queried = queried.sort_values(by=sort_col, ascending=ascending, kind='mergesort')

        total_filtered = len(queried)
        page_size = max(1, int(request.page_size or 20))
        total_pages = max(1, int(np.ceil(total_filtered / page_size))) if total_filtered > 0 else 1
        page = min(max(1, int(request.page or 1)), total_pages)
        start = (page - 1) * page_size
        end = start + page_size
        page_df = queried.iloc[start:end].copy()

        return page_df, total_filtered, total_pages, page, page_size

    def _to_outlet_model(self, row) -> OutletRFM:
        return OutletRFM(
            outlet_id=str(row['Outlet_ID']),
            final_state=str(row['Final_State']),
            first_order=row['first_order'],
            last_order=row['last_order'],
            unique_order_days=int(row['unique_order_days']),
            orders_count=int(row['orders_count']),
            aov=float(row['AOV']),
            recency_days=int(row['Recency_days']),
            recency_flag=int(row['Recency_flag']),
            r_label=str(row['R_label']),
            active_days=int(row['active_days']),
            orders_per_day=float(row['orders_per_day']),
            f_label=str(row['F_label']),
            f_cluster_id=int(row['F_cluster_id']),
            m_label=str(row['M_label']),
            m_cluster_id=float(row['M_cluster_id']) if pd.notna(row['M_cluster_id']) else None,
            rfm_segment=str(row['RFM_Segment']),
            total_net_amt=float(row['Total_Net_Amt'])
        )

    def _build_rfm_dataset(self, request: RFMRequest):
        if self.data_cache is None:
            return None

        cache_key = self._base_filter_cache_key(request)
        cached = self.rfm_result_cache.get(cache_key)
        if cached is not None:
            return cached

        df = self._apply_base_filters(self.data_cache.copy(), request)
        if df.empty:
            return None

        input_rows = len(df)
        input_outlets = df['Outlet_ID'].nunique()

        rfm, max_date, cluster_summary = self.calculate_rfm_metrics(
            df,
            recency_days=request.recency_threshold,
            frequency_threshold=request.frequency_threshold
        )

        outlet_totals = df.groupby('Outlet_ID')['Net_Amt'].sum().reset_index()
        outlet_totals.columns = ['Outlet_ID', 'Total_Net_Amt']
        rfm = rfm.merge(outlet_totals, on='Outlet_ID', how='left')
        segment_summaries = self._segment_summaries(rfm, df)

        dataset = {
            'df': df,
            'rfm': rfm,
            'max_date': max_date,
            'cluster_summary': cluster_summary,
            'segment_summaries': segment_summaries,
            'input_rows': input_rows,
            'input_outlets': input_outlets,
        }
        self.rfm_result_cache[cache_key] = dataset
        if len(self.rfm_result_cache) > self.max_cache_entries:
            oldest_key = next(iter(self.rfm_result_cache))
            self.rfm_result_cache.pop(oldest_key, None)

        return dataset

    def _build_step2_scope(self, request):
        dataset = self._build_rfm_dataset(request)
        if dataset is None:
            return None

        df = dataset['df']
        rfm = dataset['rfm']

        selected_segments = list(getattr(request, 'rfm_segments', []) or [])
        selected_classifications = list(getattr(request, 'outlet_classifications', []) or [])
        selected_slabs = [str(x) for x in (getattr(request, 'slabs', []) or [])]
        selected_outlets = [str(x) for x in (getattr(request, 'outlet_ids', []) or [])]

        rfm_scope = rfm.copy()
        if selected_segments:
            rfm_scope = rfm_scope[rfm_scope['RFM_Segment'].isin(selected_segments)]
        if selected_outlets:
            rfm_scope = rfm_scope[rfm_scope['Outlet_ID'].astype(str).isin(selected_outlets)]

        outlet_ids = set(rfm_scope['Outlet_ID'].astype(str).tolist())

        if selected_classifications and 'Final_Outlet_Classification' in df.columns:
            class_outlets = set(
                df[df['Final_Outlet_Classification'].isin(selected_classifications)]['Outlet_ID'].astype(str).tolist()
            )
            outlet_ids = outlet_ids.intersection(class_outlets)

        df_scope = df[df['Outlet_ID'].astype(str).isin(outlet_ids)].copy()
        if selected_slabs and 'Slab' in df_scope.columns:
            df_scope = df_scope[df_scope['Slab'].astype(str).isin(selected_slabs)]

        rfm_scope = rfm[rfm['Outlet_ID'].astype(str).isin(set(df_scope['Outlet_ID'].astype(str).tolist()))].copy()

        return {
            'dataset': dataset,
            'df_scope': df_scope,
            'rfm_scope': rfm_scope,
        }
    
    async def get_available_filters(self) -> Dict[str, List[str]]:
        """Get available filter options"""
        if self.data_cache is None:
            return {
                "states": [],
                "categories": [],
                "subcategories": [],
                "brands": [],
                "sizes": []
            }
        
        df = self.data_cache
        return {
            "states": sorted(df['Final_State'].dropna().unique().tolist()),
            "categories": sorted(df['Category'].dropna().unique().tolist()),
            "subcategories": sorted(df['Subcategory'].dropna().unique().tolist()),
            "brands": sorted(df['Brand'].dropna().unique().tolist()),
            "sizes": sorted(df['Sizes'].dropna().unique().tolist())
        }
    
    async def get_cascading_filters(self, current_filters: dict) -> Dict[str, List[str]]:
        """Get filtered options based on current selections (cascading filters)"""
        if self.data_cache is None:
            return await self.get_available_filters()

        df_all = self.data_cache

        states = current_filters.get('states') or []
        categories = current_filters.get('categories') or []
        subcategories = current_filters.get('subcategories') or []
        brands = current_filters.get('brands') or []

        # For each level, apply only parent filters (not the field's own filter).
        df_for_categories = df_all[df_all['Final_State'].isin(states)] if states else df_all

        df_for_subcategories = df_for_categories
        if categories:
            df_for_subcategories = df_for_subcategories[df_for_subcategories['Category'].isin(categories)]

        df_for_brands = df_for_subcategories
        if subcategories:
            df_for_brands = df_for_brands[df_for_brands['Subcategory'].isin(subcategories)]

        df_for_sizes = df_for_brands
        if brands:
            df_for_sizes = df_for_sizes[df_for_sizes['Brand'].isin(brands)]

        # States always show all options (top level), others follow parent cascade.
        return {
            "states": sorted(df_all['Final_State'].dropna().unique().tolist()),
            "categories": sorted(df_for_categories['Category'].dropna().unique().tolist()),
            "subcategories": sorted(df_for_subcategories['Subcategory'].dropna().unique().tolist()),
            "brands": sorted(df_for_brands['Brand'].dropna().unique().tolist()),
            "sizes": sorted(df_for_sizes['Sizes'].dropna().unique().tolist())
        }

    async def get_discount_options(self, request: DiscountOptionsRequest) -> DiscountOptionsResponse:
        scope = self._build_step2_scope(request)
        if scope is None:
            return DiscountOptionsResponse(
                success=False,
                message="No data matches the selected step-1 filters",
                rfm_segments=[],
                outlet_classifications=[],
                slabs=[],
                matching_outlets=0
            )

        dataset = scope['dataset']
        df = dataset['df']
        rfm = dataset['rfm']

        selected_segments = list(request.rfm_segments or [])
        selected_classifications = list(request.outlet_classifications or [])

        rfm_for_class = rfm.copy()
        if selected_segments:
            rfm_for_class = rfm_for_class[rfm_for_class['RFM_Segment'].isin(selected_segments)]

        class_outlet_ids = set(rfm_for_class['Outlet_ID'].astype(str).tolist())
        df_for_class = df[df['Outlet_ID'].astype(str).isin(class_outlet_ids)].copy()

        df_for_slab = df_for_class.copy()
        if selected_classifications and 'Final_Outlet_Classification' in df_for_slab.columns:
            df_for_slab = df_for_slab[df_for_slab['Final_Outlet_Classification'].isin(selected_classifications)]

        rfm_segments = sorted(rfm['RFM_Segment'].dropna().astype(str).unique().tolist())
        classifications = sorted(
            df_for_class['Final_Outlet_Classification'].dropna().astype(str).unique().tolist()
        ) if 'Final_Outlet_Classification' in df_for_class.columns else []
        slabs = sorted(df_for_slab['Slab'].dropna().astype(str).unique().tolist()) if 'Slab' in df_for_slab.columns else []

        matching_outlets = scope['df_scope']['Outlet_ID'].nunique() if not scope['df_scope'].empty else 0

        return DiscountOptionsResponse(
            success=True,
            message="Discount options loaded successfully",
            rfm_segments=rfm_segments,
            outlet_classifications=classifications,
            slabs=slabs,
            matching_outlets=int(matching_outlets)
        )

    def _round_discount_series(self, series, step: float = 0.5) -> pd.Series:
        safe_step = max(float(step), 0.1)
        s = pd.Series(series, copy=False).astype(float)
        return (np.round(s / safe_step) * safe_step).astype(float)

    def _slab_sort_key(self, slab_value):
        text = str(slab_value or "").strip().lower()
        m = re.search(r"(\d+)", text)
        if m:
            return (0, int(m.group(1)), text)
        return (1, text)

    def _estimate_base_discount_daily_blocks(
        self,
        period_series,
        discount_series,
        min_upward_jump_pp: float = 1.0,
        min_downward_drop_pp: float = 1.0,
        round_step: float = 0.5,
    ):
        periods = pd.to_datetime(pd.Series(period_series), errors='coerce')
        discounts = pd.Series(discount_series, copy=False).astype(float).replace([np.inf, -np.inf], np.nan)
        valid = periods.notna() & discounts.notna()
        if valid.sum() == 0:
            return np.array([]), np.array([], dtype=bool)

        work = pd.DataFrame({
            'Period': periods[valid].to_numpy(),
            'Discount': discounts[valid].to_numpy(dtype=float),
        }).sort_values('Period', kind='stable').reset_index(drop=True)

        periods_work = pd.to_datetime(work['Period'], errors='coerce')
        discounts_work = pd.Series(work['Discount'], dtype=float).interpolate(limit_direction='both').bfill().ffill()
        discounts_work = self._round_discount_series(discounts_work, step=round_step)

        n = len(discounts_work)
        if n == 0:
            return np.array([]), np.array([], dtype=bool)

        base = np.empty(n, dtype=float)
        transitions = np.zeros(n, dtype=bool)
        month_id = periods_work.dt.to_period('M')

        min_upward_jump_pp = max(0.0, float(min_upward_jump_pp))
        min_downward_drop_pp = max(0.0, float(min_downward_drop_pp))

        def _safe_median(vals):
            vals = np.asarray(vals, dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                return np.nan
            return float(np.nanmedian(vals))

        prev_base = None
        for bid in sorted(month_id.unique()):
            mask = (month_id == bid).to_numpy()
            block_vals = discounts_work[mask].to_numpy(dtype=float)

            valid_vals = block_vals[np.isfinite(block_vals)]
            if len(valid_vals) >= 8:
                q1, q3 = np.nanquantile(valid_vals, [0.25, 0.75])
                iqr = q3 - q1
                if np.isfinite(iqr) and iqr > 0:
                    lo_fence = q1 - 1.5 * iqr
                    hi_fence = q3 + 1.5 * iqr
                    block_vals = np.where((block_vals < lo_fence) | (block_vals > hi_fence), np.nan, block_vals)

            seg1 = block_vals[:10]
            seg2 = block_vals[10:20]
            seg3 = block_vals[20:]
            segment_medians = [_safe_median(seg1), _safe_median(seg2), _safe_median(seg3)]
            segment_medians = [m for m in segment_medians if np.isfinite(m)]
            candidate = float(np.min(segment_medians)) if segment_medians else _safe_median(block_vals)

            if prev_base is None:
                block_base = candidate if np.isfinite(candidate) else 0.0
            else:
                if not np.isfinite(candidate):
                    candidate = prev_base

                if candidate > prev_base and (candidate - prev_base) < min_upward_jump_pp:
                    block_base = prev_base
                elif candidate < prev_base and (prev_base - candidate) < min_downward_drop_pp:
                    block_base = prev_base
                else:
                    block_base = candidate

                first_idx = int(np.where(mask)[0][0])
                if abs(block_base - prev_base) > 1e-9:
                    transitions[first_idx] = True

            base[mask] = np.clip(block_base, 0.0, 100.0)
            prev_base = block_base

        return base, transitions

    def _compute_base_depth_result(self, df: pd.DataFrame, request: BaseDepthRequest):
        work = df.copy()
        work['Period'] = work['Date'].dt.floor('D')

        if 'Bill_No' in work.columns:
            daily = (
                work.groupby('Period', as_index=False)
                .agg(
                    orders=('Bill_No', 'nunique'),
                    quantity=('Quantity', 'sum'),
                    total_discount=('TotalDiscount', 'sum'),
                    sales_value=('SalesValue_atBasicRate', 'sum'),
                )
                .sort_values('Period')
            )
        else:
            daily = (
                work.groupby('Period', as_index=False)
                .agg(
                    orders=('Outlet_ID', 'count'),
                    quantity=('Quantity', 'sum'),
                    total_discount=('TotalDiscount', 'sum'),
                    sales_value=('SalesValue_atBasicRate', 'sum'),
                )
                .sort_values('Period')
            )

        daily['actual_discount_pct'] = (
            (daily['total_discount'] / daily['sales_value']) * 100.0
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0)

        base_daily, _ = self._estimate_base_discount_daily_blocks(
            daily['Period'],
            daily['actual_discount_pct'],
            min_upward_jump_pp=float(request.min_upward_jump_pp),
            min_downward_drop_pp=float(request.min_downward_drop_pp),
            round_step=float(request.round_step),
        )
        if len(base_daily) != len(daily):
            base_daily = np.zeros(len(daily), dtype=float)
        daily['base_discount_pct'] = base_daily

        if request.time_aggregation == "D":
            aggregated = daily.copy()
        else:
            if request.time_aggregation == "W":
                daily['OutputPeriod'] = daily['Period'].dt.to_period('W').dt.start_time
            else:
                daily['OutputPeriod'] = daily['Period'].dt.to_period('M').dt.start_time

            aggregated = (
                daily.groupby('OutputPeriod', as_index=False)
                .agg(
                    orders=('orders', 'sum'),
                    quantity=('quantity', 'sum'),
                    total_discount=('total_discount', 'sum'),
                    sales_value=('sales_value', 'sum'),
                    base_discount_pct=('base_discount_pct', 'first'),
                )
                .rename(columns={'OutputPeriod': 'Period'})
                .sort_values('Period')
            )
            aggregated['actual_discount_pct'] = (
                (aggregated['total_discount'] / aggregated['sales_value']) * 100.0
            ).replace([np.inf, -np.inf], 0.0).fillna(0.0)

        aggregated['tactical_discount_pct'] = (
            aggregated['actual_discount_pct'] - aggregated['base_discount_pct']
        ).clip(lower=0.0)

        points = [
            BaseDepthPoint(
                period=row['Period'].to_pydatetime() if hasattr(row['Period'], 'to_pydatetime') else row['Period'],
                actual_discount_pct=float(row['actual_discount_pct']),
                base_discount_pct=float(row['base_discount_pct']),
                orders=int(row['orders']),
                quantity=float(row['quantity']),
                sales_value=float(row['sales_value']),
            )
            for _, row in aggregated.iterrows()
        ]

        summary = {
            "periods": float(len(aggregated)),
            "avg_actual_discount_pct": float(aggregated['actual_discount_pct'].mean()) if len(aggregated) else 0.0,
            "avg_base_discount_pct": float(aggregated['base_discount_pct'].mean()) if len(aggregated) else 0.0,
            "avg_tactical_discount_pct": float(aggregated['tactical_discount_pct'].mean()) if len(aggregated) else 0.0,
            "max_actual_discount_pct": float(aggregated['actual_discount_pct'].max()) if len(aggregated) else 0.0,
            "max_base_discount_pct": float(aggregated['base_discount_pct'].max()) if len(aggregated) else 0.0,
        }
        return points, summary

    def _build_monthly_model_dataframe(self, df: pd.DataFrame, request: ModelingRequest) -> pd.DataFrame:
        work = df.copy()
        store_col = 'Store_ID' if 'Store_ID' in work.columns else 'Outlet_ID'
        work['Date'] = pd.to_datetime(work['Date'], errors='coerce')
        work = work.dropna(subset=['Date'])
        if work.empty:
            return pd.DataFrame()

        work['Period_D'] = work['Date'].dt.floor('D')
        daily = (
            work.groupby('Period_D', as_index=False)
            .agg(
                store_count=(store_col, 'nunique'),
                quantity=('Quantity', 'sum'),
                total_discount=('TotalDiscount', 'sum'),
                sales_value=('SalesValue_atBasicRate', 'sum'),
            )
            .rename(columns={'Period_D': 'Period'})
            .sort_values('Period')
        )
        daily['actual_discount_pct'] = (
            (daily['total_discount'] / daily['sales_value']) * 100.0
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0)

        base_daily, _ = self._estimate_base_discount_daily_blocks(
            daily['Period'],
            daily['actual_discount_pct'],
            min_upward_jump_pp=float(request.min_upward_jump_pp),
            min_downward_drop_pp=float(request.min_downward_drop_pp),
            round_step=float(request.round_step),
        )
        if len(base_daily) != len(daily):
            base_daily = np.zeros(len(daily), dtype=float)
        daily['base_discount_pct'] = base_daily
        daily['Month_Key'] = pd.to_datetime(daily['Period']).dt.to_period('M')

        monthly_base = (
            daily.groupby('Month_Key', as_index=False)
            .agg(base_discount_pct=('base_discount_pct', 'first'))
        )

        work['Month_Key'] = work['Date'].dt.to_period('M')
        monthly = (
            work.groupby('Month_Key', as_index=False)
            .agg(
                store_count=(store_col, 'nunique'),
                quantity=('Quantity', 'sum'),
                total_discount=('TotalDiscount', 'sum'),
                sales_value=('SalesValue_atBasicRate', 'sum'),
            )
        )
        monthly['Period'] = monthly['Month_Key'].dt.to_timestamp()
        monthly = monthly.merge(monthly_base, on='Month_Key', how='left')
        monthly['base_discount_pct'] = monthly['base_discount_pct'].ffill().bfill().fillna(0.0)
        monthly['actual_discount_pct'] = (
            (monthly['total_discount'] / monthly['sales_value']) * 100.0
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        monthly['tactical_discount_pct'] = (
            monthly['actual_discount_pct'] - monthly['base_discount_pct']
        ).clip(lower=0.0)
        monthly['lag1_base_discount_pct'] = monthly['base_discount_pct'].shift(1).fillna(monthly['base_discount_pct'])
        monthly['base_price'] = (
            monthly['sales_value'] / monthly['quantity'].replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        return monthly.sort_values('Period').reset_index(drop=True)

    def _predict_stage2_quantity(self, stage2_model, residual_store, structural, tactical, lag1=None):
        stage2_features = int(getattr(stage2_model, 'n_features_in_', 4))
        residual_store = np.asarray(residual_store, dtype=float)
        structural = np.asarray(structural, dtype=float)
        tactical = np.asarray(tactical, dtype=float)
        if lag1 is None:
            lag1 = np.zeros_like(structural, dtype=float)
        lag1 = np.asarray(lag1, dtype=float)

        if stage2_features >= 4:
            x = np.column_stack([residual_store, structural, tactical, lag1])
        elif stage2_features == 3:
            x = np.column_stack([residual_store, structural, tactical])
        elif stage2_features == 2:
            x = np.column_stack([residual_store, structural])
        else:
            x = residual_store.reshape(-1, 1)
        return stage2_model.predict(x)

    def _build_planner_ai_prompt(
        self,
        slab: str,
        months: List[str],
        default_structural: List[float],
        planned_structural: List[float],
        default_base_prices: List[float],
        planned_base_prices: List[float],
        metrics: Dict[str, float],
        series_rows: List[Dict[str, float]],
    ) -> str:
        def _to_num(value, default=0.0):
            try:
                v = float(value)
                return v if np.isfinite(v) else float(default)
            except Exception:
                return float(default)

        month_changes = self._extract_planner_month_changes(
            months=months,
            default_structural=default_structural,
            planned_structural=planned_structural,
            default_base_prices=default_base_prices,
            planned_base_prices=planned_base_prices,
        )

        payload = {
            "slab": slab,
            "chart_data_monthly": series_rows,
            "user_changes": month_changes,
            "metrics_summary": {
                "volume_change_pct": _to_num(metrics.get("volume_change_pct")),
                "revenue_change_pct": _to_num(metrics.get("revenue_change_pct")),
                "profit_change_pct": _to_num(metrics.get("profit_change_pct")),
                "promo_change_pct": _to_num(metrics.get("promo_change_pct")),
                "roi_revenue_x": _to_num(metrics.get("roi_revenue_x")),
                "total_spend_plan": _to_num(metrics.get("total_spend_plan")),
                "revenue_delta": _to_num(metrics.get("revenue_delta")),
                "profit_delta": _to_num(metrics.get("profit_delta")),
                "quantity_delta": _to_num(metrics.get("quantity_delta")),
            },
        }
        payload_text = json.dumps(payload, default=str, ensure_ascii=True)

        return (
            "You are a senior trade-promotion strategy analyst for FMCG retail.\n"
            "Read slab-level monthly data and user edits, then write a practical business review.\n\n"
            "Output rules (strict):\n"
            "1) Use ONLY bullet points, no paragraph blocks.\n"
            "2) Return exactly 5 bullets total.\n"
            "3) Every bullet must include at least one number from input.\n"
            "4) Mention at least 2 specific months by name.\n"
            "5) Keep each bullet complete (no cut-off), clear, and action-oriented.\n"
            "6) Cover: volume, revenue, profit, spend, ROI, risk, and recommended actions.\n"
            "7) Start each bullet as: **Short Label:** detail.\n"
            "8) If user_changes is empty, explicitly say plan is unchanged and what that implies.\n\n"
            f"Input JSON:\n{payload_text}"
        )

    def _extract_planner_month_changes(
        self,
        months: List[str],
        default_structural: List[float],
        planned_structural: List[float],
        default_base_prices: List[float],
        planned_base_prices: List[float],
    ) -> List[Dict[str, float]]:
        def _to_num(value, default=0.0):
            try:
                v = float(value)
                return v if np.isfinite(v) else float(default)
            except Exception:
                return float(default)

        month_changes = []
        for i, month in enumerate(months):
            old_struct = _to_num(default_structural[i] if i < len(default_structural) else 0.0)
            new_struct = _to_num(planned_structural[i] if i < len(planned_structural) else old_struct)
            old_price = _to_num(default_base_prices[i] if i < len(default_base_prices) else 0.0)
            new_price = _to_num(planned_base_prices[i] if i < len(planned_base_prices) else old_price)
            delta_struct = new_struct - old_struct
            delta_price = new_price - old_price
            if abs(delta_struct) > 1e-9 or abs(delta_price) > 1e-9:
                month_changes.append(
                    {
                        "month": str(month),
                        "structural_discount_change_pp": round(delta_struct, 3),
                        "base_price_change": round(delta_price, 3),
                        "new_structural_discount_pct": round(new_struct, 3),
                        "new_base_price": round(new_price, 3),
                    }
                )
        return month_changes

    def _build_planner_fallback_insights(
        self,
        slab: str,
        months: List[str],
        default_structural: List[float],
        planned_structural: List[float],
        default_base_prices: List[float],
        planned_base_prices: List[float],
        metrics: Dict[str, float],
    ) -> str:
        def _num(value, default=0.0):
            try:
                v = float(value)
                return v if np.isfinite(v) else float(default)
            except Exception:
                return float(default)

        month_changes = self._extract_planner_month_changes(
            months=months,
            default_structural=default_structural,
            planned_structural=planned_structural,
            default_base_prices=default_base_prices,
            planned_base_prices=planned_base_prices,
        )
        changed_months = [row["month"] for row in month_changes]
        first_month = changed_months[0] if changed_months else (months[0] if months else "NA")
        second_month = changed_months[1] if len(changed_months) > 1 else (months[1] if len(months) > 1 else first_month)
        max_up = max([row for row in month_changes if row["structural_discount_change_pp"] > 0], key=lambda r: r["structural_discount_change_pp"], default=None)
        max_down = min([row for row in month_changes if row["structural_discount_change_pp"] < 0], key=lambda r: r["structural_discount_change_pp"], default=None)

        qty_delta = _num(metrics.get("quantity_delta"))
        rev_delta = _num(metrics.get("revenue_delta"))
        prof_delta = _num(metrics.get("profit_delta"))
        qty_pct = _num(metrics.get("volume_change_pct"))
        rev_pct = _num(metrics.get("revenue_change_pct"))
        prof_pct = _num(metrics.get("profit_change_pct"))
        roi_x = _num(metrics.get("roi_revenue_x"))
        spend = _num(metrics.get("total_spend_plan"))
        promo_pct = _num(metrics.get("promo_change_pct"))

        bullets = [
            f"- **Business Impact:** Slab {slab} shows volume {qty_pct:.2f}% ({qty_delta:,.0f} units), revenue {rev_pct:.2f}% ({rev_delta:,.0f}), and profit {prof_pct:.2f}% ({prof_delta:,.0f}).",
            f"- **Spend and ROI:** Promo intensity shifts {promo_pct:.2f}% vs baseline with spend {spend:,.0f}; revenue ROI is {roi_x:.2f}x.",
        ]

        if max_up is not None:
            bullets.append(
                f"- **Primary Increase Month:** {max_up['month']} has the largest step-up, +{max_up['structural_discount_change_pp']:.2f} pp to {max_up['new_structural_discount_pct']:.2f}%."
            )
        else:
            bullets.append(
                f"- **Promo Pattern:** No structural step-up detected; {first_month} to {second_month} stays at baseline depth."
            )

        if max_down is not None:
            bullets.append(
                f"- **Margin Protection:** Largest step-down is {max_down['month']} at {max_down['structural_discount_change_pp']:.2f} pp, which supports margin."
            )
        else:
            bullets.append(
                f"- **Margin Protection:** No step-down between {first_month} and {second_month}; margin control depends mainly on base-price discipline."
            )

        bullets.extend([
            f"- **Action:** If ROI is below 1.50x in high-discount months such as {first_month}, reduce depth by 0.5-1.0 pp and recheck next cycle.",
        ])
        return "### Trinity Insights\n" + "\n".join(bullets)

    def _is_low_quality_insight(self, text: str) -> bool:
        if not isinstance(text, str) or not text.strip():
            return True
        body = text.strip()
        bullet_count = len([line for line in body.splitlines() if line.strip().startswith("-")])
        if bullet_count < 4:
            return True
        if len(body) < 280:
            return True
        if re.search(r"\b(and|or|with|for|to)\s*$", body.lower()):
            return True
        if body.endswith("("):
            return True
        return False

    def _generate_planner_ai_insights(
        self,
        slab: str,
        months: List[str],
        default_structural: List[float],
        planned_structural: List[float],
        default_base_prices: List[float],
        planned_base_prices: List[float],
        metrics: Dict[str, float],
        series_rows: List[Dict[str, float]],
    ) -> Dict[str, str]:
        api_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
        if not api_key:
            return {
                "status": "disabled",
                "text": "Trinity Insights are disabled. Set GEMINI_API_KEY on backend and recalculate.",
            }

        model_name = (os.getenv("GEMINI_MODEL") or "gemini-2.5-flash").strip()
        prompt = self._build_planner_ai_prompt(
            slab=slab,
            months=months,
            default_structural=default_structural,
            planned_structural=planned_structural,
            default_base_prices=default_base_prices,
            planned_base_prices=planned_base_prices,
            metrics=metrics,
            series_rows=series_rows,
        )
        endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model_name}:generateContent?key={api_key}"
        )
        request_body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.2,
                "topP": 0.9,
                "maxOutputTokens": 700,
            },
        }

        try:
            req = urllib.request.Request(
                endpoint,
                data=json.dumps(request_body).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = resp.read().decode("utf-8")
            payload = json.loads(body)
            candidates = payload.get("candidates") or []
            parts = []
            for cand in candidates:
                content = cand.get("content") or {}
                for part in (content.get("parts") or []):
                    text = part.get("text")
                    if isinstance(text, str) and text.strip():
                        parts.append(text.strip())
            joined = "\n\n".join(parts).strip()
            if self._is_low_quality_insight(joined):
                fallback = self._build_planner_fallback_insights(
                    slab=slab,
                    months=months,
                    default_structural=default_structural,
                    planned_structural=planned_structural,
                    default_base_prices=default_base_prices,
                    planned_base_prices=planned_base_prices,
                    metrics=metrics,
                )
                return {"status": "ready", "text": fallback}
            return {"status": "ready", "text": joined}
        except urllib.error.HTTPError as err:
            detail = ""
            try:
                detail = err.read().decode("utf-8", errors="ignore")
            except Exception:
                detail = str(err)
            fallback = self._build_planner_fallback_insights(
                slab=slab,
                months=months,
                default_structural=default_structural,
                planned_structural=planned_structural,
                default_base_prices=default_base_prices,
                planned_base_prices=planned_base_prices,
                metrics=metrics,
            )
            return {"status": "ready", "text": f"{fallback}\n\nNote: Trinity model call failed ({err.code}). {detail[:120]}"}
        except Exception as err:
            fallback = self._build_planner_fallback_insights(
                slab=slab,
                months=months,
                default_structural=default_structural,
                planned_structural=planned_structural,
                default_base_prices=default_base_prices,
                planned_base_prices=planned_base_prices,
                metrics=metrics,
            )
            return {"status": "ready", "text": f"{fallback}\n\nNote: Trinity model error: {str(err)[:120]}"}

    def _build_structural_roi_points(
        self,
        model_df: pd.DataFrame,
        stage2_model,
        round_step: float = 0.5,
        cogs_per_unit: float = 0.0,
    ):
        if model_df is None or model_df.empty:
            return [], {
                'episodes': 0.0,
                'avg_roi_1mo': 0.0,
                'total_spend': 0.0,
                'total_incremental_revenue': 0.0,
                'total_incremental_profit': 0.0,
                'structural_roi_1mo': 0.0,
                'structural_profit_roi_1mo': 0.0,
            }

        period_df = model_df.sort_values('Period').reset_index(drop=True).copy()
        period_df['Period'] = pd.to_datetime(period_df['Period'], errors='coerce')
        period_df = period_df.dropna(subset=['Period'])
        if period_df.empty:
            return [], {
                'episodes': 0.0,
                'avg_roi_1mo': 0.0,
                'total_spend': 0.0,
                'total_incremental_revenue': 0.0,
                'total_incremental_profit': 0.0,
                'structural_roi_1mo': 0.0,
                'structural_profit_roi_1mo': 0.0,
            }

        # Quantize structural base first, so tiny float noise does not create fake extra episodes.
        safe_step = max(float(round_step), 0.1)
        period_df['base_discount_pct'] = self._round_discount_series(period_df['base_discount_pct'], step=safe_step)

        regime_break = period_df['base_discount_pct'].diff().abs().fillna(0) > 1e-9
        period_df['regime_id'] = regime_break.cumsum()
        period_df['row_idx'] = np.arange(len(period_df))

        regimes = (
            period_df.groupby('regime_id', as_index=False)
            .agg(
                start_idx=('row_idx', 'min'),
                end_idx=('row_idx', 'max'),
                base_discount_pct=('base_discount_pct', 'first'),
            )
            .sort_values('start_idx')
            .reset_index(drop=True)
        )

        roi_points = []
        episode_rois = []
        episode_profit_rois = []
        total_spend = 0.0
        total_incremental_revenue = 0.0
        total_incremental_profit = 0.0
        cogs_per_unit = max(float(cogs_per_unit), 0.0)

        for i in range(1, len(regimes)):
            prev_base = float(regimes.loc[i - 1, 'base_discount_pct'])
            curr_base = float(regimes.loc[i, 'base_discount_pct'])
            step_up = curr_base - prev_base
            if step_up <= 0:
                continue

            start_idx = int(regimes.loc[i, 'start_idx'])
            end_idx = int(regimes.loc[i, 'end_idx'])
            hold_df = period_df.iloc[start_idx:end_idx + 1].copy()
            if hold_df.empty:
                continue

            n_rows = len(hold_df)
            residual_anchor = hold_df['residual_store'].to_numpy(dtype=float)
            base_price = hold_df['base_price'].to_numpy(dtype=float)
            actual_qty = hold_df['quantity'].to_numpy(dtype=float)
            actual_discount = hold_df['actual_discount_pct'].to_numpy(dtype=float)

            prev_struct = np.full(n_rows, prev_base, dtype=float)
            curr_struct = np.full(n_rows, curr_base, dtype=float)
            lag1_prev = np.full(n_rows, prev_base, dtype=float)
            lag1_curr = np.full(n_rows, curr_base, dtype=float)
            lag1_curr[0] = prev_base
            zeros = np.zeros(n_rows, dtype=float)

            # Structural ROI episodes: tactical term is forced to zero in both worlds.
            qty_prev = self._predict_stage2_quantity(stage2_model, residual_anchor, prev_struct, zeros, lag1_prev)
            qty_curr = self._predict_stage2_quantity(stage2_model, residual_anchor, curr_struct, zeros, lag1_curr)
            qty_prev = np.maximum(qty_prev, 0.0)
            qty_curr = np.maximum(qty_curr, 0.0)

            baseline_price = base_price * (1 - prev_base / 100.0)
            # Keep same as Streamlit structural ROI implementation.
            current_price = base_price * (1 - prev_base / 100.0)
            baseline_revenue = qty_prev * baseline_price
            predicted_revenue = qty_curr * current_price
            incremental_revenue = predicted_revenue - baseline_revenue
            spend = base_price * (step_up / 100.0) * qty_curr
            baseline_profit = baseline_revenue - (cogs_per_unit * qty_prev)
            predicted_profit = predicted_revenue - (cogs_per_unit * qty_curr)
            incremental_profit = predicted_profit - baseline_profit

            spend_sum = float(np.nansum(spend))
            incr_sum = float(np.nansum(incremental_revenue))
            incr_profit_sum = float(np.nansum(incremental_profit))
            roi_1mo = float(incr_sum / spend_sum) if spend_sum > 0 else np.nan
            profit_roi_1mo = float(incr_profit_sum / spend_sum) if spend_sum > 0 else np.nan

            episode_rois.append(roi_1mo)
            episode_profit_rois.append(profit_roi_1mo)
            total_spend += spend_sum
            total_incremental_revenue += incr_sum
            total_incremental_profit += incr_profit_sum

            first_row = hold_df.iloc[0]
            roi_points.append(
                ModelingPoint(
                    period=first_row['Period'].to_pydatetime() if hasattr(first_row['Period'], 'to_pydatetime') else first_row['Period'],
                    actual_quantity=float(actual_qty[0]) if len(actual_qty) > 0 else 0.0,
                    predicted_quantity=float(qty_curr[0]) if len(qty_curr) > 0 else 0.0,
                    actual_discount_pct=float(actual_discount[0]) if len(actual_discount) > 0 else 0.0,
                    base_discount_pct=float(curr_base),
                    tactical_discount_pct=0.0,
                    roi_1mo=float(roi_1mo) if np.isfinite(roi_1mo) else None,
                    profit_roi_1mo=float(profit_roi_1mo) if np.isfinite(profit_roi_1mo) else None,
                    spend=float(spend_sum),
                    incremental_revenue=float(incr_sum),
                    incremental_profit=float(incr_profit_sum),
                )
            )

        avg_roi = float(np.nanmean(episode_rois)) if len(episode_rois) > 0 else 0.0
        avg_profit_roi = float(np.nanmean(episode_profit_rois)) if len(episode_profit_rois) > 0 else 0.0
        structural_roi_1mo = float(total_incremental_revenue / total_spend) if total_spend > 0 else 0.0
        structural_profit_roi_1mo = float(total_incremental_profit / total_spend) if total_spend > 0 else 0.0
        summary = {
            'episodes': float(len(episode_rois)),
            'avg_roi_1mo': avg_roi,
            'avg_profit_roi_1mo': avg_profit_roi,
            'total_spend': float(total_spend),
            'total_incremental_revenue': float(total_incremental_revenue),
            'total_incremental_profit': float(total_incremental_profit),
            'structural_roi_1mo': structural_roi_1mo,
            'structural_profit_roi_1mo': structural_profit_roi_1mo,
        }
        # Keep one bar per episode month (step-up month only).
        dedup = {}
        for point in roi_points:
            key = point.period.strftime('%Y-%m-%d') if hasattr(point.period, 'strftime') else str(point.period)
            dedup[key] = point
        roi_points = [dedup[k] for k in sorted(dedup.keys())]
        return roi_points, summary

    def _run_two_stage_model(
        self,
        monthly: pd.DataFrame,
        include_lag_discount: bool = True,
        l2_penalty: float = 1.0,
        optimize_l2_penalty: bool = False,
        constraint_residual_non_negative: bool = True,
        constraint_structural_non_negative: bool = True,
        constraint_tactical_non_negative: bool = True,
        constraint_lag_non_positive: bool = True,
    ):
        if monthly is None or monthly.empty or len(monthly) < 3:
            return None

        x_discount = monthly['actual_discount_pct'].to_numpy(dtype=float)
        y_qty = monthly['quantity'].to_numpy(dtype=float)
        y_store = monthly['store_count'].to_numpy(dtype=float)
        base = monthly['base_discount_pct'].to_numpy(dtype=float)
        tactical = monthly['tactical_discount_pct'].to_numpy(dtype=float)
        lag1 = monthly['lag1_base_discount_pct'].to_numpy(dtype=float)
        periods = pd.to_datetime(monthly['Period'], errors='coerce')

        stage1 = LinearRegression()
        stage1.fit(x_discount.reshape(-1, 1), y_store)
        store_pred = stage1.predict(x_discount.reshape(-1, 1))
        residual_store = y_store - store_pred

        include_lag_discount = bool(include_lag_discount)
        l2_penalty = max(float(l2_penalty), 0.0)
        optimize_l2_penalty = bool(optimize_l2_penalty)
        constraint_residual_non_negative = bool(constraint_residual_non_negative)
        constraint_structural_non_negative = bool(constraint_structural_non_negative)
        constraint_tactical_non_negative = bool(constraint_tactical_non_negative)
        constraint_lag_non_positive = bool(constraint_lag_non_positive)

        non_negative_indices = []
        if constraint_residual_non_negative:
            non_negative_indices.append(0)
        if constraint_structural_non_negative:
            non_negative_indices.append(1)
        if constraint_tactical_non_negative:
            non_negative_indices.append(2)

        non_positive_indices = []
        if include_lag_discount and constraint_lag_non_positive:
            non_positive_indices.append(3)

        if include_lag_discount:
            X2 = np.column_stack([residual_store, base, tactical, lag1])
        else:
            X2 = np.column_stack([residual_store, base, tactical])

        def _fit_stage2_with_l2(lam: float):
            model = CustomConstrainedRidge(
                l2_penalty=float(lam),
                non_negative_indices=non_negative_indices,
                non_positive_indices=non_positive_indices if include_lag_discount else [],
                maxiter=4000,
            )
            model.fit(X2, y_qty)
            preds = model.predict(X2)
            if len(y_qty) > 1:
                ss_res = float(np.sum((y_qty - preds) ** 2))
                ss_tot = float(np.sum((y_qty - np.mean(y_qty)) ** 2))
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
            else:
                r2 = 0.0
            return model, preds, float(r2)

        def _cv_r2_for_l2(lam: float):
            n = len(y_qty)
            p = X2.shape[1]
            min_train = max(p + 2, 8)
            if n <= (min_train + 1):
                return np.nan

            y_true = []
            y_hat = []
            for split in range(min_train, n):
                X_train = X2[:split]
                y_train = y_qty[:split]
                X_val = X2[split:split + 1]
                y_val = y_qty[split:split + 1]

                model = CustomConstrainedRidge(
                    l2_penalty=float(lam),
                    non_negative_indices=non_negative_indices,
                    non_positive_indices=non_positive_indices if include_lag_discount else [],
                    maxiter=4000,
                )
                try:
                    model.fit(X_train, y_train)
                    pred = model.predict(X_val)
                except Exception:
                    continue

                if pred is None or len(pred) != 1:
                    continue
                y_true.append(float(y_val[0]))
                y_hat.append(float(pred[0]))

            if len(y_true) < 3:
                return np.nan
            y_true = np.asarray(y_true, dtype=float)
            y_hat = np.asarray(y_hat, dtype=float)
            ss_res = float(np.sum((y_true - y_hat) ** 2))
            ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
            if ss_tot <= 1e-12:
                return np.nan
            return 1.0 - (ss_res / ss_tot)

        if optimize_l2_penalty:
            l2_candidates = [0.0, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
            best = None
            for lam in l2_candidates:
                model_tmp, preds_tmp, r2_tmp = _fit_stage2_with_l2(lam)
                cv_r2_tmp = _cv_r2_for_l2(lam)
                if (best is None) or ((cv_r2_tmp if np.isfinite(cv_r2_tmp) else -np.inf) > best['cv_r2']):
                    best = {
                        'model': model_tmp,
                        'preds': preds_tmp,
                        'r2_train': r2_tmp,
                        'cv_r2': float(cv_r2_tmp) if np.isfinite(cv_r2_tmp) else -np.inf,
                        'l2': float(lam),
                    }
            stage2 = best['model']
            qty_pred = best['preds']
            r2_stage2 = float(best['r2_train'])
            r2_stage2_cv = float(best['cv_r2']) if np.isfinite(best['cv_r2']) else np.nan
            l2_penalty_effective = float(best['l2'])
            l2_candidates_evaluated = float(len(l2_candidates))
        else:
            stage2, qty_pred, r2_stage2 = _fit_stage2_with_l2(l2_penalty)
            r2_stage2_cv = float(_cv_r2_for_l2(l2_penalty))
            l2_penalty_effective = float(l2_penalty)
            l2_candidates_evaluated = 1.0

        # Reference OLS model for beta/R2 comparison only (not used for downstream calculations).
        stage2_ols = LinearRegression()
        stage2_ols.fit(X2, y_qty)
        qty_pred_ols = stage2_ols.predict(X2)
        if len(y_qty) > 1:
            ss_tot_s2 = float(np.sum((y_qty - np.mean(y_qty)) ** 2))
            ss_res_ols = float(np.sum((y_qty - qty_pred_ols) ** 2))
            r2_stage2_ols = 1 - (ss_res_ols / ss_tot_s2) if ss_tot_s2 > 1e-12 else 0.0
        else:
            r2_stage2_ols = 0.0

        qty_base = self._predict_stage2_quantity(stage2, residual_store, base, np.zeros_like(tactical), lag1)
        qty_base_ols = self._predict_stage2_quantity(stage2_ols, residual_store, base, np.zeros_like(tactical), lag1)

        actual_price = monthly['base_price'].to_numpy(dtype=float) * (1 - x_discount / 100.0)
        baseline_price = monthly['base_price'].to_numpy(dtype=float) * (1 - base / 100.0)
        predicted_revenue = qty_pred * actual_price
        baseline_revenue = qty_base * baseline_price
        spend = monthly['base_price'].to_numpy(dtype=float) * (tactical / 100.0) * qty_pred
        incremental_revenue = predicted_revenue - baseline_revenue
        roi = np.where(spend > 0, incremental_revenue / spend, np.nan)

        result_df = monthly.copy()
        result_df['predicted_quantity'] = qty_pred
        result_df['baseline_quantity'] = qty_base
        result_df['predicted_quantity_ols'] = qty_pred_ols
        result_df['baseline_quantity_ols'] = qty_base_ols
        result_df['predicted_revenue'] = predicted_revenue
        result_df['baseline_revenue'] = baseline_revenue
        result_df['spend'] = spend
        result_df['incremental_revenue'] = incremental_revenue
        result_df['roi_1mo'] = roi
        result_df['residual_store'] = residual_store

        coefficients = {
            'stage1_intercept': float(stage1.intercept_),
            'stage1_coef_discount': float(stage1.coef_[0]),
            'stage1_r2': float(stage1.score(x_discount.reshape(-1, 1), y_store)) if len(y_store) > 1 else 0.0,
            'stage2_intercept': float(stage2.intercept_),
            'coef_residual_store': float(stage2.coef_[0]),
            'coef_structural_discount': float(stage2.coef_[1]),
            'coef_tactical_discount': float(stage2.coef_[2]),
            'coef_lag1_structural_discount': float(stage2.coef_[3]) if len(stage2.coef_) >= 4 else 0.0,
            'include_lag_discount': 1.0 if include_lag_discount else 0.0,
            'l2_penalty': float(l2_penalty_effective),
            'l2_penalty_input': float(l2_penalty),
            'optimize_l2_penalty': 1.0 if optimize_l2_penalty else 0.0,
            'l2_candidates_evaluated': float(l2_candidates_evaluated),
            'stage2_cv_r2': float(r2_stage2_cv) if np.isfinite(r2_stage2_cv) else 0.0,
            'constraint_residual_non_negative': 1.0 if constraint_residual_non_negative else 0.0,
            'constraint_structural_non_negative': 1.0 if constraint_structural_non_negative else 0.0,
            'constraint_tactical_non_negative': 1.0 if constraint_tactical_non_negative else 0.0,
            'constraint_lag_non_positive': 1.0 if (include_lag_discount and constraint_lag_non_positive) else 0.0,
            'stage2_model_type': 1.0,  # 1.0 => constrained ridge
            'stage2_fit_success': 1.0 if bool(getattr(stage2, 'success_', True)) else 0.0,
            'stage2_r2': float(r2_stage2),
            # OLS reference metrics (comparison only; not used in planning/ROI outputs).
            'stage2_ols_intercept': float(stage2_ols.intercept_),
            'stage2_ols_coef_residual_store': float(stage2_ols.coef_[0]),
            'stage2_ols_coef_structural_discount': float(stage2_ols.coef_[1]),
            'stage2_ols_coef_tactical_discount': float(stage2_ols.coef_[2]),
            'stage2_ols_coef_lag1_structural_discount': float(stage2_ols.coef_[3]) if len(stage2_ols.coef_) >= 4 else 0.0,
            'stage2_ols_r2': float(r2_stage2_ols),
        }
        return {
            'model_df': result_df,
            'coefficients': coefficients,
            'stage2_model': stage2,
        }

    async def calculate_modeling(self, request: ModelingRequest) -> ModelingResponse:
        try:
            scope = self._build_step2_scope(request)
            if scope is None:
                return ModelingResponse(success=False, message="No data matches selected filters", slab_results=[], combined_summary={})

            df_scope = scope['df_scope']
            if df_scope.empty:
                return ModelingResponse(success=False, message="No data for modeling", slab_results=[], combined_summary={})

            if 'Slab' in df_scope.columns:
                slab_list = [str(s) for s in (request.slabs or [])]
                if not slab_list:
                    slab_list = df_scope['Slab'].dropna().astype(str).unique().tolist()
                slab_list = sorted(list(dict.fromkeys(slab_list)), key=self._slab_sort_key)
            else:
                slab_list = ['All']

            slab_results = []
            weighted_qty = 0.0
            weighted_r2 = 0.0
            total_spend = 0.0
            total_incremental_revenue = 0.0

            for slab in slab_list:
                slab_df = df_scope.copy()
                if 'Slab' in slab_df.columns:
                    slab_df = slab_df[slab_df['Slab'].astype(str) == str(slab)]

                if slab_df.empty:
                    slab_results.append(ModelingSlabResult(slab=str(slab), valid=False, reason="No data for slab"))
                    continue

                monthly = self._build_monthly_model_dataframe(slab_df, request)
                if monthly.empty or len(monthly) < 3:
                    slab_results.append(ModelingSlabResult(slab=str(slab), valid=False, reason="Not enough monthly points for modeling"))
                    continue

                modeled = self._run_two_stage_model(
                    monthly,
                    include_lag_discount=bool(getattr(request, 'include_lag_discount', True)),
                    l2_penalty=float(getattr(request, 'l2_penalty', 1.0)),
                    optimize_l2_penalty=bool(getattr(request, 'optimize_l2_penalty', False)),
                    constraint_residual_non_negative=bool(getattr(request, 'constraint_residual_non_negative', True)),
                    constraint_structural_non_negative=bool(getattr(request, 'constraint_structural_non_negative', True)),
                    constraint_tactical_non_negative=bool(getattr(request, 'constraint_tactical_non_negative', True)),
                    constraint_lag_non_positive=bool(getattr(request, 'constraint_lag_non_positive', True)),
                )
                if modeled is None:
                    slab_results.append(ModelingSlabResult(slab=str(slab), valid=False, reason="Model could not be fitted"))
                    continue

                model_df = modeled['model_df']
                coefficients = modeled['coefficients']
                stage2_model = modeled['stage2_model']
                base_price_series = pd.to_numeric(model_df.get('base_price', pd.Series(dtype=float)), errors='coerce').dropna()
                default_cogs = float(np.round(float(base_price_series.iloc[0]) * 0.5)) if not base_price_series.empty else 0.0
                cogs_per_unit = float(request.cogs_per_unit) if getattr(request, 'cogs_per_unit', None) is not None else default_cogs
                cogs_per_unit = max(cogs_per_unit, 0.0)
                coefficients['cogs_per_unit'] = float(cogs_per_unit)

                predicted_vs_actual = [
                    ModelingPoint(
                        period=row['Period'].to_pydatetime() if hasattr(row['Period'], 'to_pydatetime') else row['Period'],
                        actual_quantity=float(row['quantity']),
                        predicted_quantity=float(row['predicted_quantity']),
                        predicted_quantity_ols=float(row['predicted_quantity_ols']) if pd.notna(row.get('predicted_quantity_ols', np.nan)) else None,
                        actual_discount_pct=float(row['actual_discount_pct']),
                        base_discount_pct=float(row['base_discount_pct']),
                        tactical_discount_pct=float(row['tactical_discount_pct']),
                        roi_1mo=float(row['roi_1mo']) if pd.notna(row['roi_1mo']) else None,
                        spend=float(row['spend']) if pd.notna(row['spend']) else None,
                        incremental_revenue=float(row['incremental_revenue']) if pd.notna(row['incremental_revenue']) else None,
                    )
                    for _, row in model_df.iterrows()
                ]

                roi_points, roi_summary = self._build_structural_roi_points(
                    model_df,
                    stage2_model,
                    round_step=float(request.round_step),
                    cogs_per_unit=float(cogs_per_unit),
                )
                avg_roi = float(roi_summary.get('avg_roi_1mo', 0.0))
                slab_spend = float(roi_summary.get('total_spend', 0.0))
                slab_incremental = float(roi_summary.get('total_incremental_revenue', 0.0))
                slab_incremental_profit = float(roi_summary.get('total_incremental_profit', 0.0))
                slab_qty = float(model_df['quantity'].sum())

                weighted_qty += slab_qty
                weighted_r2 += slab_qty * float(coefficients.get('stage2_r2', 0.0))
                total_spend += slab_spend
                total_incremental_revenue += slab_incremental

                slab_results.append(
                    ModelingSlabResult(
                        slab=str(slab),
                        valid=True,
                        reason=None,
                        model_coefficients=coefficients,
                        predicted_vs_actual=predicted_vs_actual,
                        roi_points=roi_points,
                        summary={
                            'avg_roi_1mo': avg_roi,
                            'total_spend': slab_spend,
                            'total_incremental_revenue': slab_incremental,
                            'total_incremental_profit': slab_incremental_profit,
                            'avg_base_discount_pct': float(model_df['base_discount_pct'].mean()),
                            'periods': float(len(model_df)),
                            'structural_episodes': float(roi_summary.get('episodes', 0.0)),
                            'structural_roi_1mo': float(roi_summary.get('structural_roi_1mo', 0.0)),
                            'structural_profit_roi_1mo': float(roi_summary.get('structural_profit_roi_1mo', 0.0)),
                            'cogs_per_unit': float(cogs_per_unit),
                        },
                    )
                )

            combined_summary = {
                'weighted_stage2_r2': float(weighted_r2 / weighted_qty) if weighted_qty > 0 else 0.0,
                'total_spend': total_spend,
                'total_incremental_revenue': total_incremental_revenue,
                'combined_roi_1mo': float(total_incremental_revenue / total_spend) if total_spend > 0 else 0.0,
                'valid_slabs': float(len([x for x in slab_results if x.valid])),
            }

            return ModelingResponse(
                success=True,
                message="Modeling completed successfully",
                slab_results=slab_results,
                combined_summary=combined_summary,
            )
        except Exception as e:
            return ModelingResponse(
                success=False,
                message=f"Error in modeling: {str(e)}",
                slab_results=[],
                combined_summary={},
            )

    async def calculate_12_month_planner(self, request: PlannerRequest) -> PlannerResponse:
        try:
            def _safe_num(value, default=0.0):
                try:
                    fv = float(value)
                    return fv if np.isfinite(fv) else float(default)
                except Exception:
                    return float(default)

            scope = self._build_step2_scope(request)
            if scope is None:
                return PlannerResponse(success=False, message="No data matches selected filters")

            df_scope = scope['df_scope']
            if df_scope.empty:
                return PlannerResponse(success=False, message="No data for planner")

            if 'Slab' in df_scope.columns:
                available_slabs = sorted(
                    df_scope['Slab'].dropna().astype(str).unique().tolist(),
                    key=self._slab_sort_key
                )
            else:
                available_slabs = ['All']

            selected_slab = str(request.slab or '').strip()
            if not selected_slab:
                selected_from_list = [str(s) for s in (request.slabs or [])]
                if selected_from_list:
                    selected_slab = selected_from_list[0]
            if not selected_slab:
                selected_slab = available_slabs[0] if available_slabs else 'All'

            slab_df = df_scope.copy()
            if 'Slab' in slab_df.columns:
                slab_df = slab_df[slab_df['Slab'].astype(str) == selected_slab]

            if slab_df.empty:
                return PlannerResponse(success=False, message=f"No data for slab {selected_slab}", slab=selected_slab)

            model_request = ModelingRequest(
                run_id=request.run_id,
                states=request.states,
                categories=request.categories,
                subcategories=request.subcategories,
                brands=request.brands,
                sizes=request.sizes,
                recency_threshold=request.recency_threshold,
                frequency_threshold=request.frequency_threshold,
                round_step=request.round_step,
                min_upward_jump_pp=request.min_upward_jump_pp,
                min_downward_drop_pp=request.min_downward_drop_pp,
                include_lag_discount=request.include_lag_discount,
                l2_penalty=request.l2_penalty,
                optimize_l2_penalty=request.optimize_l2_penalty,
                constraint_residual_non_negative=request.constraint_residual_non_negative,
                constraint_structural_non_negative=request.constraint_structural_non_negative,
                constraint_tactical_non_negative=request.constraint_tactical_non_negative,
                constraint_lag_non_positive=request.constraint_lag_non_positive,
                rfm_segments=request.rfm_segments,
                outlet_classifications=request.outlet_classifications,
                slabs=[selected_slab],
                outlet_ids=request.outlet_ids,
            )

            monthly = self._build_monthly_model_dataframe(slab_df, model_request)
            if monthly.empty or len(monthly) < 3:
                return PlannerResponse(success=False, message="Not enough monthly points for planner", slab=selected_slab)

            modeled = self._run_two_stage_model(
                monthly,
                include_lag_discount=bool(request.include_lag_discount),
                l2_penalty=float(getattr(request, 'l2_penalty', 1.0)),
                optimize_l2_penalty=bool(getattr(request, 'optimize_l2_penalty', False)),
                constraint_residual_non_negative=bool(getattr(request, 'constraint_residual_non_negative', True)),
                constraint_structural_non_negative=bool(getattr(request, 'constraint_structural_non_negative', True)),
                constraint_tactical_non_negative=bool(getattr(request, 'constraint_tactical_non_negative', True)),
                constraint_lag_non_positive=bool(getattr(request, 'constraint_lag_non_positive', True)),
            )
            if modeled is None:
                return PlannerResponse(success=False, message="Model could not be fitted for planner", slab=selected_slab)

            model_df = modeled['model_df'].sort_values('Period').reset_index(drop=True).copy()
            coefficients = modeled['coefficients']
            stage2_model = modeled['stage2_model']

            model_df['Period'] = pd.to_datetime(model_df['Period'], errors='coerce')
            model_df = model_df.dropna(subset=['Period']).reset_index(drop=True)
            if model_df.empty:
                return PlannerResponse(success=False, message="No clean monthly periods for planner", slab=selected_slab)

            model_df['Month_Key'] = model_df['Period'].dt.to_period('M')
            latest_period = pd.to_datetime(model_df['Period'].max())
            if request.plan_start_year is not None:
                plan_start = pd.Period(f"{int(request.plan_start_year)}-04", freq='M')
            else:
                start_year = int(latest_period.year) if int(latest_period.month) >= 4 else int(latest_period.year) - 1
                plan_start = pd.Period(f"{start_year}-04", freq='M')

            plan_months = list(pd.period_range(start=plan_start, periods=12, freq='M'))
            if len(plan_months) < 3:
                return PlannerResponse(success=False, message="Need at least 3 months for planning", slab=selected_slab)

            rows = []
            for m in plan_months:
                mm = model_df[model_df['Month_Key'] == m]
                if mm.empty:
                    rows.append({
                        'Month_Key': m,
                        'Period': m.to_timestamp(how='start'),
                        'base_price': np.nan,
                        'actual_discount_pct': np.nan,
                        'store_count': np.nan,
                        'residual_store': np.nan,
                        'base_discount_pct': np.nan,
                    })
                else:
                    r = mm.iloc[-1]
                    rows.append({
                        'Month_Key': m,
                        'Period': m.to_timestamp(how='start'),
                        'base_price': float(r['base_price']) if pd.notna(r['base_price']) else np.nan,
                        'actual_discount_pct': float(r['actual_discount_pct']) if pd.notna(r['actual_discount_pct']) else np.nan,
                        'store_count': float(r['store_count']) if pd.notna(r['store_count']) else np.nan,
                        'residual_store': float(r['residual_store']) if pd.notna(r['residual_store']) else np.nan,
                        'base_discount_pct': float(r['base_discount_pct']) if pd.notna(r['base_discount_pct']) else np.nan,
                    })

            plan_template = pd.DataFrame(rows).sort_values('Period').reset_index(drop=True)
            if plan_template['base_discount_pct'].isna().all():
                plan_template['base_discount_pct'] = 0.0
            else:
                plan_template['base_discount_pct'] = plan_template['base_discount_pct'].ffill().bfill()
            plan_template['base_discount_pct'] = self._round_discount_series(
                plan_template['base_discount_pct'], step=float(request.round_step)
            )

            base_price_series = pd.to_numeric(model_df['base_price'], errors='coerce').dropna()
            default_bp = float(base_price_series.iloc[-1]) if not base_price_series.empty else 100.0
            default_bp = float(np.round(default_bp * 2.0) / 2.0)
            plan_template['base_price'] = pd.to_numeric(plan_template['base_price'], errors='coerce').fillna(default_bp)

            missing_resid = plan_template['residual_store'].isna()
            if missing_resid.any():
                stage1_intercept = float(coefficients.get('stage1_intercept', 0.0))
                stage1_coef_discount = float(coefficients.get('stage1_coef_discount', 0.0))
                fallback_discount = plan_template['actual_discount_pct'].fillna(plan_template['base_discount_pct']).to_numpy(dtype=float)
                fallback_store = plan_template['store_count'].fillna(0.0).to_numpy(dtype=float)
                pred_store = stage1_intercept + stage1_coef_discount * fallback_discount
                fallback_resid = fallback_store - pred_store
                plan_template.loc[missing_resid, 'residual_store'] = fallback_resid[missing_resid.to_numpy()]
            plan_template['residual_store'] = plan_template['residual_store'].fillna(0.0)

            observed_struct = plan_template['base_discount_pct'].to_numpy(dtype=float)

            prev_month = plan_months[0] - 1
            prev_row = model_df[model_df['Month_Key'] == prev_month]
            if not prev_row.empty and pd.notna(prev_row['base_discount_pct'].iloc[-1]):
                prev_struct = float(prev_row['base_discount_pct'].iloc[-1])
            else:
                prev_struct = float(observed_struct[0])
            prev_struct = float(self._round_discount_series(pd.Series([prev_struct]), step=float(request.round_step)).iloc[0])

            ref_map = {
                mk: float(v)
                for mk, v in zip(
                    model_df['Month_Key'],
                    pd.to_numeric(model_df['base_discount_pct'], errors='coerce')
                )
                if pd.notna(v)
            }
            default_struct = np.empty(len(plan_months), dtype=float)
            valid_ref_count = sum(1 for m in plan_months if (m - 12) in ref_map)
            if valid_ref_count >= 12:
                default_struct = np.asarray(
                    [float(ref_map.get(m - 12, observed_struct[i])) for i, m in enumerate(plan_months)],
                    dtype=float
                )
            else:
                default_struct[0] = prev_struct
                if len(default_struct) > 1:
                    default_struct[1:] = observed_struct[:-1]
            default_struct = np.clip(default_struct, 0.0, 60.0)
            default_struct = self._round_discount_series(default_struct, step=float(request.round_step)).to_numpy(dtype=float)

            n_plan = len(plan_months)
            planned_struct = request.planned_structural_discounts or []
            if len(planned_struct) != n_plan:
                planned_struct_arr = default_struct.copy()
            else:
                planned_struct_arr = np.asarray(planned_struct, dtype=float)
                planned_struct_arr = np.clip(planned_struct_arr, 0.0, 60.0)
                planned_struct_arr = self._round_discount_series(planned_struct_arr, step=float(request.round_step)).to_numpy(dtype=float)

            planned_base = request.planned_base_prices or []
            if len(planned_base) != n_plan:
                planned_base_arr = plan_template['base_price'].to_numpy(dtype=float)
            else:
                planned_base_arr = np.asarray(planned_base, dtype=float)
            planned_base_arr = np.maximum(planned_base_arr, 0.0)
            planned_base_arr = np.round(planned_base_arr * 2.0) / 2.0

            lag_old = np.empty(n_plan, dtype=float)
            lag_new = np.empty(n_plan, dtype=float)
            lag_old[0] = prev_struct
            lag_new[0] = prev_struct
            if n_plan > 1:
                lag_old[1:] = default_struct[:-1]
                lag_new[1:] = planned_struct_arr[:-1]

            residual_arr = plan_template['residual_store'].to_numpy(dtype=float)
            zeros_arr = np.zeros(n_plan, dtype=float)

            qty_old = self._predict_stage2_quantity(stage2_model, residual_arr, default_struct, zeros_arr, lag_old)
            qty_new = self._predict_stage2_quantity(stage2_model, residual_arr, planned_struct_arr, zeros_arr, lag_new)
            qty_old = np.maximum(qty_old, 0.0)
            qty_new = np.maximum(qty_new, 0.0)

            price_old = planned_base_arr * (1.0 - default_struct / 100.0)
            price_new = planned_base_arr * (1.0 - planned_struct_arr / 100.0)
            rev_old = qty_old * price_old
            rev_new = qty_new * price_new

            cogs_default = float(np.round(default_bp * 0.5))
            cogs_per_unit = float(request.cogs_per_unit) if request.cogs_per_unit is not None else cogs_default
            cogs_per_unit = max(cogs_per_unit, 0.0)
            prof_old = qty_old * (price_old - cogs_per_unit)
            prof_new = qty_new * (price_new - cogs_per_unit)

            total_qty_old = float(np.nansum(qty_old))
            total_qty_new = float(np.nansum(qty_new))
            total_rev_old = float(np.nansum(rev_old))
            total_rev_new = float(np.nansum(rev_new))
            total_prof_old = float(np.nansum(prof_old))
            total_prof_new = float(np.nansum(prof_new))

            qty_delta = total_qty_new - total_qty_old
            rev_delta = total_rev_new - total_rev_old
            prof_delta = total_prof_new - total_prof_old

            qty_pct = float((qty_delta / total_qty_old) * 100.0) if abs(total_qty_old) > 1e-12 else np.nan
            rev_pct = float((rev_delta / total_rev_old) * 100.0) if abs(total_rev_old) > 1e-12 else np.nan
            prof_pct = float((prof_delta / total_prof_old) * 100.0) if abs(total_prof_old) > 1e-12 else np.nan

            avg_promo_old = float(np.nanmean(default_struct))
            avg_promo_new = float(np.nanmean(planned_struct_arr))
            promo_delta_pp = avg_promo_new - avg_promo_old
            promo_pct = float((promo_delta_pp / avg_promo_old) * 100.0) if abs(avg_promo_old) > 1e-12 else np.nan

            step_up_pp = np.clip(planned_struct_arr - default_struct, 0.0, None)
            spend_monthly = planned_base_arr * (step_up_pp / 100.0) * qty_new
            total_spend_plan = float(np.nansum(spend_monthly))
            roi_revenue = float(rev_delta / total_spend_plan) if total_spend_plan > 1e-12 else np.nan
            roi_revenue_pct = float(roi_revenue * 100.0) if np.isfinite(roi_revenue) else np.nan

            metrics = {
                'volume_change_pct': _safe_num(qty_pct, 0.0),
                'revenue_change_pct': _safe_num(rev_pct, 0.0),
                'profit_change_pct': _safe_num(prof_pct, 0.0),
                'promo_change_pct': _safe_num(promo_pct, 0.0),
                'roi_change_pct': _safe_num(roi_revenue_pct, 0.0),
                'roi_revenue_x': _safe_num(roi_revenue, 0.0),
                'total_spend_plan': _safe_num(total_spend_plan, 0.0),
                'total_revenue_current': _safe_num(total_rev_old, 0.0),
                'total_revenue_planned': _safe_num(total_rev_new, 0.0),
                'total_profit_current': _safe_num(total_prof_old, 0.0),
                'total_profit_planned': _safe_num(total_prof_new, 0.0),
                'total_quantity_current': _safe_num(total_qty_old, 0.0),
                'total_quantity_planned': _safe_num(total_qty_new, 0.0),
                'revenue_delta': _safe_num(rev_delta, 0.0),
                'profit_delta': _safe_num(prof_delta, 0.0),
                'quantity_delta': _safe_num(qty_delta, 0.0),
            }

            series = [
                PlannerMonthPoint(
                    period=plan_template['Period'].iloc[i].to_pydatetime() if hasattr(plan_template['Period'].iloc[i], 'to_pydatetime') else plan_template['Period'].iloc[i],
                    current_promo_pct=_safe_num(default_struct[i], 0.0),
                    planned_promo_pct=_safe_num(planned_struct_arr[i], 0.0),
                    base_price=_safe_num(planned_base_arr[i], 0.0),
                    current_quantity=_safe_num(qty_old[i], 0.0),
                    planned_quantity=_safe_num(qty_new[i], 0.0),
                    current_revenue=_safe_num(rev_old[i], 0.0),
                    planned_revenue=_safe_num(rev_new[i], 0.0),
                )
                for i in range(n_plan)
            ]

            default_base_prices = [_safe_num(x, 0.0) for x in plan_template['base_price'].to_list()]
            series_rows = [
                {
                    "month": str(plan_months[i]),
                    "current_promo_pct": _safe_num(default_struct[i], 0.0),
                    "planned_promo_pct": _safe_num(planned_struct_arr[i], 0.0),
                    "base_price": _safe_num(planned_base_arr[i], 0.0),
                    "current_quantity": _safe_num(qty_old[i], 0.0),
                    "planned_quantity": _safe_num(qty_new[i], 0.0),
                    "current_revenue": _safe_num(rev_old[i], 0.0),
                    "planned_revenue": _safe_num(rev_new[i], 0.0),
                }
                for i in range(n_plan)
            ]
            recalculate_requested = (
                request.planned_structural_discounts is not None
                or request.planned_base_prices is not None
            )
            if recalculate_requested:
                ai_payload = self._generate_planner_ai_insights(
                    slab=selected_slab,
                    months=[str(m) for m in plan_months],
                    default_structural=[_safe_num(x, 0.0) for x in default_struct.tolist()],
                    planned_structural=[_safe_num(x, 0.0) for x in planned_struct_arr.tolist()],
                    default_base_prices=default_base_prices,
                    planned_base_prices=[_safe_num(x, 0.0) for x in planned_base_arr.tolist()],
                    metrics=metrics,
                    series_rows=series_rows,
                )
            else:
                ai_payload = {
                    "status": "pending_recalculate",
                    "text": "Trinity Insights will appear after you click Recalculate Plan.",
                }

            return PlannerResponse(
                success=True,
                message="Step 4 planner calculated successfully",
                slab=selected_slab,
                plan_start_month=str(plan_months[0]),
                months=[str(m) for m in plan_months],
                default_structural_discounts=[_safe_num(x, 0.0) for x in default_struct.tolist()],
                current_structural_discounts=[_safe_num(x, 0.0) for x in default_struct.tolist()],
                planned_structural_discounts=[_safe_num(x, 0.0) for x in planned_struct_arr.tolist()],
                planned_base_prices=[_safe_num(x, 0.0) for x in planned_base_arr.tolist()],
                cogs_per_unit=_safe_num(cogs_per_unit, 0.0),
                model_coefficients=coefficients,
                metrics=metrics,
                series=series,
                ai_insights_status=ai_payload.get("status"),
                ai_insights=ai_payload.get("text"),
            )
        except Exception as e:
            return PlannerResponse(
                success=False,
                message=f"Error in step 4 planner: {str(e)}",
                slab=request.slab,
            )
    
    def calculate_rfm_metrics(
        self, 
        df: pd.DataFrame, 
        recency_days: int = 90,
        frequency_threshold: int = 20
    ):
        """Calculate RFM metrics with clustering"""
        
        max_date = df['Date'].max()
        
        # Create order-level data
        order_level = df.groupby(['Outlet_ID', 'Date', 'Bill_No', 'Final_State']).agg({
            'Quantity': 'sum',
            'Net_Amt': 'sum'
        }).reset_index()
        
        # Calculate RFM at outlet level
        rfm = order_level.groupby(['Outlet_ID', 'Final_State']).agg({
            'Date': ['min', 'max', 'nunique'],
            'Bill_No': 'count',
            'Net_Amt': 'mean'
        }).reset_index()
        
        rfm.columns = ['Outlet_ID', 'Final_State', 'first_order', 'last_order', 
                       'unique_order_days', 'orders_count', 'AOV']
        
        # Calculate Recency
        rfm['Recency_days'] = (max_date - rfm['last_order']).dt.days
        rfm['Recency_flag'] = (rfm['Recency_days'] <= recency_days).astype(int)
        rfm['R_label'] = rfm['Recency_flag'].map({1: 'Recent', 0: 'Stale'})
        
        # Calculate Frequency
        rfm['active_days'] = (rfm['last_order'] - rfm['first_order']).dt.days + 1
        rfm['active_days'] = rfm['active_days'].clip(lower=1)
        rfm['orders_per_day'] = rfm['orders_count'] / rfm['active_days']
        
        # Fixed frequency rule
        rfm['F_label'] = np.where(rfm['unique_order_days'] >= frequency_threshold, 'High', 'Low')
        rfm['F_cluster_id'] = np.where(rfm['unique_order_days'] >= frequency_threshold, 1, 0)
        
        # Cluster Monetary (M) using K-means
        valid_m = rfm['AOV'].notna()
        if valid_m.sum() >= 10:
            m_values = np.log1p(rfm.loc[valid_m, 'AOV'].values).reshape(-1, 1)
            scaler_m = StandardScaler()
            m_scaled = scaler_m.fit_transform(m_values)
            
            kmeans_m = KMeans(n_clusters=2, random_state=42, n_init=20)
            m_clusters = kmeans_m.fit_predict(m_scaled)
            
            cluster_means = [rfm.loc[valid_m, 'AOV'][m_clusters == i].mean() for i in range(2)]
            high_cluster_m = np.argmax(cluster_means)
            
            rfm.loc[valid_m, 'M_cluster_id'] = m_clusters
            rfm.loc[valid_m, 'M_label'] = ['High' if c == high_cluster_m else 'Low' for c in m_clusters]
        else:
            median_m = rfm['AOV'].median()
            rfm['M_label'] = rfm['AOV'].apply(lambda x: 'High' if x >= median_m else 'Low')
            rfm['M_cluster_id'] = np.nan
        
        # Ensure semantic consistency
        m_means = rfm.groupby('M_label', dropna=False)['AOV'].mean()
        if 'High' in m_means.index and 'Low' in m_means.index:
            if pd.notna(m_means['High']) and pd.notna(m_means['Low']) and m_means['High'] < m_means['Low']:
                rfm['M_label'] = rfm['M_label'].replace({'High': 'Low', 'Low': 'High'})
        
        # Create RFM Segment
        rfm['RFM_Segment'] = rfm['R_label'] + '-' + rfm['F_label'] + '-' + rfm['M_label']
        
        # Cluster summaries
        freq_cluster_summary = (
            rfm.groupby('F_label', dropna=False)['orders_per_day']
            .agg(['count', 'min', 'max', 'mean'])
            .reset_index()
            .rename(columns={
                'F_label': 'Frequency_Cluster',
                'count': 'Outlets',
                'min': 'Min_Orders_Per_Day',
                'max': 'Max_Orders_Per_Day',
                'mean': 'Mean_Orders_Per_Day'
            })
            .sort_values('Frequency_Cluster')
        )
        
        monetary_cluster_summary = (
            rfm.groupby('M_label', dropna=False)['AOV']
            .agg(['count', 'min', 'max', 'mean'])
            .reset_index()
            .rename(columns={
                'M_label': 'Monetary_Cluster',
                'count': 'Outlets',
                'min': 'Min_AOV',
                'max': 'Max_AOV',
                'mean': 'Mean_AOV'
            })
            .sort_values('Monetary_Cluster')
        )
        
        cluster_summary = {
            'frequency': freq_cluster_summary.to_dict('records'),
            'monetary': monetary_cluster_summary.to_dict('records')
        }
        
        return rfm, max_date, cluster_summary
    
    async def calculate_rfm(self, request: RFMRequest) -> RFMResponse:
        """Calculate RFM based on request filters"""
        try:
            if self.data_cache is None:
                return RFMResponse(
                    success=False,
                    message="Data not loaded. Please check data files.",
                    total_outlets=0
                )

            dataset = self._build_rfm_dataset(request)
            if dataset is None:
                return RFMResponse(
                    success=False,
                    message="No data matches the selected filters",
                    total_outlets=0
                )

            rfm = dataset['rfm']
            paged_rfm, total_filtered, total_pages, page, page_size = self._apply_outlet_query(rfm, request)
            rfm_list = [self._to_outlet_model(row) for _, row in paged_rfm.iterrows()]

            return RFMResponse(
                success=True,
                message="RFM calculation completed successfully",
                rfm_data=rfm_list,
                segment_summary=dataset['segment_summaries'],
                cluster_summary=ClusterSummary(**dataset['cluster_summary']),
                max_date=dataset['max_date'],
                total_outlets=len(rfm),
                total_filtered_outlets=total_filtered,
                total_pages=total_pages,
                page=page,
                page_size=page_size,
                input_rows=dataset['input_rows'],
                input_outlets=dataset['input_outlets']
            )
            
        except Exception as e:
            return RFMResponse(
                success=False,
                message=f"Error calculating RFM: {str(e)}",
                total_outlets=0
            )

    async def export_rfm_csv(self, request: RFMRequest) -> str:
        dataset = self._build_rfm_dataset(request)
        if dataset is None:
            return ""

        rfm = dataset['rfm'].copy()

        sort_key_map = {
            'outlet_id': 'Outlet_ID',
            'final_state': 'Final_State',
            'rfm_segment': 'RFM_Segment',
            'total_net_amt': 'Total_Net_Amt',
            'orders_count': 'orders_count',
            'aov': 'AOV',
            'recency_days': 'Recency_days',
            'unique_order_days': 'unique_order_days',
            'orders_per_day': 'orders_per_day',
        }
        sort_col = sort_key_map.get((request.sort_key or 'total_net_amt').lower(), 'Total_Net_Amt')
        ascending = (request.sort_direction or 'desc').lower() == 'asc'
        search = (request.search or '').strip().lower()
        if search:
            rfm = rfm[
                rfm['Outlet_ID'].astype(str).str.lower().str.contains(search, na=False) |
                rfm['Final_State'].astype(str).str.lower().str.contains(search, na=False) |
                rfm['RFM_Segment'].astype(str).str.lower().str.contains(search, na=False)
            ]
        rfm = rfm.sort_values(by=sort_col, ascending=ascending, kind='mergesort')

        export_df = rfm[[
            'Outlet_ID', 'Final_State', 'RFM_Segment', 'Total_Net_Amt',
            'orders_count', 'AOV', 'Recency_days', 'unique_order_days',
            'orders_per_day', 'first_order', 'last_order'
        ]].copy()
        export_df.columns = [
            'Outlet_ID', 'State', 'RFM_Segment', 'Total_Net_Amt',
            'Orders', 'AOV', 'Recency_Days', 'Unique_Order_Days',
            'Orders_Per_Day', 'First_Order', 'Last_Order'
        ]
        return export_df.to_csv(index=False)

    async def calculate_base_depth(self, request: BaseDepthRequest) -> BaseDepthResponse:
        """Estimate base discount depth from filtered transactional data."""
        try:
            scope = self._build_step2_scope(request)
            if scope is None:
                return BaseDepthResponse(
                    success=False,
                    message="Data not loaded. Please check data files.",
                    points=[],
                    summary={}
                )

            df = scope['df_scope']
            if df.empty:
                return BaseDepthResponse(
                    success=False,
                    message="No data matches the selected step-2 filters",
                    points=[],
                    summary={}
                )

            required_cols = ['Date', 'TotalDiscount', 'SalesValue_atBasicRate', 'Quantity']
            missing_cols = [c for c in required_cols if c not in df.columns]
            if missing_cols:
                return BaseDepthResponse(
                    success=False,
                    message=f"Missing required columns for estimator: {', '.join(missing_cols)}",
                    points=[],
                    summary={}
                )

            points, summary = self._compute_base_depth_result(df, request)

            slab_results = []
            if 'Slab' in df.columns:
                if request.slabs:
                    slab_scope = [str(s) for s in request.slabs]
                else:
                    slab_scope = sorted(df['Slab'].dropna().astype(str).unique().tolist())

                if len(slab_scope) > 1:
                    for slab in slab_scope:
                        slab_df = df[df['Slab'].astype(str) == str(slab)].copy()
                        if slab_df.empty:
                            slab_results.append({
                                "slab": str(slab),
                                "success": False,
                                "message": "No data for this slab with current filters",
                                "points": [],
                                "summary": {},
                            })
                            continue

                        slab_points, slab_summary = self._compute_base_depth_result(slab_df, request)
                        slab_results.append({
                            "slab": str(slab),
                            "success": True,
                            "message": "Base depth estimation completed successfully",
                            "points": slab_points,
                            "summary": slab_summary,
                        })

            return BaseDepthResponse(
                success=True,
                message="Base depth estimation completed successfully",
                points=points,
                summary=summary,
                slab_results=slab_results,
            )
        except Exception as e:
            return BaseDepthResponse(
                success=False,
                message=f"Error estimating base depth: {str(e)}",
                points=[],
                summary={},
                slab_results=[],
            )
