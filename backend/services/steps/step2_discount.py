"""Step 2 discount analysis and slab/base-depth computation.\n\nThis module owns slab assignment modes and base-depth estimation outputs used\ndownstream by modeling.\n"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
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
from io import BytesIO
from datetime import datetime, date
import urllib.request
import urllib.error
try:
    from statsmodels.tsa.holtwinters import Holt
except Exception:
    Holt = None

from models.rfm_models import (
    RFMRequest, RFMResponse, OutletRFM,
    SegmentSummary, ClusterSummary,
    BaseDepthRequest, BaseDepthResponse, BaseDepthPoint,
    DiscountOptionsRequest, DiscountOptionsResponse,
    ModelingRequest, ModelingResponse, ModelingSlabResult, ModelingPoint,
    PlannerRequest, PlannerResponse, PlannerMonthPoint,
    PlannerScenarioComparisonResponse, PlannerScenarioComparisonRow,
    CrossSizePlannerRequest, CrossSizePlannerResponse, CrossSizePlannerSizeResult, CrossSizePlannerSlabState,
    BaselineForecastRequest, BaselineForecastResponse, BaselineForecastPoint,
    EDARequest, EDAResponse, EDAProductOption, EDAProductContribution,
    EDAContributionRow, EDAOptionsResponse
)


class Step2DiscountMixin:

    def _resolve_step2_discount_columns(self, df: pd.DataFrame) -> Dict[str, Optional[str]]:
        cols = list(df.columns)
        colset = set(cols)
        return {
            "scheme_col": "Scheme_Discount" if "Scheme_Discount" in colset else ("scheme_discount" if "scheme_discount" in colset else None),
            "qps_col": "Staggered_qps" if "Staggered_qps" in colset else ("staggered_qps" if "staggered_qps" in colset else None),
            "dsp_col": (
                "Basic_Rate_Per_PC_without_GST"
                if "Basic_Rate_Per_PC_without_GST" in colset
                else (
                    "basic_rate_per_pc_without_gst"
                    if "basic_rate_per_pc_without_gst" in colset
                    else (
                        "Basic_Rate_Per_PC"
                        if "Basic_Rate_Per_PC" in colset
                        else ("basic_rate_per_pc" if "basic_rate_per_pc" in colset else None)
                    )
                )
            ),
            "clp_col": (
                "Selling_Rate_Per_PC_without_GST_CLP"
                if "Selling_Rate_Per_PC_without_GST_CLP" in colset
                else (
                    "selling_rate_per_pc_without_gst_clp"
                    if "selling_rate_per_pc_without_gst_clp" in colset
                    else None
                )
            ),
        }

    def _prepare_step2_discount_basis(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        work = df.copy()
        colmap = self._resolve_step2_discount_columns(work)
        dsp_col = colmap.get("dsp_col")
        # Prefer Basic_Rate_Per_PC_without_GST when present with usable values,
        # otherwise fallback to Basic_Rate_Per_PC.
        if dsp_col in {"Basic_Rate_Per_PC_without_GST", "basic_rate_per_pc_without_gst"}:
            alt_dsp = "Basic_Rate_Per_PC" if "Basic_Rate_Per_PC" in work.columns else ("basic_rate_per_pc" if "basic_rate_per_pc" in work.columns else None)
            if alt_dsp is not None:
                dsp_series = pd.to_numeric(work[dsp_col], errors='coerce').fillna(0.0)
                if float(dsp_series.abs().sum()) <= 1e-12:
                    dsp_col = alt_dsp
        colmap["dsp_col"] = dsp_col

        missing = []
        if 'Quantity' not in work.columns:
            missing.append('Quantity')
        if colmap.get("scheme_col") is None:
            missing.append('Scheme_Discount')
        if colmap.get("qps_col") is None:
            missing.append('Staggered_qps')
        if colmap.get("dsp_col") is None:
            missing.append('Basic_Rate_Per_PC_without_GST (or Basic_Rate_Per_PC)')
        if missing:
            return work, missing

        qty = pd.to_numeric(work['Quantity'], errors='coerce').fillna(0.0)
        scheme = pd.to_numeric(work[colmap["scheme_col"]], errors='coerce').fillna(0.0)
        qps = pd.to_numeric(work[colmap["qps_col"]], errors='coerce').fillna(0.0)
        dsp = pd.to_numeric(work[colmap["dsp_col"]], errors='coerce').fillna(0.0)
        clp_col = colmap.get("clp_col")
        if clp_col is not None:
            clp = pd.to_numeric(work[clp_col], errors='coerce').fillna(0.0)
            clp = clp.where(clp > 0, dsp)
        else:
            clp = dsp

        work['Quantity'] = qty
        work['_step2_scheme_amount'] = scheme + qps
        work['_step2_dsp_sales'] = qty * dsp
        work['_step2_clp_sales'] = qty * clp
        return work, []

    def _build_summary_by_slab(self, df_scope: pd.DataFrame) -> List[Dict[str, Any]]:
        if df_scope is None or df_scope.empty or 'Slab' not in df_scope.columns:
            return []

        work, missing = self._prepare_step2_discount_basis(df_scope)
        if missing:
            return []
        outlet_col = 'Outlet_ID' if 'Outlet_ID' in work.columns else ('Store_ID' if 'Store_ID' in work.columns else None)
        if outlet_col is None:
            return []

        invoice_col = None
        for candidate in ['Invoice_No', 'Bill_No', 'Sales_Order_No']:
            if candidate in work.columns:
                invoice_col = candidate
                break
        if invoice_col is None:
            return []

        work['Slab'] = work['Slab'].astype(str)
        if 'Sizes' in work.columns:
            work['Size_Key'] = work['Sizes'].astype(str).map(self._normalize_step2_size_key)
        else:
            work['Size_Key'] = ''

        date_part = pd.to_datetime(work.get('Date'), errors='coerce').dt.strftime('%Y-%m-%d').fillna('')
        work['Invoice_Key'] = (
            work[outlet_col].astype(str).fillna('')
            + '|'
            + work[invoice_col].astype(str).fillna('')
            + '|'
            + date_part
        )

        group_cols = ['Slab']
        if work['Size_Key'].astype(str).str.len().gt(0).any():
            group_cols = ['Size_Key', 'Slab']

        slab_summary = (
            work.groupby(group_cols, as_index=False)
            .agg(
                Outlets=(outlet_col, 'nunique'),
                Invoices=('Invoice_Key', 'nunique'),
                Quantity=('Quantity', 'sum'),
                AOQ=('Quantity', 'mean'),
                Sales_Value=('_step2_dsp_sales', 'sum'),
                Total_Discount=('_step2_scheme_amount', 'sum'),
            )
        )
        slab_summary['AOV'] = (
            slab_summary['Sales_Value'] / slab_summary['Invoices'].replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan)
        slab_summary['AOQ'] = pd.to_numeric(slab_summary['AOQ'], errors='coerce')
        slab_summary['Discount_Pct'] = (
            slab_summary['Total_Discount'] / slab_summary['Sales_Value'].replace(0, np.nan) * 100.0
        ).replace([np.inf, -np.inf], np.nan)

        slab_criteria = (
            work.groupby(group_cols, as_index=False)
            .agg(
                Min_Qty=('Quantity', 'min'),
                Max_Qty=('Quantity', 'max'),
            )
        )
        slab_summary = slab_summary.merge(slab_criteria, on=group_cols, how='left')

        def _label_with_criteria(row):
            slab_name = str(row.get('Slab', '') or '')
            qmin = pd.to_numeric(row.get('Min_Qty', np.nan), errors='coerce')
            qmax = pd.to_numeric(row.get('Max_Qty', np.nan), errors='coerce')
            if pd.isna(qmin) or pd.isna(qmax):
                return slab_name
            if abs(float(qmax) - float(qmin)) < 1e-9:
                criteria = f"Qty={float(qmin):g}"
            else:
                criteria = f"Qty {float(qmin):g}-{float(qmax):g}"
            return f"{slab_name} ({criteria})"

        slab_summary['Slab_Raw'] = slab_summary['Slab'].astype(str)
        slab_summary['Slab'] = slab_summary.apply(_label_with_criteria, axis=1)

        total_invoices = float(pd.to_numeric(slab_summary['Invoices'], errors='coerce').fillna(0.0).sum())
        total_sales = float(pd.to_numeric(slab_summary['Sales_Value'], errors='coerce').fillna(0.0).sum())
        slab_summary['Invoice_Contribution_%'] = (
            slab_summary['Invoices'] / total_invoices * 100.0
        ) if total_invoices > 0 else 0.0
        slab_summary['Sales_Contribution_%'] = (
            slab_summary['Sales_Value'] / total_sales * 100.0
        ) if total_sales > 0 else 0.0

        sort_cols = ['Slab_Raw']
        if 'Size_Key' in slab_summary.columns:
            sort_cols = ['Size_Key', 'Slab_Raw']
        slab_summary = slab_summary.sort_values(
            by=sort_cols,
            key=lambda s: s.map(self._slab_sort_key) if s.name == 'Slab_Raw' else s,
            kind='mergesort'
        )

        for col in ['AOQ', 'AOV', 'Discount_Pct', 'Invoice_Contribution_%', 'Sales_Contribution_%']:
            slab_summary[col] = pd.to_numeric(slab_summary[col], errors='coerce').fillna(0.0).round(2)
        for col in ['Quantity', 'Sales_Value', 'Total_Discount']:
            slab_summary[col] = pd.to_numeric(slab_summary[col], errors='coerce').fillna(0.0)
        for col in ['Outlets', 'Invoices']:
            slab_summary[col] = pd.to_numeric(slab_summary[col], errors='coerce').fillna(0).astype(int)

        cols = []
        if 'Size_Key' in slab_summary.columns:
            cols.append('Size_Key')
        cols.extend([
            'Slab',
            'Outlets',
            'Invoices',
            'Invoice_Contribution_%',
            'Quantity',
            'AOQ',
            'AOV',
            'Sales_Value',
            'Sales_Contribution_%',
            'Total_Discount',
            'Discount_Pct',
        ])
        return slab_summary[cols].to_dict(orient='records')


    def _assign_defined_step2_slabs(
        self,
        df: pd.DataFrame,
        thresholds: List[float],
        level: str = "monthly_outlet",
        per_size_profiles: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        if 'Quantity' not in df.columns:
            return df

        outlet_col = 'Outlet_ID' if 'Outlet_ID' in df.columns else ('Store_ID' if 'Store_ID' in df.columns else None)
        if outlet_col is None:
            return df

        work = df.copy()
        qty = pd.to_numeric(work['Quantity'], errors='coerce').fillna(0.0)
        work['Quantity'] = qty

        if str(level or "monthly_outlet").strip().lower() != "monthly_outlet":
            level = "monthly_outlet"

        work['_step2_month'] = pd.to_datetime(work.get('Date'), errors='coerce').dt.to_period('M').astype(str)
        work['_step2_outlet'] = work[outlet_col].astype(str)
        if 'Sizes' in work.columns:
            work['_step2_size'] = work['Sizes'].map(self._normalize_step2_size_key)
        else:
            work['_step2_size'] = ''

        grouped = (
            work.groupby(['_step2_outlet', '_step2_month', '_step2_size'], as_index=False)
            .agg(_step2_monthly_qty=('Quantity', 'sum'))
        )
        grouped['_step2_monthly_qty'] = pd.to_numeric(grouped['_step2_monthly_qty'], errors='coerce').fillna(0.0)
        work = work.merge(grouped, on=['_step2_outlet', '_step2_month', '_step2_size'], how='left')
        monthly_qty = pd.to_numeric(work['_step2_monthly_qty'], errors='coerce').fillna(0.0)
        slab_values = pd.Series(index=work.index, dtype=object)

        size_profiles = per_size_profiles or {}
        global_thresholds = [float(x) for x in (thresholds or [])]

        unique_sizes = work['_step2_size'].dropna().astype(str).unique().tolist()
        for size_key in unique_sizes:
            size_mask = (work['_step2_size'].astype(str) == str(size_key))
            cfg = size_profiles.get(str(size_key), None)
            size_thresholds = (
                [float(x) for x in (cfg or {}).get('defined_slab_thresholds', [])]
                if cfg is not None
                else global_thresholds
            )
            if len(size_thresholds) == 0:
                size_thresholds = global_thresholds
            bins = [-np.inf] + size_thresholds + [np.inf]
            labels = [f"slab{i}" for i in range(len(bins) - 1)]
            slab_values.loc[size_mask] = pd.cut(
                monthly_qty.loc[size_mask],
                bins=bins,
                labels=labels,
                right=False,
                include_lowest=True,
            ).astype(str)

        work['Slab'] = slab_values.replace({'nan': 'slab0'}).fillna('slab0').astype(str)
        return work.drop(columns=['_step2_outlet', '_step2_month', '_step2_size', '_step2_monthly_qty'], errors='ignore')


    def _apply_step2_slab_definition(self, df: pd.DataFrame, request) -> pd.DataFrame:
        mode = self._normalize_step2_slab_definition_mode(getattr(request, 'slab_definition_mode', 'data'))
        if mode != "define":
            return df

        slab_count = self._normalize_step2_defined_slab_count(getattr(request, 'defined_slab_count', 5))
        thresholds = self._normalize_step2_defined_slab_thresholds(
            getattr(request, 'defined_slab_thresholds', None),
            slab_count=slab_count,
        )
        size_profiles = self._normalize_step2_defined_slab_profiles(
            getattr(request, 'defined_slab_profiles', None)
        )
        level = str(getattr(request, 'defined_slab_level', 'monthly_outlet') or 'monthly_outlet')
        return self._assign_defined_step2_slabs(
            df,
            thresholds=thresholds,
            level=level,
            per_size_profiles=size_profiles,
        )


    def _build_insta_fixed_base_discount(self, size_key: str, slab_index: int, plan_months: List[pd.Period]) -> Optional[np.ndarray]:
        # User-provided fixed baseline grid for planner (month x slab) for Insta Shampoo.
        fixed_grid = {
            "18-ML": {
                1: [12, 15, 17, 17],  # Apr
                2: [13, 17, 19, 19],  # May
                3: [12, 15, 17, 17],  # Jun
                4: [12, 15, 17, 17],  # Jul
                5: [12, 15, 17, 17],  # Aug
                6: [13, 17, 19, 19],  # Sep
                7: [12, 15, 17, 17],  # Oct
                8: [13, 17, 19, 19],  # Nov
                9: [12, 15, 17, 17],  # Dec
                10: [12, 15, 17, 17], # Jan
                11: [12, 15, 17, 17], # Feb
                12: [12, 15, 17, 17], # Mar
            },
            "12-ML": {
                1: [18, 24, 25, 25],  # Apr
                2: [14, 17, 18, 18],  # May
                3: [14, 17, 18, 18],  # Jun
                4: [17, 21, 18, 22],  # Jul
                5: [14, 17, 18, 18],  # Aug
                6: [18, 24, 24, 25],  # Sep
                7: [14, 17, 18, 18],  # Oct
                8: [14, 17, 18, 18],  # Nov
                9: [14, 17, 18, 18],  # Dec
                10: [14, 17, 18, 18], # Jan
                11: [14, 17, 18, 18], # Feb
                12: [14, 17, 18, 18], # Mar
            },
        }
        size_map = fixed_grid.get(size_key)
        if not size_map or slab_index < 1 or slab_index > 4:
            return None

        month_to_apr_index = {
            1: 10,  # Jan
            2: 11,  # Feb
            3: 12,  # Mar
            4: 1,   # Apr
            5: 2,   # May
            6: 3,   # Jun
            7: 4,   # Jul
            8: 5,   # Aug
            9: 6,   # Sep
            10: 7,  # Oct
            11: 8,  # Nov
            12: 9,  # Dec
        }
        values = []
        for m in plan_months:
            cal_month = int(getattr(m, "month", 0))
            apr_index = month_to_apr_index.get(cal_month)
            if apr_index is None or apr_index not in size_map:
                return None
            values.append(float(size_map[apr_index][slab_index - 1]))
        return np.asarray(values, dtype=float)


    def _planner_fixed_discount_override(self, request, slab_df: pd.DataFrame, selected_slab: str, plan_months: List[pd.Period]) -> Optional[np.ndarray]:
        if slab_df is None or slab_df.empty:
            return None
        if 'Subcategory' not in slab_df.columns or 'Sizes' not in slab_df.columns:
            return None

        valid_subcats = {"STX INSTA SHAMPOO", "STREAX INSTA SHAMPOO"}
        valid_sizes = {"12-ML", "18-ML"}

        req_sizes = [str(x).upper().replace(' ', '').strip() for x in (getattr(request, "sizes", None) or [])]
        req_sizes = [x for x in req_sizes if x in valid_sizes]
        req_subcats = [str(x).upper().strip() for x in (getattr(request, "subcategories", None) or [])]
        req_subcats = [re.sub(r"\s+", " ", x) for x in req_subcats if x]

        # Candidate rows for fixed override decision.
        cand = slab_df.copy()
        cand['__subcat'] = (
            cand['Subcategory']
            .astype(str)
            .str.upper()
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
        cand['__size'] = (
            cand['Sizes']
            .astype(str)
            .str.upper()
            .str.replace(' ', '', regex=False)
            .str.strip()
        )

        if req_subcats:
            if not all(s in valid_subcats for s in req_subcats):
                return None
            cand = cand[cand['__subcat'].isin(req_subcats)]
        else:
            cand = cand[cand['__subcat'].isin(valid_subcats)]

        if req_sizes:
            if len(set(req_sizes)) != 1:
                return None
            size_key = list(set(req_sizes))[0]
            cand = cand[cand['__size'] == size_key]
        else:
            cand_sizes = [x for x in cand['__size'].dropna().unique().tolist() if x in valid_sizes]
            if len(cand_sizes) != 1:
                return None
            size_key = cand_sizes[0]

        if cand.empty:
            return None

        slab_index = self._extract_slab_index(selected_slab)
        if slab_index is None:
            return None

        return self._build_insta_fixed_base_discount(size_key=size_key, slab_index=slab_index, plan_months=plan_months)


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


    def _estimate_base_discount_monthly_blocks(
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

        if work.empty:
            return np.array([]), np.array([], dtype=bool)

        work['Month_Key'] = pd.to_datetime(work['Period'], errors='coerce').dt.to_period('M')
        monthly = (
            work.groupby('Month_Key', as_index=False)
            .agg(
                Period=('Period', 'min'),
                Discount=('Discount', 'median'),
            )
            .sort_values('Period', kind='stable')
            .reset_index(drop=True)
        )

        if monthly.empty:
            return np.array([]), np.array([], dtype=bool)

        discounts_month = pd.Series(monthly['Discount'], dtype=float).interpolate(limit_direction='both').bfill().ffill()
        discounts_month = self._round_discount_series(discounts_month, step=round_step)

        n = len(discounts_month)
        base = np.zeros(n, dtype=float)
        transitions = np.zeros(n, dtype=bool)

        min_upward_jump_pp = max(0.0, float(min_upward_jump_pp))
        min_downward_drop_pp = max(0.0, float(min_downward_drop_pp))

        prev_base = None
        for i in range(n):
            candidate = float(discounts_month.iloc[i]) if pd.notna(discounts_month.iloc[i]) else np.nan
            if not np.isfinite(candidate):
                candidate = prev_base if prev_base is not None else 0.0

            if prev_base is None:
                block_base = candidate
            else:
                if candidate > prev_base and (candidate - prev_base) < min_upward_jump_pp:
                    block_base = prev_base
                elif candidate < prev_base and (prev_base - candidate) < min_downward_drop_pp:
                    block_base = prev_base
                else:
                    block_base = candidate
                if abs(block_base - prev_base) > 1e-9:
                    transitions[i] = True

            base[i] = np.clip(float(block_base), 0.0, 100.0)
            prev_base = float(block_base)

        return base, transitions


    def _compute_base_depth_result(self, df: pd.DataFrame, request: BaseDepthRequest):
        work, missing = self._prepare_step2_discount_basis(df)
        if missing:
            raise ValueError(f"Missing required columns for estimator: {', '.join(missing)}")
        work['Period'] = work['Date'].dt.floor('D')

        if 'Bill_No' in work.columns:
            daily = (
                work.groupby('Period', as_index=False)
                .agg(
                    orders=('Bill_No', 'nunique'),
                    quantity=('Quantity', 'sum'),
                    total_discount=('_step2_scheme_amount', 'sum'),
                    sales_value=('_step2_dsp_sales', 'sum'),
                )
                .sort_values('Period')
            )
        else:
            daily = (
                work.groupby('Period', as_index=False)
                .agg(
                    orders=('Outlet_ID', 'count'),
                    quantity=('Quantity', 'sum'),
                    total_discount=('_step2_scheme_amount', 'sum'),
                    sales_value=('_step2_dsp_sales', 'sum'),
                )
                .sort_values('Period')
            )

        daily['actual_discount_pct'] = (
            (daily['total_discount'] / daily['sales_value']) * 100.0
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0)

        slab_mode = self._normalize_step2_slab_definition_mode(getattr(request, 'slab_definition_mode', 'data'))
        if slab_mode == 'define':
            # Defined slab mode is monthly by design: one base value per month.
            monthly = (
                daily.assign(OutputPeriod=daily['Period'].dt.to_period('M').dt.start_time)
                .groupby('OutputPeriod', as_index=False)
                .agg(
                    orders=('orders', 'sum'),
                    quantity=('quantity', 'sum'),
                    total_discount=('total_discount', 'sum'),
                    sales_value=('sales_value', 'sum'),
                )
                .rename(columns={'OutputPeriod': 'Period'})
                .sort_values('Period')
            )
            monthly['actual_discount_pct'] = (
                (monthly['total_discount'] / monthly['sales_value']) * 100.0
            ).replace([np.inf, -np.inf], 0.0).fillna(0.0)

            base_monthly, _ = self._estimate_base_discount_monthly_blocks(
                monthly['Period'],
                monthly['actual_discount_pct'],
                min_upward_jump_pp=float(request.min_upward_jump_pp),
                min_downward_drop_pp=float(request.min_downward_drop_pp),
                round_step=0.5,
            )
            if len(base_monthly) != len(monthly):
                base_monthly = np.zeros(len(monthly), dtype=float)
            monthly['base_discount_pct'] = self._round_discount_series(base_monthly, step=0.5)
            aggregated = monthly
        else:
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
        selected_classifications = self._normalize_step2_outlet_classifications(
            list(request.outlet_classifications or [])
        )

        rfm_for_class = rfm.copy()
        if selected_segments:
            rfm_for_class = rfm_for_class[rfm_for_class['RFM_Segment'].isin(selected_segments)]

        class_outlet_ids = set(rfm_for_class['Outlet_ID'].astype(str).tolist())
        df_for_class = df[df['Outlet_ID'].astype(str).isin(class_outlet_ids)].copy()

        df_for_slab = df_for_class.copy()
        if selected_classifications and 'Final_Outlet_Classification' in df_for_slab.columns:
            slab_groups = self._to_step2_outlet_group_series(df_for_slab['Final_Outlet_Classification'])
            df_for_slab = df_for_slab[slab_groups.isin(selected_classifications)]
        df_for_slab = self._apply_step2_slab_definition(df_for_slab, request)

        rfm_segments = sorted(rfm['RFM_Segment'].dropna().astype(str).unique().tolist())
        if 'Final_Outlet_Classification' in df_for_class.columns:
            class_groups = self._to_step2_outlet_group_series(df_for_class['Final_Outlet_Classification'])
            raw_classifications = [x for x in class_groups.dropna().astype(str).unique().tolist() if x]
            classifications = self._normalize_step2_outlet_classifications(raw_classifications)
        else:
            classifications = []
        slabs = self._normalize_step2_slab_values(
            df_for_slab['Slab'].dropna().astype(str).unique().tolist()
        ) if 'Slab' in df_for_slab.columns else []

        matching_outlets = scope['df_scope']['Outlet_ID'].nunique() if not scope['df_scope'].empty else 0

        return DiscountOptionsResponse(
            success=True,
            message="Discount options loaded successfully",
            rfm_segments=rfm_segments,
            outlet_classifications=classifications,
            slabs=slabs,
            matching_outlets=int(matching_outlets)
        )


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

            required_cols = ['Date', 'Quantity']
            missing_cols = [c for c in required_cols if c not in df.columns]
            _, step2_missing = self._prepare_step2_discount_basis(df)
            missing_cols.extend([c for c in step2_missing if c not in missing_cols])
            if missing_cols:
                return BaseDepthResponse(
                    success=False,
                    message=f"Missing required columns for estimator: {', '.join(missing_cols)}",
                    points=[],
                    summary={}
                )

            points, summary = self._compute_base_depth_result(df, request)
            summary_source = scope.get('df_scope_all_slabs', df)
            summary_by_slab = self._build_summary_by_slab(summary_source)

            slab_results = []
            if 'Slab' in df.columns:
                if request.slabs:
                    slab_scope = self._normalize_step2_slab_values([str(s) for s in request.slabs])
                else:
                    slab_scope = self._normalize_step2_slab_values(
                        df['Slab'].dropna().astype(str).unique().tolist()
                    )
                mode = self._normalize_step2_slab_definition_mode(getattr(request, 'slab_definition_mode', 'data'))
                size_split_enabled = mode == 'define' and 'Sizes' in df.columns
                jobs: List[Dict[str, Any]] = []

                if size_split_enabled:
                    work = df.copy()
                    work['__size_key'] = work['Sizes'].map(self._normalize_step2_size_key)
                    requested_sizes = [self._normalize_step2_size_key(x) for x in (getattr(request, 'sizes', None) or []) if self._normalize_step2_size_key(x)]
                    if requested_sizes:
                        # In define mode, preserve requested size tabs even if one requested size
                        # currently has no rows after filters; user should still see explicit tabs.
                        size_scope = list(dict.fromkeys(requested_sizes))
                    else:
                        size_scope = sorted(list(dict.fromkeys(work['__size_key'].dropna().astype(str).unique().tolist())))

                    # Only split by size if user scope effectively has multiple sizes.
                    if len(size_scope) > 1:
                        for size_key in size_scope:
                            size_df = work[work['__size_key'].astype(str) == str(size_key)].copy()
                            size_slab_scope = self._normalize_step2_slab_values(
                                size_df['Slab'].dropna().astype(str).unique().tolist()
                            )
                            for slab in size_slab_scope:
                                jobs.append({
                                    "size_key": str(size_key),
                                    "slab": str(slab),
                                    "label": f"{size_key} | {slab}",
                                })
                    else:
                        for slab in slab_scope:
                            jobs.append({
                                "size_key": None,
                                "slab": str(slab),
                                "label": str(slab),
                            })
                else:
                    for slab in slab_scope:
                        jobs.append({
                            "size_key": None,
                            "slab": str(slab),
                            "label": str(slab),
                        })

                if len(jobs) > 1:
                    for job in jobs:
                        slab_df = df[df['Slab'].astype(str) == str(job['slab'])].copy()
                        if job.get("size_key") is not None and 'Sizes' in slab_df.columns:
                            slab_df = slab_df[
                                slab_df['Sizes'].map(self._normalize_step2_size_key) == str(job['size_key'])
                            ]

                        if slab_df.empty:
                            slab_results.append({
                                "slab": str(job['label']),
                                "success": False,
                                "message": "No data for this slab with current filters",
                                "points": [],
                                "summary": {},
                            })
                            continue

                        slab_points, slab_summary = self._compute_base_depth_result(slab_df, request)
                        slab_results.append({
                            "slab": str(job['label']),
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
                summary_by_slab=summary_by_slab,
            )
        except Exception as e:
            return BaseDepthResponse(
                success=False,
                message=f"Error estimating base depth: {str(e)}",
                points=[],
                summary={},
                slab_results=[],
                summary_by_slab=[],
            )
