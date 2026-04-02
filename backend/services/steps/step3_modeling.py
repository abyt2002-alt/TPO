"""Step 3 two-stage modeling and ROI generation.\n\nThis module builds monthly modeling frames, fits stage models, and computes\nROI outputs using the same Q1 discount basis as Step 2.\n"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
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
from services.core.shared_math import CustomConstrainedRidge

DEFAULT_COGS_BY_SIZE = {
    "12-ML": 8.0,
    "18-ML": 10.0,
}


class Step3ModelingMixin:

    def _resolve_modeling_cogs_for_size(self, request: ModelingRequest, size_key: Optional[str], default_cogs: float) -> float:
        normalized_size = self._normalize_step2_size_key(size_key)
        size_default_cogs = DEFAULT_COGS_BY_SIZE.get(normalized_size)
        mapping = getattr(request, 'cogs_per_size', None) or {}
        if isinstance(mapping, dict) and normalized_size:
            for raw_key, raw_value in mapping.items():
                if self._normalize_step2_size_key(raw_key) != normalized_size:
                    continue
                try:
                    parsed = float(raw_value)
                except Exception:
                    parsed = np.nan
                if np.isfinite(parsed):
                    if parsed > 0:
                        return float(parsed)
                    if size_default_cogs is not None:
                        return float(size_default_cogs)
                    return max(parsed, 0.0)
        if getattr(request, 'cogs_per_unit', None) is not None:
            try:
                parsed = float(request.cogs_per_unit)
                if parsed > 0:
                    return float(parsed)
                if size_default_cogs is not None:
                    return float(size_default_cogs)
                return max(parsed, 0.0)
            except Exception:
                pass
        if size_default_cogs is not None:
            return float(size_default_cogs)
        return max(float(default_cogs), 0.0)


    def _build_monthly_model_dataframe(self, df: pd.DataFrame, request: ModelingRequest) -> pd.DataFrame:
        work = df.copy()
        work, missing = self._prepare_step2_discount_basis(work)
        if missing:
            raise ValueError(
                "Step 3 requires Q1 discount columns. Missing: "
                + ", ".join(missing)
            )
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
                total_discount=('_step2_scheme_amount', 'sum'),
                sales_value=('_step2_dsp_sales', 'sum'),
            )
            .rename(columns={'Period_D': 'Period'})
            .sort_values('Period')
        )
        daily['actual_discount_pct'] = (
            (daily['total_discount'] / daily['sales_value']) * 100.0
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        slab_mode = self._normalize_step2_slab_definition_mode(getattr(request, 'slab_definition_mode', 'data'))
        if slab_mode == 'define':
            monthly_actual = (
                daily.assign(Month_Key=pd.to_datetime(daily['Period']).dt.to_period('M'))
                .groupby('Month_Key', as_index=False)
                .agg(
                    Period=('Period', 'min'),
                    total_discount=('total_discount', 'sum'),
                    sales_value=('sales_value', 'sum'),
                )
                .sort_values('Period')
            )
            monthly_actual['actual_discount_pct'] = (
                (monthly_actual['total_discount'] / monthly_actual['sales_value']) * 100.0
            ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
            base_monthly, _ = self._estimate_base_discount_monthly_blocks(
                monthly_actual['Period'],
                monthly_actual['actual_discount_pct'],
                min_upward_jump_pp=float(request.min_upward_jump_pp),
                min_downward_drop_pp=float(request.min_downward_drop_pp),
                round_step=0.5,
            )
            if len(base_monthly) != len(monthly_actual):
                base_monthly = np.zeros(len(monthly_actual), dtype=float)
            monthly_actual['base_discount_pct'] = self._round_discount_series(base_monthly, step=0.5)
            monthly_base = monthly_actual[['Month_Key', 'base_discount_pct']].copy()
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
                total_discount=('_step2_scheme_amount', 'sum'),
                sales_value=('_step2_dsp_sales', 'sum'),
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


    def _build_other_slabs_weighted_discount_series(
        self,
        df_scope_all_slabs: pd.DataFrame,
        request: ModelingRequest,
        size_key: str,
        slab_value: str,
    ) -> pd.DataFrame:
        work = df_scope_all_slabs.copy()
        if work.empty or 'Slab' not in work.columns or 'Sizes' not in work.columns:
            return pd.DataFrame(columns=['Period', 'other_slabs_weighted_base_discount_pct'])

        work['Size_Key'] = work['Sizes'].astype(str).map(self._normalize_step2_size_key)
        part = work[
            (work['Size_Key'].astype(str) == str(size_key))
            & (work['Slab'].astype(str) != str(slab_value))
            & (work['Slab'].astype(str).map(self._is_step2_allowed_slab))
        ].copy()
        if part.empty:
            return pd.DataFrame(columns=['Period', 'other_slabs_weighted_base_discount_pct'])

        slab_monthly = []
        for other_slab in sorted(part['Slab'].astype(str).dropna().unique().tolist(), key=self._slab_sort_key):
            other_df = part[part['Slab'].astype(str) == str(other_slab)].copy()
            if other_df.empty:
                continue
            monthly_other = self._build_monthly_model_dataframe(other_df, request)
            if monthly_other.empty:
                continue
            monthly_other = monthly_other[['Period', 'quantity', 'base_discount_pct']].copy()
            monthly_other['Slab'] = str(other_slab)
            slab_monthly.append(monthly_other)

        if not slab_monthly:
            return pd.DataFrame(columns=['Period', 'other_slabs_weighted_base_discount_pct'])

        other_monthly = pd.concat(slab_monthly, ignore_index=True)
        weight_df = (
            other_monthly.groupby('Slab', as_index=False)['quantity']
            .sum()
            .rename(columns={'quantity': 'Period_Slab_Qty'})
        )
        total_qty = float(pd.to_numeric(weight_df['Period_Slab_Qty'], errors='coerce').fillna(0.0).sum())
        if total_qty > 0:
            weight_df['Fixed_Weight'] = pd.to_numeric(weight_df['Period_Slab_Qty'], errors='coerce').fillna(0.0) / total_qty
        else:
            n = max(len(weight_df), 1)
            weight_df['Fixed_Weight'] = 1.0 / n

        other_monthly = other_monthly.merge(weight_df[['Slab', 'Fixed_Weight']], on='Slab', how='left')
        other_monthly['Fixed_Weight'] = pd.to_numeric(other_monthly['Fixed_Weight'], errors='coerce').fillna(0.0)
        other_monthly['weighted_part'] = (
            pd.to_numeric(other_monthly['base_discount_pct'], errors='coerce').fillna(0.0)
            * other_monthly['Fixed_Weight']
        )
        out = (
            other_monthly.groupby('Period', as_index=False)
            .agg(other_slabs_weighted_base_discount_pct=('weighted_part', 'sum'))
            .sort_values('Period')
            .reset_index(drop=True)
        )
        return out


    def _build_monthly_model_dataframe_new_strategy(
        self,
        df_scope_all_slabs: pd.DataFrame,
        slab_df: pd.DataFrame,
        request: ModelingRequest,
        size_key: str,
        slab_value: str,
    ) -> pd.DataFrame:
        monthly = self._build_monthly_model_dataframe(slab_df, request)
        if monthly.empty:
            return monthly
        other_series = self._build_other_slabs_weighted_discount_series(
            df_scope_all_slabs=df_scope_all_slabs,
            request=request,
            size_key=size_key,
            slab_value=slab_value,
        )
        monthly = monthly.merge(other_series, on='Period', how='left')
        monthly['other_slabs_weighted_base_discount_pct'] = (
            pd.to_numeric(monthly.get('other_slabs_weighted_base_discount_pct', 0.0), errors='coerce')
            .ffill()
            .bfill()
            .fillna(0.0)
        )
        return monthly.sort_values('Period').reset_index(drop=True)


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


    def _run_two_stage_model_new_strategy(
        self,
        monthly: pd.DataFrame,
        include_lag_discount: bool = True,
        l2_penalty: float = 0.1,
        optimize_l2_penalty: bool = True,
    ):
        if monthly is None or monthly.empty or len(monthly) < 3:
            return None

        x_discount = monthly['actual_discount_pct'].to_numpy(dtype=float)
        y_qty = monthly['quantity'].to_numpy(dtype=float)
        y_store = monthly['store_count'].to_numpy(dtype=float)
        base = monthly['base_discount_pct'].to_numpy(dtype=float)
        lag1 = monthly['lag1_base_discount_pct'].to_numpy(dtype=float)
        other = monthly['other_slabs_weighted_base_discount_pct'].to_numpy(dtype=float)

        stage1 = LinearRegression()
        stage1.fit(x_discount.reshape(-1, 1), y_store)
        store_pred = stage1.predict(x_discount.reshape(-1, 1))
        residual_store = y_store - store_pred

        feature_order = ['residual_store', 'base_discount_pct']
        x_cols = [residual_store, base]
        non_negative_indices = [0, 1]
        non_positive_indices = []

        if include_lag_discount:
            feature_order.append('lag1_base_discount_pct')
            x_cols.append(lag1)
            non_positive_indices.append(len(feature_order) - 1)

        feature_order.append('other_slabs_weighted_base_discount_pct')
        x_cols.append(other)
        non_positive_indices.append(len(feature_order) - 1)

        X2 = np.column_stack(x_cols)
        l2_floor = 0.1
        l2_penalty = max(float(l2_penalty), l2_floor)

        def _fit_stage2_with_l2(lam: float):
            model = CustomConstrainedRidge(
                l2_penalty=float(lam),
                non_negative_indices=non_negative_indices,
                non_positive_indices=non_positive_indices,
                maxiter=4000,
            )
            model.fit(X2, y_qty)
            model.feature_order_ = feature_order
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
                    non_positive_indices=non_positive_indices,
                    maxiter=4000,
                )
                try:
                    model.fit(X_train, y_train)
                    model.feature_order_ = feature_order
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
            l2_candidates = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
            best = None
            for lam in l2_candidates:
                model_tmp, preds_tmp, r2_tmp = _fit_stage2_with_l2(lam)
                cv_r2_tmp = _cv_r2_for_l2(lam)
                score = cv_r2_tmp if np.isfinite(cv_r2_tmp) else -np.inf
                if (best is None) or (score > best['score']):
                    best = {
                        'model': model_tmp,
                        'preds': preds_tmp,
                        'r2_train': r2_tmp,
                        'cv_r2': float(cv_r2_tmp) if np.isfinite(cv_r2_tmp) else np.nan,
                        'score': score,
                        'l2': float(lam),
                    }
            stage2 = best['model']
            qty_pred = best['preds']
            r2_stage2 = float(best['r2_train'])
            r2_stage2_cv = best['cv_r2']
            l2_used = float(best['l2'])
        else:
            stage2, qty_pred, r2_stage2 = _fit_stage2_with_l2(l2_penalty)
            r2_stage2_cv = float(_cv_r2_for_l2(l2_penalty))
            l2_used = float(l2_penalty)

        stage2_ols = LinearRegression()
        stage2_ols.fit(X2, y_qty)
        stage2_ols.feature_order_ = feature_order
        qty_pred_ols = stage2_ols.predict(X2)
        if len(y_qty) > 1:
            ss_tot = float(np.sum((y_qty - np.mean(y_qty)) ** 2))
            ss_res_ols = float(np.sum((y_qty - qty_pred_ols) ** 2))
            r2_ols = 1 - (ss_res_ols / ss_tot) if ss_tot > 1e-12 else 0.0
        else:
            r2_ols = 0.0

        qty_base = self._predict_stage2_quantity(
            stage2,
            residual_store,
            base,
            np.zeros_like(base),
            lag1,
            extra_feature_values={'other_slabs_weighted_base_discount_pct': other},
        )
        qty_base_ols = self._predict_stage2_quantity(
            stage2_ols,
            residual_store,
            base,
            np.zeros_like(base),
            lag1,
            extra_feature_values={'other_slabs_weighted_base_discount_pct': other},
        )

        zeros = np.zeros_like(base, dtype=float)
        qty_no_discount = self._predict_stage2_quantity(
            stage2,
            residual_store,
            zeros,
            zeros,
            zeros,
            extra_feature_values={'other_slabs_weighted_base_discount_pct': zeros},
        )
        qty_no_discount_ols = self._predict_stage2_quantity(
            stage2_ols,
            residual_store,
            zeros,
            zeros,
            zeros,
            extra_feature_values={'other_slabs_weighted_base_discount_pct': zeros},
        )
        qty_no_discount = np.maximum(qty_no_discount, 0.0)

        actual_price = monthly['base_price'].to_numpy(dtype=float) * (1 - x_discount / 100.0)
        baseline_price = monthly['base_price'].to_numpy(dtype=float) * (1 - base / 100.0)
        predicted_revenue = qty_pred * actual_price
        baseline_revenue = qty_base * baseline_price
        spend = monthly['base_price'].to_numpy(dtype=float) * (
            monthly['tactical_discount_pct'].to_numpy(dtype=float) / 100.0
        ) * qty_pred
        incremental_revenue = predicted_revenue - baseline_revenue
        roi = np.where(spend > 0, incremental_revenue / spend, np.nan)

        result_df = monthly.copy()
        result_df['predicted_quantity'] = qty_pred
        result_df['baseline_quantity'] = qty_base
        result_df['non_discount_baseline_quantity'] = qty_no_discount
        result_df['predicted_quantity_ols'] = qty_pred_ols
        result_df['baseline_quantity_ols'] = qty_base_ols
        result_df['non_discount_baseline_quantity_ols'] = qty_no_discount_ols
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
            'coef_tactical_discount': 0.0,
            'coef_lag1_structural_discount': float(stage2.coef_[2]) if include_lag_discount and len(stage2.coef_) >= 3 else 0.0,
            'coef_other_slabs_weighted_base_discount_pct': float(stage2.coef_[-1]),
            'include_lag_discount': 1.0 if include_lag_discount else 0.0,
            'l2_penalty': float(l2_used),
            'l2_penalty_input': float(l2_penalty),
            'optimize_l2_penalty': 1.0 if optimize_l2_penalty else 0.0,
            'l2_candidates_evaluated': 10.0 if optimize_l2_penalty else 1.0,
            'stage2_cv_r2': float(r2_stage2_cv) if np.isfinite(r2_stage2_cv) else 0.0,
            'constraint_residual_non_negative': 1.0,
            'constraint_structural_non_negative': 1.0,
            'constraint_tactical_non_negative': 0.0,
            'constraint_lag_non_positive': 1.0 if include_lag_discount else 0.0,
            'constraint_other_slabs_non_positive': 1.0,
            'stage2_model_type': 1.0,
            'stage2_fit_success': 1.0 if bool(getattr(stage2, 'success_', True)) else 0.0,
            'stage2_r2': float(r2_stage2),
            'stage2_ols_intercept': float(stage2_ols.intercept_),
            'stage2_ols_coef_residual_store': float(stage2_ols.coef_[0]),
            'stage2_ols_coef_structural_discount': float(stage2_ols.coef_[1]),
            'stage2_ols_coef_tactical_discount': 0.0,
            'stage2_ols_coef_lag1_structural_discount': float(stage2_ols.coef_[2]) if include_lag_discount and len(stage2_ols.coef_) >= 3 else 0.0,
            'stage2_ols_coef_other_slabs_weighted_base_discount_pct': float(stage2_ols.coef_[-1]),
            'stage2_ols_r2': float(r2_ols),
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
                return ModelingResponse(
                    success=False,
                    message="No data matches selected filters",
                    slab_results=[],
                    combined_summary={},
                    summary_by_slab=[],
                )

            df_scope = scope['df_scope']
            if df_scope.empty:
                return ModelingResponse(
                    success=False,
                    message="No data for modeling",
                    slab_results=[],
                    combined_summary={},
                    summary_by_slab=[],
                )

            summary_source = scope.get('df_scope_all_slabs', df_scope)
            summary_by_slab = self._build_summary_by_slab(summary_source)

            slab_results = []
            weighted_qty = 0.0
            weighted_r2 = 0.0
            total_spend = 0.0
            total_incremental_revenue = 0.0
            if 'Sizes' in df_scope.columns:
                size_list = sorted(
                    list(dict.fromkeys(
                        df_scope['Sizes'].astype(str).map(self._normalize_step2_size_key).dropna().tolist()
                    ))
                )
            else:
                size_list = ['']

            selected_slabs = [str(s) for s in (request.slabs or [])]

            for size_key in size_list:
                size_df = df_scope.copy()
                size_scope_all_slabs = summary_source.copy()
                if size_key:
                    size_df = size_df[
                        size_df['Sizes'].astype(str).map(self._normalize_step2_size_key) == str(size_key)
                    ]
                    size_scope_all_slabs = size_scope_all_slabs[
                        size_scope_all_slabs['Sizes'].astype(str).map(self._normalize_step2_size_key) == str(size_key)
                    ]

                if size_df.empty:
                    continue

                if 'Slab' in size_df.columns:
                    slab_list = selected_slabs[:]
                    if not slab_list:
                        slab_list = size_df['Slab'].dropna().astype(str).unique().tolist()
                    slab_list = [
                        str(s)
                        for s in sorted(list(dict.fromkeys(slab_list)), key=self._slab_sort_key)
                        if self._is_step2_allowed_slab(s)
                    ]
                else:
                    slab_list = ['All']

                for slab in slab_list:
                    slab_df = size_df.copy()
                    if 'Slab' in slab_df.columns:
                        slab_df = slab_df[slab_df['Slab'].astype(str) == str(slab)]

                    if slab_df.empty:
                        slab_results.append(
                            ModelingSlabResult(size=size_key or None, slab=str(slab), valid=False, reason="No data for slab")
                        )
                        continue

                    monthly = self._build_monthly_model_dataframe_new_strategy(
                        df_scope_all_slabs=size_scope_all_slabs,
                        slab_df=slab_df,
                        request=request,
                        size_key=size_key,
                        slab_value=str(slab),
                    )
                    if monthly.empty or len(monthly) < 3:
                        slab_results.append(
                            ModelingSlabResult(size=size_key or None, slab=str(slab), valid=False, reason="Not enough monthly points for modeling")
                        )
                        continue

                    modeled = self._run_two_stage_model_new_strategy(
                        monthly,
                        include_lag_discount=bool(getattr(request, 'include_lag_discount', True)),
                        l2_penalty=float(getattr(request, 'l2_penalty', 0.1)),
                        optimize_l2_penalty=bool(getattr(request, 'optimize_l2_penalty', True)),
                    )
                    if modeled is None:
                        slab_results.append(
                            ModelingSlabResult(size=size_key or None, slab=str(slab), valid=False, reason="Model could not be fitted")
                        )
                        continue

                    model_df = modeled['model_df']
                    coefficients = modeled['coefficients']
                    stage2_model = modeled['stage2_model']
                    base_price_series = pd.to_numeric(model_df.get('base_price', pd.Series(dtype=float)), errors='coerce').dropna()
                    default_cogs = float(np.round(float(base_price_series.iloc[0]) * 0.5)) if not base_price_series.empty else 0.0
                    cogs_per_unit = self._resolve_modeling_cogs_for_size(request, size_key, default_cogs)
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
                            size=size_key or None,
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
                                'avg_other_slabs_weighted_base_discount_pct': float(model_df['other_slabs_weighted_base_discount_pct'].mean()),
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
                summary_by_slab=summary_by_slab,
            )
        except Exception as e:
            return ModelingResponse(
                success=False,
                message=f"Error in modeling: {str(e)}",
                slab_results=[],
                combined_summary={},
                summary_by_slab=[],
            )
