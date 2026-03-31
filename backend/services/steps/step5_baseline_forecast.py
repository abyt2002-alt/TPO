"""Step 5 baseline forecast computation.\n\nThis module builds non-discount baseline histories and projects the forecast\nwindow used by planner/scenario UX.\n"""

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


class Step5BaselineForecastMixin:

    def _forecast_baseline_series(self, values: List[float], forecast_months: int) -> np.ndarray:
        arr = np.asarray(values or [], dtype=float)
        if arr.size == 0:
            return np.zeros(int(forecast_months), dtype=float)
        if arr.size == 1:
            return np.repeat(max(arr[0], 0.0), int(forecast_months))

        if Holt is not None and arr.size >= 3:
            try:
                fitted = Holt(
                    arr,
                    exponential=False,
                    damped_trend=False,
                    initialization_method="estimated",
                ).fit(optimized=True)
                forecast = np.asarray(fitted.forecast(int(forecast_months)), dtype=float)
                return np.clip(forecast, 0.0, None)
            except Exception:
                pass

        x = np.arange(arr.size, dtype=float)
        slope, intercept = np.polyfit(x, arr, 1)
        future_x = np.arange(arr.size, arr.size + int(forecast_months), dtype=float)
        forecast = intercept + (slope * future_x)
        return np.clip(np.asarray(forecast, dtype=float), 0.0, None)


    async def calculate_baseline_forecast(self, request: BaselineForecastRequest) -> BaselineForecastResponse:
        try:
            scope = self._build_step2_scope(request)
            if scope is None:
                return BaselineForecastResponse(success=False, message="No data matches selected filters", points=[])

            df_scope = scope['df_scope']
            if df_scope.empty:
                return BaselineForecastResponse(success=False, message="No data for baseline forecast", points=[])

            summary_source = scope.get('df_scope_all_slabs', df_scope)
            forecast_months = int(getattr(request, 'forecast_months', 3) or 3)
            size_histories: Dict[str, pd.DataFrame] = {}

            for size_key in ['12-ML', '18-ML']:
                size_df = df_scope[
                    df_scope['Sizes'].astype(str).map(self._normalize_step2_size_key) == str(size_key)
                ].copy()
                size_scope_all_slabs = summary_source[
                    summary_source['Sizes'].astype(str).map(self._normalize_step2_size_key) == str(size_key)
                ].copy()
                if size_df.empty:
                    continue

                slab_list = size_df['Slab'].dropna().astype(str).unique().tolist() if 'Slab' in size_df.columns else []
                slab_list = [str(s) for s in sorted(list(dict.fromkeys(slab_list)), key=self._slab_sort_key) if self._is_step2_allowed_slab(s)]
                if not slab_list:
                    continue

                size_monthly_parts: List[pd.DataFrame] = []
                size_discount_parts: List[pd.DataFrame] = []
                for slab in slab_list:
                    slab_df = size_df[size_df['Slab'].astype(str) == str(slab)].copy()
                    if slab_df.empty:
                        continue

                    monthly = self._build_monthly_model_dataframe_new_strategy(
                        df_scope_all_slabs=size_scope_all_slabs,
                        slab_df=slab_df,
                        request=request,
                        size_key=size_key,
                        slab_value=str(slab),
                    )
                    if monthly.empty or len(monthly) < 3:
                        continue

                    modeled = self._run_two_stage_model_new_strategy(
                        monthly,
                        include_lag_discount=bool(getattr(request, 'include_lag_discount', True)),
                        l2_penalty=float(getattr(request, 'l2_penalty', 0.1)),
                        optimize_l2_penalty=bool(getattr(request, 'optimize_l2_penalty', True)),
                    )
                    if modeled is None:
                        continue

                    model_df = modeled['model_df'].copy()
                    if 'non_discount_baseline_quantity' not in model_df.columns:
                        continue
                    coeff = modeled.get('coefficients', {}) or {}
                    beta_base = float(coeff.get('coef_structural_discount', 0.0))
                    beta_lag = float(coeff.get('coef_lag1_structural_discount', 0.0))
                    beta_other = float(coeff.get('coef_other_slabs_weighted_base_discount_pct', 0.0))
                    slab_discount_component = (
                        beta_base * pd.to_numeric(model_df.get('base_discount_pct', 0.0), errors='coerce').fillna(0.0)
                        + beta_lag * pd.to_numeric(model_df.get('lag1_base_discount_pct', 0.0), errors='coerce').fillna(0.0)
                        + beta_other * pd.to_numeric(model_df.get('other_slabs_weighted_base_discount_pct', 0.0), errors='coerce').fillna(0.0)
                    )
                    size_monthly_parts.append(
                        model_df[['Period', 'non_discount_baseline_quantity']]
                        .rename(columns={'non_discount_baseline_quantity': 'baseline_quantity'})
                    )
                    size_discount_parts.append(
                        pd.DataFrame({
                            'Period': model_df['Period'],
                            'discount_component_qty': slab_discount_component,
                        })
                    )

                if not size_monthly_parts:
                    continue

                size_history = pd.concat(size_monthly_parts, ignore_index=True)
                size_history_baseline = (
                    size_history.groupby('Period', as_index=False)
                    .agg(baseline_quantity=('baseline_quantity', 'sum'))
                    .sort_values('Period')
                    .reset_index(drop=True)
                )
                if size_discount_parts:
                    size_discount_history = pd.concat(size_discount_parts, ignore_index=True)
                    size_discount_history = (
                        size_discount_history.groupby('Period', as_index=False)
                        .agg(discount_component_qty=('discount_component_qty', 'sum'))
                        .sort_values('Period')
                        .reset_index(drop=True)
                    )
                else:
                    size_discount_history = pd.DataFrame(columns=['Period', 'discount_component_qty'])

                size_history_full = size_history_baseline.merge(size_discount_history, on='Period', how='left')
                size_history_full['discount_component_qty'] = pd.to_numeric(
                    size_history_full.get('discount_component_qty', 0.0), errors='coerce'
                ).fillna(0.0)
                size_histories[size_key] = size_history_full

            if not size_histories:
                return BaselineForecastResponse(success=False, message="No valid slab baselines available for forecast", points=[])

            all_periods = pd.Index(sorted(set().union(*[set(df['Period']) for df in size_histories.values()])))
            full_history = pd.DataFrame({'Period': all_periods})
            for size_key in ['12-ML', '18-ML']:
                hist = size_histories.get(size_key)
                if hist is None:
                    full_history[f'baseline_{size_key}'] = 0.0
                    full_history[f'discount_component_{size_key}'] = 0.0
                else:
                    full_history = full_history.merge(
                        hist.rename(
                            columns={
                                'baseline_quantity': f'baseline_{size_key}',
                                'discount_component_qty': f'discount_component_{size_key}',
                            }
                        ),
                        on='Period',
                        how='left',
                    )
                    full_history[f'baseline_{size_key}'] = pd.to_numeric(full_history[f'baseline_{size_key}'], errors='coerce').fillna(0.0)
                    full_history[f'discount_component_{size_key}'] = pd.to_numeric(
                        full_history[f'discount_component_{size_key}'], errors='coerce'
                    ).fillna(0.0)

            last_period = pd.to_datetime(full_history['Period'].max())
            future_periods = [last_period + pd.DateOffset(months=i) for i in range(1, forecast_months + 1)]
            forecast_df = pd.DataFrame({'Period': future_periods})
            for size_key in ['12-ML', '18-ML']:
                hist_values = full_history[f'baseline_{size_key}'].tolist()
                forecast_values = self._forecast_baseline_series(hist_values, forecast_months)
                forecast_df[f'baseline_{size_key}'] = forecast_values
                last_disc = float(pd.to_numeric(full_history[f'discount_component_{size_key}'], errors='coerce').fillna(0.0).iloc[-1]) \
                    if len(full_history) > 0 else 0.0
                forecast_df[f'discount_component_{size_key}'] = last_disc

            points: List[BaselineForecastPoint] = []
            for _, row in full_history.iterrows():
                b12 = float(row.get('baseline_12-ML', 0.0))
                b18 = float(row.get('baseline_18-ML', 0.0))
                d12 = float(row.get('discount_component_12-ML', 0.0))
                d18 = float(row.get('discount_component_18-ML', 0.0))
                points.append(BaselineForecastPoint(
                    period=pd.to_datetime(row['Period']).strftime('%Y-%m'),
                    baseline_12_ml=b12,
                    baseline_18_ml=b18,
                    discount_component_12_ml=d12,
                    discount_component_18_ml=d18,
                    total_baseline=b12 + b18,
                    is_forecast=False,
                ))

            for _, row in forecast_df.iterrows():
                b12 = float(row.get('baseline_12-ML', 0.0))
                b18 = float(row.get('baseline_18-ML', 0.0))
                d12 = float(row.get('discount_component_12-ML', 0.0))
                d18 = float(row.get('discount_component_18-ML', 0.0))
                points.append(BaselineForecastPoint(
                    period=pd.to_datetime(row['Period']).strftime('%Y-%m'),
                    baseline_12_ml=b12,
                    baseline_18_ml=b18,
                    discount_component_12_ml=d12,
                    discount_component_18_ml=d18,
                    total_baseline=b12 + b18,
                    is_forecast=True,
                ))

            next_row = forecast_df.iloc[0] if not forecast_df.empty else None
            return BaselineForecastResponse(
                success=True,
                message="Baseline forecast completed successfully",
                forecast_months=forecast_months,
                next_month_12_ml=float(next_row.get('baseline_12-ML', 0.0)) if next_row is not None else 0.0,
                next_month_18_ml=float(next_row.get('baseline_18-ML', 0.0)) if next_row is not None else 0.0,
                next_month_total=(float(next_row.get('baseline_12-ML', 0.0)) + float(next_row.get('baseline_18-ML', 0.0))) if next_row is not None else 0.0,
                points=points,
            )
        except Exception as e:
            return BaselineForecastResponse(
                success=False,
                message=f"Error in baseline forecast: {str(e)}",
                points=[],
            )
