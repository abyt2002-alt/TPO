"""Shared filtering and scope construction across steps.\n\nThis module builds reusable filtered scopes from step inputs and keeps all\nnormalization/cascade behavior in one place.\n"""

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


class ScopeBuilderMixin:

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
        if getattr(request, 'outlet_classifications', None):
            if 'Final_Outlet_Classification' in df.columns:
                df = df[df['Final_Outlet_Classification'].isin(request.outlet_classifications)]
        return df


    def _base_filter_cache_key(self, request: RFMRequest):
        return (
            tuple(sorted(request.states or [])),
            tuple(sorted(request.categories or [])),
            tuple(sorted(request.subcategories or [])),
            tuple(sorted(request.brands or [])),
            tuple(sorted(request.sizes or [])),
            tuple(sorted(getattr(request, 'outlet_classifications', None) or [])),
            int(request.recency_threshold),
            int(request.frequency_threshold),
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
        selected_classifications = self._normalize_step2_outlet_classifications(
            list(getattr(request, 'outlet_classifications', []) or [])
        )
        selected_slabs = self._normalize_step2_slab_values(
            [str(x) for x in (getattr(request, 'slabs', []) or [])]
        )
        selected_outlets = [str(x) for x in (getattr(request, 'outlet_ids', []) or [])]

        rfm_scope = rfm.copy()
        if selected_segments:
            rfm_scope = rfm_scope[rfm_scope['RFM_Segment'].isin(selected_segments)]
        if selected_outlets:
            rfm_scope = rfm_scope[rfm_scope['Outlet_ID'].astype(str).isin(selected_outlets)]

        outlet_ids = set(rfm_scope['Outlet_ID'].astype(str).tolist())

        if selected_classifications and 'Final_Outlet_Classification' in df.columns:
            class_groups = self._to_step2_outlet_group_series(df['Final_Outlet_Classification'])
            class_outlets = set(
                df[class_groups.isin(selected_classifications)]['Outlet_ID'].astype(str).tolist()
            )
            outlet_ids = outlet_ids.intersection(class_outlets)

        df_scope = df[df['Outlet_ID'].astype(str).isin(outlet_ids)].copy()
        df_scope_all_slabs = df_scope.copy()
        df_scope_all_slabs = self._apply_step2_slab_definition(df_scope_all_slabs, request)
        df_scope = self._apply_step2_slab_definition(df_scope, request)
        if 'Slab' in df_scope.columns:
            df_scope = df_scope[df_scope['Slab'].astype(str).map(self._is_step2_allowed_slab)]
            if selected_slabs:
                df_scope = df_scope[df_scope['Slab'].astype(str).isin(selected_slabs)]

        rfm_scope = rfm[rfm['Outlet_ID'].astype(str).isin(set(df_scope['Outlet_ID'].astype(str).tolist()))].copy()

        return {
            'dataset': dataset,
            'df_scope': df_scope,
            'df_scope_all_slabs': df_scope_all_slabs,
            'rfm_scope': rfm_scope,
        }


    def _to_step2_outlet_group_label(self, value: Any) -> str:
        raw = str(value or '').strip().upper().replace(' ', '')
        if not raw:
            return ''
        if raw == 'WH':
            return 'WH'
        # Business rule for Step 2: merge SS + OtherGT (and any non-WH) under OtherGT.
        return 'OtherGT'


    def _to_step2_outlet_group_series(self, series: pd.Series) -> pd.Series:
        if series is None:
            return pd.Series(dtype=str)
        normalized = (
            series.fillna('')
            .astype(str)
            .str.upper()
            .str.replace(' ', '', regex=False)
            .str.strip()
        )
        out = pd.Series(np.where(normalized.eq('WH'), 'WH', np.where(normalized.eq(''), '', 'OtherGT')), index=series.index)
        return out.astype(str)


    def _normalize_step2_outlet_classifications(self, values: List[Any]) -> List[str]:
        normalized = []
        for value in (values or []):
            label = self._to_step2_outlet_group_label(value)
            if label:
                normalized.append(label)
        # Preserve consistent display order.
        order = {'OtherGT': 0, 'WH': 1}
        unique = list(dict.fromkeys(normalized))
        return sorted(unique, key=lambda x: (order.get(x, 99), x))


    def _normalize_step2_slab_values(self, values: List[Any]) -> List[str]:
        normalized = [str(v) for v in (values or []) if self._is_step2_allowed_slab(v)]
        unique = list(dict.fromkeys(normalized))
        return sorted(unique, key=self._slab_sort_key)


    def _normalize_step2_slab_definition_mode(self, mode: Any) -> str:
        text = str(mode or "").strip().lower()
        return "define" if text == "define" else "data"


    def _normalize_step2_defined_slab_count(self, count: Any) -> int:
        try:
            value = int(count)
        except Exception:
            value = 5
        return max(2, min(20, value))


    def _normalize_step2_defined_slab_thresholds(self, thresholds: Any, slab_count: int) -> List[float]:
        default_thresholds = [8.0, 32.0, 576.0, 960.0]
        expected = max(1, int(slab_count) - 1)
        parsed = []
        for raw in (thresholds or []):
            try:
                value = float(raw)
                if np.isfinite(value):
                    parsed.append(value)
            except Exception:
                continue

        if len(parsed) == 0:
            parsed = default_thresholds[:expected]
        parsed = sorted(list(dict.fromkeys(parsed)))

        if len(parsed) < expected:
            seed = default_thresholds.copy()
            while len(seed) < expected:
                last = seed[-1] if seed else 1.0
                seed.append(last + max(1.0, abs(last) * 0.1))
            parsed = (parsed + seed)[:expected]
            parsed = sorted(list(dict.fromkeys(parsed)))
            while len(parsed) < expected:
                last = parsed[-1] if parsed else 1.0
                parsed.append(last + max(1.0, abs(last) * 0.1))

        if len(parsed) > expected:
            parsed = parsed[:expected]

        return [float(x) for x in parsed]


    def _normalize_step2_size_key(self, size_value: Any) -> str:
        return str(size_value or '').upper().replace(' ', '').strip()


    def _normalize_step2_defined_slab_profiles(self, profiles: Any) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        if not isinstance(profiles, dict):
            return out
        for raw_size, raw_cfg in profiles.items():
            size_key = self._normalize_step2_size_key(raw_size)
            if not size_key:
                continue
            cfg = raw_cfg if isinstance(raw_cfg, dict) else {}
            slab_count = self._normalize_step2_defined_slab_count(cfg.get('defined_slab_count', 5))
            thresholds = self._normalize_step2_defined_slab_thresholds(
                cfg.get('defined_slab_thresholds', None),
                slab_count=slab_count,
            )
            out[size_key] = {
                'defined_slab_count': slab_count,
                'defined_slab_thresholds': thresholds,
            }
        return out
