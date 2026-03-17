"""EDA option/scope/aggregation services.\n\nThis module owns EDA-specific filtering and aggregation logic so it can evolve\nwithout affecting RFM/discount/modeling pipelines.\n"""

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
    EDAContributionRow, EDAOptionsResponse,
    SlabTrendEDARequest, SlabTrendEDAResponse, SlabTrendEDASeries, SlabTrendEDAPoint
)


class EDAServiceMixin:

    def _build_eda_product_options(self, df: pd.DataFrame, top_n: int) -> List[EDAProductOption]:
        if df.empty or 'Sku_Code' not in df.columns:
            return []

        work = df.copy()
        if 'Sku_Name' not in work.columns:
            work['Sku_Name'] = work['Sku_Code'].astype(str)
        if 'Brand' not in work.columns:
            work['Brand'] = 'Unknown'
        if 'Sizes' not in work.columns:
            work['Sizes'] = 'NA'

        grouped = (
            work.groupby(['Sku_Code', 'Sku_Name', 'Brand', 'Sizes'], as_index=False)
            .agg(
                sales_value=('SalesValue_atBasicRate', 'sum'),
                quantity=('Quantity', 'sum'),
            )
            .sort_values(['sales_value', 'quantity'], ascending=[False, False], kind='mergesort')
            .head(max(20, int(top_n)))
        )

        options: List[EDAProductOption] = []
        for _, row in grouped.iterrows():
            code = str(row.get('Sku_Code', '') or '').strip()
            if not code:
                continue
            name = str(row.get('Sku_Name', code) or code).strip()
            brand = str(row.get('Brand', 'Unknown') or 'Unknown').strip()
            size = str(row.get('Sizes', 'NA') or 'NA').strip()
            label = f"{name} | {size} | {brand} ({code})"
            options.append(
                EDAProductOption(
                    code=code,
                    name=name,
                    brand=brand,
                    size=size,
                    label=label,
                )
            )
        return options


    def _build_eda_mix_rows(
        self,
        df: pd.DataFrame,
        group_col: str,
        total_sales_value: float,
        total_quantity: float,
        max_rows: int = 12,
    ) -> List[EDAContributionRow]:
        if df.empty or group_col not in df.columns:
            return []

        grouped = (
            df.groupby(group_col, dropna=False, as_index=False)
            .agg(
                sales_value=('SalesValue_atBasicRate', 'sum'),
                quantity=('Quantity', 'sum'),
            )
            .sort_values('sales_value', ascending=False, kind='mergesort')
            .head(max_rows)
        )

        rows: List[EDAContributionRow] = []
        for _, row in grouped.iterrows():
            label = str(row.get(group_col, 'Unknown') or 'Unknown')
            sales_value = float(row.get('sales_value', 0.0) or 0.0)
            quantity = float(row.get('quantity', 0.0) or 0.0)
            value_pct = (sales_value / total_sales_value * 100.0) if total_sales_value > 0 else 0.0
            volume_pct = (quantity / total_quantity * 100.0) if total_quantity > 0 else 0.0
            rows.append(
                EDAContributionRow(
                    key=label,
                    label=label,
                    sales_value=sales_value,
                    quantity=quantity,
                    value_pct=value_pct,
                    volume_pct=volume_pct,
                )
            )
        return rows


    def _apply_eda_scope(self, request: EDARequest) -> Optional[pd.DataFrame]:
        if self.data_cache is None:
            return None
        # Avoid copying full dataset on every EDA request; filters create new frames.
        df = self._apply_base_filters(self.data_cache, request)
        selected_classes = [str(x) for x in (request.outlet_classifications or []) if str(x).strip()]
        if selected_classes and 'Final_Outlet_Classification' in df.columns:
            df = df[df['Final_Outlet_Classification'].astype(str).isin(selected_classes)]
        return df


    async def get_eda_options(self, request: EDARequest) -> EDAOptionsResponse:
        df = self._apply_eda_scope(request)
        if df is None:
            return EDAOptionsResponse(
                success=False,
                message="Data not loaded",
                product_options=[],
                outlet_classifications=[],
                matching_rows=0,
            )
        if df.empty:
            return EDAOptionsResponse(
                success=False,
                message="No data matches selected EDA filters",
                product_options=[],
                outlet_classifications=[],
                matching_rows=0,
            )

        options = self._build_eda_product_options(df, request.top_n_products)
        classifications = []
        if 'Final_Outlet_Classification' in df.columns:
            classifications = sorted(df['Final_Outlet_Classification'].dropna().astype(str).unique().tolist())

        return EDAOptionsResponse(
            success=True,
            message="EDA options loaded successfully",
            product_options=options,
            outlet_classifications=classifications,
            matching_rows=int(len(df)),
        )


    async def get_eda_overview(self, request: EDARequest) -> EDAResponse:
        df_base = self._apply_eda_scope(request)
        if df_base is None:
            return EDAResponse(
                success=False,
                message="Data not loaded",
                summary={},
                product_options=[],
                product_contributions=[],
                state_mix=[],
                outlet_class_mix=[],
                brand_mix=[],
                category_mix=[],
            )
        if df_base.empty:
            return EDAResponse(
                success=False,
                message="No data matches selected EDA filters",
                summary={},
                product_options=[],
                product_contributions=[],
                state_mix=[],
                outlet_class_mix=[],
                brand_mix=[],
                category_mix=[],
            )

        product_options = self._build_eda_product_options(df_base, request.top_n_products)
        selected_codes = [str(x) for x in (request.product_codes or []) if str(x).strip()]
        if selected_codes and 'Sku_Code' in df_base.columns:
            df_scope = df_base[df_base['Sku_Code'].astype(str).isin(selected_codes)].copy()
        else:
            df_scope = df_base.copy()

        if df_scope.empty:
            return EDAResponse(
                success=False,
                message="No rows found for selected product(s)",
                summary={},
                product_options=product_options,
                product_contributions=[],
                state_mix=[],
                outlet_class_mix=[],
                brand_mix=[],
                category_mix=[],
            )

        if 'Sku_Name' not in df_scope.columns:
            df_scope['Sku_Name'] = df_scope['Sku_Code'].astype(str)
        if 'Brand' not in df_scope.columns:
            df_scope['Brand'] = 'Unknown'
            df_base['Brand'] = 'Unknown'
        if 'Category' not in df_scope.columns:
            df_scope['Category'] = 'Unknown'
        if 'Subcategory' not in df_scope.columns:
            df_scope['Subcategory'] = 'Unknown'
        if 'Sizes' not in df_scope.columns:
            df_scope['Sizes'] = 'NA'

        product_grouped = (
            df_scope.groupby(['Sku_Code', 'Sku_Name', 'Brand', 'Category', 'Subcategory', 'Sizes'], as_index=False)
            .agg(
                sales_value=('SalesValue_atBasicRate', 'sum'),
                quantity=('Quantity', 'sum'),
            )
            .sort_values(['sales_value', 'quantity'], ascending=[False, False], kind='mergesort')
        )
        if not selected_codes:
            product_grouped = product_grouped.head(max(50, int(request.top_n_products)))

        brand_totals = (
            df_base.groupby('Brand', as_index=False)
            .agg(
                brand_sales_value=('SalesValue_atBasicRate', 'sum'),
                brand_quantity=('Quantity', 'sum'),
            )
        )
        product_joined = product_grouped.merge(brand_totals, on='Brand', how='left')

        product_contributions: List[EDAProductContribution] = []
        for _, row in product_joined.iterrows():
            sales_value = float(row.get('sales_value', 0.0) or 0.0)
            quantity = float(row.get('quantity', 0.0) or 0.0)
            brand_sales = float(row.get('brand_sales_value', 0.0) or 0.0)
            brand_qty = float(row.get('brand_quantity', 0.0) or 0.0)
            product_contributions.append(
                EDAProductContribution(
                    code=str(row.get('Sku_Code', '') or ''),
                    name=str(row.get('Sku_Name', '') or ''),
                    brand=str(row.get('Brand', '') or ''),
                    category=str(row.get('Category', '') or ''),
                    subcategory=str(row.get('Subcategory', '') or ''),
                    size=str(row.get('Sizes', '') or ''),
                    sales_value=sales_value,
                    quantity=quantity,
                    brand_sales_value=brand_sales,
                    brand_quantity=brand_qty,
                    value_contribution_pct=(sales_value / brand_sales * 100.0) if brand_sales > 0 else 0.0,
                    volume_contribution_pct=(quantity / brand_qty * 100.0) if brand_qty > 0 else 0.0,
                )
            )

        total_sales_value = float(df_scope['SalesValue_atBasicRate'].sum())
        total_quantity = float(df_scope['Quantity'].sum())
        summary = {
            "total_sales_value": total_sales_value,
            "total_quantity": total_quantity,
            "total_outlets": float(df_scope['Outlet_ID'].nunique()) if 'Outlet_ID' in df_scope.columns else 0.0,
            "total_rows": float(len(df_scope)),
            "distinct_products": float(df_scope['Sku_Code'].nunique()) if 'Sku_Code' in df_scope.columns else 0.0,
            "distinct_brands": float(df_scope['Brand'].nunique()) if 'Brand' in df_scope.columns else 0.0,
            "selected_products": float(len(selected_codes)),
        }

        state_mix = self._build_eda_mix_rows(df_scope, 'Final_State', total_sales_value, total_quantity, max_rows=10)
        outlet_class_mix = self._build_eda_mix_rows(df_scope, 'Final_Outlet_Classification', total_sales_value, total_quantity, max_rows=10)
        brand_mix = self._build_eda_mix_rows(df_scope, 'Brand', total_sales_value, total_quantity, max_rows=12)

        category_level_mix: List[EDAContributionRow] = []
        subcategory_within_category_mix: List[EDAContributionRow] = []
        selected_categories = [str(x) for x in (request.categories or []) if str(x).strip()]
        selected_subcategories = [str(x) for x in (request.subcategories or []) if str(x).strip()]

        # Level 1: selected category contribution vs all categories (same base scope except category/subcategory).
        if selected_categories:
            category_universe_request = EDARequest(
                run_id=request.run_id,
                states=request.states,
                categories=[],
                subcategories=[],
                brands=request.brands,
                sizes=request.sizes,
                outlet_classifications=request.outlet_classifications,
                product_codes=[],
                top_n_products=request.top_n_products,
            )
            selected_category_request = EDARequest(
                run_id=request.run_id,
                states=request.states,
                categories=selected_categories,
                subcategories=[],
                brands=request.brands,
                sizes=request.sizes,
                outlet_classifications=request.outlet_classifications,
                product_codes=[],
                top_n_products=request.top_n_products,
            )
            df_category_universe = self._apply_eda_scope(category_universe_request)
            df_selected_category = self._apply_eda_scope(selected_category_request)

            if df_category_universe is not None and not df_category_universe.empty and df_selected_category is not None:
                universe_sales = float(df_category_universe['SalesValue_atBasicRate'].sum())
                universe_qty = float(df_category_universe['Quantity'].sum())
                selected_cat_sales = float(df_selected_category['SalesValue_atBasicRate'].sum())
                selected_cat_qty = float(df_selected_category['Quantity'].sum())
                other_cat_sales = max(0.0, universe_sales - selected_cat_sales)
                other_cat_qty = max(0.0, universe_qty - selected_cat_qty)
                category_level_mix.extend([
                    EDAContributionRow(
                        key='selected_categories',
                        label='Selected Category(ies)',
                        sales_value=selected_cat_sales,
                        quantity=selected_cat_qty,
                        value_pct=(selected_cat_sales / universe_sales * 100.0) if universe_sales > 0 else 0.0,
                        volume_pct=(selected_cat_qty / universe_qty * 100.0) if universe_qty > 0 else 0.0,
                    ),
                    EDAContributionRow(
                        key='other_categories',
                        label='Other Categories',
                        sales_value=other_cat_sales,
                        quantity=other_cat_qty,
                        value_pct=(other_cat_sales / universe_sales * 100.0) if universe_sales > 0 else 0.0,
                        volume_pct=(other_cat_qty / universe_qty * 100.0) if universe_qty > 0 else 0.0,
                    ),
                ])
        else:
            category_level_mix = self._build_eda_mix_rows(df_scope, 'Category', total_sales_value, total_quantity, max_rows=12)

        # Level 2: subcategory contribution within selected category scope.
        subcategory_base_request = EDARequest(
            run_id=request.run_id,
            states=request.states,
            categories=selected_categories if selected_categories else request.categories,
            subcategories=[],
            brands=request.brands,
            sizes=request.sizes,
            outlet_classifications=request.outlet_classifications,
            product_codes=[],
            top_n_products=request.top_n_products,
        )
        df_subcategory_base = self._apply_eda_scope(subcategory_base_request)
        if df_subcategory_base is not None and not df_subcategory_base.empty:
            sub_total_sales = float(df_subcategory_base['SalesValue_atBasicRate'].sum())
            sub_total_qty = float(df_subcategory_base['Quantity'].sum())
            if selected_subcategories:
                selected_sub_sales = float(df_scope['SalesValue_atBasicRate'].sum())
                selected_sub_qty = float(df_scope['Quantity'].sum())
                other_sub_sales = max(0.0, sub_total_sales - selected_sub_sales)
                other_sub_qty = max(0.0, sub_total_qty - selected_sub_qty)
                subcategory_within_category_mix = [
                    EDAContributionRow(
                        key='selected_subcategories',
                        label='Selected Subcategory(ies) in Selected Category',
                        sales_value=selected_sub_sales,
                        quantity=selected_sub_qty,
                        value_pct=(selected_sub_sales / sub_total_sales * 100.0) if sub_total_sales > 0 else 0.0,
                        volume_pct=(selected_sub_qty / sub_total_qty * 100.0) if sub_total_qty > 0 else 0.0,
                    ),
                    EDAContributionRow(
                        key='other_subcategories',
                        label='Other Subcategories in Selected Category',
                        sales_value=other_sub_sales,
                        quantity=other_sub_qty,
                        value_pct=(other_sub_sales / sub_total_sales * 100.0) if sub_total_sales > 0 else 0.0,
                        volume_pct=(other_sub_qty / sub_total_qty * 100.0) if sub_total_qty > 0 else 0.0,
                    ),
                ]
            else:
                subcategory_within_category_mix = self._build_eda_mix_rows(
                    df_subcategory_base,
                    'Subcategory',
                    sub_total_sales,
                    sub_total_qty,
                    max_rows=12
                )

        # Per-category subcategory split (for multiple selected categories, show separate sections).
        subcategory_within_category_sections: List[Dict[str, Any]] = []
        category_for_within_request = EDARequest(
            run_id=request.run_id,
            states=request.states,
            categories=selected_categories if selected_categories else request.categories,
            subcategories=[],
            brands=request.brands,
            sizes=request.sizes,
            outlet_classifications=request.outlet_classifications,
            product_codes=[],
            top_n_products=request.top_n_products,
        )
        df_for_within = self._apply_eda_scope(category_for_within_request)
        if df_for_within is not None and not df_for_within.empty and 'Category' in df_for_within.columns and 'Subcategory' in df_for_within.columns:
            grouped_within = (
                df_for_within.groupby(['Category', 'Subcategory'], as_index=False)
                .agg(
                    sales_value=('SalesValue_atBasicRate', 'sum'),
                    quantity=('Quantity', 'sum'),
                )
            )
            for category_value, cat_df in grouped_within.groupby('Category', sort=True):
                cat_sales_total = float(cat_df['sales_value'].sum())
                cat_qty_total = float(cat_df['quantity'].sum())
                rows = []
                for _, row in cat_df.sort_values('sales_value', ascending=False, kind='mergesort').head(12).iterrows():
                    sales_value = float(row.get('sales_value', 0.0) or 0.0)
                    quantity = float(row.get('quantity', 0.0) or 0.0)
                    label = str(row.get('Subcategory', 'Unknown') or 'Unknown')
                    rows.append({
                        "key": label,
                        "label": label,
                        "sales_value": sales_value,
                        "quantity": quantity,
                        "value_pct": (sales_value / cat_sales_total * 100.0) if cat_sales_total > 0 else 0.0,
                        "volume_pct": (quantity / cat_qty_total * 100.0) if cat_qty_total > 0 else 0.0,
                    })
                subcategory_within_category_sections.append({
                    "category": str(category_value),
                    "rows": rows,
                })

        # Backward-compatible alias used by current frontend section.
        category_mix = category_level_mix if category_level_mix else self._build_eda_mix_rows(
            df_scope, 'Category', total_sales_value, total_quantity, max_rows=12
        )

        return EDAResponse(
            success=True,
            message="EDA overview loaded successfully",
            summary=summary,
            product_options=product_options,
            product_contributions=product_contributions,
            state_mix=state_mix,
            outlet_class_mix=outlet_class_mix,
            brand_mix=brand_mix,
            category_level_mix=category_level_mix,
            subcategory_within_category_mix=subcategory_within_category_mix,
            subcategory_within_category_sections=subcategory_within_category_sections,
            category_mix=category_mix,
        )


    async def get_slab_trend_eda(self, request: SlabTrendEDARequest) -> SlabTrendEDAResponse:
        """Return month-wise slab discount and volume trends for 12-ML and 18-ML."""
        scope = self._build_step2_scope(request)
        if scope is None:
            return SlabTrendEDAResponse(
                success=False,
                message="Data not loaded",
                periods=[],
                series=[],
            )

        df = scope.get("df_scope_all_slabs")
        if df is None or df.empty:
            return SlabTrendEDAResponse(
                success=False,
                message="No data matches selected filters",
                periods=[],
                series=[],
            )

        work = df.copy()
        if "Date" not in work.columns or "Sizes" not in work.columns or "Slab" not in work.columns:
            return SlabTrendEDAResponse(
                success=False,
                message="Required columns missing for slab trend EDA",
                periods=[],
                series=[],
            )

        work["Date"] = pd.to_datetime(work["Date"], errors="coerce")
        work = work.dropna(subset=["Date"])
        if work.empty:
            return SlabTrendEDAResponse(
                success=False,
                message="No valid dated rows after filtering",
                periods=[],
                series=[],
            )

        work = work[work["Sizes"].astype(str).isin(["12-ML", "18-ML"])].copy()
        work["Slab"] = work["Slab"].astype(str)
        work = work[work["Slab"].map(self._is_step2_allowed_slab)]
        selected_slabs = self._normalize_step2_slab_values(getattr(request, "slabs", []) or [])
        if selected_slabs:
            work = work[work["Slab"].isin(selected_slabs)]
        if work.empty:
            return SlabTrendEDAResponse(
                success=False,
                message="No slab rows available for selected filters",
                periods=[],
                series=[],
            )

        work["period"] = work["Date"].dt.to_period("M").astype(str)
        work["MRP"] = pd.to_numeric(work.get("MRP"), errors="coerce").fillna(0.0)
        work["Quantity"] = pd.to_numeric(work.get("Quantity"), errors="coerce").fillna(0.0)
        work["mrp_x_qty"] = work["MRP"] * work["Quantity"]

        grouped = (
            work.groupby(["period", "Sizes", "Slab"], as_index=False)
            .agg(
                quantity=("Quantity", "sum"),
                sales_value=("SalesValue_atBasicRate", "sum"),
                discount_value=("TotalDiscount", "sum"),
                mrp_x_qty=("mrp_x_qty", "sum"),
            )
            .sort_values(["Sizes", "Slab", "period"], kind="mergesort")
        )

        grouped["discount_pct"] = np.where(
            grouped["sales_value"] > 0,
            grouped["discount_value"] / grouped["sales_value"] * 100.0,
            0.0,
        )
        grouped["volume_change_pct"] = (
            grouped.groupby(["Sizes", "Slab"])["quantity"]
            .pct_change()
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            * 100.0
        )
        grouped["revenue_change_pct"] = (
            grouped.groupby(["Sizes", "Slab"])["sales_value"]
            .pct_change()
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            * 100.0
        )
        grouped["mrp"] = np.where(
            grouped["quantity"] > 0,
            grouped["mrp_x_qty"] / grouped["quantity"],
            0.0,
        )

        periods = sorted(grouped["period"].astype(str).unique().tolist())
        series: List[SlabTrendEDASeries] = []

        for size_key in ["12-ML", "18-ML"]:
            size_df = grouped[grouped["Sizes"] == size_key].copy()
            if size_df.empty:
                continue
            slab_order = sorted(size_df["Slab"].astype(str).unique().tolist(), key=self._slab_sort_key)
            for slab_key in slab_order:
                slab_df = size_df[size_df["Slab"].astype(str) == slab_key].copy()
                if slab_df.empty:
                    continue
                point_map = {}
                for _, row in slab_df.iterrows():
                    point_map[str(row["period"])] = SlabTrendEDAPoint(
                        period=str(row["period"]),
                        discount_pct=float(row.get("discount_pct", 0.0) or 0.0),
                        volume=float(row.get("quantity", 0.0) or 0.0),
                        volume_change_pct=float(row.get("volume_change_pct", 0.0) or 0.0),
                        revenue=float(row.get("sales_value", 0.0) or 0.0),
                        revenue_change_pct=float(row.get("revenue_change_pct", 0.0) or 0.0),
                        mrp=float(row.get("mrp", 0.0) or 0.0),
                    )
                points = [point_map[p] for p in periods if p in point_map]
                if points:
                    series.append(
                        SlabTrendEDASeries(
                            size=size_key,
                            slab=slab_key,
                            points=points,
                        )
                    )

        return SlabTrendEDAResponse(
            success=True,
            message="Slab trend EDA loaded successfully",
            periods=periods,
            series=series,
        )
