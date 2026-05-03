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


class Step1SegmentationMixin:

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
            if not getattr(self, 'db_path', None):
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


    async def get_available_filters(self) -> Dict[str, List[str]]:
        """Get available filter options — queries SQLite, no full data load."""
        empty = {"states": [], "categories": [], "subcategories": [],
                 "brands": [], "sizes": [], "outlet_classifications": []}
        if not getattr(self, 'db_path', None):
            return empty
        try:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            def distinct(col):
                rows = conn.execute(
                    f"SELECT DISTINCT {col} FROM transactions "
                    f"WHERE {col} IS NOT NULL AND {col} != '' ORDER BY {col}"
                ).fetchall()
                return [r[0] for r in rows if r[0]]
            states = distinct("Final_State")
            categories = distinct("Category")
            subcategories = distinct("Subcategory")
            brands = distinct("Brand")
            sizes = distinct("Sizes")
            oc_raw = conn.execute(
                "SELECT DISTINCT Final_Outlet_Classification FROM transactions "
                "WHERE Final_Outlet_Classification IS NOT NULL AND Final_Outlet_Classification != ''"
            ).fetchall()
            conn.close()
            oc_series = pd.Series([r[0] for r in oc_raw if r[0]])
            return {
                "states": states,
                "categories": categories,
                "subcategories": subcategories,
                "brands": brands,
                "sizes": sizes,
                "outlet_classifications": self._normalized_outlet_classification_options(oc_series) if len(oc_series) > 0 else [],
            }
        except Exception as e:
            print(f"get_available_filters error: {e}")
            return empty


    async def get_cascading_filters(self, current_filters: dict) -> Dict[str, List[str]]:
        """Get filtered options based on current selections — pure SQL, no full data load."""
        if not getattr(self, 'db_path', None):
            return await self.get_available_filters()

        states = current_filters.get('states') or []
        categories = current_filters.get('categories') or []
        subcategories = current_filters.get('subcategories') or []
        brands = current_filters.get('brands') or []
        sizes = current_filters.get('sizes') or []

        conn = sqlite3.connect(self.db_path, check_same_thread=False)

        def _w(filters_map):
            clauses, params = [], []
            for col, vals in filters_map.items():
                if vals:
                    placeholders = ','.join(['?'] * len(vals))
                    clauses.append(f"{col} IN ({placeholders})")
                    params.extend(vals)
            return (" AND ".join(clauses) if clauses else "1=1"), params

        def distinct(col, filters_map):
            w, p = _w(filters_map)
            rows = conn.execute(
                f"SELECT DISTINCT {col} FROM transactions WHERE {w} AND {col} IS NOT NULL AND {col} != '' ORDER BY {col}", p
            ).fetchall()
            return [r[0] for r in rows if r[0]]

        all_states      = distinct("Final_State", {})
        all_categories  = distinct("Category",    {"Final_State": states})
        all_subcategories = distinct("Subcategory", {"Final_State": states, "Category": categories})
        all_brands      = distinct("Brand",       {"Final_State": states, "Category": categories, "Subcategory": subcategories})
        all_sizes       = distinct("Sizes",       {"Final_State": states, "Category": categories, "Subcategory": subcategories, "Brand": brands})

        w, p = _w({"Final_State": states, "Category": categories, "Subcategory": subcategories, "Brand": brands, "Sizes": sizes})
        oc_raw = conn.execute(
            f"SELECT DISTINCT Final_Outlet_Classification FROM transactions WHERE {w} "
            f"AND Final_Outlet_Classification IS NOT NULL AND Final_Outlet_Classification != ''", p
        ).fetchall()
        conn.close()

        oc_series = pd.Series([r[0] for r in oc_raw if r[0]])
        return {
            "states": all_states,
            "categories": all_categories,
            "subcategories": all_subcategories,
            "brands": all_brands,
            "sizes": all_sizes,
            "outlet_classifications": self._normalized_outlet_classification_options(oc_series) if len(oc_series) > 0 else [],
        }
