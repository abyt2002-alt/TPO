"""Data loading and warm-cache initialization.\n\nThis module isolates file ingestion and normalization so steps can focus on\nbusiness logic rather than storage concerns.\n"""

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
import traceback
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


class DataLoaderMixin:

    def _normalize_column_aliases(self, df: pd.DataFrame) -> pd.DataFrame:
        """Unify known lowercase/uppercase aliases from monthly parquet drops."""
        alias_map = {
            "subcategory": "Subcategory",
            "sizes": "Sizes",
            "brand": "Brand",
            "category": "Category",
            "variant": "Variant",
            "bill_no": "Bill_No",
            "invoice_no": "Invoice_No",
            "sales_order_no": "Sales_Order_No",
            "sku_code": "Sku_Code",
            "sku_name": "Sku_Name",
            "outlet_type": "Outlet_Type",
            "store_id": "Store_ID",
            "final_state": "Final_State",
            "final_outlet_classification": "Final_Outlet_Classification",
            "state": "State",
            "mrp": "MRP",
            "scheme_discount": "Scheme_Discount",
            "staggered_qps": "Staggered_qps",
            "basic_rate_per_pc_without_gst": "Basic_Rate_Per_PC_without_GST",
            "basic_rate_per_pc": "Basic_Rate_Per_PC",
            "selling_rate_without_gst_clp": "Selling_Rate_Per_PC_without_GST_CLP",
        }
        rename_map = {}
        for src, dst in alias_map.items():
            if src in df.columns and dst not in df.columns:
                rename_map[src] = dst
        if rename_map:
            df = df.rename(columns=rename_map)

        # Keep memory bounded by retaining only columns used across steps.
        keep_cols = [
            "Date",
            "Outlet_ID",
            "Bill_No",
            "Final_State",
            "Final_Outlet_Classification",
            "Outlet_Type",
            "State",
            "Category",
            "Subcategory",
            "Brand",
            "Sizes",
            "Slab",
            "Quantity",
            "MRP",
            "Net_Amt",
            "SalesValue_atBasicRate",
            "TotalDiscount",
            "Scheme_Discount",
            "Staggered_qps",
            "Basic_Rate_Per_PC_without_GST",
            "Basic_Rate_Per_PC",
            "Selling_Rate_Per_PC_without_GST_CLP",
            "Sku_Code",
            "Sku_Name",
        ]

        # Bill_No is required by Step 1; synthesize if source uses other id fields.
        if "Bill_No" not in df.columns:
            if "Invoice_No" in df.columns:
                df["Bill_No"] = df["Invoice_No"]
            elif "Sales_Order_No" in df.columns:
                df["Bill_No"] = df["Sales_Order_No"]

        # Net_Amt is required by Step 1; derive if absent.
        if "Net_Amt" not in df.columns:
            sales = pd.to_numeric(df.get("SalesValue_atBasicRate", 0.0), errors="coerce").fillna(0.0)
            disc = pd.to_numeric(df.get("TotalDiscount", 0.0), errors="coerce").fillna(0.0)
            df["Net_Amt"] = sales - disc

        # Force a stable schema across all files before concat.
        df = df.reindex(columns=keep_cols).copy()

        # Downcast numerics early to keep startup memory bounded.
        for c in [
            "Quantity",
            "MRP",
            "Net_Amt",
            "SalesValue_atBasicRate",
            "TotalDiscount",
            "Scheme_Discount",
            "Staggered_qps",
            "Basic_Rate_Per_PC_without_GST",
            "Basic_Rate_Per_PC",
            "Selling_Rate_Per_PC_without_GST_CLP",
        ]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")

        # Project scope for this app: keep only Insta Shampoo 12-ML/18-ML.
        if "Subcategory" in df.columns and "Sizes" in df.columns:
            sub = (
                df["Subcategory"]
                .astype(str)
                .str.upper()
                .str.replace(r"\s+", " ", regex=True)
                .str.strip()
            )
            siz = (
                df["Sizes"]
                .astype(str)
                .str.upper()
                .str.replace(" ", "", regex=False)
                .str.strip()
            )
            mask = sub.isin({"STX INSTA SHAMPOO", "STREAX INSTA SHAMPOO"}) & siz.isin({"12-ML", "18-ML"})
            df = df.loc[mask].copy()

        return df

    def load_data(self):
        """Load parquet files from DATA folder"""
        try:
            services_dir = Path(__file__).resolve().parent
            backend_dir = services_dir.parent.parent
            project_root = backend_dir.parent

            # Keep compatibility with old layouts + support project-root DATA folder.
            candidate_paths = [
                Path.cwd() / "DATA",                         # when running from project root
                Path.cwd().parent / "DATA",                  # when running from backend/
                backend_dir / "DATA",                        # legacy backend/DATA
                backend_dir / "step3_filtered_engineered",   # older documented path
                project_root / "DATA",                       # current user setup
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
                df = self._normalize_column_aliases(df)
                dfs.append(df)
            
            # Memory-stable combine: concatenate per-column (avoids pandas object-block vstack).
            if not dfs:
                print("Warning: No normalized parquet frames to combine")
                return None
            ordered_cols = list(dfs[0].columns)
            buffers = {c: [] for c in ordered_cols}
            for frame in dfs:
                for c in ordered_cols:
                    buffers[c].append(frame[c].to_numpy(copy=False))
            combined_df = pd.DataFrame({
                c: np.concatenate(arr_list, axis=0) if arr_list else np.array([])
                for c, arr_list in buffers.items()
            })
            combined_df['Date'] = pd.to_datetime(combined_df['Date'])

            # Normalize text dimensions so mixed-case monthly drops still match filters.
            for text_col in ['Category', 'Subcategory', 'Brand', 'Final_State', 'Final_Outlet_Classification', 'Outlet_Type']:
                if text_col in combined_df.columns:
                    combined_df[text_col] = (
                        combined_df[text_col]
                        .astype(str)
                        .str.upper()
                        .str.replace(r'\s+', ' ', regex=True)
                        .str.strip()
                    )
            if 'Sizes' in combined_df.columns:
                combined_df['Sizes'] = (
                    combined_df['Sizes']
                    .astype(str)
                    .str.upper()
                    .str.replace(' ', '', regex=False)
                    .str.strip()
                )
            
            self.data_cache = combined_df
            print(f"Loaded {len(combined_df):,} rows from {len(parquet_files)} files")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            print(traceback.format_exc())
            self.data_cache = None
