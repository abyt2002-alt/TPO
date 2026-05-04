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
        """Load parquet files from DATA folder into a SQLite file cache.

        On first run this builds the SQLite file (slow once).
        On subsequent restarts it reuses the existing file (instant).
        Memory usage stays ~50 MB instead of 1.6 GB.
        """
        try:
            services_dir = Path(__file__).resolve().parent
            backend_dir = services_dir.parent.parent
            project_root = backend_dir.parent

            candidate_paths = [
                Path.cwd() / "demo_data",
                Path.cwd().parent / "demo_data",
                backend_dir / "demo_data",
                project_root / "demo_data",
                Path.cwd() / "DATA",
                Path.cwd().parent / "DATA",
                backend_dir / "DATA",
                backend_dir / "step3_filtered_engineered",
                project_root / "DATA",
            ]

            folder_path = next((p for p in candidate_paths if p.exists()), None)

            if folder_path is None:
                print("Warning: Could not find DATA or demo_data folder")
                return

            parquet_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.parquet')])

            if not parquet_files:
                print("Warning: No parquet files found")
                return

            db_path = folder_path.parent / "qps_data.db"

            # Reuse existing SQLite if it already has rows — avoids rebuilding on every restart.
            if db_path.exists():
                try:
                    conn = sqlite3.connect(str(db_path))
                    row_count = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
                    conn.close()
                    if row_count > 0:
                        self.db_path = str(db_path)
                        self.data_cache = True  # sentinel: not None, but data is in SQLite
                        print(f"Reusing SQLite cache: {row_count:,} rows at {db_path}")
                        return
                except Exception:
                    pass  # fall through to rebuild

            print(f"Building SQLite cache from {len(parquet_files)} parquet files...")
            conn = sqlite3.connect(str(db_path))
            first_file = True
            total_rows = 0

            for file in parquet_files:
                df = pd.read_parquet(folder_path / file)
                df = self._normalize_column_aliases(df)

                # Normalize text columns in-place before writing
                for text_col in ['Category', 'Subcategory', 'Brand', 'Final_State',
                                  'Final_Outlet_Classification', 'Outlet_Type']:
                    if text_col in df.columns:
                        df[text_col] = (df[text_col].astype(str)
                                        .str.upper()
                                        .str.replace(r'\s+', ' ', regex=True)
                                        .str.strip())
                if 'Sizes' in df.columns:
                    df['Sizes'] = (df['Sizes'].astype(str)
                                   .str.upper()
                                   .str.replace(' ', '', regex=False)
                                   .str.strip())

                # Store dates as ISO strings (SQLite has no native datetime)
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')

                df.to_sql('transactions', conn,
                          if_exists='replace' if first_file else 'append',
                          index=False)
                total_rows += len(df)
                first_file = False
                del df  # free RAM immediately after writing each file

            # Single-column indexes for WHERE filtering
            conn.execute("CREATE INDEX IF NOT EXISTS idx_state  ON transactions(Final_State)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cat    ON transactions(Category)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_subcat ON transactions(Subcategory)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_brand  ON transactions(Brand)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sizes  ON transactions(Sizes)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_outlet ON transactions(Outlet_ID)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_oc     ON transactions(Final_Outlet_Classification)")
            # Composite indexes for cascade filter DISTINCT queries (index-only scans)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cas_cat    ON transactions(Final_State, Category)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cas_subcat ON transactions(Final_State, Category, Subcategory)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cas_brand  ON transactions(Final_State, Category, Subcategory, Brand)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cas_size   ON transactions(Final_State, Category, Subcategory, Brand, Sizes)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cas_oc     ON transactions(Final_State, Category, Subcategory, Brand, Sizes, Final_Outlet_Classification)")
            conn.commit()
            conn.close()

            self.db_path = str(db_path)
            self.data_cache = True  # sentinel: data lives in SQLite, not in RAM
            print(f"SQLite cache built: {total_rows:,} rows from {len(parquet_files)} files")

        except Exception as e:
            print(f"Error loading data: {e}")
            print(traceback.format_exc())
            self.data_cache = None
            self.db_path = None

    # ------------------------------------------------------------------
    # SQLite query helpers
    # ------------------------------------------------------------------

    def _get_db_conn(self) -> sqlite3.Connection:
        """Return a new SQLite connection. Caller must close it."""
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _fetch_filtered(
        self,
        states=None,
        categories=None,
        subcategories=None,
        brands=None,
        sizes=None,
        outlet_classifications=None,
    ) -> pd.DataFrame:
        """Query SQLite and return only the rows matching the given filters.

        This is the memory-efficient replacement for
        ``self.data_cache[boolean_mask]``.  Only the matching rows are
        loaded into RAM.
        """
        if not self.db_path:
            return pd.DataFrame()

        clauses: list[str] = []
        params: list = []

        def _add(col, values):
            if values:
                placeholders = ','.join(['?'] * len(values))
                clauses.append(f"{col} IN ({placeholders})")
                params.extend(values)

        _add("Final_State", states or [])
        _add("Category", categories or [])
        _add("Subcategory", subcategories or [])
        _add("Brand", brands or [])
        _add("Sizes", sizes or [])

        where = " AND ".join(clauses) if clauses else "1=1"

        conn = self._get_db_conn()
        df = pd.read_sql_query(
            f"SELECT * FROM transactions WHERE {where}", conn, params=params
        )
        conn.close()

        # Restore proper dtypes
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        for c in [
            "Quantity", "MRP", "Net_Amt", "SalesValue_atBasicRate",
            "TotalDiscount", "Scheme_Discount", "Staggered_qps",
            "Basic_Rate_Per_PC_without_GST", "Basic_Rate_Per_PC",
            "Selling_Rate_Per_PC_without_GST_CLP",
        ]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce').astype('float32')

        # Outlet classification filter needs Python-side normalization
        if outlet_classifications:
            normalized_targets = set(
                self._normalize_step2_outlet_classifications(list(outlet_classifications))
            )
            if normalized_targets and 'Final_Outlet_Classification' in df.columns:
                class_groups = self._to_step2_outlet_group_series(
                    df['Final_Outlet_Classification']
                )
                df = df[class_groups.isin(normalized_targets)]

        return df

    def _build_filter_clause(self, states=None, categories=None, subcategories=None,
                             brands=None, sizes=None):
        """Build a SQL WHERE clause and params list from filter lists."""
        clauses: list[str] = []
        params: list = []

        def _add(col, values):
            if values:
                placeholders = ','.join(['?'] * len(values))
                clauses.append(f"{col} IN ({placeholders})")
                params.extend(values)

        _add("Final_State", states or [])
        _add("Category", categories or [])
        _add("Subcategory", subcategories or [])
        _add("Brand", brands or [])
        _add("Sizes", sizes or [])

        where = " AND ".join(clauses) if clauses else "1=1"
        return where, params

    def _fetch_rfm_aggregated(self, states=None, categories=None, subcategories=None,
                              brands=None, sizes=None):
        """Return outlet-level RFM metrics computed entirely in SQL.

        Replaces two expensive pandas groupbys on millions of rows.
        Returns a DataFrame with one row per (Outlet_ID, Final_State) and
        columns: Outlet_ID, Final_State, first_order, last_order,
                 unique_order_days, orders_count, AOV, Total_Net_Amt, max_date.
        """
        if not self.db_path:
            return pd.DataFrame(), None

        where, params = self._build_filter_clause(states, categories, subcategories, brands, sizes)

        sql = f"""
        WITH bill_level AS (
            SELECT
                Outlet_ID, Date, Bill_No, Final_State,
                SUM(Net_Amt) AS bill_net_amt
            FROM transactions
            WHERE {where}
            GROUP BY Outlet_ID, Date, Bill_No, Final_State
        )
        SELECT
            Outlet_ID,
            Final_State,
            MIN(Date)            AS first_order,
            MAX(Date)            AS last_order,
            COUNT(DISTINCT Date) AS unique_order_days,
            COUNT(Bill_No)       AS orders_count,
            AVG(bill_net_amt)    AS AOV,
            SUM(bill_net_amt)    AS Total_Net_Amt
        FROM bill_level
        GROUP BY Outlet_ID, Final_State
        """

        conn = self._get_db_conn()
        rfm_agg = pd.read_sql_query(sql, conn, params=params)

        max_date_row = conn.execute(
            f"SELECT MAX(Date) FROM transactions WHERE {where}", params
        ).fetchone()
        conn.close()

        max_date = pd.to_datetime(max_date_row[0]) if max_date_row and max_date_row[0] else None
        rfm_agg['first_order'] = pd.to_datetime(rfm_agg['first_order'], errors='coerce')
        rfm_agg['last_order'] = pd.to_datetime(rfm_agg['last_order'], errors='coerce')

        return rfm_agg, max_date

    def _fetch_outlet_sales_value(self, states=None, categories=None, subcategories=None,
                                  brands=None, sizes=None):
        """Return per-outlet SalesValue_atBasicRate totals from SQL."""
        if not self.db_path:
            return {}, 0.0

        where, params = self._build_filter_clause(states, categories, subcategories, brands, sizes)

        sql = f"""
        SELECT Outlet_ID, SUM(SalesValue_atBasicRate) AS sales_value
        FROM transactions
        WHERE {where}
        GROUP BY Outlet_ID
        """
        conn = self._get_db_conn()
        df_sv = pd.read_sql_query(sql, conn, params=params)
        conn.close()

        outlet_sales = dict(zip(df_sv['Outlet_ID'].astype(str), df_sv['sales_value'].fillna(0)))
        total_sales = float(df_sv['sales_value'].sum())
        return outlet_sales, total_sales

    def _fetch_filtered_by_outlets(self, outlet_ids: list, states=None, categories=None,
                                   subcategories=None, brands=None, sizes=None) -> pd.DataFrame:
        """Load transactions for a specific set of outlet IDs only.

        Chunks the IN clause to stay within SQLite's 999-variable limit.
        If outlet_ids is very large, falls back to full filter query.
        """
        if not self.db_path or not outlet_ids:
            return pd.DataFrame()

        # If covering most outlets, just use the regular filter (no outlet restriction)
        if len(outlet_ids) > 5000:
            return self._fetch_filtered(states=states, categories=categories,
                                        subcategories=subcategories, brands=brands, sizes=sizes)

        where, params = self._build_filter_clause(states, categories, subcategories, brands, sizes)

        # Chunk outlet_ids to stay under SQLite's 999-variable limit
        CHUNK = 900
        chunks = [outlet_ids[i:i+CHUNK] for i in range(0, len(outlet_ids), CHUNK)]
        frames = []
        conn = self._get_db_conn()
        for chunk in chunks:
            placeholders = ','.join(['?'] * len(chunk))
            outlet_clause = f"Outlet_ID IN ({placeholders})"
            full_where = f"({where}) AND {outlet_clause}" if where != "1=1" else outlet_clause
            all_params = params + list(chunk)
            frames.append(pd.read_sql_query(
                f"SELECT * FROM transactions WHERE {full_where}", conn, params=all_params
            ))
        conn.close()
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        for c in ["Quantity", "MRP", "Net_Amt", "SalesValue_atBasicRate",
                  "TotalDiscount", "Scheme_Discount", "Staggered_qps",
                  "Basic_Rate_Per_PC_without_GST", "Basic_Rate_Per_PC",
                  "Selling_Rate_Per_PC_without_GST_CLP"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce').astype('float32')
        return df

    def _fetch_outlet_classifications(self, outlet_ids: list) -> pd.DataFrame:
        """Return one row per outlet with its Final_Outlet_Classification."""
        if not self.db_path or not outlet_ids:
            return pd.DataFrame(columns=['Outlet_ID', 'Final_Outlet_Classification'])
        CHUNK = 900
        chunks = [outlet_ids[i:i+CHUNK] for i in range(0, len(outlet_ids), CHUNK)]
        frames = []
        conn = self._get_db_conn()
        for chunk in chunks:
            placeholders = ','.join(['?'] * len(chunk))
            frames.append(pd.read_sql_query(
                f"SELECT Outlet_ID, Final_Outlet_Classification FROM transactions "
                f"WHERE Outlet_ID IN ({placeholders}) GROUP BY Outlet_ID",
                conn, params=list(chunk)
            ))
        conn.close()
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
            columns=['Outlet_ID', 'Final_Outlet_Classification']
        )
