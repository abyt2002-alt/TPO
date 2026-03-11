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
                dfs.append(df)
            
            combined_df = pd.concat(dfs, ignore_index=True)
            combined_df['Date'] = pd.to_datetime(combined_df['Date'])
            
            self.data_cache = combined_df
            print(f"Loaded {len(combined_df):,} rows from {len(parquet_files)} files")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            self.data_cache = None
