"""Run-state persistence layer for analysis workflow.\n\nThis module owns SQLite-backed run/session state so step logic remains stateless\nwith respect to UI flow and checkpointing.\n"""

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


class StateStoreMixin:

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
                "subcategories": ["STX INSTA SHAMPOO", "STREAX INSTA SHAMPOO"],
                "brands": [],
                "sizes": ["12-ML", "18-ML"],
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
                "slab_definition_mode": "data",
                "defined_slab_level": "monthly_outlet",
                "defined_slab_count": 5,
                "defined_slab_thresholds": [8, 32, 576, 960],
                "defined_slab_profiles": {
                    "12-ML": {
                        "defined_slab_count": 3,
                        "defined_slab_thresholds": [8, 144],
                    },
                    "18-ML": {
                        "defined_slab_count": 5,
                        "defined_slab_thresholds": [8, 32, 576, 960],
                    },
                },
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
                "step5_filters": {
                    "states": [],
                    "categories": [],
                    "subcategories": [],
                    "brands": [],
                    "sizes": [],
                    "outlet_classifications": [],
                    "product_codes": [],
                },
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
