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
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS global_reports (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        report_key TEXT UNIQUE NOT NULL,
                        run_id TEXT,
                        source_step TEXT NOT NULL,
                        report_name TEXT NOT NULL,
                        reference_mode TEXT,
                        metadata_json TEXT,
                        payload_json TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_global_reports_updated_at
                    ON global_reports(updated_at DESC)
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


    def save_global_report(
        self,
        step: str,
        name: str,
        run_id: Optional[str] = None,
        reference_mode: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        report_key = str(uuid.uuid4())
        now_iso = self._utc_now_iso()
        step_value = str(step or "").strip().lower()
        if step_value not in {"step4", "step5"}:
            raise ValueError("Unsupported report step. Allowed: step4, step5.")
        report_name = str(name or "").strip()
        if not report_name:
            raise ValueError("Report name is required.")

        report_metadata = metadata if isinstance(metadata, dict) else {}
        report_payload = payload if isinstance(payload, dict) else {}

        with sqlite3.connect(self.state_db_path) as conn:
            conn.execute(
                """
                INSERT INTO global_reports
                (report_key, run_id, source_step, report_name, reference_mode, metadata_json, payload_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    report_key,
                    (str(run_id).strip() if run_id else None),
                    step_value,
                    report_name,
                    (str(reference_mode).strip() if reference_mode else None),
                    self._safe_json_dumps(report_metadata),
                    self._safe_json_dumps(report_payload),
                    now_iso,
                    now_iso,
                ),
            )
            conn.commit()

        return {
            "report_key": report_key,
            "run_id": (str(run_id).strip() if run_id else None),
            "step": step_value,
            "name": report_name,
            "reference_mode": (str(reference_mode).strip() if reference_mode else None),
            "metadata": report_metadata,
            "created_at": now_iso,
            "updated_at": now_iso,
        }


    def list_global_reports(self, limit: int = 200) -> List[Dict[str, Any]]:
        safe_limit = max(1, min(2000, int(limit or 200)))
        with sqlite3.connect(self.state_db_path) as conn:
            rows = conn.execute(
                """
                SELECT report_key, run_id, source_step, report_name, reference_mode, metadata_json, created_at, updated_at
                FROM global_reports
                ORDER BY updated_at DESC, id DESC
                LIMIT ?
                """,
                (safe_limit,),
            ).fetchall()

        out: List[Dict[str, Any]] = []
        for row in rows:
            out.append({
                "report_key": row[0],
                "run_id": row[1],
                "step": row[2],
                "name": row[3],
                "reference_mode": row[4],
                "metadata": self._safe_json_loads(row[5], {}),
                "created_at": row[6],
                "updated_at": row[7],
            })
        return out


    def get_global_report(self, report_key: str) -> Optional[Dict[str, Any]]:
        key = str(report_key or "").strip()
        if not key:
            return None
        with sqlite3.connect(self.state_db_path) as conn:
            row = conn.execute(
                """
                SELECT report_key, run_id, source_step, report_name, reference_mode, metadata_json, payload_json, created_at, updated_at
                FROM global_reports
                WHERE report_key = ?
                """,
                (key,),
            ).fetchone()
        if row is None:
            return None
        return {
            "report_key": row[0],
            "run_id": row[1],
            "step": row[2],
            "name": row[3],
            "reference_mode": row[4],
            "metadata": self._safe_json_loads(row[5], {}),
            "payload": self._safe_json_loads(row[6], {}),
            "created_at": row[7],
            "updated_at": row[8],
        }


    def _sanitize_sheet_name(self, raw: str, fallback: str) -> str:
        text = str(raw or "").strip()
        if not text:
            text = fallback
        text = re.sub(r"[\[\]\:\*\?/\\]", "_", text)
        if not text:
            text = fallback
        return text[:31]


    def export_global_reports_excel(self, report_keys: Optional[List[str]] = None) -> bytes:
        selected_keys = [str(v).strip() for v in (report_keys or []) if str(v).strip()]
        reports: List[Dict[str, Any]]
        if selected_keys:
            reports = []
            for key in selected_keys:
                found = self.get_global_report(key)
                if found is not None:
                    reports.append(found)
        else:
            latest = self.list_global_reports(limit=1000)
            reports = [self.get_global_report(row.get("report_key")) for row in latest]
            reports = [r for r in reports if r is not None]

        if not reports:
            raise ValueError("No saved reports found for export.")

        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            used_sheet_names = set()
            for idx, report in enumerate(reports, start=1):
                base_sheet = self._sanitize_sheet_name(
                    report.get("name"),
                    fallback=f"Report_{idx}",
                )
                sheet_name = base_sheet
                suffix = 1
                while sheet_name in used_sheet_names:
                    suffix += 1
                    trunc = max(1, 31 - len(str(suffix)) - 1)
                    sheet_name = f"{base_sheet[:trunc]}_{suffix}"
                used_sheet_names.add(sheet_name)

                metadata = report.get("metadata") if isinstance(report.get("metadata"), dict) else {}
                payload = report.get("payload") if isinstance(report.get("payload"), dict) else {}
                records = payload.get("records") if isinstance(payload.get("records"), list) else []

                meta_rows = [
                    {"Meta Field": "Report Name", "Meta Value": report.get("name", "")},
                    {"Meta Field": "Step", "Meta Value": report.get("step", "")},
                    {"Meta Field": "Run ID", "Meta Value": report.get("run_id", "")},
                    {"Meta Field": "Reference Mode", "Meta Value": report.get("reference_mode", "")},
                    {"Meta Field": "Saved At (UTC)", "Meta Value": report.get("created_at", "")},
                ]
                for k in sorted(metadata.keys()):
                    meta_rows.append({"Meta Field": f"Meta: {k}", "Meta Value": metadata.get(k)})
                pd.DataFrame(meta_rows).to_excel(writer, sheet_name=sheet_name, index=False, startrow=0)

                start_row = len(meta_rows) + 2
                if records:
                    pd.DataFrame(records).to_excel(writer, sheet_name=sheet_name, index=False, startrow=start_row)
                else:
                    pd.DataFrame([{"Info": "No scenario rows saved for this report."}]).to_excel(
                        writer,
                        sheet_name=sheet_name,
                        index=False,
                        startrow=start_row,
                    )

        buffer.seek(0)
        return buffer.getvalue()
