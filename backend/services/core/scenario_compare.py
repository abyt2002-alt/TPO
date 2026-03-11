"""Scenario upload parsing and multi-scenario planner comparison.\n\nThis module isolates uploaded scenario validation/parsing and orchestration\nfor scenario comparison outputs.\n"""

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


class ScenarioCompareMixin:

    def _parse_planner_scenarios_upload(self, filename: str, file_bytes: bytes) -> Dict[str, List[float]]:
        if not file_bytes:
            raise ValueError("Uploaded file is empty.")

        suffix = Path(str(filename or "")).suffix.lower()
        try:
            if suffix == ".csv":
                df = pd.read_csv(BytesIO(file_bytes))
            elif suffix in {".xlsx", ".xls"}:
                df = pd.read_excel(BytesIO(file_bytes))
            else:
                # Fallback: try CSV first, then Excel.
                try:
                    df = pd.read_csv(BytesIO(file_bytes))
                except Exception:
                    df = pd.read_excel(BytesIO(file_bytes))
        except Exception as exc:
            raise ValueError(f"Could not parse uploaded file: {str(exc)}")

        if df is None or df.empty:
            raise ValueError("Uploaded file has no rows.")

        df = df.dropna(how="all").copy()
        if df.empty:
            raise ValueError("Uploaded file has no usable rows.")

        df.columns = [str(c).strip() for c in df.columns]
        if len(df.columns) < 2:
            raise ValueError("File must have 'Month' column and at least one scenario column.")

        month_col = df.columns[0]
        scenario_cols = [c for c in df.columns[1:] if str(c).strip()]
        if not scenario_cols:
            raise ValueError("No scenario columns found. Add columns like Scenario 1, Scenario 2, ...")

        month_name_to_num = {
            "jan": 1, "january": 1,
            "feb": 2, "february": 2,
            "mar": 3, "march": 3,
            "apr": 4, "april": 4,
            "may": 5,
            "jun": 6, "june": 6,
            "jul": 7, "july": 7,
            "aug": 8, "august": 8,
            "sep": 9, "sept": 9, "september": 9,
            "oct": 10, "october": 10,
            "nov": 11, "november": 11,
            "dec": 12, "december": 12,
        }

        def _month_to_num(value) -> Optional[int]:
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return None
            if isinstance(value, (datetime, date, pd.Timestamp)):
                return int(pd.to_datetime(value).month)
            text = str(value).strip()
            if not text:
                return None
            if re.fullmatch(r"\d{1,2}", text):
                m = int(text)
                return m if 1 <= m <= 12 else None
            dt = pd.to_datetime(text, errors="coerce")
            if not pd.isna(dt):
                return int(dt.month)
            token = re.sub(r"[^a-z]", "", text.lower())
            if token in month_name_to_num:
                return month_name_to_num[token]
            token3 = token[:3]
            return month_name_to_num.get(token3)

        month_to_row: Dict[int, int] = {}
        for idx, value in enumerate(df[month_col].tolist()):
            m = _month_to_num(value)
            if m is None:
                continue
            month_to_row[m] = idx

        fy_months = [4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3]
        month_label = {
            1: "January", 2: "February", 3: "March", 4: "April",
            5: "May", 6: "June", 7: "July", 8: "August",
            9: "September", 10: "October", 11: "November", 12: "December",
        }

        missing_months = [month_label[m] for m in fy_months if m not in month_to_row]
        if missing_months:
            raise ValueError(f"Missing month rows in upload: {', '.join(missing_months)}")

        def _parse_pct(value) -> Optional[float]:
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return None
            if isinstance(value, (int, float, np.number)):
                v = float(value)
            else:
                text = str(value).strip()
                if not text:
                    return None
                text = text.replace("%", "").strip()
                try:
                    v = float(text)
                except Exception:
                    return None
            if not np.isfinite(v):
                return None
            return float(v)

        scenarios: Dict[str, List[float]] = {}
        used_names: Dict[str, int] = {}
        for idx, col in enumerate(scenario_cols):
            raw_name = str(col).strip() or f"Scenario {idx + 1}"
            count = used_names.get(raw_name, 0) + 1
            used_names[raw_name] = count
            scenario_name = raw_name if count == 1 else f"{raw_name} ({count})"

            values: List[float] = []
            for m in fy_months:
                row_idx = month_to_row[m]
                pct = _parse_pct(df.iloc[row_idx][col])
                if pct is None:
                    raise ValueError(f"Invalid discount value for '{scenario_name}' in {month_label[m]}.")
                if pct < 0 or pct > 60:
                    raise ValueError(
                        f"Discount value out of range for '{scenario_name}' in {month_label[m]}: {pct}. Use 0 to 60."
                    )
                values.append(float(pct))
            scenarios[scenario_name] = values

        if not scenarios:
            raise ValueError("No valid scenario columns found in upload.")
        return scenarios


    async def compare_planner_scenarios_from_upload(
        self,
        request: PlannerRequest,
        filename: str,
        file_bytes: bytes,
    ) -> PlannerScenarioComparisonResponse:
        try:
            scenarios = self._parse_planner_scenarios_upload(filename=filename, file_bytes=file_bytes)
        except Exception as exc:
            return PlannerScenarioComparisonResponse(
                success=False,
                message=f"Scenario upload parse failed: {str(exc)}",
                slab=str(request.slab or ""),
                months=[],
                default_structural_discounts=[],
                default_metrics={},
                scenarios=[],
            )

        base_payload = request.model_dump(exclude_none=False)
        base_payload["planned_structural_discounts"] = None
        base_payload["planned_base_prices"] = None
        base_payload["disable_ai_insights"] = True
        base_request = PlannerRequest(**base_payload)
        base_result = await self.calculate_12_month_planner(base_request)
        if not base_result.success:
            return PlannerScenarioComparisonResponse(
                success=False,
                message=base_result.message or "Failed to run default planner baseline",
                slab=str(request.slab or ""),
                months=[],
                default_structural_discounts=[],
                default_metrics={},
                scenarios=[],
            )

        selected_slab = str(
            base_result.slab
            or request.slab
            or (request.slabs[0] if request.slabs else "")
        )
        default_base_prices = list(base_result.planned_base_prices or [])
        rows: List[PlannerScenarioComparisonRow] = []

        for scenario_name, planned_struct in scenarios.items():
            payload = request.model_dump(exclude_none=False)
            payload["slab"] = selected_slab
            payload["slabs"] = [selected_slab] if selected_slab else payload.get("slabs")
            payload["planned_structural_discounts"] = [float(x) for x in planned_struct]
            payload["planned_base_prices"] = default_base_prices
            payload["disable_ai_insights"] = True
            scenario_request = PlannerRequest(**payload)
            scenario_result = await self.calculate_12_month_planner(scenario_request)

            if not scenario_result.success:
                rows.append(
                    PlannerScenarioComparisonRow(
                        scenario=scenario_name,
                        success=False,
                        message=scenario_result.message or "Planner failed for scenario",
                        planned_structural_discounts=[float(x) for x in planned_struct],
                    )
                )
                continue

            metrics = scenario_result.metrics or {}
            roi_default_x = float(metrics.get("roi_default_x", 0.0))
            roi_planned_x = float(metrics.get("roi_planned_x", 0.0))
            roi_revenue_x = float(metrics.get("roi_revenue_x", 0.0))
            gross_margin_roi_default_x = float(metrics.get("profit_roi_default_x", 0.0))
            gross_margin_roi_planned_x = float(metrics.get("profit_roi_planned_x", 0.0))
            gross_margin_roi_revenue_x = float(metrics.get("profit_roi_revenue_x", 0.0))
            investment_change = float(metrics.get("investment_change", 0.0))

            # Scenario comparison expectation:
            # if scenario has no internal step-up episodes (flat path), use plan-vs-default ROI
            # so ROI is still visible when planned path differs from default baseline.
            if abs(roi_planned_x) <= 1e-12 and abs(roi_revenue_x) > 1e-12 and abs(investment_change) > 1e-12:
                roi_planned_x = roi_revenue_x
            if (
                abs(gross_margin_roi_planned_x) <= 1e-12
                and abs(gross_margin_roi_revenue_x) > 1e-12
                and abs(investment_change) > 1e-12
            ):
                gross_margin_roi_planned_x = gross_margin_roi_revenue_x

            roi_abs_change_x = (
                float(roi_planned_x - roi_default_x)
                if np.isfinite(roi_default_x) and np.isfinite(roi_planned_x)
                else float(metrics.get("roi_abs_change_x", 0.0))
            )
            gross_margin_roi_abs_change_x = (
                float(gross_margin_roi_planned_x - gross_margin_roi_default_x)
                if np.isfinite(gross_margin_roi_default_x) and np.isfinite(gross_margin_roi_planned_x)
                else float(metrics.get("profit_roi_abs_change_x", 0.0))
            )
            rows.append(
                PlannerScenarioComparisonRow(
                    scenario=scenario_name,
                    success=True,
                    message=None,
                    planned_structural_discounts=[float(x) for x in planned_struct],
                    volume_change_pct=float(metrics.get("volume_change_pct", 0.0)),
                    revenue_change_pct=float(metrics.get("revenue_change_pct", 0.0)),
                    profit_change_pct=float(metrics.get("profit_change_pct", 0.0)),
                    promo_change_pct=float(metrics.get("promo_change_pct", 0.0)),
                    investment_change_pct=float(metrics.get("investment_change_pct", 0.0)),
                    roi_default_x=roi_default_x,
                    roi_planned_x=roi_planned_x,
                    roi_abs_change_x=roi_abs_change_x,
                    gross_margin_roi_default_x=gross_margin_roi_default_x,
                    gross_margin_roi_planned_x=gross_margin_roi_planned_x,
                    gross_margin_roi_abs_change_x=gross_margin_roi_abs_change_x,
                )
            )

        success_count = len([r for r in rows if r.success])
        return PlannerScenarioComparisonResponse(
            success=success_count > 0,
            message=f"Processed {len(rows)} scenario(s); successful: {success_count}.",
            slab=selected_slab,
            months=list(base_result.months or []),
            default_structural_discounts=[float(x) for x in (base_result.default_structural_discounts or [])],
            default_metrics={k: float(v) for k, v in (base_result.metrics or {}).items() if isinstance(v, (int, float, np.number))},
            scenarios=rows,
        )
