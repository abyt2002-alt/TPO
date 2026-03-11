"""Step 4 cross-size planner and planner AI helpers.\n\nThis module owns cross-size coupling, elasticity application, and the existing\nplanner computation/insights paths used by Step 4/Step 6 flows.\n"""

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
    AIScenarioGenerateRequest, AIScenarioGenerateResponse, AIScenarioRow,
    BaselineForecastRequest, BaselineForecastResponse, BaselineForecastPoint,
    EDARequest, EDAResponse, EDAProductOption, EDAProductContribution,
    EDAContributionRow, EDAOptionsResponse
)


class Step4CrossSizePlannerMixin:
    def _extract_first_json_object(self, text: str) -> Optional[Dict[str, Any]]:
        raw = str(text or "").strip()
        if not raw:
            return None
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)```", raw, flags=re.IGNORECASE)
        for block in fenced:
            try:
                parsed = json.loads(block.strip())
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                continue

        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            candidate = raw[start:end + 1]
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                return None
        return None

    def _clamp_discount_5_30(self, value: Any, default: float) -> float:
        try:
            v = float(value)
            if not np.isfinite(v):
                raise ValueError
        except Exception:
            v = float(default)
        return float(min(30.0, max(5.0, v)))

    def _enforce_size_month_ladder(self, slab_values: Dict[str, float], slab_order: List[str]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        floor = 5.0
        for slab in slab_order:
            current = self._clamp_discount_5_30(slab_values.get(slab), floor)
            next_value = max(floor, current)
            out[slab] = float(round(next_value, 2))
            floor = out[slab]
        return out

    def _build_step5_ai_prompt(
        self,
        scenario_count: int,
        goal: str,
        user_prompt: str,
        periods: List[str],
        defaults_matrix: Dict[str, Dict[str, List[float]]],
    ) -> str:
        period_defaults: Dict[str, Dict[str, Dict[str, float]]] = {}
        slab_order_map: Dict[str, List[str]] = {}
        for size_key in ["12-ML", "18-ML"]:
            slab_keys = sorted(list((defaults_matrix.get(size_key) or {}).keys()), key=self._slab_sort_key)
            slab_order_map[size_key] = slab_keys

        for month_idx, period in enumerate(periods):
            period_defaults[str(period)] = {}
            for size_key in ["12-ML", "18-ML"]:
                period_defaults[str(period)][size_key] = {}
                for slab_key in slab_order_map.get(size_key, []):
                    series = defaults_matrix.get(size_key, {}).get(slab_key, [])
                    default_val = float(series[month_idx]) if month_idx < len(series) else 10.0
                    period_defaults[str(period)][size_key][slab_key] = round(default_val, 2)

        payload = {
            "scenario_count": int(scenario_count),
            "goal": str(goal or "").strip() or "maximize_revenue",
            "user_prompt": str(user_prompt or "").strip(),
            "periods": [str(p) for p in periods],
            "slab_order": slab_order_map,
            "default_discounts_by_period": period_defaults,
            "hard_constraints": [
                "Discount range must be between 5 and 30.",
                "Within each month and size, slab ladder must be non-decreasing: slab1 <= slab2 <= ...",
                "Return ONLY valid JSON object.",
            ],
            "output_schema": {
                "scenarios": [
                    {
                        "name": "AI Scenario 1",
                        "scenario_discounts_by_period": {
                            "YYYY-MM": {
                                "12-ML": {"slab1": 14.0},
                                "18-ML": {"slab1": 11.5},
                            }
                        },
                    }
                ]
            },
        }
        payload_text = json.dumps(payload, ensure_ascii=True)
        return (
            "You are generating Step-5 scenario discount plans for a trade-promo planner.\n"
            "Business objective is provided in `goal`. User preference is provided in `user_prompt`.\n"
            "Generate exactly `scenario_count` scenarios.\n"
            "Each scenario must include month-wise slab discounts for both sizes: 12-ML and 18-ML.\n"
            "Hard rules:\n"
            "1) Discounts must stay in [5, 30].\n"
            "2) Within one month and one size, ladder order must be non-decreasing by slab index.\n"
            "3) Keep monthly changes realistic; avoid extreme random jumps unless prompt explicitly asks.\n"
            "4) Use clear business names for each scenario based on objective.\n"
            "5) Return strict JSON only, no markdown, no explanation text.\n"
            "JSON shape must be exactly:\n"
            "{ \"scenarios\": [ { \"name\": \"...\", \"scenario_discounts_by_period\": { \"YYYY-MM\": { \"12-ML\": {\"slab1\": 0}, \"18-ML\": {\"slab1\": 0} } } } ] }\n\n"
            f"{payload_text}"
        )

    def _build_step5_local_ai_scenarios(
        self,
        scenario_count: int,
        goal: str,
        user_prompt: str,
        periods: List[str],
        defaults_matrix: Dict[str, Dict[str, List[float]]],
        slab_order_by_size: Dict[str, List[str]],
    ) -> List[Dict[str, Any]]:
        """Deterministic fallback when Gemini is unavailable (network/key/proxy issues)."""
        goal_key = str(goal or "").strip().lower()
        prompt_lc = str(user_prompt or "").strip().lower()

        if goal_key == "maximize_volume":
            base_moves = [2.0, 3.0, 1.0, 4.0, 0.0, -1.0]
        elif goal_key == "maximize_profit":
            base_moves = [-2.0, -1.0, 0.0, 1.0, -3.0, 2.0]
        elif goal_key == "balanced_growth":
            base_moves = [0.0, 1.0, -1.0, 2.0, -2.0, 0.5]
        else:  # maximize_revenue default
            base_moves = [1.0, 2.0, 0.0, 3.0, -1.0, 0.5]

        prompt_bias = 0.0
        if any(tok in prompt_lc for tok in ["aggressive", "deep", "high discount", "push volume"]):
            prompt_bias = 1.0
        if any(tok in prompt_lc for tok in ["conservative", "shallow", "margin protection", "protect profit"]):
            prompt_bias = -1.0

        month_count = max(1, len(periods))
        scenarios: List[Dict[str, Any]] = []
        for idx in range(max(1, int(scenario_count))):
            move = float(base_moves[idx % len(base_moves)] + prompt_bias)
            pattern_kind = idx % 4
            if pattern_kind == 0:
                month_moves = [move for _ in range(month_count)]
            elif pattern_kind == 1:
                month_moves = [move + 1.0, move, move - 1.0][:month_count]
                while len(month_moves) < month_count:
                    month_moves.append(move)
            elif pattern_kind == 2:
                month_moves = [move - 1.0, move, move + 1.0][:month_count]
                while len(month_moves) < month_count:
                    month_moves.append(move)
            else:
                month_moves = [move, move - 0.5, move][:month_count]
                while len(month_moves) < month_count:
                    month_moves.append(move)

            by_period: Dict[str, Dict[str, Dict[str, float]]] = {}
            for month_idx, period_key in enumerate(periods):
                by_period[str(period_key)] = {}
                for size_key in ["12-ML", "18-ML"]:
                    slab_order = slab_order_by_size.get(size_key, [])
                    seed: Dict[str, float] = {}
                    for slab_key in slab_order:
                        default_series = defaults_matrix.get(size_key, {}).get(slab_key, [])
                        default_val = float(default_series[month_idx]) if month_idx < len(default_series) else 10.0
                        seed[slab_key] = self._clamp_discount_5_30(default_val + month_moves[month_idx], default_val)
                    by_period[str(period_key)][size_key] = self._enforce_size_month_ladder(seed, slab_order)

            scenarios.append(
                {
                    "name": f"AI Scenario {idx + 1}",
                    "scenario_discounts_by_period": by_period,
                }
            )
        return scenarios

    async def generate_ai_scenarios(self, request: AIScenarioGenerateRequest) -> AIScenarioGenerateResponse:
        try:
            request_payload = (
                request.model_dump(exclude_none=True)
                if hasattr(request, "model_dump")
                else request.dict(exclude_none=True)
            )
            for key in ["scenario_count", "goal", "prompt"]:
                request_payload.pop(key, None)

            base_request = CrossSizePlannerRequest(**request_payload)
            planner_base = await self.calculate_cross_size_planner(base_request)
            if not planner_base.success:
                return AIScenarioGenerateResponse(
                    success=False,
                    message=planner_base.message or "Failed to initialize planner base for AI scenarios.",
                    scenarios=[],
                )

            periods = [str(p) for p in (planner_base.periods or [])]
            defaults_matrix = planner_base.defaults_matrix or {}
            if not periods or not defaults_matrix:
                return AIScenarioGenerateResponse(
                    success=False,
                    message="Planner base is missing periods/default discount matrix.",
                    scenarios=[],
                )

            slab_order_by_size: Dict[str, List[str]] = {
                size_key: sorted(list((defaults_matrix.get(size_key) or {}).keys()), key=self._slab_sort_key)
                for size_key in ["12-ML", "18-ML"]
            }
            scenario_count = int(getattr(request, "scenario_count", 5) or 5)
            goal = str(getattr(request, "goal", "") or "").strip()
            user_prompt = str(getattr(request, "prompt", "") or "").strip()

            api_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
            if not api_key:
                return AIScenarioGenerateResponse(
                    success=False,
                    message="AI generation disabled. Set GEMINI_API_KEY on backend.",
                    scenarios=[],
                )

            model_name = (os.getenv("GEMINI_MODEL") or "gemini-2.5-flash").strip()
            endpoint = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{model_name}:generateContent?key={api_key}"
            )
            prompt_text = self._build_step5_ai_prompt(
                scenario_count=scenario_count,
                goal=goal,
                user_prompt=user_prompt,
                periods=periods,
                defaults_matrix=defaults_matrix,
            )
            request_body = {
                "contents": [{"role": "user", "parts": [{"text": prompt_text}]}],
                "generationConfig": {
                    "temperature": 0.3,
                    "topP": 0.9,
                    "maxOutputTokens": 5000,
                    "responseMimeType": "application/json",
                },
            }

            req_obj = urllib.request.Request(
                endpoint,
                data=json.dumps(request_body).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            raw_scenarios: List[Any] = []
            ai_error: str = ""
            try:
                with urllib.request.urlopen(req_obj, timeout=40) as resp:
                    body = resp.read().decode("utf-8")
                payload = json.loads(body)
                raw_parts: List[str] = []
                for candidate in (payload.get("candidates") or []):
                    content = candidate.get("content") or {}
                    for part in (content.get("parts") or []):
                        txt = part.get("text")
                        if isinstance(txt, str) and txt.strip():
                            raw_parts.append(txt.strip())

                parsed = self._extract_first_json_object("\n".join(raw_parts))
                if isinstance(parsed, dict):
                    raw_scenarios = parsed.get("scenarios") or []
                if not isinstance(raw_scenarios, list):
                    raw_scenarios = []
                if not raw_scenarios:
                    ai_error = "Empty/invalid Gemini response"
            except urllib.error.HTTPError as err:
                detail = ""
                try:
                    detail = err.read().decode("utf-8", errors="ignore")
                except Exception:
                    detail = str(err)
                ai_error = f"HTTP {err.code}: {detail[:120]}"
            except Exception as err:
                ai_error = str(err)

            if ai_error:
                raw_scenarios = self._build_step5_local_ai_scenarios(
                    scenario_count=scenario_count,
                    goal=goal,
                    user_prompt=user_prompt,
                    periods=periods,
                    defaults_matrix=defaults_matrix,
                    slab_order_by_size=slab_order_by_size,
                )

            def _default_for(period_key: str, size_key: str, slab_key: str) -> float:
                month_idx = periods.index(period_key)
                series = defaults_matrix.get(size_key, {}).get(slab_key, [])
                if month_idx < len(series):
                    return float(series[month_idx])
                return 10.0

            scenarios_out: List[AIScenarioRow] = []
            for idx in range(scenario_count):
                source = raw_scenarios[idx] if idx < len(raw_scenarios) and isinstance(raw_scenarios[idx], dict) else {}
                name = str(source.get("name") or "").strip() or f"AI Scenario {idx + 1}"
                by_period_raw = source.get("scenario_discounts_by_period") or {}
                scenario_map: Dict[str, Dict[str, Dict[str, float]]] = {}

                for period_key in periods:
                    scenario_map[period_key] = {}
                    period_obj = by_period_raw.get(period_key, {}) if isinstance(by_period_raw, dict) else {}
                    for size_key in ["12-ML", "18-ML"]:
                        slab_values_in = period_obj.get(size_key, {}) if isinstance(period_obj, dict) else {}
                        slab_values_seed: Dict[str, float] = {}
                        for slab_key in slab_order_by_size.get(size_key, []):
                            default_value = _default_for(period_key, size_key, slab_key)
                            raw_value = None
                            if isinstance(slab_values_in, dict):
                                raw_value = slab_values_in.get(slab_key)
                            slab_values_seed[slab_key] = self._clamp_discount_5_30(raw_value, default_value)
                        scenario_map[period_key][size_key] = self._enforce_size_month_ladder(
                            slab_values_seed,
                            slab_order_by_size.get(size_key, []),
                        )

                scenarios_out.append(
                    AIScenarioRow(
                        name=name if name else f"AI Scenario {idx + 1}",
                        scenario_discounts_by_period=scenario_map,
                    )
                )

            return AIScenarioGenerateResponse(
                success=True,
                message=(
                    f"Generated {len(scenarios_out)} AI scenario(s)."
                    if not ai_error
                    else f"Generated {len(scenarios_out)} fallback AI scenario(s) because Gemini was unreachable ({ai_error})."
                ),
                scenarios=scenarios_out,
            )
        except Exception as err:
            return AIScenarioGenerateResponse(
                success=False,
                message=f"AI scenario generation failed: {str(err)}",
                scenarios=[],
            )

    def _pack_size_ml(self, size_key: str) -> float:
        """Extract numeric pack size in ml from size key (e.g. '12-ML' -> 12.0)."""
        try:
            m = re.search(r'(\d+(?:\.\d+)?)', str(size_key or ''))
            if not m:
                return 1.0
            v = float(m.group(1))
            return v if np.isfinite(v) and v > 0 else 1.0
        except Exception:
            return 1.0

    def _compute_step4_reference_3m(
        self,
        df_scope: pd.DataFrame,
        periods: List[str],
        pair_state: Dict[str, Dict[str, Any]],
        reference_mode: str = "ly_same_3m",
    ) -> Dict[str, Dict[str, float]]:
        """Build Step 4 reference totals for summary cards."""
        if df_scope is None or df_scope.empty or not periods:
            return {}

        work = df_scope.copy()
        if 'Date' not in work.columns:
            return {}

        work['Date'] = pd.to_datetime(work['Date'], errors='coerce')
        work = work.dropna(subset=['Date']).copy()
        if work.empty:
            return {}

        work['Month_Key'] = work['Date'].dt.to_period('M').astype(str)
        work['Size_Key'] = work.get('Sizes', pd.Series(index=work.index, dtype='object')).astype(str).map(self._normalize_step2_size_key)
        work['Quantity_Num'] = pd.to_numeric(work.get('Quantity', 0.0), errors='coerce').fillna(0.0)
        work['Sales_Num'] = pd.to_numeric(work.get('SalesValue_atBasicRate', 0.0), errors='coerce').fillna(0.0)
        work['Discount_Num'] = pd.to_numeric(work.get('TotalDiscount', 0.0), errors='coerce').fillna(0.0)
        work['Net_Revenue'] = work['Sales_Num'] - work['Discount_Num']

        ref_periods: List[str] = []
        mode = str(reference_mode or "ly_same_3m").strip().lower()
        if mode == "last_3m_before_projection":
            try:
                first_forecast_period = pd.Period(str(periods[0]), freq='M')
                for i in range(1, 4):
                    ref_periods.append((first_forecast_period - i).strftime('%Y-%m'))
            except Exception:
                ref_periods = []
        else:
            for period_key in periods:
                try:
                    ref_periods.append((pd.Period(str(period_key), freq='M') - 12).strftime('%Y-%m'))
                except Exception:
                    continue
        if not ref_periods:
            return {}

        ref_set = set(ref_periods)
        ref_work = work[work['Month_Key'].isin(ref_set)].copy()
        if ref_work.empty:
            return {}

        out: Dict[str, Dict[str, float]] = {}
        for size_key in ['12-ML', '18-ML']:
            size_part = ref_work[ref_work['Size_Key'].astype(str) == str(size_key)].copy()
            if size_part.empty:
                out[size_key] = {
                    'reference_qty': 0.0,
                    'reference_revenue': 0.0,
                    'reference_profit': 0.0,
                    'reference_available': 0.0,
                }
                continue

            qty = float(size_part['Quantity_Num'].sum())
            revenue = float(size_part['Net_Revenue'].sum())
            cogs = float(pair_state.get(size_key, {}).get('cogs_per_unit', 0.0))
            profit = float(revenue - (qty * cogs))
            out[size_key] = {
                'reference_qty': qty,
                'reference_revenue': revenue,
                'reference_profit': profit,
                'reference_available': 1.0,
            }

        total_qty = float(sum(out.get(size, {}).get('reference_qty', 0.0) for size in ['12-ML', '18-ML']))
        total_revenue = float(sum(out.get(size, {}).get('reference_revenue', 0.0) for size in ['12-ML', '18-ML']))
        total_profit = float(sum(out.get(size, {}).get('reference_profit', 0.0) for size in ['12-ML', '18-ML']))
        total_available = 1.0 if any(out.get(size, {}).get('reference_available', 0.0) > 0 for size in ['12-ML', '18-ML']) else 0.0
        out['TOTAL'] = {
            'reference_qty': total_qty,
            'reference_revenue': total_revenue,
            'reference_profit': total_profit,
            'reference_available': total_available,
        }
        return out

    def _build_planner_ai_prompt(
        self,
        slab: str,
        months: List[str],
        default_structural: List[float],
        planned_structural: List[float],
        default_base_prices: List[float],
        planned_base_prices: List[float],
        metrics: Dict[str, float],
        series_rows: List[Dict[str, float]],
    ) -> str:
        def _to_num(value, default=0.0):
            try:
                v = float(value)
                return v if np.isfinite(v) else float(default)
            except Exception:
                return float(default)

        month_changes = self._extract_planner_month_changes(
            months=months,
            default_structural=default_structural,
            planned_structural=planned_structural,
            default_base_prices=default_base_prices,
            planned_base_prices=planned_base_prices,
        )

        payload = {
            "slab": slab,
            "chart_data_monthly": series_rows,
            "user_changes": month_changes,
            "metrics_summary": {
                "volume_change_pct": _to_num(metrics.get("volume_change_pct")),
                "revenue_change_pct": _to_num(metrics.get("revenue_change_pct")),
                "profit_change_pct": _to_num(metrics.get("profit_change_pct")),
                "promo_change_pct": _to_num(metrics.get("promo_change_pct")),
                "roi_revenue_x": _to_num(metrics.get("roi_revenue_x")),
                "total_spend_plan": _to_num(metrics.get("total_spend_plan")),
                "revenue_delta": _to_num(metrics.get("revenue_delta")),
                "profit_delta": _to_num(metrics.get("profit_delta")),
                "quantity_delta": _to_num(metrics.get("quantity_delta")),
            },
        }
        payload_text = json.dumps(payload, default=str, ensure_ascii=True)

        return (
            "You are a senior trade-promotion strategy analyst for FMCG retail.\n"
            "Read slab-level monthly data and user edits, then write a practical business review.\n\n"
            "Output rules (strict):\n"
            "1) Use ONLY bullet points, no paragraph blocks.\n"
            "2) Return exactly 5 bullets total.\n"
            "3) Every bullet must include at least one number from input.\n"
            "4) Mention at least 2 specific months by name.\n"
            "5) Keep each bullet complete (no cut-off), clear, and action-oriented.\n"
            "6) Cover: volume, revenue, profit, spend, ROI, risk, and recommended actions.\n"
            "7) Start each bullet as: **Short Label:** detail.\n"
            "8) If user_changes is empty, explicitly say plan is unchanged and what that implies.\n\n"
            f"Input JSON:\n{payload_text}"
        )


    def _extract_planner_month_changes(
        self,
        months: List[str],
        default_structural: List[float],
        planned_structural: List[float],
        default_base_prices: List[float],
        planned_base_prices: List[float],
    ) -> List[Dict[str, float]]:
        def _to_num(value, default=0.0):
            try:
                v = float(value)
                return v if np.isfinite(v) else float(default)
            except Exception:
                return float(default)

        month_changes = []
        for i, month in enumerate(months):
            old_struct = _to_num(default_structural[i] if i < len(default_structural) else 0.0)
            new_struct = _to_num(planned_structural[i] if i < len(planned_structural) else old_struct)
            old_price = _to_num(default_base_prices[i] if i < len(default_base_prices) else 0.0)
            new_price = _to_num(planned_base_prices[i] if i < len(planned_base_prices) else old_price)
            delta_struct = new_struct - old_struct
            delta_price = new_price - old_price
            if abs(delta_struct) > 1e-9 or abs(delta_price) > 1e-9:
                month_changes.append(
                    {
                        "month": str(month),
                        "structural_discount_change_pp": round(delta_struct, 3),
                        "base_price_change": round(delta_price, 3),
                        "new_structural_discount_pct": round(new_struct, 3),
                        "new_base_price": round(new_price, 3),
                    }
                )
        return month_changes


    def _build_planner_fallback_insights(
        self,
        slab: str,
        months: List[str],
        default_structural: List[float],
        planned_structural: List[float],
        default_base_prices: List[float],
        planned_base_prices: List[float],
        metrics: Dict[str, float],
    ) -> str:
        def _num(value, default=0.0):
            try:
                v = float(value)
                return v if np.isfinite(v) else float(default)
            except Exception:
                return float(default)

        month_changes = self._extract_planner_month_changes(
            months=months,
            default_structural=default_structural,
            planned_structural=planned_structural,
            default_base_prices=default_base_prices,
            planned_base_prices=planned_base_prices,
        )
        changed_months = [row["month"] for row in month_changes]
        first_month = changed_months[0] if changed_months else (months[0] if months else "NA")
        second_month = changed_months[1] if len(changed_months) > 1 else (months[1] if len(months) > 1 else first_month)
        max_up = max([row for row in month_changes if row["structural_discount_change_pp"] > 0], key=lambda r: r["structural_discount_change_pp"], default=None)
        max_down = min([row for row in month_changes if row["structural_discount_change_pp"] < 0], key=lambda r: r["structural_discount_change_pp"], default=None)

        qty_delta = _num(metrics.get("quantity_delta"))
        rev_delta = _num(metrics.get("revenue_delta"))
        prof_delta = _num(metrics.get("profit_delta"))
        qty_pct = _num(metrics.get("volume_change_pct"))
        rev_pct = _num(metrics.get("revenue_change_pct"))
        prof_pct = _num(metrics.get("profit_change_pct"))
        roi_x = _num(metrics.get("roi_revenue_x"))
        spend = _num(metrics.get("total_spend_plan"))
        promo_pct = _num(metrics.get("promo_change_pct"))

        bullets = [
            f"- **Business Impact:** Slab {slab} shows volume {qty_pct:.2f}% ({qty_delta:,.0f} units), revenue {rev_pct:.2f}% ({rev_delta:,.0f}), and profit {prof_pct:.2f}% ({prof_delta:,.0f}).",
            f"- **Spend and ROI:** Promo intensity shifts {promo_pct:.2f}% vs baseline with spend {spend:,.0f}; revenue ROI is {roi_x:.2f}x.",
        ]

        if max_up is not None:
            bullets.append(
                f"- **Primary Increase Month:** {max_up['month']} has the largest step-up, +{max_up['structural_discount_change_pp']:.2f} pp to {max_up['new_structural_discount_pct']:.2f}%."
            )
        else:
            bullets.append(
                f"- **Promo Pattern:** No structural step-up detected; {first_month} to {second_month} stays at baseline depth."
            )

        if max_down is not None:
            bullets.append(
                f"- **Margin Protection:** Largest step-down is {max_down['month']} at {max_down['structural_discount_change_pp']:.2f} pp, which supports margin."
            )
        else:
            bullets.append(
                f"- **Margin Protection:** No step-down between {first_month} and {second_month}; margin control depends mainly on base-price discipline."
            )

        bullets.extend([
            f"- **Action:** If ROI is below 1.50x in high-discount months such as {first_month}, reduce depth by 0.5-1.0 pp and recheck next cycle.",
        ])
        return "### Trinity Insights\n" + "\n".join(bullets)


    def _is_low_quality_insight(self, text: str) -> bool:
        if not isinstance(text, str) or not text.strip():
            return True
        body = text.strip()
        bullet_count = len([line for line in body.splitlines() if line.strip().startswith("-")])
        if bullet_count < 4:
            return True
        if len(body) < 280:
            return True
        if re.search(r"\b(and|or|with|for|to)\s*$", body.lower()):
            return True
        if body.endswith("("):
            return True
        return False


    def _generate_planner_ai_insights(
        self,
        slab: str,
        months: List[str],
        default_structural: List[float],
        planned_structural: List[float],
        default_base_prices: List[float],
        planned_base_prices: List[float],
        metrics: Dict[str, float],
        series_rows: List[Dict[str, float]],
    ) -> Dict[str, str]:
        api_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
        if not api_key:
            return {
                "status": "disabled",
                "text": "Trinity Insights are disabled. Set GEMINI_API_KEY on backend and recalculate.",
            }

        model_name = (os.getenv("GEMINI_MODEL") or "gemini-2.5-flash").strip()
        prompt = self._build_planner_ai_prompt(
            slab=slab,
            months=months,
            default_structural=default_structural,
            planned_structural=planned_structural,
            default_base_prices=default_base_prices,
            planned_base_prices=planned_base_prices,
            metrics=metrics,
            series_rows=series_rows,
        )
        endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model_name}:generateContent?key={api_key}"
        )
        request_body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.2,
                "topP": 0.9,
                "maxOutputTokens": 700,
            },
        }

        try:
            req = urllib.request.Request(
                endpoint,
                data=json.dumps(request_body).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = resp.read().decode("utf-8")
            payload = json.loads(body)
            candidates = payload.get("candidates") or []
            parts = []
            for cand in candidates:
                content = cand.get("content") or {}
                for part in (content.get("parts") or []):
                    text = part.get("text")
                    if isinstance(text, str) and text.strip():
                        parts.append(text.strip())
            joined = "\n\n".join(parts).strip()
            if self._is_low_quality_insight(joined):
                fallback = self._build_planner_fallback_insights(
                    slab=slab,
                    months=months,
                    default_structural=default_structural,
                    planned_structural=planned_structural,
                    default_base_prices=default_base_prices,
                    planned_base_prices=planned_base_prices,
                    metrics=metrics,
                )
                return {"status": "ready", "text": fallback}
            return {"status": "ready", "text": joined}
        except urllib.error.HTTPError as err:
            detail = ""
            try:
                detail = err.read().decode("utf-8", errors="ignore")
            except Exception:
                detail = str(err)
            fallback = self._build_planner_fallback_insights(
                slab=slab,
                months=months,
                default_structural=default_structural,
                planned_structural=planned_structural,
                default_base_prices=default_base_prices,
                planned_base_prices=planned_base_prices,
                metrics=metrics,
            )
            return {"status": "ready", "text": f"{fallback}\n\nNote: Trinity model call failed ({err.code}). {detail[:120]}"}
        except Exception as err:
            fallback = self._build_planner_fallback_insights(
                slab=slab,
                months=months,
                default_structural=default_structural,
                planned_structural=planned_structural,
                default_base_prices=default_base_prices,
                planned_base_prices=planned_base_prices,
                metrics=metrics,
            )
            return {"status": "ready", "text": f"{fallback}\n\nNote: Trinity model error: {str(err)[:120]}"}


    async def calculate_12_month_planner(self, request: PlannerRequest) -> PlannerResponse:
        try:
            def _safe_num(value, default=0.0):
                try:
                    fv = float(value)
                    return fv if np.isfinite(fv) else float(default)
                except Exception:
                    return float(default)

            scope = self._build_step2_scope(request)
            if scope is None:
                return PlannerResponse(success=False, message="No data matches selected filters")

            df_scope = scope['df_scope']
            if df_scope.empty:
                return PlannerResponse(success=False, message="No data for planner")

            if 'Slab' in df_scope.columns:
                available_slabs = sorted(
                    df_scope['Slab'].dropna().astype(str).unique().tolist(),
                    key=self._slab_sort_key
                )
            else:
                available_slabs = ['All']

            selected_slab = str(request.slab or '').strip()
            if not selected_slab:
                selected_from_list = [str(s) for s in (request.slabs or [])]
                if selected_from_list:
                    selected_slab = selected_from_list[0]
            if not selected_slab:
                selected_slab = available_slabs[0] if available_slabs else 'All'

            slab_df = df_scope.copy()
            if 'Slab' in slab_df.columns:
                slab_df = slab_df[slab_df['Slab'].astype(str) == selected_slab]

            if slab_df.empty:
                return PlannerResponse(success=False, message=f"No data for slab {selected_slab}", slab=selected_slab)

            model_request = ModelingRequest(
                run_id=request.run_id,
                states=request.states,
                categories=request.categories,
                subcategories=request.subcategories,
                brands=request.brands,
                sizes=request.sizes,
                recency_threshold=request.recency_threshold,
                frequency_threshold=request.frequency_threshold,
                round_step=request.round_step,
                min_upward_jump_pp=request.min_upward_jump_pp,
                min_downward_drop_pp=request.min_downward_drop_pp,
                include_lag_discount=request.include_lag_discount,
                l2_penalty=request.l2_penalty,
                optimize_l2_penalty=request.optimize_l2_penalty,
                constraint_residual_non_negative=request.constraint_residual_non_negative,
                constraint_structural_non_negative=request.constraint_structural_non_negative,
                constraint_tactical_non_negative=request.constraint_tactical_non_negative,
                constraint_lag_non_positive=request.constraint_lag_non_positive,
                rfm_segments=request.rfm_segments,
                outlet_classifications=request.outlet_classifications,
                slabs=[selected_slab],
                outlet_ids=request.outlet_ids,
            )

            monthly = self._build_monthly_model_dataframe(slab_df, model_request)
            if monthly.empty or len(monthly) < 3:
                return PlannerResponse(success=False, message="Not enough monthly points for planner", slab=selected_slab)

            modeled = self._run_two_stage_model(
                monthly,
                include_lag_discount=bool(request.include_lag_discount),
                l2_penalty=float(getattr(request, 'l2_penalty', 1.0)),
                optimize_l2_penalty=bool(getattr(request, 'optimize_l2_penalty', False)),
                constraint_residual_non_negative=bool(getattr(request, 'constraint_residual_non_negative', True)),
                constraint_structural_non_negative=bool(getattr(request, 'constraint_structural_non_negative', True)),
                constraint_tactical_non_negative=bool(getattr(request, 'constraint_tactical_non_negative', True)),
                constraint_lag_non_positive=bool(getattr(request, 'constraint_lag_non_positive', True)),
            )
            if modeled is None:
                return PlannerResponse(success=False, message="Model could not be fitted for planner", slab=selected_slab)

            model_df = modeled['model_df'].sort_values('Period').reset_index(drop=True).copy()
            coefficients = modeled['coefficients']
            stage2_model = modeled['stage2_model']

            model_df['Period'] = pd.to_datetime(model_df['Period'], errors='coerce')
            model_df = model_df.dropna(subset=['Period']).reset_index(drop=True)
            if model_df.empty:
                return PlannerResponse(success=False, message="No clean monthly periods for planner", slab=selected_slab)

            model_df['Month_Key'] = model_df['Period'].dt.to_period('M')
            latest_period = pd.to_datetime(model_df['Period'].max())
            if request.plan_start_year is not None:
                plan_start = pd.Period(f"{int(request.plan_start_year)}-04", freq='M')
            else:
                start_year = int(latest_period.year) if int(latest_period.month) >= 4 else int(latest_period.year) - 1
                plan_start = pd.Period(f"{start_year}-04", freq='M')

            plan_months = list(pd.period_range(start=plan_start, periods=12, freq='M'))
            if len(plan_months) < 3:
                return PlannerResponse(success=False, message="Need at least 3 months for planning", slab=selected_slab)

            rows = []
            for m in plan_months:
                mm = model_df[model_df['Month_Key'] == m]
                if mm.empty:
                    rows.append({
                        'Month_Key': m,
                        'Period': m.to_timestamp(how='start'),
                        'base_price': np.nan,
                        'actual_discount_pct': np.nan,
                        'store_count': np.nan,
                        'residual_store': np.nan,
                        'base_discount_pct': np.nan,
                    })
                else:
                    r = mm.iloc[-1]
                    rows.append({
                        'Month_Key': m,
                        'Period': m.to_timestamp(how='start'),
                        'base_price': float(r['base_price']) if pd.notna(r['base_price']) else np.nan,
                        'actual_discount_pct': float(r['actual_discount_pct']) if pd.notna(r['actual_discount_pct']) else np.nan,
                        'store_count': float(r['store_count']) if pd.notna(r['store_count']) else np.nan,
                        'residual_store': float(r['residual_store']) if pd.notna(r['residual_store']) else np.nan,
                        'base_discount_pct': float(r['base_discount_pct']) if pd.notna(r['base_discount_pct']) else np.nan,
                    })

            plan_template = pd.DataFrame(rows).sort_values('Period').reset_index(drop=True)
            if plan_template['base_discount_pct'].isna().all():
                plan_template['base_discount_pct'] = 0.0
            else:
                plan_template['base_discount_pct'] = plan_template['base_discount_pct'].ffill().bfill()
            plan_template['base_discount_pct'] = self._round_discount_series(
                plan_template['base_discount_pct'], step=float(request.round_step)
            )

            base_price_series = pd.to_numeric(model_df['base_price'], errors='coerce').dropna()
            default_bp = float(base_price_series.iloc[-1]) if not base_price_series.empty else 100.0
            default_bp = float(np.round(default_bp * 2.0) / 2.0)
            plan_template['base_price'] = pd.to_numeric(plan_template['base_price'], errors='coerce').fillna(default_bp)

            missing_resid = plan_template['residual_store'].isna()
            if missing_resid.any():
                stage1_intercept = float(coefficients.get('stage1_intercept', 0.0))
                stage1_coef_discount = float(coefficients.get('stage1_coef_discount', 0.0))
                fallback_discount = plan_template['actual_discount_pct'].fillna(plan_template['base_discount_pct']).to_numpy(dtype=float)
                fallback_store = plan_template['store_count'].fillna(0.0).to_numpy(dtype=float)
                pred_store = stage1_intercept + stage1_coef_discount * fallback_discount
                fallback_resid = fallback_store - pred_store
                plan_template.loc[missing_resid, 'residual_store'] = fallback_resid[missing_resid.to_numpy()]
            plan_template['residual_store'] = plan_template['residual_store'].fillna(0.0)

            observed_struct = plan_template['base_discount_pct'].to_numpy(dtype=float)

            prev_month = plan_months[0] - 1
            prev_row = model_df[model_df['Month_Key'] == prev_month]
            if not prev_row.empty and pd.notna(prev_row['base_discount_pct'].iloc[-1]):
                prev_struct = float(prev_row['base_discount_pct'].iloc[-1])
            else:
                prev_struct = float(observed_struct[0])
            prev_struct = float(self._round_discount_series(pd.Series([prev_struct]), step=float(request.round_step)).iloc[0])

            ref_map = {
                mk: float(v)
                for mk, v in zip(
                    model_df['Month_Key'],
                    pd.to_numeric(model_df['base_discount_pct'], errors='coerce')
                )
                if pd.notna(v)
            }
            default_struct = np.empty(len(plan_months), dtype=float)
            fixed_override = self._planner_fixed_discount_override(
                request=request,
                slab_df=slab_df,
                selected_slab=selected_slab,
                plan_months=plan_months,
            )
            if fixed_override is not None and len(fixed_override) == len(plan_months):
                default_struct = fixed_override
            else:
                valid_ref_count = sum(1 for m in plan_months if (m - 12) in ref_map)
                if valid_ref_count >= 12:
                    default_struct = np.asarray(
                        [float(ref_map.get(m - 12, observed_struct[i])) for i, m in enumerate(plan_months)],
                        dtype=float
                    )
                else:
                    default_struct[0] = prev_struct
                    if len(default_struct) > 1:
                        default_struct[1:] = observed_struct[:-1]
            default_struct = np.clip(default_struct, 0.0, 60.0)
            default_struct = self._round_discount_series(default_struct, step=float(request.round_step)).to_numpy(dtype=float)

            n_plan = len(plan_months)
            planned_struct = request.planned_structural_discounts or []
            if len(planned_struct) != n_plan:
                planned_struct_arr = default_struct.copy()
            else:
                planned_struct_arr = np.asarray(planned_struct, dtype=float)
                planned_struct_arr = np.clip(planned_struct_arr, 0.0, 60.0)
                planned_struct_arr = self._round_discount_series(planned_struct_arr, step=float(request.round_step)).to_numpy(dtype=float)

            planned_base = request.planned_base_prices or []
            if len(planned_base) != n_plan:
                planned_base_arr = plan_template['base_price'].to_numpy(dtype=float)
            else:
                planned_base_arr = np.asarray(planned_base, dtype=float)
            planned_base_arr = np.maximum(planned_base_arr, 0.0)
            planned_base_arr = np.round(planned_base_arr * 2.0) / 2.0

            lag_old = np.empty(n_plan, dtype=float)
            lag_new = np.empty(n_plan, dtype=float)
            lag_old[0] = prev_struct
            lag_new[0] = prev_struct
            if n_plan > 1:
                lag_old[1:] = default_struct[:-1]
                lag_new[1:] = planned_struct_arr[:-1]

            residual_arr = plan_template['residual_store'].to_numpy(dtype=float)
            zeros_arr = np.zeros(n_plan, dtype=float)

            qty_old = self._predict_stage2_quantity(stage2_model, residual_arr, default_struct, zeros_arr, lag_old)
            qty_new = self._predict_stage2_quantity(stage2_model, residual_arr, planned_struct_arr, zeros_arr, lag_new)
            qty_old = np.maximum(qty_old, 0.0)
            qty_new = np.maximum(qty_new, 0.0)
            qty_zero = self._predict_stage2_quantity(stage2_model, residual_arr, zeros_arr, zeros_arr, zeros_arr)
            qty_zero = np.maximum(qty_zero, 0.0)

            price_old = planned_base_arr * (1.0 - default_struct / 100.0)
            price_new = planned_base_arr * (1.0 - planned_struct_arr / 100.0)
            # Keep zero-structural comparison on the same default effective price basis.
            price_zero = price_old.copy()
            rev_old = qty_old * price_old
            rev_new = qty_new * price_new
            rev_zero = qty_zero * price_zero

            cogs_default = float(np.round(default_bp * 0.5))
            cogs_per_unit = float(request.cogs_per_unit) if request.cogs_per_unit is not None else cogs_default
            cogs_per_unit = max(cogs_per_unit, 0.0)
            prof_old = qty_old * (price_old - cogs_per_unit)
            prof_new = qty_new * (price_new - cogs_per_unit)

            total_qty_old = float(np.nansum(qty_old))
            total_qty_new = float(np.nansum(qty_new))
            total_rev_old = float(np.nansum(rev_old))
            total_rev_new = float(np.nansum(rev_new))
            total_prof_old = float(np.nansum(prof_old))
            total_prof_new = float(np.nansum(prof_new))

            qty_delta = total_qty_new - total_qty_old
            rev_delta = total_rev_new - total_rev_old
            prof_delta = total_prof_new - total_prof_old

            qty_pct = float((qty_delta / total_qty_old) * 100.0) if abs(total_qty_old) > 1e-12 else np.nan
            rev_pct = float((rev_delta / total_rev_old) * 100.0) if abs(total_rev_old) > 1e-12 else np.nan
            prof_pct = float((prof_delta / total_prof_old) * 100.0) if abs(total_prof_old) > 1e-12 else np.nan

            avg_promo_old = float(np.nanmean(default_struct))
            avg_promo_new = float(np.nanmean(planned_struct_arr))
            promo_delta_pp = avg_promo_new - avg_promo_old
            promo_pct = float((promo_delta_pp / avg_promo_old) * 100.0) if abs(avg_promo_old) > 1e-12 else np.nan

            step_up_pp = np.clip(planned_struct_arr - default_struct, 0.0, None)
            spend_monthly = planned_base_arr * (step_up_pp / 100.0) * qty_new
            total_spend_plan = float(np.nansum(spend_monthly))
            roi_revenue = float(rev_delta / total_spend_plan) if total_spend_plan > 1e-12 else np.nan
            profit_roi_revenue = float(prof_delta / total_spend_plan) if total_spend_plan > 1e-12 else np.nan
            roi_revenue_pct = float(roi_revenue * 100.0) if np.isfinite(roi_revenue) else np.nan

            # Absolute structural ROI for baseline/default and user-planned scenarios.
            spend_default_monthly = planned_base_arr * (default_struct / 100.0) * qty_old
            spend_planned_monthly = planned_base_arr * (planned_struct_arr / 100.0) * qty_new
            total_spend_default = float(np.nansum(spend_default_monthly))
            total_spend_planned = float(np.nansum(spend_planned_monthly))
            investment_change = float(total_spend_planned - total_spend_default)
            investment_change_pct = float((investment_change / total_spend_default) * 100.0) if total_spend_default > 1e-12 else np.nan

            # ROI @ Default / Planned should follow Step 3 structural episode logic:
            # For each positive step-up, include the full hold window at the increased level.
            # Compare against counterfactual of previous structural level held constant.
            def _planner_structural_episode_totals(struct_arr: np.ndarray) -> tuple[float, float, float]:
                struct = np.asarray(struct_arr, dtype=float)
                if struct.size == 0:
                    return 0.0, 0.0, 0.0

                regime_break = np.abs(np.diff(struct, prepend=struct[0])) > 1e-9
                regime_id = np.cumsum(regime_break)
                row_idx = np.arange(struct.size, dtype=int)
                regime_df = pd.DataFrame(
                    {
                        'regime_id': regime_id,
                        'row_idx': row_idx,
                        'base_discount_pct': struct,
                    }
                )
                regimes = (
                    regime_df.groupby('regime_id', as_index=False)
                    .agg(
                        start_idx=('row_idx', 'min'),
                        end_idx=('row_idx', 'max'),
                        base_discount_pct=('base_discount_pct', 'first'),
                    )
                    .sort_values('start_idx')
                    .reset_index(drop=True)
                )

                total_inc = 0.0
                total_inc_profit = 0.0
                total_spend_step = 0.0

                for r in range(1, len(regimes)):
                    prev_base = float(regimes.loc[r - 1, 'base_discount_pct'])
                    curr_base = float(regimes.loc[r, 'base_discount_pct'])
                    step_up = curr_base - prev_base
                    if step_up <= 0:
                        continue

                    s_idx = int(regimes.loc[r, 'start_idx'])
                    e_idx = int(regimes.loc[r, 'end_idx'])
                    if e_idx < s_idx:
                        continue

                    hold_slice = slice(s_idx, e_idx + 1)
                    n_hold = e_idx - s_idx + 1
                    if n_hold <= 0:
                        continue

                    resid_hold = residual_arr[hold_slice]
                    base_hold = planned_base_arr[hold_slice]
                    zeros_hold = np.zeros(n_hold, dtype=float)

                    prev_struct = np.full(n_hold, prev_base, dtype=float)
                    curr_struct = np.full(n_hold, curr_base, dtype=float)
                    lag_prev = np.full(n_hold, prev_base, dtype=float)
                    lag_curr = np.full(n_hold, curr_base, dtype=float)
                    lag_curr[0] = prev_base

                    qty_prev = self._predict_stage2_quantity(stage2_model, resid_hold, prev_struct, zeros_hold, lag_prev)
                    qty_curr = self._predict_stage2_quantity(stage2_model, resid_hold, curr_struct, zeros_hold, lag_curr)
                    qty_prev = np.maximum(qty_prev, 0.0)
                    qty_curr = np.maximum(qty_curr, 0.0)

                    prev_price = base_hold * (1.0 - prev_base / 100.0)
                    inc_rev = (qty_curr - qty_prev) * prev_price
                    inc_profit = (qty_curr * (prev_price - cogs_per_unit)) - (qty_prev * (prev_price - cogs_per_unit))
                    spend_step = base_hold * (step_up / 100.0) * qty_curr

                    total_inc += float(np.nansum(inc_rev))
                    total_inc_profit += float(np.nansum(inc_profit))
                    total_spend_step += float(np.nansum(spend_step))

                return total_inc, total_inc_profit, total_spend_step

            inc_rev_default, inc_profit_default, spend_default_step = _planner_structural_episode_totals(default_struct)
            inc_rev_planned, inc_profit_planned, spend_planned_step = _planner_structural_episode_totals(planned_struct_arr)

            roi_default_x = float(inc_rev_default / spend_default_step) if spend_default_step > 1e-12 else np.nan
            roi_planned_x = float(inc_rev_planned / spend_planned_step) if spend_planned_step > 1e-12 else np.nan
            roi_abs_change_x = float(roi_planned_x - roi_default_x) if np.isfinite(roi_default_x) and np.isfinite(roi_planned_x) else np.nan
            profit_roi_default_x = float(inc_profit_default / spend_default_step) if spend_default_step > 1e-12 else np.nan
            profit_roi_planned_x = float(inc_profit_planned / spend_planned_step) if spend_planned_step > 1e-12 else np.nan
            profit_roi_abs_change_x = (
                float(profit_roi_planned_x - profit_roi_default_x)
                if np.isfinite(profit_roi_default_x) and np.isfinite(profit_roi_planned_x)
                else np.nan
            )

            metrics = {
                'volume_change_pct': _safe_num(qty_pct, 0.0),
                'revenue_change_pct': _safe_num(rev_pct, 0.0),
                'profit_change_pct': _safe_num(prof_pct, 0.0),
                'promo_change_pct': _safe_num(promo_pct, 0.0),
                'roi_change_pct': _safe_num(roi_revenue_pct, 0.0),
                'roi_revenue_x': _safe_num(roi_revenue, 0.0),
                'profit_roi_revenue_x': _safe_num(profit_roi_revenue, 0.0),
                'total_spend_plan': _safe_num(total_spend_plan, 0.0),
                'total_revenue_current': _safe_num(total_rev_old, 0.0),
                'total_revenue_planned': _safe_num(total_rev_new, 0.0),
                'total_profit_current': _safe_num(total_prof_old, 0.0),
                'total_profit_planned': _safe_num(total_prof_new, 0.0),
                'total_quantity_current': _safe_num(total_qty_old, 0.0),
                'total_quantity_planned': _safe_num(total_qty_new, 0.0),
                'revenue_delta': _safe_num(rev_delta, 0.0),
                'profit_delta': _safe_num(prof_delta, 0.0),
                'quantity_delta': _safe_num(qty_delta, 0.0),
                'investment_default': _safe_num(total_spend_default, 0.0),
                'investment_planned': _safe_num(total_spend_planned, 0.0),
                'investment_change': _safe_num(investment_change, 0.0),
                'investment_change_pct': _safe_num(investment_change_pct, 0.0),
                'roi_default_x': _safe_num(roi_default_x, 0.0),
                'roi_planned_x': _safe_num(roi_planned_x, 0.0),
                'roi_abs_change_x': _safe_num(roi_abs_change_x, 0.0),
                'profit_roi_default_x': _safe_num(profit_roi_default_x, 0.0),
                'profit_roi_planned_x': _safe_num(profit_roi_planned_x, 0.0),
                'profit_roi_abs_change_x': _safe_num(profit_roi_abs_change_x, 0.0),
            }

            series = [
                PlannerMonthPoint(
                    period=plan_template['Period'].iloc[i].to_pydatetime() if hasattr(plan_template['Period'].iloc[i], 'to_pydatetime') else plan_template['Period'].iloc[i],
                    current_promo_pct=_safe_num(default_struct[i], 0.0),
                    planned_promo_pct=_safe_num(planned_struct_arr[i], 0.0),
                    base_price=_safe_num(planned_base_arr[i], 0.0),
                    current_quantity=_safe_num(qty_old[i], 0.0),
                    planned_quantity=_safe_num(qty_new[i], 0.0),
                    current_revenue=_safe_num(rev_old[i], 0.0),
                    planned_revenue=_safe_num(rev_new[i], 0.0),
                )
                for i in range(n_plan)
            ]

            default_base_prices = [_safe_num(x, 0.0) for x in plan_template['base_price'].to_list()]
            series_rows = [
                {
                    "month": str(plan_months[i]),
                    "current_promo_pct": _safe_num(default_struct[i], 0.0),
                    "planned_promo_pct": _safe_num(planned_struct_arr[i], 0.0),
                    "base_price": _safe_num(planned_base_arr[i], 0.0),
                    "current_quantity": _safe_num(qty_old[i], 0.0),
                    "planned_quantity": _safe_num(qty_new[i], 0.0),
                    "current_revenue": _safe_num(rev_old[i], 0.0),
                    "planned_revenue": _safe_num(rev_new[i], 0.0),
                }
                for i in range(n_plan)
            ]
            recalculate_requested = (
                request.planned_structural_discounts is not None
                or request.planned_base_prices is not None
            )
            if recalculate_requested and not bool(getattr(request, "disable_ai_insights", False)):
                ai_payload = self._generate_planner_ai_insights(
                    slab=selected_slab,
                    months=[str(m) for m in plan_months],
                    default_structural=[_safe_num(x, 0.0) for x in default_struct.tolist()],
                    planned_structural=[_safe_num(x, 0.0) for x in planned_struct_arr.tolist()],
                    default_base_prices=default_base_prices,
                    planned_base_prices=[_safe_num(x, 0.0) for x in planned_base_arr.tolist()],
                    metrics=metrics,
                    series_rows=series_rows,
                )
            else:
                ai_payload = {
                    "status": "pending_recalculate",
                    "text": "Trinity Insights will appear after you click Recalculate Plan.",
                }

            return PlannerResponse(
                success=True,
                message="Step 4 planner calculated successfully",
                slab=selected_slab,
                plan_start_month=str(plan_months[0]),
                months=[str(m) for m in plan_months],
                default_structural_discounts=[_safe_num(x, 0.0) for x in default_struct.tolist()],
                current_structural_discounts=[_safe_num(x, 0.0) for x in default_struct.tolist()],
                planned_structural_discounts=[_safe_num(x, 0.0) for x in planned_struct_arr.tolist()],
                planned_base_prices=[_safe_num(x, 0.0) for x in planned_base_arr.tolist()],
                cogs_per_unit=_safe_num(cogs_per_unit, 0.0),
                model_coefficients=coefficients,
                metrics=metrics,
                series=series,
                ai_insights_status=ai_payload.get("status"),
                ai_insights=ai_payload.get("text"),
            )
        except Exception as e:
            return PlannerResponse(
                success=False,
                message=f"Error in step 4 planner: {str(e)}",
                slab=request.slab,
            )


    def _build_cross_size_monthly_quantity(self, df_scope: pd.DataFrame, size_key: str) -> pd.DataFrame:
        if df_scope is None or df_scope.empty or 'Date' not in df_scope.columns or 'Quantity' not in df_scope.columns or 'Sizes' not in df_scope.columns:
            return pd.DataFrame(columns=['Period', 'quantity'])
        work = df_scope.copy()
        normalized_size = self._normalize_step2_size_key(size_key)
        work = work[work['Sizes'].astype(str).map(self._normalize_step2_size_key) == normalized_size].copy()
        if work.empty:
            return pd.DataFrame(columns=['Period', 'quantity'])
        work['Date'] = pd.to_datetime(work['Date'], errors='coerce')
        work = work.dropna(subset=['Date']).copy()
        if work.empty:
            return pd.DataFrame(columns=['Period', 'quantity'])
        work['Period'] = work['Date'].dt.to_period('M').dt.to_timestamp(how='start')
        monthly = (
            work.groupby('Period', as_index=False)
            .agg(quantity=('Quantity', 'sum'))
            .sort_values('Period')
            .reset_index(drop=True)
        )
        monthly['quantity'] = pd.to_numeric(monthly['quantity'], errors='coerce')
        monthly = monthly.dropna(subset=['quantity']).copy()
        return monthly


    def _fit_cross_size_pct_change_model(self, df_scope: pd.DataFrame) -> Optional[Dict[str, float]]:
        m12 = self._build_cross_size_monthly_quantity(df_scope, '12-ML')
        m18 = self._build_cross_size_monthly_quantity(df_scope, '18-ML')
        if m12.empty or m18.empty:
            return None

        merged = (
            m12.rename(columns={'quantity': 'quantity_12'})
            .merge(
                m18.rename(columns={'quantity': 'quantity_18'}),
                on='Period',
                how='inner',
            )
            .sort_values('Period')
            .reset_index(drop=True)
        )
        if merged.empty or len(merged) < 6:
            return None

        merged['quantity_12'] = pd.to_numeric(merged['quantity_12'], errors='coerce')
        merged['quantity_18'] = pd.to_numeric(merged['quantity_18'], errors='coerce')
        merged = merged.dropna(subset=['quantity_12', 'quantity_18']).copy()
        merged = merged[(merged['quantity_12'] > 0) & (merged['quantity_18'] > 0)].copy()
        if merged.empty or len(merged) < 6:
            return None

        merged['pct_change_12'] = (
            merged['quantity_12'].pct_change().replace([np.inf, -np.inf], np.nan) * 100.0
        )
        merged['pct_change_18'] = (
            merged['quantity_18'].pct_change().replace([np.inf, -np.inf], np.nan) * 100.0
        )
        merged = merged.dropna(subset=['pct_change_12', 'pct_change_18']).copy()
        if merged.empty or len(merged) < 6:
            return None

        mdl12 = LinearRegression()
        mdl18 = LinearRegression()
        X12 = merged[['pct_change_18']].to_numpy(dtype=float)
        y12 = merged['pct_change_12'].to_numpy(dtype=float)
        X18 = merged[['pct_change_12']].to_numpy(dtype=float)
        y18 = merged['pct_change_18'].to_numpy(dtype=float)
        mdl12.fit(X12, y12)
        mdl18.fit(X18, y18)

        return {
            'cross_elasticity_12_from_18': float(mdl12.coef_[0]) if len(mdl12.coef_) >= 1 else np.nan,
            'cross_elasticity_18_from_12': float(mdl18.coef_[0]) if len(mdl18.coef_) >= 1 else np.nan,
            'r2_12': float(mdl12.score(X12, y12)) if len(y12) > 1 else np.nan,
            'r2_18': float(mdl18.score(X18, y18)) if len(y18) > 1 else np.nan,
        }


    def _run_cross_size_coupled_prediction(
        self,
        slab_state_local: Dict[str, Dict[str, float]],
        discount_map: Dict[str, float],
        max_iter: int = 8,
    ) -> tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        slabs_local = list((slab_state_local or {}).keys())
        qty_map_local = {s: float(slab_state_local[s].get('anchor_qty', 0.0)) for s in slabs_local}
        other_map_local = {s: 0.0 for s in slabs_local}
        for _ in range(max_iter):
            weight_map_local = self._normalize_weight_map_from_qty(qty_map_local)
            next_qty_local: Dict[str, float] = {}
            next_other_local: Dict[str, float] = {}
            for slab_key in slabs_local:
                state = slab_state_local.get(slab_key) or {}
                model = state.get('stage2_model')
                if model is None:
                    next_qty_local[slab_key] = max(float(state.get('anchor_qty', 0.0)), 0.0)
                    next_other_local[slab_key] = 0.0
                    continue

                own_disc = float(discount_map.get(slab_key, state.get('default_discount_pct', 0.0)))
                other_disc = self._compute_other_weighted_discount_for_slab(
                    slab_key,
                    discount_map,
                    weight_map_local,
                )
                pred = float(
                    self._predict_stage2_quantity(
                        model,
                        np.array([float(state.get('residual_store', 0.0))], dtype=float),
                        np.array([own_disc], dtype=float),
                        np.array([0.0], dtype=float),
                        np.array([float(state.get('lag1_base_discount_pct', 0.0))], dtype=float),
                        extra_feature_values={
                            'other_slabs_weighted_base_discount_pct': np.array([other_disc], dtype=float)
                        },
                    )[0]
                )
                next_qty_local[slab_key] = max(pred, 0.0)
                next_other_local[slab_key] = float(other_disc)
            delta_local = float(sum(abs(next_qty_local[s] - qty_map_local.get(s, 0.0)) for s in slabs_local))
            qty_map_local = next_qty_local
            other_map_local = next_other_local
            if delta_local <= 1e-6:
                break
        return qty_map_local, self._normalize_weight_map_from_qty(qty_map_local), other_map_local


    def _normalize_period_key(self, value: Any) -> str:
        try:
            ts = pd.to_datetime(value, errors='coerce')
            if pd.notna(ts):
                return ts.to_period('M').strftime('%Y-%m')
        except Exception:
            pass
        return str(value or '').strip()


    def _resolve_step4_modeled_weights(
        self,
        size_scope_all_slabs: pd.DataFrame,
        slab_list: List[str],
    ) -> tuple[Dict[str, float], str]:
        clean_slab_list = [str(s) for s in slab_list]
        if not clean_slab_list:
            return {}, "fallback_latest_mix"

        modeled_weights: Dict[str, float] = {}
        try:
            work = size_scope_all_slabs.copy()
            if not work.empty and 'Slab' in work.columns and 'Quantity' in work.columns:
                work = work[work['Slab'].astype(str).isin(clean_slab_list)].copy()
                grouped = (
                    work.groupby(work['Slab'].astype(str), as_index=False)['Quantity']
                    .sum()
                    .rename(columns={'Quantity': 'qty'})
                )
                total_qty = float(pd.to_numeric(grouped['qty'], errors='coerce').fillna(0.0).sum())
                if total_qty > 0:
                    for _, row in grouped.iterrows():
                        slab_key = str(row['Slab'])
                        modeled_weights[slab_key] = max(float(row['qty']) / total_qty, 0.0)
        except Exception:
            modeled_weights = {}

        for slab in clean_slab_list:
            modeled_weights.setdefault(slab, 0.0)

        total_weight = float(sum(v for v in modeled_weights.values() if np.isfinite(v) and v > 0))
        if total_weight > 0:
            normalized = {
                slab: max(float(modeled_weights.get(slab, 0.0)), 0.0) / total_weight
                for slab in clean_slab_list
            }
            return normalized, "modeled"

        fallback_weights: Dict[str, float] = {}
        try:
            latest = size_scope_all_slabs.copy()
            if not latest.empty and 'Date' in latest.columns and 'Slab' in latest.columns and 'Quantity' in latest.columns:
                latest['Date'] = pd.to_datetime(latest['Date'], errors='coerce')
                latest = latest.dropna(subset=['Date']).copy()
                if not latest.empty:
                    last_period = latest['Date'].dt.to_period('M').max()
                    latest = latest[latest['Date'].dt.to_period('M') == last_period].copy()
                    latest = latest[latest['Slab'].astype(str).isin(clean_slab_list)].copy()
                    grouped = (
                        latest.groupby(latest['Slab'].astype(str), as_index=False)['Quantity']
                        .sum()
                        .rename(columns={'Quantity': 'qty'})
                    )
                    total_qty = float(pd.to_numeric(grouped['qty'], errors='coerce').fillna(0.0).sum())
                    if total_qty > 0:
                        for _, row in grouped.iterrows():
                            slab_key = str(row['Slab'])
                            fallback_weights[slab_key] = max(float(row['qty']) / total_qty, 0.0)
        except Exception:
            fallback_weights = {}

        for slab in clean_slab_list:
            fallback_weights.setdefault(slab, 0.0)
        fallback_total = float(sum(v for v in fallback_weights.values() if np.isfinite(v) and v > 0))
        if fallback_total > 0:
            fallback_weights = {
                slab: max(float(fallback_weights.get(slab, 0.0)), 0.0) / fallback_total
                for slab in clean_slab_list
            }
        else:
            n = max(len(clean_slab_list), 1)
            fallback_weights = {slab: (1.0 / n) for slab in clean_slab_list}

        return fallback_weights, "fallback_latest_mix"


    def _extract_period_scenario_discounts(
        self,
        request: CrossSizePlannerRequest,
        periods: List[str],
        defaults_matrix: Dict[str, Dict[str, List[float]]],
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        period_payload = request.scenario_discounts_by_period or {}
        legacy_payload = request.scenario_discounts_by_size or {}

        normalized_period_payload: Dict[str, Dict[str, Dict[str, float]]] = {}
        for raw_period, per_size in period_payload.items():
            period_key = self._normalize_period_key(raw_period)
            size_map: Dict[str, Dict[str, float]] = {}
            for raw_size, slab_map in (per_size or {}).items():
                size_key = self._normalize_step2_size_key(raw_size)
                if not size_key:
                    continue
                size_map[size_key] = {
                    str(k): float(v)
                    for k, v in (slab_map or {}).items()
                    if v is not None
                }
            normalized_period_payload[period_key] = size_map

        normalized_legacy_payload: Dict[str, Dict[str, float]] = {}
        for raw_size, slab_map in legacy_payload.items():
            size_key = self._normalize_step2_size_key(raw_size)
            if not size_key:
                continue
            normalized_legacy_payload[size_key] = {
                str(k): float(v)
                for k, v in (slab_map or {}).items()
                if v is not None
            }

        out: Dict[str, Dict[str, Dict[str, float]]] = {}
        for month_idx, period_key in enumerate(periods):
            month_map: Dict[str, Dict[str, float]] = {}
            period_override = normalized_period_payload.get(period_key, {})
            for size_key, slab_series_map in defaults_matrix.items():
                resolved: Dict[str, float] = {}
                for slab_key, default_series in slab_series_map.items():
                    default_value = float(default_series[month_idx]) if month_idx < len(default_series) else 0.0
                    override = None
                    if slab_key in (period_override.get(size_key, {}) or {}):
                        override = period_override[size_key].get(slab_key)
                    elif slab_key in (normalized_legacy_payload.get(size_key, {}) or {}):
                        override = normalized_legacy_payload[size_key].get(slab_key)
                    scenario_value = float(override) if override is not None else default_value
                    resolved[str(slab_key)] = max(scenario_value, 0.0)
                month_map[str(size_key)] = resolved
            out[str(period_key)] = month_map
        return out


    async def calculate_cross_size_planner(self, request: CrossSizePlannerRequest) -> CrossSizePlannerResponse:
        try:
            scope = self._build_step2_scope(request)
            if scope is None:
                return CrossSizePlannerResponse(success=False, message="No data matches selected filters", size_results=[], impact_summary={})

            df_scope = scope['df_scope']
            if df_scope.empty:
                return CrossSizePlannerResponse(success=False, message="No data for cross-size planner", size_results=[], impact_summary={})

            summary_source = scope.get('df_scope_all_slabs', df_scope)
            forecast_months = int(getattr(request, 'forecast_months', 3) or 3)
            size_results: List[CrossSizePlannerSizeResult] = []
            pair_state: Dict[str, Dict[str, Any]] = {}
            global_last_period: Optional[pd.Timestamp] = None

            for size_key in ['12-ML', '18-ML']:
                size_df = df_scope.copy()
                size_scope_all_slabs = summary_source.copy()
                if 'Sizes' in size_df.columns:
                    size_df = size_df[
                        size_df['Sizes'].astype(str).map(self._normalize_step2_size_key) == self._normalize_step2_size_key(size_key)
                    ]
                    size_scope_all_slabs = size_scope_all_slabs[
                        size_scope_all_slabs['Sizes'].astype(str).map(self._normalize_step2_size_key) == self._normalize_step2_size_key(size_key)
                    ]
                if size_df.empty:
                    continue

                slab_list = (
                    size_df['Slab'].dropna().astype(str).unique().tolist()
                    if 'Slab' in size_df.columns else []
                )
                slab_list = [
                    str(s)
                    for s in sorted(list(dict.fromkeys(slab_list)), key=self._slab_sort_key)
                    if self._is_step2_allowed_slab(s)
                ]
                if not slab_list:
                    continue

                slab_rows: List[CrossSizePlannerSlabState] = []
                slab_state: Dict[str, Dict[str, Any]] = {}

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

                    model_df = modeled['model_df'].sort_values('Period').reset_index(drop=True).copy()
                    model_df['Period'] = pd.to_datetime(model_df['Period'], errors='coerce')
                    model_df = model_df.dropna(subset=['Period']).copy()
                    if model_df.empty:
                        continue

                    slab_last_period = pd.to_datetime(model_df['Period'].max(), errors='coerce')
                    if pd.notna(slab_last_period):
                        if global_last_period is None or slab_last_period > global_last_period:
                            global_last_period = slab_last_period

                    last_row = model_df.iloc[-1]
                    coeff = modeled['coefficients']
                    default_discount = float(last_row.get('base_discount_pct', 0.0))
                    anchor_qty = max(float(last_row.get('quantity', 0.0)), 1.0)
                    base_price_series = pd.to_numeric(model_df.get('base_price', pd.Series(dtype=float)), errors='coerce').dropna()
                    default_base_price = float(base_price_series.iloc[-1]) if not base_price_series.empty else 0.0
                    base_price_for_slab = float(last_row.get('base_price', default_base_price)) if pd.notna(last_row.get('base_price', np.nan)) else default_base_price
                    cogs_for_slab = float(self._resolve_modeling_cogs_for_size(request, size_key, 0.0))

                    baseline_series = pd.to_numeric(
                        model_df.get('non_discount_baseline_quantity', pd.Series(dtype=float)),
                        errors='coerce',
                    ).fillna(0.0).to_numpy(dtype=float)
                    if baseline_series.size == 0:
                        baseline_series = pd.to_numeric(
                            model_df.get('baseline_quantity', pd.Series(dtype=float)),
                            errors='coerce',
                        ).fillna(0.0).to_numpy(dtype=float)
                    baseline_forecast = self._forecast_baseline_series(baseline_series.tolist(), forecast_months)
                    baseline_forecast = np.clip(np.asarray(baseline_forecast, dtype=float), 0.0, None)

                    slab_rows.append(
                        CrossSizePlannerSlabState(
                            slab=str(slab),
                            default_discount_pct=float(default_discount),
                            scenario_discount_pct=float(default_discount),
                            residual_store=float(last_row.get('residual_store', 0.0)),
                            lag1_base_discount_pct=float(last_row.get('lag1_base_discount_pct', 0.0)),
                            anchor_qty=float(anchor_qty),
                            base_price=float(base_price_for_slab),
                            cogs_per_unit=float(cogs_for_slab),
                            stage2_intercept=float(coeff.get('stage2_intercept', 0.0)),
                            coef_residual_store=float(coeff.get('coef_residual_store', 0.0)),
                            coef_base_discount_pct=float(coeff.get('coef_structural_discount', 0.0)),
                            coef_lag1_base_discount_pct=float(coeff.get('coef_lag1_structural_discount', 0.0)),
                            coef_other_slabs_weighted_base_discount_pct=float(
                                coeff.get('coef_other_slabs_weighted_base_discount_pct', 0.0)
                            ),
                        )
                    )
                    slab_state[str(slab)] = {
                        'coefficients': {
                            'coef_base_discount_pct': float(coeff.get('coef_structural_discount', 0.0)),
                            'coef_lag1_base_discount_pct': float(coeff.get('coef_lag1_structural_discount', 0.0)),
                            'coef_other_slabs_weighted_base_discount_pct': float(
                                coeff.get('coef_other_slabs_weighted_base_discount_pct', 0.0)
                            ),
                        },
                        'default_discount_pct': float(default_discount),
                        'anchor_qty': float(anchor_qty),
                        'baseline_forecast': baseline_forecast.tolist(),
                        'base_price': float(base_price_for_slab),
                        'cogs_per_unit': float(cogs_for_slab),
                    }

                if not slab_rows:
                    continue

                modeled_weights, weight_source = self._resolve_step4_modeled_weights(size_scope_all_slabs, slab_list)
                size_results.append(
                    CrossSizePlannerSizeResult(
                        size=size_key,
                        slabs=slab_rows,
                        modeled_weights={str(k): float(v) for k, v in modeled_weights.items()},
                        weight_source=str(weight_source),
                    )
                )
                total_qty_scope = float(pd.to_numeric(size_df.get('Quantity', pd.Series(dtype=float)), errors='coerce').fillna(0.0).sum())
                total_sales_scope = float(pd.to_numeric(size_df.get('SalesValue_atBasicRate', pd.Series(dtype=float)), errors='coerce').fillna(0.0).sum())
                avg_base_price_scope = (total_sales_scope / total_qty_scope) if total_qty_scope > 0 else 0.0
                cogs_scope = self._resolve_modeling_cogs_for_size(request, size_key, avg_base_price_scope * 0.5 if avg_base_price_scope > 0 else 0.0)
                pair_state[size_key] = {
                    'slab_state': slab_state,
                    'default_map': {row.slab: row.default_discount_pct for row in slab_rows},
                    'modeled_weights': {str(k): float(v) for k, v in modeled_weights.items()},
                    'weight_source': str(weight_source),
                    'avg_base_price': float(avg_base_price_scope),
                    'cogs_per_unit': float(cogs_scope),
                }

            if not size_results:
                return CrossSizePlannerResponse(success=False, message="No valid size/slab models for Step 4", size_results=[], impact_summary={})

            if global_last_period is None or pd.isna(global_last_period):
                global_last_period = pd.Timestamp.today().to_period('M').to_timestamp()
            periods = [
                (pd.to_datetime(global_last_period) + pd.DateOffset(months=i + 1)).to_period('M').strftime('%Y-%m')
                for i in range(forecast_months)
            ]

            defaults_matrix: Dict[str, Dict[str, List[float]]] = {}
            baseline_slab_matrix: Dict[str, Dict[str, List[float]]] = {}
            for size_key, block in pair_state.items():
                defaults_matrix[size_key] = {}
                baseline_slab_matrix[size_key] = {}
                for slab_key, slab_payload in block.get('slab_state', {}).items():
                    default_discount = float(slab_payload.get('default_discount_pct', 0.0))
                    defaults_matrix[size_key][str(slab_key)] = [default_discount for _ in range(forecast_months)]
                    base_series = [
                        max(float(v), 0.0)
                        for v in (slab_payload.get('baseline_forecast') or [0.0] * forecast_months)
                    ]
                    if len(base_series) < forecast_months:
                        tail = base_series[-1] if base_series else 0.0
                        base_series.extend([tail] * (forecast_months - len(base_series)))
                    baseline_slab_matrix[size_key][str(slab_key)] = base_series[:forecast_months]

            scenario_by_period = self._extract_period_scenario_discounts(
                request=request,
                periods=periods,
                defaults_matrix=defaults_matrix,
            )
            scenario_matrix: Dict[str, Dict[str, List[float]]] = {
                size_key: {slab_key: [0.0] * forecast_months for slab_key in slab_map.keys()}
                for size_key, slab_map in defaults_matrix.items()
            }
            for month_idx, period_key in enumerate(periods):
                month_map = scenario_by_period.get(period_key, {})
                for size_key, slab_map in defaults_matrix.items():
                    for slab_key in slab_map.keys():
                        scenario_val = float((month_map.get(size_key, {}) or {}).get(slab_key, slab_map[slab_key][month_idx]))
                        scenario_matrix[size_key][slab_key][month_idx] = max(scenario_val, 0.0)

            cross_fit = self._fit_cross_size_pct_change_model(df_scope)
            e12_from_18 = float(cross_fit.get('cross_elasticity_12_from_18', 0.0)) if cross_fit else 0.0
            e18_from_12 = float(cross_fit.get('cross_elasticity_18_from_12', 0.0)) if cross_fit else 0.0
            cross_r2_12 = float(cross_fit.get('r2_12_model', 0.0)) if cross_fit else 0.0
            cross_r2_18 = float(cross_fit.get('r2_18_model', 0.0)) if cross_fit else 0.0

            monthly_results: List[Dict[str, Any]] = []
            for month_idx, period_key in enumerate(periods):
                month_default_maps = {
                    size_key: {slab_key: float(defaults_matrix[size_key][slab_key][month_idx]) for slab_key in defaults_matrix[size_key].keys()}
                    for size_key in defaults_matrix.keys()
                }
                month_scenario_maps = {
                    size_key: {slab_key: float(scenario_matrix[size_key][slab_key][month_idx]) for slab_key in scenario_matrix[size_key].keys()}
                    for size_key in scenario_matrix.keys()
                }
                month_sizes: Dict[str, Dict[str, Any]] = {}

                for size_key in ['12-ML', '18-ML']:
                    block = pair_state.get(size_key, {})
                    slab_state = block.get('slab_state', {})
                    if not slab_state:
                        continue
                    default_map = month_default_maps.get(size_key, {})
                    scenario_map = month_scenario_maps.get(size_key, {})
                    weight_map = block.get('modeled_weights', {}) or {}

                    slab_rows_month: List[Dict[str, Any]] = []
                    baseline_total_qty_default_world = 0.0
                    baseline_total_qty_non_discount = 0.0
                    pre_cross_total_qty = 0.0
                    for slab_key in sorted(list(slab_state.keys()), key=self._slab_sort_key):
                        slab_payload = slab_state.get(slab_key, {})
                        coeff = slab_payload.get('coefficients', {})
                        baseline_series = baseline_slab_matrix.get(size_key, {}).get(slab_key, [0.0] * forecast_months)
                        baseline_non_discount_qty = float(baseline_series[month_idx]) if month_idx < len(baseline_series) else 0.0
                        default_discount = float(default_map.get(slab_key, slab_payload.get('default_discount_pct', 0.0)))
                        scenario_discount = float(scenario_map.get(slab_key, default_discount))
                        default_lag = float(default_map.get(slab_key, default_discount)) if month_idx > 0 else float(slab_payload.get('default_discount_pct', default_discount))
                        scenario_lag = (
                            float(scenario_matrix.get(size_key, {}).get(slab_key, [scenario_discount])[month_idx - 1])
                            if month_idx > 0
                            else float(slab_payload.get('default_discount_pct', scenario_discount))
                        )
                        other_default = self._compute_other_weighted_discount_for_slab(slab_key, default_map, weight_map)
                        other_scenario = self._compute_other_weighted_discount_for_slab(slab_key, scenario_map, weight_map)

                        coef_base = float(coeff.get('coef_base_discount_pct', 0.0))
                        coef_lag = float(coeff.get('coef_lag1_base_discount_pct', 0.0))
                        coef_other = float(coeff.get('coef_other_slabs_weighted_base_discount_pct', 0.0))
                        discount_component_default = (
                            (coef_base * default_discount)
                            + (coef_lag * default_lag)
                            + (coef_other * other_default)
                        )
                        discount_component_scenario = (
                            (coef_base * scenario_discount)
                            + (coef_lag * scenario_lag)
                            + (coef_other * other_scenario)
                        )
                        delta_qty_discount = discount_component_scenario - discount_component_default
                        default_world_qty = max(baseline_non_discount_qty + float(discount_component_default), 0.0)
                        pre_cross_qty = max(baseline_non_discount_qty + float(discount_component_scenario), 0.0)

                        baseline_total_qty_default_world += default_world_qty
                        baseline_total_qty_non_discount += baseline_non_discount_qty
                        pre_cross_total_qty += pre_cross_qty
                        slab_rows_month.append(
                            {
                                'slab': str(slab_key),
                                'default_discount_pct': float(default_discount),
                                'scenario_discount_pct': float(scenario_discount),
                                'default_lag_used_pct': float(default_lag),
                                'lag_used_pct': float(scenario_lag),
                                'other_weighted_default_pct': float(other_default),
                                'other_weighted_scenario_pct': float(other_scenario),
                                'delta_qty_discount': float(delta_qty_discount),
                                'discount_component_default_qty': float(discount_component_default),
                                'discount_component_scenario_qty': float(discount_component_scenario),
                                'non_discount_baseline_qty': float(baseline_non_discount_qty),
                                'baseline_qty': float(baseline_non_discount_qty),
                                'default_world_qty': float(default_world_qty),
                                'pre_cross_qty': float(pre_cross_qty),
                                'final_qty': float(pre_cross_qty),
                                'base_price': float(slab_payload.get('base_price', 0.0)),
                                'cogs_per_unit': float(slab_payload.get('cogs_per_unit', 0.0)),
                            }
                        )

                    month_sizes[size_key] = {
                        'size': size_key,
                        'baseline_total_qty': float(baseline_total_qty_non_discount),
                        'baseline_total_qty_default_world': float(baseline_total_qty_default_world),
                        'pre_cross_total_qty': float(pre_cross_total_qty),
                        'final_total_qty': float(pre_cross_total_qty),
                        'slabs': slab_rows_month,
                    }

                monthly_results.append(
                    {
                        'period': str(period_key),
                        'sizes': month_sizes,
                        'impact': {
                            'prev12_qty': float(month_sizes.get('12-ML', {}).get('baseline_total_qty', 0.0)),
                            'prev18_qty': float(month_sizes.get('18-ML', {}).get('baseline_total_qty', 0.0)),
                            'pre12_qty': float(month_sizes.get('12-ML', {}).get('pre_cross_total_qty', 0.0)),
                            'pre18_qty': float(month_sizes.get('18-ML', {}).get('pre_cross_total_qty', 0.0)),
                            'final12_qty': float(month_sizes.get('12-ML', {}).get('pre_cross_total_qty', 0.0)),
                            'final18_qty': float(month_sizes.get('18-ML', {}).get('pre_cross_total_qty', 0.0)),
                            'own12_pct': 0.0,
                            'own18_pct': 0.0,
                            'overall12_pct': 0.0,
                            'overall18_pct': 0.0,
                        },
                    }
                )

            reference_3m = self._compute_step4_reference_3m(
                df_scope=df_scope,
                periods=periods,
                pair_state=pair_state,
                reference_mode=str(getattr(request, 'reference_mode', 'ly_same_3m') or 'ly_same_3m'),
            )

            baseline_qty_12_3m = float(sum(row.get('sizes', {}).get('12-ML', {}).get('baseline_total_qty', 0.0) for row in monthly_results))
            baseline_qty_18_3m = float(sum(row.get('sizes', {}).get('18-ML', {}).get('baseline_total_qty', 0.0) for row in monthly_results))
            precross_qty_12_3m = float(sum(row.get('sizes', {}).get('12-ML', {}).get('pre_cross_total_qty', 0.0) for row in monthly_results))
            precross_qty_18_3m = float(sum(row.get('sizes', {}).get('18-ML', {}).get('pre_cross_total_qty', 0.0) for row in monthly_results))

            ref_qty_12_3m = float(reference_3m.get('12-ML', {}).get('reference_qty', 0.0))
            ref_qty_18_3m = float(reference_3m.get('18-ML', {}).get('reference_qty', 0.0))

            own12_3m = ((precross_qty_12_3m - ref_qty_12_3m) / ref_qty_12_3m * 100.0) if ref_qty_12_3m > 0 else 0.0
            own18_3m = ((precross_qty_18_3m - ref_qty_18_3m) / ref_qty_18_3m * 100.0) if ref_qty_18_3m > 0 else 0.0

            # Step 4 planner is additive-only: no cross-pack volume readjustment applied.
            overall12_3m = own12_3m
            overall18_3m = own18_3m
            final_qty_12_3m = max(precross_qty_12_3m, 0.0)
            final_qty_18_3m = max(precross_qty_18_3m, 0.0)

            target_size_totals = {'12-ML': float(final_qty_12_3m), '18-ML': float(final_qty_18_3m)}

            # Redistribute final 3M size totals back to slab-month using scenario slab-month shares.
            for size_key in ['12-ML', '18-ML']:
                cells: List[Dict[str, Any]] = []
                sum_pre_cross = 0.0
                sum_baseline = 0.0
                for row in monthly_results:
                    size_payload = row.get('sizes', {}).get(size_key, {})
                    for slab_row in size_payload.get('slabs', []):
                        pre_v = max(float(slab_row.get('pre_cross_qty', 0.0)), 0.0)
                        base_v = max(float(slab_row.get('non_discount_baseline_qty', 0.0)), 0.0)
                        sum_pre_cross += pre_v
                        sum_baseline += base_v
                        cells.append({'row': row, 'size_payload': size_payload, 'slab_row': slab_row, 'pre': pre_v, 'base': base_v})

                n_cells = len(cells)
                target_total = float(target_size_totals.get(size_key, 0.0))
                if n_cells <= 0:
                    continue

                if sum_pre_cross > 0:
                    shares = [c['pre'] / sum_pre_cross for c in cells]
                elif sum_baseline > 0:
                    shares = [c['base'] / sum_baseline for c in cells]
                else:
                    shares = [1.0 / float(n_cells)] * n_cells

                for idx, c in enumerate(cells):
                    c['slab_row']['final_qty'] = float(max(target_total * shares[idx], 0.0))

            # Recompute month totals and size summaries from redistributed final slab-month qty.
            summary_acc = {
                '12-ML': {'baseline_qty': 0.0, 'pre_cross_qty': 0.0, 'final_qty': 0.0, 'baseline_revenue': 0.0, 'scenario_revenue': 0.0, 'baseline_profit': 0.0, 'scenario_profit': 0.0},
                '18-ML': {'baseline_qty': 0.0, 'pre_cross_qty': 0.0, 'final_qty': 0.0, 'baseline_revenue': 0.0, 'scenario_revenue': 0.0, 'baseline_profit': 0.0, 'scenario_profit': 0.0},
            }
            for row in monthly_results:
                period_sizes = row.get('sizes', {})
                for size_key in ['12-ML', '18-ML']:
                    size_payload = period_sizes.get(size_key)
                    if size_payload is None:
                        continue

                    baseline_total_qty_non_discount = float(size_payload.get('baseline_total_qty', 0.0))
                    pre_cross_total_qty = float(size_payload.get('pre_cross_total_qty', 0.0))
                    slabs_for_month = size_payload.get('slabs', [])
                    final_total_qty = float(sum(max(float(s.get('final_qty', 0.0)), 0.0) for s in slabs_for_month))

                    baseline_revenue_total = 0.0
                    scenario_revenue_total = 0.0
                    baseline_profit_total = 0.0
                    scenario_profit_total = 0.0
                    for slab_row in slabs_for_month:
                        final_qty = max(float(slab_row.get('final_qty', 0.0)), 0.0)
                        base_price = float(slab_row.get('base_price', 0.0))
                        cogs_per_unit = float(slab_row.get('cogs_per_unit', 0.0))
                        default_discount = float(slab_row.get('default_discount_pct', 0.0))
                        scenario_discount = float(slab_row.get('scenario_discount_pct', 0.0))
                        baseline_qty = float(slab_row.get('non_discount_baseline_qty', 0.0))

                        baseline_revenue = baseline_qty * base_price * (1.0 - (default_discount / 100.0))
                        scenario_revenue = final_qty * base_price * (1.0 - (scenario_discount / 100.0))
                        baseline_profit = baseline_revenue - (baseline_qty * cogs_per_unit)
                        scenario_profit = scenario_revenue - (final_qty * cogs_per_unit)

                        slab_row['baseline_revenue'] = float(baseline_revenue)
                        slab_row['scenario_revenue'] = float(scenario_revenue)
                        slab_row['baseline_profit'] = float(baseline_profit)
                        slab_row['scenario_profit'] = float(scenario_profit)

                        baseline_revenue_total += baseline_revenue
                        scenario_revenue_total += scenario_revenue
                        baseline_profit_total += baseline_profit
                        scenario_profit_total += scenario_profit

                    size_payload['final_total_qty'] = float(final_total_qty)
                    size_payload['volume_delta_pct'] = ((final_total_qty - baseline_total_qty_non_discount) / baseline_total_qty_non_discount * 100.0) if baseline_total_qty_non_discount > 0 else 0.0
                    size_payload['baseline_revenue_total'] = float(baseline_revenue_total)
                    size_payload['scenario_revenue_total'] = float(scenario_revenue_total)
                    size_payload['revenue_delta_pct'] = ((scenario_revenue_total - baseline_revenue_total) / baseline_revenue_total * 100.0) if baseline_revenue_total > 0 else 0.0
                    size_payload['baseline_profit_total'] = float(baseline_profit_total)
                    size_payload['scenario_profit_total'] = float(scenario_profit_total)
                    size_payload['profit_delta_pct'] = ((scenario_profit_total - baseline_profit_total) / abs(baseline_profit_total) * 100.0) if abs(baseline_profit_total) > 1e-9 else 0.0

                    summary_acc[size_key]['baseline_qty'] += baseline_total_qty_non_discount
                    summary_acc[size_key]['pre_cross_qty'] += pre_cross_total_qty
                    summary_acc[size_key]['final_qty'] += final_total_qty
                    summary_acc[size_key]['baseline_revenue'] += baseline_revenue_total
                    summary_acc[size_key]['scenario_revenue'] += scenario_revenue_total
                    summary_acc[size_key]['baseline_profit'] += baseline_profit_total
                    summary_acc[size_key]['scenario_profit'] += scenario_profit_total

                row['impact'] = {
                    'prev12_qty': float(row.get('sizes', {}).get('12-ML', {}).get('baseline_total_qty', 0.0)),
                    'prev18_qty': float(row.get('sizes', {}).get('18-ML', {}).get('baseline_total_qty', 0.0)),
                    'pre12_qty': float(row.get('sizes', {}).get('12-ML', {}).get('pre_cross_total_qty', 0.0)),
                    'pre18_qty': float(row.get('sizes', {}).get('18-ML', {}).get('pre_cross_total_qty', 0.0)),
                    'final12_qty': float(row.get('sizes', {}).get('12-ML', {}).get('final_total_qty', 0.0)),
                    'final18_qty': float(row.get('sizes', {}).get('18-ML', {}).get('final_total_qty', 0.0)),
                    'own12_pct': float(own12_3m),
                    'own18_pct': float(own18_3m),
                    'overall12_pct': float(overall12_3m),
                    'overall18_pct': float(overall18_3m),
                }

            summary_3m: Dict[str, Dict[str, float]] = {}
            for size_key in ['12-ML', '18-ML']:
                agg = summary_acc.get(size_key, {})
                baseline_qty = float(agg.get('baseline_qty', 0.0))
                pre_cross_qty = float(agg.get('pre_cross_qty', 0.0))
                final_qty = float(agg.get('final_qty', 0.0))
                baseline_revenue = float(agg.get('baseline_revenue', 0.0))
                scenario_revenue = float(agg.get('scenario_revenue', 0.0))
                baseline_profit = float(agg.get('baseline_profit', 0.0))
                scenario_profit = float(agg.get('scenario_profit', 0.0))
                reference_qty = float(reference_3m.get(size_key, {}).get('reference_qty', 0.0))
                reference_revenue = float(reference_3m.get(size_key, {}).get('reference_revenue', 0.0))
                reference_profit = float(reference_3m.get(size_key, {}).get('reference_profit', 0.0))
                reference_available = float(reference_3m.get(size_key, {}).get('reference_available', 0.0))
                summary_3m[size_key] = {
                    'baseline_qty': baseline_qty,
                    'scenario_qty_additive': pre_cross_qty,
                    'discount_component_qty': float(pre_cross_qty - baseline_qty),
                    'final_qty': final_qty,
                    'volume_delta_pct': ((final_qty - baseline_qty) / baseline_qty * 100.0) if baseline_qty > 0 else 0.0,
                    'volume_delta_additive_pct': ((pre_cross_qty - baseline_qty) / baseline_qty * 100.0) if baseline_qty > 0 else 0.0,
                    'baseline_revenue': baseline_revenue,
                    'scenario_revenue': scenario_revenue,
                    'revenue_delta_pct': ((scenario_revenue - baseline_revenue) / baseline_revenue * 100.0) if baseline_revenue > 0 else 0.0,
                    'baseline_profit': baseline_profit,
                    'scenario_profit': scenario_profit,
                    'profit_delta_pct': ((scenario_profit - baseline_profit) / abs(baseline_profit) * 100.0) if abs(baseline_profit) > 1e-9 else 0.0,
                    'reference_qty': reference_qty,
                    'reference_revenue': reference_revenue,
                    'reference_profit': reference_profit,
                    'own_delta_vs_reference_pct': ((pre_cross_qty - reference_qty) / reference_qty * 100.0) if reference_qty > 0 else 0.0,
                    'adjusted_delta_vs_reference_pct': ((final_qty - reference_qty) / reference_qty * 100.0) if reference_qty > 0 else 0.0,
                    'vs_reference_volume_pct': ((final_qty - reference_qty) / reference_qty * 100.0) if reference_qty > 0 else 0.0,
                    'vs_reference_revenue_pct': ((scenario_revenue - reference_revenue) / reference_revenue * 100.0) if reference_revenue > 0 else 0.0,
                    'vs_reference_profit_pct': ((scenario_profit - reference_profit) / abs(reference_profit) * 100.0) if abs(reference_profit) > 1e-9 else 0.0,
                    'reference_available': reference_available,
                }

            total_baseline_qty = float(sum(summary_3m.get(size, {}).get('baseline_qty', 0.0) for size in ['12-ML', '18-ML']))
            total_pre_cross_qty = float(sum(summary_3m.get(size, {}).get('scenario_qty_additive', 0.0) for size in ['12-ML', '18-ML']))
            total_final_qty = float(sum(summary_3m.get(size, {}).get('final_qty', 0.0) for size in ['12-ML', '18-ML']))
            total_baseline_revenue = float(sum(summary_3m.get(size, {}).get('baseline_revenue', 0.0) for size in ['12-ML', '18-ML']))
            total_scenario_revenue = float(sum(summary_3m.get(size, {}).get('scenario_revenue', 0.0) for size in ['12-ML', '18-ML']))
            total_baseline_profit = float(sum(summary_3m.get(size, {}).get('baseline_profit', 0.0) for size in ['12-ML', '18-ML']))
            total_scenario_profit = float(sum(summary_3m.get(size, {}).get('scenario_profit', 0.0) for size in ['12-ML', '18-ML']))
            reference_total_qty = float(reference_3m.get('TOTAL', {}).get('reference_qty', 0.0))
            reference_total_revenue = float(reference_3m.get('TOTAL', {}).get('reference_revenue', 0.0))
            reference_total_profit = float(reference_3m.get('TOTAL', {}).get('reference_profit', 0.0))
            baseline_volume_ml = float(
                sum(
                    summary_3m.get(size, {}).get('baseline_qty', 0.0) * self._pack_size_ml(size)
                    for size in ['12-ML', '18-ML']
                )
            )
            scenario_volume_ml_additive = float(
                sum(
                    summary_3m.get(size, {}).get('scenario_qty_additive', 0.0) * self._pack_size_ml(size)
                    for size in ['12-ML', '18-ML']
                )
            )
            final_volume_ml = float(
                sum(
                    summary_3m.get(size, {}).get('final_qty', 0.0) * self._pack_size_ml(size)
                    for size in ['12-ML', '18-ML']
                )
            )
            reference_volume_ml = float(
                sum(
                    float(reference_3m.get(size, {}).get('reference_qty', 0.0)) * self._pack_size_ml(size)
                    for size in ['12-ML', '18-ML']
                )
            )
            summary_3m['TOTAL'] = {
                'baseline_qty': total_baseline_qty,
                'scenario_qty_additive': total_pre_cross_qty,
                'discount_component_qty': float(total_pre_cross_qty - total_baseline_qty),
                'final_qty': total_final_qty,
                'volume_delta_pct': ((total_final_qty - total_baseline_qty) / total_baseline_qty * 100.0) if total_baseline_qty > 0 else 0.0,
                'volume_delta_additive_pct': ((total_pre_cross_qty - total_baseline_qty) / total_baseline_qty * 100.0) if total_baseline_qty > 0 else 0.0,
                'baseline_revenue': total_baseline_revenue,
                'scenario_revenue': total_scenario_revenue,
                'revenue_delta_pct': ((total_scenario_revenue - total_baseline_revenue) / total_baseline_revenue * 100.0) if total_baseline_revenue > 0 else 0.0,
                'baseline_profit': total_baseline_profit,
                'scenario_profit': total_scenario_profit,
                'profit_delta_pct': ((total_scenario_profit - total_baseline_profit) / abs(total_baseline_profit) * 100.0) if abs(total_baseline_profit) > 1e-9 else 0.0,
                'reference_qty': reference_total_qty,
                'reference_revenue': reference_total_revenue,
                'reference_profit': reference_total_profit,
                'vs_reference_volume_pct': (
                    ((total_final_qty - reference_total_qty) / reference_total_qty * 100.0)
                    if reference_total_qty > 0 else 0.0
                ),
                'vs_reference_revenue_pct': (
                    ((total_scenario_revenue - reference_total_revenue) / reference_total_revenue * 100.0)
                    if reference_total_revenue > 0 else 0.0
                ),
                'vs_reference_profit_pct': (
                    ((total_scenario_profit - reference_total_profit) / abs(reference_total_profit) * 100.0)
                    if abs(reference_total_profit) > 1e-9 else 0.0
                ),
                'reference_available': float(reference_3m.get('TOTAL', {}).get('reference_available', 0.0)),
                'baseline_volume_ml': baseline_volume_ml,
                'scenario_volume_ml_additive': scenario_volume_ml_additive,
                'final_volume_ml': final_volume_ml,
                'reference_volume_ml': reference_volume_ml,
                'volume_ml_delta_pct': ((final_volume_ml - baseline_volume_ml) / baseline_volume_ml * 100.0) if baseline_volume_ml > 0 else 0.0,
                'volume_ml_delta_additive_pct': ((scenario_volume_ml_additive - baseline_volume_ml) / baseline_volume_ml * 100.0) if baseline_volume_ml > 0 else 0.0,
                'vs_reference_volume_ml_pct': (
                    ((final_volume_ml - reference_volume_ml) / reference_volume_ml * 100.0)
                    if reference_volume_ml > 0 else 0.0
                ),
            }

            impact_summary = {
                'delta_12_due_to_own_pct': float(summary_3m.get('12-ML', {}).get('own_delta_vs_reference_pct', 0.0)),
                'delta_18_due_to_own_pct': float(summary_3m.get('18-ML', {}).get('own_delta_vs_reference_pct', 0.0)),
                'delta_12_overall_pct': float(summary_3m.get('12-ML', {}).get('adjusted_delta_vs_reference_pct', 0.0)),
                'delta_18_overall_pct': float(summary_3m.get('18-ML', {}).get('adjusted_delta_vs_reference_pct', 0.0)),
                'delta_12_vs_reference_pct': float(summary_3m.get('12-ML', {}).get('adjusted_delta_vs_reference_pct', 0.0)),
                'delta_18_vs_reference_pct': float(summary_3m.get('18-ML', {}).get('adjusted_delta_vs_reference_pct', 0.0)),
                'avg_base_price_12': float(pair_state.get('12-ML', {}).get('avg_base_price', 0.0)),
                'avg_base_price_18': float(pair_state.get('18-ML', {}).get('avg_base_price', 0.0)),
                'cogs_per_unit_12': float(pair_state.get('12-ML', {}).get('cogs_per_unit', 0.0)),
                'cogs_per_unit_18': float(pair_state.get('18-ML', {}).get('cogs_per_unit', 0.0)),
            }
            return CrossSizePlannerResponse(
                success=True,
                message="Cross-size Step 4 planner calculated successfully",
                size_results=size_results,
                impact_summary=impact_summary,
                periods=periods,
                defaults_matrix=defaults_matrix,
                scenario_matrix=scenario_matrix,
                baseline_slab_matrix=baseline_slab_matrix,
                monthly_results=monthly_results,
                summary_3m=summary_3m,
                cross_elasticity_12_from_18=float(e12_from_18),
                cross_elasticity_18_from_12=float(e18_from_12),
                cross_model_r2_12=float(cross_r2_12),
                cross_model_r2_18=float(cross_r2_18),
                reference_mode=str(getattr(request, 'reference_mode', 'ly_same_3m') or 'ly_same_3m'),
            )
        except Exception as e:
            return CrossSizePlannerResponse(
                success=False,
                message=f"Error in cross-size planner: {str(e)}",
                size_results=[],
                impact_summary={},
            )

