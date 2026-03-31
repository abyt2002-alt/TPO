"""Step 4 cross-size planner and planner AI helpers.\n\nThis module owns cross-size coupling, elasticity application, and the existing\nplanner computation/insights paths used by Step 4/Step 6 flows.\n"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import asyncio
import threading
import time
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
import random
import hashlib
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
    AIScenarioJobCreateResponse, AIScenarioJobStatusResponse, AIScenarioJobResultsResponse,
    BaselineForecastRequest, BaselineForecastResponse, BaselineForecastPoint,
    EDARequest, EDAResponse, EDAProductOption, EDAProductContribution,
    EDAContributionRow, EDAOptionsResponse
)


class _AIJobCancelled(Exception):
    pass


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

    def _clamp_discount_5_30_int(self, value: Any, default: float) -> int:
        try:
            v = float(value)
            if not np.isfinite(v):
                raise ValueError
        except Exception:
            v = float(default)
        return int(min(30, max(5, int(round(v)))))

    def _enforce_size_month_ladder(self, slab_values: Dict[str, float], slab_order: List[str]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        floor = 5.0
        for slab in slab_order:
            current = self._clamp_discount_5_30(slab_values.get(slab), floor)
            next_value = max(floor, current)
            out[slab] = float(round(next_value, 2))
            floor = out[slab]
        return out

    def _enforce_size_month_ladder_int(self, slab_values: Dict[str, Any], slab_order: List[str]) -> Dict[str, int]:
        out: Dict[str, int] = {}
        floor = 5
        for slab in slab_order:
            current = self._clamp_discount_5_30_int(slab_values.get(slab), floor)
            next_value = max(floor, current)
            out[slab] = int(next_value)
            floor = out[slab]
        return out

    def _parse_step5_discount_constraints(
        self,
        request: AIScenarioGenerateRequest,
        periods: List[str],
        slab_order_by_size: Dict[str, List[str]],
    ) -> Dict[str, Dict[str, Optional[int]]]:
        out: Dict[str, Dict[str, Optional[int]]] = {}
        period_set = {str(p) for p in (periods or [])}
        slab_set_by_size = {
            str(size_key): {str(slab) for slab in (slabs or [])}
            for size_key, slabs in (slab_order_by_size or {}).items()
        }
        raw_constraints = getattr(request, 'discount_constraints', None) or []
        for raw in raw_constraints:
            item = raw if isinstance(raw, dict) else {}
            period_key = self._normalize_period_key(item.get('period'))
            size_key = self._normalize_step2_size_key(item.get('size'))
            slab_key = str(item.get('slab') or '').strip()
            if period_key not in period_set:
                continue
            if size_key not in slab_set_by_size:
                continue
            if slab_key not in slab_set_by_size.get(size_key, set()):
                continue

            min_raw = item.get('min')
            max_raw = item.get('max')
            min_val: Optional[int] = None
            max_val: Optional[int] = None
            try:
                if min_raw is not None and str(min_raw) != '':
                    min_val = self._clamp_discount_5_30_int(min_raw, 5.0)
            except Exception:
                min_val = None
            try:
                if max_raw is not None and str(max_raw) != '':
                    max_val = self._clamp_discount_5_30_int(max_raw, 30.0)
            except Exception:
                max_val = None

            if min_val is not None and max_val is not None and min_val > max_val:
                raise RuntimeError(
                    f"Invalid discount constraints for {period_key} {size_key} {slab_key}: min ({min_val}) > max ({max_val})."
                )

            out[f"{period_key}|{size_key}|{slab_key}"] = {
                'min': min_val,
                'max': max_val,
            }
        return out

    def _enforce_size_month_ladder_int_with_bounds(
        self,
        slab_values: Dict[str, Any],
        slab_order: List[str],
        bounds_by_slab: Dict[str, Dict[str, Optional[int]]],
    ) -> Optional[Dict[str, int]]:
        if not slab_order:
            return {}

        lows: List[int] = []
        highs: List[int] = []
        seeds: List[int] = []
        for slab in slab_order:
            bound = bounds_by_slab.get(str(slab), {}) or {}
            low = 5 if bound.get('min') is None else int(bound.get('min'))
            high = 30 if bound.get('max') is None else int(bound.get('max'))
            low = int(min(30, max(5, low)))
            high = int(min(30, max(5, high)))
            if low > high:
                return None
            lows.append(low)
            highs.append(high)
            seeds.append(self._clamp_discount_5_30_int(slab_values.get(slab), low))

        # Forward adjust minimum bounds for monotonic ladder.
        for idx in range(1, len(lows)):
            lows[idx] = max(lows[idx], lows[idx - 1])
        # Backward adjust maximum bounds for monotonic ladder.
        for idx in range(len(highs) - 2, -1, -1):
            highs[idx] = min(highs[idx], highs[idx + 1])

        for idx in range(len(lows)):
            if lows[idx] > highs[idx]:
                return None

        out: Dict[str, int] = {}
        prev = lows[0]
        for idx, slab in enumerate(slab_order):
            low = lows[idx]
            high = highs[idx]
            val = max(int(seeds[idx]), low, prev)
            if val > high:
                val = high
            if val < max(low, prev):
                return None
            out[str(slab)] = int(val)
            prev = out[str(slab)]

        return out

    def _step5_clamp_int(self, value: Any, lo: int, hi: int, default: int) -> int:
        try:
            v = int(round(float(value)))
        except Exception:
            v = int(default)
        return max(int(lo), min(int(hi), int(v)))

    def _step5_allowed_patterns(self) -> set:
        return {"flat", "up", "down", "wave", "pulse"}

    def _step5_normalize_weights(self, weights: Dict[str, Any], keys: List[str], fallback: float = 1.0) -> Dict[str, float]:
        parsed: Dict[str, float] = {}
        for key in keys:
            try:
                val = float((weights or {}).get(key, fallback))
            except Exception:
                val = float(fallback)
            parsed[str(key)] = max(0.0, val)
        total = float(sum(parsed.values()))
        if total <= 0:
            eq = 1.0 / max(1, len(keys))
            return {str(k): eq for k in keys}
        return {str(k): float(parsed[str(k)] / total) for k in keys}

    def _step5_default_family(self, index: int) -> Dict[str, Any]:
        templates = [
            {
                "name": "Balanced realistic",
                "priority_weight": 0.40,
                "base_min": 10,
                "base_max": 16,
                "gap_min": 1,
                "gap_max": 2,
                "month_pattern": "flat",
                "month_shift_strength": 1,
                "size_bias_12": 0,
                "size_bias_18": 0,
                "volatility": 1,
                "anchor_weights": {
                    "latest_month": 0.40,
                    "last_3m_avg": 0.40,
                    "ly_same_3m": 0.10,
                    "stress_explore": 0.10,
                },
            },
            {
                "name": "Revenue push",
                "priority_weight": 0.35,
                "base_min": 12,
                "base_max": 20,
                "gap_min": 1,
                "gap_max": 3,
                "month_pattern": "up",
                "month_shift_strength": 1,
                "size_bias_12": 1,
                "size_bias_18": 1,
                "volatility": 2,
                "anchor_weights": {
                    "latest_month": 0.30,
                    "last_3m_avg": 0.20,
                    "ly_same_3m": 0.10,
                    "stress_explore": 0.40,
                },
            },
            {
                "name": "Margin safe",
                "priority_weight": 0.25,
                "base_min": 8,
                "base_max": 14,
                "gap_min": 1,
                "gap_max": 2,
                "month_pattern": "down",
                "month_shift_strength": 1,
                "size_bias_12": -1,
                "size_bias_18": 0,
                "volatility": 1,
                "anchor_weights": {
                    "latest_month": 0.20,
                    "last_3m_avg": 0.50,
                    "ly_same_3m": 0.20,
                    "stress_explore": 0.10,
                },
            },
        ]
        return copy.deepcopy(templates[index % len(templates)])

    def _step5_sanitize_family(self, item: Dict[str, Any], idx: int) -> Dict[str, Any]:
        raw = dict(item or {})
        pattern = str(raw.get("month_pattern", "flat")).strip().lower()
        if pattern not in self._step5_allowed_patterns():
            pattern = "flat"

        base_min = self._step5_clamp_int(raw.get("base_min", 10), 5, 30, 10)
        base_max = self._step5_clamp_int(raw.get("base_max", 18), 5, 30, 18)
        if base_max < base_min:
            base_min, base_max = base_max, base_min
        if base_max - base_min < 2:
            base_max = min(30, base_min + 2)

        gap_min = self._step5_clamp_int(raw.get("gap_min", 1), 1, 10, 1)
        gap_max = self._step5_clamp_int(raw.get("gap_max", 3), 1, 10, 3)
        if gap_max < gap_min:
            gap_min, gap_max = gap_max, gap_min
        if gap_max - gap_min < 1:
            gap_max = min(10, gap_min + 1)

        anchor_keys = ["latest_month", "last_3m_avg", "ly_same_3m", "stress_explore"]
        return {
            "name": str(raw.get("name") or f"Family {idx + 1}"),
            "priority_weight": max(0.0, float(raw.get("priority_weight", 1.0))),
            "base_min": int(base_min),
            "base_max": int(base_max),
            "gap_min": int(gap_min),
            "gap_max": int(gap_max),
            "month_pattern": pattern,
            "month_shift_strength": self._step5_clamp_int(raw.get("month_shift_strength", 2), 0, 6, 2),
            "size_bias_12": self._step5_clamp_int(raw.get("size_bias_12", 0), -6, 6, 0),
            "size_bias_18": self._step5_clamp_int(raw.get("size_bias_18", 0), -6, 6, 0),
            "volatility": self._step5_clamp_int(raw.get("volatility", 1), 0, 6, 1),
            "anchor_weights": self._step5_normalize_weights(raw.get("anchor_weights", {}), anchor_keys),
        }

    def _step5_family_similarity_score(self, a: Dict[str, Any], b: Dict[str, Any]) -> float:
        pattern_score = 1.0 if a.get("month_pattern") == b.get("month_pattern") else 0.0
        numeric_parts = [
            max(0.0, 1.0 - (abs(float(a["base_min"]) - float(b["base_min"])) / 6.0)),
            max(0.0, 1.0 - (abs(float(a["base_max"]) - float(b["base_max"])) / 6.0)),
            max(0.0, 1.0 - (abs(float(a["gap_min"]) - float(b["gap_min"])) / 3.0)),
            max(0.0, 1.0 - (abs(float(a["gap_max"]) - float(b["gap_max"])) / 3.0)),
            max(0.0, 1.0 - (abs(float(a["month_shift_strength"]) - float(b["month_shift_strength"])) / 3.0)),
            max(0.0, 1.0 - (abs(float(a["size_bias_12"]) - float(b["size_bias_12"])) / 4.0)),
            max(0.0, 1.0 - (abs(float(a["size_bias_18"]) - float(b["size_bias_18"])) / 4.0)),
            max(0.0, 1.0 - (abs(float(a["volatility"]) - float(b["volatility"])) / 3.0)),
        ]
        numeric_score = float(sum(numeric_parts) / max(1, len(numeric_parts)))
        anchor_keys = ["latest_month", "last_3m_avg", "ly_same_3m", "stress_explore"]
        l1 = sum(
            abs(float((a.get("anchor_weights") or {}).get(k, 0.0)) - float((b.get("anchor_weights") or {}).get(k, 0.0)))
            for k in anchor_keys
        )
        anchor_score = max(0.0, 1.0 - (l1 / 2.0))
        return (0.35 * pattern_score) + (0.45 * numeric_score) + (0.20 * anchor_score)

    def _step5_families_too_similar(self, a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        return self._step5_family_similarity_score(a, b) >= 0.90

    def _step5_diversify_family(self, family: Dict[str, Any], idx: int) -> Dict[str, Any]:
        cycle = ["flat", "up", "down", "wave", "pulse"]
        mutated = dict(family or {})
        current = str(mutated.get("month_pattern", "flat"))
        if current not in cycle:
            current = "flat"
        mutated["name"] = f"{family.get('name', f'Family {idx + 1}')} (diversified)"
        mutated["month_pattern"] = cycle[(cycle.index(current) + 1) % len(cycle)]
        mutated["base_min"] = max(5, int(family.get("base_min", 10)) - (1 + (idx % 2)))
        mutated["base_max"] = min(30, int(family.get("base_max", 18)) + 2)
        if int(mutated["base_max"]) - int(mutated["base_min"]) < 3:
            mutated["base_max"] = min(30, int(mutated["base_min"]) + 3)
        mutated["gap_min"] = max(1, int(family.get("gap_min", 1)))
        mutated["gap_max"] = min(10, max(int(family.get("gap_max", 3)) + 1, int(mutated["gap_min"]) + 1))
        mutated["month_shift_strength"] = min(6, max(1, int(family.get("month_shift_strength", 2)) + 1))
        mutated["volatility"] = min(6, max(2, int(family.get("volatility", 1)) + 1))
        mutated["size_bias_12"] = self._step5_clamp_int(int(family.get("size_bias_12", 0)) + (1 if idx % 2 == 0 else -1), -6, 6, 0)
        mutated["size_bias_18"] = self._step5_clamp_int(int(family.get("size_bias_18", 0)) + (-1 if idx % 2 == 0 else 1), -6, 6, 0)
        default_anchor = self._step5_sanitize_family(self._step5_default_family(idx), idx).get("anchor_weights", {})
        anchor_keys = ["latest_month", "last_3m_avg", "ly_same_3m", "stress_explore"]
        blended = {
            k: (0.70 * float((family.get("anchor_weights") or {}).get(k, 0.0))) + (0.30 * float(default_anchor.get(k, 0.0)))
            for k in anchor_keys
        }
        mutated["anchor_weights"] = self._step5_normalize_weights(blended, anchor_keys)
        return self._step5_sanitize_family(mutated, idx)

    def _step5_diversify_families(self, families: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        diversified = [dict(f) for f in (families or [])]
        actions: List[Dict[str, Any]] = []
        for i in range(len(diversified)):
            for j in range(i + 1, len(diversified)):
                if not self._step5_families_too_similar(diversified[i], diversified[j]):
                    continue
                before_score = round(self._step5_family_similarity_score(diversified[i], diversified[j]), 4)
                candidate = self._step5_diversify_family(diversified[j], j)
                action_type = "widened"
                if self._step5_families_too_similar(diversified[i], candidate):
                    candidate = self._step5_sanitize_family(self._step5_default_family(j), j)
                    action_type = "replaced_with_default"
                after_score = round(self._step5_family_similarity_score(diversified[i], candidate), 4)
                diversified[j] = candidate
                actions.append({
                    "pair": f"{i + 1}-{j + 1}",
                    "action": action_type,
                    "before_similarity": before_score,
                    "after_similarity": after_score,
                })
        return diversified, actions

    def _step5_sanitize_families(self, payload: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        families_raw = payload.get("families", []) if isinstance(payload, dict) else []
        if not isinstance(families_raw, list):
            families_raw = []
        sanitized = [self._step5_sanitize_family(item, idx) for idx, item in enumerate(families_raw[:3])]
        while len(sanitized) < 3:
            sanitized.append(self._step5_sanitize_family(self._step5_default_family(len(sanitized)), len(sanitized)))
        sanitized, diversity_actions = self._step5_diversify_families(sanitized)
        weight_map = self._step5_normalize_weights({str(i): f.get("priority_weight", 1.0) for i, f in enumerate(sanitized)}, ["0", "1", "2"])
        for i, fam in enumerate(sanitized):
            fam["priority_weight"] = float(weight_map.get(str(i), 0.0))
        return sanitized, {
            "provided_count": len(families_raw),
            "final_count": len(sanitized),
            "diversity_actions": diversity_actions,
        }

    def _step5_allocate_count_by_family(self, total: int, family_weights: List[float]) -> List[int]:
        total = int(total or 0)
        n = len(family_weights or [])
        if total <= 0 or n == 0:
            return [0] * n
        weight_keys = [str(i) for i in range(n)]
        normalized = self._step5_normalize_weights({str(i): float(family_weights[i]) for i in range(n)}, weight_keys)
        weights = [float(normalized[str(i)]) for i in range(n)]
        if total >= n:
            base = [1] * n
            remaining = total - n
            raw = [w * remaining for w in weights]
        else:
            base = [0] * n
            remaining = total
            raw = [w * remaining for w in weights]
        floors = [int(v) for v in raw]
        allocated = [base[i] + floors[i] for i in range(n)]
        remainder = total - sum(allocated)
        fractions = sorted([(raw[i] - floors[i], i) for i in range(n)], reverse=True)
        idx = 0
        while remainder > 0 and idx < len(fractions):
            allocated[fractions[idx][1]] += 1
            remainder -= 1
            idx += 1
        return allocated

    def _step5_month_shift(self, pattern: str, idx: int, strength: int) -> int:
        if pattern == "up":
            return idx * int(strength)
        if pattern == "down":
            return -idx * int(strength)
        if pattern == "wave":
            seq = [0, int(strength), -int(strength)]
            return seq[idx % 3]
        if pattern == "pulse":
            seq = [int(strength), 0, int(strength)]
            return seq[idx % 3]
        return 0

    def _build_step5_ai_prompt(
        self,
        scenario_count: int,
        goal: str,
        user_prompt: str,
        periods: List[str],
        defaults_matrix: Dict[str, Dict[str, List[float]]],
        planner_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        slab_order_map: Dict[str, List[str]] = {
            size_key: sorted(list((defaults_matrix.get(size_key) or {}).keys()), key=self._slab_sort_key)
            for size_key in ["12-ML", "18-ML"]
        }
        period_defaults: Dict[str, Dict[str, Dict[str, float]]] = {}
        for idx, period in enumerate(periods):
            pkey = str(period)
            period_defaults[pkey] = {}
            for size_key in ["12-ML", "18-ML"]:
                period_defaults[pkey][size_key] = {}
                for slab_key in slab_order_map.get(size_key, []):
                    series = defaults_matrix.get(size_key, {}).get(slab_key, []) or []
                    default_val = float(series[idx]) if idx < len(series) else 10.0
                    period_defaults[pkey][size_key][slab_key] = float(round(default_val, 2))

        schema = {
            "families": [
                {
                    "name": "Balanced realistic",
                    "priority_weight": 0.4,
                    "base_min": 10,
                    "base_max": 17,
                    "gap_min": 1,
                    "gap_max": 3,
                    "month_pattern": "flat",
                    "month_shift_strength": 2,
                    "size_bias_12": 0,
                    "size_bias_18": 0,
                    "volatility": 1,
                    "anchor_weights": {
                        "latest_month": 0.4,
                        "last_3m_avg": 0.3,
                        "ly_same_3m": 0.2,
                        "stress_explore": 0.1,
                    },
                }
            ]
        }
        prompt_text = str(user_prompt or "").strip()
        prompt_lower = prompt_text.lower()

        intent_hints: List[str] = []
        if ("12" in prompt_lower or "12ml" in prompt_lower or "12-ml" in prompt_lower) and any(
            k in prompt_lower for k in ["increase", "grow", "growth", "up", "volume"]
        ):
            intent_hints.append(
                "If user asks 12-ML volume growth, prioritize stronger 12-ML discount moves (within constraints) "
                "before changing 18-ML."
            )
        if ("18" in prompt_lower or "18ml" in prompt_lower or "18-ml" in prompt_lower) and any(
            k in prompt_lower for k in ["increase", "grow", "growth", "up", "volume"]
        ):
            intent_hints.append(
                "If user asks 18-ML volume growth, prioritize stronger 18-ML discount moves (within constraints)."
            )
        if any(k in prompt_lower for k in ["profit", "margin", "gm"]):
            intent_hints.append(
                "For profit/margin goals, prefer shallower discount ladders, especially for 18-ML "
                "(higher margin pack), unless prompt explicitly asks aggressive 18-ML discounting."
            )
        if any(k in prompt_lower for k in ["no deep", "not deep", "avoid deep", "shallow"]) and (
            "18" in prompt_lower or "18ml" in prompt_lower or "18-ml" in prompt_lower
        ):
            intent_hints.append(
                "If prompt says no deep 18-ML discount, keep 18-ML ladder in shallow-to-moderate range and avoid "
                "high deep-discount tops."
            )
        if any(k in prompt_lower for k in ["revenue", "sales"]):
            intent_hints.append(
                "For revenue goals, allow moderate-to-high discount ladders but keep family diversity and realism."
            )
        if not intent_hints:
            intent_hints.append(
                "Map user intent explicitly by pack: 12-ML actions should primarily come from 12-ML ladders, "
                "18-ML actions from 18-ML ladders."
            )

        prompt_payload = {
            "scenario_count": int(scenario_count),
            "goal": str(goal or "").strip() or "maximize_revenue",
            "user_prompt": prompt_text,
            "periods": [str(p) for p in periods],
            "default_discounts_by_period": period_defaults,
            "slab_order": slab_order_map,
            "planner_context": planner_context or {},
            "intent_hints": intent_hints,
            "intent_resolution_framework": {
                "step_1": "Infer pack target from prompt (12-ML, 18-ML, both).",
                "step_2": "Infer objective priority (volume, revenue, profit, balanced).",
                "step_3": "Use coefficient signs/magnitudes to set family direction by slab.",
                "step_4": "Use margin profile to avoid unnecessary deep discounting when objective is profit.",
                "step_5": "Generate diverse but business-realistic families aligned to inferred intent.",
            },
            "constraints": [
                "return EXACTLY 3 families",
                "all discounts are later repaired to integer [5,30]",
                "ladder is repaired monotonic later; still keep realistic family ranges",
                "avoid degenerate families (all ones / all flat tiny values)",
            ],
            "schema": schema,
        }
        return (
            "Return ONE JSON object only (no markdown, no code fences).\n"
            "You are generating scenario families for discount ladders.\n"
            "Create exactly 3 families with this schema:\n"
            f"{json.dumps(schema)}\n"
            "Use planner_context deeply (do not ignore it):\n"
            "- size_slab_coefficients gives per-slab model coefficients.\n"
            "- pack_margin_profile gives relative pricing/margin context by pack.\n"
            "- cross elasticities indicate cross-pack relationship direction/strength.\n"
            "How to interpret coefficients:\n"
            "- coef_base_discount_pct: own discount sensitivity for that slab (stronger absolute magnitude => stronger response).\n"
            "- coef_lag1_base_discount_pct: carryover/drag from prior month discount.\n"
            "- coef_other_slabs_weighted_base_discount_pct: coupling/cannibalization signal within same pack.\n"
            "Pack-aware decision rules (must follow):\n"
            "- If user asks 12-ML growth, move 12-ML ladders up first; do not rely mainly on 18-ML changes.\n"
            "- If user asks 18-ML growth, move 18-ML ladders up first.\n"
            "- If user asks profit/margin increase, avoid deep discounting (especially in 18-ML unless explicitly asked).\n"
            "- Respect pack-specific constraints mentioned in prompt (e.g., keep 18-ML shallow).\n"
            "- If prompt is ambiguous, choose balanced families and explicitly hedge with one conservative and one growth family.\n"
            "Hard constraints:\n"
            "- month_pattern must be one of: flat, up, down, wave, pulse.\n"
            "- base_min/base_max/gap_min/gap_max must define meaningful spread.\n"
            "- priority_weight and anchor_weights can be floats; they will be normalized.\n"
            "- focus on realistic business scenarios, but keep diversity across families.\n"
            f"Input:\n{json.dumps(prompt_payload)}"
        )

    def _step5_build_anchor_vectors(
        self,
        periods: List[str],
        defaults_matrix: Dict[str, Dict[str, List[float]]],
        slab_order_by_size: Dict[str, List[str]],
    ) -> Tuple[Dict[str, Dict[str, Dict[str, Dict[str, float]]]], Dict[str, Any]]:
        anchor_keys = ["latest_month", "last_3m_avg", "ly_same_3m", "stress_explore"]

        def _empty() -> Dict[str, Dict[str, Dict[str, float]]]:
            out: Dict[str, Dict[str, Dict[str, float]]] = {}
            for period in periods:
                pkey = str(period)
                out[pkey] = {}
                for size_key in ["12-ML", "18-ML"]:
                    out[pkey][size_key] = {}
                    for slab_key in slab_order_by_size.get(size_key, []):
                        out[pkey][size_key][slab_key] = float(10.0)
            return out

        anchors: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {k: _empty() for k in anchor_keys}
        flat_cells = 0
        total_cells = 0
        synthetic_diversification_applied = False

        for size_key in ["12-ML", "18-ML"]:
            slabs = slab_order_by_size.get(size_key, [])
            if not slabs:
                continue
            for slab_key in slabs:
                hist_vals: List[float] = []
                series = defaults_matrix.get(size_key, {}).get(slab_key, []) or []
                for idx in range(len(periods)):
                    if idx < len(series):
                        hist_vals.append(float(series[idx]))
                    else:
                        hist_vals.append(float(series[-1] if series else 10.0))
                if not hist_vals:
                    hist_vals = [10.0 for _ in periods]
                latest_val = float(hist_vals[-1])
                avg_val = float(sum(hist_vals) / max(1, len(hist_vals)))
                hist_span = float(max(hist_vals) - min(hist_vals)) if hist_vals else 0.0
                is_flat = hist_span < 1e-9
                total_cells += max(1, len(periods))
                if is_flat:
                    flat_cells += max(1, len(periods))

                avg_month_step = max(1.0, min(2.5, hist_span if hist_span > 0 else 1.5))
                stress_amp = max(2.0, min(6.0, (hist_span + 2.0)))
                size_bias = 0.8 if size_key == "12-ML" else -0.6
                slab_rank = slabs.index(slab_key)
                slab_bias = 0.4 if slab_rank >= 2 else 0.0

                for month_idx, period in enumerate(periods):
                    pkey = str(period)
                    avg_offsets = [-avg_month_step, 0.0, avg_month_step]
                    avg_offset = avg_offsets[min(month_idx, len(avg_offsets) - 1)]
                    seasonal_offsets = [1.5, -0.5, 1.0]
                    seasonal_offset = seasonal_offsets[min(month_idx, len(seasonal_offsets) - 1)]
                    stress_offsets = [-(stress_amp + 0.5), stress_amp * 0.5, stress_amp + 1.0]
                    stress_offset = stress_offsets[min(month_idx, len(stress_offsets) - 1)]

                    latest_anchor = latest_val
                    last3_anchor = avg_val + avg_offset
                    if is_flat:
                        ly_anchor = avg_val + seasonal_offset
                        synthetic_diversification_applied = True
                    else:
                        ly_anchor = hist_vals[min(month_idx, len(hist_vals) - 1)]
                    stress_anchor = avg_val + stress_offset + size_bias + slab_bias

                    anchors["latest_month"][pkey][size_key][slab_key] = float(self._clamp_discount_5_30(latest_anchor, latest_val))
                    anchors["last_3m_avg"][pkey][size_key][slab_key] = float(self._clamp_discount_5_30(last3_anchor, avg_val))
                    anchors["ly_same_3m"][pkey][size_key][slab_key] = float(self._clamp_discount_5_30(ly_anchor, avg_val))
                    anchors["stress_explore"][pkey][size_key][slab_key] = float(self._clamp_discount_5_30(stress_anchor, avg_val))

        for anchor_key in anchor_keys:
            for period in periods:
                pkey = str(period)
                for size_key in ["12-ML", "18-ML"]:
                    slabs = slab_order_by_size.get(size_key, [])
                    if not slabs:
                        continue
                    ladd = {slab: anchors[anchor_key][pkey][size_key].get(slab, 10.0) for slab in slabs}
                    anchors[anchor_key][pkey][size_key] = self._enforce_size_month_ladder(ladd, slabs)

        info = {
            "flat_defaults_cells": int(flat_cells),
            "total_anchor_cells": int(total_cells),
            "synthetic_diversification_applied": bool(synthetic_diversification_applied),
        }
        return anchors, info

    def _step5_choose_weighted_anchor(self, rng: random.Random, weights: Dict[str, float]) -> str:
        keys = ["latest_month", "last_3m_avg", "ly_same_3m", "stress_explore"]
        probs = self._step5_normalize_weights(weights or {}, keys)
        r = float(rng.random())
        cum = 0.0
        for key in keys:
            cum += float(probs.get(key, 0.0))
            if r <= cum:
                return key
        return keys[-1]

    def _step5_sample_from_family(
        self,
        family: Dict[str, Any],
        anchors: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
        rng: random.Random,
        periods: List[str],
        slab_order_by_size: Dict[str, List[str]],
    ) -> Tuple[Dict[str, Dict[str, Dict[str, float]]], str]:
        anchor_key = self._step5_choose_weighted_anchor(rng, family.get("anchor_weights", {}))
        anchor = anchors.get(anchor_key, anchors.get("last_3m_avg", {}))
        scenario: Dict[str, Dict[str, Dict[str, float]]] = {}

        for month_idx, period in enumerate(periods):
            pkey = str(period)
            scenario[pkey] = {}
            shift = self._step5_month_shift(str(family.get("month_pattern", "flat")), month_idx, int(family.get("month_shift_strength", 0)))
            for size_key in ["12-ML", "18-ML"]:
                slabs = slab_order_by_size.get(size_key, [])
                scenario[pkey][size_key] = {}
                if not slabs:
                    continue
                bias = int(family.get("size_bias_12", 0)) if size_key == "12-ML" else int(family.get("size_bias_18", 0))
                volatility = int(family.get("volatility", 0))
                noise = rng.randint(-volatility, volatility) if volatility > 0 else 0
                anchor_vals = [float(((anchor.get(pkey, {}).get(size_key, {}) or {}).get(slab, 10.0))) for slab in slabs]

                base_raw = anchor_vals[0] + bias + shift + noise + rng.randint(-1, 1)
                base_low = int(family.get("base_min", 10))
                base_high = int(family.get("base_max", 16))
                base = self._clamp_discount_5_30(max(base_low, min(base_high, base_raw)), 10.0)
                ladder = [float(base)]

                for i in range(1, len(slabs)):
                    anchor_gap = max(1, int(round(anchor_vals[i] - anchor_vals[i - 1])))
                    gap_raw = anchor_gap + rng.randint(-1, 1)
                    gap_min = int(family.get("gap_min", 1))
                    gap_max = int(family.get("gap_max", 3))
                    gap = max(gap_min, min(gap_max, gap_raw))
                    ladder.append(float(ladder[-1] + gap))

                mapped = {slabs[i]: float(ladder[i]) for i in range(len(slabs))}
                scenario[pkey][size_key] = self._enforce_size_month_ladder(mapped, slabs)
        return scenario, anchor_key

    def _step5_is_degenerate(
        self,
        scenario_map: Dict[str, Dict[str, Dict[str, float]]],
        periods: List[str],
        slab_order_by_size: Dict[str, List[str]],
    ) -> bool:
        values: List[int] = []
        month_signatures: List[Tuple[int, ...]] = []
        for period in periods:
            pkey = str(period)
            month_vals: List[int] = []
            for size_key in ["12-ML", "18-ML"]:
                for slab in slab_order_by_size.get(size_key, []):
                    raw_val = (scenario_map.get(pkey, {}).get(size_key, {}) or {}).get(slab, 0.0)
                    val = int(round(float(raw_val)))
                    values.append(val)
                    month_vals.append(val)
            month_signatures.append(tuple(month_vals))
        if not values:
            return True
        unique_vals = set(values)
        if len(unique_vals) <= 2:
            return True
        if max(values) <= 6:
            return True
        if (max(values) - min(values)) <= 2:
            return True
        if len(set(month_signatures)) <= 1:
            return True
        return False

    # Step 5 AI scenario generation now supports async jobs + polling.
    def _ensure_ai_job_store(self):
        if not hasattr(self, "_step5_ai_jobs"):
            self._step5_ai_jobs: Dict[str, Dict[str, Any]] = {}
        if not hasattr(self, "_step5_ai_job_lock"):
            self._step5_ai_job_lock = threading.RLock()

    def _persist_ai_job_snapshot(self, run_id: Optional[str]):
        if not run_id:
            return
        self._ensure_ai_job_store()
        try:
            with self._step5_ai_job_lock:
                summaries: Dict[str, Dict[str, Any]] = {}
                for job in self._step5_ai_jobs.values():
                    if str(job.get("run_id") or "") != str(run_id):
                        continue
                    summaries[str(job.get("job_id"))] = {
                        "job_id": str(job.get("job_id")),
                        "status": str(job.get("status") or "queued"),
                        "progress_current": int(job.get("progress_current") or 0),
                        "progress_total": int(job.get("progress_total") or 0),
                        "result_count": int(job.get("result_count") or 0),
                        "error_detail": job.get("error_detail"),
                        "created_at": str(job.get("created_at") or ""),
                        "updated_at": str(job.get("updated_at") or ""),
                    }
            self.save_run_state(
                run_id=run_id,
                state_update={"ui_state": {"step5_ai_jobs": summaries}},
            )
        except Exception:
            return

    def _create_ai_job(self, request: AIScenarioGenerateRequest) -> Dict[str, Any]:
        self._ensure_ai_job_store()
        run_id = str(getattr(request, "run_id", "") or "").strip()
        run_id = self.create_run(run_id if run_id else None)
        payload = (
            request.model_dump(exclude_none=True)
            if hasattr(request, "model_dump")
            else request.dict(exclude_none=True)
        )
        payload["run_id"] = run_id

        now_iso = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        job_id = str(uuid.uuid4())
        with self._step5_ai_job_lock:
            for job in self._step5_ai_jobs.values():
                if str(job.get("run_id") or "") != run_id:
                    continue
                if str(job.get("status") or "") in {"queued", "running"}:
                    job["cancel_requested"] = True
                    job["status"] = "cancelled"
                    job["error_detail"] = "Cancelled by newer AI generation job."
                    job["updated_at"] = now_iso

            job = {
                "job_id": job_id,
                "run_id": run_id,
                "status": "queued",
                "progress_current": 0,
                "progress_total": int(getattr(request, "scenario_count", 0) or 0),
                "result_count": 0,
                "error_detail": None,
                "request_payload": payload,
                "results": [],
                "cancel_requested": False,
                "created_at": now_iso,
                "updated_at": now_iso,
            }
            self._step5_ai_jobs[job_id] = job
        self._persist_ai_job_snapshot(run_id)
        return copy.deepcopy(job)

    def _update_ai_job_status(
        self,
        job_id: str,
        *,
        status: Optional[str] = None,
        progress_current: Optional[int] = None,
        progress_total: Optional[int] = None,
        result_count: Optional[int] = None,
        error_detail: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        self._ensure_ai_job_store()
        now_iso = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        with self._step5_ai_job_lock:
            job = self._step5_ai_jobs.get(str(job_id))
            if job is None:
                return None
            if status is not None:
                job["status"] = str(status)
            if progress_current is not None:
                job["progress_current"] = int(max(0, progress_current))
            if progress_total is not None:
                job["progress_total"] = int(max(0, progress_total))
            if result_count is not None:
                job["result_count"] = int(max(0, result_count))
            if error_detail is not None:
                job["error_detail"] = str(error_detail) if error_detail else None
            job["updated_at"] = now_iso
            snapshot = copy.deepcopy(job)
        self._persist_ai_job_snapshot(snapshot.get("run_id"))
        return snapshot

    def _append_ai_job_results(self, job_id: str, scenarios: List[AIScenarioRow]) -> Optional[Dict[str, Any]]:
        self._ensure_ai_job_store()
        now_iso = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        with self._step5_ai_job_lock:
            job = self._step5_ai_jobs.get(str(job_id))
            if job is None:
                return None
            normalized_rows: List[Dict[str, Any]] = []
            for row in (scenarios or []):
                if hasattr(row, "model_dump"):
                    normalized_rows.append(row.model_dump())
                elif isinstance(row, dict):
                    normalized_rows.append(copy.deepcopy(row))
            job["results"] = normalized_rows
            job["result_count"] = len(normalized_rows)
            job["updated_at"] = now_iso
            snapshot = copy.deepcopy(job)
        self._persist_ai_job_snapshot(snapshot.get("run_id"))
        return snapshot

    def _get_ai_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        self._ensure_ai_job_store()
        with self._step5_ai_job_lock:
            job = self._step5_ai_jobs.get(str(job_id))
            if job is None:
                return None
            return copy.deepcopy(job)

    def _list_ai_job_results(self, job_id: str, offset: int, limit: int) -> Optional[Dict[str, Any]]:
        self._ensure_ai_job_store()
        with self._step5_ai_job_lock:
            job = self._step5_ai_jobs.get(str(job_id))
            if job is None:
                return None
            total = len(job.get("results") or [])
            safe_offset = max(0, int(offset or 0))
            safe_limit = max(1, min(1000, int(limit or 200)))
            rows = (job.get("results") or [])[safe_offset:safe_offset + safe_limit]
            return {
                "job_id": str(job.get("job_id")),
                "status": str(job.get("status") or "queued"),
                "offset": safe_offset,
                "limit": safe_limit,
                "total_results": total,
                "rows": copy.deepcopy(rows),
            }

    def _is_ai_job_cancelled(self, job_id: str) -> bool:
        self._ensure_ai_job_store()
        with self._step5_ai_job_lock:
            job = self._step5_ai_jobs.get(str(job_id))
            if job is None:
                return True
            return bool(job.get("cancel_requested"))

    def _request_gemini_ai_scenarios(
        self,
        *,
        scenario_count: int,
        goal: str,
        user_prompt: str,
        periods: List[str],
        defaults_matrix: Dict[str, Dict[str, List[float]]],
        planner_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        api_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
        if not api_key:
            raise RuntimeError("AI generation disabled. Set GEMINI_API_KEY on backend.")

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
            planner_context=planner_context or {},
        )
        request_body = {
            "contents": [{"role": "user", "parts": [{"text": prompt_text}]}],
            "generationConfig": {
                "temperature": 0.35,
                "topP": 0.9,
                "maxOutputTokens": 9000,
                "responseMimeType": "application/json",
            },
        }
        last_error: Optional[str] = None
        retry_delays = [0.8, 1.5, 2.5]

        for attempt_idx in range(len(retry_delays) + 1):
            req_obj = urllib.request.Request(
                endpoint,
                data=json.dumps(request_body).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req_obj, timeout=75) as resp:
                    body = resp.read().decode("utf-8")
            except urllib.error.HTTPError as err:
                detail = ""
                try:
                    detail = err.read().decode("utf-8", errors="ignore")
                except Exception:
                    detail = str(err)
                last_error = f"Gemini HTTP {err.code}: {detail[:160]}"
                if attempt_idx < len(retry_delays):
                    time.sleep(retry_delays[attempt_idx])
                    continue
                raise RuntimeError(last_error) from err
            except Exception as err:
                last_error = f"Gemini unavailable ({str(err)})"
                if attempt_idx < len(retry_delays):
                    time.sleep(retry_delays[attempt_idx])
                    continue
                raise RuntimeError(last_error) from err

            try:
                payload = json.loads(body)
            except Exception:
                last_error = "Invalid Gemini JSON envelope"
                if attempt_idx < len(retry_delays):
                    time.sleep(retry_delays[attempt_idx])
                    continue
                raise RuntimeError(last_error)

            raw_parts: List[str] = []
            for candidate in (payload.get("candidates") or []):
                content = candidate.get("content") or {}
                for part in (content.get("parts") or []):
                    txt = part.get("text")
                    if isinstance(txt, str) and txt.strip():
                        raw_parts.append(txt.strip())

            families_payload: Optional[Dict[str, Any]] = None
            for part_text in raw_parts:
                parsed_any: Any = None
                try:
                    parsed_any = json.loads(part_text)
                except Exception:
                    parsed_any = self._extract_first_json_object(part_text)

                if isinstance(parsed_any, dict):
                    candidate_list = parsed_any.get("families")
                    if isinstance(candidate_list, list) and len(candidate_list) > 0:
                        families_payload = parsed_any
                        break

            if isinstance(families_payload, dict) and isinstance(families_payload.get("families"), list):
                return families_payload

            last_error = "Empty/invalid Gemini response"
            if attempt_idx < len(retry_delays):
                time.sleep(retry_delays[attempt_idx])
                continue
            raise RuntimeError(last_error)

        raise RuntimeError(last_error or "Empty/invalid Gemini response")

    def _scenario_signature(self, scenario_map: Dict[str, Dict[str, Dict[str, float]]]) -> str:
        parts: List[str] = []
        for period_key in sorted(list((scenario_map or {}).keys())):
            by_size = scenario_map.get(period_key) or {}
            for size_key in ["12-ML", "18-ML"]:
                by_slab = by_size.get(size_key) or {}
                slab_keys = sorted(list(by_slab.keys()), key=self._slab_sort_key)
                for slab_key in slab_keys:
                    v = float(by_slab.get(slab_key, 0.0))
                    parts.append(f"{period_key}|{size_key}|{slab_key}|{round(v, 2):.2f}")
        return "||".join(parts)

    def _sanitize_ai_scenarios(
        self,
        *,
        raw_scenarios: List[Any],
        periods: List[str],
        defaults_matrix: Dict[str, Dict[str, List[float]]],
        slab_order_by_size: Dict[str, List[str]],
        constraint_bounds_by_key: Optional[Dict[str, Dict[str, Optional[int]]]] = None,
        start_index: int = 1,
    ) -> List[AIScenarioRow]:
        month_index = {str(p): idx for idx, p in enumerate(periods)}
        constraint_bounds_by_key = constraint_bounds_by_key or {}

        def _default_for(period_key: str, size_key: str, slab_key: str) -> float:
            idx = month_index.get(period_key, 0)
            series = defaults_matrix.get(size_key, {}).get(slab_key, [])
            if idx < len(series):
                return float(series[idx])
            return 10.0

        out: List[AIScenarioRow] = []
        for raw_idx, source in enumerate(raw_scenarios):
            source_obj = source if isinstance(source, dict) else {}
            name = str(source_obj.get("name") or "").strip() or f"AI Scenario {start_index + raw_idx}"
            by_period_raw = source_obj.get("scenario_discounts_by_period") or {}
            scenario_map: Dict[str, Dict[str, Dict[str, float]]] = {}

            for period_key in periods:
                period_key = str(period_key)
                scenario_map[period_key] = {}
                period_obj = by_period_raw.get(period_key, {}) if isinstance(by_period_raw, dict) else {}
                for size_key in ["12-ML", "18-ML"]:
                    slab_values_in = period_obj.get(size_key, {}) if isinstance(period_obj, dict) else {}
                    slab_seed: Dict[str, float] = {}
                    bound_map: Dict[str, Dict[str, Optional[int]]] = {}
                    for slab_key in slab_order_by_size.get(size_key, []):
                        default_value = _default_for(period_key, size_key, slab_key)
                        raw_value = slab_values_in.get(slab_key) if isinstance(slab_values_in, dict) else None
                        slab_seed[slab_key] = float(self._clamp_discount_5_30_int(raw_value, default_value))
                        bound_map[slab_key] = constraint_bounds_by_key.get(
                            f"{period_key}|{size_key}|{slab_key}",
                            {'min': None, 'max': None},
                        )
                    bounded = self._enforce_size_month_ladder_int_with_bounds(
                        slab_seed,
                        slab_order_by_size.get(size_key, []),
                        bound_map,
                    )
                    if bounded is None:
                        scenario_map = {}
                        break
                    scenario_map[period_key][size_key] = bounded
                if not scenario_map:
                    break
            if not scenario_map:
                continue
            out.append(AIScenarioRow(name=name, scenario_discounts_by_period=scenario_map))
        return out

    async def _generate_ai_scenarios_rows(
        self,
        request: AIScenarioGenerateRequest,
        progress_callback=None,
        cancel_check=None,
    ) -> List[AIScenarioRow]:
        request_payload = (
            request.model_dump(exclude_none=True)
            if hasattr(request, "model_dump")
            else request.dict(exclude_none=True)
        )
        for key in ["scenario_count", "goal", "prompt", "discount_constraints", "metric_thresholds"]:
            request_payload.pop(key, None)

        base_request = CrossSizePlannerRequest(**request_payload)
        planner_base = await self.calculate_cross_size_planner(base_request)
        if not planner_base.success:
            raise RuntimeError(planner_base.message or "Failed to initialize planner base for AI scenarios.")

        periods = [str(p) for p in (planner_base.periods or [])]
        defaults_matrix = planner_base.defaults_matrix or {}
        if not periods or not defaults_matrix:
            raise RuntimeError("Planner base is missing periods/default discount matrix.")

        slab_order_by_size: Dict[str, List[str]] = {
            size_key: sorted(list((defaults_matrix.get(size_key) or {}).keys()), key=self._slab_sort_key)
            for size_key in ["12-ML", "18-ML"]
        }
        discount_constraint_bounds = self._parse_step5_discount_constraints(
            request=request,
            periods=periods,
            slab_order_by_size=slab_order_by_size,
        )
        target_count = int(getattr(request, "scenario_count", 5) or 5)
        goal = str(getattr(request, "goal", "") or "").strip()
        user_prompt = str(getattr(request, "prompt", "") or "").strip()

        if progress_callback:
            progress_callback(0, target_count, 0)

        accepted: List[AIScenarioRow] = []
        seen_signatures: set = set()

        planner_context = {
            "cross_elasticity_12_from_18": float(planner_base.cross_elasticity_12_from_18 or 0.0),
            "cross_elasticity_18_from_12": float(planner_base.cross_elasticity_18_from_12 or 0.0),
            "cross_model_r2_12": float(planner_base.cross_model_r2_12 or 0.0),
            "cross_model_r2_18": float(planner_base.cross_model_r2_18 or 0.0),
            "model_build_notes": (
                "There is a negative relationship between 12-ML and 18-ML. "
                "Planner volume is built slab-wise from beta*own_discount + beta*lag_discount + beta*other_slabs_weighted_discount, "
                "then slabs are summed to pack totals. Cross adjustment is applied on pack changes."
            ),
        }
        size_slab_coefficients: Dict[str, Dict[str, Dict[str, float]]] = {}
        pack_margin_profile: Dict[str, Dict[str, float]] = {}
        for size_payload in (getattr(planner_base, "size_results", None) or []):
            size_key = str(getattr(size_payload, "size", "") or "").strip()
            if not size_key:
                continue
            slab_map: Dict[str, Dict[str, float]] = {}
            weighted_price_num = 0.0
            weighted_cogs_num = 0.0
            weighted_den = 0.0
            for slab_state in (getattr(size_payload, "slabs", None) or []):
                slab_key = str(getattr(slab_state, "slab", "") or "").strip()
                if not slab_key:
                    continue
                anchor_qty = float(getattr(slab_state, "anchor_qty", 0.0) or 0.0)
                base_price = float(getattr(slab_state, "base_price", 0.0) or 0.0)
                cogs_per_unit = float(getattr(slab_state, "cogs_per_unit", 0.0) or 0.0)
                coef_base = float(getattr(slab_state, "coef_base_discount_pct", 0.0) or 0.0)
                coef_lag = float(getattr(slab_state, "coef_lag1_base_discount_pct", 0.0) or 0.0)
                coef_other = float(getattr(slab_state, "coef_other_slabs_weighted_base_discount_pct", 0.0) or 0.0)
                default_disc = float(getattr(slab_state, "default_discount_pct", 0.0) or 0.0)
                slab_map[slab_key] = {
                    "default_discount_pct": float(round(default_disc, 4)),
                    "anchor_qty": float(round(anchor_qty, 4)),
                    "base_price": float(round(base_price, 4)),
                    "cogs_per_unit": float(round(cogs_per_unit, 4)),
                    "coef_base_discount_pct": float(round(coef_base, 6)),
                    "coef_lag1_base_discount_pct": float(round(coef_lag, 6)),
                    "coef_other_slabs_weighted_base_discount_pct": float(round(coef_other, 6)),
                    "coef_signal": float(round(abs(coef_base) + abs(coef_lag) + abs(coef_other), 6)),
                }
                w = max(anchor_qty, 0.0)
                weighted_price_num += w * base_price
                weighted_cogs_num += w * cogs_per_unit
                weighted_den += w
            if slab_map:
                size_slab_coefficients[size_key] = slab_map
            if weighted_den > 0:
                avg_price = weighted_price_num / weighted_den
                avg_cogs = weighted_cogs_num / weighted_den
            else:
                avg_price = 0.0
                avg_cogs = 0.0
            pack_margin_profile[size_key] = {
                "weighted_avg_base_price": float(round(avg_price, 6)),
                "weighted_avg_cogs_per_unit": float(round(avg_cogs, 6)),
                "weighted_avg_unit_margin": float(round(avg_price - avg_cogs, 6)),
            }
        planner_context["size_slab_coefficients"] = size_slab_coefficients
        planner_context["pack_margin_profile"] = pack_margin_profile
        planner_context["ai_intent_usage_notes"] = [
            "Read user prompt and map objective by pack before creating families.",
            "For pack-specific growth asks, move that pack primarily; avoid accidental opposite-pack domination.",
            "Use slab coefficients to prioritize slabs with stronger modeled response.",
            "For margin/profit asks, use shallower ladders where margin profile is stronger unless prompt says otherwise.",
        ]
        planner_context["hard_discount_constraints"] = [
            {
                "period": str(k).split("|")[0],
                "size": str(k).split("|")[1],
                "slab": str(k).split("|")[2],
                "min": v.get("min"),
                "max": v.get("max"),
            }
            for k, v in (discount_constraint_bounds or {}).items()
        ]
        families_payload = self._request_gemini_ai_scenarios(
            scenario_count=target_count,
            goal=goal,
            user_prompt=user_prompt,
            periods=periods,
            defaults_matrix=defaults_matrix,
            planner_context=planner_context,
        )
        families, _sanitize_info = self._step5_sanitize_families(families_payload)
        allocation = self._step5_allocate_count_by_family(
            total=target_count,
            family_weights=[float(f.get("priority_weight", 0.0)) for f in families],
        )
        anchors, _anchor_info = self._step5_build_anchor_vectors(
            periods=periods,
            defaults_matrix=defaults_matrix,
            slab_order_by_size=slab_order_by_size,
        )

        seed_input = json.dumps(
            {
                "prompt": user_prompt,
                "goal": goal,
                "planner_context": planner_context,
                "families": families,
                "target": target_count,
                "periods": periods,
            },
            sort_keys=True,
        )
        seed_base = int(hashlib.sha256(seed_input.encode("utf-8")).hexdigest()[:12], 16) % (10**9)

        for fam_idx, family in enumerate(families):
            target = int(allocation[fam_idx]) if fam_idx < len(allocation) else 0
            if target <= 0:
                continue
            rng = random.Random(seed_base + (fam_idx + 1) * 104729)
            attempts = 0
            max_attempts = max(200, target * 100)
            accepted_in_family = 0

            while accepted_in_family < target and attempts < max_attempts:
                if cancel_check and bool(cancel_check()):
                    raise _AIJobCancelled("Cancelled by newer job")
                attempts += 1

                sampled_map, anchor_used = self._step5_sample_from_family(
                    family=family,
                    anchors=anchors,
                    rng=rng,
                    periods=periods,
                    slab_order_by_size=slab_order_by_size,
                )
                sanitized_rows = self._sanitize_ai_scenarios(
                    raw_scenarios=[{
                        "name": f"AI {len(accepted) + 1:03d} - {family.get('name', 'Family')} ({anchor_used})",
                        "scenario_discounts_by_period": sampled_map,
                    }],
                    periods=periods,
                    defaults_matrix=defaults_matrix,
                    slab_order_by_size=slab_order_by_size,
                    constraint_bounds_by_key=discount_constraint_bounds,
                    start_index=len(accepted) + 1,
                )
                if not sanitized_rows:
                    continue
                row = sanitized_rows[0]
                if self._step5_is_degenerate(row.scenario_discounts_by_period or {}, periods, slab_order_by_size):
                    continue
                signature = self._scenario_signature(row.scenario_discounts_by_period or {})
                if signature in seen_signatures:
                    continue
                seen_signatures.add(signature)
                accepted.append(row)
                accepted_in_family += 1
                if progress_callback:
                    progress_callback(len(accepted), target_count, len(accepted))

        if len(accepted) < target_count:
            extra_attempts = 0
            extra_max = max(1000, (target_count - len(accepted)) * 250)
            rr_idx = 0
            while len(accepted) < target_count and extra_attempts < extra_max:
                if cancel_check and bool(cancel_check()):
                    raise _AIJobCancelled("Cancelled by newer job")
                fam_idx = rr_idx % len(families)
                family = families[fam_idx]
                rng = random.Random(seed_base + 700001 + extra_attempts * 37)
                extra_attempts += 1
                rr_idx += 1
                sampled_map, anchor_used = self._step5_sample_from_family(
                    family=family,
                    anchors=anchors,
                    rng=rng,
                    periods=periods,
                    slab_order_by_size=slab_order_by_size,
                )
                sanitized_rows = self._sanitize_ai_scenarios(
                    raw_scenarios=[{
                        "name": f"AI {len(accepted) + 1:03d} - {family.get('name', 'Family')} ({anchor_used})",
                        "scenario_discounts_by_period": sampled_map,
                    }],
                    periods=periods,
                    defaults_matrix=defaults_matrix,
                    slab_order_by_size=slab_order_by_size,
                    constraint_bounds_by_key=discount_constraint_bounds,
                    start_index=len(accepted) + 1,
                )
                if not sanitized_rows:
                    continue
                row = sanitized_rows[0]
                if self._step5_is_degenerate(row.scenario_discounts_by_period or {}, periods, slab_order_by_size):
                    continue
                signature = self._scenario_signature(row.scenario_discounts_by_period or {})
                if signature in seen_signatures:
                    continue
                seen_signatures.add(signature)
                accepted.append(row)
                if progress_callback:
                    progress_callback(len(accepted), target_count, len(accepted))

        if len(accepted) < target_count:
            raise RuntimeError(
                f"Generated {len(accepted)} unique scenario(s) out of requested {target_count}. "
                "Try lowering scenario count or using a more specific prompt."
            )
        return accepted[:target_count]

    def _run_ai_job_worker(self, job_id: str):
        try:
            job = self._get_ai_job(job_id)
            if not job:
                return
            request_payload = copy.deepcopy(job.get("request_payload") or {})
            request = AIScenarioGenerateRequest(**request_payload)
            self._update_ai_job_status(
                job_id,
                status="running",
                progress_current=0,
                progress_total=int(getattr(request, "scenario_count", 0) or 0),
                result_count=0,
                error_detail=None,
            )

            def _progress(curr: int, total: int, result_count: int):
                if self._is_ai_job_cancelled(job_id):
                    raise _AIJobCancelled("Cancelled by newer job")
                self._update_ai_job_status(
                    job_id,
                    status="running",
                    progress_current=int(curr),
                    progress_total=int(total),
                    result_count=int(result_count),
                )

            rows = asyncio.run(
                self._generate_ai_scenarios_rows(
                    request=request,
                    progress_callback=_progress,
                    cancel_check=lambda: self._is_ai_job_cancelled(job_id),
                )
            )
            if self._is_ai_job_cancelled(job_id):
                self._update_ai_job_status(
                    job_id,
                    status="cancelled",
                    error_detail="Cancelled by newer AI generation job.",
                )
                return
            self._append_ai_job_results(job_id, rows)
            self._update_ai_job_status(
                job_id,
                status="completed",
                progress_current=len(rows),
                progress_total=len(rows),
                result_count=len(rows),
                error_detail=None,
            )
        except _AIJobCancelled as err:
            self._update_ai_job_status(job_id, status="cancelled", error_detail=str(err))
        except Exception as err:
            self._update_ai_job_status(job_id, status="failed", error_detail=str(err))

    async def create_ai_scenario_job(self, request: AIScenarioGenerateRequest) -> AIScenarioJobCreateResponse:
        try:
            job = self._create_ai_job(request)
            worker = threading.Thread(
                target=self._run_ai_job_worker,
                args=(str(job.get("job_id")),),
                daemon=True,
                name=f"step5-ai-job-{job.get('job_id')}",
            )
            worker.start()
            return AIScenarioJobCreateResponse(
                success=True,
                message="AI scenario generation job started.",
                job_id=str(job.get("job_id")),
                status="queued",
            )
        except Exception as err:
            return AIScenarioJobCreateResponse(
                success=False,
                message=f"Failed to start AI scenario generation job: {str(err)}",
                job_id="",
                status="failed",
            )

    def get_ai_scenario_job_status(self, job_id: str) -> Optional[AIScenarioJobStatusResponse]:
        job = self._get_ai_job(job_id)
        if not job:
            return None
        status = str(job.get("status") or "queued")
        return AIScenarioJobStatusResponse(
            success=status in {"queued", "running", "completed"},
            message=f"AI scenario job is {status}.",
            job_id=str(job.get("job_id")),
            status=status,
            progress_current=int(job.get("progress_current") or 0),
            progress_total=int(job.get("progress_total") or 0),
            result_count=int(job.get("result_count") or 0),
            error_detail=job.get("error_detail"),
        )

    def get_ai_scenario_job_results(self, job_id: str, offset: int = 0, limit: int = 200) -> Optional[AIScenarioJobResultsResponse]:
        snapshot = self._list_ai_job_results(job_id=job_id, offset=offset, limit=limit)
        if snapshot is None:
            return None
        rows: List[AIScenarioRow] = []
        for row in (snapshot.get("rows") or []):
            try:
                rows.append(AIScenarioRow(**(row or {})))
            except Exception:
                continue
        return AIScenarioJobResultsResponse(
            success=True,
            message="AI scenario job results loaded.",
            job_id=str(snapshot.get("job_id")),
            status=str(snapshot.get("status") or "queued"),
            offset=int(snapshot.get("offset") or 0),
            limit=int(snapshot.get("limit") or 200),
            total_results=int(snapshot.get("total_results") or 0),
            scenarios=rows,
        )

    async def generate_ai_scenarios(self, request: AIScenarioGenerateRequest) -> AIScenarioGenerateResponse:
        try:
            rows = await self._generate_ai_scenarios_rows(request=request)
            return AIScenarioGenerateResponse(
                success=True,
                message=f"Generated {len(rows)} AI scenario(s).",
                scenarios=rows,
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
                    'reference_investment': 0.0,
                    'reference_available': 0.0,
                }
                continue

            qty = float(size_part['Quantity_Num'].sum())
            revenue = float(size_part['Net_Revenue'].sum())
            investment = float(size_part['Discount_Num'].sum())
            cogs = float(pair_state.get(size_key, {}).get('cogs_per_unit', 0.0))
            profit = float(revenue - (qty * cogs))
            out[size_key] = {
                'reference_qty': qty,
                'reference_revenue': revenue,
                'reference_profit': profit,
                'reference_investment': investment,
                'reference_available': 1.0,
            }

        total_qty = float(sum(out.get(size, {}).get('reference_qty', 0.0) for size in ['12-ML', '18-ML']))
        total_revenue = float(sum(out.get(size, {}).get('reference_revenue', 0.0) for size in ['12-ML', '18-ML']))
        total_profit = float(sum(out.get(size, {}).get('reference_profit', 0.0) for size in ['12-ML', '18-ML']))
        total_investment = float(sum(out.get(size, {}).get('reference_investment', 0.0) for size in ['12-ML', '18-ML']))
        total_available = 1.0 if any(out.get(size, {}).get('reference_available', 0.0) > 0 for size in ['12-ML', '18-ML']) else 0.0
        out['TOTAL'] = {
            'reference_qty': total_qty,
            'reference_revenue': total_revenue,
            'reference_profit': total_profit,
            'reference_investment': total_investment,
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
                '12-ML': {
                    'baseline_qty': 0.0,
                    'pre_cross_qty': 0.0,
                    'final_qty': 0.0,
                    'baseline_revenue': 0.0,
                    'scenario_revenue': 0.0,
                    'baseline_profit': 0.0,
                    'scenario_profit': 0.0,
                    'baseline_investment': 0.0,
                    'scenario_investment': 0.0,
                    'scenario_investment_positive': 0.0,
                },
                '18-ML': {
                    'baseline_qty': 0.0,
                    'pre_cross_qty': 0.0,
                    'final_qty': 0.0,
                    'baseline_revenue': 0.0,
                    'scenario_revenue': 0.0,
                    'baseline_profit': 0.0,
                    'scenario_profit': 0.0,
                    'baseline_investment': 0.0,
                    'scenario_investment': 0.0,
                    'scenario_investment_positive': 0.0,
                },
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
                    baseline_investment_total = 0.0
                    scenario_investment_total = 0.0
                    scenario_investment_positive_total = 0.0
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
                        baseline_investment = baseline_qty * base_price * (default_discount / 100.0)
                        scenario_investment = final_qty * base_price * (scenario_discount / 100.0)
                        scenario_investment_positive = final_qty * base_price * (max(0.0, scenario_discount - default_discount) / 100.0)

                        slab_row['baseline_revenue'] = float(baseline_revenue)
                        slab_row['scenario_revenue'] = float(scenario_revenue)
                        slab_row['baseline_profit'] = float(baseline_profit)
                        slab_row['scenario_profit'] = float(scenario_profit)
                        slab_row['baseline_investment'] = float(baseline_investment)
                        slab_row['scenario_investment'] = float(scenario_investment)
                        slab_row['scenario_investment_positive'] = float(scenario_investment_positive)

                        baseline_revenue_total += baseline_revenue
                        scenario_revenue_total += scenario_revenue
                        baseline_profit_total += baseline_profit
                        scenario_profit_total += scenario_profit
                        baseline_investment_total += baseline_investment
                        scenario_investment_total += scenario_investment
                        scenario_investment_positive_total += scenario_investment_positive

                    size_payload['final_total_qty'] = float(final_total_qty)
                    size_payload['volume_delta_pct'] = ((final_total_qty - baseline_total_qty_non_discount) / baseline_total_qty_non_discount * 100.0) if baseline_total_qty_non_discount > 0 else 0.0
                    size_payload['baseline_revenue_total'] = float(baseline_revenue_total)
                    size_payload['scenario_revenue_total'] = float(scenario_revenue_total)
                    size_payload['revenue_delta_pct'] = ((scenario_revenue_total - baseline_revenue_total) / baseline_revenue_total * 100.0) if baseline_revenue_total > 0 else 0.0
                    size_payload['baseline_profit_total'] = float(baseline_profit_total)
                    size_payload['scenario_profit_total'] = float(scenario_profit_total)
                    size_payload['profit_delta_pct'] = ((scenario_profit_total - baseline_profit_total) / abs(baseline_profit_total) * 100.0) if abs(baseline_profit_total) > 1e-9 else 0.0
                    size_payload['baseline_investment_total'] = float(baseline_investment_total)
                    size_payload['scenario_investment_total'] = float(scenario_investment_total)
                    size_payload['scenario_investment_positive_total'] = float(scenario_investment_positive_total)
                    size_payload['investment_delta_pct'] = ((scenario_investment_total - baseline_investment_total) / baseline_investment_total * 100.0) if baseline_investment_total > 0 else 0.0

                    summary_acc[size_key]['baseline_qty'] += baseline_total_qty_non_discount
                    summary_acc[size_key]['pre_cross_qty'] += pre_cross_total_qty
                    summary_acc[size_key]['final_qty'] += final_total_qty
                    summary_acc[size_key]['baseline_revenue'] += baseline_revenue_total
                    summary_acc[size_key]['scenario_revenue'] += scenario_revenue_total
                    summary_acc[size_key]['baseline_profit'] += baseline_profit_total
                    summary_acc[size_key]['scenario_profit'] += scenario_profit_total
                    summary_acc[size_key]['baseline_investment'] += baseline_investment_total
                    summary_acc[size_key]['scenario_investment'] += scenario_investment_total
                    summary_acc[size_key]['scenario_investment_positive'] += scenario_investment_positive_total

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
                baseline_investment = float(agg.get('baseline_investment', 0.0))
                scenario_investment = float(agg.get('scenario_investment', 0.0))
                scenario_investment_positive = float(agg.get('scenario_investment_positive', 0.0))
                reference_qty = float(reference_3m.get(size_key, {}).get('reference_qty', 0.0))
                reference_revenue = float(reference_3m.get(size_key, {}).get('reference_revenue', 0.0))
                reference_profit = float(reference_3m.get(size_key, {}).get('reference_profit', 0.0))
                reference_investment = float(reference_3m.get(size_key, {}).get('reference_investment', 0.0))
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
                    'baseline_investment': baseline_investment,
                    'scenario_investment': scenario_investment,
                    'investment_change_positive': scenario_investment_positive,
                    'investment_delta_pct': ((scenario_investment - baseline_investment) / baseline_investment * 100.0) if baseline_investment > 0 else 0.0,
                    'reference_qty': reference_qty,
                    'reference_revenue': reference_revenue,
                    'reference_profit': reference_profit,
                    'reference_investment': reference_investment,
                    'own_delta_vs_reference_pct': ((pre_cross_qty - reference_qty) / reference_qty * 100.0) if reference_qty > 0 else 0.0,
                    'adjusted_delta_vs_reference_pct': ((final_qty - reference_qty) / reference_qty * 100.0) if reference_qty > 0 else 0.0,
                    'vs_reference_volume_pct': ((final_qty - reference_qty) / reference_qty * 100.0) if reference_qty > 0 else 0.0,
                    'vs_reference_revenue_pct': ((scenario_revenue - reference_revenue) / reference_revenue * 100.0) if reference_revenue > 0 else 0.0,
                    'vs_reference_profit_pct': ((scenario_profit - reference_profit) / abs(reference_profit) * 100.0) if abs(reference_profit) > 1e-9 else 0.0,
                    'vs_reference_investment_pct': ((scenario_investment - reference_investment) / reference_investment * 100.0) if reference_investment > 0 else 0.0,
                    'investment_change_positive_vs_reference_pct': (
                        ((scenario_investment_positive - reference_investment) / reference_investment * 100.0)
                        if reference_investment > 0 else 0.0
                    ),
                    'reference_available': reference_available,
                }

            total_baseline_qty = float(sum(summary_3m.get(size, {}).get('baseline_qty', 0.0) for size in ['12-ML', '18-ML']))
            total_pre_cross_qty = float(sum(summary_3m.get(size, {}).get('scenario_qty_additive', 0.0) for size in ['12-ML', '18-ML']))
            total_final_qty = float(sum(summary_3m.get(size, {}).get('final_qty', 0.0) for size in ['12-ML', '18-ML']))
            total_baseline_revenue = float(sum(summary_3m.get(size, {}).get('baseline_revenue', 0.0) for size in ['12-ML', '18-ML']))
            total_scenario_revenue = float(sum(summary_3m.get(size, {}).get('scenario_revenue', 0.0) for size in ['12-ML', '18-ML']))
            total_baseline_profit = float(sum(summary_3m.get(size, {}).get('baseline_profit', 0.0) for size in ['12-ML', '18-ML']))
            total_scenario_profit = float(sum(summary_3m.get(size, {}).get('scenario_profit', 0.0) for size in ['12-ML', '18-ML']))
            total_baseline_investment = float(sum(summary_3m.get(size, {}).get('baseline_investment', 0.0) for size in ['12-ML', '18-ML']))
            total_scenario_investment = float(sum(summary_3m.get(size, {}).get('scenario_investment', 0.0) for size in ['12-ML', '18-ML']))
            total_scenario_investment_positive = float(sum(summary_3m.get(size, {}).get('investment_change_positive', 0.0) for size in ['12-ML', '18-ML']))
            reference_total_qty = float(reference_3m.get('TOTAL', {}).get('reference_qty', 0.0))
            reference_total_revenue = float(reference_3m.get('TOTAL', {}).get('reference_revenue', 0.0))
            reference_total_profit = float(reference_3m.get('TOTAL', {}).get('reference_profit', 0.0))
            reference_total_investment = float(reference_3m.get('TOTAL', {}).get('reference_investment', 0.0))
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
                'baseline_investment': total_baseline_investment,
                'scenario_investment': total_scenario_investment,
                'investment_change_positive': total_scenario_investment_positive,
                'investment_delta_pct': ((total_scenario_investment - total_baseline_investment) / total_baseline_investment * 100.0) if total_baseline_investment > 0 else 0.0,
                'reference_qty': reference_total_qty,
                'reference_revenue': reference_total_revenue,
                'reference_profit': reference_total_profit,
                'reference_investment': reference_total_investment,
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
                'vs_reference_investment_pct': (
                    ((total_scenario_investment - reference_total_investment) / reference_total_investment * 100.0)
                    if reference_total_investment > 0 else 0.0
                ),
                'investment_change_positive_vs_reference_pct': (
                    ((total_scenario_investment_positive - reference_total_investment) / reference_total_investment * 100.0)
                    if reference_total_investment > 0 else 0.0
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

