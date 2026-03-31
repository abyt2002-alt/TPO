import json
import os
import random
import re
import hashlib
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st

MONTHS = ["Dec", "Jan", "Feb"]
SIZES = {
    "12-ML": ["slab1", "slab2"],
    "18-ML": ["slab1", "slab2", "slab3", "slab4"],
}
DEFAULTS = {
    "Dec": {
        "12-ML": {"slab1": 14.0, "slab2": 21.0},
        "18-ML": {"slab1": 11.5, "slab2": 15.5, "slab3": 16.5, "slab4": 17.0},
    },
    "Jan": {
        "12-ML": {"slab1": 14.0, "slab2": 21.0},
        "18-ML": {"slab1": 11.5, "slab2": 15.5, "slab3": 16.5, "slab4": 17.0},
    },
    "Feb": {
        "12-ML": {"slab1": 14.0, "slab2": 21.0},
        "18-ML": {"slab1": 11.5, "slab2": 15.5, "slab3": 16.5, "slab4": 17.0},
    },
}
ALLOWED_PATTERNS = {"flat", "up", "down", "wave", "pulse"}
ANCHOR_KEYS = ["latest_month", "last_3m_avg", "ly_same_3m", "stress_explore"]


def clamp_discount(value: float) -> int:
    return max(1, min(30, int(round(float(value)))))


def _clamp_int(value: Any, lo: int, hi: int, default: int) -> int:
    try:
        val = int(round(float(value)))
    except Exception:
        val = default
    return max(lo, min(hi, val))


def enforce_ladder(values: List[float]) -> Tuple[List[int], int, int]:
    fixed: List[int] = []
    range_or_rounding_fixes = 0
    ladder_fixes = 0
    for i, v in enumerate(values):
        raw = float(v)
        rounded = int(round(raw))
        vv = max(1, min(30, rounded))
        if rounded != vv or abs(raw - rounded) > 1e-9:
            range_or_rounding_fixes += 1
        if i > 0 and vv < fixed[i - 1]:
            vv = fixed[i - 1]
            ladder_fixes += 1
        fixed.append(vv)
    return fixed, range_or_rounding_fixes, ladder_fixes


def _normalize_weights(weights: Dict[str, Any], keys: List[str], fallback: float = 1.0) -> Dict[str, float]:
    parsed: Dict[str, float] = {}
    for k in keys:
        try:
            v = float(weights.get(k, fallback))
        except Exception:
            v = fallback
        parsed[k] = max(0.0, v)
    total = sum(parsed.values())
    if total <= 0:
        eq = 1.0 / max(1, len(keys))
        return {k: eq for k in keys}
    return {k: parsed[k] / total for k in keys}


def build_empty_scenario() -> Dict[str, Any]:
    return {
        month: {
            size: {slab: DEFAULTS[month][size][slab] for slab in slabs}
            for size, slabs in SIZES.items()
        }
        for month in MONTHS
    }


def normalize_scenario(item: Dict[str, Any], index: int) -> Dict[str, Any]:
    months_data = item.get("months", {})
    out = build_empty_scenario()
    fixes = {
        "missing_filled": 0,
        "range_or_rounding_fixes": 0,
        "ladder_fixes": 0,
    }

    for month in MONTHS:
        for size, slabs in SIZES.items():
            incoming = months_data.get(month, {}).get(size, {})
            ladder_values = []
            for slab in slabs:
                if slab not in incoming:
                    fixes["missing_filled"] += 1
                raw = incoming.get(slab, out[month][size][slab])
                ladder_values.append(raw)
            fixed, rr_fix, ladder_fix = enforce_ladder(ladder_values)
            fixes["range_or_rounding_fixes"] += rr_fix
            fixes["ladder_fixes"] += ladder_fix
            for slab, val in zip(slabs, fixed):
                out[month][size][slab] = val

    result = {
        "name": item.get("name") or f"SCN-{index + 1:03d}",
        "months": out,
        "corrections": fixes,
    }
    if item.get("family_name"):
        result["family_name"] = str(item.get("family_name"))
    if item.get("anchor_used"):
        result["anchor_used"] = str(item.get("anchor_used"))
    return result


def extract_json_object(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    match = re.search(r"\{.*\}", cleaned, flags=re.S)
    if not match:
        raise ValueError("No JSON object found in model response")
    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("Model response is not a JSON object")
    return parsed

def _gemini_families_prompt(user_prompt: str, planner_context: Dict[str, Any]) -> str:
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

    return (
        "Return ONE JSON object only (no markdown, no code fences).\n"
        "You are generating scenario families for discount ladders.\n"
        "Create exactly 3 families with this schema:\n"
        f"{json.dumps(schema)}\n"
        "Hard constraints:\n"
        "- Integer-like controls preferred.\n"
        "- Discounts are later repaired to [1,30].\n"
        "- Slab ladders are repaired non-decreasing.\n"
        "- Avoid degenerate families (do NOT collapse all months/slabs to same tiny values like all 1s).\n"
        "- Ensure meaningful spread: base_max should be above base_min and gaps should create ladder separation.\n"
        "- month_pattern must be one of: flat, up, down, wave, pulse.\n"
        "- priority_weight and anchor_weights can be floats; they will be normalized.\n"
        f"Default ladder reference: {json.dumps(DEFAULTS)}\n"
        f"Planner/model context: {json.dumps(planner_context)}\n"
        f"Business goal: {user_prompt}\n"
        "Focus on realistic, diverse families suitable for business planner testing."
    )


def _generate_families_with_gemini(
    session: requests.Session,
    api_key: str,
    user_prompt: str,
    planner_context: Dict[str, Any],
    temperature: float = 0.4,
    max_attempts: int = 4,
    stage_callback: Optional[Callable[[str], None]] = None,
) -> Tuple[Dict[str, Any], str]:
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.5-flash:generateContent?key={api_key}"
    )
    payload = {
        "contents": [{"parts": [{"text": _gemini_families_prompt(user_prompt, planner_context)}]}],
        "generationConfig": {
            "temperature": float(temperature),
            "responseMimeType": "application/json",
        },
    }

    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            if stage_callback:
                stage_callback(f"Gemini thinking... attempt {attempt}/{max_attempts}")
            resp = session.post(url, json=payload, timeout=(10, 180))
            resp.raise_for_status()
            data = resp.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            parsed = extract_json_object(text)
            if stage_callback:
                stage_callback("Gemini families received")
            return parsed, text
        except Exception as exc:
            last_error = exc
            if attempt < max_attempts:
                time.sleep(min(2 ** (attempt - 1), 8))

    raise RuntimeError(f"Gemini family generation failed after retries: {last_error}")


def _default_family(index: int) -> Dict[str, Any]:
    templates = [
        {
            "name": "Balanced realistic",
            "priority_weight": 0.4,
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
                "latest_month": 0.4,
                "last_3m_avg": 0.4,
                "ly_same_3m": 0.1,
                "stress_explore": 0.1,
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
                "latest_month": 0.3,
                "last_3m_avg": 0.2,
                "ly_same_3m": 0.1,
                "stress_explore": 0.4,
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
                "latest_month": 0.2,
                "last_3m_avg": 0.5,
                "ly_same_3m": 0.2,
                "stress_explore": 0.1,
            },
        },
    ]
    return templates[index % len(templates)]


def _sanitize_family(item: Dict[str, Any], idx: int) -> Dict[str, Any]:
    raw = dict(item or {})
    pattern = str(raw.get("month_pattern", "flat")).lower().strip()
    if pattern not in ALLOWED_PATTERNS:
        pattern = "flat"

    base_min = _clamp_int(raw.get("base_min", 10), 1, 30, 10)
    base_max = _clamp_int(raw.get("base_max", 18), 1, 30, 18)
    if base_max < base_min:
        base_min, base_max = base_max, base_min

    gap_min = _clamp_int(raw.get("gap_min", 1), 1, 10, 1)
    gap_max = _clamp_int(raw.get("gap_max", 3), 1, 10, 3)
    if gap_max < gap_min:
        gap_min, gap_max = gap_max, gap_min

    # Keep enough range so Monte Carlo does not collapse into flat/degenerate ladders.
    if base_max - base_min < 2:
        base_max = min(30, base_min + 2)
    if gap_max - gap_min < 1:
        gap_max = min(10, gap_min + 1)

    return {
        "name": str(raw.get("name") or f"Family {idx + 1}"),
        "priority_weight": max(0.0, float(raw.get("priority_weight", 1.0))),
        "base_min": base_min,
        "base_max": base_max,
        "gap_min": gap_min,
        "gap_max": gap_max,
        "month_pattern": pattern,
        "month_shift_strength": _clamp_int(raw.get("month_shift_strength", 2), 0, 6, 2),
        "size_bias_12": _clamp_int(raw.get("size_bias_12", 0), -6, 6, 0),
        "size_bias_18": _clamp_int(raw.get("size_bias_18", 0), -6, 6, 0),
        "volatility": _clamp_int(raw.get("volatility", 1), 0, 6, 1),
        "anchor_weights": _normalize_weights(raw.get("anchor_weights", {}), ANCHOR_KEYS),
    }


def _family_similarity_score(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    """Return 0..1 similarity score where 1 is near-identical."""
    pattern_score = 1.0 if a.get("month_pattern") == b.get("month_pattern") else 0.0
    numeric_parts = []
    numeric_parts.append(max(0.0, 1.0 - (abs(a["base_min"] - b["base_min"]) / 6.0)))
    numeric_parts.append(max(0.0, 1.0 - (abs(a["base_max"] - b["base_max"]) / 6.0)))
    numeric_parts.append(max(0.0, 1.0 - (abs(a["gap_min"] - b["gap_min"]) / 3.0)))
    numeric_parts.append(max(0.0, 1.0 - (abs(a["gap_max"] - b["gap_max"]) / 3.0)))
    numeric_parts.append(
        max(0.0, 1.0 - (abs(a["month_shift_strength"] - b["month_shift_strength"]) / 3.0))
    )
    numeric_parts.append(max(0.0, 1.0 - (abs(a["size_bias_12"] - b["size_bias_12"]) / 4.0)))
    numeric_parts.append(max(0.0, 1.0 - (abs(a["size_bias_18"] - b["size_bias_18"]) / 4.0)))
    numeric_parts.append(max(0.0, 1.0 - (abs(a["volatility"] - b["volatility"]) / 3.0)))
    numeric_score = sum(numeric_parts) / max(1, len(numeric_parts))

    l1 = sum(
        abs(float(a["anchor_weights"].get(k, 0.0)) - float(b["anchor_weights"].get(k, 0.0)))
        for k in ANCHOR_KEYS
    )
    # L1 on probability vectors ranges 0..2
    anchor_score = max(0.0, 1.0 - (l1 / 2.0))

    return (0.35 * pattern_score) + (0.45 * numeric_score) + (0.20 * anchor_score)


def _families_too_similar(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    return _family_similarity_score(a, b) >= 0.90


def _diversify_family(family: Dict[str, Any], idx: int) -> Dict[str, Any]:
    pattern_cycle = ["flat", "up", "down", "wave", "pulse"]
    mutated = dict(family)
    mutated["name"] = f"{family.get('name', f'Family {idx + 1}')} (diversified)"
    mutated["month_pattern"] = pattern_cycle[(pattern_cycle.index(family["month_pattern"]) + 1) % len(pattern_cycle)]
    mutated["base_min"] = max(1, family["base_min"] - (1 + (idx % 2)))
    mutated["base_max"] = min(30, family["base_max"] + 2)
    if mutated["base_max"] - mutated["base_min"] < 3:
        mutated["base_max"] = min(30, mutated["base_min"] + 3)
    mutated["gap_min"] = max(1, family["gap_min"])
    mutated["gap_max"] = min(10, max(family["gap_max"] + 1, mutated["gap_min"] + 1))
    mutated["month_shift_strength"] = min(6, max(1, family["month_shift_strength"] + 1))
    mutated["volatility"] = min(6, max(2, family["volatility"] + 1))
    mutated["size_bias_12"] = _clamp_int(family["size_bias_12"] + (1 if idx % 2 == 0 else -1), -6, 6, 0)
    mutated["size_bias_18"] = _clamp_int(family["size_bias_18"] + (-1 if idx % 2 == 0 else 1), -6, 6, 0)

    default_anchor = _sanitize_family(_default_family(idx), idx)["anchor_weights"]
    blended = {
        k: (0.70 * float(family["anchor_weights"].get(k, 0.0))) + (0.30 * float(default_anchor.get(k, 0.0)))
        for k in ANCHOR_KEYS
    }
    mutated["anchor_weights"] = _normalize_weights(blended, ANCHOR_KEYS)
    return _sanitize_family(mutated, idx)


def _diversify_families(families: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    diversified = [dict(f) for f in families]
    actions: List[Dict[str, Any]] = []

    for i in range(len(diversified)):
        for j in range(i + 1, len(diversified)):
            if not _families_too_similar(diversified[i], diversified[j]):
                continue

            before_score = round(_family_similarity_score(diversified[i], diversified[j]), 4)
            candidate = _diversify_family(diversified[j], j)
            action_type = "widened"

            # If still too similar after widening, replace with default template.
            if _families_too_similar(diversified[i], candidate):
                candidate = _sanitize_family(_default_family(j), j)
                action_type = "replaced_with_default"

            after_score = round(_family_similarity_score(diversified[i], candidate), 4)
            diversified[j] = candidate
            actions.append(
                {
                    "pair": f"{i + 1}-{j + 1}",
                    "action": action_type,
                    "before_similarity": before_score,
                    "after_similarity": after_score,
                }
            )

    return diversified, actions


def _sanitize_families(payload: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    families_raw = payload.get("families", []) if isinstance(payload, dict) else []
    if not isinstance(families_raw, list):
        families_raw = []

    sanitized = [_sanitize_family(f, i) for i, f in enumerate(families_raw[:3])]
    while len(sanitized) < 3:
        sanitized.append(_sanitize_family(_default_family(len(sanitized)), len(sanitized)))

    sanitized, diversity_actions = _diversify_families(sanitized)

    weights = _normalize_weights({str(i): f["priority_weight"] for i, f in enumerate(sanitized)}, ["0", "1", "2"])
    for i, f in enumerate(sanitized):
        f["priority_weight"] = float(weights[str(i)])

    return sanitized, {
        "provided_count": len(families_raw),
        "final_count": len(sanitized),
        "diversity_actions": diversity_actions,
    }

def _allocate_count_by_family(total: int, family_weights: List[float]) -> List[int]:
    total = int(total)
    n = len(family_weights)
    if total <= 0 or n == 0:
        return [0] * n

    normalized = _normalize_weights({str(i): w for i, w in enumerate(family_weights)}, [str(i) for i in range(n)])
    weights = [normalized[str(i)] for i in range(n)]

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


def _month_shift(pattern: str, idx: int, strength: int) -> int:
    if pattern == "up":
        return idx * strength
    if pattern == "down":
        return -idx * strength
    if pattern == "wave":
        seq = [0, strength, -strength]
        return seq[idx % 3]
    if pattern == "pulse":
        seq = [strength, 0, strength]
        return seq[idx % 3]
    return 0


def _build_anchor_vectors() -> Tuple[Dict[str, Dict[str, Dict[str, Dict[str, float]]]], Dict[str, Any]]:
    latest_month = MONTHS[-1]
    anchors: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {
        "latest_month": build_empty_scenario(),
        "last_3m_avg": build_empty_scenario(),
        "ly_same_3m": build_empty_scenario(),
        "stress_explore": build_empty_scenario(),
    }

    flat_cells = 0
    total_cells = 0
    synthetic_diversification_applied = False

    for month in MONTHS:
        for size, slabs in SIZES.items():
            for slab in slabs:
                hist_vals = [float(DEFAULTS[m][size][slab]) for m in MONTHS]
                latest_val = float(DEFAULTS[latest_month][size][slab])
                avg_val = float(sum(hist_vals) / len(hist_vals))
                month_idx = MONTHS.index(month)

                hist_span = max(hist_vals) - min(hist_vals)
                is_flat = hist_span < 1e-9
                total_cells += 1
                if is_flat:
                    flat_cells += 1

                avg_month_step = max(1.0, min(2.5, hist_span if hist_span > 0 else 1.5))

                # Keep latest-month anchor as strict carry-forward across all forecast months.
                latest_anchor = latest_val

                # Build a month-distinct average anchor to avoid collapse when defaults are flat.
                avg_offsets = [-avg_month_step, 0.0, avg_month_step]
                last3_anchor = avg_val + avg_offsets[month_idx]

                # LY anchor: use month-wise history when available; otherwise use a synthetic seasonal shape.
                if is_flat:
                    seasonal_offsets = [1.5, -0.5, 1.0]
                    ly_anchor = avg_val + seasonal_offsets[month_idx]
                    synthetic_diversification_applied = True
                else:
                    ly_anchor = hist_vals[month_idx]

                # Stress anchor: broader exploratory profile, not just (latest - 3).
                stress_amp = max(2.0, min(6.0, (hist_span + 2.0)))
                size_bias = 0.8 if size == "12-ML" else -0.6
                slab_bias = 0.4 if slab in ("slab3", "slab4") else 0.0
                stress_offsets = [-(stress_amp + 0.5), stress_amp * 0.5, stress_amp + 1.0]
                stress_anchor = avg_val + stress_offsets[month_idx] + size_bias + slab_bias

                anchors["latest_month"][month][size][slab] = float(clamp_discount(latest_anchor))
                anchors["last_3m_avg"][month][size][slab] = float(clamp_discount(last3_anchor))
                anchors["ly_same_3m"][month][size][slab] = float(clamp_discount(ly_anchor))
                anchors["stress_explore"][month][size][slab] = float(clamp_discount(stress_anchor))

    # Enforce ladder shape inside each month/size/anchor after per-slab construction.
    for anchor_key in ANCHOR_KEYS:
        for month in MONTHS:
            for size, slabs in SIZES.items():
                ladder = [anchors[anchor_key][month][size][slab] for slab in slabs]
                fixed, _, _ = enforce_ladder(ladder)
                for slab, val in zip(slabs, fixed):
                    anchors[anchor_key][month][size][slab] = float(val)

    info = {
        "ly_anchor_available": False,
        "ly_anchor_fallback": "month-wise history if available else synthetic seasonal profile",
        "flat_defaults_cells": flat_cells,
        "total_anchor_cells": total_cells,
        "synthetic_diversification_applied": synthetic_diversification_applied,
    }
    return anchors, info


def _choose_weighted_key(rng: random.Random, weights: Dict[str, float]) -> str:
    probs = _normalize_weights(weights, ANCHOR_KEYS)
    r = rng.random()
    cum = 0.0
    for k in ANCHOR_KEYS:
        cum += probs[k]
        if r <= cum:
            return k
    return ANCHOR_KEYS[-1]


def _scenario_signature(item: Dict[str, Any]) -> str:
    flat: List[int] = []
    months = item.get("months", {})
    for month in MONTHS:
        for size, slabs in SIZES.items():
            for slab in slabs:
                flat.append(int(months.get(month, {}).get(size, {}).get(slab, 0)))
    return ",".join(str(v) for v in flat)


def _is_degenerate(item: Dict[str, Any]) -> bool:
    months = item.get("months", {})
    values: List[int] = []
    month_signatures: List[Tuple[int, ...]] = []
    for month in MONTHS:
        month_vals: List[int] = []
        for size, slabs in SIZES.items():
            for slab in slabs:
                val = int(months.get(month, {}).get(size, {}).get(slab, 0))
                values.append(val)
                month_vals.append(val)
        month_signatures.append(tuple(month_vals))

    if not values:
        return True
    unique_vals = set(values)
    if len(unique_vals) <= 2:
        return True
    if max(values) <= 3:
        return True
    if (max(values) - min(values)) <= 2:
        return True
    if len(set(month_signatures)) <= 1:
        return True
    return False


def _sample_from_family(
    family: Dict[str, Any],
    anchors: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
    rng: random.Random,
) -> Tuple[Dict[str, Any], str]:
    months = build_empty_scenario()
    anchor_key = _choose_weighted_key(rng, family.get("anchor_weights", {}))
    anchor = anchors.get(anchor_key, anchors["last_3m_avg"])

    for m_idx, month in enumerate(MONTHS):
        shift = _month_shift(family["month_pattern"], m_idx, family["month_shift_strength"])
        for size, slabs in SIZES.items():
            bias = family["size_bias_12"] if size == "12-ML" else family["size_bias_18"]
            noise = rng.randint(-family["volatility"], family["volatility"]) if family["volatility"] > 0 else 0
            anchor_vals = [float(anchor[month][size][slab]) for slab in slabs]

            base_raw = anchor_vals[0] + bias + shift + noise + rng.randint(-1, 1)
            base = clamp_discount(max(family["base_min"], min(family["base_max"], base_raw)))
            ladder = [base]

            for i in range(1, len(slabs)):
                anchor_gap = max(1, int(round(anchor_vals[i] - anchor_vals[i - 1])))
                gap_raw = anchor_gap + rng.randint(-1, 1)
                gap = max(family["gap_min"], min(family["gap_max"], gap_raw))
                ladder.append(ladder[-1] + gap)

            fixed, _, _ = enforce_ladder(ladder)
            for slab, val in zip(slabs, fixed):
                months[month][size][slab] = int(val)

    return {"months": months}, anchor_key

def generate_with_gemini(
    user_prompt: str,
    count: int,
    planner_context: Dict[str, Any],
    temperature: float = 0.4,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    stage_callback: Optional[Callable[[str], None]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing")

    total_count = max(1, int(count))
    total_work = total_count + 1

    if progress_callback:
        progress_callback(0, total_work)
    if stage_callback:
        stage_callback("Sending family-planning prompt to Gemini...")

    with requests.Session() as session:
        families_payload, raw_text = _generate_families_with_gemini(
            session=session,
            api_key=api_key,
            user_prompt=user_prompt,
            planner_context=planner_context,
            temperature=temperature,
            stage_callback=stage_callback,
        )

    families, sanitize_info = _sanitize_families(families_payload)
    allocation = _allocate_count_by_family(total_count, [f["priority_weight"] for f in families])
    anchors, anchor_info = _build_anchor_vectors()

    if progress_callback:
        progress_callback(1, total_work)
    if stage_callback:
        stage_callback("Generating scenarios from AI families with Monte Carlo...")

    global_seen: set = set()
    scenarios: List[Dict[str, Any]] = []
    accepted_total = 0
    family_stats: List[Dict[str, Any]] = []

    seed_input = json.dumps(
        {
            "prompt": user_prompt,
            "planner_context": planner_context,
            "families": families,
            "total": total_count,
        },
        sort_keys=True,
    )
    seed_base = int(hashlib.sha256(seed_input.encode("utf-8")).hexdigest()[:12], 16) % (10**9)

    for fam_idx, family in enumerate(families):
        target = int(allocation[fam_idx]) if fam_idx < len(allocation) else 0
        stats = {
            "family_index": fam_idx + 1,
            "family_name": family["name"],
            "target": target,
            "accepted": 0,
            "duplicates": 0,
            "degenerate": 0,
            "corrections_missing": 0,
            "corrections_range": 0,
            "corrections_ladder": 0,
            "attempts": 0,
        }
        family_stats.append(stats)
        if target <= 0:
            continue

        rng = random.Random(seed_base + (fam_idx + 1) * 104729)
        max_attempts = max(200, target * 100)

        while stats["accepted"] < target and stats["attempts"] < max_attempts:
            stats["attempts"] += 1
            raw_scenario, anchor_used = _sample_from_family(family, anchors, rng)
            raw_scenario["family_name"] = family["name"]
            raw_scenario["anchor_used"] = anchor_used
            normalized = normalize_scenario(raw_scenario, index=accepted_total)

            if _is_degenerate(normalized):
                stats["degenerate"] += 1
                continue

            sig = _scenario_signature(normalized)
            if sig in global_seen:
                stats["duplicates"] += 1
                continue

            global_seen.add(sig)
            scenarios.append(normalized)
            stats["accepted"] += 1
            accepted_total += 1

            corr = normalized.get("corrections", {})
            stats["corrections_missing"] += int(corr.get("missing_filled", 0))
            stats["corrections_range"] += int(corr.get("range_or_rounding_fixes", 0))
            stats["corrections_ladder"] += int(corr.get("ladder_fixes", 0))

            if progress_callback:
                progress_callback(1 + accepted_total, total_work)

    if accepted_total < total_count:
        if stage_callback:
            stage_callback("Filling shortfall after dedupe/rejection...")
        extra_attempts = 0
        extra_max = max(1000, (total_count - accepted_total) * 250)
        rr_idx = 0

        while accepted_total < total_count and extra_attempts < extra_max:
            fam_idx = rr_idx % len(families)
            family = families[fam_idx]
            stats = family_stats[fam_idx]
            rng = random.Random(seed_base + 700001 + extra_attempts * 37)
            extra_attempts += 1
            stats["attempts"] += 1

            raw_scenario, anchor_used = _sample_from_family(family, anchors, rng)
            raw_scenario["family_name"] = family["name"]
            raw_scenario["anchor_used"] = anchor_used
            normalized = normalize_scenario(raw_scenario, index=accepted_total)

            if _is_degenerate(normalized):
                stats["degenerate"] += 1
                rr_idx += 1
                continue

            sig = _scenario_signature(normalized)
            if sig in global_seen:
                stats["duplicates"] += 1
                rr_idx += 1
                continue

            global_seen.add(sig)
            scenarios.append(normalized)
            stats["accepted"] += 1
            accepted_total += 1
            corr = normalized.get("corrections", {})
            stats["corrections_missing"] += int(corr.get("missing_filled", 0))
            stats["corrections_range"] += int(corr.get("range_or_rounding_fixes", 0))
            stats["corrections_ladder"] += int(corr.get("ladder_fixes", 0))
            if progress_callback:
                progress_callback(1 + accepted_total, total_work)
            rr_idx += 1

    if accepted_total < total_count:
        raise RuntimeError(
            f"Could not generate enough unique valid scenarios. Requested={total_count}, generated={accepted_total}. "
            "Try smaller count or broaden prompt goal/ranges."
        )

    for i, scn in enumerate(scenarios[:total_count]):
        scn["name"] = f"SCN-{i + 1:03d}"

    diagnostics = {
        "raw_gemini": raw_text,
        "raw_gemini_json": families_payload,
        "sanitize_info": sanitize_info,
        "sanitized_families": families,
        "allocation": [
            {
                "family_index": i + 1,
                "family_name": families[i]["name"],
                "priority_weight": families[i]["priority_weight"],
                "allocated": allocation[i],
            }
            for i in range(len(families))
        ],
        "family_stats": family_stats,
        "anchor_info": anchor_info,
    }

    if stage_callback:
        stage_callback("Generation complete")
    if progress_callback:
        progress_callback(total_work, total_work)
    return scenarios[:total_count], diagnostics


def flatten_scenarios(scenarios: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in scenarios:
        corr = item.get("corrections", {})
        row: Dict[str, Any] = {
            "scenario_name": item["name"],
            "family_name": item.get("family_name", ""),
            "anchor_used": item.get("anchor_used", ""),
            "missing_filled": int(corr.get("missing_filled", 0)),
            "range_or_rounding_fixes": int(corr.get("range_or_rounding_fixes", 0)),
            "ladder_fixes": int(corr.get("ladder_fixes", 0)),
        }
        for month in MONTHS:
            for size, slabs in SIZES.items():
                for slab in slabs:
                    col = f"{month}_{size}_{slab}".replace("-", "")
                    row[col] = int(item["months"][month][size][slab])
        rows.append(row)
    return pd.DataFrame(rows)


def month_table(scenario: Dict[str, Any], month: str) -> pd.DataFrame:
    rows = []
    for size, slabs in SIZES.items():
        for slab in slabs:
            rows.append({"Size": size, "Slab": slab, "Discount %": int(scenario["months"][month][size][slab])})
    return pd.DataFrame(rows)

def main() -> None:
    st.set_page_config(page_title="AI Scenario Testing", layout="wide")
    st.title("AI Scenario Testing")
    st.caption("3-family AI planner + Monte Carlo scenario generation (Streamlit)")

    col1, col2, col3 = st.columns([4, 1, 1])
    with col1:
        prompt = st.text_area(
            "Prompt",
            value="Generate realistic scenarios with focus on higher revenue but controlled discount depth.",
            height=120,
        )
    with col2:
        scenario_count = st.number_input("Scenarios", min_value=1, max_value=10000, value=200, step=1)
    with col3:
        temperature = st.slider("AI Temperature", min_value=0.0, max_value=1.0, value=0.4, step=0.1)

    with st.expander("Planner Context Sent To Gemini", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            e12_from_18 = st.number_input("Cross Elasticity (12 wrt 18)", value=-1.020, step=0.001, format="%.3f")
            reference_window = st.selectbox("Reference Window", options=["LY same 3M", "Last 3M"], index=0)
        with c2:
            e18_from_12 = st.number_input("Cross Elasticity (18 wrt 12)", value=-0.305, step=0.001, format="%.3f")
            model_notes = st.text_area(
                "Model Build Notes",
                value=(
                    "Scenario planner context (important):\n"
                    "1) There is a negative relationship between 12-ML and 18-ML. They can eat into each other's volume. "
                    "Plan scenarios according to the prompt goal while respecting this cannibalization behavior.\n"
                    "2) Slab volume is computed from discount drivers:\n"
                    "   - beta_own * current slab discount\n"
                    "   - beta_lag * previous month discount (lag discount)\n"
                    "   - beta_other * weighted discount of other slabs in the same size\n"
                    "3) For each size, slab volumes are summed to get total 12-ML volume and total 18-ML volume.\n"
                    "4) Cross-elasticity adjustment is applied on month-on-month change from the previous month.\n"
                    "5) Keep scenario ladders valid, realistic, and business-applicable. Think like a scenario planner simulating "
                    "real-world, reasonable cases."
                ),
                height=140,
            )

    planner_context = {
        "cross_elasticity_12_wrt_18": float(e12_from_18),
        "cross_elasticity_18_wrt_12": float(e18_from_12),
        "reference_window": reference_window,
        "model_build_notes": model_notes.strip(),
        "family_count_locked": 3,
    }

    stage_text = st.empty()
    progress_text = st.empty()
    progress_bar = st.progress(0, text="AI idle")

    if st.button("Generate", type="primary", use_container_width=True):
        try:
            total = int(scenario_count)
            if total > 1000:
                st.info("Large request detected. This may take a few minutes.")

            def on_stage(message: str) -> None:
                stage_text.info(message)

            def on_progress(done: int, total_count: int) -> None:
                pct = 0 if total_count <= 0 else int((done / total_count) * 100)
                progress_text.info(f"AI generating scenarios... {done} / {total_count}")
                progress_bar.progress(pct, text=f"Generating ({pct}%)")

            scenarios, diagnostics = generate_with_gemini(
                user_prompt=prompt,
                count=total,
                planner_context=planner_context,
                temperature=temperature,
                progress_callback=on_progress,
                stage_callback=on_stage,
            )

            st.session_state["ai_test_scenarios"] = scenarios
            st.session_state["ai_test_diagnostics"] = diagnostics
            progress_bar.progress(100, text="Generation complete")
            stage_text.success("Gemini generation complete.")
            st.success(f"Generated {len(scenarios)} scenario(s) using 3 AI families")

        except Exception as exc:
            st.session_state.pop("ai_test_scenarios", None)
            st.session_state.pop("ai_test_diagnostics", None)
            progress_bar.progress(0, text="Generation failed")
            stage_text.error("Gemini request failed.")
            progress_text.error("AI scenario generation failed.")
            st.error(f"Gemini generation failed: {exc}")
            return

    scenarios = st.session_state.get("ai_test_scenarios", [])
    diagnostics = st.session_state.get("ai_test_diagnostics", {})
    if not scenarios:
        st.info("Click Generate to create scenarios.")
        return

    flat_df = flatten_scenarios(scenarios)
    corr_missing = int(flat_df["missing_filled"].sum()) if "missing_filled" in flat_df.columns else 0
    corr_rr = int(flat_df["range_or_rounding_fixes"].sum()) if "range_or_rounding_fixes" in flat_df.columns else 0
    corr_ladder = int(flat_df["ladder_fixes"].sum()) if "ladder_fixes" in flat_df.columns else 0
    total_corrections = corr_missing + corr_rr + corr_ladder
    if total_corrections > 0:
        st.warning(
            f"Validation corrections applied: missing-filled={corr_missing}, "
            f"range/rounding fixes={corr_rr}, ladder fixes={corr_ladder}"
        )
    else:
        st.success("No validation corrections were needed in generated scenarios.")

    st.subheader("Family Diagnostics")
    alloc_df = pd.DataFrame(diagnostics.get("allocation", []))
    stats_df = pd.DataFrame(diagnostics.get("family_stats", []))

    dleft, dright = st.columns(2)
    with dleft:
        st.markdown("**Family Allocation**")
        if alloc_df.empty:
            st.caption("No allocation diagnostics available")
        else:
            st.dataframe(alloc_df, use_container_width=True, hide_index=True)
    with dright:
        st.markdown("**Accepted / Corrected by Family**")
        if stats_df.empty:
            st.caption("No family stats available")
        else:
            cols = [
                "family_name",
                "target",
                "accepted",
                "duplicates",
                "degenerate",
                "corrections_missing",
                "corrections_range",
                "corrections_ladder",
                "attempts",
            ]
            present = [c for c in cols if c in stats_df.columns]
            st.dataframe(stats_df[present], use_container_width=True, hide_index=True)

    st.subheader("Preview")
    page_size = st.selectbox("Rows per page", options=[5, 10, 20, 50], index=2)
    total_rows = len(flat_df)
    max_start = max(0, total_rows - page_size)
    start = 0 if max_start == 0 else st.slider("Start row", min_value=0, max_value=max_start, value=0, step=1)
    st.dataframe(flat_df.iloc[start : start + page_size], use_container_width=True, hide_index=True)

    st.subheader("Scenario Detail")
    selected = st.selectbox(
        "Select scenario",
        options=list(range(len(scenarios))),
        format_func=lambda i: (
            f"{scenarios[i].get('name')} | {scenarios[i].get('family_name', 'NA')} | {scenarios[i].get('anchor_used', 'NA')}"
        ),
    )
    detail = scenarios[selected]
    c_dec, c_jan, c_feb = st.columns(3)
    with c_dec:
        st.markdown("**Dec**")
        st.dataframe(month_table(detail, "Dec"), use_container_width=True, hide_index=True)
    with c_jan:
        st.markdown("**Jan**")
        st.dataframe(month_table(detail, "Jan"), use_container_width=True, hide_index=True)
    with c_feb:
        st.markdown("**Feb**")
        st.dataframe(month_table(detail, "Feb"), use_container_width=True, hide_index=True)

    csv_data = flat_df.to_csv(index=False).encode("utf-8")
    json_data = json.dumps(scenarios, indent=2).encode("utf-8")

    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            "Download CSV",
            data=csv_data,
            file_name="generated_scenarios.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with dl2:
        st.download_button(
            "Download JSON",
            data=json_data,
            file_name="generated_scenarios.json",
            mime="application/json",
            use_container_width=True,
        )

    with st.expander("Raw Gemini Outputs (Used by App)", expanded=False):
        raw_text = diagnostics.get("raw_gemini", "")
        raw_json = diagnostics.get("raw_gemini_json", {})
        sanitized = diagnostics.get("sanitized_families", [])
        allocation = diagnostics.get("allocation", [])
        anchor_info = diagnostics.get("anchor_info", {})

        st.markdown("**Raw Gemini Response**")
        st.code(raw_text or "(empty)", language="json")
        st.markdown("**Raw Parsed JSON**")
        st.code(json.dumps(raw_json, indent=2), language="json")
        st.markdown("**Sanitized Families**")
        st.code(json.dumps(sanitized, indent=2), language="json")
        st.markdown("**Allocation Summary**")
        st.code(json.dumps(allocation, indent=2), language="json")
        st.markdown("**Anchor Info**")
        st.code(json.dumps(anchor_info, indent=2), language="json")

        raw_jsonl = "\n".join(
            [
                json.dumps({"type": "raw_gemini", "payload": raw_text}),
                json.dumps({"type": "raw_parsed", "payload": raw_json}),
                json.dumps({"type": "sanitized_families", "payload": sanitized}),
                json.dumps({"type": "allocation", "payload": allocation}),
                json.dumps({"type": "family_stats", "payload": diagnostics.get("family_stats", [])}),
                json.dumps({"type": "anchor_info", "payload": anchor_info}),
            ]
        )
        st.download_button(
            "Download Diagnostics (JSONL)",
            data=raw_jsonl.encode("utf-8"),
            file_name="ai_family_diagnostics.jsonl",
            mime="application/json",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
