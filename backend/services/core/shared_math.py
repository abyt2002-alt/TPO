"""Shared numeric utilities and constrained model primitives.\n\nThis module contains generic math/rounding/sorting/prediction helpers reused\nby multiple steps to keep formulas consistent.\n"""

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


class CustomConstrainedRidge:
    """Ridge regression with sign-constrained coefficients."""

    def __init__(self, l2_penalty=1.0, non_negative_indices=None, non_positive_indices=None, maxiter=2000):
        self.l2_penalty = float(l2_penalty)
        self.non_negative_indices = tuple(non_negative_indices or [])
        self.non_positive_indices = tuple(non_positive_indices or [])
        self.maxiter = int(maxiter)
        self.intercept_ = 0.0
        self.coef_ = None
        self.n_features_in_ = 0
        self.success_ = False
        self.message_ = ""

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, p = X.shape
        self.n_features_in_ = p

        x_mean = X.mean(axis=0)
        x_std = X.std(axis=0)
        x_std = np.where(x_std <= 1e-12, 1.0, x_std)
        Xs = (X - x_mean) / x_std

        y_mean = float(np.mean(y))
        theta0 = np.zeros(p + 1, dtype=float)
        theta0[0] = y_mean

        bounds = [(None, None)] * (p + 1)
        for idx in self.non_negative_indices:
            if 0 <= idx < p:
                bounds[idx + 1] = (0.0, None)
        for idx in self.non_positive_indices:
            if 0 <= idx < p:
                bounds[idx + 1] = (None, 0.0)

        lam = self.l2_penalty

        def obj(theta):
            b0 = theta[0]
            w = theta[1:]
            resid = y - (Xs @ w + b0)
            return float(np.mean(resid * resid) + lam * np.sum(w * w))

        def grad(theta):
            b0 = theta[0]
            w = theta[1:]
            resid = (Xs @ w + b0) - y
            g_b0 = 2.0 * np.mean(resid)
            g_w = 2.0 * (Xs.T @ resid) / n + 2.0 * lam * w
            return np.concatenate([[g_b0], g_w])

        res = minimize(
            obj,
            theta0,
            jac=grad,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': self.maxiter}
        )
        self.success_ = bool(res.success)
        self.message_ = str(res.message)

        theta = np.asarray(res.x, dtype=float)
        b0_s = float(theta[0])
        w_s = theta[1:]

        coef = w_s / x_std
        intercept = b0_s - float(np.sum((w_s * x_mean) / x_std))

        if self.non_negative_indices:
            coef[list(self.non_negative_indices)] = np.maximum(coef[list(self.non_negative_indices)], 0.0)
        if self.non_positive_indices:
            coef[list(self.non_positive_indices)] = np.minimum(coef[list(self.non_positive_indices)], 0.0)

        self.intercept_ = float(intercept)
        self.coef_ = coef.astype(float)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercept_

class SharedMathMixin:

    def _round_discount_series(self, series, step: float = 0.5) -> pd.Series:
        safe_step = max(float(step), 0.1)
        s = pd.Series(series, copy=False).astype(float)
        return (np.round(s / safe_step) * safe_step).astype(float)


    def _slab_sort_key(self, slab_value):
        text = str(slab_value or "").strip().lower()
        m = re.search(r"(\d+)", text)
        if m:
            return (0, int(m.group(1)), text)
        return (1, text)


    def _is_step2_allowed_slab(self, slab_value, include_zero: bool = False) -> bool:
        text = str(slab_value or "").strip().lower()
        m = re.search(r"(\d+)", text)
        if not m:
            return False
        try:
            idx = int(m.group(1))
        except Exception:
            return False
        if include_zero:
            return idx >= 0
        return idx >= 1


    def _extract_slab_index(self, slab_value) -> Optional[int]:
        text = str(slab_value or "").strip().lower()
        m = re.search(r"(\d+)", text)
        if not m:
            return None
        try:
            idx = int(m.group(1))
        except Exception:
            return None
        return idx if 1 <= idx <= 4 else None


    def _predict_stage2_quantity(self, stage2_model, residual_store, structural, tactical, lag1=None, extra_feature_values=None):
        residual_store = np.asarray(residual_store, dtype=float)
        structural = np.asarray(structural, dtype=float)
        tactical = np.asarray(tactical, dtype=float)
        if lag1 is None:
            lag1 = np.zeros_like(structural, dtype=float)
        lag1 = np.asarray(lag1, dtype=float)

        feature_order = list(getattr(stage2_model, 'feature_order_', []) or [])
        if feature_order:
            feature_map = {
                'residual_store': residual_store,
                'base_discount_pct': structural,
                'tactical_discount_pct': tactical,
                'lag1_base_discount_pct': lag1,
            }
            for key, value in (extra_feature_values or {}).items():
                feature_map[str(key)] = np.asarray(value, dtype=float)
            cols = []
            for feat in feature_order:
                arr = feature_map.get(feat)
                if arr is None:
                    arr = np.zeros_like(structural, dtype=float)
                cols.append(np.asarray(arr, dtype=float))
            x = np.column_stack(cols)
        else:
            stage2_features = int(getattr(stage2_model, 'n_features_in_', 4))
            if stage2_features >= 4:
                x = np.column_stack([residual_store, structural, tactical, lag1])
            elif stage2_features == 3:
                x = np.column_stack([residual_store, structural, tactical])
            elif stage2_features == 2:
                x = np.column_stack([residual_store, structural])
            else:
                x = residual_store.reshape(-1, 1)
        return stage2_model.predict(x)


    def _normalize_weight_map_from_qty(self, qty_map: Dict[str, float]) -> Dict[str, float]:
        clean = {str(k): max(float(v), 0.0) for k, v in (qty_map or {}).items()}
        total = float(sum(clean.values()))
        if total <= 0:
            n = max(len(clean), 1)
            return {k: (1.0 / n) for k in clean.keys()}
        return {k: (v / total) for k, v in clean.items()}


    def _compute_other_weighted_discount_for_slab(
        self,
        slab_key: str,
        discount_map: Dict[str, float],
        weight_map: Dict[str, float],
    ) -> float:
        weighted_sum = 0.0
        weight_sum = 0.0
        for other_slab, disc in (discount_map or {}).items():
            other_key = str(other_slab)
            if other_key == str(slab_key):
                continue
            w = float(weight_map.get(other_key, 0.0))
            if not np.isfinite(w) or w <= 0:
                continue
            weighted_sum += w * float(disc)
            weight_sum += w
        if weight_sum <= 0:
            return 0.0
        return float(weighted_sum / weight_sum)
