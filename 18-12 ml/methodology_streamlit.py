from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pyarrow.parquet as pq
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import Holt
import streamlit as st


TARGET_SUBCATEGORIES = {"STX INSTA SHAMPOO", "STREAX INSTA SHAMPOO"}
TARGET_SIZES = {"12-ML", "18-ML"}
REQUIRED_COLUMNS = {"Subcategory", "Sizes"}
PREFERRED_COLUMNS = [
    "Date",
    "Brand",
    "Subcategory",
    "Sizes",
    "Sku_Code",
    "Sku_Name",
    "Material_ID",
    "MG-10(Base_Sku)",
    "T SKU Code",
    "MRP",
    "Outlet_ID",
    "Bill_No",
    "Invoice_Key",
    "Final_Outlet_Classification",
    "SalesValue_atBasicRate",
    "Quantity",
    "TotalDiscount",
    "Net_Amt",
]


def _normalize_text(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.upper()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def _normalize_size(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.upper()
        .str.replace(" ", "", regex=False)
        .str.strip()
    )


def _find_data_dir() -> Path | None:
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / "DATA",
        script_dir.parent / "DATA",
        Path.cwd() / "DATA",
    ]
    for folder in candidates:
        if folder.exists():
            return folder
    return None


def _parse_breaks(value: str) -> list[float]:
    parts = [x.strip() for x in value.split(",") if x.strip()]
    if not parts:
        raise ValueError("Please enter at least one slab break.")
    breaks = sorted({float(x) for x in parts})
    return breaks


def _breaks_to_text(breaks: list[float]) -> str:
    out: list[str] = []
    for x in breaks:
        if float(x).is_integer():
            out.append(str(int(x)))
        else:
            out.append(str(x))
    return ",".join(out)


def _assign_slab(series: pd.Series, breaks: list[float]) -> pd.Categorical:
    bins = [-float("inf"), *breaks, float("inf")]
    labels = [f"slab{i}" for i in range(len(bins) - 1)]
    return pd.cut(series, bins=bins, labels=labels, right=False)


def build_defined_slabs(df: pd.DataFrame, breaks_12: list[float], breaks_18: list[float]) -> pd.DataFrame:
    required = {"Date", "Outlet_ID", "Sizes", "Quantity"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for slab definition: {', '.join(missing)}")

    work = df.copy()
    work = work.dropna(subset=["Date"]).copy()
    work["Month"] = work["Date"].dt.to_period("M").dt.to_timestamp()
    monthly = (
        work.groupby(["Sizes", "Outlet_ID", "Month"], as_index=False)
        .agg(Monthly_Quantity=("Quantity", "sum"))
    )

    monthly["Defined_Slab"] = None
    mask_12 = monthly["Sizes"] == "12-ML"
    mask_18 = monthly["Sizes"] == "18-ML"
    if mask_12.any():
        monthly.loc[mask_12, "Defined_Slab"] = _assign_slab(monthly.loc[mask_12, "Monthly_Quantity"], breaks_12).astype(str)
    if mask_18.any():
        monthly.loc[mask_18, "Defined_Slab"] = _assign_slab(monthly.loc[mask_18, "Monthly_Quantity"], breaks_18).astype(str)

    out = work.merge(monthly, on=["Sizes", "Outlet_ID", "Month"], how="left")
    out = out[out["Defined_Slab"].astype(str) != "slab0"].copy()
    return out


def build_slab_rules_df(breaks: list[float]) -> pd.DataFrame:
    bins = [-float("inf"), *breaks, float("inf")]
    rows = []
    for idx in range(len(bins) - 1):
        left = bins[idx]
        right = bins[idx + 1]
        if left == -float("inf"):
            rule = f"Qty < {right:g}"
        elif right == float("inf"):
            rule = f"Qty >= {left:g}"
        else:
            rule = f"{left:g} <= Qty < {right:g}"
        rows.append({"Slab": f"slab{idx}", "Rule": rule})
    rules_df = pd.DataFrame(rows)
    return rules_df[rules_df["Slab"] != "slab0"].reset_index(drop=True)


def build_slab_mix_df(defined_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["Sizes", "Outlet_ID", "Month", "Defined_Slab", "Monthly_Quantity"]
    monthly_unique = defined_df[cols].drop_duplicates().copy()
    monthly_unique = monthly_unique.dropna(subset=["Defined_Slab"])

    mix = (
        monthly_unique.groupby(["Sizes", "Defined_Slab"], as_index=False)
        .agg(
            Outlet_Months=("Outlet_ID", "count"),
            Unique_Outlets=("Outlet_ID", "nunique"),
            Quantity=("Monthly_Quantity", "sum"),
        )
    )
    total_qty = mix.groupby("Sizes")["Quantity"].transform("sum")
    mix["Quantity_%"] = (mix["Quantity"] / total_qty * 100.0).round(2)
    mix["Slab_Order"] = pd.to_numeric(
        mix["Defined_Slab"].astype(str).str.extract(r"(\d+)")[0], errors="coerce"
    )
    mix = mix[mix["Defined_Slab"].astype(str) != "slab0"].copy()
    mix = mix.sort_values(["Sizes", "Slab_Order"], na_position="last").drop(columns=["Slab_Order"])
    mix["Quantity"] = mix["Quantity"].round(2)
    return mix


def build_outlet_discount_variation(
    defined_df: pd.DataFrame, size_value: str, slab_value: str, month_value: pd.Timestamp
) -> tuple[pd.DataFrame, float, int, float, float]:
    scope = defined_df[
        (defined_df["Sizes"] == size_value)
        & (defined_df["Defined_Slab"].astype(str) == str(slab_value))
        & (defined_df["Month"] == month_value)
    ].copy()

    if scope.empty:
        return pd.DataFrame(), np.nan, 0, 0.0, np.nan

    agg_map: dict[str, tuple[str, str]] = {
        "Sales_Value": ("SalesValue_atBasicRate", "sum"),
        "Total_Discount": ("TotalDiscount", "sum"),
        "Quantity": ("Quantity", "sum"),
        "Rows": ("Quantity", "count"),
    }
    if "Bill_No" in scope.columns:
        agg_map["Invoices"] = ("Bill_No", "nunique")

    by_outlet = scope.groupby("Outlet_ID", as_index=False).agg(**agg_map)
    by_outlet = by_outlet[by_outlet["Sales_Value"] > 0].copy()

    if by_outlet.empty:
        return pd.DataFrame(), np.nan, 0, 0.0, np.nan

    overall_discount_pct = float(scope["TotalDiscount"].sum() / scope["SalesValue_atBasicRate"].sum() * 100.0)
    by_outlet["Outlet_Discount_%"] = (by_outlet["Total_Discount"] / by_outlet["Sales_Value"] * 100.0).round(2)
    by_outlet["Deviation_pp"] = (by_outlet["Outlet_Discount_%"] - overall_discount_pct).round(2)
    by_outlet["Abs_Deviation_pp"] = by_outlet["Deviation_pp"].abs().round(2)
    by_outlet = by_outlet.sort_values(["Abs_Deviation_pp", "Sales_Value"], ascending=[False, False]).reset_index(drop=True)

    total_qty = float(by_outlet["Quantity"].sum())
    outlet_count = int(by_outlet["Outlet_ID"].nunique())
    avg_abs_deviation = float(by_outlet["Abs_Deviation_pp"].mean()) if not by_outlet.empty else np.nan
    return by_outlet, round(overall_discount_pct, 2), outlet_count, round(total_qty, 2), round(avg_abs_deviation, 2)


def _round_to_step(value: float, step: float) -> float:
    return round(float(value) / step) * step


def _apply_fixed_step_rule(actual_series: pd.Series, round_step: float = 0.5, min_step_pp: float = 1.0) -> pd.Series:
    out: list[float] = []
    prev_base: float | None = None
    for raw in actual_series:
        if pd.isna(raw):
            out.append(np.nan if prev_base is None else prev_base)
            continue
        curr = _round_to_step(float(raw), round_step)
        if prev_base is None:
            prev_base = curr
        else:
            if curr >= (prev_base + min_step_pp):
                prev_base = curr
            elif curr <= (prev_base - min_step_pp):
                prev_base = curr
            # else: keep previous base (no 0.5pp shift)
        out.append(float(prev_base))
    return pd.Series(out, index=actual_series.index, dtype=float)


def _safe_corr(x: pd.Series, y: pd.Series) -> float:
    x_num = pd.to_numeric(x, errors="coerce")
    y_num = pd.to_numeric(y, errors="coerce")
    valid = ~(x_num.isna() | y_num.isna())
    if valid.sum() < 2:
        return np.nan
    xv = x_num[valid].to_numpy(dtype=float)
    yv = y_num[valid].to_numpy(dtype=float)
    if np.nanstd(xv) <= 1e-12 or np.nanstd(yv) <= 1e-12:
        return np.nan
    return float(np.corrcoef(xv, yv)[0, 1])


def render_xy_diagnostics(
    df: pd.DataFrame,
    y_col: str,
    feature_cols: list[str],
    title_prefix: str,
    chart_key_prefix: str = "xy_diag",
) -> None:
    if df is None or df.empty:
        st.info("No data available for diagnostics.")
        return
    if y_col not in df.columns:
        st.info(f"'{y_col}' not found in model table.")
        return

    available_features = [f for f in feature_cols if f in df.columns]
    if not available_features:
        st.info("No feature columns available for diagnostics.")
        return

    corr_rows: list[dict[str, float | str]] = []
    for feat in available_features:
        corr_rows.append(
            {
                "Feature": feat,
                "Corr_with_Y": _safe_corr(df[feat], df[y_col]),
            }
        )

    corr_df = pd.DataFrame(corr_rows)
    corr_df["Abs_Corr"] = corr_df["Corr_with_Y"].abs()
    # Keep the same input feature order to make slab-to-slab comparison stable.
    corr_df = corr_df.reset_index(drop=True)
    st.dataframe(corr_df[["Feature", "Corr_with_Y"]], use_container_width=True, height=190)

    plot_cols = st.columns(2)
    for idx, feat in enumerate(corr_df["Feature"].tolist()):
        work = df[[feat, y_col]].copy()
        if "Period" in df.columns:
            work["Period"] = pd.to_datetime(df["Period"], errors="coerce")
            work = work.dropna(subset=["Period"]).sort_values("Period")
        else:
            work["Period"] = np.arange(len(work), dtype=int)

        work = work.dropna(subset=[feat, y_col]).copy()
        if work.empty:
            continue

        corr_val = _safe_corr(work[feat], work[y_col])
        corr_txt = f"{corr_val:.3f}" if np.isfinite(corr_val) else "NA"

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=work["Period"],
                y=work[y_col],
                mode="lines+markers",
                name=y_col,
                line=dict(color="#1d4ed8", width=2.5),
                hovertemplate=(
                    f"{y_col}: %{{y:.4f}}<extra></extra>"
                ),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=work["Period"],
                y=work[feat],
                mode="lines+markers",
                name=feat,
                yaxis="y2",
                line=dict(color="#0b7a75", width=2.5),
                hovertemplate=f"{feat}: %{{y:.4f}}<extra></extra>",
            )
        )

        fig.update_layout(
            title=f"{title_prefix}: {y_col} and {feat} over time (corr={corr_txt})",
            height=320,
            margin=dict(l=20, r=20, t=45, b=20),
            xaxis_title="Period",
            yaxis_title=y_col,
            yaxis2=dict(
                title=feat,
                overlaying="y",
                side="right",
                showgrid=False,
            ),
            legend=dict(orientation="h", y=-0.2),
        )
        with plot_cols[idx % 2]:
            chart_key = f"{chart_key_prefix}_{idx}_{str(feat)}".replace(" ", "_")
            st.plotly_chart(fig, use_container_width=True, key=chart_key)


def render_store_quantity_chart(
    df: pd.DataFrame,
    quantity_col: str,
    store_col: str,
    title: str,
    chart_key: str = "",
) -> None:
    if df is None or df.empty:
        st.info("No data available for Store Count vs Quantity chart.")
        return
    needed = [quantity_col, store_col]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        st.info(f"Missing columns for chart: {', '.join(missing)}")
        return

    work = df.copy()
    if "Period" in work.columns:
        work["Period"] = pd.to_datetime(work["Period"], errors="coerce")
        work = work.dropna(subset=["Period"]).sort_values("Period")
        x_vals = work["Period"]
        x_title = "Period"
    else:
        x_vals = np.arange(len(work), dtype=int)
        x_title = "Index"

    fig = go.Figure()
    corr_val = _safe_corr(work[quantity_col], work[store_col])
    corr_txt = f"{corr_val:.3f}" if np.isfinite(corr_val) else "NA"
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=pd.to_numeric(work[quantity_col], errors="coerce"),
            mode="lines+markers",
            name=quantity_col,
            line=dict(color="#1d4ed8", width=2.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=pd.to_numeric(work[store_col], errors="coerce"),
            mode="lines+markers",
            name=store_col,
            yaxis="y2",
            line=dict(color="#0b7a75", width=2.5),
        )
    )
    fig.update_layout(
        title=f"{title} (corr={corr_txt})",
        height=340,
        margin=dict(l=20, r=20, t=45, b=20),
        xaxis_title=x_title,
        yaxis_title=quantity_col,
        yaxis2=dict(
            title=store_col,
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        legend=dict(orientation="h", y=-0.2),
    )
    resolved_key = chart_key or f"sq_{quantity_col}_{store_col}_{title}".replace(" ", "_")
    st.plotly_chart(fig, use_container_width=True, key=resolved_key)
    st.caption(f"Correlation ({quantity_col} vs {store_col}): {corr_txt}")


def compute_base_discount(defined_df: pd.DataFrame, round_step: float = 0.5, min_step_pp: float = 1.0) -> pd.DataFrame:
    required = {"Sizes", "Defined_Slab", "Month", "SalesValue_atBasicRate", "TotalDiscount", "Monthly_Quantity"}
    missing = [c for c in required if c not in defined_df.columns]
    if missing:
        raise KeyError(f"Missing required columns for base discount: {', '.join(missing)}")

    monthly_unique = defined_df[
        ["Sizes", "Defined_Slab", "Outlet_ID", "Month", "Monthly_Quantity"]
    ].drop_duplicates()

    monthly_disc = (
        defined_df.groupby(["Sizes", "Defined_Slab", "Month"], as_index=False)
        .agg(
            Sales_Value=("SalesValue_atBasicRate", "sum"),
            Total_Discount=("TotalDiscount", "sum"),
        )
    )
    monthly_qty = (
        monthly_unique.groupby(["Sizes", "Defined_Slab", "Month"], as_index=False)
        .agg(Monthly_Quantity=("Monthly_Quantity", "sum"))
    )

    out = monthly_disc.merge(monthly_qty, on=["Sizes", "Defined_Slab", "Month"], how="left")
    out = out[out["Defined_Slab"].astype(str) != "slab0"].copy()
    out["Actual_Discount_%"] = np.where(
        out["Sales_Value"] > 0,
        (out["Total_Discount"] / out["Sales_Value"]) * 100.0,
        np.nan,
    )
    out = out.sort_values(["Sizes", "Defined_Slab", "Month"]).reset_index(drop=True)
    out["Estimated_Base_%"] = (
        out.groupby(["Sizes", "Defined_Slab"], group_keys=False)["Actual_Discount_%"]
        .apply(lambda s: _apply_fixed_step_rule(s, round_step=round_step, min_step_pp=min_step_pp))
    )
    out["Actual_Discount_%"] = out["Actual_Discount_%"].round(2)
    out["Estimated_Base_%"] = out["Estimated_Base_%"].round(2)
    return out


def build_base_summary(base_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        base_df.groupby(["Sizes", "Defined_Slab"], as_index=False)
        .agg(
            Months=("Month", "nunique"),
            Avg_Actual_Discount=("Actual_Discount_%", "mean"),
            Avg_Estimated_Base=("Estimated_Base_%", "mean"),
            Quantity=("Monthly_Quantity", "sum"),
        )
    )
    total_qty = summary.groupby("Sizes")["Quantity"].transform("sum")
    summary["Quantity_%"] = (summary["Quantity"] / total_qty * 100.0).round(2)
    summary["Avg_Actual_Discount"] = summary["Avg_Actual_Discount"].round(2)
    summary["Avg_Estimated_Base"] = summary["Avg_Estimated_Base"].round(2)
    summary["Slab_Order"] = pd.to_numeric(
        summary["Defined_Slab"].astype(str).str.extract(r"(\d+)")[0], errors="coerce"
    )
    summary = summary.sort_values(["Sizes", "Slab_Order"], na_position="last").drop(columns=["Slab_Order"])
    return summary


def _slab_sort_key(value: str) -> float:
    v = pd.to_numeric(pd.Series([value]).astype(str).str.extract(r"(\d+)")[0], errors="coerce").iloc[0]
    return float(v) if pd.notna(v) else 1e9


def render_slab_charts(base_df: pd.DataFrame, size_label: str, two_per_row: bool = True) -> None:
    size_df = base_df[base_df["Sizes"] == size_label].copy()
    slabs = sorted(size_df["Defined_Slab"].dropna().astype(str).unique(), key=_slab_sort_key)
    if not slabs:
        st.info(f"No slab data for {size_label}.")
        return

    st.markdown(f"**{size_label} - Slab Wise Base Discount**")
    chunk_size = 2 if two_per_row else 1
    for i in range(0, len(slabs), chunk_size):
        cols = st.columns(chunk_size) if two_per_row else [st.container()]
        for j, slab in enumerate(slabs[i : i + chunk_size]):
            slab_df = size_df[size_df["Defined_Slab"].astype(str) == slab].sort_values("Month")
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=slab_df["Month"],
                    y=slab_df["Actual_Discount_%"],
                    mode="lines",
                    name="Actual Discount %",
                    line=dict(color="#9ec5ff", width=2, shape="hv"),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=slab_df["Month"],
                    y=slab_df["Estimated_Base_%"],
                    mode="lines",
                    name="Estimated Base %",
                    line=dict(color="#0b7a75", width=4, shape="hv"),
                )
            )
            fig.update_layout(
                title=f"{slab}",
                height=300,
                margin=dict(l=20, r=20, t=40, b=20),
                legend=dict(orientation="h", y=-0.25),
                xaxis_title="Month",
                yaxis_title="Discount %",
            )
            cols[j].plotly_chart(fig, use_container_width=True)


class CustomConstrainedRidge:
    def __init__(self, l2_penalty=1.0, non_negative_indices=None, non_positive_indices=None, maxiter=2000):
        self.l2_penalty = float(l2_penalty)
        self.non_negative_indices = tuple(non_negative_indices or [])
        self.non_positive_indices = tuple(non_positive_indices or [])
        self.maxiter = int(maxiter)
        self.intercept_ = 0.0
        self.coef_ = None
        self.n_features_in_ = 0
        self.success_ = False

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, p = X.shape
        self.n_features_in_ = p

        x_mean = X.mean(axis=0)
        x_std = np.where(X.std(axis=0) <= 1e-12, 1.0, X.std(axis=0))
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

        lam = float(self.l2_penalty)

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
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": self.maxiter},
        )
        self.success_ = bool(res.success)
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


def _predict_stage2_quantity(stage2_model, residual_store, structural, tactical, lag1=None, extra_feature_values=None):
    residual_store = np.asarray(residual_store, dtype=float)
    structural = np.asarray(structural, dtype=float)
    tactical = np.asarray(tactical, dtype=float)
    if lag1 is None:
        lag1 = np.zeros_like(structural, dtype=float)
    lag1 = np.asarray(lag1, dtype=float)

    feature_order = getattr(stage2_model, "feature_order_", None)
    if feature_order:
        mapping = {
            "residual_store": residual_store,
            "base_discount_pct": structural,
            "tactical_discount_pct": tactical,
            "lag1_base_discount_pct": lag1,
        }
        if extra_feature_values:
            for k, v in extra_feature_values.items():
                mapping[k] = np.asarray(v, dtype=float)
        x_cols = []
        for name in feature_order:
            if name in mapping:
                x_cols.append(mapping[name])
            else:
                x_cols.append(np.zeros_like(structural, dtype=float))
        x = np.column_stack(x_cols)
    else:
        stage2_features = int(getattr(stage2_model, "n_features_in_", 3))
        if stage2_features >= 3:
            x = np.column_stack([residual_store, structural, lag1])
        elif stage2_features == 2:
            x = np.column_stack([residual_store, structural])
        else:
            x = residual_store.reshape(-1, 1)
    return stage2_model.predict(x)


class _Stage2CoeffModel:
    def __init__(self, intercept: float, coefs: list[float], feature_order: list[str]):
        self.intercept_ = float(intercept)
        self.coef_ = np.asarray(coefs, dtype=float)
        self.feature_order_ = list(feature_order)
        self.n_features_in_ = len(self.feature_order_)

    def predict(self, X):
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] != len(self.coef_):
            raise ValueError(
                f"Feature mismatch in _Stage2CoeffModel: got {arr.shape[1]}, expected {len(self.coef_)}"
            )
        return self.intercept_ + arr @ self.coef_


def _build_stage2_model_from_coefficients(coeff: dict | None):
    if not isinstance(coeff, dict):
        return None
    feature_order = coeff.get("feature_order", [])
    if not feature_order:
        return None
    intercept = coeff.get("stage2_intercept", None)
    if intercept is None or not np.isfinite(float(intercept)):
        return None
    coefs: list[float] = []
    for feat in feature_order:
        v = coeff.get(f"coef_{feat}", np.nan)
        if not np.isfinite(float(v)):
            return None
        coefs.append(float(v))
    return _Stage2CoeffModel(intercept=float(intercept), coefs=coefs, feature_order=feature_order)


def _build_stage2_ols_model_from_coefficients(coeff: dict | None):
    if not isinstance(coeff, dict):
        return None
    feature_order = coeff.get("feature_order", [])
    if not feature_order:
        return None
    intercept = coeff.get("stage2_ols_intercept", None)
    if intercept is None or not np.isfinite(float(intercept)):
        return None
    coefs: list[float] = []
    for feat in feature_order:
        v = coeff.get(f"stage2_ols_coef_{feat}", np.nan)
        if not np.isfinite(float(v)):
            return None
        coefs.append(float(v))
    return _Stage2CoeffModel(intercept=float(intercept), coefs=coefs, feature_order=feature_order)


def _build_stage2_constrained_model_from_coefficients(coeff: dict | None):
    if not isinstance(coeff, dict):
        return None
    feature_order = coeff.get("feature_order", [])
    if not feature_order:
        return None

    intercept = coeff.get("stage2_constrained_intercept", np.nan)
    if not np.isfinite(float(intercept)):
        # Backward-compatible fallback: if selected model itself is constrained.
        if str(coeff.get("model_selected", "")).strip().lower() == "constrained_ridge":
            intercept = coeff.get("stage2_intercept", np.nan)
            coef_values = [coeff.get(f"coef_{feat}", np.nan) for feat in feature_order]
        else:
            return None
    else:
        coef_values = [coeff.get(f"stage2_constrained_coef_{feat}", np.nan) for feat in feature_order]

    coefs: list[float] = []
    for v in coef_values:
        if not np.isfinite(float(v)):
            return None
        coefs.append(float(v))
    return _Stage2CoeffModel(intercept=float(intercept), coefs=coefs, feature_order=feature_order)


def build_monthly_model_df(defined_df: pd.DataFrame, base_df: pd.DataFrame, size_value: str, slab_value: str) -> pd.DataFrame:
    scope = defined_df[
        (defined_df["Sizes"] == size_value)
        & (defined_df["Defined_Slab"].astype(str) == str(slab_value))
    ].copy()
    if scope.empty:
        return pd.DataFrame()

    monthly = (
        scope.groupby("Month", as_index=False)
        .agg(
            store_count=("Outlet_ID", "nunique"),
            quantity=("Quantity", "sum"),
            total_discount=("TotalDiscount", "sum"),
            sales_value=("SalesValue_atBasicRate", "sum"),
        )
        .rename(columns={"Month": "Period"})
        .sort_values("Period")
    )
    monthly["actual_discount_pct"] = (
        (monthly["total_discount"] / monthly["sales_value"]) * 100.0
    ).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    base_map = base_df[
        (base_df["Sizes"] == size_value) & (base_df["Defined_Slab"].astype(str) == str(slab_value))
    ][["Month", "Estimated_Base_%"]].copy()
    base_map = base_map.rename(columns={"Month": "Period", "Estimated_Base_%": "base_discount_pct"})

    monthly = monthly.merge(base_map, on="Period", how="left")
    monthly["base_discount_pct"] = monthly["base_discount_pct"].ffill().bfill().fillna(0.0)
    monthly["tactical_discount_pct"] = (monthly["actual_discount_pct"] - monthly["base_discount_pct"]).clip(lower=0.0)
    monthly["lag1_base_discount_pct"] = monthly["base_discount_pct"].shift(1).fillna(monthly["base_discount_pct"])
    monthly["base_price"] = (
        monthly["sales_value"] / monthly["quantity"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return monthly.sort_values("Period").reset_index(drop=True)


def build_other_slabs_weighted_discount_series(
    base_df: pd.DataFrame,
    size_value: str,
    slab_value: str,
) -> pd.DataFrame:
    part = base_df[
        (base_df["Sizes"] == size_value)
        & (base_df["Defined_Slab"].astype(str) != str(slab_value))
        & (base_df["Defined_Slab"].astype(str) != "slab0")
    ].copy()
    if part.empty:
        return pd.DataFrame(columns=["Period", "other_slabs_weighted_base_discount_pct"])

    part["Monthly_Quantity"] = pd.to_numeric(part["Monthly_Quantity"], errors="coerce").fillna(0.0)
    part["Estimated_Base_%"] = pd.to_numeric(part["Estimated_Base_%"], errors="coerce").fillna(0.0)

    slab_weights = (
        part.groupby("Defined_Slab", as_index=False)["Monthly_Quantity"]
        .sum()
        .rename(columns={"Monthly_Quantity": "Period_Slab_Qty"})
    )
    total_qty = float(slab_weights["Period_Slab_Qty"].sum())
    if total_qty > 0:
        slab_weights["Fixed_Weight"] = slab_weights["Period_Slab_Qty"] / total_qty
    else:
        n = max(len(slab_weights), 1)
        slab_weights["Fixed_Weight"] = 1.0 / n

    part = part.merge(slab_weights[["Defined_Slab", "Fixed_Weight"]], on="Defined_Slab", how="left")
    part["Fixed_Weight"] = part["Fixed_Weight"].fillna(0.0)
    part["weighted_part"] = part["Estimated_Base_%"] * part["Fixed_Weight"]

    out = (
        part.groupby("Month", as_index=False)
        .agg(other_slabs_weighted_base_discount_pct=("weighted_part", "sum"))
        .rename(columns={"Month": "Period"})
        .sort_values("Period")
        .reset_index(drop=True)
    )
    return out


def get_fixed_slab_weight_map(base_df: pd.DataFrame, size_value: str) -> dict[str, float]:
    part = base_df[
        (base_df["Sizes"] == size_value) & (base_df["Defined_Slab"].astype(str) != "slab0")
    ].copy()
    if part.empty:
        return {}
    part["Monthly_Quantity"] = pd.to_numeric(part["Monthly_Quantity"], errors="coerce").fillna(0.0)
    slab_qty = (
        part.groupby("Defined_Slab", as_index=False)["Monthly_Quantity"]
        .sum()
        .rename(columns={"Monthly_Quantity": "Period_Slab_Qty"})
    )
    total_qty = float(slab_qty["Period_Slab_Qty"].sum())
    if total_qty <= 0:
        n = max(len(slab_qty), 1)
        slab_qty["Fixed_Weight"] = 1.0 / n
    else:
        slab_qty["Fixed_Weight"] = slab_qty["Period_Slab_Qty"] / total_qty
    return {
        str(row["Defined_Slab"]): float(row["Fixed_Weight"])
        for _, row in slab_qty.iterrows()
    }


def compute_other_weighted_discount_for_slab(
    target_slab: str,
    scenario_discount_map: dict[str, float],
    fixed_weight_map: dict[str, float],
) -> float:
    other_slabs = [s for s in fixed_weight_map.keys() if str(s) != str(target_slab)]
    if not other_slabs:
        return 0.0
    denom = float(sum(fixed_weight_map.get(s, 0.0) for s in other_slabs))
    if denom <= 0:
        return 0.0
    weighted = 0.0
    for s in other_slabs:
        w_excl = float(fixed_weight_map.get(s, 0.0)) / denom
        d = float(scenario_discount_map.get(s, 0.0))
        weighted += w_excl * d
    return float(weighted)


def build_monthly_model_df_new_strategy(
    defined_df: pd.DataFrame,
    base_df: pd.DataFrame,
    size_value: str,
    slab_value: str,
) -> pd.DataFrame:
    monthly = build_monthly_model_df(defined_df, base_df, size_value, slab_value)
    if monthly.empty:
        return monthly

    other_series = build_other_slabs_weighted_discount_series(base_df, size_value, slab_value)
    monthly = monthly.merge(other_series, on="Period", how="left")
    monthly["other_slabs_weighted_base_discount_pct"] = (
        monthly["other_slabs_weighted_base_discount_pct"].ffill().bfill().fillna(0.0)
    )
    return monthly.sort_values("Period").reset_index(drop=True)


def run_two_stage_model(
    monthly: pd.DataFrame,
    include_lag_discount: bool = True,
    l2_penalty: float = 0.1,
    optimize_l2_penalty: bool = True,
):
    if monthly is None or monthly.empty or len(monthly) < 3:
        return None

    x_discount = monthly["actual_discount_pct"].to_numpy(dtype=float)
    y_qty = monthly["quantity"].to_numpy(dtype=float)
    y_store = monthly["store_count"].to_numpy(dtype=float)
    base = monthly["base_discount_pct"].to_numpy(dtype=float)
    tactical = monthly["tactical_discount_pct"].to_numpy(dtype=float)
    lag1 = monthly["lag1_base_discount_pct"].to_numpy(dtype=float)

    stage1 = LinearRegression()
    stage1.fit(x_discount.reshape(-1, 1), y_store)
    store_pred = stage1.predict(x_discount.reshape(-1, 1))
    residual_store = y_store - store_pred

    include_lag_discount = bool(include_lag_discount)
    feature_order = ["residual_store", "base_discount_pct"]
    if include_lag_discount:
        X2 = np.column_stack([residual_store, base, lag1])
        feature_order.append("lag1_base_discount_pct")
    else:
        X2 = np.column_stack([residual_store, base])

    non_negative_indices = [0, 1]
    non_positive_indices = [2] if include_lag_discount else []

    def _fit_stage2_with_l2(lam: float):
        model = CustomConstrainedRidge(
            l2_penalty=float(lam),
            non_negative_indices=non_negative_indices,
            non_positive_indices=non_positive_indices,
            maxiter=4000,
        )
        model.fit(X2, y_qty)
        model.feature_order_ = feature_order
        preds = model.predict(X2)
        if len(y_qty) > 1:
            ss_res = float(np.sum((y_qty - preds) ** 2))
            ss_tot = float(np.sum((y_qty - np.mean(y_qty)) ** 2))
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
        else:
            r2 = 0.0
        return model, preds, float(r2)

    def _cv_r2_for_l2(lam: float):
        n = len(y_qty)
        p = X2.shape[1]
        min_train = max(p + 2, 8)
        if n <= (min_train + 1):
            return np.nan
        y_true = []
        y_hat = []
        for split in range(min_train, n):
            X_train = X2[:split]
            y_train = y_qty[:split]
            X_val = X2[split : split + 1]
            y_val = y_qty[split : split + 1]
            model = CustomConstrainedRidge(
                l2_penalty=float(lam),
                non_negative_indices=non_negative_indices,
                non_positive_indices=non_positive_indices,
                maxiter=4000,
            )
            try:
                model.fit(X_train, y_train)
                model.feature_order_ = feature_order
                pred = model.predict(X_val)
            except Exception:
                continue
            if pred is None or len(pred) != 1:
                continue
            y_true.append(float(y_val[0]))
            y_hat.append(float(pred[0]))
        if len(y_true) < 3:
            return np.nan
        y_true = np.asarray(y_true, dtype=float)
        y_hat = np.asarray(y_hat, dtype=float)
        ss_res = float(np.sum((y_true - y_hat) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        if ss_tot <= 1e-12:
            return np.nan
        return 1.0 - (ss_res / ss_tot)

    if optimize_l2_penalty:
        l2_candidates = [0.0, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
        best = None
        for lam in l2_candidates:
            model_tmp, preds_tmp, r2_tmp = _fit_stage2_with_l2(lam)
            cv_r2_tmp = _cv_r2_for_l2(lam)
            score = cv_r2_tmp if np.isfinite(cv_r2_tmp) else -np.inf
            if (best is None) or (score > best["score"]):
                best = {
                    "model": model_tmp,
                    "preds": preds_tmp,
                    "r2_train": r2_tmp,
                    "cv_r2": float(cv_r2_tmp) if np.isfinite(cv_r2_tmp) else np.nan,
                    "score": score,
                    "l2": float(lam),
                }
        stage2 = best["model"]
        qty_pred = best["preds"]
        r2_stage2 = float(best["r2_train"])
        r2_stage2_cv = best["cv_r2"]
        l2_used = float(best["l2"])
    else:
        stage2, qty_pred, r2_stage2 = _fit_stage2_with_l2(l2_penalty)
        r2_stage2_cv = float(_cv_r2_for_l2(l2_penalty))
        l2_used = float(l2_penalty)

    stage2_ols = LinearRegression()
    stage2_ols.fit(X2, y_qty)
    stage2_ols.feature_order_ = feature_order
    qty_pred_ols = stage2_ols.predict(X2)
    if len(y_qty) > 1:
        ss_tot = float(np.sum((y_qty - np.mean(y_qty)) ** 2))
        ss_res_ols = float(np.sum((y_qty - qty_pred_ols) ** 2))
        r2_ols = 1 - (ss_res_ols / ss_tot) if ss_tot > 1e-12 else 0.0
    else:
        r2_ols = 0.0

    def _ols_respects_sign_rules(coef_values: np.ndarray, tol: float = 1e-9) -> tuple[bool, list[str]]:
        violations: list[str] = []
        for idx in non_negative_indices:
            if idx < len(coef_values) and float(coef_values[idx]) < -tol:
                feat = feature_order[idx] if idx < len(feature_order) else f"idx_{idx}"
                violations.append(f"{feat}<{0}")
        for idx in non_positive_indices:
            if idx < len(coef_values) and float(coef_values[idx]) > tol:
                feat = feature_order[idx] if idx < len(feature_order) else f"idx_{idx}"
                violations.append(f"{feat}>{0}")
        return (len(violations) == 0), violations

    ols_constraints_ok, ols_constraint_violations = _ols_respects_sign_rules(stage2_ols.coef_)
    model_selected = "ols" if ols_constraints_ok else "constrained_ridge"
    stage2_selected = stage2_ols if ols_constraints_ok else stage2
    qty_pred_selected = qty_pred_ols if ols_constraints_ok else qty_pred
    r2_stage2_selected = float(r2_ols) if ols_constraints_ok else float(r2_stage2)
    cv_r2_selected = np.nan if ols_constraints_ok else r2_stage2_cv
    l2_used_selected = 0.0 if ols_constraints_ok else float(l2_used)

    qty_base_constrained = _predict_stage2_quantity(stage2, residual_store, base, np.zeros_like(base), lag1)
    qty_base_ols = _predict_stage2_quantity(stage2_ols, residual_store, base, np.zeros_like(base), lag1)
    qty_base_selected = qty_base_ols if ols_constraints_ok else qty_base_constrained
    zeros = np.zeros_like(base, dtype=float)
    qty_no_discount_constrained = _predict_stage2_quantity(stage2, residual_store, zeros, zeros, zeros)
    qty_no_discount_ols = _predict_stage2_quantity(stage2_ols, residual_store, zeros, zeros, zeros)
    qty_no_discount_selected = qty_no_discount_ols if ols_constraints_ok else qty_no_discount_constrained
    qty_no_discount_selected = np.maximum(qty_no_discount_selected, 0.0)

    actual_price = monthly["base_price"].to_numpy(dtype=float) * (1 - x_discount / 100.0)
    baseline_price = monthly["base_price"].to_numpy(dtype=float) * (1 - base / 100.0)
    predicted_revenue = qty_pred_selected * actual_price
    baseline_revenue = qty_base_selected * baseline_price
    spend = monthly["base_price"].to_numpy(dtype=float) * (tactical / 100.0) * qty_pred_selected
    incremental_revenue = predicted_revenue - baseline_revenue
    roi = np.full(len(spend), np.nan, dtype=float)
    mask_spend = spend > 0
    roi[mask_spend] = incremental_revenue[mask_spend] / spend[mask_spend]

    model_df = monthly.copy()
    model_df["predicted_quantity"] = qty_pred_selected
    model_df["baseline_quantity"] = qty_base_selected
    model_df["predicted_quantity_constrained"] = qty_pred
    model_df["baseline_quantity_constrained"] = qty_base_constrained
    model_df["predicted_quantity_ols"] = qty_pred_ols
    model_df["baseline_quantity_ols"] = qty_base_ols
    model_df["non_discount_baseline_quantity"] = qty_no_discount_selected
    model_df["non_discount_baseline_quantity_constrained"] = qty_no_discount_constrained
    model_df["non_discount_baseline_quantity_ols"] = qty_no_discount_ols
    model_df["predicted_revenue"] = predicted_revenue
    model_df["baseline_revenue"] = baseline_revenue
    model_df["spend"] = spend
    model_df["incremental_revenue"] = incremental_revenue
    model_df["roi_1mo"] = roi
    model_df["residual_store"] = residual_store

    coefficients = {
        "stage1_intercept": float(stage1.intercept_),
        "stage1_coef_discount": float(stage1.coef_[0]),
        "stage1_r2": float(stage1.score(x_discount.reshape(-1, 1), y_store)) if len(y_store) > 1 else 0.0,
        "model_selected": model_selected,
        "ols_constraints_respected": bool(ols_constraints_ok),
        "ols_constraint_violations": "; ".join(ols_constraint_violations),
        "stage2_intercept": float(stage2_selected.intercept_),
        "coef_residual_store": float(stage2_selected.coef_[0]),
        "coef_structural_discount": float(stage2_selected.coef_[1]),
        "coef_tactical_discount": 0.0,
        "coef_lag1_structural_discount": float(stage2_selected.coef_[2]) if len(stage2_selected.coef_) >= 3 else 0.0,
        "stage2_r2": float(r2_stage2_selected),
        "stage2_cv_r2": float(cv_r2_selected) if np.isfinite(cv_r2_selected) else np.nan,
        "l2_used": float(l2_used_selected),
        "stage2_constrained_intercept": float(stage2.intercept_),
        "stage2_constrained_r2": float(r2_stage2),
        "stage2_constrained_cv_r2": float(r2_stage2_cv) if np.isfinite(r2_stage2_cv) else np.nan,
        "stage2_constrained_coef_residual_store": float(stage2.coef_[0]),
        "stage2_constrained_coef_structural_discount": float(stage2.coef_[1]),
        "stage2_constrained_coef_lag1_structural_discount": float(stage2.coef_[2]) if len(stage2.coef_) >= 3 else 0.0,
        "stage2_ols_r2": float(r2_ols),
        "stage2_ols_intercept": float(stage2_ols.intercept_),
        "stage2_ols_coef_residual_store": float(stage2_ols.coef_[0]),
        "stage2_ols_coef_structural_discount": float(stage2_ols.coef_[1]),
        "stage2_ols_coef_tactical_discount": 0.0,
        "stage2_ols_coef_lag1_structural_discount": float(stage2_ols.coef_[2]) if len(stage2_ols.coef_) >= 3 else 0.0,
    }
    return {"model_df": model_df, "coefficients": coefficients, "stage2_model": stage2_selected}


def run_two_stage_model_new_strategy(
    monthly: pd.DataFrame,
    include_lag_discount: bool = True,
    l2_penalty: float = 0.1,
    optimize_l2_penalty: bool = True,
):
    if monthly is None or monthly.empty or len(monthly) < 3:
        return None

    x_discount = monthly["actual_discount_pct"].to_numpy(dtype=float)
    y_qty = monthly["quantity"].to_numpy(dtype=float)
    y_store = monthly["store_count"].to_numpy(dtype=float)
    base = monthly["base_discount_pct"].to_numpy(dtype=float)
    tactical = monthly["tactical_discount_pct"].to_numpy(dtype=float)
    lag1 = monthly["lag1_base_discount_pct"].to_numpy(dtype=float)
    other = monthly["other_slabs_weighted_base_discount_pct"].to_numpy(dtype=float)

    stage1 = LinearRegression()
    stage1.fit(x_discount.reshape(-1, 1), y_store)
    store_pred = stage1.predict(x_discount.reshape(-1, 1))
    residual_store = y_store - store_pred

    feature_order = ["residual_store", "base_discount_pct"]
    x_cols = [residual_store, base]
    non_negative_indices = [0, 1]
    non_positive_indices = []

    if include_lag_discount:
        feature_order.append("lag1_base_discount_pct")
        x_cols.append(lag1)
        non_positive_indices.append(len(feature_order) - 1)

    feature_order.append("other_slabs_weighted_base_discount_pct")
    x_cols.append(other)
    # Constrain cross-slab effect as non-positive (cannibalization effect).
    non_positive_indices.append(len(feature_order) - 1)

    X2 = np.column_stack(x_cols)
    l2_floor = 0.1
    l2_penalty = max(float(l2_penalty), l2_floor)

    def _fit_stage2_with_l2(lam: float):
        model = CustomConstrainedRidge(
            l2_penalty=float(lam),
            non_negative_indices=non_negative_indices,
            non_positive_indices=non_positive_indices,
            maxiter=4000,
        )
        model.fit(X2, y_qty)
        model.feature_order_ = feature_order
        preds = model.predict(X2)
        if len(y_qty) > 1:
            ss_res = float(np.sum((y_qty - preds) ** 2))
            ss_tot = float(np.sum((y_qty - np.mean(y_qty)) ** 2))
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
        else:
            r2 = 0.0
        return model, preds, float(r2)

    def _cv_r2_for_l2(lam: float):
        n = len(y_qty)
        p = X2.shape[1]
        min_train = max(p + 2, 8)
        if n <= (min_train + 1):
            return np.nan
        y_true = []
        y_hat = []
        for split in range(min_train, n):
            X_train = X2[:split]
            y_train = y_qty[:split]
            X_val = X2[split : split + 1]
            y_val = y_qty[split : split + 1]
            model = CustomConstrainedRidge(
                l2_penalty=float(lam),
                non_negative_indices=non_negative_indices,
                non_positive_indices=non_positive_indices,
                maxiter=4000,
            )
            try:
                model.fit(X_train, y_train)
                model.feature_order_ = feature_order
                pred = model.predict(X_val)
            except Exception:
                continue
            if pred is None or len(pred) != 1:
                continue
            y_true.append(float(y_val[0]))
            y_hat.append(float(pred[0]))
        if len(y_true) < 3:
            return np.nan
        y_true = np.asarray(y_true, dtype=float)
        y_hat = np.asarray(y_hat, dtype=float)
        ss_res = float(np.sum((y_true - y_hat) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        if ss_tot <= 1e-12:
            return np.nan
        return 1.0 - (ss_res / ss_tot)

    if optimize_l2_penalty:
        l2_candidates = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
        best = None
        for lam in l2_candidates:
            model_tmp, preds_tmp, r2_tmp = _fit_stage2_with_l2(lam)
            cv_r2_tmp = _cv_r2_for_l2(lam)
            score = cv_r2_tmp if np.isfinite(cv_r2_tmp) else -np.inf
            if (best is None) or (score > best["score"]):
                best = {
                    "model": model_tmp,
                    "preds": preds_tmp,
                    "r2_train": r2_tmp,
                    "cv_r2": float(cv_r2_tmp) if np.isfinite(cv_r2_tmp) else np.nan,
                    "score": score,
                    "l2": float(lam),
                }
        stage2 = best["model"]
        qty_pred = best["preds"]
        r2_stage2 = float(best["r2_train"])
        r2_stage2_cv = best["cv_r2"]
        l2_used = float(best["l2"])
    else:
        stage2, qty_pred, r2_stage2 = _fit_stage2_with_l2(l2_penalty)
        r2_stage2_cv = float(_cv_r2_for_l2(l2_penalty))
        l2_used = float(l2_penalty)

    stage2_ols = LinearRegression()
    stage2_ols.fit(X2, y_qty)
    stage2_ols.feature_order_ = feature_order
    qty_pred_ols = stage2_ols.predict(X2)
    if len(y_qty) > 1:
        ss_tot = float(np.sum((y_qty - np.mean(y_qty)) ** 2))
        ss_res_ols = float(np.sum((y_qty - qty_pred_ols) ** 2))
        r2_ols = 1 - (ss_res_ols / ss_tot) if ss_tot > 1e-12 else 0.0
    else:
        r2_ols = 0.0

    def _ols_respects_sign_rules(coef_values: np.ndarray, tol: float = 1e-9) -> tuple[bool, list[str]]:
        violations: list[str] = []
        for idx in non_negative_indices:
            if idx < len(coef_values) and float(coef_values[idx]) < -tol:
                feat = feature_order[idx] if idx < len(feature_order) else f"idx_{idx}"
                violations.append(f"{feat}<{0}")
        for idx in non_positive_indices:
            if idx < len(coef_values) and float(coef_values[idx]) > tol:
                feat = feature_order[idx] if idx < len(feature_order) else f"idx_{idx}"
                violations.append(f"{feat}>{0}")
        return (len(violations) == 0), violations

    ols_constraints_ok, ols_constraint_violations = _ols_respects_sign_rules(stage2_ols.coef_)
    model_selected = "constrained_ridge"
    stage2_selected = stage2
    qty_pred_selected = qty_pred
    r2_stage2_selected = float(r2_stage2)
    cv_r2_selected = r2_stage2_cv
    l2_used_selected = float(l2_used)

    qty_base_constrained = _predict_stage2_quantity(
        stage2,
        residual_store,
        base,
        np.zeros_like(base),
        lag1,
        extra_feature_values={"other_slabs_weighted_base_discount_pct": other},
    )
    qty_base_ols = _predict_stage2_quantity(
        stage2_ols,
        residual_store,
        base,
        np.zeros_like(base),
        lag1,
        extra_feature_values={"other_slabs_weighted_base_discount_pct": other},
    )
    qty_base_selected = qty_base_constrained

    zeros = np.zeros_like(base, dtype=float)
    qty_no_discount_constrained = _predict_stage2_quantity(
        stage2,
        residual_store,
        zeros,
        zeros,
        zeros,
        extra_feature_values={"other_slabs_weighted_base_discount_pct": zeros},
    )
    qty_no_discount_ols = _predict_stage2_quantity(
        stage2_ols,
        residual_store,
        zeros,
        zeros,
        zeros,
        extra_feature_values={"other_slabs_weighted_base_discount_pct": zeros},
    )
    qty_no_discount_selected = qty_no_discount_constrained
    qty_no_discount_selected = np.maximum(qty_no_discount_selected, 0.0)

    actual_price = monthly["base_price"].to_numpy(dtype=float) * (1 - x_discount / 100.0)
    baseline_price = monthly["base_price"].to_numpy(dtype=float) * (1 - base / 100.0)
    predicted_revenue = qty_pred_selected * actual_price
    baseline_revenue = qty_base_selected * baseline_price
    spend = monthly["base_price"].to_numpy(dtype=float) * (tactical / 100.0) * qty_pred_selected
    incremental_revenue = predicted_revenue - baseline_revenue
    roi = np.full(len(spend), np.nan, dtype=float)
    mask_spend = spend > 0
    roi[mask_spend] = incremental_revenue[mask_spend] / spend[mask_spend]

    model_df = monthly.copy()
    model_df["predicted_quantity"] = qty_pred_selected
    model_df["baseline_quantity"] = qty_base_selected
    model_df["predicted_quantity_constrained"] = qty_pred
    model_df["baseline_quantity_constrained"] = qty_base_constrained
    model_df["predicted_quantity_ols"] = qty_pred_ols
    model_df["baseline_quantity_ols"] = qty_base_ols
    model_df["non_discount_baseline_quantity"] = qty_no_discount_selected
    model_df["non_discount_baseline_quantity_constrained"] = qty_no_discount_constrained
    model_df["non_discount_baseline_quantity_ols"] = qty_no_discount_ols
    model_df["predicted_revenue"] = predicted_revenue
    model_df["baseline_revenue"] = baseline_revenue
    model_df["spend"] = spend
    model_df["incremental_revenue"] = incremental_revenue
    model_df["roi_1mo"] = roi
    model_df["residual_store"] = residual_store

    coefficients = {
        "stage1_intercept": float(stage1.intercept_),
        "stage1_coef_discount": float(stage1.coef_[0]),
        "stage1_r2": float(stage1.score(x_discount.reshape(-1, 1), y_store)) if len(y_store) > 1 else 0.0,
        "model_selected": model_selected,
        "ols_constraints_respected": bool(ols_constraints_ok),
        "ols_constraint_violations": "; ".join(ols_constraint_violations),
        "stage2_intercept": float(stage2_selected.intercept_),
        "stage2_r2": float(r2_stage2_selected),
        "stage2_cv_r2": float(cv_r2_selected) if np.isfinite(cv_r2_selected) else np.nan,
        "l2_used": float(l2_used_selected),
        "stage2_constrained_intercept": float(stage2.intercept_),
        "stage2_constrained_r2": float(r2_stage2),
        "stage2_constrained_cv_r2": float(r2_stage2_cv) if np.isfinite(r2_stage2_cv) else np.nan,
        "stage2_ols_r2": float(r2_ols),
        "stage2_ols_intercept": float(stage2_ols.intercept_),
        "feature_order": feature_order,
    }
    for idx, feat in enumerate(feature_order):
        coef_key = f"coef_{feat}"
        ols_key = f"stage2_ols_coef_{feat}"
        constrained_key = f"stage2_constrained_coef_{feat}"
        coefficients[coef_key] = float(stage2_selected.coef_[idx]) if len(stage2_selected.coef_) > idx else np.nan
        coefficients[constrained_key] = float(stage2.coef_[idx]) if len(stage2.coef_) > idx else np.nan
        coefficients[ols_key] = float(stage2_ols.coef_[idx]) if len(stage2_ols.coef_) > idx else np.nan
    return {"model_df": model_df, "coefficients": coefficients, "stage2_model": stage2_selected}


def _build_size_monthly_for_combined(defined_df: pd.DataFrame, base_df: pd.DataFrame, size_value: str) -> pd.DataFrame:
    scope = defined_df[defined_df["Sizes"] == size_value].copy()
    if scope.empty:
        return pd.DataFrame()

    monthly = (
        scope.groupby("Month", as_index=False)
        .agg(
            store_count=("Outlet_ID", "nunique"),
            quantity=("Quantity", "sum"),
            total_discount=("TotalDiscount", "sum"),
            sales_value=("SalesValue_atBasicRate", "sum"),
        )
        .rename(columns={"Month": "Period"})
        .sort_values("Period")
        .reset_index(drop=True)
    )
    monthly["actual_discount_pct"] = np.where(
        monthly["sales_value"] > 0,
        (monthly["total_discount"] / monthly["sales_value"]) * 100.0,
        0.0,
    )

    base_scope = base_df[base_df["Sizes"] == size_value].copy()
    if not base_scope.empty:
        slab_qty = (
            base_scope.groupby("Defined_Slab", as_index=False)["Monthly_Quantity"]
            .sum()
            .rename(columns={"Monthly_Quantity": "Period_Slab_Qty"})
        )
        total_period_qty = float(slab_qty["Period_Slab_Qty"].sum())
        if total_period_qty > 0:
            slab_qty["Fixed_Weight"] = slab_qty["Period_Slab_Qty"] / total_period_qty
        else:
            n = max(len(slab_qty), 1)
            slab_qty["Fixed_Weight"] = 1.0 / n

        base_scope = base_scope.merge(slab_qty[["Defined_Slab", "Fixed_Weight"]], on="Defined_Slab", how="left")
        base_scope["Fixed_Weight"] = base_scope["Fixed_Weight"].fillna(0.0)
        base_scope["weighted_base_part"] = base_scope["Estimated_Base_%"] * base_scope["Fixed_Weight"]
        base_month = (
            base_scope.groupby("Month", as_index=False)
            .agg(base_discount_pct=("weighted_base_part", "sum"))
            .rename(columns={"Month": "Period"})
        )
        monthly = monthly.merge(base_month[["Period", "base_discount_pct"]], on="Period", how="left")
    else:
        monthly["base_discount_pct"] = np.nan

    monthly["base_discount_pct"] = monthly["base_discount_pct"].ffill().bfill().fillna(0.0)
    monthly["lag1_base_discount_pct"] = monthly["base_discount_pct"].shift(1).fillna(monthly["base_discount_pct"])

    if "MRP" in scope.columns:
        mrp_work = scope[["Month", "MRP", "Quantity"]].copy()
        mrp_work["MRP"] = pd.to_numeric(mrp_work["MRP"], errors="coerce")
        mrp_work["Quantity"] = pd.to_numeric(mrp_work["Quantity"], errors="coerce").fillna(0.0)
        mrp_work = mrp_work.dropna(subset=["MRP"])
        if not mrp_work.empty:
            mrp_work["mrp_x_qty"] = mrp_work["MRP"] * mrp_work["Quantity"]
            mrp_month = (
                mrp_work.groupby("Month", as_index=False)
                .agg(sum_mrp_qty=("mrp_x_qty", "sum"), sum_qty=("Quantity", "sum"))
                .rename(columns={"Month": "Period"})
            )
            mrp_month["mrp_weighted"] = np.where(
                mrp_month["sum_qty"] > 0,
                mrp_month["sum_mrp_qty"] / mrp_month["sum_qty"],
                np.nan,
            )
            monthly = monthly.merge(mrp_month[["Period", "mrp_weighted"]], on="Period", how="left")
        else:
            monthly["mrp_weighted"] = np.nan
    else:
        monthly["mrp_weighted"] = np.nan

    if monthly["mrp_weighted"].isna().all():
        monthly["mrp_weighted"] = np.where(
            monthly["quantity"] > 0,
            monthly["sales_value"] / monthly["quantity"],
            np.nan,
        )
    monthly["mrp_weighted"] = monthly["mrp_weighted"].ffill().bfill().fillna(0.0)
    return monthly


def _add_stage1_residual(df_size: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    work = df_size.copy().sort_values("Period").reset_index(drop=True)
    if work.empty:
        return work, {"stage1_intercept": 0.0, "stage1_coef_discount": 0.0, "stage1_r2": np.nan}
    x = work["actual_discount_pct"].to_numpy(dtype=float).reshape(-1, 1)
    y = work["store_count"].to_numpy(dtype=float)
    if len(work) < 2:
        work["store_hat"] = y
        work["residual_store"] = 0.0
        return work, {"stage1_intercept": float(y[0]) if len(y) else 0.0, "stage1_coef_discount": 0.0, "stage1_r2": np.nan}

    s1 = LinearRegression()
    s1.fit(x, y)
    hat = s1.predict(x)
    work["store_hat"] = hat
    work["residual_store"] = y - hat
    r2 = float(s1.score(x, y)) if len(y) > 1 else np.nan
    return work, {
        "stage1_intercept": float(s1.intercept_),
        "stage1_coef_discount": float(s1.coef_[0]),
        "stage1_r2": r2,
    }


def run_combined_two_size_model(
    defined_df: pd.DataFrame,
    base_df: pd.DataFrame,
    constraint_map: dict[str, str] | None = None,
    l2_selection_mode: str = "train_r2",
    min_l2_train_floor: float = 0.1,
):
    m12 = _build_size_monthly_for_combined(defined_df, base_df, "12-ML")
    m18 = _build_size_monthly_for_combined(defined_df, base_df, "18-ML")
    if m12.empty or m18.empty:
        return None

    m12, s1_12 = _add_stage1_residual(m12)
    m18, s1_18 = _add_stage1_residual(m18)

    rename_12 = {
        "store_count": "store_count_12",
        "quantity": "quantity_12",
        "actual_discount_pct": "actual_discount_pct_12",
        "base_discount_pct": "base_discount_pct_12",
        "lag1_base_discount_pct": "lag1_base_discount_pct_12",
        "mrp_weighted": "mrp_12",
        "residual_store": "residual_store_12",
    }
    rename_18 = {
        "store_count": "store_count_18",
        "quantity": "quantity_18",
        "actual_discount_pct": "actual_discount_pct_18",
        "base_discount_pct": "base_discount_pct_18",
        "lag1_base_discount_pct": "lag1_base_discount_pct_18",
        "mrp_weighted": "mrp_18",
        "residual_store": "residual_store_18",
    }
    c12 = m12[["Period", *rename_12.keys()]].rename(columns=rename_12)
    c18 = m18[["Period", *rename_18.keys()]].rename(columns=rename_18)
    combined = c12.merge(c18, on="Period", how="inner").sort_values("Period").reset_index(drop=True)
    if combined.empty or len(combined) < 8:
        return None

    combined["combined_quantity"] = combined["quantity_12"] + combined["quantity_18"]
    combined["combined_volume_ml"] = (combined["quantity_12"] * 12.0) + (combined["quantity_18"] * 18.0)
    feature_order = [
        "residual_store_12",
        "residual_store_18",
        "base_discount_pct_12",
        "base_discount_pct_18",
        "lag1_base_discount_pct_12",
        "lag1_base_discount_pct_18",
        "mrp_12",
        "mrp_18",
    ]
    X = combined[feature_order].to_numpy(dtype=float)
    y = combined["combined_volume_ml"].to_numpy(dtype=float)

    default_constraint_map = {
        "residual_store_12": ">=0",
        "residual_store_18": ">=0",
        "base_discount_pct_12": ">=0",
        "base_discount_pct_18": ">=0",
        "lag1_base_discount_pct_12": "<=0",
        "lag1_base_discount_pct_18": "<=0",
        "mrp_12": "<=0",
        "mrp_18": "<=0",
    }
    merged_constraints = default_constraint_map.copy()
    if constraint_map:
        for k, v in constraint_map.items():
            if k in merged_constraints and v in {">=0", "<=0", "none"}:
                merged_constraints[k] = v

    non_negative_indices = [
        idx for idx, feat in enumerate(feature_order) if merged_constraints.get(feat, "none") == ">=0"
    ]
    non_positive_indices = [
        idx for idx, feat in enumerate(feature_order) if merged_constraints.get(feat, "none") == "<=0"
    ]

    def _fit_with_l2(lam: float):
        model = CustomConstrainedRidge(
            l2_penalty=float(lam),
            non_negative_indices=non_negative_indices,
            non_positive_indices=non_positive_indices,
            maxiter=4000,
        )
        model.fit(X, y)
        model.feature_order_ = feature_order
        preds = model.predict(X)
        if len(y) > 1:
            ss_res = float(np.sum((y - preds) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
        else:
            r2 = 0.0
        return model, preds, float(r2)

    def _cv_r2(lam: float):
        n = len(y)
        p = X.shape[1]
        min_train = max(p + 2, 8)
        if n <= (min_train + 1):
            return np.nan
        y_true = []
        y_hat = []
        for split in range(min_train, n):
            X_train = X[:split]
            y_train = y[:split]
            X_val = X[split : split + 1]
            y_val = y[split : split + 1]
            model = CustomConstrainedRidge(
                l2_penalty=float(lam),
                non_negative_indices=non_negative_indices,
                non_positive_indices=non_positive_indices,
                maxiter=4000,
            )
            try:
                model.fit(X_train, y_train)
                pred = model.predict(X_val)
            except Exception:
                continue
            if pred is None or len(pred) != 1:
                continue
            y_true.append(float(y_val[0]))
            y_hat.append(float(pred[0]))
        if len(y_true) < 3:
            return np.nan
        y_true = np.asarray(y_true, dtype=float)
        y_hat = np.asarray(y_hat, dtype=float)
        ss_res = float(np.sum((y_true - y_hat) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        if ss_tot <= 1e-12:
            return np.nan
        return 1.0 - (ss_res / ss_tot)

    mode = str(l2_selection_mode).strip().lower()
    if mode not in {"train_r2", "cv_r2"}:
        mode = "train_r2"

    l2_candidates = [0.0, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    if mode == "train_r2":
        floor = max(float(min_l2_train_floor), 0.0)
        l2_candidates = sorted(set([lam for lam in l2_candidates if lam >= floor] + [floor]))
    best = None
    for lam in l2_candidates:
        model_tmp, preds_tmp, r2_tmp = _fit_with_l2(lam)
        cv_r2_tmp = _cv_r2(lam)
        if mode == "train_r2":
            score = r2_tmp if np.isfinite(r2_tmp) else -np.inf
        else:
            score = cv_r2_tmp if np.isfinite(cv_r2_tmp) else -np.inf
        if (best is None) or (score > best["score"]):
            best = {
                "model": model_tmp,
                "preds": preds_tmp,
                "r2_train": r2_tmp,
                "cv_r2": float(cv_r2_tmp) if np.isfinite(cv_r2_tmp) else np.nan,
                "score": score,
                "l2": float(lam),
            }

    stage2 = best["model"]
    y_hat = best["preds"]
    r2_stage2 = float(best["r2_train"])
    r2_stage2_cv = best["cv_r2"]
    l2_used = float(best["l2"])

    ols = LinearRegression()
    ols.fit(X, y)
    y_hat_ols = ols.predict(X)
    if len(y) > 1:
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        ss_res_ols = float(np.sum((y - y_hat_ols) ** 2))
        r2_ols = 1 - (ss_res_ols / ss_tot) if ss_tot > 1e-12 else 0.0
    else:
        r2_ols = 0.0

    X_no_disc = combined[feature_order].copy()
    X_no_disc["base_discount_pct_12"] = 0.0
    X_no_disc["base_discount_pct_18"] = 0.0
    X_no_disc["lag1_base_discount_pct_12"] = 0.0
    X_no_disc["lag1_base_discount_pct_18"] = 0.0
    y_no_disc_raw = stage2.predict(X_no_disc.to_numpy(dtype=float))
    y_no_disc = np.maximum(y_no_disc_raw, 0.0)
    baseline_zero_clip_months = int(np.sum(y_no_disc_raw <= 0.0))
    baseline_zero_clip_pct = (baseline_zero_clip_months / len(y_no_disc_raw) * 100.0) if len(y_no_disc_raw) else 0.0

    model_df = combined.copy()
    model_df["predicted_volume_ml"] = y_hat
    model_df["predicted_volume_ml_ols"] = y_hat_ols
    model_df["non_discount_baseline_volume_ml"] = y_no_disc
    model_df["discount_impact_volume_ml"] = model_df["predicted_volume_ml"] - model_df["non_discount_baseline_volume_ml"]

    # Backward-compatible aliases for any existing downstream view logic.
    model_df["predicted_quantity"] = model_df["predicted_volume_ml"]
    model_df["predicted_quantity_ols"] = model_df["predicted_volume_ml_ols"]
    model_df["non_discount_baseline_quantity"] = model_df["non_discount_baseline_volume_ml"]
    model_df["discount_impact_quantity"] = model_df["discount_impact_volume_ml"]

    coef = {
        "stage1_12_intercept": s1_12.get("stage1_intercept", np.nan),
        "stage1_12_coef_discount": s1_12.get("stage1_coef_discount", np.nan),
        "stage1_12_r2": s1_12.get("stage1_r2", np.nan),
        "stage1_18_intercept": s1_18.get("stage1_intercept", np.nan),
        "stage1_18_coef_discount": s1_18.get("stage1_coef_discount", np.nan),
        "stage1_18_r2": s1_18.get("stage1_r2", np.nan),
        "stage2_intercept": float(stage2.intercept_),
        "stage2_r2": r2_stage2,
        "stage2_cv_r2": r2_stage2_cv,
        "stage2_ols_r2": float(r2_ols),
        "l2_used": l2_used,
        "l2_selection_mode": mode,
        "min_l2_train_floor": max(float(min_l2_train_floor), 0.0),
        "baseline_zero_clip_months": baseline_zero_clip_months,
        "baseline_zero_clip_pct": float(baseline_zero_clip_pct),
    }
    for feat in feature_order:
        coef[f"constraint_{feat}"] = merged_constraints.get(feat, "none")
    for i, name in enumerate(feature_order):
        coef[f"coef_{name}"] = float(stage2.coef_[i])
        coef[f"ols_coef_{name}"] = float(ols.coef_[i])
    coef["stage2_ols_intercept"] = float(ols.intercept_)

    return {"model_df": model_df, "coefficients": coef, "feature_order": feature_order}


def build_weighted_discount_detail(base_df: pd.DataFrame, size_value: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    part = base_df[base_df["Sizes"] == size_value].copy()
    if part.empty:
        return pd.DataFrame(), pd.DataFrame()

    part = part[
        ["Month", "Defined_Slab", "Monthly_Quantity", "Estimated_Base_%", "Actual_Discount_%"]
    ].copy()
    part["Monthly_Quantity"] = pd.to_numeric(part["Monthly_Quantity"], errors="coerce").fillna(0.0)
    part["Estimated_Base_%"] = pd.to_numeric(part["Estimated_Base_%"], errors="coerce").fillna(0.0)
    part["Actual_Discount_%"] = pd.to_numeric(part["Actual_Discount_%"], errors="coerce").fillna(0.0)

    slab_qty = (
        part.groupby("Defined_Slab", as_index=False)["Monthly_Quantity"]
        .sum()
        .rename(columns={"Monthly_Quantity": "Period_Slab_Qty"})
    )
    total_period_qty = float(slab_qty["Period_Slab_Qty"].sum())
    if total_period_qty > 0:
        slab_qty["Fixed_Weight"] = slab_qty["Period_Slab_Qty"] / total_period_qty
    else:
        n = max(len(slab_qty), 1)
        slab_qty["Fixed_Weight"] = 1.0 / n

    detail = part.merge(slab_qty, on="Defined_Slab", how="left")
    detail["Fixed_Weight"] = detail["Fixed_Weight"].fillna(0.0)
    detail["Weighted_Base_Contribution"] = detail["Fixed_Weight"] * detail["Estimated_Base_%"]
    detail["Weighted_Actual_Contribution"] = detail["Fixed_Weight"] * detail["Actual_Discount_%"]

    monthly = (
        detail.groupby("Month", as_index=False)
        .agg(
            Weighted_Base_Discount=("Weighted_Base_Contribution", "sum"),
            Weighted_Actual_Discount=("Weighted_Actual_Contribution", "sum"),
        )
        .sort_values("Month")
        .reset_index(drop=True)
    )

    detail = detail.sort_values(["Month", "Defined_Slab"]).reset_index(drop=True)
    detail["Fixed_Weight"] = detail["Fixed_Weight"].round(6)
    detail["Weighted_Base_Contribution"] = detail["Weighted_Base_Contribution"].round(4)
    detail["Weighted_Actual_Contribution"] = detail["Weighted_Actual_Contribution"].round(4)
    monthly["Weighted_Base_Discount"] = monthly["Weighted_Base_Discount"].round(4)
    monthly["Weighted_Actual_Discount"] = monthly["Weighted_Actual_Discount"].round(4)

    return detail, monthly


def build_structural_roi_points(model_df: pd.DataFrame, stage2_model, cogs_per_unit: float = 0.0, round_step: float = 0.5):
    if model_df is None or model_df.empty:
        return pd.DataFrame(), {
            "episodes": 0.0,
            "structural_roi_1mo": 0.0,
            "structural_profit_roi_1mo": 0.0,
            "total_spend": 0.0,
            "total_incremental_revenue": 0.0,
            "total_incremental_profit": 0.0,
        }

    period_df = model_df.sort_values("Period").reset_index(drop=True).copy()
    period_df = period_df.dropna(subset=["Period"])
    if period_df.empty:
        return pd.DataFrame(), {
            "episodes": 0.0,
            "structural_roi_1mo": 0.0,
            "structural_profit_roi_1mo": 0.0,
            "total_spend": 0.0,
            "total_incremental_revenue": 0.0,
            "total_incremental_profit": 0.0,
        }

    safe_step = max(float(round_step), 0.1)
    period_df["base_discount_pct"] = period_df["base_discount_pct"].apply(lambda x: _round_to_step(float(x), safe_step))
    regime_break = period_df["base_discount_pct"].diff().abs().fillna(0) > 1e-9
    period_df["regime_id"] = regime_break.cumsum()
    period_df["row_idx"] = np.arange(len(period_df))

    regimes = (
        period_df.groupby("regime_id", as_index=False)
        .agg(
            start_idx=("row_idx", "min"),
            end_idx=("row_idx", "max"),
            base_discount_pct=("base_discount_pct", "first"),
        )
        .sort_values("start_idx")
        .reset_index(drop=True)
    )

    roi_rows = []
    episode_rois = []
    episode_profit_rois = []
    total_spend = 0.0
    total_incremental_revenue = 0.0
    total_incremental_profit = 0.0
    cogs_per_unit = max(float(cogs_per_unit), 0.0)

    for i in range(1, len(regimes)):
        prev_base = float(regimes.loc[i - 1, "base_discount_pct"])
        curr_base = float(regimes.loc[i, "base_discount_pct"])
        step_up = curr_base - prev_base
        if step_up <= 0:
            continue
        start_idx = int(regimes.loc[i, "start_idx"])
        end_idx = int(regimes.loc[i, "end_idx"])
        hold_df = period_df.iloc[start_idx : end_idx + 1].copy()
        if hold_df.empty:
            continue

        n_rows = len(hold_df)
        residual_anchor = hold_df["residual_store"].to_numpy(dtype=float)
        base_price = hold_df["base_price"].to_numpy(dtype=float)
        actual_discount = hold_df["actual_discount_pct"].to_numpy(dtype=float)
        feature_order = getattr(stage2_model, "feature_order_", []) or []
        known_features = {
            "residual_store",
            "base_discount_pct",
            "tactical_discount_pct",
            "lag1_base_discount_pct",
        }
        extra_feature_values = {}
        for feat in feature_order:
            if feat in known_features:
                continue
            if feat in hold_df.columns:
                extra_feature_values[feat] = hold_df[feat].to_numpy(dtype=float)
        prev_struct = np.full(n_rows, prev_base, dtype=float)
        curr_struct = np.full(n_rows, curr_base, dtype=float)
        lag1_prev = np.full(n_rows, prev_base, dtype=float)
        lag1_curr = np.full(n_rows, curr_base, dtype=float)
        lag1_curr[0] = prev_base
        zeros = np.zeros(n_rows, dtype=float)

        qty_prev = _predict_stage2_quantity(
            stage2_model,
            residual_anchor,
            prev_struct,
            zeros,
            lag1_prev,
            extra_feature_values=extra_feature_values,
        )
        qty_curr = _predict_stage2_quantity(
            stage2_model,
            residual_anchor,
            curr_struct,
            zeros,
            lag1_curr,
            extra_feature_values=extra_feature_values,
        )
        qty_prev = np.maximum(qty_prev, 0.0)
        qty_curr = np.maximum(qty_curr, 0.0)

        baseline_price = base_price * (1 - prev_base / 100.0)
        current_price = base_price * (1 - prev_base / 100.0)
        baseline_revenue = qty_prev * baseline_price
        predicted_revenue = qty_curr * current_price
        incremental_revenue = predicted_revenue - baseline_revenue
        spend = base_price * (step_up / 100.0) * qty_curr
        baseline_profit = baseline_revenue - (cogs_per_unit * qty_prev)
        predicted_profit = predicted_revenue - (cogs_per_unit * qty_curr)
        incremental_profit = predicted_profit - baseline_profit

        spend_sum = float(np.nansum(spend))
        incr_sum = float(np.nansum(incremental_revenue))
        incr_profit_sum = float(np.nansum(incremental_profit))
        roi_1mo = float(incr_sum / spend_sum) if spend_sum > 0 else np.nan
        profit_roi_1mo = float(incr_profit_sum / spend_sum) if spend_sum > 0 else np.nan

        episode_rois.append(roi_1mo)
        episode_profit_rois.append(profit_roi_1mo)
        total_spend += spend_sum
        total_incremental_revenue += incr_sum
        total_incremental_profit += incr_profit_sum

        roi_rows.append(
            {
                "Period": hold_df.iloc[0]["Period"],
                "base_discount_pct": curr_base,
                "actual_discount_pct": float(actual_discount[0]) if len(actual_discount) else 0.0,
                "structural_roi_1mo": float(roi_1mo) if np.isfinite(roi_1mo) else np.nan,
                "profit_roi_1mo": float(profit_roi_1mo) if np.isfinite(profit_roi_1mo) else np.nan,
                "spend": float(spend_sum),
                "incremental_revenue": float(incr_sum),
                "incremental_profit": float(incr_profit_sum),
            }
        )

    points_df = pd.DataFrame(roi_rows).drop_duplicates(subset=["Period"]).sort_values("Period").reset_index(drop=True)
    summary = {
        "episodes": float(len(episode_rois)),
        "avg_roi_1mo": float(np.nanmean(episode_rois)) if len(episode_rois) else 0.0,
        "avg_profit_roi_1mo": float(np.nanmean(episode_profit_rois)) if len(episode_profit_rois) else 0.0,
        "total_spend": float(total_spend),
        "total_incremental_revenue": float(total_incremental_revenue),
        "total_incremental_profit": float(total_incremental_profit),
        "structural_roi_1mo": float(total_incremental_revenue / total_spend) if total_spend > 0 else 0.0,
        "structural_profit_roi_1mo": float(total_incremental_profit / total_spend) if total_spend > 0 else 0.0,
    }
    return points_df, summary


def run_modeling_all_slabs(
    defined_df: pd.DataFrame,
    base_df: pd.DataFrame,
    cogs_per_unit: float = 0.0,
    cogs_by_size: dict[str, float] | None = None,
    size_filter: str | None = None,
    include_lag_discount: bool = True,
    optimize_l2_penalty: bool = True,
    l2_penalty: float = 0.1,
):
    sizes = sorted(base_df["Sizes"].dropna().astype(str).unique().tolist())
    if size_filter:
        sizes = [s for s in sizes if s == size_filter]
    cogs_by_size = cogs_by_size or {}
    results: list[dict] = []
    summary_rows: list[dict] = []
    for size in sizes:
        slabs = sorted(
            base_df[base_df["Sizes"] == size]["Defined_Slab"].dropna().astype(str).unique().tolist(),
            key=_slab_sort_key,
        )
        for slab in slabs:
            monthly = build_monthly_model_df(defined_df, base_df, size, slab)
            if monthly.empty or len(monthly) < 3:
                results.append({"size": size, "slab": slab, "valid": False, "reason": "Not enough monthly points"})
                continue

            modeled = run_two_stage_model(
                monthly=monthly,
                include_lag_discount=include_lag_discount,
                l2_penalty=l2_penalty,
                optimize_l2_penalty=optimize_l2_penalty,
            )
            if modeled is None:
                results.append({"size": size, "slab": slab, "valid": False, "reason": "Model fit failed"})
                continue

            model_df = modeled["model_df"]
            coefficients = modeled["coefficients"]
            points_df, roi_summary = build_structural_roi_points(
                model_df=model_df,
                stage2_model=modeled["stage2_model"],
                cogs_per_unit=float(cogs_by_size.get(size, cogs_per_unit)),
                round_step=0.5,
            )

            result = {
                "size": size,
                "slab": slab,
                "valid": True,
                "monthly": monthly,
                "model_df": model_df,
                "coefficients": coefficients,
                "roi_points": points_df,
                "roi_summary": roi_summary,
            }
            results.append(result)
            summary_rows.append(
                {
                    "Size": size,
                    "Slab": slab,
                    "Months": len(monthly),
                    "Stage1_R2": coefficients.get("stage1_r2", np.nan),
                    "Stage2_R2": coefficients.get("stage2_r2", np.nan),
                    "OLS_R2": coefficients.get("stage2_ols_r2", np.nan),
                    "CV_R2": coefficients.get("stage2_cv_r2", np.nan),
                    "L2_Used": coefficients.get("l2_used", np.nan),
                    "Topline_ROI_x": roi_summary.get("structural_roi_1mo", np.nan),
                    "GrossMargin_ROI_x": roi_summary.get("structural_profit_roi_1mo", np.nan),
                    "Episodes": roi_summary.get("episodes", 0.0),
                }
            )

        # Additional size-level combined model (all slabs together),
        # with base discount built as fixed-weight average across slabs.
        monthly_combined = _build_size_monthly_for_combined(defined_df, base_df, size)
        if monthly_combined.empty or len(monthly_combined) < 3:
            results.append(
                {
                    "size": size,
                    "slab": "combined_all_slabs",
                    "valid": False,
                    "reason": "Not enough monthly points (combined)",
                }
            )
        else:
            monthly_combined = monthly_combined.copy()
            monthly_combined["tactical_discount_pct"] = (
                monthly_combined["actual_discount_pct"] - monthly_combined["base_discount_pct"]
            ).clip(lower=0.0)
            monthly_combined["base_price"] = (
                monthly_combined["sales_value"] / monthly_combined["quantity"].replace(0, np.nan)
            ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

            modeled_combined = run_two_stage_model(
                monthly=monthly_combined,
                include_lag_discount=include_lag_discount,
                l2_penalty=l2_penalty,
                optimize_l2_penalty=optimize_l2_penalty,
            )
            if modeled_combined is None:
                results.append(
                    {
                        "size": size,
                        "slab": "combined_all_slabs",
                        "valid": False,
                        "reason": "Model fit failed (combined)",
                    }
                )
            else:
                model_df_c = modeled_combined["model_df"]
                coefficients_c = modeled_combined["coefficients"]
                points_df_c, roi_summary_c = build_structural_roi_points(
                    model_df=model_df_c,
                    stage2_model=modeled_combined["stage2_model"],
                    cogs_per_unit=float(cogs_by_size.get(size, cogs_per_unit)),
                    round_step=0.5,
                )
                results.append(
                    {
                        "size": size,
                        "slab": "combined_all_slabs",
                        "valid": True,
                        "monthly": monthly_combined,
                        "model_df": model_df_c,
                        "coefficients": coefficients_c,
                        "roi_points": points_df_c,
                        "roi_summary": roi_summary_c,
                    }
                )
                summary_rows.append(
                    {
                        "Size": size,
                        "Slab": "combined_all_slabs",
                        "Months": len(monthly_combined),
                        "Stage1_R2": coefficients_c.get("stage1_r2", np.nan),
                        "Stage2_R2": coefficients_c.get("stage2_r2", np.nan),
                        "OLS_R2": coefficients_c.get("stage2_ols_r2", np.nan),
                        "CV_R2": coefficients_c.get("stage2_cv_r2", np.nan),
                        "L2_Used": coefficients_c.get("l2_used", np.nan),
                        "Topline_ROI_x": roi_summary_c.get("structural_roi_1mo", np.nan),
                        "GrossMargin_ROI_x": roi_summary_c.get("structural_profit_roi_1mo", np.nan),
                        "Episodes": roi_summary_c.get("episodes", 0.0),
                    }
                )
    summary_df = pd.DataFrame(summary_rows)
    return results, summary_df


def run_new_strategy_all_slabs(
    defined_df: pd.DataFrame,
    base_df: pd.DataFrame,
    cogs_per_unit: float = 0.0,
    cogs_by_size: dict[str, float] | None = None,
    include_lag_discount: bool = True,
    optimize_l2_penalty: bool = True,
    l2_penalty: float = 0.1,
):
    sizes = sorted(base_df["Sizes"].dropna().astype(str).unique().tolist())
    cogs_by_size = cogs_by_size or {}
    results: list[dict] = []
    summary_rows: list[dict] = []

    for size in sizes:
        slabs = sorted(
            base_df[base_df["Sizes"] == size]["Defined_Slab"].dropna().astype(str).unique().tolist(),
            key=_slab_sort_key,
        )
        slabs = [s for s in slabs if s != "slab0"]
        for slab in slabs:
            monthly = build_monthly_model_df_new_strategy(defined_df, base_df, size, slab)
            if monthly.empty or len(monthly) < 3:
                results.append(
                    {"size": size, "slab": slab, "valid": False, "reason": "Not enough monthly points"}
                )
                continue

            modeled = run_two_stage_model_new_strategy(
                monthly=monthly,
                include_lag_discount=include_lag_discount,
                l2_penalty=l2_penalty,
                optimize_l2_penalty=optimize_l2_penalty,
            )
            if modeled is None:
                results.append({"size": size, "slab": slab, "valid": False, "reason": "Model fit failed"})
                continue

            model_df = modeled["model_df"]
            coefficients = modeled["coefficients"]
            points_df, roi_summary = build_structural_roi_points(
                model_df=model_df,
                stage2_model=modeled["stage2_model"],
                cogs_per_unit=float(cogs_by_size.get(size, cogs_per_unit)),
                round_step=0.5,
            )

            results.append(
                {
                    "size": size,
                    "slab": slab,
                    "valid": True,
                    "monthly": monthly,
                    "model_df": model_df,
                    "coefficients": coefficients,
                    "stage2_model": modeled["stage2_model"],
                    "roi_points": points_df,
                    "roi_summary": roi_summary,
                }
            )
            summary_rows.append(
                {
                    "Size": size,
                    "Slab": slab,
                    "Months": len(monthly),
                    "Stage1_R2": coefficients.get("stage1_r2", np.nan),
                    "Stage2_R2": coefficients.get("stage2_r2", np.nan),
                    "OLS_R2": coefficients.get("stage2_ols_r2", np.nan),
                    "CV_R2": coefficients.get("stage2_cv_r2", np.nan),
                    "L2_Used": coefficients.get("l2_used", np.nan),
                    "Topline_ROI_x": roi_summary.get("structural_roi_1mo", np.nan),
                    "GrossMargin_ROI_x": roi_summary.get("structural_profit_roi_1mo", np.nan),
                    "Episodes": roi_summary.get("episodes", 0.0),
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    return results, summary_df


def _add_stage1_residual_column(monthly_df: pd.DataFrame) -> pd.DataFrame:
    out = monthly_df.copy()
    if out.empty or "actual_discount_pct" not in out.columns or "store_count" not in out.columns:
        out["residual_store"] = 0.0
        return out
    x = pd.to_numeric(out["actual_discount_pct"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    y = pd.to_numeric(out["store_count"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if len(out) < 2:
        out["residual_store"] = 0.0
        return out
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)
    y_hat = model.predict(x.reshape(-1, 1))
    out["residual_store"] = y - y_hat
    return out


def fit_cross_size_elasticity_model(defined_df: pd.DataFrame, base_df: pd.DataFrame) -> dict | None:
    m12 = _build_size_monthly_for_combined(defined_df, base_df, "12-ML")
    m18 = _build_size_monthly_for_combined(defined_df, base_df, "18-ML")
    if m12.empty or m18.empty:
        return None

    d12 = m12.rename(
        columns={
            "quantity": "quantity_12",
        }
    )[["Period", "quantity_12"]]
    d18 = m18.rename(
        columns={
            "quantity": "quantity_18",
        }
    )[["Period", "quantity_18"]]

    mm = d12.merge(d18, on="Period", how="inner").sort_values("Period").reset_index(drop=True)
    if mm.empty or len(mm) < 6:
        return None

    for c in ["quantity_12", "quantity_18"]:
        mm[c] = pd.to_numeric(mm[c], errors="coerce")
    mm = mm.dropna(subset=["quantity_12", "quantity_18"]).copy()
    mm = mm[(mm["quantity_12"] > 0) & (mm["quantity_18"] > 0)].copy()
    if mm.empty or len(mm) < 6:
        return None

    mm["pct_change_12"] = (
        mm["quantity_12"].pct_change().replace([np.inf, -np.inf], np.nan) * 100.0
    )
    mm["pct_change_18"] = (
        mm["quantity_18"].pct_change().replace([np.inf, -np.inf], np.nan) * 100.0
    )
    mm = mm.dropna(subset=["pct_change_12", "pct_change_18"]).copy()
    if mm.empty or len(mm) < 6:
        return None

    mdl12 = LinearRegression()
    mdl18 = LinearRegression()
    X12 = mm[["pct_change_18"]].to_numpy(dtype=float)
    y12 = mm["pct_change_12"].to_numpy(dtype=float)
    X18 = mm[["pct_change_12"]].to_numpy(dtype=float)
    y18 = mm["pct_change_18"].to_numpy(dtype=float)
    mdl12.fit(X12, y12)
    mdl18.fit(X18, y18)
    r2_12 = float(mdl12.score(X12, y12)) if len(y12) > 1 else np.nan
    r2_18 = float(mdl18.score(X18, y18)) if len(y18) > 1 else np.nan

    e12_from_18 = float(mdl12.coef_[0]) if len(mdl12.coef_) >= 1 else np.nan
    e18_from_12 = float(mdl18.coef_[0]) if len(mdl18.coef_) >= 1 else np.nan

    coef_df = pd.DataFrame(
        [
            {"Model": "Q12 model", "Feature": "Intercept", "Beta": float(mdl12.intercept_)},
            {"Model": "Q12 model", "Feature": "pct_change_18", "Beta": e12_from_18},
            {"Model": "Q18 model", "Feature": "Intercept", "Beta": float(mdl18.intercept_)},
            {"Model": "Q18 model", "Feature": "pct_change_12", "Beta": e18_from_12},
        ]
    )

    return {
        "months": int(len(mm)),
        "r2_12": r2_12,
        "r2_18": r2_18,
        "cross_elasticity_12_from_18": e12_from_18,
        "cross_elasticity_18_from_12": e18_from_12,
        "model_type": "pct_change_volume_cross",
        "coef_table": coef_df,
    }


def build_new_strategy_baseline_projection(ns_results: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    for result in ns_results:
        if not result.get("valid"):
            continue
        size_value = str(result.get("size", ""))
        slab_value = str(result.get("slab", ""))
        if slab_value == "combined_all_slabs":
            continue
        model_df = result.get("model_df", pd.DataFrame())
        if model_df is None or model_df.empty:
            continue
        work = model_df.copy()
        if "Period" not in work.columns or "non_discount_baseline_quantity" not in work.columns:
            continue
        for _, row in work.iterrows():
            rows.append(
                {
                    "Period": row["Period"],
                    "Size": size_value,
                    "Slab": slab_value,
                    "BaselineQty": float(pd.to_numeric(row["non_discount_baseline_quantity"], errors="coerce")),
                }
            )
    detail_df = pd.DataFrame(rows)
    if detail_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    detail_df["Period"] = pd.to_datetime(detail_df["Period"], errors="coerce")
    detail_df = detail_df.dropna(subset=["Period"]).copy()
    detail_df["Series"] = detail_df["Size"] + " " + detail_df["Slab"]
    size_df = (
        detail_df.groupby(["Period", "Size"], as_index=False)["BaselineQty"]
        .sum()
        .sort_values(["Period", "Size"])
        .reset_index(drop=True)
    )
    total_df = (
        size_df.groupby("Period", as_index=False)["BaselineQty"]
        .sum()
        .rename(columns={"BaselineQty": "TotalBaselineQty"})
        .sort_values("Period")
        .reset_index(drop=True)
    )
    detail_df = detail_df.sort_values(["Size", "Slab", "Period"]).reset_index(drop=True)
    return total_df, size_df, detail_df


def build_three_month_baseline_forecast(
    size_baseline_df: pd.DataFrame,
    elasticity: dict | None,
    horizon: int = 3,
    apply_cross_adjustment: bool = False,
) -> pd.DataFrame:
    if size_baseline_df is None or size_baseline_df.empty:
        return pd.DataFrame()

    wide = (
        size_baseline_df.pivot(index="Period", columns="Size", values="BaselineQty")
        .reset_index()
        .sort_values("Period")
        .reset_index(drop=True)
    )
    for col in ["12-ML", "18-ML"]:
        if col not in wide.columns:
            wide[col] = np.nan
        wide[col] = pd.to_numeric(wide[col], errors="coerce")

    last_period = pd.to_datetime(wide["Period"]).max()
    if pd.isna(last_period):
        return pd.DataFrame()

    def _holt_forecast(series: pd.Series, steps: int) -> np.ndarray:
        y = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
        if len(y) < 4:
            last_val = float(y[-1]) if len(y) else 0.0
            return np.repeat(max(last_val, 0.0), steps)
        try:
            fitted = Holt(
                y,
                exponential=False,
                damped_trend=False,
                initialization_method="estimated",
            ).fit(optimized=True)
            fc = np.asarray(fitted.forecast(steps), dtype=float)
        except Exception:
            last_val = float(y[-1]) if len(y) else 0.0
            fc = np.repeat(max(last_val, 0.0), steps)
        return np.clip(fc, 0.0, None)

    fc_12 = _holt_forecast(wide["12-ML"], horizon)
    fc_18 = _holt_forecast(wide["18-ML"], horizon)
    future_periods = pd.date_range(
        start=(last_period + pd.offsets.MonthBegin(1)),
        periods=horizon,
        freq="MS",
    )
    future = pd.DataFrame(
        {
            "Period": future_periods,
            "12-ML": fc_12,
            "18-ML": fc_18,
        }
    )
    if apply_cross_adjustment and elasticity:
        e12_cross = float(elasticity.get("cross_elasticity_12_from_18", np.nan))
        e18_cross = float(elasticity.get("cross_elasticity_18_from_12", np.nan))
    else:
        e12_cross = np.nan
        e18_cross = np.nan

    wide["Type"] = "History"
    future["Type"] = "Forecast"

    out = pd.concat([wide, future], ignore_index=True).reset_index(drop=True)
    out["raw_change_pct_12"] = (
        out["12-ML"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0) * 100.0
    )
    out["raw_change_pct_18"] = (
        out["18-ML"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0) * 100.0
    )
    out["own_change_pct_12"] = out["raw_change_pct_12"]
    out["own_change_pct_18"] = out["raw_change_pct_18"]
    out["baseline_12_pre_cross"] = out["12-ML"].clip(lower=0.0)
    out["baseline_18_pre_cross"] = out["18-ML"].clip(lower=0.0)
    out["baseline_12_adjusted"] = out["baseline_12_pre_cross"]
    out["baseline_18_adjusted"] = out["baseline_18_pre_cross"]

    if apply_cross_adjustment:
        forecast_mask = out["Type"] == "Forecast"
        if np.isfinite(e12_cross):
            out.loc[forecast_mask, "baseline_12_adjusted"] = (
                out.loc[forecast_mask, "baseline_12_pre_cross"]
                * (1.0 + ((e12_cross * out.loc[forecast_mask, "raw_change_pct_18"]) / 100.0))
            )
        if np.isfinite(e18_cross):
            out.loc[forecast_mask, "baseline_18_adjusted"] = (
                out.loc[forecast_mask, "baseline_18_pre_cross"]
                * (1.0 + ((e18_cross * out.loc[forecast_mask, "raw_change_pct_12"]) / 100.0))
            )
        out["baseline_12_adjusted"] = out["baseline_12_adjusted"].clip(lower=0.0)
        out["baseline_18_adjusted"] = out["baseline_18_adjusted"].clip(lower=0.0)

    out["baseline_total_adjusted"] = out["baseline_12_adjusted"] + out["baseline_18_adjusted"]
    return out


def build_monthly_size_trend(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    required_core = {"Sizes", "Quantity", "TotalDiscount", "SalesValue_atBasicRate"}
    missing_core = [c for c in required_core if c not in work.columns]
    if missing_core:
        raise KeyError(f"Missing required columns for trend view: {', '.join(missing_core)}")

    if "Month" not in work.columns:
        if "Date" not in work.columns:
            raise KeyError("Missing required columns for trend view: need Date or Month")
        work["Date"] = pd.to_datetime(work["Date"], errors="coerce")
        work = work.dropna(subset=["Date"]).copy()
        work["Month"] = work["Date"].dt.to_period("M").dt.to_timestamp()

    out = (
        work.groupby(["Month", "Sizes"], as_index=False)
        .agg(
            Quantity=("Quantity", "sum"),
            Total_Discount=("TotalDiscount", "sum"),
            Sales_Value=("SalesValue_atBasicRate", "sum"),
        )
        .sort_values(["Month", "Sizes"])
        .reset_index(drop=True)
    )
    out["Discount_%"] = np.where(
        out["Sales_Value"] > 0,
        (out["Total_Discount"] / out["Sales_Value"]) * 100.0,
        np.nan,
    )
    out["Discount_%"] = out["Discount_%"].round(2)
    return out


def build_price_change_months(df: pd.DataFrame) -> dict[str, list[pd.Timestamp]]:
    if "MRP" not in df.columns or "Sizes" not in df.columns:
        return {}

    work = df.copy()
    if "Month" not in work.columns:
        if "Date" not in work.columns:
            return {}
        work["Date"] = pd.to_datetime(work["Date"], errors="coerce")
        work = work.dropna(subset=["Date"]).copy()
        work["Month"] = work["Date"].dt.to_period("M").dt.to_timestamp()

    work["MRP"] = pd.to_numeric(work["MRP"], errors="coerce")
    work = work.dropna(subset=["MRP", "Month", "Sizes"]).copy()
    if work.empty:
        return {}

    product_candidates = ["Sku_Code", "Sku_Name", "Material_ID", "MG-10(Base_Sku)", "T SKU Code"]
    product_col = next((c for c in product_candidates if c in work.columns and work[c].notna().any()), None)

    if product_col is None:
        monthly = (
            work.groupby(["Sizes", "Month"], as_index=False)
            .agg(MRP=("MRP", "median"))
            .sort_values(["Sizes", "Month"])
            .reset_index(drop=True)
        )
        monthly["Prev_MRP"] = monthly.groupby("Sizes")["MRP"].shift(1)
        monthly["Changed"] = monthly["Prev_MRP"].notna() & ((monthly["MRP"] - monthly["Prev_MRP"]).abs() > 1e-9)
        changed = monthly[monthly["Changed"]]
    else:
        work[product_col] = work[product_col].astype(str).str.strip()
        work = work[work[product_col] != ""].copy()
        monthly = (
            work.groupby(["Sizes", product_col, "Month"], as_index=False)
            .agg(MRP=("MRP", "median"))
            .sort_values(["Sizes", product_col, "Month"])
            .reset_index(drop=True)
        )
        monthly["Prev_MRP"] = monthly.groupby(["Sizes", product_col])["MRP"].shift(1)
        monthly["Changed"] = monthly["Prev_MRP"].notna() & ((monthly["MRP"] - monthly["Prev_MRP"]).abs() > 1e-9)
        changed = monthly[monthly["Changed"]]

    if changed.empty:
        return {}

    out: dict[str, list[pd.Timestamp]] = {}
    for size, part in changed.groupby("Sizes"):
        out[str(size)] = sorted(pd.to_datetime(part["Month"]).dropna().unique())
    return out


def _add_price_change_lines(fig: go.Figure, months: list[pd.Timestamp], color: str, label: str) -> None:
    for dt in months:
        x0 = pd.to_datetime(dt)
        x1 = x0 + pd.offsets.MonthBegin(1)
        fig.add_vrect(
            x0=x0,
            x1=x1,
            fillcolor=color,
            opacity=0.10,
            line_width=0,
            layer="below",
        )
        fig.add_vline(
            x=x0,
            line_width=2,
            line_dash="dash",
            line_color=color,
            opacity=0.9,
        )
        fig.add_annotation(
            x=x0,
            y=1.03,
            yref="paper",
            text=f"{label} Price Change",
            showarrow=False,
            font=dict(size=9, color=color),
            xanchor="left",
        )


@st.cache_data(show_spinner=False)
def load_and_filter_data() -> tuple[pd.DataFrame, Path, int, int]:
    data_dir = _find_data_dir()
    if data_dir is None:
        raise FileNotFoundError("DATA folder not found. Expected at project root: DATA/")

    files = sorted(data_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    dfs: list[pd.DataFrame] = []
    files_loaded = 0
    for file in files:
        schema_cols = set(pq.read_schema(file).names)
        if not REQUIRED_COLUMNS.issubset(schema_cols):
            continue

        use_cols = [col for col in PREFERRED_COLUMNS if col in schema_cols]
        part = pd.read_parquet(file, columns=use_cols)
        part["Subcategory"] = _normalize_text(part["Subcategory"])
        part["Sizes"] = _normalize_size(part["Sizes"])
        part = part[
            part["Subcategory"].isin(TARGET_SUBCATEGORIES)
            & part["Sizes"].isin(TARGET_SIZES)
        ].copy()
        if not part.empty:
            dfs.append(part)
        files_loaded += 1

    filtered = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=PREFERRED_COLUMNS)

    if "Date" in filtered.columns:
        filtered["Date"] = pd.to_datetime(filtered["Date"], errors="coerce")

    return filtered, data_dir, len(files), files_loaded


def main() -> None:
    st.set_page_config(page_title="12ML + 18ML Methodology", layout="wide")
    st.title("12ML + 18ML Slab Definition")

    try:
        with st.spinner("Loading parquet files and filtering data..."):
            filtered, _data_dir, _file_count, _files_loaded = load_and_filter_data()
    except Exception as exc:
        st.error(str(exc))
        return

    if filtered.empty:
        st.warning("No matching rows for the selected subcategories and sizes.")
        return

    if "breaks_12" not in st.session_state:
        st.session_state["breaks_12"] = [8.0, 144.0]
    if "breaks_18" not in st.session_state:
        st.session_state["breaks_18"] = [8.0, 32.0, 576.0, 960.0]

    tab_preview, tab_qd, tab_slabs, tab_base, tab_new_strategy = st.tabs(
        [
            "Data Preview",
            "Qty & Discount",
            "Slab Setup",
            "Step 3 - Base Discount",
            "New Strategy",
        ]
    )

    with tab_preview:
        preview_rows = st.number_input("Rows to preview", min_value=100, max_value=50000, value=2000, step=100)
        st.dataframe(filtered.head(int(preview_rows)), use_container_width=True, height=520)

    with tab_qd:
        trend_df = build_monthly_size_trend(filtered)
        price_change_map = build_price_change_months(filtered)
        if trend_df.empty:
            st.info("No trend data available.")
        else:
            st.markdown("**Combined View (12-ML + 18-ML)**")
            c1, c2 = st.columns(2)
            with c1:
                fig_qty_combined = go.Figure()
                for size, color in [("12-ML", "#2563eb"), ("18-ML", "#059669")]:
                    part = trend_df[trend_df["Sizes"] == size].sort_values("Month")
                    fig_qty_combined.add_trace(
                        go.Scatter(
                            x=part["Month"],
                            y=part["Quantity"],
                            mode="lines+markers",
                            name=f"{size} Quantity",
                            line=dict(color=color, width=2),
                        )
                    )
                for size, color in [("12-ML", "#1d4ed8"), ("18-ML", "#047857")]:
                    part = trend_df[trend_df["Sizes"] == size].sort_values("Month")
                    fig_qty_combined.add_trace(
                        go.Scatter(
                            x=part["Month"],
                            y=part["Sales_Value"],
                            mode="lines+markers",
                            name=f"{size} Sales Value",
                            yaxis="y2",
                            line=dict(color=color, width=2, dash="dot"),
                            marker=dict(size=5),
                        )
                    )
                _add_price_change_lines(fig_qty_combined, price_change_map.get("12-ML", []), "#1d4ed8", "12-ML")
                _add_price_change_lines(fig_qty_combined, price_change_map.get("18-ML", []), "#dc2626", "18-ML")
                fig_qty_combined.update_layout(
                    title="Monthly Quantity & Sales Value (Combined)",
                    height=340,
                    margin=dict(l=20, r=20, t=45, b=20),
                    xaxis_title="Month",
                    yaxis_title="Quantity",
                    yaxis2=dict(
                        title="Sales Value",
                        overlaying="y",
                        side="right",
                    ),
                )
                st.plotly_chart(fig_qty_combined, use_container_width=True)
            with c2:
                fig_disc_combined = go.Figure()
                for size, color in [("12-ML", "#f59e0b"), ("18-ML", "#dc2626")]:
                    part = trend_df[trend_df["Sizes"] == size].sort_values("Month")
                    fig_disc_combined.add_trace(
                        go.Scatter(
                            x=part["Month"],
                            y=part["Discount_%" ],
                            mode="lines+markers",
                            name=size,
                            line=dict(color=color, width=2, shape="hv"),
                        )
                    )
                for size, color in [("12-ML", "#f59e0b"), ("18-ML", "#dc2626")]:
                    part = trend_df[trend_df["Sizes"] == size].sort_values("Month")
                    fig_disc_combined.add_trace(
                        go.Scatter(
                            x=part["Month"],
                            y=part["Sales_Value"],
                            mode="lines+markers",
                            name=f"{size} Sales Value",
                            yaxis="y2",
                            line=dict(color=color, width=2, dash="dot"),
                            marker=dict(size=5),
                        )
                    )
                _add_price_change_lines(fig_disc_combined, price_change_map.get("12-ML", []), "#1d4ed8", "12-ML")
                _add_price_change_lines(fig_disc_combined, price_change_map.get("18-ML", []), "#dc2626", "18-ML")
                fig_disc_combined.update_layout(
                    title="Monthly Discount % (Combined)",
                    height=340,
                    margin=dict(l=20, r=20, t=45, b=20),
                    xaxis_title="Month",
                    yaxis_title="Discount %",
                    yaxis2=dict(
                        title="Sales Value",
                        overlaying="y",
                        side="right",
                    ),
                )
                st.plotly_chart(fig_disc_combined, use_container_width=True)
            m12 = [pd.to_datetime(x).strftime("%b %Y") for x in price_change_map.get("12-ML", [])]
            m18 = [pd.to_datetime(x).strftime("%b %Y") for x in price_change_map.get("18-ML", [])]
            st.caption(f"Price change months - 12-ML: {', '.join(m12) if m12 else 'None'} | 18-ML: {', '.join(m18) if m18 else 'None'}")

            st.markdown("**Separate View (12-ML and 18-ML)**")
            s_left, s_right = st.columns(2)
            for col, size, bar_color, line_color in [
                (s_left, "12-ML", "#93c5fd", "#1d4ed8"),
                (s_right, "18-ML", "#a7f3d0", "#047857"),
            ]:
                with col:
                    part = trend_df[trend_df["Sizes"] == size].sort_values("Month")
                    fig_sep = go.Figure()
                    fig_sep.add_trace(
                        go.Bar(
                            x=part["Month"],
                            y=part["Quantity"],
                            name="Quantity",
                            marker_color=bar_color,
                        )
                    )
                    fig_sep.add_trace(
                        go.Scatter(
                            x=part["Month"],
                            y=part["Discount_%" ],
                            mode="lines+markers",
                            name="Discount %",
                            yaxis="y2",
                            line=dict(color=line_color, width=3, shape="hv"),
                        )
                    )
                    fig_sep.update_layout(
                        title=f"{size} - Quantity & Discount %",
                        height=360,
                        margin=dict(l=20, r=20, t=45, b=20),
                        xaxis_title="Month",
                        yaxis_title="Quantity",
                        yaxis2=dict(
                            title="Discount %",
                            overlaying="y",
                            side="right",
                        ),
                    )
                    st.plotly_chart(fig_sep, use_container_width=True)

            with st.expander("Monthly Quantity & Discount Table", expanded=False):
                st.dataframe(trend_df, use_container_width=True, height=300)

    with tab_slabs:
        in_col1, in_col2 = st.columns(2)
        with in_col1:
            st.markdown("### 12-ML")
            with st.container(border=True):
                breaks_12_text = st.text_input(
                    "Break points (comma separated)",
                    value=_breaks_to_text(st.session_state["breaks_12"]),
                    key="breaks_12_text",
                )
        with in_col2:
            st.markdown("### 18-ML")
            with st.container(border=True):
                breaks_18_text = st.text_input(
                    "Break points (comma separated)",
                    value=_breaks_to_text(st.session_state["breaks_18"]),
                    key="breaks_18_text",
                )
        apply_clicked = st.button("Save and Apply Slabs", type="primary")

        if apply_clicked or "defined_df" not in st.session_state:
            try:
                breaks_12 = _parse_breaks(breaks_12_text)
                breaks_18 = _parse_breaks(breaks_18_text)
                st.session_state["defined_df"] = build_defined_slabs(filtered, breaks_12, breaks_18)
                st.session_state["breaks_12"] = breaks_12
                st.session_state["breaks_18"] = breaks_18
            except Exception as exc:
                st.error(str(exc))
                return

        defined_df = st.session_state["defined_df"]
        slab_mix = build_slab_mix_df(defined_df)

        left, right = st.columns(2)
        with left:
            st.markdown("**12-ML Rules**")
            st.dataframe(build_slab_rules_df(st.session_state["breaks_12"]), use_container_width=True, height=190)
            st.markdown("**12-ML Slab Mix (Quantity %)**")
            mix_12 = slab_mix[slab_mix["Sizes"] == "12-ML"][
                ["Defined_Slab", "Quantity", "Quantity_%", "Unique_Outlets", "Outlet_Months"]
            ].copy()
            st.dataframe(mix_12, use_container_width=True, height=250)
        with right:
            st.markdown("**18-ML Rules**")
            st.dataframe(build_slab_rules_df(st.session_state["breaks_18"]), use_container_width=True, height=190)
            st.markdown("**18-ML Slab Mix (Quantity %)**")
            mix_18 = slab_mix[slab_mix["Sizes"] == "18-ML"][
                ["Defined_Slab", "Quantity", "Quantity_%", "Unique_Outlets", "Outlet_Months"]
            ].copy()
            st.dataframe(mix_18, use_container_width=True, height=250)

        with st.expander("Defined Slab Output Preview", expanded=False):
            st.dataframe(defined_df.head(3000), use_container_width=True, height=380)
        if st.button("Prepare Full Defined Slab CSV", key="prepare_defined_slab_full_csv", use_container_width=True):
            st.session_state["defined_slab_full_csv_bytes"] = defined_df.to_csv(index=False).encode("utf-8")
            st.session_state["defined_slab_full_csv_rows"] = len(defined_df)
        full_csv_bytes = st.session_state.get("defined_slab_full_csv_bytes")
        full_csv_rows = st.session_state.get("defined_slab_full_csv_rows")
        if full_csv_bytes is not None and full_csv_rows == len(defined_df):
            st.download_button(
                label="Download Full Defined Slab CSV",
                data=full_csv_bytes,
                file_name="insta_shampoo_12ml_18ml_defined_slab_full.csv",
                mime="text/csv",
                use_container_width=True,
                key="download_defined_slab_full_csv",
            )

        st.markdown("### Outlet-Level Sanity Check")
        size_opts = sorted(defined_df["Sizes"].dropna().astype(str).unique().tolist())
        if size_opts:
            sc1, sc2, sc3 = st.columns(3)
            selected_size = sc1.selectbox("Size", options=size_opts, key="sanity_size")

            slab_opts = sorted(
                defined_df[defined_df["Sizes"] == selected_size]["Defined_Slab"]
                .dropna()
                .astype(str)
                .unique()
                .tolist(),
                key=_slab_sort_key,
            )
            if slab_opts:
                selected_slab = sc2.selectbox("Slab", options=slab_opts, key="sanity_slab")

                month_values = (
                    defined_df[
                        (defined_df["Sizes"] == selected_size)
                        & (defined_df["Defined_Slab"].astype(str) == selected_slab)
                    ]["Month"]
                    .dropna()
                    .sort_values()
                    .unique()
                    .tolist()
                )
                month_map = {pd.Timestamp(m).strftime("%Y-%m"): pd.Timestamp(m) for m in month_values}
                month_labels = list(month_map.keys())
                if month_labels:
                    selected_month_label = sc3.selectbox("Month", options=month_labels, key="sanity_month")
                    selected_month = month_map[selected_month_label]

                    outlet_view, overall_pct, outlet_count, total_qty, avg_abs_dev = build_outlet_discount_variation(
                        defined_df=defined_df,
                        size_value=selected_size,
                        slab_value=selected_slab,
                        month_value=selected_month,
                    )

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Overall Discount %", f"{overall_pct:.2f}%" if pd.notna(overall_pct) else "NA")
                    m2.metric("Outlets", f"{outlet_count:,}")
                    m3.metric("Quantity", f"{total_qty:,.2f}")
                    m4.metric("Avg Abs Deviation (pp)", f"{avg_abs_dev:.2f}" if pd.notna(avg_abs_dev) else "NA")

                    if outlet_view.empty:
                        st.info("No outlet-level data for this selection.")
                    else:
                        top10 = outlet_view.head(10).copy()
                        st.dataframe(
                            top10[
                                [
                                    "Outlet_ID",
                                    "Outlet_Discount_%",
                                    "Deviation_pp",
                                    "Abs_Deviation_pp",
                                    "Sales_Value",
                                    "Total_Discount",
                                    "Quantity",
                                    "Rows",
                                ]
                                + (["Invoices"] if "Invoices" in top10.columns else [])
                            ],
                            use_container_width=True,
                            height=320,
                        )

                        disc = outlet_view["Outlet_Discount_%"].dropna().astype(float)
                        if disc.empty:
                            st.info("No discount distribution available for this selection.")
                        else:
                            bin_size = 0.5
                            min_edge = np.floor(disc.min() / bin_size) * bin_size
                            max_edge = np.ceil(disc.max() / bin_size) * bin_size
                            if max_edge <= min_edge:
                                max_edge = min_edge + bin_size
                            edges = np.arange(min_edge, max_edge + bin_size, bin_size)

                            bands = pd.cut(disc, bins=edges, include_lowest=True, right=False)
                            counts = bands.value_counts(sort=False)
                            counts = counts[counts > 0]
                            freq_pct = (counts / counts.sum() * 100.0).round(2)

                            chart_df = pd.DataFrame(
                                {
                                    "Band": [f"{iv.left:.1f}% to {iv.right:.1f}%" for iv in counts.index],
                                    "Band_Mid": [round((iv.left + iv.right) / 2, 2) for iv in counts.index],
                                    "Outlet_Count": counts.values,
                                    "Frequency_%": freq_pct.values,
                                }
                            )
                            chart_df["Bar_Label"] = chart_df.apply(
                                lambda r: f"{r['Frequency_%']:.1f}%\n{r['Band_Mid']:.1f}%",
                                axis=1,
                            )

                            fig = go.Figure(
                                data=[
                                    go.Bar(
                                        x=chart_df["Band"],
                                        y=chart_df["Frequency_%"],
                                        text=chart_df["Bar_Label"],
                                        textposition="outside",
                                        marker_color="#3b82f6",
                                        customdata=np.stack(
                                            [chart_df["Outlet_Count"], chart_df["Band_Mid"]],
                                            axis=1,
                                        ),
                                        hovertemplate=(
                                            "Band: %{x}<br>"
                                            "Frequency: %{y:.2f}%<br>"
                                            "Outlets: %{customdata[0]:,.0f}<br>"
                                            "Band Mid: %{customdata[1]:.2f}%<extra></extra>"
                                        ),
                                    )
                                ]
                            )
                            title_suffix = f" | Overall: {overall_pct:.2f}%" if pd.notna(overall_pct) else ""
                            fig.update_layout(
                                title=f"Outlet Discount % Frequency (0.5pp bands){title_suffix}",
                                height=340,
                                margin=dict(l=20, r=20, t=50, b=20),
                                xaxis_title="Discount Band",
                                yaxis_title="Frequency %",
                                yaxis=dict(range=[0, max(chart_df['Frequency_%'].max() * 1.2, 5)]),
                                bargap=0.12,
                            )
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No month data available for selected size and slab.")
            else:
                st.info("No slab data available for selected size.")

    with tab_base:
        if "defined_df" not in st.session_state:
            st.info("Please complete Slab Setup first.")
            return
        defined_df = st.session_state["defined_df"]

        run_base = st.button("Compute Base Discount for All Slabs", type="primary")

        if run_base or "base_df" not in st.session_state:
            try:
                st.session_state["base_df"] = compute_base_discount(defined_df=defined_df, round_step=0.5)
            except Exception as exc:
                st.error(str(exc))
                return

        base_df = st.session_state["base_df"]
        b_left, b_right = st.columns(2)
        with b_left:
            render_slab_charts(base_df, "12-ML", two_per_row=False)
        with b_right:
            render_slab_charts(base_df, "18-ML", two_per_row=False)

        with st.expander("Monthly Base Discount Output", expanded=False):
            st.dataframe(
                base_df[
                    [
                        "Sizes",
                        "Defined_Slab",
                        "Month",
                        "Monthly_Quantity",
                        "Actual_Discount_%",
                        "Estimated_Base_%",
                    ]
                ],
                use_container_width=True,
                height=420,
            )

        with st.expander("Download Base Discount CSV", expanded=False):
            if st.button("Prepare Base CSV", use_container_width=True):
                st.session_state["methodology_base_csv_bytes"] = base_df.to_csv(index=False).encode("utf-8")
                st.session_state["methodology_base_csv_rows"] = len(base_df)

            csv_bytes = st.session_state.get("methodology_base_csv_bytes")
            csv_rows = st.session_state.get("methodology_base_csv_rows")
            if csv_bytes is not None and csv_rows == len(base_df):
                st.download_button(
                    label="Download Base Discount CSV",
                    data=csv_bytes,
                    file_name="insta_shampoo_12ml_18ml_base_discount.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

    with tab_new_strategy:
        if "defined_df" not in st.session_state or "base_df" not in st.session_state:
            st.info("Please complete Slab Setup and Step 3 - Base Discount first.")
        else:
            defined_df = st.session_state["defined_df"]
            base_df = st.session_state["base_df"]

            nc1, nc2 = st.columns(2)
            cogs_12_ns = nc1.number_input(
                "COGS Per Unit (12-ML) - New Strategy",
                min_value=0.0,
                max_value=10000.0,
                value=float(st.session_state.get("model_cogs_12", 8.0)),
                step=0.5,
                key="new_strategy_cogs_12",
            )
            cogs_18_ns = nc2.number_input(
                "COGS Per Unit (18-ML) - New Strategy",
                min_value=0.0,
                max_value=10000.0,
                value=float(st.session_state.get("model_cogs_18", 8.0)),
                step=0.5,
                key="new_strategy_cogs_18",
            )
            st.caption(
                "New Strategy Stage 2 = existing slab model + one extra feature: "
                "other_slabs_weighted_base_discount_pct (fixed-weight weighted base discount from other slabs)."
            )
            run_new_strategy = st.button("Run New Strategy Modeling (12-ML + 18-ML)", type="primary")
            new_strategy_cache_version = 3
            needs_new_strategy_refresh = (
                run_new_strategy
                or "new_strategy_results" not in st.session_state
                or st.session_state.get("new_strategy_cache_version") != new_strategy_cache_version
            )

            if needs_new_strategy_refresh:
                with st.spinner("Running New Strategy models..."):
                    ns_results, ns_summary = run_new_strategy_all_slabs(
                        defined_df=defined_df,
                        base_df=base_df,
                        cogs_per_unit=float(cogs_12_ns),
                        cogs_by_size={"12-ML": float(cogs_12_ns), "18-ML": float(cogs_18_ns)},
                        include_lag_discount=True,
                        optimize_l2_penalty=True,
                        l2_penalty=0.1,
                    )
                st.session_state["new_strategy_results"] = ns_results
                st.session_state["new_strategy_summary"] = ns_summary
                st.session_state["new_strategy_cache_version"] = new_strategy_cache_version

            ns_results = st.session_state.get("new_strategy_results", [])
            ns_summary = st.session_state.get("new_strategy_summary", pd.DataFrame())

            if ns_summary.empty:
                st.warning("No valid New Strategy model results. Check slab coverage and monthly points.")
            else:
                ntab12, ntab18 = st.tabs(["12-ML", "18-ML"])
                for ntab, active_size in zip([ntab12, ntab18], ["12-ML", "18-ML"]):
                    with ntab:
                        st.markdown(f"**{active_size} New Strategy Summary**")
                        d = ns_summary[ns_summary["Size"] == active_size].copy()
                        st.dataframe(d.reset_index(drop=True), use_container_width=True, height=250, hide_index=True)

                        valid_results = [r for r in ns_results if r.get("valid") and r.get("size") == active_size]
                        if not valid_results:
                            st.info(f"No valid New Strategy models for {active_size}.")
                            continue

                        slab_opts = sorted([r["slab"] for r in valid_results], key=_slab_sort_key)
                        slab_to_result = {str(r["slab"]): r for r in valid_results}

                        with st.expander(f"{active_size} Slab-Level Deep Dive", expanded=False):
                            selected_slab = st.selectbox(
                                f"{active_size} New Strategy Detail Slab",
                                options=slab_opts,
                                key=f"new_strategy_detail_slab_{active_size}",
                            )
                            chosen = slab_to_result.get(str(selected_slab))
                            if chosen is not None:
                                coeff = chosen["coefficients"]
                                roi_s = chosen["roi_summary"]
                                model_used = str(coeff.get("model_selected", "constrained_ridge")).strip().lower()
                                using_ols = model_used == "ols"
                                model_used_label = "OLS" if using_ols else "Constrained Ridge"
                                m1, m2, m3, m4 = st.columns(4)
                                m1.metric("Topline ROI", f"{roi_s.get('structural_roi_1mo', 0.0):.2f}x")
                                m2.metric("Gross Margin ROI", f"{roi_s.get('structural_profit_roi_1mo', 0.0):.2f}x")
                                m3.metric("Structural Episodes", f"{int(roi_s.get('episodes', 0.0))}")
                                m4.metric("Stage2 R2", f"{float(coeff.get('stage2_r2', np.nan)):.2f}")
                                st.caption(f"Model used for outputs: {model_used_label}")
                                ols_ok = coeff.get("ols_constraints_respected", None)
                                if ols_ok is not None:
                                    if bool(ols_ok):
                                        st.caption("OLS sign-check: PASS")
                                    else:
                                        viol = str(coeff.get("ols_constraint_violations", "")).strip()
                                        st.caption(
                                            f"OLS sign-check: FAIL"
                                            + (f" ({viol})" if viol else "")
                                        )

                                feature_order = coeff.get("feature_order", [])
                                selected_col_name = f"{model_used_label} (Selected)"
                                reference_col_name = "Constrained Ridge Reference" if using_ols else "OLS Reference"
                                coef_rows = []
                                for feat in feature_order:
                                    ref_key = (
                                        f"stage2_constrained_coef_{feat}"
                                        if using_ols
                                        else f"stage2_ols_coef_{feat}"
                                    )
                                    coef_rows.append(
                                        {
                                            "Feature": feat,
                                            selected_col_name: coeff.get(f"coef_{feat}", np.nan),
                                            reference_col_name: coeff.get(ref_key, np.nan),
                                        }
                                    )
                                coef_df = pd.DataFrame(coef_rows)
                                st.markdown("**Coefficient Comparison (New Strategy Stage 2)**")
                                st.dataframe(coef_df.reset_index(drop=True), use_container_width=True, height=250, hide_index=True)

                                model_df = chosen["model_df"].copy()
                                roi_pts = chosen["roi_points"].copy()

                                fig_qty = go.Figure()
                                fig_qty.add_trace(
                                    go.Scatter(
                                        x=model_df["Period"],
                                        y=model_df["quantity"],
                                        mode="lines+markers",
                                        name="Actual Quantity",
                                        line=dict(color="#1d4ed8", width=2),
                                    )
                                )
                                fig_qty.add_trace(
                                    go.Scatter(
                                        x=model_df["Period"],
                                        y=model_df["predicted_quantity"],
                                        mode="lines+markers",
                                        name="Predicted Quantity",
                                        line=dict(color="#0b7a75", width=2),
                                    )
                                )
                                fig_qty.update_layout(
                                    title=f"New Strategy: Actual vs Predicted Quantity - {active_size} {selected_slab}",
                                    height=360,
                                    margin=dict(l=20, r=20, t=45, b=20),
                                    xaxis_title="Month",
                                    yaxis_title="Quantity",
                                )
                                st.plotly_chart(
                                    fig_qty,
                                    use_container_width=True,
                                    key=f"new_strategy_qty_{active_size}_{selected_slab}",
                                )

                                fig_roi = go.Figure()
                                if not roi_pts.empty:
                                    fig_roi.add_trace(
                                        go.Bar(
                                            x=roi_pts["Period"],
                                            y=roi_pts["structural_roi_1mo"],
                                            name="Topline ROI",
                                            marker_color="#3b82f6",
                                        )
                                    )
                                    fig_roi.add_trace(
                                        go.Bar(
                                            x=roi_pts["Period"],
                                            y=roi_pts["profit_roi_1mo"],
                                            name="Gross Margin ROI",
                                            marker_color="#f59e0b",
                                        )
                                    )
                                fig_roi.add_trace(
                                    go.Scatter(
                                        x=model_df["Period"],
                                        y=model_df["base_discount_pct"],
                                        mode="lines+markers",
                                        name="Base Discount %",
                                        yaxis="y2",
                                        line=dict(color="#0b7a75", width=3, shape="hv"),
                                    )
                                )
                                fig_roi.update_layout(
                                    title=f"New Strategy ROI Episodes - {active_size} {selected_slab}",
                                    barmode="group",
                                    height=380,
                                    margin=dict(l=20, r=20, t=45, b=20),
                                    xaxis_title="Month",
                                    yaxis_title="ROI (x)",
                                    yaxis2=dict(
                                        title="Base Discount %",
                                        overlaying="y",
                                        side="right",
                                    ),
                                )
                                st.plotly_chart(
                                    fig_roi,
                                    use_container_width=True,
                                    key=f"new_strategy_roi_{active_size}_{selected_slab}",
                                )

                                with st.expander("New Strategy Diagnostics", expanded=False):
                                    render_store_quantity_chart(
                                        df=model_df,
                                        quantity_col="quantity",
                                        store_col="store_count",
                                        title=f"{active_size} {selected_slab} - Quantity and Store Count over time",
                                        chart_key=f"new_strategy_sq_{active_size}_{selected_slab}",
                                    )
                                    render_xy_diagnostics(
                                        df=model_df,
                                        y_col="quantity",
                                        feature_cols=[
                                            "residual_store",
                                            "base_discount_pct",
                                            "lag1_base_discount_pct",
                                            "other_slabs_weighted_base_discount_pct",
                                        ],
                                        title_prefix=f"{active_size} {selected_slab} - New Strategy Stage 2",
                                        chart_key_prefix=f"new_strategy_xy_{active_size}_{selected_slab}",
                                    )

                                with st.expander("New Strategy Equation & Monthly Table", expanded=False):
                                    s1_intercept = float(coeff.get("stage1_intercept", 0.0))
                                    s1_beta_discount = float(coeff.get("stage1_coef_discount", 0.0))
                                    s2_intercept = float(coeff.get("stage2_intercept", 0.0))
                                    st.markdown("**Stage 1 Equation (Store Count Model)**")
                                    st.code(
                                        f"store_count_hat = {s1_intercept:.6f} + ({s1_beta_discount:.6f} * actual_discount_pct)",
                                        language="text",
                                    )
                                    s2_line = f"qty_hat = {s2_intercept:.6f}"
                                    for feat in feature_order:
                                        s2_line += f" + ({float(coeff.get(f'coef_{feat}', 0.0)):.6f} * {feat})"
                                    st.markdown(
                                        f"**Stage 2 Equation ({model_used_label} + Other-Slabs Weighted Discount)**"
                                    )
                                    st.code(s2_line, language="text")
                                    if using_ols:
                                        st.markdown(
                                            f"Stage 2 R2 (OLS selected): **{float(coeff.get('stage2_r2', np.nan)):.4f}**, "
                                            f"Constrained R2: **{float(coeff.get('stage2_constrained_r2', np.nan)):.4f}**, "
                                            f"CV R2 (Constrained): **{float(coeff.get('stage2_constrained_cv_r2', np.nan)):.4f}**, "
                                            f"L2 Used (Constrained): **{float(coeff.get('l2_used', np.nan)):.4f}**"
                                        )
                                    else:
                                        st.markdown(
                                            f"Stage 2 R2: **{float(coeff.get('stage2_r2', np.nan)):.4f}**, "
                                            f"CV R2: **{float(coeff.get('stage2_cv_r2', np.nan)):.4f}**, "
                                            f"L2 Used: **{float(coeff.get('l2_used', np.nan)):.4f}**, "
                                            f"OLS R2: **{float(coeff.get('stage2_ols_r2', np.nan)):.4f}**"
                                        )
                                    cols_show = [
                                        "Period",
                                        "store_count",
                                        "actual_discount_pct",
                                        "residual_store",
                                        "base_discount_pct",
                                        "lag1_base_discount_pct",
                                        "other_slabs_weighted_base_discount_pct",
                                        "quantity",
                                        "predicted_quantity",
                                        "baseline_quantity",
                                        "spend",
                                        "incremental_revenue",
                                        "roi_1mo",
                                    ]
                                    cols_show = [c for c in cols_show if c in model_df.columns]
                                    st.dataframe(
                                        model_df[cols_show].reset_index(drop=True),
                                        use_container_width=True,
                                        height=320,
                                        hide_index=True,
                                    )

                        st.markdown("**Last-Month Scenario Simulator**")
                        st.caption(
                            "Edit slab discounts for this size. Prediction uses each slab model with last month residual_store and lag1 discount."
                        )

                        slab_state: dict[str, dict] = {}
                        for slab in slab_opts:
                            slab_key = str(slab)
                            rr = slab_to_result[slab_key]
                            coeff = rr.get("coefficients", {})
                            model_df = rr["model_df"].sort_values("Period").copy()
                            if model_df.empty:
                                continue
                            last = model_df.iloc[-1]
                            stage2_model = rr.get("stage2_model")
                            if stage2_model is None:
                                stage2_model = _build_stage2_model_from_coefficients(coeff)
                            if stage2_model is None:
                                continue
                            qty_anchor = float(last.get("quantity", 0.0))
                            selected_mode = str(coeff.get("model_selected", "constrained_ridge")).strip().lower()
                            selected_mode_label = "OLS" if selected_mode == "ols" else "Constrained Ridge"
                            slab_state[slab_key] = {
                                "model_auto": stage2_model,
                                "model_ols": _build_stage2_ols_model_from_coefficients(coeff),
                                "model_constrained": _build_stage2_constrained_model_from_coefficients(coeff),
                                "model_auto_label": selected_mode_label,
                                "coeff": coeff,
                                "residual": float(last.get("residual_store", 0.0)),
                                "lag1": float(last.get("lag1_base_discount_pct", 0.0)),
                                "own_default": float(last.get("base_discount_pct", 0.0)),
                                "qty_anchor": max(qty_anchor, 1.0),
                            }

                        if not slab_state:
                            st.info("Scenario simulation unavailable for this size.")
                        else:
                            if st.button(
                                "Reset Scenario to Default",
                                key=f"ns_reset_{active_size}",
                            ):
                                for slab_key_reset, st_state_reset in slab_state.items():
                                    sk = str(slab_key_reset)
                                    st.session_state[f"ns_scenario_{active_size}_{sk}"] = round(
                                        float(st_state_reset.get("own_default", 0.0)),
                                        2,
                                    )
                                st.rerun()

                            st.markdown("**Scenario Inputs by Slab (row-wise)**")
                            h1, h2, h3, h4, h5, h6 = st.columns([1.0, 1.7, 1.2, 1.2, 1.2, 1.2])
                            h1.markdown("**Slab**")
                            h2.markdown("**Scenario Discount %**")
                            h3.markdown("**Own Beta**")
                            h4.markdown("**Other Beta**")
                            h5.markdown("**Anchor Qty**")
                            h6.markdown("**Volume Delta %**")

                            scenario_inputs: dict[str, float] = {}
                            delta_placeholders: dict[str, object] = {}

                            for slab_key in slab_opts:
                                slab_key = str(slab_key)
                                st_state = slab_state.get(slab_key)
                                if st_state is None:
                                    continue
                                coeff = st_state["coeff"]
                                c1, c2, c3, c4, c5, c6 = st.columns([1.0, 1.7, 1.2, 1.2, 1.2, 1.2])
                                c1.write(slab_key)
                                scenario_key = f"ns_scenario_{active_size}_{slab_key}"
                                if scenario_key not in st.session_state:
                                    st.session_state[scenario_key] = round(float(st_state["own_default"]), 2)
                                scenario_inputs[slab_key] = c2.number_input(
                                    f"{slab_key} discount",
                                    min_value=0.0,
                                    max_value=100.0,
                                    step=0.5,
                                    key=scenario_key,
                                    label_visibility="collapsed",
                                )
                                beta_own = coeff.get("coef_base_discount_pct", np.nan)
                                beta_other = coeff.get("coef_other_slabs_weighted_base_discount_pct", np.nan)
                                c3.write(f"{float(beta_own):.4f}" if np.isfinite(beta_own) else "NA")
                                c4.write(f"{float(beta_other):.4f}" if np.isfinite(beta_other) else "NA")
                                c5.write(f"{float(st_state['qty_anchor']):,.0f}")
                                ph = c6.empty()
                                ph.write("-")
                                delta_placeholders[slab_key] = ph

                            def _normalize_weight_map_from_qty(q_map: dict[str, float]) -> dict[str, float]:
                                clean = {str(k): max(float(v), 0.0) for k, v in q_map.items()}
                                total = float(sum(clean.values()))
                                if total <= 0:
                                    n = max(len(clean), 1)
                                    return {k: (1.0 / n) for k in clean.keys()}
                                return {k: (v / total) for k, v in clean.items()}

                            def _run_coupled_prediction(discount_map: dict[str, float], max_iter: int = 8):
                                slabs = list(slab_state.keys())
                                qty_map = {s: float(slab_state[s]["qty_anchor"]) for s in slabs}
                                other_map = {s: 0.0 for s in slabs}
                                for _ in range(max_iter):
                                    weight_map = _normalize_weight_map_from_qty(qty_map)
                                    next_qty: dict[str, float] = {}
                                    next_other: dict[str, float] = {}
                                    for s in slabs:
                                        stt = slab_state[s]
                                        model_for_pred = stt.get("model_constrained") or stt.get("model_auto")
                                        if model_for_pred is None:
                                            next_qty[s] = max(float(stt["qty_anchor"]), 0.0)
                                            next_other[s] = float(next_other.get(s, 0.0))
                                            continue
                                        own_disc = float(discount_map.get(s, stt["own_default"]))
                                        other_disc = compute_other_weighted_discount_for_slab(s, discount_map, weight_map)
                                        pred = float(
                                            _predict_stage2_quantity(
                                                model_for_pred,
                                                np.array([stt["residual"]], dtype=float),
                                                np.array([own_disc], dtype=float),
                                                np.array([0.0], dtype=float),
                                                np.array([stt["lag1"]], dtype=float),
                                                extra_feature_values={
                                                    "other_slabs_weighted_base_discount_pct": np.array([other_disc], dtype=float)
                                                },
                                            )[0]
                                        )
                                        next_qty[s] = max(pred, 0.0)
                                        next_other[s] = float(other_disc)
                                    delta = float(sum(abs(next_qty[s] - qty_map.get(s, 0.0)) for s in slabs))
                                    qty_map = next_qty
                                    other_map = next_other
                                    if delta <= 1e-6:
                                        break
                                return qty_map, _normalize_weight_map_from_qty(qty_map), other_map

                            default_map = {s: float(st["own_default"]) for s, st in slab_state.items()}
                            qty_default_map, w_default_map, other_default_map = _run_coupled_prediction(default_map)
                            qty_scenario_map, w_scenario_map, other_scenario_map = _run_coupled_prediction(scenario_inputs)

                            sim_rows = []
                            for slab_key in slab_opts:
                                slab_key = str(slab_key)
                                if slab_key not in slab_state:
                                    continue
                                own_default = float(default_map.get(slab_key, 0.0))
                                own_scenario = float(scenario_inputs.get(slab_key, own_default))
                                pred_default = float(qty_default_map.get(slab_key, 0.0))
                                pred_scenario = float(qty_scenario_map.get(slab_key, pred_default))
                                delta = pred_scenario - pred_default
                                pct = (delta / pred_default * 100.0) if pred_default > 0 else np.nan
                                sim_rows.append(
                                    {
                                        "Slab": slab_key,
                                        "Own Discount Default %": round(own_default, 2),
                                        "Own Discount Scenario %": round(own_scenario, 2),
                                        "Other Weighted Default %": round(float(other_default_map.get(slab_key, 0.0)), 2),
                                        "Other Weighted Scenario %": round(float(other_scenario_map.get(slab_key, 0.0)), 2),
                                        "Weight Default %": round(float(w_default_map.get(slab_key, 0.0)) * 100.0, 2),
                                        "Weight Scenario %": round(float(w_scenario_map.get(slab_key, 0.0)) * 100.0, 2),
                                        "Pred Qty Default": round(pred_default, 2),
                                        "Pred Qty Scenario": round(pred_scenario, 2),
                                        "Delta Qty": round(delta, 2),
                                        "Volume Delta %": round(pct, 2) if np.isfinite(pct) else np.nan,
                                    }
                                )

                            sim_df = pd.DataFrame(sim_rows)
                            if not sim_df.empty and "Slab" in sim_df.columns and "Volume Delta %" in sim_df.columns:
                                delta_pct_map = {
                                    str(row["Slab"]): row["Volume Delta %"]
                                    for _, row in sim_df.iterrows()
                                }
                                for slab_key, ph in delta_placeholders.items():
                                    v = delta_pct_map.get(slab_key, np.nan)
                                    if np.isfinite(v):
                                        sign = "+" if float(v) > 0 else ""
                                        ph.write(f"{sign}{float(v):.2f}%")
                                    else:
                                        ph.write("NA")
                            st.dataframe(sim_df, use_container_width=True, height=300, hide_index=True)

                            total_default = float(sum(qty_default_map.values()))
                            total_scenario = float(sum(qty_scenario_map.values()))
                            own_change_pct = (
                                ((total_scenario - total_default) / total_default) * 100.0
                                if total_default > 0
                                else np.nan
                            )
                            st.session_state[f"ns_size_delta_pct_{active_size}"] = own_change_pct
                            st.session_state[f"ns_size_total_default_{active_size}"] = total_default

                st.markdown("**Cross-Size Scenario Inputs (Side-by-Side)**")
                st.caption(
                    "Edit slab discounts for both sizes together. Volume deltas are computed from constrained ridge models "
                    "using dynamic slab weights."
                )

                def _normalize_weight_map_from_qty(q_map: dict[str, float]) -> dict[str, float]:
                    clean = {str(k): max(float(v), 0.0) for k, v in q_map.items()}
                    total = float(sum(clean.values()))
                    if total <= 0:
                        n = max(len(clean), 1)
                        return {k: (1.0 / n) for k in clean.keys()}
                    return {k: (v / total) for k, v in clean.items()}

                def _run_coupled_prediction_for_state(
                    slab_state_local: dict[str, dict],
                    discount_map: dict[str, float],
                    max_iter: int = 8,
                ):
                    slabs_local = list(slab_state_local.keys())
                    qty_map_local = {s: float(slab_state_local[s]["qty_anchor"]) for s in slabs_local}
                    other_map_local = {s: 0.0 for s in slabs_local}
                    for _ in range(max_iter):
                        weight_map_local = _normalize_weight_map_from_qty(qty_map_local)
                        next_qty_local: dict[str, float] = {}
                        next_other_local: dict[str, float] = {}
                        for s_local in slabs_local:
                            stt_local = slab_state_local[s_local]
                            model_for_pred_local = stt_local.get("model_constrained")
                            if model_for_pred_local is None:
                                next_qty_local[s_local] = max(float(stt_local["qty_anchor"]), 0.0)
                                next_other_local[s_local] = 0.0
                                continue
                            own_disc_local = float(discount_map.get(s_local, stt_local["own_default"]))
                            other_disc_local = compute_other_weighted_discount_for_slab(
                                s_local,
                                discount_map,
                                weight_map_local,
                            )
                            pred_local = float(
                                _predict_stage2_quantity(
                                    model_for_pred_local,
                                    np.array([stt_local["residual"]], dtype=float),
                                    np.array([own_disc_local], dtype=float),
                                    np.array([0.0], dtype=float),
                                    np.array([stt_local["lag1"]], dtype=float),
                                    extra_feature_values={
                                        "other_slabs_weighted_base_discount_pct": np.array([other_disc_local], dtype=float)
                                    },
                                )[0]
                            )
                            next_qty_local[s_local] = max(pred_local, 0.0)
                            next_other_local[s_local] = float(other_disc_local)
                        delta_local = float(
                            sum(abs(next_qty_local[s] - qty_map_local.get(s, 0.0)) for s in slabs_local)
                        )
                        qty_map_local = next_qty_local
                        other_map_local = next_other_local
                        if delta_local <= 1e-6:
                            break
                    return qty_map_local, _normalize_weight_map_from_qty(qty_map_local), other_map_local

                pair_state: dict[str, dict] = {}
                for pair_size in ["12-ML", "18-ML"]:
                    valid_pair_results = [r for r in ns_results if r.get("valid") and r.get("size") == pair_size]
                    slab_opts_pair = sorted(
                        [r["slab"] for r in valid_pair_results if str(r.get("slab")) != "combined_all_slabs"],
                        key=_slab_sort_key,
                    )
                    slab_to_result_pair = {str(r["slab"]): r for r in valid_pair_results}
                    slab_state_pair: dict[str, dict] = {}
                    for slab_pair in slab_opts_pair:
                        r_pair = slab_to_result_pair.get(str(slab_pair))
                        if r_pair is None:
                            continue
                        coeff_pair = r_pair.get("coefficients", {})
                        model_df_pair = r_pair.get("model_df", pd.DataFrame()).sort_values("Period")
                        if model_df_pair.empty:
                            continue
                        last_pair = model_df_pair.iloc[-1]
                        model_pair = _build_stage2_constrained_model_from_coefficients(coeff_pair)
                        if model_pair is None:
                            continue
                        slab_state_pair[str(slab_pair)] = {
                            "model_constrained": model_pair,
                            "coeff": coeff_pair,
                            "residual": float(last_pair.get("residual_store", 0.0)),
                            "lag1": float(last_pair.get("lag1_base_discount_pct", 0.0)),
                            "own_default": float(last_pair.get("base_discount_pct", 0.0)),
                            "qty_anchor": max(float(last_pair.get("quantity", 0.0)), 1.0),
                        }
                    pair_state[pair_size] = {"slab_opts": slab_opts_pair, "slab_state": slab_state_pair}

                if any(len(pair_state.get(s, {}).get("slab_state", {})) == 0 for s in ["12-ML", "18-ML"]):
                    st.info("Cross-size scenario needs valid slab models for both 12-ML and 18-ML.")
                else:
                    if st.button("Reset Both Sizes to Default", key="ns_pair_reset_all"):
                        for pair_size in ["12-ML", "18-ML"]:
                            state_block = pair_state[pair_size]["slab_state"]
                            for skey, sval in state_block.items():
                                st.session_state[f"ns_pair_{pair_size}_{skey}"] = round(
                                    float(sval.get("own_default", 0.0)),
                                    2,
                                )
                        st.rerun()

                    st.markdown("**Top Impact View**")
                    ti1, ti2, ti3, ti4 = st.columns(4)
                    top_placeholders = {
                        "12_overall": ti1.empty(),
                        "12_own": ti2.empty(),
                        "18_overall": ti3.empty(),
                        "18_own": ti4.empty(),
                    }

                    col12, col18 = st.columns(2)
                    pair_outputs: dict[str, dict] = {}
                    for panel_col, panel_size in zip([col12, col18], ["12-ML", "18-ML"]):
                        with panel_col:
                            st.markdown(f"**{panel_size} Slab Inputs**")
                            slab_state_panel = pair_state[panel_size]["slab_state"]
                            slab_opts_panel = pair_state[panel_size]["slab_opts"]

                            scenario_panel: dict[str, float] = {}
                            delta_placeholders: dict[str, object] = {}
                            qty_placeholders: dict[str, object] = {}
                            st.markdown("`Slab` | `Scenario Discount %` | `Volume Delta %` | `Scenario Qty`")
                            for slab_panel in slab_opts_panel:
                                s_panel = str(slab_panel)
                                default_panel = float(slab_state_panel[s_panel]["own_default"])
                                key_panel = f"ns_pair_{panel_size}_{s_panel}"
                                if key_panel not in st.session_state:
                                    st.session_state[key_panel] = round(default_panel, 2)
                                i1, i2, i3, i4 = st.columns([1, 2, 1.2, 1.3])
                                i1.caption(s_panel)
                                scenario_panel[s_panel] = i2.number_input(
                                    f"{panel_size} {s_panel} Discount %",
                                    min_value=0.0,
                                    max_value=100.0,
                                    step=0.5,
                                    key=key_panel,
                                    label_visibility="collapsed",
                                )
                                delta_placeholders[s_panel] = i3.empty()
                                qty_placeholders[s_panel] = i4.empty()

                            default_panel_map = {
                                s: float(stv["own_default"]) for s, stv in slab_state_panel.items()
                            }
                            qty_default_panel, _, _ = _run_coupled_prediction_for_state(
                                slab_state_panel,
                                default_panel_map,
                            )
                            qty_scenario_panel, _, _ = _run_coupled_prediction_for_state(
                                slab_state_panel,
                                scenario_panel,
                            )

                            for slab_panel in slab_opts_panel:
                                s_panel = str(slab_panel)
                                pred_def = float(qty_default_panel.get(s_panel, 0.0))
                                pred_scn = float(qty_scenario_panel.get(s_panel, pred_def))
                                d_qty = pred_scn - pred_def
                                d_pct = (d_qty / pred_def * 100.0) if pred_def > 0 else np.nan
                                d_ph = delta_placeholders.get(s_panel)
                                q_ph = qty_placeholders.get(s_panel)
                                if d_ph is not None:
                                    if np.isfinite(d_pct):
                                        if d_pct > 0:
                                            d_ph.success(f"+{d_pct:.2f}%")
                                        elif d_pct < 0:
                                            d_ph.error(f"{d_pct:.2f}%")
                                        else:
                                            d_ph.info("0.00%")
                                    else:
                                        d_ph.caption("NA")
                                if q_ph is not None:
                                    q_ph.caption(f"{pred_scn:,.0f}")

                            total_def = float(sum(qty_default_panel.values()))
                            total_scn = float(sum(qty_scenario_panel.values()))
                            total_delta_pct = (
                                ((total_scn - total_def) / total_def) * 100.0
                                if total_def > 0
                                else np.nan
                            )
                            pair_outputs[panel_size] = {
                                "total_default": total_def,
                                "total_scenario": total_scn,
                                "total_delta_pct": total_delta_pct,
                            }

                    d12 = float(pair_outputs.get("12-ML", {}).get("total_delta_pct", np.nan))
                    d18 = float(pair_outputs.get("18-ML", {}).get("total_delta_pct", np.nan))
                    with top_placeholders["12_own"].container():
                        st.metric("12-ML Due To Own", f"{d12:+.2f}%" if np.isfinite(d12) else "NA")
                    with top_placeholders["18_own"].container():
                        st.metric("18-ML Due To Own", f"{d18:+.2f}%" if np.isfinite(d18) else "NA")
                    with top_placeholders["12_overall"].container():
                        st.metric("12-ML Overall", "NA")
                    with top_placeholders["18_overall"].container():
                        st.metric("18-ML Overall", "NA")

                    st.markdown("**Cross-Size Elasticity (Estimated from Data)**")
                    elasticity = fit_cross_size_elasticity_model(defined_df=defined_df, base_df=base_df)
                    if not elasticity:
                        st.warning("Could not estimate cross-size elasticity model (insufficient common months).")
                    else:
                        e12 = float(elasticity.get("cross_elasticity_12_from_18", np.nan))
                        e18 = float(elasticity.get("cross_elasticity_18_from_12", np.nan))
                        r2_12 = float(elasticity.get("r2_12", np.nan))
                        r2_18 = float(elasticity.get("r2_18", np.nan))

                        em1, em2 = st.columns(2)
                        em1.metric("Q12 Model Cross Elasticity (12 wrt 18)", f"{e12:.3f}" if np.isfinite(e12) else "NA")
                        em2.metric("Q18 Model Cross Elasticity (18 wrt 12)", f"{e18:.3f}" if np.isfinite(e18) else "NA")

                        rv1, rv2 = st.columns(2)
                        rv1.caption(f"Q12 model R2: {r2_12:.3f}" if np.isfinite(r2_12) else "Q12 model R2: NA")
                        rv2.caption(f"Q18 model R2: {r2_18:.3f}" if np.isfinite(r2_18) else "Q18 model R2: NA")

                        if np.isfinite(d12) and np.isfinite(d18) and np.isfinite(e12) and np.isfinite(e18):
                            # One-way cross pass (no recursive bounce-back):
                            # own stays fixed; each size gets direct impact from the other size's own change once.
                            final_12 = d12 + (e12 * d18)
                            final_18 = d18 + (e18 * d12)
                            with top_placeholders["12_overall"].container():
                                st.metric("12-ML Overall", f"{final_12:+.2f}%")
                            with top_placeholders["18_overall"].container():
                                st.metric("18-ML Overall", f"{final_18:+.2f}%")
                        else:
                            st.warning("Cross impact could not be computed (check deltas/elasticities).")

                        with st.expander("Elasticity Model Coefficients", expanded=False):
                            coef_tbl = elasticity.get("coef_table", pd.DataFrame())
                            if coef_tbl is None or coef_tbl.empty:
                                st.caption("No coefficient details available.")
                            else:
                                q12 = coef_tbl[coef_tbl["Model"] == "Q12 model"]
                                q18 = coef_tbl[coef_tbl["Model"] == "Q18 model"]
                                c12, c18 = st.columns(2)
                                with c12:
                                    st.markdown("**Q12 Model Betas**")
                                    for _, r in q12.iterrows():
                                        st.caption(f"{r['Feature']}: {float(r['Beta']):.4f}")
                                with c18:
                                    st.markdown("**Q18 Model Betas**")
                                    for _, r in q18.iterrows():
                                        st.caption(f"{r['Feature']}: {float(r['Beta']):.4f}")

                    st.markdown("**Baseline Projection (No Discount Components)**")
                    total_baseline_df, size_baseline_df, detail_baseline_df = build_new_strategy_baseline_projection(ns_results)
                    if total_baseline_df.empty or size_baseline_df.empty:
                        st.info("Baseline projection is not available.")
                    else:
                        latest_size = (
                            size_baseline_df.sort_values("Period")
                            .groupby("Size", as_index=False)
                            .tail(1)
                            .reset_index(drop=True)
                        )
                        latest_12 = latest_size[latest_size["Size"] == "12-ML"]["BaselineQty"]
                        latest_18 = latest_size[latest_size["Size"] == "18-ML"]["BaselineQty"]
                        b1, b2, b3 = st.columns(3)
                        b1.metric(
                            "12-ML Baseline Qty",
                            f"{float(latest_12.iloc[0]):,.0f}" if not latest_12.empty else "NA",
                        )
                        b2.metric(
                            "18-ML Baseline Qty",
                            f"{float(latest_18.iloc[0]):,.0f}" if not latest_18.empty else "NA",
                        )
                        b3.metric(
                            "Total Baseline Qty",
                            f"{float(total_baseline_df['TotalBaselineQty'].iloc[-1]):,.0f}",
                        )

                        elasticity = fit_cross_size_elasticity_model(defined_df=defined_df, base_df=base_df)
                        baseline_wide_for_top = (
                            size_baseline_df.pivot(index="Period", columns="Size", values="BaselineQty")
                            .reset_index()
                            .sort_values("Period")
                        )
                        if "12-ML" not in baseline_wide_for_top.columns:
                            baseline_wide_for_top["12-ML"] = np.nan
                        if "18-ML" not in baseline_wide_for_top.columns:
                            baseline_wide_for_top["18-ML"] = np.nan
                        baseline_wide_for_top["change_pct_12"] = (
                            baseline_wide_for_top["12-ML"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0) * 100.0
                        )
                        baseline_wide_for_top["change_pct_18"] = (
                            baseline_wide_for_top["18-ML"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0) * 100.0
                        )

                        if elasticity:
                            e12_cross = float(elasticity.get("cross_elasticity_12_from_18", np.nan))
                            e18_cross = float(elasticity.get("cross_elasticity_18_from_12", np.nan))
                            baseline_wide_for_top["baseline_12_adjusted"] = baseline_wide_for_top["12-ML"] * (
                                1.0 + ((e12_cross * baseline_wide_for_top["change_pct_18"]) / 100.0)
                            )
                            baseline_wide_for_top["baseline_18_adjusted"] = baseline_wide_for_top["18-ML"] * (
                                1.0 + ((e18_cross * baseline_wide_for_top["change_pct_12"]) / 100.0)
                            )
                        else:
                            baseline_wide_for_top["baseline_12_adjusted"] = baseline_wide_for_top["12-ML"]
                            baseline_wide_for_top["baseline_18_adjusted"] = baseline_wide_for_top["18-ML"]

                        baseline_wide_for_top["baseline_12_adjusted"] = baseline_wide_for_top["baseline_12_adjusted"].clip(lower=0.0)
                        baseline_wide_for_top["baseline_18_adjusted"] = baseline_wide_for_top["baseline_18_adjusted"].clip(lower=0.0)
                        baseline_wide_for_top["baseline_total_adjusted"] = (
                            baseline_wide_for_top["baseline_12_adjusted"] + baseline_wide_for_top["baseline_18_adjusted"]
                        )

                        fig_baseline_total = go.Figure()
                        fig_baseline_total.add_trace(
                            go.Scatter(
                                x=baseline_wide_for_top["Period"],
                                y=baseline_wide_for_top["baseline_12_adjusted"],
                                mode="lines+markers",
                                name="12-ML Final Baseline",
                                line=dict(color="#2563eb", width=2),
                            )
                        )
                        fig_baseline_total.add_trace(
                            go.Scatter(
                                x=baseline_wide_for_top["Period"],
                                y=baseline_wide_for_top["baseline_18_adjusted"],
                                mode="lines+markers",
                                name="18-ML Final Baseline",
                                line=dict(color="#dc2626", width=2),
                            )
                        )
                        fig_baseline_total.add_trace(
                            go.Scatter(
                                x=baseline_wide_for_top["Period"],
                                y=baseline_wide_for_top["baseline_total_adjusted"],
                                mode="lines+markers",
                                name="Total Baseline Volume (12-ML + 18-ML)",
                                line=dict(color="#0b7a75", width=4),
                            )
                        )
                        fig_baseline_total.update_layout(
                            title="Total Baseline Volume Projection (Adjusted for Cross Effect)",
                            height=380,
                            margin=dict(l=20, r=20, t=45, b=20),
                            xaxis_title="Month",
                            yaxis_title="Baseline Quantity",
                        )
                        st.plotly_chart(
                            fig_baseline_total,
                            use_container_width=True,
                            key="new_strategy_total_baseline_projection",
                        )

                        apply_cross_adjustment_forecast = st.checkbox(
                            "Apply Cross Adjustment In 3-Month Forecast",
                            value=True,
                            key="apply_cross_adjustment_forecast",
                        )
                        forecast_df = build_three_month_baseline_forecast(
                            size_baseline_df=size_baseline_df,
                            elasticity=elasticity,
                            horizon=3,
                            apply_cross_adjustment=apply_cross_adjustment_forecast,
                        )
                        if not forecast_df.empty:
                            st.markdown("**Next 3-Month Baseline Forecast**")
                            fc_only = forecast_df[forecast_df["Type"] == "Forecast"].copy()
                            if not fc_only.empty:
                                f1, f2, f3 = st.columns(3)
                                f1.metric(
                                    "12-ML Forecast (Next Month)",
                                    f"{float(fc_only['baseline_12_adjusted'].iloc[0]):,.0f}",
                                )
                                f2.metric(
                                    "18-ML Forecast (Next Month)",
                                    f"{float(fc_only['baseline_18_adjusted'].iloc[0]):,.0f}",
                                )
                                f3.metric(
                                    "Total Forecast (Next Month)",
                                    f"{float(fc_only['baseline_total_adjusted'].iloc[0]):,.0f}",
                                )

                            fig_fc = go.Figure()
                            hist = forecast_df[forecast_df["Type"] == "History"].copy()
                            fut = forecast_df[forecast_df["Type"] == "Forecast"].copy()
                            forecast_start = fut["Period"].min() if not fut.empty else None
                            for col_name, label, color in [
                                ("baseline_12_adjusted", "12-ML Final Baseline", "#2563eb"),
                                ("baseline_18_adjusted", "18-ML Final Baseline", "#dc2626"),
                                ("baseline_total_adjusted", "Total Final Baseline", "#0b7a75"),
                            ]:
                                fig_fc.add_trace(
                                    go.Scatter(
                                        x=hist["Period"],
                                        y=hist[col_name],
                                        mode="lines+markers",
                                        name=f"{label} - History",
                                        line=dict(color=color, width=3 if "Total" in label else 2),
                                    )
                                )
                                if not hist.empty and not fut.empty:
                                    fut_plot = pd.concat(
                                        [
                                            hist[["Period", col_name]].tail(1),
                                            fut[["Period", col_name]],
                                        ],
                                        ignore_index=True,
                                    )
                                else:
                                    fut_plot = fut[["Period", col_name]].copy()
                                fig_fc.add_trace(
                                    go.Scatter(
                                        x=fut_plot["Period"],
                                        y=fut_plot[col_name],
                                        mode="lines+markers",
                                        name=f"{label} - Forecast",
                                        line=dict(color=color, width=3 if "Total" in label else 2, dash="dash"),
                                    )
                                )
                            if forecast_start is not None:
                                fig_fc.add_vrect(
                                    x0=forecast_start,
                                    x1=fut["Period"].max() + pd.offsets.MonthBegin(1),
                                    fillcolor="#e5e7eb",
                                    opacity=0.18,
                                    line_width=0,
                                    layer="below",
                                )
                            fig_fc.update_layout(
                                title=(
                                    "3-Month Baseline Forecast (12-ML, 18-ML, Total)"
                                    if not apply_cross_adjustment_forecast
                                    else "3-Month Baseline Forecast (Cross-Adjusted)"
                                ),
                                height=420,
                                margin=dict(l=20, r=20, t=45, b=20),
                                xaxis_title="Month",
                                yaxis_title="Baseline Quantity",
                                legend=dict(orientation="h", y=-0.22, x=0),
                                xaxis=dict(tickformat="%b %Y"),
                            )
                            st.plotly_chart(
                                fig_fc,
                                use_container_width=True,
                                key="new_strategy_baseline_forecast_chart",
                            )

                        with st.expander("Slab-Level Baseline Lines", expanded=False):
                            fig_baseline_detail = go.Figure()
                            palette = ["#1d4ed8", "#0b7a75", "#f59e0b", "#7c3aed", "#dc2626", "#0891b2", "#65a30d"]
                            for idx, series_name in enumerate(detail_baseline_df["Series"].dropna().unique().tolist()):
                                part = detail_baseline_df[detail_baseline_df["Series"] == series_name].copy()
                                fig_baseline_detail.add_trace(
                                    go.Scatter(
                                        x=part["Period"],
                                        y=part["BaselineQty"],
                                        mode="lines+markers",
                                        name=series_name,
                                        line=dict(color=palette[idx % len(palette)], width=2),
                                    )
                                )
                            fig_baseline_detail.update_layout(
                                title="Slab-Level Baseline Projection",
                                height=480,
                                margin=dict(l=20, r=20, t=45, b=20),
                                xaxis_title="Month",
                                yaxis_title="Baseline Quantity",
                                legend_title="Series",
                            )
                            st.plotly_chart(
                                fig_baseline_detail,
                                use_container_width=True,
                                key="new_strategy_slab_baseline_projection",
                            )



if __name__ == "__main__":
    main()

