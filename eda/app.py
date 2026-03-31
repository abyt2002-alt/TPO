from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import pyarrow.parquet as pq


st.set_page_config(page_title="EDA Tool", layout="wide")
PLOT_CONFIG = {"displayModeBar": False, "displaylogo": False, "responsive": True}


def _safe_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def _normalize_text(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.upper()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def _find_data_dir() -> Path:
    candidates = [Path("DATA"), Path("../DATA"), Path.cwd() / "DATA"]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("DATA folder not found. Expected DATA or ../DATA")


def _round_discount_series(values, step: float = 0.5):
    arr = pd.Series(values, dtype=float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    step = float(step) if float(step) > 0 else 0.5
    rounded = np.round(arr / step) * step
    return np.clip(rounded, 0.0, 100.0)


def _estimate_base_discount_monthly_blocks(
    period_series,
    discount_series,
    min_upward_jump_pp: float = 1.0,
    min_downward_drop_pp: float = 1.0,
    round_step: float = 0.5,
):
    periods = pd.to_datetime(pd.Series(period_series), errors="coerce")
    discounts = pd.Series(discount_series, copy=False).astype(float).replace([np.inf, -np.inf], np.nan)
    valid = periods.notna() & discounts.notna()
    if valid.sum() == 0:
        return np.array([]), np.array([], dtype=bool)

    work = pd.DataFrame(
        {
            "Period": periods[valid].to_numpy(),
            "Discount": discounts[valid].to_numpy(dtype=float),
        }
    ).sort_values("Period", kind="stable").reset_index(drop=True)

    if work.empty:
        return np.array([]), np.array([], dtype=bool)

    work["Month_Key"] = pd.to_datetime(work["Period"], errors="coerce").dt.to_period("M")
    monthly = (
        work.groupby("Month_Key", as_index=False)
        .agg(
            Period=("Period", "min"),
            Discount=("Discount", "median"),
        )
        .sort_values("Period", kind="stable")
        .reset_index(drop=True)
    )

    if monthly.empty:
        return np.array([]), np.array([], dtype=bool)

    discounts_month = pd.Series(monthly["Discount"], dtype=float).interpolate(limit_direction="both").bfill().ffill()
    discounts_month = _round_discount_series(discounts_month, step=round_step)

    n = len(discounts_month)
    base = np.zeros(n, dtype=float)
    transitions = np.zeros(n, dtype=bool)

    min_upward_jump_pp = max(0.0, float(min_upward_jump_pp))
    min_downward_drop_pp = max(0.0, float(min_downward_drop_pp))

    prev_base = None
    for i in range(n):
        candidate = float(discounts_month[i]) if pd.notna(discounts_month[i]) else np.nan
        if not np.isfinite(candidate):
            candidate = prev_base if prev_base is not None else 0.0

        if prev_base is None:
            block_base = candidate
        else:
            if candidate > prev_base and (candidate - prev_base) < min_upward_jump_pp:
                block_base = prev_base
            elif candidate < prev_base and (prev_base - candidate) < min_downward_drop_pp:
                block_base = prev_base
            else:
                block_base = candidate
            if abs(block_base - prev_base) > 1e-9:
                transitions[i] = True

        base[i] = np.clip(float(block_base), 0.0, 100.0)
        prev_base = float(block_base)

    return base, transitions


def _apply_step2_base_discount_monthly(grp: pd.DataFrame) -> pd.DataFrame:
    out = grp.sort_values("PeriodStart", kind="stable").reset_index(drop=True).copy()
    out["Discount_Level_%"] = np.where(out["Sales"] != 0, out["Discount"] / out["Sales"] * 100.0, np.nan)
    base_monthly, _ = _estimate_base_discount_monthly_blocks(
        out["PeriodStart"],
        out["Discount_Level_%"],
        min_upward_jump_pp=1.0,
        min_downward_drop_pp=1.0,
        round_step=0.5,
    )
    if len(base_monthly) == len(out):
        out["Base_Discount_%"] = _round_discount_series(base_monthly, step=0.5)
    else:
        out["Base_Discount_%"] = np.nan
    return out


@st.cache_data(show_spinner=True)
def load_data() -> pd.DataFrame:
    data_dir = _find_data_dir()
    files = sorted(data_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    want_cols = [
        "Date",
        "Subcategory",
        "subcategory",
        "Sizes",
        "sizes",
        "Outlet_ID",
        "outlet_id",
        "OutletId",
        "outletid",
        "Final_State",
        "State",
        "state",
        "Outlet_Type",
        "outlet_type",
        "SalesValue_atBasicRate",
        "salesvalue_atbasicrate",
        "Quantity",
        "quantity",
        "TotalDiscount",
        "totaldiscount",
        "MRP",
        "mrp",
        "Basic_Rate_Per_PC",
        "basic_rate_per_pc",
    ]

    frames: List[pd.DataFrame] = []
    for f in files:
        schema_cols = set(pq.ParquetFile(f).schema.names)
        cols = [c for c in want_cols if c in schema_cols]
        if "Date" not in cols:
            continue
        df = pd.read_parquet(f, columns=cols)
        frames.append(df)

    if not frames:
        raise RuntimeError("No parquet file had required Date column.")

    df = pd.concat(frames, ignore_index=True)

    # Canonical columns
    if "Sizes" not in df.columns and "sizes" in df.columns:
        df["Sizes"] = df["sizes"]
    if "Subcategory" not in df.columns and "subcategory" in df.columns:
        df["Subcategory"] = df["subcategory"]
    if "Outlet_ID" not in df.columns:
        for alt in ["outlet_id", "OutletId", "outletid"]:
            if alt in df.columns:
                df["Outlet_ID"] = df[alt]
                break
    if "State" not in df.columns and "state" in df.columns:
        df["State"] = df["state"]
    if "Outlet_Type" not in df.columns and "outlet_type" in df.columns:
        df["Outlet_Type"] = df["outlet_type"]
    if "SalesValue_atBasicRate" not in df.columns and "salesvalue_atbasicrate" in df.columns:
        df["SalesValue_atBasicRate"] = df["salesvalue_atbasicrate"]
    if "Quantity" not in df.columns and "quantity" in df.columns:
        df["Quantity"] = df["quantity"]
    if "TotalDiscount" not in df.columns and "totaldiscount" in df.columns:
        df["TotalDiscount"] = df["totaldiscount"]
    if "MRP" not in df.columns and "mrp" in df.columns:
        df["MRP"] = df["mrp"]
    if "Basic_Rate_Per_PC" not in df.columns and "basic_rate_per_pc" in df.columns:
        df["Basic_Rate_Per_PC"] = df["basic_rate_per_pc"]

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).copy()

    if "Sizes" not in df.columns:
        df["Sizes"] = "UNKNOWN"
    if "Subcategory" not in df.columns:
        raise RuntimeError("Subcategory column not found in parquet data.")
    if "Outlet_ID" not in df.columns:
        df["Outlet_ID"] = "UNKNOWN"

    if "Final_State" in df.columns:
        df["State_Use"] = df["Final_State"]
    elif "State" in df.columns:
        df["State_Use"] = df["State"]
    else:
        df["State_Use"] = "UNKNOWN"

    if "Outlet_Type" in df.columns:
        df["Channel_Use"] = df["Outlet_Type"]
    else:
        df["Channel_Use"] = "UNKNOWN"

    df["Subcategory"] = _normalize_text(df["Subcategory"])
    df["Sizes"] = _normalize_text(df["Sizes"]).str.replace(" ", "", regex=False)
    df["Sizes"] = df["Sizes"].str.replace(r"^12(?:-)?ML$", "12-ML", regex=True)
    df["Sizes"] = df["Sizes"].str.replace(r"^18(?:-)?ML$", "18-ML", regex=True)
    df["Outlet_Use"] = _normalize_text(df["Outlet_ID"])
    df["State_Use"] = _normalize_text(df["State_Use"])
    df["Channel_Use"] = _normalize_text(df["Channel_Use"])

    # Fixed scope for this EDA: STX + STREAX Insta Shampoo combined.
    target_subcats = {"STX INSTA SHAMPOO", "STREAX INSTA SHAMPOO"}
    df = df[df["Subcategory"].isin(target_subcats)].copy()
    df = df[df["Sizes"].isin({"12-ML", "18-ML"})].copy()

    df["Sales"] = _safe_num(df["SalesValue_atBasicRate"]) if "SalesValue_atBasicRate" in df.columns else 0.0
    df["Volume"] = _safe_num(df["Quantity"]) if "Quantity" in df.columns else 0.0
    df["Discount"] = _safe_num(df["TotalDiscount"]) if "TotalDiscount" in df.columns else 0.0
    df["MRP_Use"] = _safe_num(df["MRP"]) if "MRP" in df.columns else np.nan
    df["BasicRate_Use"] = _safe_num(df["Basic_Rate_Per_PC"]) if "Basic_Rate_Per_PC" in df.columns else np.nan

    df["Period"] = df["Date"].dt.to_period("M").astype(str)
    df["PeriodStart"] = df["Date"].dt.to_period("M").dt.to_timestamp()

    # Step-2 style slab definition (ignore dataset slab):
    # monthly quantity per outlet per size -> slab assignment.
    outlet_month = (
        df.groupby(["Period", "Sizes", "Outlet_Use"], as_index=False)["Volume"]
        .sum()
        .rename(columns={"Volume": "Monthly_Outlet_Qty"})
    )
    df = df.merge(outlet_month, on=["Period", "Sizes", "Outlet_Use"], how="left")
    q = pd.to_numeric(df["Monthly_Outlet_Qty"], errors="coerce").fillna(0.0)
    s = df["Sizes"].astype(str)
    df["Slab"] = np.where(
        s == "12-ML",
        np.select([q < 8, q < 144], ["SLAB0", "SLAB1"], default="SLAB2"),
        np.where(
            s == "18-ML",
            np.select([q < 8, q < 32, q < 576, q < 960], ["SLAB0", "SLAB1", "SLAB2", "SLAB3"], default="SLAB4"),
            "UNKNOWN",
        ),
    )

    return df[
        [
            "Date",
            "Period",
            "PeriodStart",
            "Subcategory",
            "Sizes",
            "Slab",
            "State_Use",
            "Channel_Use",
            "Sales",
            "Volume",
            "Discount",
            "MRP_Use",
            "BasicRate_Use",
        ]
    ].copy()


def make_monthly(df: pd.DataFrame) -> pd.DataFrame:
    w = df.copy()
    w["MRP_x_Vol"] = w["MRP_Use"] * w["Volume"]
    w["BR_x_Vol"] = w["BasicRate_Use"] * w["Volume"]

    grp = (
        w.groupby(["Period", "PeriodStart"], as_index=False)
        .agg(
            Sales=("Sales", "sum"),
            Volume=("Volume", "sum"),
            Discount=("Discount", "sum"),
            MRP_x_Vol=("MRP_x_Vol", "sum"),
            BR_x_Vol=("BR_x_Vol", "sum"),
        )
        .sort_values("PeriodStart")
        .reset_index(drop=True)
    )

    grp = _apply_step2_base_discount_monthly(grp)
    grp["Base_Price"] = np.where(grp["Volume"] != 0, grp["Sales"] / grp["Volume"], np.nan)
    grp["MRP"] = np.where(grp["Volume"] != 0, grp["MRP_x_Vol"] / grp["Volume"], np.nan)

    grp["Sales_Growth_%"] = grp["Sales"].pct_change() * 100.0
    grp["Volume_Growth_%"] = grp["Volume"].pct_change() * 100.0
    return grp


def make_monthly_by(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    w = df.copy()
    w["MRP_x_Vol"] = w["MRP_Use"] * w["Volume"]
    w["BR_x_Vol"] = w["BasicRate_Use"] * w["Volume"]

    grp = (
        w.groupby(["Period", "PeriodStart", group_col], as_index=False)
        .agg(
            Sales=("Sales", "sum"),
            Volume=("Volume", "sum"),
            Discount=("Discount", "sum"),
            MRP_x_Vol=("MRP_x_Vol", "sum"),
            BR_x_Vol=("BR_x_Vol", "sum"),
        )
        .sort_values(["PeriodStart", group_col])
        .reset_index(drop=True)
    )
    out_parts = []
    for gval, part in grp.groupby(group_col, sort=False):
        part2 = _apply_step2_base_discount_monthly(part)
        part2[group_col] = gval
        out_parts.append(part2)
    grp = pd.concat(out_parts, ignore_index=True) if out_parts else grp
    grp["Base_Price"] = np.where(grp["Volume"] != 0, grp["Sales"] / grp["Volume"], np.nan)
    grp["MRP"] = np.where(grp["Volume"] != 0, grp["MRP_x_Vol"] / grp["Volume"], np.nan)
    return grp


def line_chart_single(x, y, name, color, title_text: str | None = None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name=name, line=dict(color=color, width=2)))
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=40, r=18, t=54, b=40),
        height=360,
        title=dict(text=title_text or name, x=0.01, xanchor="left"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
        font=dict(size=12, color="#334155"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e2e8f0", title="Month", tickformat="%b %Y")
    fig.update_yaxes(showgrid=True, gridcolor="#e2e8f0")
    return fig


def line_chart_sales_volume(df_monthly: pd.DataFrame, title_text: str = "Sales vs Volume"):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_monthly["PeriodStart"],
            y=df_monthly["Sales"],
            mode="lines+markers",
            name="Sales",
            line=dict(color="#1f77b4", width=2),
            yaxis="y1",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_monthly["PeriodStart"],
            y=df_monthly["Volume"],
            mode="lines+markers",
            name="Volume",
            line=dict(color="#ff7f0e", width=2),
            yaxis="y2",
        )
    )
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=40, r=18, t=54, b=40),
        height=360,
        title=dict(text=title_text, x=0.01, xanchor="left"),
        yaxis=dict(title="Sales"),
        yaxis2=dict(title="Volume", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
        font=dict(size=12, color="#334155"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e2e8f0", title="Month", tickformat="%b %Y")
    fig.update_yaxes(showgrid=True, gridcolor="#e2e8f0")
    return fig


def multi_line_chart(df: pd.DataFrame, group_col: str, y_col: str, title: str, y_title: str):
    fig = go.Figure()
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]
    groups = sorted(df[group_col].dropna().astype(str).unique().tolist())
    for i, g in enumerate(groups):
        s = df[df[group_col].astype(str) == g].sort_values("PeriodStart")
        fig.add_trace(
            go.Scatter(
                x=s["PeriodStart"],
                y=s[y_col],
                mode="lines+markers",
                name=g,
                line=dict(color=palette[i % len(palette)], width=2),
            )
        )
    fig.update_layout(
        title=dict(text=title, x=0.01, xanchor="left"),
        template="plotly_white",
        margin=dict(l=40, r=18, t=58, b=40),
        height=360,
        yaxis=dict(title=y_title),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
        font=dict(size=12, color="#334155"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e2e8f0", title="Month", tickformat="%b %Y")
    fig.update_yaxes(showgrid=True, gridcolor="#e2e8f0")
    return fig


def render_size_tab(
    size_df: pd.DataFrame,
    size_label: str,
    sel_months: List[str],
    sel_states: List[str],
    sel_slabs: List[str],
    sel_channels: List[str],
):
    if size_df.empty:
        st.warning(f"No data for {size_label} with current filters.")
        return

    monthly = make_monthly(size_df)
    if monthly.empty:
        st.warning(f"No monthly data available for {size_label}.")
        return

    latest = monthly.iloc[-1]
    prev = monthly.iloc[-2] if len(monthly) > 1 else None
    total_sales = monthly["Sales"].sum()
    total_volume = monthly["Volume"].sum()
    sales_growth = ((latest["Sales"] - prev["Sales"]) / prev["Sales"] * 100.0) if prev is not None and prev["Sales"] != 0 else np.nan
    volume_growth = ((latest["Volume"] - prev["Volume"]) / prev["Volume"] * 100.0) if prev is not None and prev["Volume"] != 0 else np.nan
    latest_period = str(monthly["Period"].iloc[-1])

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Absolute Sales", f"{total_sales:,.0f}")
    m2.metric("Absolute Volume", f"{total_volume:,.0f}")
    m3.metric("Sales Growth (Latest MoM)", "NA" if pd.isna(sales_growth) else f"{sales_growth:+.2f}%")
    m4.metric("Volume Growth (Latest MoM)", "NA" if pd.isna(volume_growth) else f"{volume_growth:+.2f}%")

    st.markdown(f"**Latest Month in View:** `{latest_period}`")
    l1, l2, l3, l4 = st.columns(4)
    l1.metric("Sales (Latest Month)", f"{latest['Sales']:,.0f}")
    l2.metric("Volume (Latest Month)", f"{latest['Volume']:,.0f}")
    l3.metric("Base Discount % (Latest)", "NA" if pd.isna(latest["Base_Discount_%"]) else f"{latest['Base_Discount_%']:.2f}%")
    l4.metric("Base Price (Latest)", "NA" if pd.isna(latest["Base_Price"]) else f"{latest['Base_Price']:.2f}")

    st.markdown("### Trends")
    c0, c1 = st.columns(2)
    c0.plotly_chart(
        line_chart_sales_volume(monthly, f"{size_label} Sales vs Volume"),
        use_container_width=True,
        config=PLOT_CONFIG,
    )
    c1.plotly_chart(
        line_chart_single(
            monthly["PeriodStart"],
            monthly["Base_Discount_%"],
            f"{size_label} Base Discount %",
            "#2ca02c",
            title_text=f"{size_label} Base Discount % (Step 2 logic)",
        ),
        use_container_width=True,
        config=PLOT_CONFIG,
    )

    fig_price = go.Figure()
    fig_price.add_trace(
        go.Scatter(
            x=monthly["PeriodStart"],
            y=monthly["Base_Price"],
            mode="lines+markers",
            name="Base Price",
            line=dict(color="#9467bd", width=2),
        )
    )
    fig_price.add_trace(
        go.Scatter(
            x=monthly["PeriodStart"],
            y=monthly["MRP"],
            mode="lines+markers",
            name="MRP",
            line=dict(color="#d62728", width=2),
        )
    )
    fig_price.update_layout(
        title=dict(text=f"{size_label} Base Price & MRP", x=0.01, xanchor="left"),
        template="plotly_white",
        margin=dict(l=40, r=18, t=58, b=40),
        height=360,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
        font=dict(size=12, color="#334155"),
    )
    fig_price.update_xaxes(showgrid=True, gridcolor="#e2e8f0", title="Month", tickformat="%b %Y")
    fig_price.update_yaxes(showgrid=True, gridcolor="#e2e8f0", title="Price")
    st.plotly_chart(fig_price, use_container_width=True, config=PLOT_CONFIG)

    by_state = make_monthly_by(size_df, "State_Use")
    by_slab = make_monthly_by(size_df, "Slab")

    st.markdown("### State Level")
    s1, s2 = st.columns(2)
    s1.plotly_chart(
        multi_line_chart(by_state, "State_Use", "Sales", f"{size_label} Sales by State", "Sales"),
        use_container_width=True,
        config=PLOT_CONFIG,
    )
    s2.plotly_chart(
        multi_line_chart(by_state, "State_Use", "Volume", f"{size_label} Volume by State", "Volume"),
        use_container_width=True,
        config=PLOT_CONFIG,
    )

    st.markdown("### Slab Level")
    b1, b2 = st.columns(2)
    b1.plotly_chart(
        multi_line_chart(by_slab, "Slab", "Sales", f"{size_label} Sales by Slab", "Sales"),
        use_container_width=True,
        config=PLOT_CONFIG,
    )
    b2.plotly_chart(
        multi_line_chart(by_slab, "Slab", "Volume", f"{size_label} Volume by Slab", "Volume"),
        use_container_width=True,
        config=PLOT_CONFIG,
    )

    st.markdown("### Data Tables")
    with st.expander("Show selected filter context", expanded=False):
        cxt1, cxt2, cxt3 = st.columns(3)
        cxt1.markdown(f"<div class='small-muted'><b>Months:</b> {len(sel_months) if sel_months else 0}</div>", unsafe_allow_html=True)
        cxt2.markdown(
            f"<div class='small-muted'><b>States:</b> {', '.join(sel_states[:6]) + (' ...' if len(sel_states) > 6 else '') if sel_states else 'All'}</div>",
            unsafe_allow_html=True,
        )
        cxt3.markdown(
            f"<div class='small-muted'><b>Slabs:</b> {', '.join(sel_slabs) if sel_slabs else 'All'} | <b>Channels:</b> {', '.join(sel_channels) if sel_channels else 'All'}</div>",
            unsafe_allow_html=True,
        )

    size_table = monthly[
        [
            "Period",
            "Sales",
            "Volume",
            "Sales_Growth_%",
            "Volume_Growth_%",
            "Base_Discount_%",
            "Base_Price",
            "MRP",
        ]
    ].copy()
    st.markdown("#### Size Level Monthly")
    st.dataframe(size_table, use_container_width=True, height=240)

    state_table = by_state[
        ["Period", "State_Use", "Sales", "Volume", "Base_Discount_%", "Base_Price", "MRP"]
    ].copy()
    st.markdown("#### State Level Monthly")
    st.dataframe(state_table, use_container_width=True, height=240)

    slab_table = by_slab[
        ["Period", "Slab", "Sales", "Volume", "Base_Discount_%", "Base_Price", "MRP"]
    ].copy()
    st.markdown("#### Slab Level Monthly")
    st.dataframe(slab_table, use_container_width=True, height=240)


def main():
    st.markdown(
        """
        <h1 style="margin:0 0 4px 0; line-height:1.25; font-size:2rem; color:#0f172a;">
          EDA Tool
        </h1>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Professional EDA | Fixed scope: STX INSTA SHAMPOO + STREAX INSTA SHAMPOO | Slab: monthly outlet quantity logic | Discount: Step 2 monthly base-discount logic")
    st.markdown(
        """
        <style>
          .block-container {padding-top: 0.55rem; padding-bottom: 0.6rem; max-width: 100%;}
          div[data-testid="stMetric"] {
            background:#f8fafc; border:1px solid #e2e8f0; padding:8px 10px; border-radius:8px;
          }
          div[data-testid="stMultiSelect"] label, div[data-testid="stSelectbox"] label {
            font-weight: 600; font-size: 0.9rem;
          }
          div[data-testid="stTabs"] button {font-weight: 600;}
          .small-muted {color:#64748b; font-size:0.84rem;}
          h1, h2, h3 {overflow: visible !important; text-overflow: unset !important; white-space: normal !important;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    df = load_data()

    with st.sidebar:
        st.markdown("### Filters")
        month_options = sorted(df["Period"].dropna().unique().tolist())
        default_months = month_options
        sel_months = st.multiselect("Month", month_options, default=default_months)

        slab_options = sorted(df["Slab"].dropna().unique().tolist(), key=lambda x: (999 if x == "NAN" else 0, x))
        sel_slabs = st.multiselect("Slab", slab_options, default=slab_options)

        state_options = sorted(df["State_Use"].dropna().unique().tolist())
        sel_states = st.multiselect("State", state_options, default=state_options)

        channel_options = sorted(df["Channel_Use"].dropna().unique().tolist())
        sel_channels = st.multiselect("Channel", channel_options, default=channel_options)

        st.markdown(
            "<div class='small-muted'>Subcategory fixed: STX INSTA SHAMPOO + STREAX INSTA SHAMPOO</div>",
            unsafe_allow_html=True,
        )

    f = df.copy()
    if sel_months:
        f = f[f["Period"].isin(sel_months)]
    if sel_slabs:
        f = f[f["Slab"].isin(sel_slabs)]
    if sel_states:
        f = f[f["State_Use"].isin(sel_states)]
    if sel_channels:
        f = f[f["Channel_Use"].isin(sel_channels)]

    if f.empty:
        st.warning("No data for selected filters.")
        return

    tab_pack, tab_12, tab_18 = st.tabs(["Pack Size Trends", "12-ML", "18-ML"])
    with tab_pack:
        by_size = make_monthly_by(f, "Sizes")
        p1, p2 = st.columns(2)
        p1.plotly_chart(
            multi_line_chart(by_size, "Sizes", "Sales", "Sales by Pack Size", "Sales"),
            use_container_width=True,
            config=PLOT_CONFIG,
        )
        p2.plotly_chart(
            multi_line_chart(by_size, "Sizes", "Volume", "Volume by Pack Size", "Volume"),
            use_container_width=True,
            config=PLOT_CONFIG,
        )
        p3, p4 = st.columns(2)
        p3.plotly_chart(
            multi_line_chart(by_size, "Sizes", "Base_Discount_%", "Base Discount % by Pack Size (Step 2 logic)", "Base Discount %"),
            use_container_width=True,
            config=PLOT_CONFIG,
        )
        p4.plotly_chart(
            multi_line_chart(by_size, "Sizes", "Base_Price", "Base Price by Pack Size", "Base Price"),
            use_container_width=True,
            config=PLOT_CONFIG,
        )
    with tab_12:
        render_size_tab(
            f[f["Sizes"] == "12-ML"].copy(),
            "12-ML",
            sel_months,
            sel_states,
            sel_slabs,
            sel_channels,
        )
    with tab_18:
        render_size_tab(
            f[f["Sizes"] == "18-ML"].copy(),
            "18-ML",
            sel_months,
            sel_states,
            sel_slabs,
            sel_channels,
        )


if __name__ == "__main__":
    main()
