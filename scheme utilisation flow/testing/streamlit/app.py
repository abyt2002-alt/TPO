import pandas as pd
import streamlit as st
from pathlib import Path

PARQUET_PATH = Path(__file__).resolve().parents[2] / "output" / "step2_deduped_sales_lines.parquet"
PAGE_SIZE = 2000

st.set_page_config(page_title="Step 2 Sales Lines Validator", layout="wide")
st.title("Step 2 — Deduped Sales Lines Validator")

@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_parquet(PARQUET_PATH, engine="pyarrow")
    df["_Year"] = pd.to_datetime(df["Date"], errors="coerce").dt.year.astype("Int64")
    return df

df = load_data()
st.caption(f"Total rows loaded: **{len(df):,}**")

# ── Filters ───────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    years = sorted(df["_Year"].dropna().unique().astype(int).tolist())
    selected_years = st.multiselect("Year", years, default=years)

with col2:
    months = sorted(df["Month"].dropna().unique().tolist())
    selected_months = st.multiselect("Month", months, default=months)

with col3:
    sizes = sorted(df["Sizes"].dropna().unique().tolist())
    selected_sizes = st.multiselect("Size", sizes, default=sizes)

with col4:
    mrp_min = float(df["MRP"].min())
    mrp_max = float(df["MRP"].max())
    mrp_range = st.slider("MRP range", min_value=mrp_min, max_value=mrp_max,
                          value=(mrp_min, mrp_max), step=1.0)

outlet_search = st.text_input("Outlet ID (type to filter, comma-separate multiple)", value="")

# ── Apply filters ─────────────────────────────────────────────────────────────
outlet_ids = [o.strip() for o in outlet_search.split(",") if o.strip()]

mask = (
    df["_Year"].isin([int(y) for y in selected_years])
    & df["Month"].isin(selected_months)
    & df["Sizes"].isin(selected_sizes)
    & df["MRP"].between(mrp_range[0], mrp_range[1])
)
if outlet_ids:
    mask = mask & df["Outlet_ID"].astype(str).isin(outlet_ids)
filtered = df[mask].reset_index(drop=True)

# ── Summary metrics ───────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
m1.metric("Rows", f"{len(filtered):,}")
m2.metric("Total Quantity", f"{filtered['Quantity'].sum():,.0f}")
m3.metric("Total Net Amt", f"₹{filtered['Net_Amt'].sum():,.0f}")
m4.metric("Total Claimable Discount", f"₹{filtered['Claimable_Scheme_Discount'].sum():,.0f}")

# ── Paginated table ───────────────────────────────────────────────────────────
st.markdown("---")

display_cols = [c for c in filtered.columns if c != "_Year"]

total_rows = len(filtered)
max_page = max(1, (total_rows - 1) // PAGE_SIZE + 1)

pc1, pc2 = st.columns([1, 5])
with pc1:
    page = st.number_input("Page", min_value=1, max_value=max_page, value=1, step=1)

start = (page - 1) * PAGE_SIZE
end = start + PAGE_SIZE
st.caption(f"Rows {start + 1:,} – {min(end, total_rows):,} of {total_rows:,}  (page {page}/{max_page})")
st.dataframe(filtered[display_cols].iloc[start:end], use_container_width=True, height=600)
