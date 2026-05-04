"""
Creates demo_data/ folder with top 10% outlets by SalesValue_atBasicRate.
Run once on the server: python3 create_demo_data.py
"""
import os
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "DATA"
DEMO_DIR = Path(__file__).parent.parent / "demo_data"

DEMO_DIR.mkdir(exist_ok=True)

parquet_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.parquet')])
print(f"Found {len(parquet_files)} parquet files in {DATA_DIR}")

# Load all parquets to compute per-outlet total SalesValue_atBasicRate
print("Computing outlet sales totals...")
frames = []
for f in parquet_files:
    df = pd.read_parquet(DATA_DIR / f, columns=['Outlet_ID', 'SalesValue_atBasicRate'])
    frames.append(df)

all_data = pd.concat(frames, ignore_index=True)
outlet_totals = (
    all_data.groupby('Outlet_ID')['SalesValue_atBasicRate']
    .sum()
    .reset_index()
    .sort_values('SalesValue_atBasicRate', ascending=False)
)

top_10_pct = int(len(outlet_totals) * 0.10)
top_outlets = set(outlet_totals.head(top_10_pct)['Outlet_ID'].tolist())
print(f"Total outlets: {len(outlet_totals):,} — keeping top 10%: {top_10_pct:,} outlets")
del all_data, frames, outlet_totals

# Filter each parquet and write to demo_data/
total_rows_in = 0
total_rows_out = 0
for f in parquet_files:
    df = pd.read_parquet(DATA_DIR / f)
    total_rows_in += len(df)
    df_filtered = df[df['Outlet_ID'].isin(top_outlets)]
    total_rows_out += len(df_filtered)
    df_filtered.to_parquet(DEMO_DIR / f, index=False)
    print(f"  {f}: {len(df):,} → {len(df_filtered):,} rows")

print(f"\nDone! {total_rows_in:,} → {total_rows_out:,} rows ({total_rows_out/total_rows_in*100:.1f}%)")
print(f"demo_data saved to: {DEMO_DIR}")
