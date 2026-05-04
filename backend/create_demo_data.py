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

# Load all parquets to compute per-outlet total SalesValue_atBasicRate per state
print("Computing outlet sales totals per state...")
frames = []
for f in parquet_files:
    df = pd.read_parquet(DATA_DIR / f, columns=['Outlet_ID', 'Final_State', 'SalesValue_atBasicRate'])
    frames.append(df)

all_data = pd.concat(frames, ignore_index=True)
outlet_totals = (
    all_data.groupby(['Outlet_ID', 'Final_State'])['SalesValue_atBasicRate']
    .sum()
    .reset_index()
)

top_outlets = set()
for state, group in outlet_totals.groupby('Final_State'):
    group_sorted = group.sort_values('SalesValue_atBasicRate', ascending=False)
    top_n = max(1, int(len(group_sorted) * 0.10))
    state_top = set(group_sorted.head(top_n)['Outlet_ID'].tolist())
    top_outlets.update(state_top)
    print(f"  {state}: {len(group_sorted):,} outlets → keeping top 10%: {top_n:,}")

print(f"Total outlets kept: {len(top_outlets):,}")
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
