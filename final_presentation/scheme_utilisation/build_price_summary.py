import pandas as pd
import os
import re
from datetime import datetime

FOLDER = r"Raw files"
OUTPUT = "price_movement_12ml_18ml.xlsx"

MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "june": 6,
    "jul": 7, "july": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

def parse_month_year(filename):
    name = filename.lower()
    match = re.match(r"([a-z]+)\s+(\d{2})\s+scheme", name)
    if match:
        mon_str, yr_str = match.group(1), match.group(2)
        mon = MONTH_MAP.get(mon_str)
        yr = 2000 + int(yr_str)
        if mon:
            return datetime(yr, mon, 1)
    return None

files = sorted(
    [f for f in os.listdir(FOLDER) if f.endswith(".xlsx") and not f.startswith("~")],
    key=lambda f: parse_month_year(f) or datetime(9999, 1, 1),
)

print(f"Reading {len(files)} files...")

all_frames = []
for f in files:
    dt = parse_month_year(f)
    if dt is None:
        print(f"  WARNING: Could not parse date from '{f}', skipping.")
        continue
    df = pd.read_excel(os.path.join(FOLDER, f), usecols=["SUb_Brand", "MRP", "Selling_Price"])
    df["Month"] = dt.strftime("%b %Y")
    df["_sort"] = dt
    all_frames.append(df)

data = pd.concat(all_frames, ignore_index=True)

# Filter 12 ML and 18 ML
mask_12 = data["SUb_Brand"].str.contains(r"\b12\s*ML\b", case=False, na=False)
mask_18 = data["SUb_Brand"].str.contains(r"\b18\s*ML\b", case=False, na=False)
filtered = data[mask_12 | mask_18].copy()

print(f"Rows after filtering (12 ML + 18 ML): {len(filtered):,}")

# Median price per Month x SUb_Brand (median handles outliers better than mean)
summary = (
    filtered.groupby(["_sort", "Month", "SUb_Brand"], sort=True)
    .agg(MRP=("MRP", "median"), Selling_Price=("Selling_Price", "median"))
    .reset_index()
    .sort_values(["SUb_Brand", "_sort"])
)

# Price change flag: compare to previous month for same product
summary["MRP_Prev"] = summary.groupby("SUb_Brand")["MRP"].shift(1)
summary["SP_Prev"] = summary.groupby("SUb_Brand")["Selling_Price"].shift(1)
summary["MRP_Changed"] = (summary["MRP"] != summary["MRP_Prev"]) & summary["MRP_Prev"].notna()
summary["SP_Changed"] = (summary["Selling_Price"] != summary["SP_Prev"]) & summary["SP_Prev"].notna()
summary["Price_Change_Flag"] = summary.apply(
    lambda r: "MRP+SP changed" if r["MRP_Changed"] and r["SP_Changed"]
    else "MRP changed" if r["MRP_Changed"]
    else "SP changed" if r["SP_Changed"]
    else ("First record" if pd.isna(r["MRP_Prev"]) else "No change"),
    axis=1,
)

# Size tag
summary["Size"] = summary["SUb_Brand"].str.extract(r"(\b(?:12|18)\s*ML\b)", expand=False)

out_cols = ["Month", "Size", "SUb_Brand", "MRP", "Selling_Price", "MRP_Prev", "SP_Prev", "Price_Change_Flag"]
final = summary[out_cols].copy()
final.columns = ["Month", "Size", "Sub_Brand", "MRP", "Selling_Price", "MRP_Prev_Month", "SP_Prev_Month", "Price_Change_Flag"]

final.to_excel(OUTPUT, index=False)
print(f"\nOutput saved: {OUTPUT}")
print(f"Rows: {len(final)}, Unique products: {final['Sub_Brand'].nunique()}")

print("\n=== PRICE CHANGES DETECTED ===")
changes = final[~final["Price_Change_Flag"].isin(["No change", "First record"])]
if changes.empty:
    print("No price changes found across any month.")
else:
    print(changes[["Month", "Sub_Brand", "MRP", "MRP_Prev_Month", "Selling_Price", "SP_Prev_Month", "Price_Change_Flag"]].to_string(index=False))

print("\n=== SAMPLE SUMMARY (first 5 rows per size) ===")
for sz in ["12 ML", "18 ML"]:
    sub = final[final["Size"].str.contains(sz, na=False)].head(5)
    print(f"\n{sz}:")
    print(sub[["Month", "Sub_Brand", "MRP", "Selling_Price", "Price_Change_Flag"]].to_string(index=False))
