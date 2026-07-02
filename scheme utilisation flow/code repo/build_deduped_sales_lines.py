from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


FLOW_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = FLOW_ROOT / "output"
DEFAULT_OUTPUT_PATH = FLOW_ROOT / "output" / "step2_deduped_sales_lines.parquet"
DEFAULT_REPORT_PATH = FLOW_ROOT / "output" / "step2_deduped_sales_lines_report.json"

LINE_KEY_COLS = ["Outlet_Id", "Bill No", "Batch_Id"]

OUTLET_CLASSIFICATION_MAPPING = {
    "GRO [Grocer Kirana]": "OtherGT",
    "NOV [Novelty/ Cosmetics/Fancy store]": "OtherGT",
    "WH [Wholesale]": "WH",
    "CH [Chemist/Pharmacy]": "OtherGT",
    "GM [General Merchant]": "OtherGT",
    "Others PanBidi Cloths toy store": "OtherGT",
    "Self Service": "SS",
    "Saloon [Barber 2-3 chair]": "OtherGT",
    "Retailer": "OtherGT",
    "Beauty Shop": "OtherGT",
    "B": "OtherGT",
}

COLS_NEEDED = [
    "source_file",
    "source_row_number",
    "Plant_Name",
    "Month",
    "Distributor_Type",
    "skunitid",
    "skucode",
    "SKU Name",
    "Category",
    "Subcategory",
    "SUb_Brand",
    "Batch_Id",
    "Batch No",
    "MRP",
    "Selling_Price",
    "gst",
    "Bill No",
    "Invoice Date",
    "Invoice qty. pieces",
    "Scheme Name",
    "Scheme_discount",
    "%_Scheme",
    "Sale_Value",
    "Distributor_Sale_Value",
    "Scheme_Reason",
    "Outlet_Classification",
    "Outlet_Id",
    "Outlet_Name",
]

FIRST_VALUE_COLS = [
    "source_file",
    "source_row_number",
    "Plant_Name",
    "Month",
    "Distributor_Type",
    "skunitid",
    "skucode",
    "SKU Name",
    "Category",
    "Subcategory",
    "SUb_Brand",
    "Batch_Id",
    "Batch No",
    "MRP",
    "Selling_Price",
    "gst",
    "Bill No",
    "Invoice Date",
    "Invoice qty. pieces",
    "Scheme Name",
    "Sale_Value",
    "Distributor_Sale_Value",
    "Outlet_Classification",
    "Outlet_Id",
    "Outlet_Name",
]


def read_monthly_parquets(input_dir: Path, output_path: Path) -> pd.DataFrame:
    files = sorted(
        path
        for path in input_dir.glob("*.parquet")
        if path.name != output_path.name and not path.name.startswith("step2_")
    )
    if not files:
        raise FileNotFoundError(f"No monthly parquet files found in: {input_dir}")

    frames = []
    for path in files:
        parquet_cols = pq.ParquetFile(path).schema_arrow.names
        usecols = [col for col in COLS_NEEDED if col in parquet_cols]
        missing = sorted(set(COLS_NEEDED) - set(usecols))
        if missing:
            raise ValueError(f"{path.name} is missing required columns: {missing}")
        print(f"Reading {path.name}", flush=True)
        frames.append(pd.read_parquet(path, columns=COLS_NEEDED, engine="pyarrow"))

    return pd.concat(frames, ignore_index=True)


def normalize_state(plant_name: object) -> str:
    text = str(plant_name or "").upper()
    if "MUMBAI" in text or "NAGPUR" in text:
        return "MAH"
    if "GHAZIABAD" in text or "LUCKNOW" in text or "VARANASI" in text:
        return "UP"
    return ""


def normalize_size(sub_brand: object) -> str:
    text = str(sub_brand or "").upper().replace("-", " ")
    text = " ".join(text.split())
    if "12 ML" in text or "12ML" in text:
        return "12-ML"
    if "18 ML" in text or "18ML" in text:
        return "18-ML"
    return ""


def scheme_reason_group(reason: object) -> str:
    text = str(reason or "").strip().upper()
    if text == "CLAIMABLE":
        return "Claimable"
    if text.startswith("NON-CLAIMABLE"):
        return "Non_Claimable"
    return "Other"


def clean_key_part(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def build_line_key(df: pd.DataFrame) -> pd.Series:
    parts = [clean_key_part(df[col]) for col in LINE_KEY_COLS]
    return parts[0] + "|" + parts[1] + "|" + parts[2]


def to_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def build_deduped_sales_lines(
    df: pd.DataFrame,
    *,
    stockiest_only: bool,
    exclude_samt: bool,
    keep_only_12_18: bool,
) -> tuple[pd.DataFrame, dict]:
    raw_rows = int(len(df))

    df = df.copy()
    df["Sizes"] = df["SUb_Brand"].map(normalize_size)

    if stockiest_only:
        df = df[df["Distributor_Type"].astype(str).str.strip().isin(["STOCKIEST DMS", "STOCKIEST"])].copy()

    if exclude_samt:
        df = df[~df["Scheme Name"].astype(str).str.upper().str.contains("SAMT", na=False)].copy()

    if keep_only_12_18:
        df = df[df["Sizes"].isin(["12-ML", "18-ML"])].copy()

    df["MRP"] = to_number(df["MRP"])
    invalid_mrp_rows = int((df["MRP"] > 25.0).sum())
    invalid_mrp_quantity = float(to_number(df.loc[df["MRP"] > 25.0, "Invoice qty. pieces"]).sum())
    df = df[df["MRP"] <= 25.0].copy()

    filtered_rows = int(len(df))
    df["_line_key"] = build_line_key(df)
    df["_scheme_reason_group"] = df["Scheme_Reason"].map(scheme_reason_group)

    for col in [
        "Invoice qty. pieces",
        "MRP",
        "Selling_Price",
        "gst",
        "Scheme_discount",
        "%_Scheme",
        "Sale_Value",
        "Distributor_Sale_Value",
    ]:
        df[col] = to_number(df[col])

    raw_qty_sum = float(df["Invoice qty. pieces"].sum())

    df = df.sort_values(["source_file", "source_row_number"], kind="mergesort")
    first_values = df.drop_duplicates("_line_key", keep="first")[["_line_key", *FIRST_VALUE_COLS, "Sizes"]].copy()

    scheme_pivot = (
        df.pivot_table(
            index="_line_key",
            columns="_scheme_reason_group",
            values=["%_Scheme", "Scheme_discount"],
            aggfunc="sum",
            fill_value=0.0,
        )
        .reset_index()
    )
    scheme_pivot.columns = [
        "_line_key"
        if col[0] == "_line_key"
        else f"{str(col[1]).lower()}_{str(col[0]).replace('%_', '').lower()}"
        for col in scheme_pivot.columns.to_flat_index()
    ]

    for col in [
        "claimable_scheme",
        "non_claimable_scheme",
        "other_scheme",
        "claimable_scheme_discount",
        "non_claimable_scheme_discount",
        "other_scheme_discount",
    ]:
        if col not in scheme_pivot.columns:
            scheme_pivot[col] = 0.0

    line_row_counts = df.groupby("_line_key").size().rename("raw_scheme_rows").reset_index()

    lines = first_values.merge(scheme_pivot, on="_line_key", how="left")
    lines = lines.merge(line_row_counts, on="_line_key", how="left")

    qty = to_number(lines["Invoice qty. pieces"])
    mrp = to_number(lines["MRP"])
    gst = to_number(lines["gst"])
    selling_price = to_number(lines["Selling_Price"])

    clp = mrp / 1.12 / 1.07 / (1.0 + (gst / 100.0))
    distributor_revenue = selling_price * qty
    company_revenue = clp * qty

    lines["Claimable_Scheme_Pct"] = to_number(lines["claimable_scheme"])
    lines["Non_Claimable_Scheme_Pct"] = to_number(lines["non_claimable_scheme"])
    lines["Other_Scheme_Pct"] = to_number(lines["other_scheme"])
    lines["Total_Scheme_Pct"] = (
        lines["Claimable_Scheme_Pct"]
        + lines["Non_Claimable_Scheme_Pct"]
        + lines["Other_Scheme_Pct"]
    )

    lines["Claimable_Scheme_Discount"] = to_number(lines["claimable_scheme_discount"])
    lines["Non_Claimable_Scheme_Discount"] = to_number(lines["non_claimable_scheme_discount"])
    lines["Other_Scheme_Discount"] = to_number(lines["other_scheme_discount"])

    lines["Scheme_Discount"] = lines["Claimable_Scheme_Discount"]
    lines["Staggered_qps"] = lines["Non_Claimable_Scheme_Discount"] + lines["Other_Scheme_Discount"]
    lines["TotalDiscount"] = lines["Scheme_Discount"] + lines["Staggered_qps"]

    lines["Date"] = pd.to_datetime(lines["Invoice Date"], errors="coerce")
    lines["Outlet_ID"] = lines["Outlet_Id"].astype(str)
    lines["Bill_No"] = lines["Bill No"].astype(str)
    lines["Final_State"] = lines["Plant_Name"].map(normalize_state)
    lines["State"] = lines["Final_State"]
    lines["Final_Outlet_Classification"] = lines["Outlet_Classification"].map(OUTLET_CLASSIFICATION_MAPPING).fillna("OtherGT")
    lines["Brand"] = "Streax"
    lines["Quantity"] = qty
    lines["Selling_Rate_Per_PC_without_GST_CLP"] = clp
    lines["Net_Amt"] = company_revenue
    lines["SalesValue_atBasicRate"] = distributor_revenue
    lines["Basic_Rate_Per_PC_without_GST"] = selling_price
    lines["Basic_Rate_Per_PC"] = selling_price
    lines["Sku_Code"] = lines["skucode"].astype(str)
    lines["Sku_Name"] = lines["SKU Name"].astype(str)
    lines["Transaction_Key"] = lines["_line_key"]

    output_cols = [
        "Transaction_Key",
        "Date",
        "Month",
        "source_file",
        "source_row_number",
        "Plant_Name",
        "Final_State",
        "State",
        "Distributor_Type",
        "Outlet_ID",
        "Outlet_Name",
        "Bill_No",
        "Batch_Id",
        "Batch No",
        "Category",
        "Subcategory",
        "Brand",
        "SUb_Brand",
        "Sizes",
        "Sku_Code",
        "Sku_Name",
        "MRP",
        "gst",
        "Selling_Price",
        "Basic_Rate_Per_PC_without_GST",
        "Basic_Rate_Per_PC",
        "Selling_Rate_Per_PC_without_GST_CLP",
        "Quantity",
        "Sale_Value",
        "Distributor_Sale_Value",
        "SalesValue_atBasicRate",
        "Net_Amt",
        "Claimable_Scheme_Pct",
        "Non_Claimable_Scheme_Pct",
        "Other_Scheme_Pct",
        "Total_Scheme_Pct",
        "Claimable_Scheme_Discount",
        "Non_Claimable_Scheme_Discount",
        "Other_Scheme_Discount",
        "Scheme_Discount",
        "Staggered_qps",
        "TotalDiscount",
        "Final_Outlet_Classification",
        "raw_scheme_rows",
    ]
    lines = lines[output_cols].copy()

    report = {
        "raw_rows_before_filters": raw_rows,
        "raw_rows_after_filters": filtered_rows,
        "deduped_sale_lines": int(len(lines)),
        "raw_quantity_sum_after_filters": round(raw_qty_sum, 4),
        "deduped_quantity_sum": round(float(lines["Quantity"].sum()), 4),
        "quantity_overcount_factor": round(raw_qty_sum / float(lines["Quantity"].sum()), 6)
        if float(lines["Quantity"].sum()) else None,
        "raw_scheme_row_distribution": {
            str(k): int(v) for k, v in lines["raw_scheme_rows"].value_counts().sort_index().items()
        },
        "size_distribution": {
            str(k): int(v) for k, v in lines["Sizes"].value_counts(dropna=False).sort_index().items()
        },
        "state_distribution": {
            str(k): int(v) for k, v in lines["Final_State"].value_counts(dropna=False).sort_index().items()
        },
        "stockiest_only": bool(stockiest_only),
        "exclude_samt": bool(exclude_samt),
        "keep_only_12_18": bool(keep_only_12_18),
        "mrp_max_allowed": 25.0,
        "rows_removed_mrp_above_25": invalid_mrp_rows,
        "quantity_removed_mrp_above_25": round(invalid_mrp_quantity, 4),
    }

    return lines, report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build one deduplicated sale-line parquet from raw scheme utilisation parquet files."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument(
        "--include-all-distributor-types",
        action="store_true",
        help="Do not filter to Distributor_Type == STOCKIEST DMS.",
    )
    parser.add_argument(
        "--include-samt",
        action="store_true",
        help="Do not exclude schemes whose Scheme Name contains SAMT.",
    )
    parser.add_argument(
        "--include-other-sizes",
        action="store_true",
        help="Do not filter to derived Sizes 12-ML and 18-ML.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_path = args.output.resolve()
    report_path = args.report.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    df = read_monthly_parquets(input_dir, output_path)
    lines, report = build_deduped_sales_lines(
        df,
        stockiest_only=not args.include_all_distributor_types,
        exclude_samt=not args.include_samt,
        keep_only_12_18=not args.include_other_sizes,
    )

    lines.to_parquet(output_path, index=False, engine="pyarrow", compression="snappy")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"deduped_sales_lines={output_path}")
    print(f"report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
