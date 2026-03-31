#!/usr/bin/env python3
"""
Validate monthly 12-ML/18-ML parity between raw Excel and existing parquet files
for a selected numeric value column (default: Quantity), after identical filters.

Filters:
- State in configured states (default: MAH,UP)
- Outlet_Type == GT
- Quantity > 0
- Subcategory in {STX INSTA SHAMPOO, STREAX INSTA SHAMPOO}
- Sizes in {12-ML, 18-ML}

Exit codes:
- 0: full match within tolerance
- 1: mismatch or validation/schema error
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import re
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow.parquet as pq


ALLOWED_SUBCATEGORIES = {"STX INSTA SHAMPOO", "STREAX INSTA SHAMPOO"}
ALLOWED_SIZES = {"12-ML", "18-ML"}
BASE_REQUIRED_COLUMNS = ("Date", "State", "Outlet_Type", "Subcategory", "Sizes", "Quantity")
ALIASES = {
    "date": "Date",
    "state": "State",
    "outlet_type": "Outlet_Type",
    "subcategory": "Subcategory",
    "sizes": "Sizes",
    "quantity": "Quantity",
    "taxable_amount": "Taxable_Amount",
    "salesvalue_atbasicrate": "SalesValue_atBasicRate",
}
STATE_CANONICAL_MAP = {
    "MAHARASHTRA": "MAH",
    "MAHARASHTRA STATE": "MAH",
    "MH": "MAH",
    "MAH": "MAH",
    "UTTARPRADESH": "UP",
    "UTTAR PRADESH": "UP",
    "U P": "UP",
    "UP": "UP",
}
MAH_STATE_TOKENS = ("MAHARASHTRA", "MUMBAI", "VIDARBHA")
UP_STATE_TOKENS = ("U.P", "UP", "UTTAR PRADESH")


def _norm_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).strip().lower()).strip("_")


def _normalize_size_value(raw_size: pd.Series) -> pd.Series:
    clean = (
        raw_size.astype(str)
        .str.upper()
        .str.replace(" ", "", regex=False)
        .str.strip()
    )
    return clean.replace(
        {
            "12": "12-ML",
            "12ML": "12-ML",
            "12-ML": "12-ML",
            "18": "18-ML",
            "18ML": "18-ML",
            "18-ML": "18-ML",
        }
    )


def _normalize_state_value(raw_state: pd.Series) -> pd.Series:
    clean = (
        raw_state.astype(str)
        .str.upper()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    canonical = clean.replace({k: v for k, v in STATE_CANONICAL_MAP.items()})
    mah_mask = canonical.str.contains("|".join(map(re.escape, MAH_STATE_TOKENS)), regex=True, na=False)
    up_mask = canonical.str.contains("|".join(map(re.escape, UP_STATE_TOKENS)), regex=True, na=False)
    canonical = canonical.where(~mah_mask, "MAH")
    canonical = canonical.where(~up_mask, "UP")
    return canonical


def _build_canonical_to_source(columns: Iterable[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for src in columns:
        key = _norm_key(src)
        canonical = ALIASES.get(key)
        if canonical and canonical not in mapping:
            mapping[canonical] = src
    return mapping


def _read_excel_required_columns(source_file: Path, required_columns: tuple[str, ...]) -> pd.DataFrame:
    header_only = pd.read_excel(source_file, nrows=0)
    canonical_to_source = _build_canonical_to_source(list(header_only.columns))
    missing = [c for c in required_columns if c not in canonical_to_source]
    if missing:
        raise ValueError(f"Missing required columns in {source_file.name}: {', '.join(missing)}")
    source_cols = [canonical_to_source[c] for c in required_columns]
    df = pd.read_excel(source_file, usecols=source_cols)
    rename_map = {canonical_to_source[c]: c for c in required_columns}
    return df.rename(columns=rename_map)


def _read_parquet_required_columns(source_file: Path, required_columns: tuple[str, ...]) -> pd.DataFrame:
    schema_names = pq.ParquetFile(source_file).schema.names
    canonical_to_source = _build_canonical_to_source(schema_names)
    missing = [c for c in required_columns if c not in canonical_to_source]
    if missing:
        raise ValueError(f"Missing required columns in {source_file.name}: {', '.join(missing)}")
    source_cols = [canonical_to_source[c] for c in required_columns]
    df = pd.read_parquet(source_file, columns=source_cols)
    rename_map = {canonical_to_source[c]: c for c in required_columns}
    return df.rename(columns=rename_map)


def _apply_common_filters(df: pd.DataFrame, states: set[str], value_column: str) -> pd.DataFrame:
    work = df.copy()
    work["Date"] = pd.to_datetime(work["Date"], errors="coerce")
    work["Quantity"] = pd.to_numeric(work["Quantity"], errors="coerce")
    work[value_column] = pd.to_numeric(work[value_column], errors="coerce")

    state = _normalize_state_value(work["State"])
    outlet_type = (
        work["Outlet_Type"]
        .astype(str)
        .str.upper()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    subcategory = (
        work["Subcategory"]
        .astype(str)
        .str.upper()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    sizes = _normalize_size_value(work["Sizes"])

    mask = (
        work["Date"].notna()
        & work["Quantity"].notna()
        & (work["Quantity"] > 0)
        & work[value_column].notna()
        & state.isin(states)
        & outlet_type.eq("GT")
        & subcategory.isin(ALLOWED_SUBCATEGORIES)
        & sizes.isin(ALLOWED_SIZES)
    )

    filtered = work.loc[mask, ["Date", "Sizes", value_column]].copy()
    filtered["month"] = filtered["Date"].dt.to_period("M").astype(str)
    filtered["size"] = _normalize_size_value(filtered["Sizes"])
    filtered["value"] = pd.to_numeric(filtered[value_column], errors="coerce").fillna(0.0)
    return filtered


def _aggregate_month_size(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["month", "size", "value"])
    out = (
        df.groupby(["month", "size"], as_index=False)
        .agg(value=("value", "sum"))
        .sort_values(["month", "size"])
        .reset_index(drop=True)
    )
    out["value"] = pd.to_numeric(out["value"], errors="coerce").fillna(0.0)
    return out


def _process_single_file(
    file_path: str,
    states_sorted: tuple[str, ...],
    value_column: str,
    required_columns: tuple[str, ...],
) -> tuple[pd.DataFrame, int, int]:
    fp = Path(file_path)
    states = set(states_sorted)
    if fp.suffix.lower() == ".xlsx":
        df = _read_excel_required_columns(fp, required_columns)
    elif fp.suffix.lower() == ".parquet":
        df = _read_parquet_required_columns(fp, required_columns)
    else:
        raise ValueError(f"Unsupported file extension: {fp.name}")

    rows_loaded = len(df)
    filtered = _apply_common_filters(df, states, value_column)
    rows_after_filters = len(filtered)
    agg = _aggregate_month_size(filtered)
    return agg, rows_loaded, rows_after_filters


def _load_and_filter_files(
    folder: Path,
    pattern: str,
    states: set[str],
    label: str,
    workers: int,
    value_column: str,
    required_columns: tuple[str, ...],
) -> tuple[pd.DataFrame, dict]:
    files = sorted(folder.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found in {folder} matching {pattern}")

    frames: list[pd.DataFrame] = []
    stats = {
        "files": len(files),
        "rows_loaded": 0,
        "rows_after_filters": 0,
    }

    worker_count = workers if workers > 0 else max(1, min(8, (os.cpu_count() or 2), len(files)))
    states_sorted = tuple(sorted(states))
    print(f"[{label}] workers={worker_count} | files={len(files)}")

    with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as pool:
        future_map = {
            pool.submit(
                _process_single_file,
                str(fp),
                states_sorted,
                value_column,
                required_columns,
            ): fp.name
            for fp in files
        }
        completed = 0
        for future in concurrent.futures.as_completed(future_map):
            fname = future_map[future]
            completed += 1
            try:
                agg_df, rows_loaded, rows_after = future.result()
            except Exception as exc:
                raise RuntimeError(f"{label} file failed: {fname} | {exc}") from exc
            stats["rows_loaded"] += rows_loaded
            stats["rows_after_filters"] += rows_after
            frames.append(agg_df)
            print(f"[{label}] completed {completed}/{len(files)}: {fname}")

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        combined = (
            combined.groupby(["month", "size"], as_index=False)
            .agg(value=("value", "sum"))
            .sort_values(["month", "size"])
            .reset_index(drop=True)
        )
    else:
        combined = pd.DataFrame(columns=["month", "size", "value"])

    print(
        f"[{label}] files={stats['files']} | rows_loaded={stats['rows_loaded']:,} | "
        f"rows_after_filters={stats['rows_after_filters']:,}"
    )
    return combined, stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate monthly raw-vs-parquet match for 12-ML and 18-ML."
    )
    project_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--raw-folder", type=Path, default=project_root / "Updated HRI DATA")
    parser.add_argument("--parquet-folder", type=Path, default=project_root / "DATA")
    parser.add_argument("--states", type=str, default="MAH,UP")
    parser.add_argument("--tolerance", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--show-month-tests", action="store_true")
    parser.add_argument("--value-column", type=str, default="Quantity")
    args = parser.parse_args()

    raw_folder = args.raw_folder.resolve()
    parquet_folder = args.parquet_folder.resolve()
    if not raw_folder.exists():
        print(f"ERROR: Raw folder not found: {raw_folder}")
        return 1
    if not parquet_folder.exists():
        print(f"ERROR: Parquet folder not found: {parquet_folder}")
        return 1

    states = {s.strip().upper() for s in (args.states or "").split(",") if s.strip()}
    if not states:
        print("ERROR: states set is empty.")
        return 1

    requested_value = str(args.value_column).strip()
    if not requested_value:
        print("ERROR: value column is empty.")
        return 1
    value_column = ALIASES.get(_norm_key(requested_value), requested_value)
    required_columns = tuple(dict.fromkeys(BASE_REQUIRED_COLUMNS + (value_column,)))

    print("=== Raw vs Parquet Monthly Validation ===")
    print(f"Raw folder: {raw_folder}")
    print(f"Parquet folder: {parquet_folder}")
    print(f"States: {sorted(states)}")
    print(f"Value column: {value_column}")
    print(f"Tolerance: {args.tolerance}")
    print(f"Workers: {args.workers if args.workers > 0 else 'auto'}")
    print()

    try:
        raw_agg_by_file, _ = _load_and_filter_files(
            raw_folder, "*.xlsx", states, "RAW", args.workers, value_column, required_columns
        )
        pq_agg_by_file, _ = _load_and_filter_files(
            parquet_folder, "*.parquet", states, "PARQUET", args.workers, value_column, required_columns
        )
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1

    raw_agg = raw_agg_by_file.rename(columns={"value": "raw_value"})
    pq_agg = pq_agg_by_file.rename(columns={"value": "parquet_value"})

    merged = raw_agg.merge(pq_agg, on=["month", "size"], how="outer")
    merged["raw_value"] = pd.to_numeric(merged["raw_value"], errors="coerce")
    merged["parquet_value"] = pd.to_numeric(merged["parquet_value"], errors="coerce")
    merged["diff_value"] = merged["parquet_value"].fillna(0.0) - merged["raw_value"].fillna(0.0)
    merged["match"] = (
        merged["raw_value"].notna()
        & merged["parquet_value"].notna()
        & (merged["diff_value"].abs() <= float(args.tolerance))
    )

    size_order = {"12-ML": 0, "18-ML": 1}
    merged["_size_sort"] = merged["size"].map(lambda x: size_order.get(str(x), 99))
    merged = merged.sort_values(["month", "_size_sort", "size"]).drop(columns=["_size_sort"])

    total_rows = len(merged)
    matched_rows = int(merged["match"].sum()) if total_rows > 0 else 0
    mismatch_df = merged[~merged["match"]].copy()

    print("\n=== Coverage Summary ===")
    print(f"Raw month-size rows: {len(raw_agg)}")
    print(f"Parquet month-size rows: {len(pq_agg)}")
    print(f"Compared month-size rows: {total_rows}")
    raw_months = sorted(raw_agg["month"].unique().tolist()) if not raw_agg.empty else []
    pq_months = sorted(pq_agg["month"].unique().tolist()) if not pq_agg.empty else []
    print(f"Raw month range: {raw_months[0]} to {raw_months[-1]}" if raw_months else "Raw month range: NA")
    print(f"Parquet month range: {pq_months[0]} to {pq_months[-1]}" if pq_months else "Parquet month range: NA")
    print(f"Raw sizes present: {sorted(raw_agg['size'].dropna().unique().tolist()) if not raw_agg.empty else []}")
    print(f"Parquet sizes present: {sorted(pq_agg['size'].dropna().unique().tolist()) if not pq_agg.empty else []}")

    print("\n=== Match Summary ===")
    print(f"Matched rows: {matched_rows}")
    print(f"Mismatched rows: {len(mismatch_df)}")

    month_tests = (
        merged.assign(row_pass=merged["match"])
        .groupby("month", as_index=False)
        .agg(
            test_rows=("size", "count"),
            passed_rows=("row_pass", "sum"),
            failed_rows=("row_pass", lambda s: int((~s).sum())),
        )
        .sort_values("month")
    )
    month_tests["status"] = month_tests["failed_rows"].map(lambda x: "PASS" if x == 0 else "FAIL")
    print("\n=== Monthly Independent Tests ===")
    print(f"Total month tests: {len(month_tests)}")
    print(f"Passed month tests: {int((month_tests['status'] == 'PASS').sum())}")
    print(f"Failed month tests: {int((month_tests['status'] == 'FAIL').sum())}")
    if args.show_month_tests:
        print(month_tests.to_string(index=False))

    if len(mismatch_df) > 0:
        show = mismatch_df[["month", "size", "raw_value", "parquet_value", "diff_value"]].copy()
        for col in ["raw_value", "parquet_value", "diff_value"]:
            show[col] = pd.to_numeric(show[col], errors="coerce").round(6)
        print("\n=== Mismatch Details ===")
        print(show.to_string(index=False))
        return 1

    print(f"\nPASS: Raw and parquet monthly {value_column} sums match for all month-size combinations.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
