from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from pandas.api.types import is_object_dtype


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "final_presentation" / "Raw files"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"


def safe_stem(name: str) -> str:
    text = Path(name).stem.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "scheme_utilisation"


def read_first_sheet(path: Path) -> tuple[pd.DataFrame, str]:
    excel = pd.ExcelFile(path, engine="openpyxl")
    sheet_name = excel.sheet_names[0]
    df = pd.read_excel(excel, sheet_name=sheet_name)
    df.columns = [str(col).strip() for col in df.columns]
    return df, sheet_name


def normalize_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """Avoid pyarrow failures from mixed object columns in exported Excel data."""
    out = df.copy()
    for col in out.columns:
        if is_object_dtype(out[col]):
            out[col] = out[col].map(lambda value: None if pd.isna(value) else str(value))
    return out


def convert_file(path: Path, output_dir: Path) -> dict:
    df, sheet_name = read_first_sheet(path)
    df.insert(0, "source_file", path.name)
    df.insert(1, "source_sheet", sheet_name)
    df.insert(2, "source_row_number", range(2, len(df) + 2))
    df = normalize_for_parquet(df)

    output_path = output_dir / f"{safe_stem(path.name)}.parquet"
    try:
        df.to_parquet(output_path, index=False, engine="pyarrow", compression="snappy")
    except Exception:
        output_path.unlink(missing_ok=True)
        raise

    return {
        "source_file": path.name,
        "source_sheet": sheet_name,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "output_file": output_path.name,
        "status": "converted",
    }


def summarize_existing_parquet(source_path: Path, output_path: Path) -> dict:
    parquet_file = pq.ParquetFile(output_path)
    source_sheet = ""
    if "source_sheet" in parquet_file.schema_arrow.names and parquet_file.metadata.num_row_groups:
        table = parquet_file.read_row_group(0, columns=["source_sheet"])
        if table.num_rows:
            source_sheet = str(table.column("source_sheet")[0].as_py() or "")
    return {
        "source_file": source_path.name,
        "source_sheet": source_sheet,
        "rows": int(parquet_file.metadata.num_rows),
        "columns": int(len(parquet_file.schema_arrow.names)),
        "output_file": output_path.name,
        "status": "skipped_existing",
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert raw monthly Scheme Utilization Excel files to parquet."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help=f"Folder containing raw .xlsx files. Default: {DEFAULT_RAW_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Folder where parquet files will be written. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing parquet files.",
    )
    args = parser.parse_args()

    raw_dir = args.raw_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw folder not found: {raw_dir}")

    files = sorted(p for p in raw_dir.glob("*.xlsx") if not p.name.startswith("~$"))
    if not files:
        raise FileNotFoundError(f"No .xlsx files found in: {raw_dir}")

    manifest = []
    for path in files:
        output_path = output_dir / f"{safe_stem(path.name)}.parquet"
        if output_path.exists() and not args.overwrite:
            print(f"SKIP existing: {output_path.name}", flush=True)
            manifest.append(summarize_existing_parquet(path, output_path))
            continue

        print(f"Converting: {path.name}", flush=True)
        manifest.append(convert_file(path, output_dir))

    if manifest:
        manifest_df = pd.DataFrame(manifest)
        manifest_df.to_csv(output_dir / "conversion_manifest.csv", index=False)

    print(f"Done. Converted {len(manifest)} file(s) into {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
