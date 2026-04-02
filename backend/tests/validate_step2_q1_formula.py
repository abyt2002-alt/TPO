"""Validate Step 2 actual discount % against Q1 formula on current backend data.

Q1 formula:
actual_discount_pct = sum(Scheme_Discount + Staggered_qps) / sum(Quantity * DSP) * 100
where DSP = Basic_Rate_Per_PC_without_GST fallback Basic_Rate_Per_PC.
"""

import argparse
import asyncio
import sys
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Validate Step 2 Q1 discount formula")
    parser.add_argument("--size", default="12-ML", help="Pack size filter (e.g., 12-ML, 18-ML)")
    parser.add_argument("--slab", default="slab1", help="Slab filter (e.g., slab1)")
    parser.add_argument("--tolerance", type=float, default=1e-6, help="Max allowed abs diff")
    return parser.parse_args()


async def main():
    args = parse_args()
    sys.path.append(str(Path(__file__).resolve().parents[1]))

    from services.rfm_service import RFMService
    from models.rfm_models import BaseDepthRequest

    svc = RFMService()
    req = BaseDepthRequest(time_aggregation="M", sizes=[args.size], slabs=[args.slab])
    resp = await svc.calculate_base_depth(req)

    if not resp.success:
        print("FAIL: Step 2 API returned error:", resp.message)
        raise SystemExit(1)

    scope = svc._build_step2_scope(req)
    if scope is None or scope.get("df_scope") is None or scope["df_scope"].empty:
        print("FAIL: No scope rows after filters.")
        raise SystemExit(1)

    work, missing = svc._prepare_step2_discount_basis(scope["df_scope"])
    if missing:
        print("FAIL: Missing required columns:", ", ".join(missing))
        raise SystemExit(1)

    work["Period"] = pd.to_datetime(work["Date"], errors="coerce").dt.to_period("M").dt.start_time
    manual = (
        work.groupby("Period", as_index=False)
        .agg(scheme_amt=("_step2_scheme_amount", "sum"), dsp_sales=("_step2_dsp_sales", "sum"))
        .sort_values("Period")
    )
    manual["manual_pct"] = (
        (manual["scheme_amt"] / manual["dsp_sales"]) * 100.0
    ).replace([float("inf"), float("-inf")], 0.0).fillna(0.0)

    api = pd.DataFrame(
        [{"Period": pd.to_datetime(p.period), "api_pct": float(p.actual_discount_pct)} for p in resp.points]
    ).sort_values("Period")

    merged = api.merge(manual[["Period", "manual_pct"]], on="Period", how="inner")
    if merged.empty:
        print("FAIL: No overlapping periods between API and manual calculation.")
        raise SystemExit(1)

    merged["abs_diff"] = (merged["api_pct"] - merged["manual_pct"]).abs()
    max_diff = float(merged["abs_diff"].max())
    print(f"Rows checked: {len(merged)} | Max abs diff: {max_diff:.10f}")
    print(merged.tail(5).to_string(index=False))

    if max_diff > float(args.tolerance):
        print(f"FAIL: Max diff {max_diff:.10f} exceeds tolerance {args.tolerance:.10f}")
        raise SystemExit(1)

    print("PASS: Step 2 actual_discount_pct matches Q1 formula within tolerance.")
    raise SystemExit(0)


if __name__ == "__main__":
    asyncio.run(main())

