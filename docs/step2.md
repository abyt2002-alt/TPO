# Step 2 Documentation (Discount Analysis / Base Depth)

This document describes how Step 2 is computed in the current React + FastAPI app.

## Entry Points

- Backend API:
  - `POST /api/discount/options`
  - `POST /api/discount/base-depth`
- Main backend functions:
  - `backend/services/core/scope_builder.py:_build_step2_scope`
  - `backend/services/steps/step2_discount.py:_apply_step2_slab_definition`
  - `backend/services/steps/step2_discount.py:_assign_defined_step2_slabs`
  - `backend/services/steps/step2_discount.py:_compute_base_depth_result`
  - `backend/services/steps/step2_discount.py:calculate_base_depth`

## 1) Scope Build

Step 2 always starts from Step 1 dataset/cache and then applies Step 2 filters:

1. Filter RFM rows by selected `rfm_segments` and optional `outlet_ids`.
2. Convert outlet classification to Step 2 business groups:
   - `WH`
   - `OtherGT` (all non-WH, including SS and OtherGT)
3. Intersect outlet set and build `df_scope`.
4. Apply slab definition to both:
   - `df_scope` (active slab-filtered data)
   - `df_scope_all_slabs` (for summary tables and cross-slab logic)
5. Keep only allowed slabs (`slab1+`) for Step 2 charts; apply user slab selection if passed.

## 2) Slab Definition Modes

Step 2 supports two slab definition modes.

### Mode A: `data`

- Uses slab already present in data column `Slab`.
- Base-depth estimator follows Data/TDP daily logic.

### Mode B: `define`

- Slab is re-assigned by backend using monthly outlet quantity.
- Group key is:
  - outlet + month + size
- Quantity is summed at that level and binned using thresholds.
- Supports per-size profiles through `defined_slab_profiles`.
- Example bins:
  - thresholds `[8, 32, 576, 960]` -> `slab0..slab4`.

## 3) Base Discount Computation

`calculate_base_depth` calls `_compute_base_depth_result` on scope data.

### 3.1 Actual Discount

At aggregated grain:

- `actual_discount_pct = (total_discount / sales_value) * 100`

### 3.2 Base Discount Logic

#### In `data` mode

1. Build daily series.
2. Estimate daily base blocks using min step-up and min step-down rules.
3. Apply requested output aggregation:
   - daily / weekly / monthly
4. Carry base value to output periods.

#### In `define` mode

1. Force monthly series.
2. Monthly discount is recomputed from monthly sums.
3. Base is estimated at monthly level only.
4. Base is rounded to `0.5` step.

### 3.3 Tactical Discount

- `tactical_discount_pct = max(actual_discount_pct - base_discount_pct, 0)`

## 4) Step 2 Outputs

`BaseDepthResponse` includes:

- `points`:
  - period, actual discount, base discount, orders, quantity, sales value
- `summary`:
  - avg/max actual/base/tactical, period count
- `slab_results`:
  - per-slab base-depth outputs
- `summary_by_slab`:
  - slab-level business summary table used in UI

## 5) Downstream Dependency

Step 3 modeling depends on Step 2 output schema and slab assignment:

- same scope filters
- same slab definition mode
- same size and slab selections
