# Step 3 Documentation (Modeling and ROI)

This document describes current Step 3 modeling and ROI flow.

## Entry Point

- Backend API:
  - `POST /api/discount/modeling`
- Main backend functions:
  - `backend/services/steps/step3_modeling.py:calculate_modeling`
  - `backend/services/steps/step3_modeling.py:_build_monthly_model_dataframe_new_strategy`
  - `backend/services/steps/step3_modeling.py:_run_two_stage_model_new_strategy`
  - `backend/services/steps/step3_modeling.py:_build_structural_roi_points`

## 1) Scope and Grain

Step 3 reuses Step 2 scope and slab assignment.

Model is run per:

- size (`12-ML`, `18-ML`)
- slab (`slab1+`, based on active config)

Each slab model is monthly.

## 2) Monthly Feature Frame

For each slab:

1. Build monthly aggregates:
   - store count
   - quantity
   - sales value
   - total discount
2. Compute discounts:
   - `actual_discount_pct`
   - `base_discount_pct` (from Step 2 estimation rules)
   - `lag1_base_discount_pct`
3. Compute:
   - `residual_store` through Stage 1
4. Compute cross-slab feature:
   - `other_slabs_weighted_base_discount_pct`
   - weighted by fixed slab quantity shares (within same size, excluding own slab)

## 3) Two-Stage Model

## Stage 1

- Model:
  - `store_count ~ actual_discount_pct`
- Residual used in Stage 2:
  - `residual_store = actual_store_count - predicted_store_count`

## Stage 2 (New Strategy)

- Target:
  - monthly slab quantity
- Features:
  - `residual_store`
  - `base_discount_pct`
  - `lag1_base_discount_pct` (if enabled)
  - `other_slabs_weighted_base_discount_pct`

Fitted with constrained ridge (`CustomConstrainedRidge`), with defaults:

- residual coefficient >= 0
- base discount coefficient >= 0
- lag coefficient <= 0 (if lag used)
- other-slabs weighted coefficient <= 0

Also fits OLS reference for comparison.

If auto-tune is enabled, L2 is selected from candidate list.

## 4) Baseline/Prediction Series

For each slab/month:

- `predicted_quantity` from actual discount conditions
- `baseline_quantity` from base discount (tactical set to zero)
- `non_discount_baseline_quantity` from zero discount inputs

Also keeps OLS reference variants in output.

## 5) ROI Computation

ROI is built from model outputs and base price/cogs:

- predicted revenue vs baseline revenue
- spend from tactical component
- structural episodes where base increases
- topline ROI and gross margin ROI aggregates

Returned per slab in:

- `roi_points`
- `summary` fields including:
  - structural ROI
  - gross margin ROI
  - total spend
  - episode count

## 6) Response Structure

`ModelingResponse` includes:

- `slab_results[]`:
  - model coefficients
  - predicted vs actual series
  - ROI series and summary
- `combined_summary`:
  - weighted R2 and combined ROI summaries
- `summary_by_slab`:
  - slab table used in expanders/tabs
