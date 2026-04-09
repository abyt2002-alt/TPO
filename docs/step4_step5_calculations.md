# Step 4 / Step 5 Calculation Reference

This document explains the calculations used on the Step 4 and Step 5 pages:

- raw source columns
- internal aliases created in Step 2
- baseline and discount decomposition from Step 3
- Step 4 planner formulas
- Step 5 scenario-generator formulas
- what each UI number actually means

## 1. Raw Source Columns Used

The main raw columns used by Step 4 and Step 5 come from the sales data loaded by the backend:

- `Quantity`
- `Scheme_Discount`
- `Staggered_qps`
- `Basic_Rate_Per_PC_without_GST`
- `Basic_Rate_Per_PC`
- `Selling_Rate_Per_PC_without_GST_CLP`
- `Sizes`
- `Date`

Column normalization is handled in:

- `backend/services/core/data_loader.py`
- `backend/services/steps/step2_discount.py`

### Price meaning

- `DSP`:
  - preferred column: `Basic_Rate_Per_PC_without_GST`
  - fallback column: `Basic_Rate_Per_PC`
- `CLP`:
  - column: `Selling_Rate_Per_PC_without_GST_CLP`
  - if missing or non-positive, backend falls back to `DSP`

### Important note on EASP

`EASP` is not used in the current Step 4 / Step 5 planner calculations.

The current planner uses:

- `DSP` for gross revenue and investment
- `CLP` for net revenue
- `COGS per unit` from modeling settings, not from a raw transaction column

## 2. Step 2 Internal Basis

Step 2 creates these internal fields:

- `_step2_scheme_amount = Scheme_Discount + Staggered_qps`
- `_step2_dsp_sales = Quantity * DSP`
- `_step2_clp_sales = Quantity * CLP`

These are created in:

- `backend/services/steps/step2_discount.py`

### Step 2 actual discount %

For a slab-month:

`actual_discount_pct = sum(_step2_scheme_amount) / sum(_step2_dsp_sales) * 100`

So:

- numerator = `Scheme_Discount + Staggered_qps`
- denominator = `Quantity * DSP`

## 3. Step 3 Modeling Decomposition

Step 3 creates the baseline and discount-driven quantity split used later by Step 4 and Step 5.

### 3.1 Monthly derived fields

At slab-month level:

- `total_discount = sum(_step2_scheme_amount)`
- `sales_value = sum(_step2_dsp_sales)`
- `actual_discount_pct = total_discount / sales_value * 100`
- `tactical_discount_pct = actual_discount_pct - base_discount_pct`
- `lag1_base_discount_pct = previous month base discount`
- `base_price = sales_value / quantity`

In practice, `base_price` is the monthly DSP-like unit price used by the planner.

### 3.2 Stage 1 and Stage 2 model idea

The modeling logic is effectively:

`quantity = intercept + residual_store_term + own_discount_term + lag_discount_term + other_slab_discount_term`

More explicitly:

`predicted_qty = stage2_intercept + coef_residual_store * residual_store + coef_base_discount * base_discount_pct + coef_lag * lag1_base_discount_pct + coef_other * other_slabs_weighted_base_discount_pct`

The generic stage-2 prediction helper is in:

- `backend/services/core/shared_math.py`

### 3.3 Non-discount baseline quantity

The non-discount baseline keeps the store residual active, but turns all discount inputs off:

`non_discount_baseline_quantity = stage2(intercept + residual_store term, base=0, tactical=0, lag=0, other=0)`

So conceptually:

`non_discount_baseline_quantity ~= intercept + coef_residual_store * residual_store`

This is built in:

- `backend/services/steps/step3_modeling.py`

## 4. Step 4 Scenario Planner

Step 4 forecasts 3 future months, then computes volume, revenue, profit, margin, investment, and CTS.

## 4.1 Forecast baseline for each slab

Current logic for each slab:

1. Take historical `residual_store`
2. Forecast `residual_store` forward
3. Recompute baseline using the stage-2 model with all discount inputs zero

So forecast baseline is:

`forecast_baseline_qty = stage2_model(forecast_residual_store, base=0, tactical=0, lag=0, other=0)`

This is implemented in:

- `backend/services/steps/step4_cross_size_planner.py`

## 4.2 Forecast discount inputs

Default future discounts come from the last 3 historical `base_discount_pct` values for each slab.

For any scenario entered by the user:

- `scenario_discount_pct` comes from the Step 4 input boxes
- `lag_used_pct` is previous month scenario discount
- `other_weighted_scenario_pct` is weighted average discount of the other slabs

## 4.3 Discount component for each slab-month

For each future slab-month:

`discount_component_scenario_qty = coef_base * scenario_discount_pct + coef_lag * lag_used_pct + coef_other * other_weighted_scenario_pct`

Then:

`pre_cross_qty = non_discount_baseline_qty + discount_component_scenario_qty`

Current planner mode is additive only, so:

`final_qty = pre_cross_qty`

No extra cross-pack volume redistribution is applied beyond the planner's own slab-month share logic.

## 4.4 Step 4 revenue, profit, investment

For each slab-month:

- `baseline_revenue_gross = baseline_qty * DSP`
- `scenario_revenue_gross = final_qty * DSP`
- `baseline_revenue_net = baseline_qty * CLP`
- `scenario_revenue_net = final_qty * CLP`
- `baseline_profit = baseline_revenue_net - (baseline_qty * COGS_per_unit)`
- `scenario_profit = scenario_revenue_net - (final_qty * COGS_per_unit)`
- `baseline_investment = baseline_qty * DSP * default_discount_pct / 100`
- `scenario_investment = final_qty * DSP * scenario_discount_pct / 100`

These slab-month values are then summed:

- by size: `12-ML`, `18-ML`
- then again into `TOTAL`

## 4.5 Step 4 page KPI formulas

### Volume

For size cards:

`Volume = scenario final_qty`

For TOTAL card:

`Volume Units = total final_qty of 12-ML + 18-ML`

Note:

- the TOTAL card label now says `Volume Units`
- the displayed value is still total unit quantity, not ML

The planner also computes ML-based totals separately:

- `final_volume_ml = final_qty_12 * 12 + final_qty_18 * 18`

but that ML total is not the big Step 4 TOTAL card number.

### Gross Revenue

`Gross Revenue = scenario_revenue_gross = final_qty * DSP`

### Net Revenue

`Net Revenue = scenario_revenue_net = final_qty * CLP`

### Net Margin %

Current Step 4 UI formula:

`Net Margin % = scenario_profit / scenario_revenue_net * 100`

### Investment

`Investment = scenario_investment = final_qty * DSP * scenario_discount_pct / 100`

### CTS %

Current Step 4 UI formula:

`CTS % = scenario_investment / scenario_revenue_gross * 100`

## 4.6 Step 4 comparison percentages

The small green/red number under each KPI is a comparison against the selected reference mode.

### Y-o-Y

If forecast months are:

- `2026-04`
- `2026-05`
- `2026-06`

then Y-o-Y reference is:

- `2025-04`
- `2025-05`
- `2025-06`

### Q-o-Q

For the same forecast months, Q-o-Q reference is:

- `2026-01`
- `2026-02`
- `2026-03`

### Comparison formulas

- `Volume % = (scenario_qty - reference_qty) / reference_qty * 100`
- `Gross Revenue % = (scenario_revenue_gross - reference_revenue_gross) / reference_revenue_gross * 100`
- `Net Revenue % = (scenario_revenue_net - reference_revenue_net) / reference_revenue_net * 100`
- `Net Margin delta = current_net_margin_pct - reference_net_margin_pct`
- `Investment % = (scenario_investment - reference_investment) / reference_investment * 100`
- `CTS delta = current_cts_pct - reference_cts_pct`

Important:

- the big Step 4 KPI value usually stays the same when switching `Y-o-Y` / `Q-o-Q`
- only the small comparison number should change

## 4.7 Step 4 lower chart

The lower Step 4 chart currently behaves like this:

### Historical months

- top line = actual monthly total quantity from history
- displayed baseline = `min(modeled baseline, actual total)`
- displayed discount component = `max(actual total - displayed baseline, 0)`

### Forecast months

- baseline = Step 4 planner `baseline_total_qty`
- total = Step 4 planner `final_total_qty`
- discount component = `final_total_qty - baseline_total_qty`

So the chart uses:

`Total = Baseline + Discount Component`

## 5. Step 5 Scenario Generator

Step 5 uses Step 4 planner logic to compare multiple scenario ladders.

Each scenario is recomputed with:

- same Step 4 planner base
- same slab coefficients
- same revenue / profit / investment formulas
- same selected reference mode (`Y-o-Y` or `Q-o-Q`)

Frontend scenario recompute happens via:

- `frontend/src/utils/crossSizePlannerCompute.js`
- `frontend/src/components/rfm/ScenarioComparison.jsx`
- `frontend/src/pages/RFMAnalysis.jsx`

## 5.1 Step 5 scenario metrics

For each scenario summary:

- `volume = final_qty`
- `revenue = scenario_revenue_gross`
- `profit = scenario_profit`
- `investment = scenario_investment`

For TOTAL, Step 5 also carries:

- `volume_ml = final_volume_ml`

## 5.2 Step 5 net margin %

Current Step 5 UI uses net margin, not gross margin.

Formula:

`Net Margin % = (scenario_profit / scenario_revenue_net * 100) - (reference_profit / reference_revenue_net * 100)`

This net-margin basis is used in:

- filter threshold
- sorting
- scenario comparison chart
- modal summary cards
- saved scenario workbook export

## 5.3 Step 5 volume %

For size-level:

`volume_pct = vs_reference_volume_pct`

For TOTAL chart and ranking:

`volume_pct = vs_reference_volume_ml_pct`

So the Step 5 TOTAL volume comparison is ML-based.

## 5.4 Step 5 revenue %

`revenue_pct = (scenario_revenue_gross - reference_revenue_gross) / reference_revenue_gross * 100`

## 5.5 Step 5 investment %

`investment_pct = (scenario_investment - reference_investment) / reference_investment * 100`

## 5.6 Step 5 CTS %

`CTS % = scenario_investment / scenario_revenue_gross * 100`

## 5.7 Step 5 chart meaning

The Step 5 grouped bar chart is not showing absolute values.

It shows comparison percentages:

- blue = `Volume %`
- green = `Revenue %`
- orange = `Net Margin %`

So for example:

- `+33% Volume`
- `+33% Revenue`
- `+0.72% Net Margin`

means:

- volume is 33% above the selected reference
- revenue is 33% above the selected reference
- net margin is 0.72 percentage points above the selected reference net margin

## 6. Worked Example

Assume one slab-month forecast:

- `final_qty = 100,000`
- `baseline_qty = 30,000`
- `DSP = 12`
- `CLP = 11`
- `COGS = 6`
- `scenario_discount_pct = 20`

Then:

- `discount_component_qty = 100,000 - 30,000 = 70,000`
- `Gross Revenue = 100,000 * 12 = 1,200,000`
- `Net Revenue = 100,000 * 11 = 1,100,000`
- `Profit = 1,100,000 - (100,000 * 6) = 500,000`
- `Net Margin % = 500,000 / 1,100,000 * 100 = 45.45%`
- `Investment = 100,000 * 12 * 20% = 240,000`
- `CTS % = 240,000 / 1,200,000 * 100 = 20.00%`

If reference net margin was `44.10%`, then:

- `Net Margin delta = 45.45% - 44.10% = +1.35%`

## 7. Code References

Main files behind these calculations:

- `backend/services/steps/step2_discount.py`
- `backend/services/steps/step3_modeling.py`
- `backend/services/core/shared_math.py`
- `backend/services/steps/step4_cross_size_planner.py`
- `backend/services/steps/step5_baseline_forecast.py`
- `frontend/src/components/rfm/CrossSizePlanner.jsx`
- `frontend/src/components/rfm/ScenarioComparison.jsx`
- `frontend/src/utils/crossSizePlannerCompute.js`
- `frontend/src/pages/RFMAnalysis.jsx`
