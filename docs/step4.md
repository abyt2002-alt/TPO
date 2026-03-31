# Step 4: Cross-Size Scenario Planner

This document explains the current Step 4 logic end to end.

It covers:

- where the baseline comes from
- how the discount component is computed
- how slab quantities become size totals
- how the cross-size readjustment is applied
- how volume, revenue, and profit are produced

Backend entry point:

- [backend/services/steps/step4_cross_size_planner.py](C:/Users/abqua/Desktop/HRI%20app/backend/services/steps/step4_cross_size_planner.py):`calculate_cross_size_planner`

Frontend view:

- [frontend/src/components/rfm/CrossSizePlanner.jsx](C:/Users/abqua/Desktop/HRI%20app/frontend/src/components/rfm/CrossSizePlanner.jsx)

## 1. What Step 4 needs before it can run

Step 4 depends on Step 3 output.

For every active size and slab, Step 4 reuses the fitted Step 3 stage-2 model information:

- structural discount beta
- lag structural discount beta
- other-slabs-weighted discount beta
- latest default discount
- latest anchor quantity
- slab base price
- slab COGS
- slab baseline forecast series

So Step 4 is not an independent model. It is a planner built on top of Step 3 model artifacts.

## 2. Baseline used in Step 4

Each slab gets a 3-month projected **non-discount baseline quantity**.

This is stored in:

- `baseline_slab_matrix[size][slab][month]`

Code:

- [backend/services/steps/step4_cross_size_planner.py](C:/Users/abqua/Desktop/HRI%20app/backend/services/steps/step4_cross_size_planner.py):1320
- [backend/services/steps/step4_cross_size_planner.py](C:/Users/abqua/Desktop/HRI%20app/backend/services/steps/step4_cross_size_planner.py):1334

Important distinction:

- `non_discount_baseline_qty` = projected quantity with no discount component inside it
- this is the clean baseline coming from the baseline forecast

This baseline is the starting quantity for each slab-month in Step 4.

## 3. Discount inputs used in Step 4

For every size, slab, and month, Step 4 builds three discount inputs:

1. `own discount`
2. `lag discount`
3. `other slabs weighted discount`

### 3.1 Own discount

This is the scenario discount for that slab and that month.

Example:

- `12-ML`
- `slab2`
- `Dec`
- user changes `21` to `24`

Then own discount for that slab-month becomes `24`.

### 3.2 Lag discount

Lag is sequential across the 3 forecast months.

Rule:

- Month 1 uses the latest historical default discount as lag
- Month 2 uses Month 1 scenario discount as lag
- Month 3 uses Month 2 scenario discount as lag

Code:

- [backend/services/steps/step4_cross_size_planner.py](C:/Users/abqua/Desktop/HRI%20app/backend/services/steps/step4_cross_size_planner.py):1398
- [backend/services/steps/step4_cross_size_planner.py](C:/Users/abqua/Desktop/HRI%20app/backend/services/steps/step4_cross_size_planner.py):1402

### 3.3 Other-slabs weighted discount

For one slab, Step 4 also computes the weighted average discount of the **other slabs of the same size**.

It excludes the slab itself.

If slab `s` is being evaluated:

- take all other slabs `j != s`
- multiply each slab discount by its stored Step 3 weight
- divide by the sum of weights of those other slabs

This produces:

- `other_weighted_default_pct`
- `other_weighted_scenario_pct`

The weights come from Step 3 modeled weights whenever available.

## 4. Discount component for one slab-month

Once the three discount inputs are ready, Step 4 computes the slab discount component using Step 3 coefficients.

Inputs:

- `beta_base`
- `beta_lag`
- `beta_other`

Default-world discount component:

```text
discount_component_default
= beta_base * default_discount
+ beta_lag * default_lag
+ beta_other * other_default
```

Scenario discount component:

```text
discount_component_scenario
= beta_base * scenario_discount
+ beta_lag * scenario_lag
+ beta_other * other_scenario
```

Code:

- [backend/services/steps/step4_cross_size_planner.py](C:/Users/abqua/Desktop/HRI%20app/backend/services/steps/step4_cross_size_planner.py):1411
- [backend/services/steps/step4_cross_size_planner.py](C:/Users/abqua/Desktop/HRI%20app/backend/services/steps/step4_cross_size_planner.py):1416

## 5. Quantity for one slab-month

After the discount component is computed, Step 4 creates two slab quantities.

### 5.1 Default-world quantity

```text
default_world_qty
= non_discount_baseline_qty
+ discount_component_default
```

### 5.2 Scenario quantity before cross-size adjustment

```text
pre_cross_qty
= non_discount_baseline_qty
+ discount_component_scenario
```

Both are clipped at zero.

Code:

- [backend/services/steps/step4_cross_size_planner.py](C:/Users/abqua/Desktop/HRI%20app/backend/services/steps/step4_cross_size_planner.py):1422
- [backend/services/steps/step4_cross_size_planner.py](C:/Users/abqua/Desktop/HRI%20app/backend/services/steps/step4_cross_size_planner.py):1423

This is the most important point:

Step 4 first creates slab-level quantities **size by size**, before any cross-size correction.

## 6. From slab quantities to size totals

For each month:

- sum all slab `non_discount_baseline_qty` to get size baseline quantity
- sum all slab `pre_cross_qty` to get size pre-cross quantity

So for `12-ML` in a month:

```text
size_baseline_12 = sum of slab baselines
size_precross_12 = sum of slab pre_cross quantities
```

Same for `18-ML`.

At this point:

- the within-size discount model is already applied
- cross-size elasticity is **not** applied yet

## 7. Current cross-size adjustment design

This is the final corrected design now.

Cross-size elasticity is **not** applied month by month anymore.

Instead, Step 4 does:

1. Build all 3 months additively first
2. Sum 3-month pre-cross totals for `12-ML`
3. Sum 3-month pre-cross totals for `18-ML`
4. Compare those totals to the 3-month baseline totals
5. Apply cross-size elasticity once on the 3-month total change
6. Push the result back to monthly/slab rows proportionally

This avoids the old rebound problem caused by month-to-month chaining.

## 8. 3-month own change before cross adjustment

For each size:

```text
own_change_3m
= (pre_cross_qty_3m - baseline_qty_3m) / baseline_qty_3m
```

So:

- `own12_3m` = 12-ML additive 3-month growth before cross correction
- `own18_3m` = 18-ML additive 3-month growth before cross correction

Code:

- [backend/services/steps/step4_cross_size_planner.py](C:/Users/abqua/Desktop/HRI%20app/backend/services/steps/step4_cross_size_planner.py):1488
- [backend/services/steps/step4_cross_size_planner.py](C:/Users/abqua/Desktop/HRI%20app/backend/services/steps/step4_cross_size_planner.py):1489

## 9. Cross-size elasticity readjustment

Step 4 fits two historical size-level elasticity numbers:

- `12 wrt 18`
- `18 wrt 12`

Then it solves the 3-month total overall change:

```text
overall12_3m = (own12_3m + e12_from_18 * own18_3m) / denom
overall18_3m = (own18_3m + e18_from_12 * own12_3m) / denom
denom = 1 - (e12_from_18 * e18_from_12)
```

Code:

- [backend/services/steps/step4_cross_size_planner.py](C:/Users/abqua/Desktop/HRI%20app/backend/services/steps/step4_cross_size_planner.py):1491
- [backend/services/steps/step4_cross_size_planner.py](C:/Users/abqua/Desktop/HRI%20app/backend/services/steps/step4_cross_size_planner.py):1495

This gives a final 3-month corrected total for each size.

## 10. Final 3-month size quantity after cross adjustment

Once `overall12_3m` and `overall18_3m` are known:

```text
final_qty_12_3m = baseline_qty_12_3m * (1 + overall12_3m)
final_qty_18_3m = baseline_qty_18_3m * (1 + overall18_3m)
```

Code:

- [backend/services/steps/step4_cross_size_planner.py](C:/Users/abqua/Desktop/HRI%20app/backend/services/steps/step4_cross_size_planner.py):1498
- [backend/services/steps/step4_cross_size_planner.py](C:/Users/abqua/Desktop/HRI%20app/backend/services/steps/step4_cross_size_planner.py):1499

## 11. Push the 3-month adjustment back to months and slabs

Step 4 does not solve a separate cross effect for each month anymore.

Instead, for each size it computes one 3-month scale:

```text
scale_12 = final_qty_12_3m / pre_cross_qty_12_3m
scale_18 = final_qty_18_3m / pre_cross_qty_18_3m
```

Then every month and every slab inside that size is scaled by the same size-level factor:

```text
final_slab_qty = pre_cross_slab_qty * scale_size
```

Code:

- [backend/services/steps/step4_cross_size_planner.py](C:/Users/abqua/Desktop/HRI%20app/backend/services/steps/step4_cross_size_planner.py):1500
- [backend/services/steps/step4_cross_size_planner.py](C:/Users/abqua/Desktop/HRI%20app/backend/services/steps/step4_cross_size_planner.py):1520

Why this is done:

- keep the month/slab shape created by the discount model
- apply the cross-size correction only once at the 3-month level
- avoid month-to-month rebound instability

## 12. Revenue calculation

Revenue is computed after final slab quantities are known.

For each slab-month:

```text
baseline_revenue
= baseline_qty * base_price * (1 - default_discount / 100)

scenario_revenue
= final_qty * base_price * (1 - scenario_discount / 100)
```

Code:

- [backend/services/steps/step4_cross_size_planner.py](C:/Users/abqua/Desktop/HRI%20app/backend/services/steps/step4_cross_size_planner.py):1528
- [backend/services/steps/step4_cross_size_planner.py](C:/Users/abqua/Desktop/HRI%20app/backend/services/steps/step4_cross_size_planner.py):1529

## 13. Profit calculation

Profit is also computed after final slab quantities are known.

For each slab-month:

```text
baseline_profit = baseline_revenue - baseline_qty * cogs_per_unit
scenario_profit = scenario_revenue - final_qty * cogs_per_unit
```

Code:

- [backend/services/steps/step4_cross_size_planner.py](C:/Users/abqua/Desktop/HRI%20app/backend/services/steps/step4_cross_size_planner.py):1530
- [backend/services/steps/step4_cross_size_planner.py](C:/Users/abqua/Desktop/HRI%20app/backend/services/steps/step4_cross_size_planner.py):1531

## 14. What the top cards show

The top cards on Step 4 show 3-month totals.

For each size:

- Volume = summed final quantity over 3 months
- Revenue = summed scenario revenue over 3 months
- Profit = summed scenario profit over 3 months

For total:

- Volume is shown in `ml`, not simple packs
- total volume uses:
  - `12-ML quantity * 12`
  - `18-ML quantity * 18`

Code:

- [backend/services/steps/step4_cross_size_planner.py](C:/Users/abqua/Desktop/HRI%20app/backend/services/steps/step4_cross_size_planner.py):1635
- [backend/services/steps/step4_cross_size_planner.py](C:/Users/abqua/Desktop/HRI%20app/backend/services/steps/step4_cross_size_planner.py):1637

## 15. What the comparison percent is measured against

The Step 4 top card percent is compared against **same 3 months last year**, not against the Step 4 baseline.

Those LY values come from:

- `reference_qty`
- `reference_revenue`
- `reference_profit`
- `reference_volume_ml`

Code:

- [backend/services/steps/step4_cross_size_planner.py](C:/Users/abqua/Desktop/HRI%20app/backend/services/steps/step4_cross_size_planner.py):1575
- [backend/services/steps/step4_cross_size_planner.py](C:/Users/abqua/Desktop/HRI%20app/backend/services/steps/step4_cross_size_planner.py):1648

So on the UI:

- big number = current Step 4 scenario result
- green/red percent below = versus LY same 3M

## 16. Debug terms shown in the trace

Trace terms mean:

- `non_discount_baseline_qty`: pure forecast baseline with no discount component
- `default_world_qty`: baseline plus default discount component
- `pre_cross_qty`: baseline plus scenario discount component
- `final_qty`: post cross-size 3-month-adjusted slab quantity

So the quantity path is:

```text
non_discount_baseline_qty
    -> add default discount component -> default_world_qty
    -> add scenario discount component -> pre_cross_qty
    -> apply 3M size-level cross adjustment -> final_qty
```

## 17. Short summary

Step 4 currently works like this:

1. Forecast non-discount slab baseline for the next 3 months
2. For each slab-month, compute discount effect from:
   - own discount
   - lag discount
   - other slabs weighted discount
3. Add baseline + discount component to get additive slab quantity
4. Sum to get 3-month additive totals for `12-ML` and `18-ML`
5. Apply cross-size elasticity once on those 3-month totals
6. Scale monthly/slab quantities back from that 3-month correction
7. Compute revenue and profit from final quantities
8. Compare the 3-month totals against LY same 3M in the cards

That is the current Step 4 design.
