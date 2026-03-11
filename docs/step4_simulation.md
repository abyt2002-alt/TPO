# Step 4 Simulation (Plain-English + Equations)

This document simulates one full Step 4 run in simple terms.
It explains exactly how quantity, revenue, and profit are produced.

## What Step 4 uses

For each size (`12-ML`, `18-ML`) and slab (`slab1`, `slab2`, ...), Step 4 needs:

1. `baseline_non_discount_qty` for each forecast month (from baseline projection).
2. Step 3 coefficients for that slab:
   - `beta_base`
   - `beta_lag`
   - `beta_other`
3. Default discount per slab.
4. Scenario discount input per slab/month (user edits).
5. `base_price` and `cogs_per_unit`.
6. Cross-size elasticities:
   - `e12_from_18`
   - `e18_from_12`

---

## Step 1: Compute one slab quantity (before cross-size adjustment)

For slab `s` in month `m`:

\[
\text{discount\_component}_{s,m}
=
\beta_{\text{base},s}\cdot D_{s,m}
+
\beta_{\text{lag},s}\cdot L_{s,m}
+
\beta_{\text{other},s}\cdot O_{s,m}
\]

Where:
- \(D_{s,m}\): slab's own scenario discount for month `m`
- \(L_{s,m}\): lag discount (M1 uses last historical default, M2 uses M1 scenario, M3 uses M2 scenario)
- \(O_{s,m}\): weighted discount of other slabs in same size (excluding self)

Then:

\[
\text{pre\_cross\_qty}_{s,m}
=
\max\left(
\text{baseline\_non\_discount\_qty}_{s,m}
+
\text{discount\_component}_{s,m},
0
\right)
\]

---

## Step 2: Mini numeric example for one slab

Example (`12-ML`, `slab1`, month `Dec`):

- `baseline_non_discount_qty = 30,000`
- `beta_base = 1,000`
- `beta_lag = -200`
- `beta_other = 50`
- `scenario_discount = 16`
- `lag_discount = 14`
- `other_weighted_discount = 21`

Discount component:

\[
1{,}000\times16 + (-200)\times14 + 50\times21
= 16{,}000 - 2{,}800 + 1{,}050
= 14{,}250
\]

Pre-cross quantity:

\[
30{,}000 + 14{,}250 = 44{,}250
\]

So this slab contributes **44,250 units** before any cross-size adjustment.

---

## Step 3: Build pack totals (12-ML and 18-ML)

For each month:

\[
\text{pre\_cross\_qty}_{12,m}
=
\sum_{s\in\text{12-ML slabs}} \text{pre\_cross\_qty}_{s,m}
\]

\[
\text{pre\_cross\_qty}_{18,m}
=
\sum_{s\in\text{18-ML slabs}} \text{pre\_cross\_qty}_{s,m}
\]

Also sum the non-discount baseline the same way:

\[
\text{baseline\_qty}_{12,m}
=
\sum_{s\in\text{12-ML slabs}} \text{baseline\_non\_discount\_qty}_{s,m}
\]

\[
\text{baseline\_qty}_{18,m}
=
\sum_{s\in\text{18-ML slabs}} \text{baseline\_non\_discount\_qty}_{s,m}
\]

Then sum each over 3 months to get 3M totals.

---

## Step 4: Cross-size adjustment on 3M totals

Step 4 uses 3-month pack totals (not month-by-month chaining) for stability.

Own 3M change:

\[
\text{own12}
=
\frac{\text{pre12\_3m} - \text{base12\_3m}}{\text{base12\_3m}}\times 100
\]

\[
\text{own18}
=
\frac{\text{pre18\_3m} - \text{base18\_3m}}{\text{base18\_3m}}\times 100
\]

Simultaneous system:

\[
\text{overall12} = \text{own12} + e12\_{\text{from18}}\cdot \text{overall18}
\]

\[
\text{overall18} = \text{own18} + e18\_{\text{from12}}\cdot \text{overall12}
\]

Closed form:

\[
\text{denom} = 1 - e12\_{\text{from18}}\cdot e18\_{\text{from12}}
\]

\[
\text{overall12}
=
\frac{\text{own12} + e12\_{\text{from18}}\cdot \text{own18}}{\text{denom}}
\]

\[
\text{overall18}
=
\frac{\text{own18} + e18\_{\text{from12}}\cdot \text{own12}}{\text{denom}}
\]

Then target final 3M pack quantities:

\[
\text{final12\_3m}
=
\text{base12\_3m}\cdot\left(1 + \frac{\text{overall12}}{100}\right)
\]

\[
\text{final18\_3m}
=
\text{base18\_3m}\cdot\left(1 + \frac{\text{overall18}}{100}\right)
\]

Scales:

\[
\text{scale12}=\frac{\text{final12\_3m}}{\text{pre12\_3m}},\quad
\text{scale18}=\frac{\text{final18\_3m}}{\text{pre18\_3m}}
\]

Apply back to every slab-month:

\[
\text{final\_qty}_{s,m}
=
\text{pre\_cross\_qty}_{s,m}\times \text{scale(size)}
\]

---

## Step 5: Revenue and profit after final quantity

For each slab-month:

\[
\text{scenario\_revenue}_{s,m}
=
\text{final\_qty}_{s,m}\cdot \text{base\_price}_{s}\cdot\left(1-\frac{\text{scenario\_discount}_{s,m}}{100}\right)
\]

\[
\text{scenario\_profit}_{s,m}
=
\text{scenario\_revenue}_{s,m}
-
\text{final\_qty}_{s,m}\cdot \text{cogs\_per\_unit}_{\text{size}}
\]

Then sum slab-month values to size 3M totals and total (12 + 18).

---

## Step 6: Reference comparison (for card percentages)

Reference is always from actual historical data under current filters.

Two modes:
1. `LY same 3M`: same months one year earlier.
2. `last 3M before projection`: immediate 3 months before forecast start.

Reference metrics:
- `reference_qty`
- `reference_revenue = sum(SalesValue_atBasicRate - TotalDiscount)`
- `reference_profit = reference_revenue - reference_qty*cogs_per_unit`

Displayed % is:

\[
\%\Delta\text{Volume}
=
\frac{\text{final\_qty\_3m} - \text{reference\_qty\_3m}}{\text{reference\_qty\_3m}}\times 100
\]

and similarly for revenue/profit.

---

## End-to-end flow in one sentence

Step 4 takes projected non-discount slab baseline, adds model-based discount effect, aggregates to pack totals, applies 3M cross-size correction once, pushes that back to slab quantities, then computes revenue/profit and compares against selected historical reference window.
