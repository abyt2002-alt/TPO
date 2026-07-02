# Forecast & Chart Issues — Root Cause Document

> Generated: 2026-05-26  
> Covers: Step 5 Baseline Forecast chart, Step 4 Planner forecast override, historical baseline collapse

---

## How the chart is built (important context first)

The "Historical Baseline + Discount Component" chart on the Step 5 page has two halves:

| Section | Source |
|---|---|
| **Historical period** (Apr 24 → Apr 26) | Step 5 backend — `non_discount_baseline_quantity` from the 2-stage model per slab, summed |
| **Forecast period** (May 26 → Jul 26) | **Step 4 planner output overrides Step 5** — uses `plannerData.monthly_results[].sizes['12-ML' / '18-ML'].final_total_qty` |

Frontend code (`BaselineForecast.jsx:186`):
```js
const combinedPoints = [
  ...historicalPoints,
  ...(plannerForecastPoints.length > 0
    ? plannerForecastPoints          // ← Step 4 wins when available
    : parsedPoints.filter(row => row.is_forecast)),
]
```

And the "discount component" in the chart for the forecast period is:
```js
discount18: total18 - baseline18    // total from planner, baseline from planner
```

So if planner says `baseline = 0` and `total = 2.8M`, the chart shows 2.8M of yellow "discount" area.

---

## Issue 1 — Both baselines collapse to near-zero throughout history

### What you see
Both the 12-ML blue area and 18-ML red area are essentially flat at the bottom of the chart. The entire volume shows as "discount component" (teal/yellow fill). The blue/red area is barely visible.

### Why it happens
The historical `baseline_quantity` plotted = `non_discount_baseline_quantity` from the model. This is:

```
baseline = predicted_quantity − (coef_base × base_discount_pct
                                + coef_lag  × lag1_base_discount_pct
                                + coef_other × other_slabs_discount)
```

The frontend then clamps it (`BaselineForecast.jsx:172`):
```js
const baseline12 = Math.min(Math.max(Number(row.baseline12 || 0), 0), total12)
```

When the discount component (right-hand side of the subtraction) is **larger than predicted_quantity**, baseline goes negative → clamped to 0 → discount fills the entire chart.

### Actual discount percentages in the data
Both sizes run 15–24% discount almost every month (from `Total_Scheme_Pct`, quantity-weighted):

**12-ML disc % by month:**
```
Apr 24: 17.9%   May 24: 17.9%   Jun 24: 17.9%   Jul 24: 17.9%
Aug 24: 17.0%   Sep 24: 15.3%   Oct 24: 15.1%   Nov 24: 15.8%
Dec 24: 16.3%   Jan 25: 16.3%   Feb 25: 16.5%   Mar 25: 16.8%
Apr 25: 23.9%   May 25: 16.2%   Jun 25: 16.6%   Jul 25: 22.7%
Aug 25: 16.2%   Sep 25: 24.1%   Oct 25: 16.2%   Nov 25: 20.1%
Dec 25: 16.2%   Jan 26: 20.7%   Feb 26: 16.6%   Mar 26: 17.0%
Apr 26: 19.4%
```

**18-ML disc % by month:**
```
Apr 24: 16.1%   May 24: 18.1%   Jun 24: 16.2%   Jul 24: 17.7%
Aug 24: 16.3%   Sep 24: 17.8%   Oct 24: 17.3%   Nov 24: 17.9%
Dec 24: 16.4%   Jan 25: 17.7%   Feb 25: 16.4%   Mar 25: 17.0%
Apr 25: 15.1%   May 25: 17.8%   Jun 25: 15.2%   Jul 25: 15.5%
Aug 25: 16.9%   Sep 25: 15.1%   Oct 25: 16.8%   Nov 25: 15.2%
Dec 25: 17.2%   Jan 26: 15.1%   Feb 26: 17.2%   Mar 26: 15.6%
Apr 26: 15.4%
```

Since discount is present in every single month (never zero), the regression cannot isolate an "organic" baseline — it attributes most volume to discount, leaving a near-zero baseline. This is a **multicollinearity problem**: volume and discount are correlated in the same months, so the model assigns most of the volume to the discount coefficient.

---

## Issue 2 — 12-ML zigzag pattern in history (up-down-up-down)

### What you see
12-ML alternates dramatically: Apr 25 → 934K, May 25 → 241K, Jul 25 → 712K, Aug 25 → 255K, Sep 25 → 939K, Oct 25 → 302K... The chart shows a series of sharp peaks and valleys.

### Why it happens
This is **trade loading / forward buying** — a real business pattern in the data, not a bug.

When the discount jumps (e.g. Apr 25 base disc goes from 16.8% to 23.9%), distributors **bulk-buy** in that month, then sit on stock and **don't order** the following month. The model captures this through the negative `lag1_base_discount_pct` coefficient.

**Actual pattern in the data:**
```
Month     Qty         Disc%   Notes
Mar 25    482,586     16.8%   — normal
Apr 25    934,107     23.9%   ← disc jumps 7pp → bulk buy
May 25    241,706     16.2%   ← hangover (outlets stocked up)
Jun 25    353,064     16.6%   — recovering
Jul 25    711,945     22.7%   ← disc jumps again → bulk buy
Aug 25    255,035     16.2%   ← hangover again
```

**This is correct behaviour** — the data is showing trade loading. The model's negative lag coefficient is intended to capture exactly this.

---

## Issue 3 — 12-ML forecast drops to 500K then goes flat/straight

### What you see
After the Apr 26 peak of 1,847,104 units, the forecast (May–Jul 26, from Step 4 planner) drops to ~500K then recovers flatly to ~580K → ~640K.

### Why it happens
**Cause A — Lag correction from Apr 26's high discount.**

For the first forecast month (May 26), the planner uses Apr 26's actual discount as the lag (`step4_cross_size_planner.py:2961`):
```python
scenario_lag = float(slab_payload.get('default_discount_pct', scenario_discount))
# = Apr 26 disc% ≈ 19.4%
```

The Stage 2 model then computes:
```
May 26 discount_component = coef_base × Feb26_disc% + coef_lag × 19.4%
```

With `coef_lag` being **negative** (hangover), a 19.4% lag wipes out a large portion of the predicted volume → drops to ~500K.

**Cause B — No seasonality in the model.**

The Holt model in step5 (`_forecast_baseline_series`) has no seasonal component. It doesn't know that May is historically always a **low month** for 12-ML:
```
May 24: 308,754   May 25: 241,706
```
The flat recovery from 500K → 580K → 640K is just the lag effect dissipating across 3 months, not a realistic seasonal forecast.

**What Step 5 Holt actually forecasts** (on raw quantity as proxy):
```
Current code (damped_trend=False):  May 26 → 1,148,399 | Jun 26 → 1,226,228 | Jul 26 → 1,304,056
Proposed fix  (damped_trend=True):  May 26 → 1,140,457 | Jun 26 → 1,215,228 | Jul 26 → 1,289,253
```
Note: the step5 Holt result is overridden by the Step 4 planner in the chart, so what you actually see is the planner's 500K–640K range (driven by the lag correction above).

---

## Issue 4 — 18-ML forecast jumps to 2.8M (the main wrong number)

### What you see
After Apr 26 actual of 731,560 units, the forecast for May 26 jumps to **~2,800,000** — nearly 4× the last actual. Then it drops to ~2,500,000 and ~2,700,000 for Jun and Jul 26. Almost the entire forecast bar is yellow ("discount component"), with near-zero red baseline.

### The actual 18-ML outlet and volume trend
```
Month     Qty           Outlets    Disc%
Jan 25    1,361,837     25,041     17.7%  ← peak outlets
Mar 25    1,736,799     25,675     17.0%  ← peak volume
Sep 25      719,487     16,859     15.1%
Nov 25      761,553     14,493     15.2%
Jan 26      691,134     15,185     15.1%
Mar 26      821,143     13,397     15.6%
Apr 26      731,560     12,075     15.4%  ← last actual
```
18-ML lost **~13,600 outlets** (−53%) between Jan 25 and Apr 26. Volume halved. The discount percentage barely changed (~15–17% throughout).

### Why the forecast is 2.8M instead of ~800K

**Step 1 — Step 5 Holt baseline (not shown, overridden by planner):**

The Step 5 Holt forecast on the 18-ML quantity series (`_forecast_baseline_series`, `step5_baseline_forecast.py:55`):
```
Current code (damped_trend=False):  May 26 → 819,565 | Jun 26 → 778,854 | Jul 26 → 738,142
Proposed fix  (damped_trend=True):  May 26 → 831,402 | Jun 26 → 794,740 | Jul 26 → 758,446
```
These numbers are **reasonable**. But they are never shown — the planner overrides them.

**Step 2 — Step 4 planner computes 2.8M:**

The planner (`step4_cross_size_planner.py:2793`) runs the 2-stage model for each 18-ML slab:
```python
baseline_forecast = _predict_stage2_quantity(
    stage2_model,
    residual_forecast,   # ← Holt forecast of stage-1 residuals (ALSO damped_trend=False)
    zeros_forecast,      # base_discount = 0
    zeros_forecast,      # lag = 0
    zeros_forecast,
)
pre_cross_qty = max(baseline_non_discount_qty + discount_component_scenario, 0.0)
```

**The 2.8M comes from two compounding problems:**

**Problem A — Model coefficients are anchored to the high-volume era.**
The 2-stage model is fitted on the full 25-month history. During Jan–Mar 2025, 18-ML had 25K outlets and 1.4–1.7M monthly volume. The regression coefficients for `coef_base` (discount → volume uplift) were calibrated when the underlying scale was ~1.5M/month. Now the scale is ~750K/month (half as many outlets). Summed across 4–5 18-ML slabs, each slab's model may predict 500–700K at a ~16% discount — totalling 2.5–3M, even though actual is 731K.

**Problem B — `_forecast_unbounded_series` in Step 4 also has `damped_trend=False`** (`step4_cross_size_planner.py:65`).
The stage-1 residual series (store count variation not explained by discount) is forecasted with undamped Holt. If these residuals are noisy/volatile, the undamped Holt can project them upward. A positive residual forecast → higher `baseline_non_discount_qty` → further inflates the total.

**Result in the chart:**
```
Planner baseline_18_ml  ≈ 0      (model baseline collapses, Issue 1)
Planner final_total_qty ≈ 2,800,000
Discount component shown = 2,800,000 − 0 = 2,800,000  ← entire bar is yellow
```

The 2.8M is a **model artifact from outdated coefficients + undamped trend + baseline collapse** — not a real business forecast.

---

## Summary table

| Issue | Where | Root Cause | Impact on Chart |
|---|---|---|---|
| Baseline ≈ 0 throughout history | Step 3 model output | High discount coefficients fitted on always-discounted data; baseline = actual − discount_component < 0 → clamped to 0 | All volume shows as yellow/teal "discount component"; thin baseline area |
| 12-ML zigzag (up-down-up) | Historical data | Trade loading: bulk-buy in high-discount months, hangover the next month. Negative lag coefficient captures this. **This is correct behaviour.** | Alternating sharp peaks and valleys in the red/green line |
| 12-ML forecast flat at 500–640K | Step 4 planner | Negative lag correction from Apr 26's 19.4% discount suppresses May 26. No seasonal model. | Line drops from 1.85M to 500K then slowly recovers — looks artificially flat |
| 18-ML forecast jumps to 2.8M | Step 4 planner | Model coefficients from high-outlet era (25K outlets, 1.5M volume) applied to current low-outlet era (12K outlets, 731K volume). Baseline = 0 so entire prediction = discount component. | Massive yellow spike in forecast zone; completely wrong scale |
| Undamped trend in step 5 Holt | `step5_baseline_forecast.py:55` | `damped_trend=False` — Holt extrapolates recent trend indefinitely | 12-ML baseline would project to 1.3M (too high); 18-ML baseline would project to 738K (ok but declining) |
| Undamped trend in step 4 residual Holt | `step4_cross_size_planner.py:65` | `damped_trend=False` — same bug in the residual_store forecast used for planner baseline | Inflates planner baseline_non_discount_qty for 12-ML; may contribute to 18-ML inflation |

---

## Proposed fixes (not yet applied)

### Fix 1 — `damped_trend=True` in both step4 and step5 (1-line change each)

**`step5_baseline_forecast.py:55`**
```python
# Before
damped_trend=False,
# After
damped_trend=True,
```

**`step4_cross_size_planner.py:65`**
```python
# Before
damped_trend=False,
# After
damped_trend=True,
```

Effect: Holt's trend extrapolation gradually dampens to flat instead of continuing forever. Low cost, no new dependency.

### Fix 2 — Switch step5 to Holt-Winters (seasonal forecasting)

With 25 months of data (Apr 24–Apr 26), we have just over 2 full seasonal cycles — the minimum for `ExponentialSmoothing` with `seasonal_periods=12`. This would capture that April is always high for 12-ML, May is always low, etc.

```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing

fitted = ExponentialSmoothing(
    arr,
    trend='add',
    seasonal='add',
    seasonal_periods=12,
    damped_trend=True,
    initialization_method='estimated',
).fit(optimized=True)
```

**Why not ARIMA/SARIMA?** SARIMA(p,d,q)(P,D,Q)[12] needs 3+ full seasonal cycles (36+ months) to fit reliably. With 25 months it is unstable. Holt-Winters is the right choice at this data length.

### Fix 3 — Refit or re-weight the Step 4 model for 18-ML's structural outlet decline

This is the deeper issue. The 18-ML model must be told that the outlet universe has halved. Options:
- Fit the model only on the last 12 months (Apr 25–Apr 26, the declining era) rather than all 25 months
- Add outlet count as an explicit covariate in the forecast
- Weight recent months more heavily in the regression (recency weighting)

This requires a model change discussion, not just a parameter flip.
