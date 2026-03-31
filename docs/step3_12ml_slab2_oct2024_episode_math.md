# Step 3 Structural ROI Episode Math
## Case: 12-ML, slab2, Episode starting Oct-2024

This note explains exactly how the Step 3 **structural ROI point** is computed for:
- Size: `12-ML`
- Slab: `slab2`
- ROI point period: `2024-10-01`

All formulas below match backend logic in:
- `backend/services/steps/step3_modeling.py` (`_build_structural_roi_points`)

---

## 1) Episode definition used by backend

Backend builds base-discount regimes month-wise and creates an ROI episode only when:

\[
\Delta d_{\text{base}} = d_{\text{curr}} - d_{\text{prev}} > 0
\]

For this ROI point:
- Previous base discount: \(d_{prev} = 18.0\%\)
- Current base discount: \(d_{curr} = 19.5\%\)
- Step-up: \(\Delta d = 1.5\%\)

This current regime spans **2 months**:
- Oct-2024
- Nov-2024

So this ROI point is a **2-month episode aggregate**, not a single-month-only calculation.

---

## 2) Stage-2 quantity model used inside episode

For each month \(t\) in the episode, structural worlds are predicted with tactical forced to zero:

\[
\hat{Q}_{prev,t} = f(\text{residual\_store}_t, d_{prev}, 0, \text{lag}_{prev,t})
\]
\[
\hat{Q}_{curr,t} = f(\text{residual\_store}_t, d_{curr}, 0, \text{lag}_{curr,t})
\]

Lag setup in this episode:
- Oct: \(\text{lag}_{curr}=d_{prev}\)
- Nov: \(\text{lag}_{curr}=d_{curr}\)
- Both months for prev-world use \(\text{lag}_{prev}=d_{prev}\)

---

## 3) Revenue and spend formulas per month

Backend uses **same Streamlit structural convention** in this function:

\[
P^{base}_t = \text{BasePrice}_t \times (1 - d_{prev})
\]
\[
P^{curr}_t = \text{BasePrice}_t \times (1 - d_{prev})
\]

So both worlds are valued at the same net price tied to \(d_{prev}\).

Per month:

\[
R_{base,t} = \hat{Q}_{prev,t} \times P^{base}_t
\]
\[
R_{curr,t} = \hat{Q}_{curr,t} \times P^{curr}_t
\]
\[
\Delta R_t = R_{curr,t} - R_{base,t}
\]
\[
Spend_t = \text{BasePrice}_t \times \left(\frac{\Delta d}{100}\right) \times \hat{Q}_{curr,t}
\]

Episode sums:

\[
\Delta R_{episode} = \sum_t \Delta R_t
\]
\[
Spend_{episode} = \sum_t Spend_t
\]

ROI outputs:

\[
Topline\ ROI = \frac{\Delta R_{episode}}{Spend_{episode}}
\]
\[
Profit\ ROI = \frac{\Delta GP_{episode}}{Spend_{episode}}
\]

---

## 4) Exact numbers from this ROI point (payload)

From `roi_points` for `12-ML / slab2 / 2024-10-01`:

- `spend = 87,019.71129972197`
- `incremental_revenue = 2,377,035.7864385573`
- `incremental_profit = 843,950.0363672494`
- `roi_1mo = 27.316061509918523`
- `profit_roi_1mo = 9.698377801558461`

Verification:

\[
\frac{2,377,035.7864385573}{87,019.71129972197} = 27.316061509918523
\]

\[
\frac{843,950.0363672494}{87,019.71129972197} = 9.698377801558461
\]

---

## 5) Useful derived checks

Because \(\Delta d = 1.5\%\):

\[
\sum_t (\text{BasePrice}_t \times \hat{Q}_{curr,t})
= \frac{Spend_{episode}}{0.015}
= 5,801,314.086648132
\]

Because valuation uses \((1-d_{prev})=(1-0.18)=0.82\):

\[
\sum_t \left[\text{BasePrice}_t \times (\hat{Q}_{curr,t}-\hat{Q}_{prev,t})\right]
= \frac{\Delta R_{episode}}{0.82}
= 2,898,824.1298031183
\]

These are consistent with the payload totals.

---

## 6) What the period label means

The plotted bar at `2024-10-01` is the **episode start month label**.
The bar values (`spend`, `incremental_revenue`, ROI) are for the **full hold window** (Oct + Nov), not Oct alone.

---

## 7) First-row context fields (display fields)

The same ROI point also carries first-row display context:
- `actual_quantity = 229,404`
- `predicted_quantity = 258,818.722588795`
- `actual_discount_pct = 19.701709747314453`
- `base_discount_pct = 19.5`

These first-row fields are informational; ROI numerator/denominator are episode sums.

---

## 8) What is **not** shown in the current UI/API point

No, the current `roi_points` output does **not** show the detailed month-by-month intermediate values like:

- Oct BasePrice
- Nov BasePrice
- Oct \(\hat{Q}_{prev}\), \(\hat{Q}_{curr}\)
- Nov \(\hat{Q}_{prev}\), \(\hat{Q}_{curr}\)
- Oct spend / Nov spend
- Oct incremental revenue / Nov incremental revenue

Those are computed internally and then summed into:
- `spend` (episode total)
- `incremental_revenue` (episode total)
- `incremental_profit` (episode total)

---

## 9) Raw Oct/Nov decomposition (fully expanded from payload)

This section expands the same episode into month-wise internals.

Known payload values (12-ML, slab2):
- Stage-2 coefficients:
  - \(b_0=-1,237,112.860423829\)
  - \(b_{res}=616.9570217897353\)
  - \(b_{struct}=85,684.93123222867\)
  - \(b_{tact}=0\)
  - \(b_{lag}=-1,027.0013454232524\)
- \(d_{prev}=18.0\%\), \(d_{curr}=19.5\%\), \(\Delta d=1.5\%\)
- COGS per unit \(=6.0\)

Observed monthly payload rows:
- Oct-2024:
  - \(\hat Q_{actual}=258,818.722588795\)
  - \(d_{actual}=19.7017097473\%\)
  - \(d_{base}=19.5\%\)
  - tactical \(=0.2017097473\%\)
  - tactical spend (from `predicted_vs_actual`) \(=5,919.807993230353\)
- Nov-2024:
  - \(\hat Q_{actual}=252,534.62491727085\)
  - \(d_{actual}=19.2799892426\%\)
  - \(d_{base}=19.5\%\)
  - tactical \(=0\%\)
  - tactical incremental revenue (from `predicted_vs_actual`) \(=6,306.606273556128\)

### 9.1 Recover residuals used for structural episode

Because \(b_{tact}=0\), month-level prediction equation is:

\[
\hat Q_t=b_0+b_{res}\cdot Residual_t+b_{struct}\cdot d_{base,t}+b_{lag}\cdot lag_t
\]

So:

\[
Residual_t=\frac{\hat Q_t-b_0-b_{struct}d_{base,t}-b_{lag}lag_t}{b_{res}}
\]

Results:
- Oct residual \(=-253.5647480669\)
- Nov residual \(=-261.2534451493\)

### 9.2 Structural world quantities for episode

For each month:

\[
\hat Q_{prev,t}=f(Residual_t,d_{prev},0,lag_{prev,t})
\]
\[
\hat Q_{curr,t}=f(Residual_t,d_{curr},0,lag_{curr,t})
\]

Lag setup:
- Oct: \(lag_{prev}=18.0,\ lag_{curr}=18.0\)
- Nov: \(lag_{prev}=18.0,\ lag_{curr}=19.5\)

Computed quantities:
- Oct:
  - \(\hat Q_{prev}=130,291.3257404520\)
  - \(\hat Q_{curr}=258,818.7225887951\)
- Nov:
  - \(\hat Q_{prev}=125,547.7300870627\)
  - \(\hat Q_{curr}=252,534.6249172709\)

### 9.3 BasePrice used in Oct and Nov

Oct BasePrice from tactical spend identity:

\[
Spend^{tactical}_{Oct}=BasePrice_{Oct}\cdot\frac{tactical_{Oct}}{100}\cdot \hat Q_{actual,Oct}
\]
\[
BasePrice_{Oct}=\frac{5,919.807993230353}{(0.2017097473/100)\cdot258,818.722588795}=11.3392686844
\]

Nov BasePrice from tactical incremental revenue identity (\(tactical=0\)):

\[
\Delta R^{tactical}_{Nov}
=
\hat Q_{actual,Nov}\cdot BasePrice_{Nov}\cdot
\left(\frac{d_{base,Nov}-d_{actual,Nov}}{100}\right)
\]
\[
BasePrice_{Nov}
=
\frac{6,306.606273556128}
{252,534.62491727085\cdot((19.5-19.2799892426)/100)}
=11.3509149551
\]

### 9.4 Month-wise structural episode math (exact)

Price factor for structural episode valuation:

\[
P_t=BasePrice_t\cdot(1-d_{prev})
=BasePrice_t\cdot0.82
\]

Per month:

\[
R_{base,t}=\hat Q_{prev,t}\cdot P_t,\quad
R_{curr,t}=\hat Q_{curr,t}\cdot P_t,\quad
\Delta R_t=R_{curr,t}-R_{base,t}
\]
\[
Spend_t=BasePrice_t\cdot\frac{\Delta d}{100}\cdot\hat Q_{curr,t}
\]
\[
\Delta GP_t=\left(R_{curr,t}-COGS\cdot\hat Q_{curr,t}\right)-\left(R_{base,t}-COGS\cdot\hat Q_{prev,t}\right)
\]

Computed month-wise components:

| Month | BasePrice | \( \hat Q_{prev} \) | \( \hat Q_{curr} \) | \( \Delta R_t \) | \( Spend_t \) | \( \Delta GP_t \) |
|---|---:|---:|---:|---:|---:|---:|
| Oct-2024 | 11.3392686844 | 130,291.3257404520 | 258,818.7225887951 | 1,195,073.4826579436 | 44,022.2255397634 | 423,909.1015678849 |
| Nov-2024 | 11.3509149551 | 125,547.7300870627 | 252,534.6249172709 | 1,181,962.3037805718 | 42,997.4857599571 | 420,040.9347993223 |

Episode totals (match payload):

\[
\sum \Delta R_t = 2,377,035.7864385573
\]
\[
\sum Spend_t = 87,019.71129972197
\]
\[
\sum \Delta GP_t = 843,950.0363672494
\]

And therefore:

\[
Topline\ ROI=\frac{2,377,035.7864385573}{87,019.71129972197}=27.3160615099
\]
\[
Profit\ ROI=\frac{843,950.0363672494}{87,019.71129972197}=9.6983778016
\]
