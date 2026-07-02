# Transition Notes: DMS Sales Data → Scheme Utilisation Data

**Purpose of this document**: A future agent or developer picking up this work should be
able to read this file and understand exactly what changed, why, where the files are, and
what still needs to be done. Keep updating this file as each step of the tool is migrated.

---

## Column Dictionary

Key computed columns in `step2_deduped_sales_lines.parquet` — what they mean and how they are derived.

| Column | Formula | Meaning |
|--------|---------|---------|
| `Net_Amt` | `CLP × Quantity` | Company revenue at CLP. CLP = `MRP ÷ 1.12 ÷ 1.07 ÷ (1 + GST%)` — strips out retailer margin (12%), distributor margin (7%), and GST from MRP. This is the price the company actually realises per unit. |
| `SalesValue_atBasicRate` | `Selling_Price × Quantity` | Distributor revenue. Selling_Price (DSP) is the price the distributor pays the company. Always higher than CLP because it includes distributor margin. |
| `Selling_Rate_Per_PC_without_GST_CLP` | `MRP ÷ 1.12 ÷ 1.07 ÷ (1 + GST%)` | CLP per unit — same as the rate used in `Net_Amt`. |
| `Basic_Rate_Per_PC_without_GST` | `Selling_Price` (from raw data) | DSP per unit without GST. Used as `base_price` in the model. |
| `Basic_Rate_Per_PC` | `Selling_Price` (from raw data) | Same as above — retained as a fallback alias. |
| `Total_Scheme_Pct` | `Claimable_Pct + Non_Claimable_Pct + Other_Pct` | Total % discount for a sale line — sum of `%_Scheme` across all scheme reason rows for that invoice line. Used for all discount % calculations as `Σ(Total_Scheme_Pct × Qty) / Σ(Qty)`. |
| `Scheme_Discount` | Sum of claimable `Scheme_discount` ₹ rows | ₹ claimable discount for the invoice line. |
| `Staggered_qps` | Non-claimable + Other `Scheme_discount` ₹ rows | ₹ non-claimable / staggered QPS discount. |
| `TotalDiscount` | `Scheme_Discount + Staggered_qps` | Total ₹ discount for the invoice line. |
| `Claimable_Scheme_Pct` | Sum of `%_Scheme` where reason = Claimable | Claimable scheme % only. |
| `Non_Claimable_Scheme_Pct` | Sum of `%_Scheme` where reason starts with Non-Claimable | Non-claimable scheme % only. |
| `Final_State` | Derived from `Plant_Name` | MAH (Mumbai/Nagpur plants) or UP (Ghaziabad/Lucknow/Varanasi plants). |
| `Final_Outlet_Classification` | Mapped from raw `Outlet_Classification` string | OtherGT / WH / SS — see `OUTLET_CLASSIFICATION_MAPPING` in `build_deduped_sales_lines.py`. |
| `Sizes` | Derived from `SUb_Brand` | 12-ML or 18-ML — extracted by matching "12 ML" or "18 ML" in the sub-brand string. |

### Why Net_Amt and not SalesValue_atBasicRate for sales charts?

`SalesValue_atBasicRate` (DSP × qty) includes the distributor's margin — it is the distributor's
revenue, not the company's. `Net_Amt` (CLP × qty) is the company's actual revenue. For any
business-level sales charts or revenue computations in the tool, `Net_Amt` is the correct column.
`SalesValue_atBasicRate` is retained for base_price computation in the model (Step 3) because the
model needs the distributor-facing price as the reference for discount % calculations.

---

## Background — Why we switched data sources

The original tool (`backend/`) was built on **DMS sales data** stored as monthly parquet
files in a `DATA/` folder. That data was found to be unreliable.

The replacement source is the **scheme utilisation Excel files** provided monthly by the
business. These are first converted to parquet (Step 1 of the scheme utilisation flow),
then deduplicated and enriched (Step 2), and the Step 2 output is now what the main tool
reads. The scheme utilisation data is the authoritative record.

---

## Folder structure (scheme utilisation flow)

```
scheme utilisation flow/
  code repo/
    convert_raw_scheme_utilisation_to_parquet.py   # converts Excel → parquet (run first)
    build_deduped_sales_lines.py                   # deduplicates + enriches → final source
    validate_parquet_outputs.py                    # validation helper
  output/                                          # raw monthly parquets (25 files, Apr24–Mar26)
  final_source_data/                               # ← THE SOURCE THE MAIN TOOL NOW READS
    step2_deduped_sales_lines.parquet              # single enriched deduplicated file
    step2_deduped_sales_lines_report.json          # row counts, distributions, QC report
  testing/
    streamlit/
      app.py                                       # simple Streamlit validator for step 2 data
  TRANSITION_NOTES.md                              # this file
```

---

## What build_deduped_sales_lines.py does (Step 2 pipeline)

Reads all monthly raw parquet files from `output/`, deduplicates on
`Outlet_Id | Bill No | Batch_Id` as the unique sale-line key, pivots scheme discount rows
into Claimable / Non-Claimable / Other columns, and computes derived financials (CLP,
Net_Amt, SalesValue_atBasicRate, Scheme_Discount, Staggered_qps, TotalDiscount).

Default filters applied (all ON by default):
- Only `STOCKIEST DMS` distributor type
- Exclude schemes with "SAMT" in the name
- Keep only 12-ML and 18-ML pack sizes

### Changes made to build_deduped_sales_lines.py for this transition

**Change 1 — Distributor Type filter updated**

Original filter kept only `STOCKIEST DMS`. Updated to keep both `STOCKIEST DMS` and `STOCKIEST`
since both are valid distributor types across the data.

Excluded types and why:
- `SUPER STOCKIEST DMS` — only appears in Nov 2025 (393 outlets, 2,096 rows). Those outlets
  appear in no other month under any label. Treated as a one-off data anomaly, excluded.
- All other distributor types — not present in the data.

SAMT schemes also confirmed as Dec 2025 only (4,988 rows, scheme names starting with
"All India SAMT"). These are a distinct scheme category and are excluded by the
`exclude_samt=True` filter which remains unchanged.

**Change 2 — Outlet Classification mapping**

The raw data has detailed outlet type strings like `"GRO [Grocer Kirana]"`,
`"WH [Wholesale]"` etc. The main tool expects three groups: `OtherGT`, `WH`, `SS`.

Added `OUTLET_CLASSIFICATION_MAPPING` constant at the top of the file and applied it to
the `Final_Outlet_Classification` column instead of copying raw values directly.
Any value not in the mapping defaults to `OtherGT`.

Mapping:
```python
OUTLET_CLASSIFICATION_MAPPING = {
    "GRO [Grocer Kirana]": "OtherGT",
    "NOV [Novelty/ Cosmetics/Fancy store]": "OtherGT",
    "WH [Wholesale]": "WH",
    "CH [Chemist/Pharmacy]": "OtherGT",
    "GM [General Merchant]": "OtherGT",
    "Others PanBidi Cloths toy store": "OtherGT",
    "Self Service": "SS",
    "Saloon [Barber 2-3 chair]": "OtherGT",
    "Retailer": "OtherGT",
    "Beauty Shop": "OtherGT",
    "B": "OtherGT",
}
```

**Change 2 — Output written to final_source_data/**

The script is now run with explicit `--output` and `--report` flags pointing to
`final_source_data/` so the main tool's source of truth is kept separate from the
intermediate `output/` folder.

### How to re-run Step 2

```
cd "C:\Users\abqua\Desktop\HRI app\scheme utilisation flow"
python "code repo/build_deduped_sales_lines.py" ^
  --output "final_source_data/step2_deduped_sales_lines.parquet" ^
  --report "final_source_data/step2_deduped_sales_lines_report.json"
```

The script will NOT accidentally read the previous step 2 output as input — it explicitly
skips any file whose name starts with `step2_`.

### Output file stats (last run: 2026-05-25)

`final_source_data/step2_deduped_sales_lines.parquet`
- **1,189,572 rows** across 25 months (Apr 2024 – Mar 2026)
- 12-ML: 284,572 rows | 18-ML: 905,000 rows
- MAH: 827,753 rows | UP: 361,819 rows
- Final_Outlet_Classification: OtherGT: 1,117,706 | WH: 66,129 | SS: 5,737
- Validated: scheme % cross-checked against raw files for 6 transactions — all matched exactly

---

## Main tool migration status

The main tool lives in `backend/` (FastAPI) + `frontend/` (React). It has 6 steps.
Each step is being migrated one at a time to use the new data source.

### STEP 1 — RFM Segmentation ✅ DONE

**What Step 1 does**: Groups transactions by outlet, calculates Recency / Frequency /
Monetary metrics, assigns each outlet to one of 8 RFM segments
(e.g. Recent-High-High, Stale-Low-Low), and provides filters for State, Category,
Subcategory, Brand, Size, Outlet Classification.

**File changed**: `backend/services/core/data_loader.py`

**Change**: Added `scheme utilisation flow/final_source_data/` as the **first** entry in
`candidate_paths` inside `load_data()`. The backend now loads
`step2_deduped_sales_lines.parquet` as its single source file instead of the old monthly
DMS parquet files.

**Why no other changes were needed for Step 1**: All columns Step 1 requires are present
in the step 2 output with identical names. See column map below.

**Column map — Step 1 needs vs step 2 provides**:

| Column | In step 2? | Notes |
|--------|-----------|-------|
| Date | YES | Invoice date (datetime) |
| Outlet_ID | YES | String outlet identifier |
| Bill_No | YES | Used to count unique orders per outlet |
| Final_State | YES | MAH or UP |
| Final_Outlet_Classification | YES | Mapped to OtherGT / WH / SS |
| Category | YES | Value: STREAX |
| Subcategory | YES | STREAX INSTA SHAMPOO. STX HC SMART also present but filtered out by the data loader's scope filter |
| Brand | YES | Constant "Streax" |
| Sizes | YES | 12-ML or 18-ML |
| Quantity | YES | Units sold |
| Net_Amt | YES | Used for AOV (Monetary metric) |
| SalesValue_atBasicRate | YES | Used for market share % in segment summaries |
| Outlet_Type | NO | Not used by Step 1 — safe, data loader fills with NaN |
| Slab | NO | Not used by Step 1 — only needed from Step 2 (discount) onward |

**UI note**: The "Outlet Type(s)" dropdown in the frontend maps to `Final_Outlet_Classification`.
The scope builder (`backend/services/core/scope_builder.py`) internally collapses
SS → OtherGT (anything not WH becomes OtherGT), so SS outlets still appear under OtherGT
in the UI. This is correct behaviour.

**Verified**: Backend restarted and `/api/rfm/filters` returns correct values:
- states: [MAH, UP]
- subcategories: [STREAX INSTA SHAMPOO]
- sizes: [12-ML, 18-ML]
- outlet_classifications: [OtherGT, WH]

---

### STEP 2 — Discount Analysis ✅ DONE

**What Step 2 does**: User defines slab boundaries (quantity bins per outlet per month per size).
The backend assigns each outlet-month to a slab, then computes:
- Weighted discount % per slab: `Σ(Total_Scheme_Pct × Qty) / Σ(Qty)`
- Base discount calendar: round actual % to nearest 0.5pp, only accept step-up or step-down
  if the change is ≥ 1pp (`min_upward_jump_pp` / `min_downward_drop_pp` controls)
- Summary table per slab: outlets, invoices, quantity, AOV, AOQ, discount %

**File changed**: `backend/services/steps/step2_discount.py`

**Key change — discount formula replaced**

The old code derived discount % from ₹ amounts divided by DSP (Basic_Rate_Per_PC), which
required a reliable DSP column. The scheme utilisation data already has `Total_Scheme_Pct`
computed in the Step 2 pipeline (= Claimable + Non-Claimable + Other scheme %).

New primary path in `_prepare_step2_discount_basis()`:
```python
if 'Total_Scheme_Pct' in work.columns:
    pct = pd.to_numeric(work['Total_Scheme_Pct'], errors='coerce').fillna(0.0)
    work['_step2_weighted_disc'] = pct * qty   # numerator: pct × qty
    work['_step2_qty_denom'] = qty             # denominator: qty
    # ₹ summary still uses Scheme_Discount + Staggered_qps if available
    if colmap.get("scheme_col") and colmap.get("qps_col"):
        work['_step2_scheme_amount'] = scheme + qps
    else:
        work['_step2_scheme_amount'] = (pct / 100.0) * qty
```

DSP-based fallback path is preserved for any future data source that lacks `Total_Scheme_Pct`.

**Aggregation fix** — `_build_summary_by_slab`, `_compute_base_depth_result`, weekly/monthly
re-aggregation blocks: all now compute weighted discount as:
```python
'Discount_Pct' = _weighted_disc.sum() / _qty_denom.sum()
```
instead of the old `Total_Discount / Sales_Value` ratio.

**Column dependency** — `data_loader.py` `keep_cols` was updated to include `Total_Scheme_Pct`
so the column survives normalization and reaches Step 2.

**Column map — Step 2 needs vs step 2 parquet provides**:

| Column | In step 2 parquet? | Notes |
|--------|-------------------|-------|
| Total_Scheme_Pct | YES | = Claimable + Non-Claimable + Other scheme % per transaction |
| Scheme_Discount | YES | Claimable ₹ discount |
| Staggered_qps | YES | Non-Claimable + Other ₹ discount |
| Quantity | YES | Units for weighting |
| Date | YES | For monthly slab assignment |
| Outlet_ID | YES | Grouping key |
| Sizes | YES | 12-ML / 18-ML — slabs are per-size |
| Bill_No | YES | Invoice count |
| Slab | NO | Assigned by backend at query time from user-defined boundaries |

**Verified**: Weighted discount % cross-checked against reference Excel
(`final_presentation/scheme_utilisation/scheme_utilisation_all_months.xlsx`) for all 25 months,
both sizes, all slabs — all matched exactly.

---

### STEP 3 — Modeling & ROI ⚠️ IN PROGRESS

**What Step 3 does**: Fits a two-stage regression model per size per slab over the 25-month
history, then computes structural ROI for each episode where the base discount stepped up.

---

#### Two-stage model structure

**Stage 1** — removes the store-count signal from discount:
```
LinearRegression: actual_discount_pct → store_count
residual_store = actual_store_count − predicted_store_count
```
This isolates the outlet-count variation that is NOT explained by the discount level.

**Stage 2** — predicts quantity from structural drivers:
```
CustomConstrainedRidge:
  quantity ~ residual_store
           + base_discount_pct          (structural level — positive coef constrained)
           + lag1_base_discount_pct     (prior month base — negative coef constrained)
           + other_slabs_weighted_base_discount_pct  (cross-slab weighted base — negative)
```
Tactical discount (actual − base) is NOT a feature in the new strategy model.

**Slab assignment**: `Outlet_ID × Month × Size` monthly quantity → cut into bins defined
by user-set thresholds. One outlet can move between slabs across months.

**Base discount**: actual weighted % → rounded to 0.5pp steps → step-up/step-down controlled
by `min_upward_jump_pp` / `min_downward_drop_pp` settings.

---

#### Discount % formula (fixed for new data source)

`actual_discount_pct` in `_build_monthly_model_dataframe` now uses:
```python
Σ(_step2_weighted_disc) / Σ(_step2_qty_denom)
= Σ(Total_Scheme_Pct × Qty) / Σ(Qty)   # quantity-weighted average
```
Previously used `Σ(₹ TotalDiscount) / Σ(₹ DSP sales) × 100` — changed for consistency
with Step 2. File: `backend/services/steps/step3_modeling.py`, function
`_build_monthly_model_dataframe`, 3 aggregation blocks.

---

#### Structural ROI — current implementation (as of 2026-05-25)

**File**: `backend/services/steps/step3_modeling.py` → `_build_structural_roi_points()`

**Step 1 — Identify regimes**:
The 25-month base discount series is rounded to 0.5pp and broken into flat blocks
(regimes). Each time the base steps UP, that is an "episode". Step-downs and flat periods
are skipped.

Example: Mar=24%, Apr=23%, May=26%, Jun=28%
```
Mar→Apr: 24%→23% = step-DOWN  → SKIP
Apr→May: 23%→26% = +3pp       → EPISODE 1  (step_up=3)
May→Jun: 26%→28% = +2pp       → EPISODE 2  (step_up=2)
```

**Step 2 — Counterfactual quantity prediction** for each episode month:
```python
# World A: base stayed at prev_base
qty_prev = model(base=prev_base, lag=prev_base,  tactical=0)

# World B: base moved to curr_base
qty_curr = model(base=curr_base, lag=prev_base,  tactical=0)
#                               ↑ FIRST month of episode: lag = prev_base
#                                 subsequent months: lag = curr_base
```
Tactical term is forced to 0 in both worlds — this isolates the structural (base) effect.

**Step 3 — Revenue, spend, ROI equations**:
```
baseline_price     = base_price × (1 − prev_base / 100)
current_price      = base_price × (1 − prev_base / 100)   ← uses prev_base, NOT curr_base

incremental_revenue = (qty_curr − qty_prev) × current_price
spend               = base_price × (step_up / 100) × qty_curr
ROI (Topline)       = incremental_revenue / spend
```

For profit ROI, COGS is deducted:
```
incremental_profit = incremental_revenue − cogs_per_unit × (qty_curr − qty_prev)
Profit ROI         = incremental_profit / spend
```

**Concrete example — Episode 1 (Apr 23% → May 26%, step_up=3pp)**:
```
qty_prev = model(base=23%, lag=23%, tactical=0)
qty_curr = model(base=26%, lag=23%, tactical=0)  ← lag=23% not 26%

Δqty = coef_base × 3  +  coef_lag × (23−23)
     = 154,398 × 3    +  (−16,901) × 0
     = 463,194 units   (lag term = 0 for first month of episode)

baseline_price = current_price = P × 0.77  (both use prev_base=23%)

incremental_revenue = 463,194 × 0.77P
spend               = P × 0.03 × qty_curr
ROI                 = (463,194 × 0.77) / (0.03 × qty_curr)
```

**Why the numbers look high (11x–14x)**:

The formula simplifies to:
```
ROI ≈ (Δqty / qty_curr) × (1 − prev_base%) / step_up%
```
The ratio `(1 − prev_base%) / step_up%` = `(1 − 0.23) / 0.03` = **25.7×**.
Even a 40% quantity lift gives ROI = 25.7 × 0.40 = **10.3×**.

Two structural reasons ROI is overstated:
1. **Trade loading**: The first-month quantity spike includes forward buying by outlets
   stocking up on the new deal. The June hangover (outlets don't reorder because they
   have inventory) is NOT counted in the episode — only the step-up month is measured.
2. **Marginal spend denominator**: Spend = only the EXTRA discount from the step-up
   (step_up% × qty). The total discount cost in the new regime is much higher but
   ignored.

**Trade loading detail**:
The lag coefficient (−16,901) IS the model's learned forward-buying signal. At steady
state after a 3pp step-up the net quantity lift is:
```
Δqty_steady = (coef_base + coef_lag) × step_up
            = (154,398 + (−16,901)) × 3
            = 137,497 × 3 = 412,491 units
```
vs the first-month bump of 463,194. Difference = 50,703 units that are "borrowed" from
future months. The current ROI counts 463K, the correct steady-state ROI would count 412K.

---

#### Proposed ROI fix (NOT yet applied — decision pending)

Two possible corrections:

**Option A — Steady-state qty lift (fixes trade loading)**:
Replace first-month qty prediction with steady-state prediction (where lag = curr_base
in both worlds):
```
Δqty_steady = (coef_base + coef_lag) × step_up
```
This deducts the forward-buying effect.

**Option B — Full cost denominator (fixes marginal spend)**:
Replace marginal spend with total incremental discount cost:
```
incremental_spend = base_price × curr_base% × qty_curr
                  − base_price × prev_base% × qty_prev
```
This charges the full cost of moving to a higher discount regime.

**Option C — Both A and B together** (most conservative, arguably most accurate).

Current status: formula documented, fix not applied. Discuss with business before changing
as it will significantly lower the displayed ROI numbers.

---

### STEP 4 — Cross-Size Scenario Planner 🔲 TODO
### STEP 5 — AI Scenario Generation + Baseline Forecast 🔲 TODO
### STEP 6 — EDA & Validation 🔲 TODO
