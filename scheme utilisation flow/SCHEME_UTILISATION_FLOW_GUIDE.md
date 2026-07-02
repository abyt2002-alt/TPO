# Scheme Utilisation Data Flow Guide

This guide captures the confirmed mapping and calculation rules for replacing the old `DATA` sales files with the new monthly scheme utilisation files.

## Source Data

- Raw Excel folder: `final_presentation/Raw files`
- Converted parquet folder: `scheme utilisation flow/output`
- Current raw coverage: 25 monthly files, April 2024 through April 2026.
- Converted output coverage: 25 parquet files.
- Validation status:
  - Raw rows: `2,117,512`
  - Parquet rows: `2,117,512`
  - Raw and parquet row counts match.
  - Raw and parquet column counts match.
  - Source file names and source row ranges match.

The existing notebook used for reference is:

`final_presentation/scheme_utilisation/code repo/scheme_utilisation_all_months.ipynb`

## Raw Row Grain

The raw file row is a scheme utilisation row, not a clean sale transaction row.

A single sale line can appear multiple times because primary and secondary scheme rows are stored separately. Therefore, do not sum raw row quantity directly.

Use this key to identify one unique sale line:

```text
Outlet_Id + Bill No + Batch_Id
```

Full-data check:

```text
Raw rows: 2,117,512
Unique sale lines by Outlet_Id + Bill No + Batch_Id: 1,201,463
Raw quantity sum: 82,535,854
Quantity counted once per sale line: 42,843,362
Raw quantity overcount factor: 1.93x
```

Quantity is consistent inside duplicate sale-line groups, so the transaction quantity should be taken once using `first` or `max`.

## Confirmed Business Mappings

### State

Derive state from `Plant_Name`.

```text
HRIPL - MUMBAI       -> MAH
HRIPL - NAGPUR       -> MAH
HRIPL - GHAZIABAD-RT -> UP
HRIPL - LUCKNOW - RT -> UP
HRIPL - VARANASI     -> UP
```

These are the only `Plant_Name` values found in the full converted data.

### Product Fields

```text
Category    -> Category
Subcategory -> Subcategory
Brand       -> constant "Streax" for now
Sizes       -> derive from SUb_Brand
MRP         -> MRP
```

Size should be identified from `SUb_Brand`, not from `Scheme Name`.

Use this normalization:

```text
SUb_Brand contains 12 ML -> 12-ML
SUb_Brand contains 18 ML -> 18-ML
```

The older notebook also identified pack size from `SUb_Brand`.

### Transaction And Outlet Fields

```text
Invoice Date        -> Date
Outlet_Id           -> Outlet_ID
Bill No             -> Bill_No
Invoice qty. pieces -> Quantity
Outlet_Classification -> Final_Outlet_Classification
skucode             -> Sku_Code
SKU Name            -> Sku_Name
```

### Rate And Revenue Fields

`Selling_Price` is the DSP/basic rate.

CLP is calculated from `MRP` and `gst`:

```text
CLP = MRP / 1.12 / 1.07 / (1 + gst / 100)
```

Revenue mappings:

```text
Distributor revenue = Selling_Price * Quantity
Company realised revenue = CLP * Quantity
Net_Amt = Company realised revenue
SalesValue_atBasicRate = Distributor revenue
Selling_Rate_Per_PC_without_GST_CLP = CLP
```

COGS remains a frontend input and does not come from the source file.

## Scheme Rows

Confirmed scheme reason mapping:

```text
Claimable                  -> primary scheme
Non-Claimable_*            -> secondary / QPS scheme
```

For each unique sale line:

```text
total_scheme_percent = sum(%_Scheme)
Scheme_Discount = sum(Scheme_discount where Scheme_Reason == Claimable)
Staggered_qps = sum(Scheme_discount where Scheme_Reason starts with Non-Claimable)
TotalDiscount = Scheme_Discount + Staggered_qps
```

The old notebook summed both `Claimable` and `Non-Claimable` percentages into one sale-line discount percentage before slab-level aggregation.

## Filters From Existing Scheme Utilisation Notebook

The reference notebook applies these filters before building slab summaries:

```text
Distributor_Type == STOCKIEST DMS
Pack is 12 ML or 18 ML from SUb_Brand
Scheme Name does not contain SAMT
```

These filters should be preserved unless the app needs a wider scope.

## Slab Assignment

Slab is assigned per outlet per month, after deduplicating sale lines.

Flow:

```text
Raw scheme rows
-> filter eligible rows
-> group to one row per sale line
-> count Quantity once per sale line
-> sum monthly quantity by outlet and size
-> assign one slab to the outlet for that month and size
-> all sale lines for that outlet-month-size inherit that slab
```

Reference notebook slab definitions:

```text
12-ML:
  Slab 1: 8 <= outlet monthly quantity < 144
  Slab 2: outlet monthly quantity >= 144

18-ML:
  Slab 1: 8 <= outlet monthly quantity < 32
  Slab 2: 32 <= outlet monthly quantity < 576
  Slab 3: 576 <= outlet monthly quantity < 960
  Slab 4: outlet monthly quantity >= 960
```

For the main app, these slab cutoffs can still come from the frontend. The important rule is that slab assignment must use deduplicated monthly outlet quantity.

## Slab-Level Scheme Percent Calculation

The reference notebook calculates each slab's scheme percentage as a quantity-weighted average of sale-line scheme percentages.

This is the agreed method for the main app as well.

Per sale line:

```text
line_scheme_percent = sum(%_Scheme) across all scheme rows for that sale-line key
```

Per slab:

```text
slab_total_discount_percent =
    sum(line_scheme_percent * line_quantity) / sum(line_quantity)
```

Full agreed flow:

```text
1. Deduplicate raw scheme rows into sale lines.

2. For each sale line:
   transaction_scheme_percent = sum(%_Scheme)
   transaction_quantity = Invoice qty. pieces counted once

3. For each month + outlet + size:
   outlet_month_size_quantity = sum(transaction_quantity)

4. Apply the frontend slab cutoffs for that product size.
   The outlet-month-size is assigned to exactly one slab.

5. All sale lines for that outlet-month-size inherit that slab.

6. For each slab:
   required_slab_scheme_percent =
       sum(transaction_scheme_percent * transaction_quantity)
       / sum(transaction_quantity)
```

Important: do not average raw scheme rows and do not sum raw row quantity. The weighting must happen only after raw rows have been deduplicated into sale lines.

## Canonical App Output Row

After transformation, the app should consume one row per deduplicated sale line with these canonical fields:

```text
Date
Outlet_ID
Bill_No
Final_State
State
Final_Outlet_Classification
Category
Subcategory
Brand
Sizes
Slab
Quantity
MRP
Net_Amt
SalesValue_atBasicRate
TotalDiscount
Scheme_Discount
Staggered_qps
Basic_Rate_Per_PC_without_GST
Basic_Rate_Per_PC
Selling_Rate_Per_PC_without_GST_CLP
Sku_Code
Sku_Name
```

Recommended mapping for basic rate fields:

```text
Basic_Rate_Per_PC_without_GST = Selling_Price
Basic_Rate_Per_PC = Selling_Price
```

If `Selling_Price` is later confirmed to include GST, split these two fields properly.

## Step 2 Deduped Sale-Line Script

Script:

```text
scheme utilisation flow/code repo/build_deduped_sales_lines.py
```

Default input:

```text
scheme utilisation flow/output/*.parquet
```

Default outputs:

```text
scheme utilisation flow/output/step2_deduped_sales_lines.parquet
scheme utilisation flow/output/step2_deduped_sales_lines_report.json
```

The script creates one row per sale line using:

```text
Outlet_Id + Bill No + Batch_Id
```

It keeps non-additive transaction fields once:

```text
Quantity
Sale_Value
Distributor_Sale_Value
MRP
Selling_Price
gst
Product and outlet dimensions
```

It pivots additive scheme fields into separate claimable and non-claimable columns:

```text
Claimable_Scheme_Pct
Non_Claimable_Scheme_Pct
Total_Scheme_Pct
Claimable_Scheme_Discount
Non_Claimable_Scheme_Discount
Scheme_Discount
Staggered_qps
TotalDiscount
```

Default filters match the old scheme utilisation notebook:

```text
Distributor_Type == STOCKIEST DMS
Scheme Name does not contain SAMT
Sizes are 12-ML or 18-ML from SUb_Brand
```

## Remaining Implementation Checks

- Inspect the few rows where `SUb_Brand` does not clearly map to `12-ML` or `18-ML`.
- Confirm whether `Distributor_Type == STOCKIEST DMS` should be mandatory for the app source, as it was in the notebook.
- Confirm whether `SAMT` schemes should always be excluded in the app source, as they were in the notebook.
- Keep an audit column for `total_scheme_percent` even if the existing app does not currently require it.
