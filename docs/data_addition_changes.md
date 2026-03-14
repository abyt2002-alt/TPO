# Data Addition Change Log (Dec 2025, Jan 2026, Feb 2026)

## Why new months were not appearing
- New parquet files were added, but key columns were lowercase in those files:
  - `subcategory`, `sizes`, `brand`
- App logic filters using canonical names:
  - `Subcategory`, `Sizes`, `Brand`
- Size values in new files were lowercase (`12-ml`, `18-ml`), while filters expected normalized values (`12-ML`, `18-ML`).
- Result: Dec/Jan/Feb rows were loaded from disk but dropped during filtering.

## Backend file changed
- `backend/services/core/data_loader.py`

## What had to change

### 1) Column alias normalization
- Added alias mapping so mixed-schema files are standardized during load:
  - `subcategory -> Subcategory`
  - `sizes -> Sizes`
  - `brand -> Brand`
  - `category -> Category`
  - `bill_no -> Bill_No`
  - `invoice_no -> Invoice_No`
  - `sales_order_no -> Sales_Order_No`
  - `sku_code -> Sku_Code`
  - `sku_name -> Sku_Name`
  - `mrp -> MRP`
  - and related state/outlet aliases.

### 2) Text/value normalization
- Standardized text dimensions to uppercase/trimmed values.
- Standardized `Sizes` to uppercase and no spaces.
- This ensures `12-ml` and `18-ml` map to `12-ML` and `18-ML`.

### 3) Stable schema before merge
- Reindexed every file to one fixed set of required columns before combining.
- Added derivations for mandatory fields when absent:
  - `Bill_No` from `Invoice_No`/`Sales_Order_No`
  - `Net_Amt` from `SalesValue_atBasicRate - TotalDiscount`

### 4) Low-memory safe loading
- Reduced loaded columns to only app-required fields.
- Downcasted numeric columns (`Quantity`, `Net_Amt`, `SalesValue_atBasicRate`, `TotalDiscount`) to `float32`.
- Replaced direct large-frame concat with memory-stable column-wise combine to avoid pandas consolidation crashes.

### 5) Scope-safe preload (current app usage)
- Loader now keeps only the tool scope rows:
  - Subcategory in `{STX INSTA SHAMPOO, STREAX INSTA SHAMPOO}`
  - Sizes in `{12-ML, 18-ML}`
- This keeps startup fast/stable and avoids out-of-memory failures on full raw dumps.

## Validation after fix
- Backend `/api/rfm/calculate` now returns:
  - `max_date = 2026-02-28`
- Included month tails for scoped data:
  - `2025-12`, `2026-01`, `2026-02` now present.

## Operational note for future monthly drops
- You can keep adding monthly files in `DATA/`.
- Mixed case column names and size strings are now handled automatically.
- After adding files:
  1. restart backend
  2. refresh frontend
  3. run Step 1 once to rebuild run state from latest data.
