# EDA Tool

Standalone Streamlit EDA app for parquet data in `DATA/`.

## Features
- Filters: `Month`, `Pack Size`, `Slab`, `State`, `Channel`
- Absolute values: `Sales`, `Volume`
- Growth rates: `Sales Growth`, `Volume Growth` (latest MoM)
- Trend lines:
  - Sales + Volume
  - Discount Level %
  - Base Price
  - MRP
- Download monthly metrics CSV

## Run
From project root:

```powershell
streamlit run eda/app.py
```

