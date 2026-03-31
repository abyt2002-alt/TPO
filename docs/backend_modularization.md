# Backend Modularization (Step/Core Split)

## What changed

`backend/services/rfm_service.py` is now a thin compatibility facade.  
It keeps the same `RFMService` class and public methods used by `backend/main.py`, but delegates logic to mixins from:

- `backend/services/core/`
- `backend/services/steps/`

## Core modules

- `core/context.py`: shared runtime context and dependency initialization
- `core/state_store.py`: run-state SQLite persistence
- `core/data_loader.py`: parquet loading/caching
- `core/scope_builder.py`: shared filter/scope/slab normalization
- `core/shared_math.py`: common math helpers and constrained ridge primitives
- `core/scenario_compare.py`: scenario upload parsing/comparison
- `core/eda_service.py`: EDA option/scope/aggregation

## Step modules

- `steps/step1_segmentation.py`
- `steps/step2_discount.py`
- `steps/step3_modeling.py`
- `steps/step4_cross_size_planner.py`
- `steps/step5_baseline_forecast.py`

## Compatibility guarantees

- API routes in `backend/main.py` unchanged
- Pydantic models in `backend/models/rfm_models.py` unchanged
- Request/response schemas unchanged
- Formulas and business outputs intended unchanged (method-body extraction split)

## Snapshot parity harness

Location: `backend/tests/parity/`

- `requests.json`: fixed endpoint payloads
- `run_snapshot_parity.py --mode capture`: write `snapshots.json`
- `run_snapshot_parity.py --mode compare`: strict compare with:
  - exact key matching
  - exact list length/order
  - float tolerance `1e-9`
  - exact datetime string equality
