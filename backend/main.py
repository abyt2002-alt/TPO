from fastapi import FastAPI, HTTPException, Response, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from typing import List, Dict
import json

from services.rfm_service import RFMService
from models.rfm_models import (
    RFMRequest, RFMResponse,
    BaseDepthRequest, BaseDepthResponse,
    DiscountOptionsRequest, DiscountOptionsResponse,
    RunCreateRequest, RunCreateResponse,
    RunStateUpdateRequest, RunStateResponse,
    ModelingRequest, ModelingResponse,
    PlannerRequest, PlannerResponse, PlannerScenarioComparisonResponse,
    EDARequest, EDAResponse, EDAOptionsResponse
)

app = FastAPI(title="Trade Promo Optimization API", version="1.0.0")

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow local-network frontend hosts
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
rfm_service = RFMService()


def _model_to_dict(model_obj):
    if hasattr(model_obj, "model_dump"):
        return model_obj.model_dump(exclude_none=True)
    return model_obj.dict(exclude_none=True)

@app.get("/")
async def root():
    return {"message": "Trade Promo Optimization API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# RFM Endpoints
@app.post("/api/rfm/calculate", response_model=RFMResponse)
async def calculate_rfm(request: RFMRequest):
    """Calculate RFM metrics for outlets"""
    try:
        result = await rfm_service.calculate_rfm(request)
        if request.run_id:
            step1_filters = {
                "states": request.states or [],
                "categories": request.categories or [],
                "subcategories": request.subcategories or [],
                "brands": request.brands or [],
                "sizes": request.sizes or [],
                "recency_threshold": request.recency_threshold,
                "frequency_threshold": request.frequency_threshold,
            }
            table_query = {
                "page": request.page,
                "page_size": request.page_size,
                "search": request.search or "",
                "sort_key": request.sort_key or "total_net_amt",
                "sort_direction": request.sort_direction or "desc",
            }
            rfm_service.save_run_state(
                run_id=request.run_id,
                state_update={
                    "active_step": "step1",
                    "filters": step1_filters,
                    "last_calculated_filters": step1_filters,
                    "table_query": table_query,
                },
                step1_result=jsonable_encoder(result),
            )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/rfm/export")
async def export_rfm(request: RFMRequest):
    """Export full outlet dataset as CSV for selected filters"""
    try:
        csv_content = await rfm_service.export_rfm_csv(request)
        if not csv_content:
            raise HTTPException(status_code=404, detail="No data matches the selected filters")
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=rfm_outlets.csv"}
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/discount/base-depth", response_model=BaseDepthResponse)
async def estimate_base_depth(request: BaseDepthRequest):
    """Estimate base discount depth for selected filters."""
    try:
        result = await rfm_service.calculate_base_depth(request)
        if request.run_id:
            step1_filters = {
                "states": request.states or [],
                "categories": request.categories or [],
                "subcategories": request.subcategories or [],
                "brands": request.brands or [],
                "sizes": request.sizes or [],
                "recency_threshold": request.recency_threshold,
                "frequency_threshold": request.frequency_threshold,
            }
            base_depth_config = {
                "time_aggregation": request.time_aggregation,
                "rolling_window_periods": request.rolling_window_periods,
                "quantile": request.quantile,
                "round_step": request.round_step,
                "min_upward_jump_pp": request.min_upward_jump_pp,
                "min_downward_drop_pp": request.min_downward_drop_pp,
            }
            step2_filters = {
                "rfm_segments": request.rfm_segments or [],
                "outlet_classifications": request.outlet_classifications or [],
                "slabs": request.slabs or [],
            }
            rfm_service.save_run_state(
                run_id=request.run_id,
                state_update={
                    "active_step": "step2",
                    "last_calculated_filters": step1_filters,
                    "base_depth_config": base_depth_config,
                    "step2_filters": step2_filters,
                },
                step2_result=jsonable_encoder(result),
            )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/discount/options", response_model=DiscountOptionsResponse)
async def get_discount_options(request: DiscountOptionsRequest):
    """Get step-2 discount analysis filter options."""
    try:
        return await rfm_service.get_discount_options(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/discount/modeling", response_model=ModelingResponse)
async def run_modeling(request: ModelingRequest):
    """Run Step 3 modeling and ROI on slab-level monthly data."""
    try:
        result = await rfm_service.calculate_modeling(request)
        if request.run_id:
            rfm_service.save_run_state(
                run_id=request.run_id,
                state_update={
                    "active_step": "step3",
                    "step3_result": jsonable_encoder(result),
                },
            )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/planner/12-month", response_model=PlannerResponse)
async def run_12_month_planner(request: PlannerRequest):
    """Run Step 4: 12-month planner for a selected slab."""
    try:
        result = await rfm_service.calculate_12_month_planner(request)
        if request.run_id:
            rfm_service.save_run_state(
                run_id=request.run_id,
                state_update={
                    "active_step": "step4",
                    "step4_result": jsonable_encoder(result),
                },
            )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/planner/scenario-compare", response_model=PlannerScenarioComparisonResponse)
async def compare_planner_scenarios(
    payload_json: str = Form(...),
    file: UploadFile = File(...),
):
    """Run Step 5: scenario comparison by uploading CSV/XLSX structural plans."""
    try:
        payload = json.loads(payload_json)
        request = PlannerRequest(**(payload or {}))
        file_bytes = await file.read()
        result = await rfm_service.compare_planner_scenarios_from_upload(
            request=request,
            filename=str(file.filename or ""),
            file_bytes=file_bytes,
        )
        if request.run_id:
            rfm_service.save_run_state(
                run_id=request.run_id,
                state_update={
                    "active_step": "step5",
                    "step5_result": jsonable_encoder(result),
                },
            )
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/eda/options", response_model=EDAOptionsResponse)
async def get_eda_options(request: EDARequest):
    """Get Step 5 EDA dropdown options for products/outlet classes."""
    try:
        return await rfm_service.get_eda_options(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/eda/overview", response_model=EDAResponse)
async def get_eda_overview(request: EDARequest):
    """Run Step 5 EDA aggregation views."""
    try:
        result = await rfm_service.get_eda_overview(request)
        if request.run_id:
            rfm_service.save_run_state(
                run_id=request.run_id,
                state_update={
                    "active_step": "step5",
                    "step5_result": jsonable_encoder(result),
                },
            )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/runs/create", response_model=RunCreateResponse)
async def create_run(request: RunCreateRequest = RunCreateRequest()):
    try:
        run_id = rfm_service.create_run(request.run_id)
        return RunCreateResponse(success=True, message="Run initialized", run_id=run_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/runs/{run_id}/state", response_model=RunStateResponse)
async def get_run_state(run_id: str):
    try:
        state = rfm_service.get_run_state(run_id)
        if state is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return RunStateResponse(
            success=True,
            message="Run state loaded",
            run_id=run_id,
            state=state.get("state") or {},
            step1_result=state.get("step1_result"),
            step2_result=state.get("step2_result"),
            step3_result=(state.get("state") or {}).get("step3_result"),
            step4_result=(state.get("state") or {}).get("step4_result"),
            step5_result=(state.get("state") or {}).get("step5_result"),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/runs/{run_id}/state", response_model=RunStateResponse)
async def save_run_state(run_id: str, request: RunStateUpdateRequest):
    try:
        state_update = _model_to_dict(request)
        rfm_service.save_run_state(run_id=run_id, state_update=state_update)
        state = rfm_service.get_run_state(run_id)
        return RunStateResponse(
            success=True,
            message="Run state saved",
            run_id=run_id,
            state=state.get("state") if state else {},
            step1_result=state.get("step1_result") if state else None,
            step2_result=state.get("step2_result") if state else None,
            step3_result=((state.get("state") or {}).get("step3_result") if state else None),
            step4_result=((state.get("state") or {}).get("step4_result") if state else None),
            step5_result=((state.get("state") or {}).get("step5_result") if state else None),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/rfm/filters")
async def get_available_filters():
    """Get available filter options for RFM calculation"""
    try:
        filters = await rfm_service.get_available_filters()
        return filters
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/rfm/filters/cascade")
async def get_cascading_filters(filters: Dict[str, List[str]]):
    """Get cascading filter options based on current selections"""
    try:
        if rfm_service.data_cache is None:
            return await rfm_service.get_available_filters()

        df_all = rfm_service.data_cache
        states = filters.get("states") or []
        categories = filters.get("categories") or []
        subcategories = filters.get("subcategories") or []
        brands = filters.get("brands") or []

        df_for_categories = df_all[df_all["Final_State"].isin(states)] if states else df_all

        df_for_subcategories = df_for_categories
        if categories:
            df_for_subcategories = df_for_subcategories[df_for_subcategories["Category"].isin(categories)]

        df_for_brands = df_for_subcategories
        if subcategories:
            df_for_brands = df_for_brands[df_for_brands["Subcategory"].isin(subcategories)]

        df_for_sizes = df_for_brands
        if brands:
            df_for_sizes = df_for_sizes[df_for_sizes["Brand"].isin(brands)]

        return {
            "states": sorted(df_all["Final_State"].dropna().unique().tolist()),
            "categories": sorted(df_for_categories["Category"].dropna().unique().tolist()),
            "subcategories": sorted(df_for_subcategories["Subcategory"].dropna().unique().tolist()),
            "brands": sorted(df_for_brands["Brand"].dropna().unique().tolist()),
            "sizes": sorted(df_for_sizes["Sizes"].dropna().unique().tolist()),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/rfm/segments")
async def get_rfm_segments():
    """Get all RFM segment definitions"""
    return {
        "segments": [
            "Recent-High-High",
            "Recent-High-Low",
            "Recent-Low-High",
            "Recent-Low-Low",
            "Stale-High-High",
            "Stale-High-Low",
            "Stale-Low-High",
            "Stale-Low-Low"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
