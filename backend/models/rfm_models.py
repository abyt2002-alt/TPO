from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class RFMRequest(BaseModel):
    """Request model for RFM calculation"""
    run_id: Optional[str] = None
    states: Optional[List[str]] = None
    categories: Optional[List[str]] = None
    subcategories: Optional[List[str]] = None
    brands: Optional[List[str]] = None
    sizes: Optional[List[str]] = None
    recency_threshold: int = Field(default=90, ge=30, le=180)
    frequency_threshold: int = Field(default=20, ge=1, le=100)
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=500)
    search: Optional[str] = None
    sort_key: Optional[str] = "total_net_amt"
    sort_direction: Optional[str] = "desc"
    
    class Config:
        json_schema_extra = {
            "example": {
                "states": ["MAH", "UP"],
                "categories": [],
                "subcategories": [],
                "brands": [],
                "sizes": [],
                "recency_threshold": 90,
                "frequency_threshold": 20,
                "page": 1,
                "page_size": 20,
                "search": "",
                "sort_key": "total_net_amt",
                "sort_direction": "desc"
            }
        }

class OutletRFM(BaseModel):
    """Individual outlet RFM data"""
    outlet_id: str
    final_state: str
    first_order: datetime
    last_order: datetime
    unique_order_days: int
    orders_count: int
    aov: float
    recency_days: int
    recency_flag: int
    r_label: str
    active_days: int
    orders_per_day: float
    f_label: str
    f_cluster_id: int
    m_label: str
    m_cluster_id: Optional[float]
    rfm_segment: str
    total_net_amt: float

class SegmentSummary(BaseModel):
    """Summary statistics for an RFM segment"""
    segment: str
    total_outlets: int
    percentage: float
    mah_count: int
    up_count: int
    avg_order_days: float
    avg_aov: float
    avg_recency: float
    market_share: float

class ClusterSummary(BaseModel):
    """Cluster range summary"""
    frequency: List[Dict[str, Any]]
    monetary: List[Dict[str, Any]]

class RFMResponse(BaseModel):
    """Response model for RFM calculation"""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    rfm_data: Optional[List[OutletRFM]] = None
    segment_summary: Optional[List[SegmentSummary]] = None
    cluster_summary: Optional[ClusterSummary] = None
    max_date: Optional[datetime] = None
    total_outlets: int = 0
    total_filtered_outlets: int = 0
    total_pages: int = 0
    page: int = 1
    page_size: int = 20
    input_rows: int = 0
    input_outlets: int = 0
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "RFM calculation completed successfully",
                "total_outlets": 150,
                "input_rows": 50000,
                "input_outlets": 150
            }
        }


class BaseDepthRequest(BaseModel):
    """Request model for base depth estimation"""
    run_id: Optional[str] = None
    states: Optional[List[str]] = None
    categories: Optional[List[str]] = None
    subcategories: Optional[List[str]] = None
    brands: Optional[List[str]] = None
    sizes: Optional[List[str]] = None
    recency_threshold: int = Field(default=90, ge=30, le=180)
    frequency_threshold: int = Field(default=20, ge=1, le=100)
    time_aggregation: str = Field(default="D", pattern="^(D|W|M)$")
    rolling_window_periods: int = Field(default=10, ge=1, le=90)
    quantile: float = Field(default=0.5, ge=0.0, le=1.0)
    round_step: float = Field(default=0.5, gt=0.0, le=10.0)
    min_upward_jump_pp: float = Field(default=1.0, ge=0.0, le=5.0)
    min_downward_drop_pp: float = Field(default=1.0, ge=0.0, le=5.0)
    rfm_segments: Optional[List[str]] = None
    outlet_classifications: Optional[List[str]] = None
    slabs: Optional[List[str]] = None
    outlet_ids: Optional[List[str]] = None


class BaseDepthPoint(BaseModel):
    period: datetime
    actual_discount_pct: float
    base_discount_pct: float
    orders: int
    quantity: float
    sales_value: float


class BaseDepthResponse(BaseModel):
    success: bool
    message: str
    points: List[BaseDepthPoint] = []
    summary: Dict[str, float] = {}
    slab_results: List[Dict[str, Any]] = []


class DiscountOptionsRequest(BaseModel):
    """Request model for step-2 discount filters options."""
    run_id: Optional[str] = None
    states: Optional[List[str]] = None
    categories: Optional[List[str]] = None
    subcategories: Optional[List[str]] = None
    brands: Optional[List[str]] = None
    sizes: Optional[List[str]] = None
    recency_threshold: int = Field(default=90, ge=30, le=180)
    frequency_threshold: int = Field(default=20, ge=1, le=100)
    rfm_segments: Optional[List[str]] = None
    outlet_classifications: Optional[List[str]] = None
    slabs: Optional[List[str]] = None


class DiscountOptionsResponse(BaseModel):
    success: bool
    message: str
    rfm_segments: List[str] = []
    outlet_classifications: List[str] = []
    slabs: List[str] = []
    matching_outlets: int = 0


class RunCreateRequest(BaseModel):
    run_id: Optional[str] = None


class RunCreateResponse(BaseModel):
    success: bool
    message: str
    run_id: str


class RunStateUpdateRequest(BaseModel):
    active_step: Optional[str] = Field(default=None, pattern="^(step1|step2|step3|step4)$")
    filters: Optional[Dict[str, Any]] = None
    table_query: Optional[Dict[str, Any]] = None
    step2_filters: Optional[Dict[str, Any]] = None
    base_depth_config: Optional[Dict[str, Any]] = None
    last_calculated_filters: Optional[Dict[str, Any]] = None
    ui_state: Optional[Dict[str, Any]] = None


class RunStateResponse(BaseModel):
    success: bool
    message: str
    run_id: str
    state: Dict[str, Any] = {}
    step1_result: Optional[Dict[str, Any]] = None
    step2_result: Optional[Dict[str, Any]] = None
    step3_result: Optional[Dict[str, Any]] = None
    step4_result: Optional[Dict[str, Any]] = None


class ModelingRequest(BaseModel):
    run_id: Optional[str] = None
    states: Optional[List[str]] = None
    categories: Optional[List[str]] = None
    subcategories: Optional[List[str]] = None
    brands: Optional[List[str]] = None
    sizes: Optional[List[str]] = None
    recency_threshold: int = Field(default=90, ge=30, le=180)
    frequency_threshold: int = Field(default=20, ge=1, le=100)
    time_aggregation: str = Field(default="M", pattern="^(D|W|M)$")
    round_step: float = Field(default=0.5, gt=0.0, le=10.0)
    min_upward_jump_pp: float = Field(default=1.0, ge=0.0, le=5.0)
    min_downward_drop_pp: float = Field(default=1.0, ge=0.0, le=5.0)
    include_lag_discount: bool = True
    l2_penalty: float = Field(default=1.0, ge=0.0, le=100.0)
    optimize_l2_penalty: bool = False
    constraint_residual_non_negative: bool = True
    constraint_structural_non_negative: bool = True
    constraint_tactical_non_negative: bool = True
    constraint_lag_non_positive: bool = True
    cogs_per_unit: Optional[float] = Field(default=None, ge=0.0)
    rfm_segments: Optional[List[str]] = None
    outlet_classifications: Optional[List[str]] = None
    slabs: Optional[List[str]] = None
    outlet_ids: Optional[List[str]] = None


class ModelingPoint(BaseModel):
    period: datetime
    actual_quantity: float
    predicted_quantity: float
    predicted_quantity_ols: Optional[float] = None
    actual_discount_pct: float
    base_discount_pct: float
    tactical_discount_pct: float
    roi_1mo: Optional[float] = None
    profit_roi_1mo: Optional[float] = None
    spend: Optional[float] = None
    incremental_revenue: Optional[float] = None
    incremental_profit: Optional[float] = None


class ModelingSlabResult(BaseModel):
    slab: str
    valid: bool = True
    reason: Optional[str] = None
    model_coefficients: Dict[str, float] = {}
    predicted_vs_actual: List[ModelingPoint] = []
    roi_points: List[ModelingPoint] = []
    summary: Dict[str, float] = {}


class ModelingResponse(BaseModel):
    success: bool
    message: str
    slab_results: List[ModelingSlabResult] = []
    combined_summary: Dict[str, float] = {}


class PlannerRequest(BaseModel):
    run_id: Optional[str] = None
    states: Optional[List[str]] = None
    categories: Optional[List[str]] = None
    subcategories: Optional[List[str]] = None
    brands: Optional[List[str]] = None
    sizes: Optional[List[str]] = None
    recency_threshold: int = Field(default=90, ge=30, le=180)
    frequency_threshold: int = Field(default=20, ge=1, le=100)
    round_step: float = Field(default=0.5, gt=0.0, le=10.0)
    min_upward_jump_pp: float = Field(default=1.0, ge=0.0, le=5.0)
    min_downward_drop_pp: float = Field(default=1.0, ge=0.0, le=5.0)
    include_lag_discount: bool = True
    l2_penalty: float = Field(default=1.0, ge=0.0, le=100.0)
    optimize_l2_penalty: bool = False
    constraint_residual_non_negative: bool = True
    constraint_structural_non_negative: bool = True
    constraint_tactical_non_negative: bool = True
    constraint_lag_non_positive: bool = True
    rfm_segments: Optional[List[str]] = None
    outlet_classifications: Optional[List[str]] = None
    slabs: Optional[List[str]] = None
    outlet_ids: Optional[List[str]] = None
    slab: Optional[str] = None
    plan_start_year: Optional[int] = None
    planned_structural_discounts: Optional[List[float]] = None
    planned_base_prices: Optional[List[float]] = None
    cogs_per_unit: Optional[float] = Field(default=None, ge=0.0)


class PlannerMonthPoint(BaseModel):
    period: datetime
    current_promo_pct: float
    planned_promo_pct: float
    base_price: float
    current_quantity: float
    planned_quantity: float
    current_revenue: float
    planned_revenue: float


class PlannerResponse(BaseModel):
    success: bool
    message: str
    slab: Optional[str] = None
    plan_start_month: Optional[str] = None
    months: List[str] = []
    default_structural_discounts: List[float] = []
    current_structural_discounts: List[float] = []
    planned_structural_discounts: List[float] = []
    planned_base_prices: List[float] = []
    cogs_per_unit: float = 0.0
    model_coefficients: Dict[str, float] = {}
    metrics: Dict[str, float] = {}
    series: List[PlannerMonthPoint] = []
    ai_insights_status: Optional[str] = None
    ai_insights: Optional[str] = None
