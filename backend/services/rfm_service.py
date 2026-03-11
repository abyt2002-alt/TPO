import os
from pathlib import Path

from services.core.context import ContextMixin
from services.core.state_store import StateStoreMixin
from services.core.data_loader import DataLoaderMixin
from services.core.scope_builder import ScopeBuilderMixin
from services.core.shared_math import SharedMathMixin
from services.core.scenario_compare import ScenarioCompareMixin
from services.core.eda_service import EDAServiceMixin
from services.steps.step1_segmentation import Step1SegmentationMixin
from services.steps.step2_discount import Step2DiscountMixin
from services.steps.step3_modeling import Step3ModelingMixin
from services.steps.step4_cross_size_planner import Step4CrossSizePlannerMixin
from services.steps.step5_baseline_forecast import Step5BaselineForecastMixin


class RFMService(
    ContextMixin,
    StateStoreMixin,
    DataLoaderMixin,
    ScopeBuilderMixin,
    SharedMathMixin,
    ScenarioCompareMixin,
    EDAServiceMixin,
    Step1SegmentationMixin,
    Step2DiscountMixin,
    Step3ModelingMixin,
    Step4CrossSizePlannerMixin,
    Step5BaselineForecastMixin,
):
    """Facade service that composes step/core mixins for backward compatibility."""

    def __init__(self):
        self._init_service_context()
        self._init_state_db()
        self.load_data()
