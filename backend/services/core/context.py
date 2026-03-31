"""Service context helpers for shared runtime state.\n\nThis module centralizes basic context initialization so the facade service can stay thin\nwhile step/core mixins use a consistent set of in-memory caches and DB paths.\n"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
import os
import sqlite3
import json
import uuid
import copy
import re
from io import BytesIO
from datetime import datetime, date
import urllib.request
import urllib.error
try:
    from statsmodels.tsa.holtwinters import Holt
except Exception:
    Holt = None

from models.rfm_models import (
    RFMRequest, RFMResponse, OutletRFM,
    SegmentSummary, ClusterSummary,
    BaseDepthRequest, BaseDepthResponse, BaseDepthPoint,
    DiscountOptionsRequest, DiscountOptionsResponse,
    ModelingRequest, ModelingResponse, ModelingSlabResult, ModelingPoint,
    PlannerRequest, PlannerResponse, PlannerMonthPoint,
    PlannerScenarioComparisonResponse, PlannerScenarioComparisonRow,
    CrossSizePlannerRequest, CrossSizePlannerResponse, CrossSizePlannerSizeResult, CrossSizePlannerSlabState,
    BaselineForecastRequest, BaselineForecastResponse, BaselineForecastPoint,
    EDARequest, EDAResponse, EDAProductOption, EDAProductContribution,
    EDAContributionRow, EDAOptionsResponse
)


class ContextMixin:

    def _init_service_context(self):

        self.data_cache = None

        self.rfm_result_cache = {}

        self.max_cache_entries = 5

        self.state_db_path = Path(__file__).resolve().parent.parent.parent / 'analysis_state.db'
