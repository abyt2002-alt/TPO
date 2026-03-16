import { useState, useEffect, useCallback, useRef, useMemo } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import { useSearchParams } from 'react-router-dom'
import {
  calculateRFM,
  getAvailableFilters,
  getCascadingFilters,
  exportRFMOutlets,
  calculateBaseDepth,
  calculateModeling,
  calculateCrossSizePlanner,
  calculateBaselineForecast,
  startAIScenarioJob,
  getAIScenarioJobStatus,
  getAIScenarioJobResults,
  getEDAOptions,
  getEDAOverview,
  getSlabTrendEDA,
  getDiscountOptions,
  createRun,
  getRunState,
  saveRunState,
} from '../services/api'
import Layout from '../components/Layout'
import FilterPanel from '../components/rfm/FilterPanel'
import RFMSummary from '../components/rfm/RFMSummary'
import SegmentGrid from '../components/rfm/SegmentGrid'
import ClusterSummary from '../components/rfm/ClusterSummary'
import OutletTable from '../components/rfm/OutletTable'
import BaseDepthEstimator from '../components/rfm/BaseDepthEstimator'
import ModelingROI from '../components/rfm/ModelingROI'
import CrossSizePlanner from '../components/rfm/CrossSizePlanner'
import BaselineForecast from '../components/rfm/BaselineForecast'
import ScenarioComparison from '../components/rfm/ScenarioComparison'
import EDAInsights from '../components/rfm/EDAInsights'
import SlabTrendEDA from '../components/rfm/SlabTrendEDA'
import DiscountStepFilters from '../components/rfm/DiscountStepFilters'
import { Step2SlabDefinitionPanel } from '../components/rfm/DiscountStepFilters'
import { Loader2, AlertCircle, BarChart3, ChevronDown, ChevronUp, Search } from 'lucide-react'
import { computeCrossSizePlannerData, normalizePlannerPeriodsFromData } from '../utils/crossSizePlannerCompute'

const DEFAULT_STEP2_FILTERS = {
  rfm_segments: [],
  outlet_classifications: [],
  slabs: [],
  slab_definition_mode: 'define',
  defined_slab_level: 'monthly_outlet',
  defined_slab_count: 5,
  defined_slab_thresholds: [8, 32, 576, 960],
  defined_slab_profiles: {
    '12-ML': {
      defined_slab_count: 3,
      defined_slab_thresholds: [8, 144],
    },
    '18-ML': {
      defined_slab_count: 5,
      defined_slab_thresholds: [8, 32, 576, 960],
    },
  },
}

const parseSlabIndex = (value) => {
  const match = String(value || '').trim().toLowerCase().match(/(\d+)/)
  if (!match) return null
  const idx = Number(match[1])
  if (!Number.isFinite(idx)) return null
  return idx
}

const normalizeStep2Slabs = (values = [], allowedOptions = null) => {
  let slabs = (values || [])
    .map((v) => String(v))
    .filter((v) => {
      const idx = parseSlabIndex(v)
      return idx !== null && idx >= 1
    })

  slabs = Array.from(new Set(slabs))
  slabs.sort((a, b) => {
    const ai = parseSlabIndex(a) ?? Number.MAX_SAFE_INTEGER
    const bi = parseSlabIndex(b) ?? Number.MAX_SAFE_INTEGER
    if (ai !== bi) return ai - bi
    return a.localeCompare(b)
  })

  if (Array.isArray(allowedOptions) && allowedOptions.length > 0) {
    const allowed = new Set(allowedOptions.map((v) => String(v)))
    slabs = slabs.filter((v) => allowed.has(v))
  }

  return slabs
}

const normalizeStep2SizeKey = (value) => String(value || '').toUpperCase().replace(/\s+/g, '').trim()

const normalizeDefinedSlabCount = (value) => {
  const parsed = Number(value)
  if (!Number.isFinite(parsed)) return DEFAULT_STEP2_FILTERS.defined_slab_count
  return Math.min(20, Math.max(2, Math.round(parsed)))
}

const normalizeDefinedSlabThresholds = (values = [], slabCount = DEFAULT_STEP2_FILTERS.defined_slab_count) => {
  const expected = Math.max(1, Number(slabCount || DEFAULT_STEP2_FILTERS.defined_slab_count) - 1)
  const fallback = [...DEFAULT_STEP2_FILTERS.defined_slab_thresholds]
  while (fallback.length < expected) {
    const last = fallback[fallback.length - 1] ?? 1
    fallback.push(last + Math.max(1, Math.abs(last) * 0.1))
  }
  const parsed = (values || [])
    .map((v) => Number(v))
    .filter((v) => Number.isFinite(v))
    .slice(0, expected)
  const merged = [...parsed]
  for (let i = merged.length; i < expected; i += 1) {
    merged.push(fallback[i])
  }
  return merged
}

const normalizeDefinedSlabProfiles = (profiles = {}) => {
  if (!profiles || typeof profiles !== 'object' || Array.isArray(profiles)) return {}
  const out = {}
  Object.entries(profiles).forEach(([rawSize, rawConfig]) => {
    const sizeKey = normalizeStep2SizeKey(rawSize)
    if (!sizeKey) return
    const cfg = rawConfig && typeof rawConfig === 'object' ? rawConfig : {}
    const count = normalizeDefinedSlabCount(cfg.defined_slab_count)
    out[sizeKey] = {
      defined_slab_count: count,
      defined_slab_thresholds: normalizeDefinedSlabThresholds(cfg.defined_slab_thresholds || [], count),
    }
  })
  return out
}

const normalizeStep2OutletClassifications = (values = []) => {
  const out = []
  for (const value of values || []) {
    const raw = String(value || '').trim().toUpperCase().replace(/\s+/g, '')
    if (!raw) continue
    out.push(raw === 'WH' ? 'WH' : 'OtherGT')
  }
  return Array.from(new Set(out)).sort((a, b) => {
    const order = { OtherGT: 0, WH: 1 }
    return (order[a] ?? 99) - (order[b] ?? 99) || a.localeCompare(b)
  })
}

const normalizeStep2Filters = (filters = {}) => ({
  rfm_segments: Array.isArray(filters?.rfm_segments) ? filters.rfm_segments.map((v) => String(v)) : [],
  outlet_classifications: normalizeStep2OutletClassifications(filters?.outlet_classifications || []),
  slabs: [],
  slab_definition_mode: 'define',
  defined_slab_level: 'monthly_outlet',
  defined_slab_count: normalizeDefinedSlabCount(filters?.defined_slab_count),
  defined_slab_thresholds: normalizeDefinedSlabThresholds(
    filters?.defined_slab_thresholds || [],
    normalizeDefinedSlabCount(filters?.defined_slab_count)
  ),
  defined_slab_profiles: normalizeDefinedSlabProfiles(filters?.defined_slab_profiles || {}),
})

const normalizeDiscountOptions = (options = {}) => ({
  ...options,
  rfm_segments: Array.isArray(options?.rfm_segments) ? options.rfm_segments.map((v) => String(v)) : [],
  outlet_classifications: normalizeStep2OutletClassifications(options?.outlet_classifications || []),
  slabs: normalizeStep2Slabs(options?.slabs || []),
  matching_outlets: Number(options?.matching_outlets || 0),
})

const DEFAULT_STEP1_FILTERS = {
  states: [],
  categories: [],
  subcategories: ['STX INSTA SHAMPOO', 'STREAX INSTA SHAMPOO'],
  brands: [],
  sizes: ['12-ML', '18-ML'],
  recency_threshold: 90,
  frequency_threshold: 20,
}

const DEFAULT_STEP5_FILTERS = {
  states: [],
  categories: [],
  subcategories: [],
  brands: [],
  sizes: [],
  outlet_classifications: [],
  product_codes: [],
}

const EMPTY_FILTERS = {
  states: [],
  categories: [],
  subcategories: [],
  brands: [],
  sizes: [],
}

const arraysEqualAsStrings = (left = [], right = []) => {
  if (!Array.isArray(left) || !Array.isArray(right) || left.length !== right.length) return false
  return left.every((value, index) => String(value) === String(right[index]))
}

const isStaleForecastResult = (result) => {
  if (!result || typeof result !== 'object') return true
  const points = Array.isArray(result.points) ? result.points : []
  if (points.length === 0) return true
  const hasMissingFields = points.some((point) => (
    point == null ||
    !Object.prototype.hasOwnProperty.call(point, 'discount_component_12_ml') ||
    !Object.prototype.hasOwnProperty.call(point, 'discount_component_18_ml')
  ))
  if (hasMissingFields) return true

  const allComponentsZero = points.every((point) => {
    const d12 = Number(point?.discount_component_12_ml || 0)
    const d18 = Number(point?.discount_component_18_ml || 0)
    return Math.abs(d12) < 1e-9 && Math.abs(d18) < 1e-9
  })
  return allComponentsZero
}

const isLegacyDefaultScope = (restoredFilters = {}, state = {}, restored = {}) => {
  const states = Array.isArray(restoredFilters.states) ? restoredFilters.states : []
  const categories = Array.isArray(restoredFilters.categories) ? restoredFilters.categories : []
  const brands = Array.isArray(restoredFilters.brands) ? restoredFilters.brands : []
  const subcategories = Array.isArray(restoredFilters.subcategories) ? restoredFilters.subcategories : []
  const sizes = Array.isArray(restoredFilters.sizes) ? restoredFilters.sizes : []
  const hasCalculatedOutput = Boolean(restored?.step1_result || state?.last_calculated_filters)

  return (
    !hasCalculatedOutput &&
    states.length === 0 &&
    categories.length === 0 &&
    brands.length === 0 &&
    arraysEqualAsStrings(subcategories, DEFAULT_STEP1_FILTERS.subcategories) &&
    arraysEqualAsStrings(sizes, ['18-ML'])
  )
}

const RUN_STORAGE_KEY = 'rfm_analysis_run_id'
const BOOTSTRAP_TIMEOUT_MS = 8000

const withTimeout = async (promise, timeoutMs = BOOTSTRAP_TIMEOUT_MS) => {
  let timeoutId
  const timeoutPromise = new Promise((_, reject) => {
    timeoutId = setTimeout(() => reject(new Error('Request timeout')), timeoutMs)
  })
  try {
    const result = await Promise.race([promise, timeoutPromise])
    return result
  } finally {
    clearTimeout(timeoutId)
  }
}

const getRequestErrorMessage = (error, fallbackMessage) => {
  const raw = String(error?.message || '').trim()
  const text = raw.toLowerCase()
  if (
    text.includes('aborted') ||
    text.includes('canceled') ||
    text.includes('cancelled') ||
    text.includes('network error')
  ) {
    return 'Request interrupted. Run Step 2 again.'
  }
  if (text.includes('timeout')) {
    return 'Step 2 took too long. Run it again.'
  }
  return raw || fallbackMessage
}

const resolveStepTabFromQuery = (stepParam) => {
  if (stepParam === '2') return 'step2'
  if (stepParam === '3') return 'step3'
  if (stepParam === '4') return 'step4'
  if (stepParam === '5') return 'step5'
  if (stepParam === '6') return 'step_eda'
  return 'step1'
}

const resolveStepQueryFromTab = (stepTab) => {
  if (stepTab === 'step2') return '2'
  if (stepTab === 'step3') return '3'
  if (stepTab === 'step4') return '4'
  if (stepTab === 'step5') return '5'
  if (stepTab === 'step_eda') return '6'
  return null
}

const FIXED_STAGE3_SETTINGS = {
  roi_mode: 'both',
  l2_penalty: 1.0,
  optimize_l2_penalty: true,
  constraint_residual_non_negative: true,
  constraint_structural_non_negative: true,
  constraint_tactical_non_negative: true,
  constraint_lag_non_positive: true,
}

const DEFAULT_MODELING_COGS_BY_SIZE = {
  '12-ML': 8,
  '18-ML': 10,
}

const normalizeModelingCogsBySize = (raw = {}) => {
  const out = { ...DEFAULT_MODELING_COGS_BY_SIZE }
  const input = raw && typeof raw === 'object' ? raw : {}
  Object.keys(out).forEach((sizeKey) => {
    const parsed = Number(input[sizeKey])
    if (Number.isFinite(parsed) && parsed > 0) out[sizeKey] = parsed
  })
  return out
}

const DEFAULT_STEP5_SCENARIO_BUILDER = {
  mode: 'fixed_historical_ladders_v2',
}
const DEFAULT_STEP5_AI_SETTINGS = {
  scenario_count: 5,
  prompt: '',
}

const STEP5_ANCHOR_SCENARIOS = [
  { key: 'anchor_last_3m_exact', id: 'last_3m_exact', name: 'Last 3 Months Exact' },
  { key: 'anchor_same_season_last_year', id: 'same_season_last_year', name: 'Same Season Last Year' },
  { key: 'anchor_most_common_historical_ladder', id: 'most_common_historical_ladder', name: 'Most Common Historical Ladder' },
  { key: 'anchor_highest_historical_promo_ladder', id: 'highest_historical_promo_ladder', name: 'Deep Discount' },
  { key: 'anchor_lowest_historical_promo_ladder', id: 'lowest_historical_promo_ladder', name: 'Shallow Discount' },
]

const normalizeStep5ScenarioBuilder = (builder = {}) => ({
  ...DEFAULT_STEP5_SCENARIO_BUILDER,
  ...(builder || {}),
})

const buildStep5ScenarioDefinitions = (rawBuilder = {}) => {
  const builder = normalizeStep5ScenarioBuilder(rawBuilder)
  const anchors = STEP5_ANCHOR_SCENARIOS.map((row) => ({ ...row, type: 'anchor' }))
  if (builder.mode !== DEFAULT_STEP5_SCENARIO_BUILDER.mode) return anchors
  return anchors
}

const toPeriodKey = (value) => {
  if (!value) return null
  if (typeof value === 'string') {
    const raw = value.trim()
    const m = raw.match(/^(\d{4})-(\d{2})/)
    if (m) return `${m[1]}-${m[2]}`
    const dt = new Date(raw)
    if (!Number.isNaN(dt.getTime())) {
      const y = dt.getFullYear()
      const mm = String(dt.getMonth() + 1).padStart(2, '0')
      return `${y}-${mm}`
    }
    return null
  }
  const dt = new Date(value)
  if (Number.isNaN(dt.getTime())) return null
  const y = dt.getFullYear()
  const mm = String(dt.getMonth() + 1).padStart(2, '0')
  return `${y}-${mm}`
}

const shiftPeriodKey = (periodKey, deltaMonths) => {
  const m = String(periodKey || '').match(/^(\d{4})-(\d{2})$/)
  if (!m) return null
  const year = Number(m[1])
  const month = Number(m[2]) - 1
  const dt = new Date(year, month, 1)
  dt.setMonth(dt.getMonth() + Number(deltaMonths || 0))
  const y = dt.getFullYear()
  const mm = String(dt.getMonth() + 1).padStart(2, '0')
  return `${y}-${mm}`
}

const sortPeriodKeys = (keys = []) =>
  [...keys].sort((a, b) => String(a).localeCompare(String(b)))

const EdaMultiSelect = ({
  label,
  options = [],
  selectedValues = [],
  onChange,
  placeholder = 'All',
  disabled = false,
}) => {
  const [isOpen, setIsOpen] = useState(false)
  const [search, setSearch] = useState('')
  const normalizedOptions = useMemo(
    () =>
      (options || []).map((opt) =>
        typeof opt === 'string'
          ? { value: String(opt), label: String(opt) }
          : { value: String(opt.value), label: String(opt.label ?? opt.value) }
      ),
    [options]
  )

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase()
    if (!q) return normalizedOptions
    return normalizedOptions.filter((opt) => opt.label.toLowerCase().includes(q))
  }, [normalizedOptions, search])

  const selectedSet = useMemo(() => new Set((selectedValues || []).map((v) => String(v))), [selectedValues])
  const selectedCount = selectedSet.size
  const displayText =
    selectedCount === 0
      ? placeholder
      : selectedCount === 1
        ? normalizedOptions.find((opt) => selectedSet.has(opt.value))?.label || '1 selected'
        : `${selectedCount} selected`

  const toggle = (value) => {
    if (disabled) return
    const valueStr = String(value)
    if (selectedSet.has(valueStr)) {
      onChange((selectedValues || []).filter((v) => String(v) !== valueStr))
    } else {
      onChange([...(selectedValues || []), valueStr])
    }
  }

  const handleSelectAllFiltered = () => {
    if (disabled) return
    const merged = new Set((selectedValues || []).map((v) => String(v)))
    filtered.forEach((opt) => merged.add(opt.value))
    onChange(Array.from(merged))
  }

  const handleClear = () => {
    if (disabled) return
    onChange([])
  }

  return (
    <div className="relative">
      <label className="block text-sm font-medium text-gray-700 mb-1">{label}</label>
      <button
        type="button"
        onClick={() => !disabled && setIsOpen((v) => !v)}
        className={`w-full px-3 py-2 text-left border border-gray-300 rounded-lg bg-white focus:outline-none focus:ring-2 focus:ring-primary flex items-center justify-between text-sm ${
          disabled ? 'opacity-60 cursor-not-allowed' : 'hover:bg-gray-50'
        }`}
      >
        <span className={selectedCount === 0 ? 'text-gray-400' : 'text-gray-900'}>{displayText}</span>
        {isOpen ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
      </button>
      {isOpen && (
        <>
          <div className="fixed inset-0 z-40" onClick={() => setIsOpen(false)} />
          <div className="absolute z-50 w-full mt-1 bg-white border border-gray-300 rounded-lg shadow-lg overflow-hidden">
            <div className="p-2 border-b border-gray-200 bg-gray-50">
              <div className="relative">
                <Search className="absolute left-2 top-2.5 w-4 h-4 text-gray-400" />
                <input
                  type="text"
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  placeholder="Search options..."
                  className="w-full pl-8 pr-3 py-2 text-sm border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-primary"
                />
              </div>
              <div className="grid grid-cols-2 gap-2 mt-2">
                <button
                  type="button"
                  onClick={handleSelectAllFiltered}
                  className="text-xs px-2 py-1 bg-primary text-white rounded hover:bg-opacity-90"
                >
                  Select All
                </button>
                <button
                  type="button"
                  onClick={handleClear}
                  className="text-xs px-2 py-1 bg-gray-200 text-gray-700 rounded hover:bg-gray-300"
                >
                  Clear
                </button>
              </div>
            </div>
            <div className="max-h-64 overflow-y-auto">
              {filtered.length === 0 ? (
                <div className="px-3 py-2 text-sm text-gray-500">No options found</div>
              ) : (
                filtered.map((opt) => (
                  <label key={opt.value} className="flex items-center px-3 py-2 hover:bg-gray-50 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={selectedSet.has(opt.value)}
                      onChange={() => toggle(opt.value)}
                      className="w-4 h-4 text-primary border-gray-300 rounded focus:ring-primary"
                    />
                    <span className="ml-2 text-sm text-gray-700">{opt.label}</span>
                  </label>
                ))
              )}
            </div>
          </div>
        </>
      )}
    </div>
  )
}

const RFMAnalysis = () => {
  const [searchParams, setSearchParams] = useSearchParams()
  const [filters, setFilters] = useState(DEFAULT_STEP1_FILTERS)
  const [rfmData, setRfmData] = useState(null)
  const [activeStepTab, setActiveStepTab] = useState(resolveStepTabFromQuery(searchParams.get('step')))
  const [baseDepthConfig, setBaseDepthConfig] = useState({
    time_aggregation: 'D',
    round_step: 0.5,
    rolling_window_periods: 10,
    quantile: 0.5,
    min_upward_jump_pp: 1.0,
    min_downward_drop_pp: 1.0,
  })
  const [runId, setRunId] = useState('')
  const [isRunBootstrapping, setIsRunBootstrapping] = useState(true)
  const hydrationCompletedRef = useRef(false)
  const runInitializationStartedRef = useRef(false)
  const [step2Filters, setStep2Filters] = useState(DEFAULT_STEP2_FILTERS)
  const [discountOptions, setDiscountOptions] = useState({
    rfm_segments: [],
    outlet_classifications: [],
    slabs: [],
    matching_outlets: 0,
  })
  const [isDiscountOptionsLoading, setIsDiscountOptionsLoading] = useState(false)
  const [lastCalculatedFilters, setLastCalculatedFilters] = useState(null)
  const [tableQuery, setTableQuery] = useState({
    page: 1,
    page_size: 20,
    search: '',
    sort_key: 'total_net_amt',
    sort_direction: 'desc',
  })
  const [cascadedFilters, setCascadedFilters] = useState(null)
  const [isCascadeLoading, setIsCascadeLoading] = useState(false)
  const [isBaseDepthConfigExpanded, setIsBaseDepthConfigExpanded] = useState(true)
  const [baseDepthResult, setBaseDepthResult] = useState(null)
  const [baseDepthErrorMessage, setBaseDepthErrorMessage] = useState('')
  const [modelingResult, setModelingResult] = useState(null)
  const [modelingErrorMessage, setModelingErrorMessage] = useState('')
  const [plannerResult, setPlannerResult] = useState(null)
  const [plannerErrorMessage, setPlannerErrorMessage] = useState('')
  const [plannerDefaultByReference, setPlannerDefaultByReference] = useState({})
  const [forecastResult, setForecastResult] = useState(null)
  const [forecastErrorMessage, setForecastErrorMessage] = useState('')
  const [scenarioResult, setScenarioResult] = useState(null)
  const [scenarioErrorMessage, setScenarioErrorMessage] = useState('')
  const [step5Filters, setStep5Filters] = useState(DEFAULT_STEP5_FILTERS)
  const [step5ScenarioBuilder, setStep5ScenarioBuilder] = useState(DEFAULT_STEP5_SCENARIO_BUILDER)
  const [step5AISettings, setStep5AISettings] = useState(DEFAULT_STEP5_AI_SETTINGS)
  const [step5AIJob, setStep5AIJob] = useState(null)
  const [step5CurrentFilterContext, setStep5CurrentFilterContext] = useState({
    discount_constraints: [],
    metric_thresholds: {},
  })
  const [step5CreateScenarioRequestId, setStep5CreateScenarioRequestId] = useState(0)
  const step5AutoScenarioInitRef = useRef(false)
  const [edaOptions, setEdaOptions] = useState({
    product_options: [],
    outlet_classifications: [],
    matching_rows: 0,
  })
  const [isEdaOptionsLoading, setIsEdaOptionsLoading] = useState(false)
  const [edaResult, setEdaResult] = useState(null)
  const [edaErrorMessage, setEdaErrorMessage] = useState('')
  const [slabTrendResult, setSlabTrendResult] = useState(null)
  const [slabTrendErrorMessage, setSlabTrendErrorMessage] = useState('')
  const [modelingSettings, setModelingSettings] = useState({
    include_lag_discount: true,
    cogs_per_unit: 0,
    cogs_per_size: { ...DEFAULT_MODELING_COGS_BY_SIZE },
    ...FIXED_STAGE3_SETTINGS,
  })
  const [step4DisplayReferenceMode, setStep4DisplayReferenceMode] = useState('ly_same_3m')
  const step5ScenarioDefs = useMemo(() => buildStep5ScenarioDefinitions(step5ScenarioBuilder), [step5ScenarioBuilder])

  // Fetch initial available filters
  const { data: availableFilters, isLoading: filtersLoading } = useQuery({
    queryKey: ['filters', 'default_scope'],
    queryFn: () => getCascadingFilters({
      states: DEFAULT_STEP1_FILTERS.states,
      categories: DEFAULT_STEP1_FILTERS.categories,
      subcategories: DEFAULT_STEP1_FILTERS.subcategories,
      brands: DEFAULT_STEP1_FILTERS.brands,
    }),
    retry: 3,
    retryDelay: (attempt) => Math.min(1000 * (2 ** attempt), 8000),
    staleTime: 5 * 60 * 1000,
    placeholderData: EMPTY_FILTERS,
  })

  // Fetch cascading filters when selections change
  useEffect(() => {
    if (!availableFilters) return

    let isActive = true
    const fetchCascadingFilters = async () => {
      try {
        setIsCascadeLoading(true)
        const [categoriesLevel, subcategoriesLevel, brandsLevel, sizesLevel] = await Promise.all([
          getCascadingFilters({
            states: filters.states,
            categories: [],
            subcategories: [],
            brands: [],
          }),
          getCascadingFilters({
            states: filters.states,
            categories: filters.categories,
            subcategories: [],
            brands: [],
          }),
          getCascadingFilters({
            states: filters.states,
            categories: filters.categories,
            subcategories: filters.subcategories,
            brands: [],
          }),
          getCascadingFilters({
            states: filters.states,
            categories: filters.categories,
            subcategories: filters.subcategories,
            brands: filters.brands,
          }),
        ])

        if (isActive) {
          setCascadedFilters({
            states: categoriesLevel?.states || [],
            categories: categoriesLevel?.categories || [],
            subcategories: subcategoriesLevel?.subcategories || [],
            brands: brandsLevel?.brands || [],
            sizes: sizesLevel?.sizes || [],
          })
        }
      } catch (error) {
        if (isActive) {
          // Fallback to initial filters if cascade fails
          setCascadedFilters(availableFilters)
        }
      } finally {
        if (isActive) {
          setIsCascadeLoading(false)
        }
      }
    }

    const debounceTimer = setTimeout(() => {
      fetchCascadingFilters()
    }, 500)

    return () => {
      isActive = false
      clearTimeout(debounceTimer)
    }
  }, [filters.states, filters.categories, filters.subcategories, filters.brands, availableFilters])

  // Calculate RFM mutation
  const calculateMutation = useMutation({
    mutationFn: calculateRFM,
    onSuccess: (data) => {
      if (data.success) {
        setRfmData(data)
      }
    },
  })
  const mutateRFM = calculateMutation.mutate

  const exportMutation = useMutation({
    mutationFn: exportRFMOutlets,
    onSuccess: (response) => {
      const blob = new Blob([response.data], { type: 'text/csv' })
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = 'rfm_outlets_full.csv'
      a.click()
      window.URL.revokeObjectURL(url)
    },
  })
  const mutateExport = exportMutation.mutate

  const baseDepthMutation = useMutation({
    mutationFn: calculateBaseDepth,
    onSuccess: (data) => {
      setBaseDepthResult(data)
      setBaseDepthErrorMessage('')
    },
    onError: (error) => {
      setBaseDepthErrorMessage(getRequestErrorMessage(error, 'Failed to estimate base depth'))
    },
  })

  const modelingMutation = useMutation({
    mutationFn: calculateModeling,
    onSuccess: (data) => {
      setModelingResult(data)
      setModelingErrorMessage('')
    },
    onError: (error) => {
      setModelingErrorMessage(error?.message || 'Failed to run Step 3 modeling')
    },
  })

  const plannerMutation = useMutation({
    mutationFn: calculateCrossSizePlanner,
    onSuccess: (data, variables) => {
      const isCacheOnly = Boolean(variables?.cache_only)
      if (!isCacheOnly) {
        setPlannerResult(data)
        setPlannerErrorMessage('')
      }
      const scenarioByPeriod = variables?.scenario_discounts_by_period
      const hasScenarioOverride =
        scenarioByPeriod &&
        typeof scenarioByPeriod === 'object' &&
        Object.keys(scenarioByPeriod).length > 0
      if (!hasScenarioOverride) {
        const referenceKey = String(
          variables?.reference_mode || data?.reference_mode || 'ly_same_3m'
        )
        setPlannerDefaultByReference((prev) => ({
          ...prev,
          [referenceKey]: data,
        }))
      }
    },
    onError: (error) => {
      setPlannerErrorMessage(error?.message || 'Failed to run Step 4 planner')
    },
  })

  const forecastMutation = useMutation({
    mutationFn: calculateBaselineForecast,
    onSuccess: (data) => {
      setForecastResult(data)
      setForecastErrorMessage('')
    },
    onError: (error) => {
      setForecastErrorMessage(error?.message || 'Failed to run Step 5 baseline forecast')
    },
  })

  const scenarioMutation = useMutation({
    mutationFn: async (variables = {}) => {
      if (!modelingResult?.success) {
        throw new Error('Run Step 3 modeling before Step 5 scenario comparison.')
      }

      const basePayload = buildPlannerPayload({
        reference_mode: step4DisplayReferenceMode || 'ly_same_3m',
      })

      const defaultResponse = plannerResult?.success
        ? plannerResult
        : await calculateCrossSizePlanner(basePayload)

      if (!defaultResponse?.success) {
        throw new Error(defaultResponse?.message || 'Failed to initialize Step 5 scenarios from Step 4 planner.')
      }
      const periods = normalizePlannerPeriodsFromData(defaultResponse)

      const readMetric = (summaryBlock = {}) => ({
        volume: Number(summaryBlock?.final_qty ?? summaryBlock?.scenario_qty_additive ?? 0),
        revenue: Number(summaryBlock?.scenario_revenue ?? 0),
        profit: Number(summaryBlock?.scenario_profit ?? 0),
        volume_pct: Number(summaryBlock?.vs_reference_volume_pct ?? 0),
        revenue_pct: Number(summaryBlock?.vs_reference_revenue_pct ?? 0),
        gross_margin_pct: (
          Number(summaryBlock?.scenario_revenue ?? 0) > 0 &&
          Number(summaryBlock?.reference_revenue ?? 0) > 0
        )
          ? (
            ((Number(summaryBlock?.scenario_profit ?? 0) / Number(summaryBlock?.scenario_revenue ?? 0)) * 100) -
            ((Number(summaryBlock?.reference_profit ?? 0) / Number(summaryBlock?.reference_revenue ?? 0)) * 100)
          )
          : Number(summaryBlock?.vs_reference_profit_pct ?? 0),
        profit_pct: (
          Number(summaryBlock?.scenario_revenue ?? 0) > 0 &&
          Number(summaryBlock?.reference_revenue ?? 0) > 0
        )
          ? (
            ((Number(summaryBlock?.scenario_profit ?? 0) / Number(summaryBlock?.scenario_revenue ?? 0)) * 100) -
            ((Number(summaryBlock?.reference_profit ?? 0) / Number(summaryBlock?.reference_revenue ?? 0)) * 100)
          )
          : Number(summaryBlock?.vs_reference_profit_pct ?? 0),
        scenario_investment: Number(summaryBlock?.scenario_investment ?? 0),
        reference_investment: Number(summaryBlock?.reference_investment ?? 0),
        investment_pct: Number(
          summaryBlock?.vs_reference_investment_pct ??
          summaryBlock?.investment_delta_pct ??
          summaryBlock?.investment_change_positive_vs_reference_pct ??
          0
        ),
      })

      const historicalContext = buildStep5HistoricalDiscountContext(defaultResponse)
      const buildScenarioRow = ({ key, id, name, scenarioByPeriod }) => {
        try {
          const response = computeCrossSizePlannerData({
            data: defaultResponse,
            periods,
            scenarioDiscountsByPeriod: scenarioByPeriod,
            applyTerminalLagStart: true,
          })
          const summary = response?.summary_3m || {}
          return {
            key,
            scenario: name,
            scenario_id: id,
            success: Boolean(response?.success),
            message: response?.message || '',
            scenario_discounts_by_period: scenarioByPeriod,
            summary: {
              '12-ML': readMetric(summary?.['12-ML']),
              '18-ML': readMetric(summary?.['18-ML']),
              TOTAL: {
                ...readMetric(summary?.TOTAL),
                volume_ml: Number(summary?.TOTAL?.final_volume_ml ?? summary?.TOTAL?.scenario_volume_ml_additive ?? 0),
                volume_ml_pct: Number(summary?.TOTAL?.vs_reference_volume_ml_pct ?? 0),
              },
            },
          }
        } catch (error) {
          return {
            key,
            scenario: name,
            scenario_id: id,
            success: false,
            message: error?.message || 'Failed to compute scenario',
            scenario_discounts_by_period: scenarioByPeriod,
            summary: null,
          }
        }
      }

      let scenarioInputs = []
      const mode = String(variables?.mode || 'fixed')

      if (mode === 'fixed') {
        scenarioInputs = step5ScenarioDefs.map((scenarioDef) => ({
          key: scenarioDef.key,
          id: scenarioDef.id,
          name: scenarioDef.name,
          scenarioByPeriod: buildStep5GeneratedScenarioMap(defaultResponse, scenarioDef, historicalContext),
        }))
      } else if (mode === 'custom') {
        scenarioInputs = Array.isArray(variables?.scenarioInputs) ? variables.scenarioInputs : []
        if (!scenarioInputs.length) {
          throw new Error('No custom scenarios provided.')
        }
      } else {
        throw new Error('Unsupported Step 5 scenario mode.')
      }

      const scenarios = scenarioInputs.map((scenarioDef) => buildScenarioRow({
        key: scenarioDef.key,
        id: scenarioDef.id,
        name: scenarioDef.name,
        scenarioByPeriod: scenarioDef.scenarioByPeriod,
      }))

      const successCount = scenarios.filter((row) => row?.success).length
      const successLabel = mode === 'custom' ? 'Updated' : 'Generated'
      return {
        success: successCount > 0,
        message: String(variables?.message || `${successLabel} ${scenarios.length} scenario(s); successful: ${successCount}.`),
        generation_mode: mode,
        reference_mode: String(basePayload.reference_mode || defaultResponse?.reference_mode || 'ly_same_3m'),
        periods,
        planner_base: defaultResponse,
        scenarios,
      }
    },
    onSuccess: (data) => {
      const incomingRows = Array.isArray(data?.scenarios) ? data.scenarios : []
      const scoreRevenue = (row) => Number(row?.summary?.TOTAL?.revenue_pct ?? -1e9)
      const scoreProfit = (row) => Number(row?.summary?.TOTAL?.profit_pct ?? -1e9)
      const scoreVolume = (row) => Number(row?.summary?.TOTAL?.volume_ml_pct ?? row?.summary?.TOTAL?.volume_pct ?? -1e9)
      const sortByRevenue = (rows) =>
        [...rows].sort((a, b) => {
          const ar = scoreRevenue(a)
          const br = scoreRevenue(b)
          if (br !== ar) return br - ar
          const ap = scoreProfit(a)
          const bp = scoreProfit(b)
          if (bp !== ap) return bp - ap
          const av = scoreVolume(a)
          const bv = scoreVolume(b)
          return bv - av
        })
      const keyFor = (row) =>
        String(
          row?.key ||
          row?.scenario_id ||
          row?.scenario ||
          `row_${Math.random().toString(36).slice(2)}`
        )

      const prevRows = Array.isArray(scenarioResult?.scenarios) ? scenarioResult.scenarios : []
      const mergedMap = new Map()
      prevRows.forEach((row) => mergedMap.set(keyFor(row), row))
      incomingRows.forEach((row) => mergedMap.set(keyFor(row), row))
      const mergedRows = sortByRevenue(Array.from(mergedMap.values()))
      setScenarioResult({
        ...data,
        scenarios: mergedRows,
      })
      setScenarioErrorMessage('')
    },
    onError: (error) => {
      setScenarioErrorMessage(error?.message || 'Failed to run Step 5 scenario generation')
    },
  })

  const aiScenarioJobMutation = useMutation({
    mutationFn: async (variables = {}) => {
      if (!modelingResult?.success) {
        throw new Error('Run Step 3 modeling before AI scenario generation.')
      }
      const basePayload = buildPlannerPayload({
        reference_mode: step4DisplayReferenceMode || 'ly_same_3m',
      })
      const aiPayload = {
        ...basePayload,
        scenario_count: Math.min(10000, Math.max(1, Number(variables?.scenario_count || step5AISettings.scenario_count || 5))),
        prompt: String(variables?.prompt || step5AISettings.prompt || ''),
        discount_constraints: Array.isArray(variables?.discount_constraints) ? variables.discount_constraints : [],
        metric_thresholds: variables?.metric_thresholds && typeof variables.metric_thresholds === 'object'
          ? variables.metric_thresholds
          : {},
      }
      const createResp = await startAIScenarioJob(aiPayload)
      if (!createResp?.success || !createResp?.job_id) {
        throw new Error(createResp?.message || 'Failed to start AI scenario job.')
      }
      const jobId = String(createResp.job_id)
      const totalTarget = Number(aiPayload.scenario_count || 0)
      setStep5AIJob({
        jobId,
        status: String(createResp.status || 'queued'),
        progressCurrent: 0,
        progressTotal: totalTarget,
        resultCount: 0,
        errorDetail: '',
      })

      const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms))
      let statusResp = null
      let pollRounds = 0
      let consecutivePollFailures = 0
      while (pollRounds < 3600) {
        pollRounds += 1
        await sleep(1500)
        try {
          statusResp = await getAIScenarioJobStatus(jobId)
          consecutivePollFailures = 0
        } catch (pollError) {
          consecutivePollFailures += 1
          const pollMsg = String(pollError?.message || '').toLowerCase()
          const transient =
            pollMsg.includes('timeout') ||
            pollMsg.includes('network error') ||
            pollMsg.includes('aborted') ||
            pollMsg.includes('canceled') ||
            pollMsg.includes('cancelled')
          if (transient && consecutivePollFailures <= 5) {
            continue
          }
          throw pollError
        }
        const status = String(statusResp?.status || 'queued')
        setStep5AIJob({
          jobId,
          status,
          progressCurrent: Number(statusResp?.progress_current || 0),
          progressTotal: Number(statusResp?.progress_total || totalTarget),
          resultCount: Number(statusResp?.result_count || 0),
          errorDetail: String(statusResp?.error_detail || ''),
        })
        if (status === 'completed') break
        if (status === 'failed' || status === 'cancelled') {
          throw new Error(statusResp?.error_detail || statusResp?.message || 'AI scenario generation failed.')
        }
      }
      if (!statusResp || String(statusResp?.status || '') !== 'completed') {
        throw new Error('AI scenario generation timed out.')
      }

      const rows = []
      const pageSize = 200
      let offset = 0
      const totalRows = Number(statusResp?.result_count || 0)
      while (offset < totalRows) {
        const page = await getAIScenarioJobResults(jobId, { offset, limit: pageSize })
        const pageRows = Array.isArray(page?.scenarios) ? page.scenarios : []
        rows.push(...pageRows)
        offset += pageRows.length
        if (pageRows.length === 0) break
      }

      const stamp = Date.now()
      const scenarioInputs = rows.map((row, idx) => ({
        key: `ai_${jobId}_${idx + 1}`,
        id: `ai_${jobId}_${idx + 1}`,
        name: String(row?.name || `AI Scenario ${idx + 1}`),
        scenarioByPeriod: row?.scenario_discounts_by_period || {},
      }))
      return {
        scenarioInputs,
        message: `Generated ${scenarioInputs.length} AI scenario(s); successful: ${scenarioInputs.length}.`,
        stamp,
      }
    },
    onSuccess: (data) => {
      scenarioMutation.mutate({
        mode: 'custom',
        scenarioInputs: Array.isArray(data?.scenarioInputs) ? data.scenarioInputs : [],
        message: String(data?.message || 'Generated AI scenarios.'),
      })
    },
    onError: (error) => {
      setScenarioErrorMessage(error?.message || 'AI scenario generation failed')
      setStep5AIJob((prev) => ({
        ...(prev || {}),
        status: 'failed',
        errorDetail: error?.message || 'AI scenario generation failed',
      }))
    },
  })

  const edaMutation = useMutation({
    mutationFn: getEDAOverview,
    onSuccess: (data) => {
      setEdaResult(data)
      setEdaErrorMessage('')
      if (Array.isArray(data?.product_options) && data.product_options.length > 0) {
        setEdaOptions((prev) => ({
          ...prev,
          product_options: data.product_options,
        }))
      }
    },
    onError: (error) => {
      setEdaErrorMessage(error?.message || 'Failed to run Step 5 EDA')
    },
  })

  const slabTrendMutation = useMutation({
    mutationFn: getSlabTrendEDA,
    onSuccess: (data) => {
      setSlabTrendResult(data)
      setSlabTrendErrorMessage('')
    },
    onError: (error) => {
      setSlabTrendErrorMessage(error?.message || 'Failed to run slab trend EDA')
    },
  })

  const setStepTab = useCallback((step) => {
    setActiveStepTab(step)
    const nextParams = new URLSearchParams(searchParams)
    if (runId) {
      nextParams.set('run_id', runId)
    }
    const stepQuery = resolveStepQueryFromTab(step)
    if (stepQuery) {
      nextParams.set('step', stepQuery)
    } else {
      nextParams.delete('step')
    }
    setSearchParams(nextParams)
  }, [searchParams, setSearchParams, runId])

  useEffect(() => {
    const stepFromUrl = resolveStepTabFromQuery(searchParams.get('step'))
    setActiveStepTab((prev) => (prev === stepFromUrl ? prev : stepFromUrl))
  }, [searchParams])

  useEffect(() => {
    if (runInitializationStartedRef.current) return
    runInitializationStartedRef.current = true

    let isActive = true
    const initialParams = new URLSearchParams(window.location.search)
    const hardStopTimer = setTimeout(() => {
      if (!isActive) return
      hydrationCompletedRef.current = true
      setIsRunBootstrapping(false)
    }, BOOTSTRAP_TIMEOUT_MS)

    const initializeRun = async () => {
      setIsRunBootstrapping(true)
      try {
        const urlRunId = initialParams.get('run_id')
        const storedRunId = window.localStorage.getItem(RUN_STORAGE_KEY)
        const requestedRunId = urlRunId || storedRunId || null

        const runResponse = await withTimeout(createRun(requestedRunId))
        const resolvedRunId = runResponse?.run_id
        if (!resolvedRunId || !isActive) return

        setRunId(resolvedRunId)
        window.localStorage.setItem(RUN_STORAGE_KEY, resolvedRunId)

        const restored = await withTimeout(getRunState(resolvedRunId))
        if (!isActive) return

        const state = restored?.state || {}
        const restoredFilters = state.filters || {}
        const useUpdatedDefaultScope = isLegacyDefaultScope(restoredFilters, state, restored)
        const hasSavedStep1Selection = ['states', 'categories', 'subcategories', 'brands', 'sizes']
          .some((key) => Array.isArray(restoredFilters[key]) && restoredFilters[key].length > 0)
        const restoredTableQuery = state.table_query || {}
        const restoredStep2 = state.step2_filters || {}
        const restoredConfig = state.base_depth_config || {}
        const restoredUi = state.ui_state || {}
        const restoredStep5Filters = restoredUi.step5_filters || {}
        const restoredStep5Builder = normalizeStep5ScenarioBuilder(restoredUi.step5_builder || {})
        const restoredStep5AI = restoredUi.step5_ai_settings || {}

        setFilters((prev) => ({
          ...prev,
          ...restoredFilters,
          states: useUpdatedDefaultScope
            ? DEFAULT_STEP1_FILTERS.states
            : (restoredFilters.states || (hasSavedStep1Selection ? [] : DEFAULT_STEP1_FILTERS.states)),
          categories: useUpdatedDefaultScope
            ? DEFAULT_STEP1_FILTERS.categories
            : (restoredFilters.categories || (hasSavedStep1Selection ? [] : DEFAULT_STEP1_FILTERS.categories)),
          subcategories: useUpdatedDefaultScope
            ? DEFAULT_STEP1_FILTERS.subcategories
            : (restoredFilters.subcategories || (hasSavedStep1Selection ? [] : DEFAULT_STEP1_FILTERS.subcategories)),
          brands: useUpdatedDefaultScope
            ? DEFAULT_STEP1_FILTERS.brands
            : (restoredFilters.brands || (hasSavedStep1Selection ? [] : DEFAULT_STEP1_FILTERS.brands)),
          sizes: useUpdatedDefaultScope
            ? DEFAULT_STEP1_FILTERS.sizes
            : (restoredFilters.sizes || (hasSavedStep1Selection ? [] : DEFAULT_STEP1_FILTERS.sizes)),
          recency_threshold: Number(restoredFilters.recency_threshold || DEFAULT_STEP1_FILTERS.recency_threshold),
          frequency_threshold: Number(restoredFilters.frequency_threshold || DEFAULT_STEP1_FILTERS.frequency_threshold),
        }))
        setTableQuery((prev) => ({ ...prev, ...restoredTableQuery }))
        setStep2Filters(normalizeStep2Filters(restoredStep2))
        setBaseDepthConfig((prev) => ({ ...prev, ...restoredConfig }))
        setLastCalculatedFilters(state.last_calculated_filters || null)
        if (typeof restoredUi.is_base_depth_config_expanded === 'boolean') {
          setIsBaseDepthConfigExpanded(restoredUi.is_base_depth_config_expanded)
        }
        setStep5Filters((prev) => ({
          ...prev,
          ...restoredStep5Filters,
          states: restoredStep5Filters.states || [],
          categories: restoredStep5Filters.categories || [],
          subcategories: restoredStep5Filters.subcategories || [],
          brands: restoredStep5Filters.brands || [],
          sizes: restoredStep5Filters.sizes || [],
          outlet_classifications: restoredStep5Filters.outlet_classifications || [],
          product_codes: restoredStep5Filters.product_codes || [],
        }))
        setStep5ScenarioBuilder(restoredStep5Builder)
        setStep5AISettings((prev) => ({
          ...prev,
          ...restoredStep5AI,
          scenario_count: Math.min(10000, Math.max(1, Number(restoredStep5AI?.scenario_count || prev.scenario_count || 5))),
          prompt: String(restoredStep5AI?.prompt || prev.prompt || ''),
        }))
        if (restored?.step1_result) {
          setRfmData(restored.step1_result)
        }
        if (restored?.step2_result) {
          setBaseDepthResult(restored.step2_result)
          if (restored.step2_result.success === false) {
            setBaseDepthErrorMessage(restored.step2_result.message || 'Failed to estimate base depth')
          }
        }
        if (restored?.step3_result) {
          setModelingResult(restored.step3_result)
        } else if (state?.step3_result) {
          setModelingResult(state.step3_result)
        }
        if (restored?.step4_result) {
          setPlannerResult(restored.step4_result)
        } else if (state?.step4_result) {
          setPlannerResult(state.step4_result)
        }
        const restoredForecast = restored?.step5_result || state?.step5_result || null
        setForecastResult(isStaleForecastResult(restoredForecast) ? null : restoredForecast)
        if (restored?.step6_result) {
          setScenarioResult(restored.step6_result)
        } else if (state?.step6_result) {
          setScenarioResult(state.step6_result)
        }

        const stepFromUrl = resolveStepTabFromQuery(initialParams.get('step'))
        const restoredStep = ['step1', 'step2', 'step3', 'step4', 'step5', 'step6'].includes(state.active_step)
          ? (state.active_step === 'step6' ? 'step5' : state.active_step)
          : 'step1'
        const effectiveStep = initialParams.get('step') === null ? restoredStep : stepFromUrl
        setActiveStepTab(effectiveStep)

        const nextParams = new URLSearchParams(initialParams)
        nextParams.set('run_id', resolvedRunId)
        const stepQuery = resolveStepQueryFromTab(effectiveStep)
        if (stepQuery) {
          nextParams.set('step', stepQuery)
        } else {
          nextParams.delete('step')
        }
        setSearchParams(nextParams, { replace: true })
      } catch (error) {
        // Continue with in-memory state if run bootstrap fails.
      } finally {
        if (isActive) {
          clearTimeout(hardStopTimer)
          hydrationCompletedRef.current = true
          setIsRunBootstrapping(false)
        }
      }
    }

    initializeRun()
    return () => {
      isActive = false
      clearTimeout(hardStopTimer)
    }
  }, [setSearchParams])

  useEffect(() => {
    if (!runId || !hydrationCompletedRef.current) return
    if (aiScenarioJobMutation.isPending) return

    let isActive = true
    const timer = setTimeout(async () => {
      try {
        const persistedStep5Builder = {
          mode: String(step5ScenarioBuilder?.mode || DEFAULT_STEP5_SCENARIO_BUILDER.mode),
        }
        await saveRunState(runId, {
          active_step: activeStepTab,
          filters,
          table_query: tableQuery,
          step2_filters: step2Filters,
          base_depth_config: baseDepthConfig,
          last_calculated_filters: lastCalculatedFilters,
          ui_state: {
            is_base_depth_config_expanded: isBaseDepthConfigExpanded,
            step5_filters: step5Filters,
            step5_builder: persistedStep5Builder,
            step5_ai_settings: step5AISettings,
          },
        })
      } catch {
        if (!isActive) return
      }
    }, 2000)

    return () => {
      isActive = false
      clearTimeout(timer)
    }
  }, [
    runId,
    activeStepTab,
    filters,
    tableQuery,
    step2Filters,
    baseDepthConfig,
    lastCalculatedFilters,
    isBaseDepthConfigExpanded,
    step5Filters,
    step5ScenarioBuilder,
    step5AISettings,
    aiScenarioJobMutation.isPending,
  ])

  const handleCalculate = () => {
    const baseFilters = { ...filters }
    const initialQuery = {
      page: 1,
      page_size: 20,
      search: '',
      sort_key: 'total_net_amt',
      sort_direction: 'desc',
    }
    setTableQuery(initialQuery)
    setLastCalculatedFilters(baseFilters)
    setStepTab('step1')
    setStep2Filters(DEFAULT_STEP2_FILTERS)
    setBaseDepthResult(null)
    setBaseDepthErrorMessage('')
    setModelingResult(null)
    setModelingErrorMessage('')
    setModelingSettings({
      include_lag_discount: true,
      cogs_per_unit: 0,
      cogs_per_size: { ...DEFAULT_MODELING_COGS_BY_SIZE },
      ...FIXED_STAGE3_SETTINGS,
    })
    setPlannerResult(null)
    setPlannerErrorMessage('')
    setPlannerDefaultByReference({})
    setForecastResult(null)
    setForecastErrorMessage('')
    setScenarioResult(null)
    setScenarioErrorMessage('')
    setSlabTrendResult(null)
    setSlabTrendErrorMessage('')
    mutateRFM({ run_id: runId || undefined, ...baseFilters, ...initialQuery })
  }

  const handleTableQueryChange = useCallback((query) => {
    if (!lastCalculatedFilters) return
    setTableQuery(query)
    mutateRFM({ run_id: runId || undefined, ...lastCalculatedFilters, ...query })
  }, [lastCalculatedFilters, mutateRFM, runId])

  const handleExport = useCallback(() => {
    if (!lastCalculatedFilters) return
    mutateExport({
      run_id: runId || undefined,
      ...lastCalculatedFilters,
      page: 1,
      page_size: 20,
      search: '',
      sort_key: 'total_net_amt',
      sort_direction: 'desc',
    })
  }, [lastCalculatedFilters, mutateExport, runId])

  const handleBaseDepthConfigChange = (key, value) => {
    setBaseDepthConfig((prev) => ({
      ...prev,
      [key]: value,
    }))
    setModelingResult(null)
    setPlannerResult(null)
    setPlannerDefaultByReference({})
    setForecastResult(null)
    setScenarioResult(null)
    setForecastErrorMessage('')
    setScenarioErrorMessage('')
  }

  const handleRunBaseDepth = () => {
    const baseFilters = lastCalculatedFilters || filters
    setBaseDepthResult(null)
    setBaseDepthErrorMessage('')
    setModelingResult(null)
    setModelingErrorMessage('')
    setPlannerResult(null)
    setPlannerErrorMessage('')
    setPlannerDefaultByReference({})
    setForecastResult(null)
    setForecastErrorMessage('')
    setScenarioResult(null)
    setScenarioErrorMessage('')
    const payload = {
      run_id: runId || undefined,
      ...baseFilters,
      ...baseDepthConfig,
      time_aggregation: step2Filters?.slab_definition_mode === 'define' ? 'M' : 'D',
      ...step2Filters,
    }
    baseDepthMutation.mutate(payload)
  }

  const handleRunModeling = (settings = {}) => {
    if (!baseDepthResult?.success) {
      setModelingErrorMessage('Run Step 2 base depth estimation before Step 3 modeling.')
      return
    }

    const effectiveSettings = {
      include_lag_discount: settings.include_lag_discount ?? modelingSettings.include_lag_discount,
      cogs_per_unit: settings.cogs_per_unit ?? modelingSettings.cogs_per_unit,
      cogs_per_size: normalizeModelingCogsBySize(settings.cogs_per_size ?? modelingSettings.cogs_per_size),
      ...FIXED_STAGE3_SETTINGS,
    }
    setModelingSettings((prev) => ({
      ...prev,
      include_lag_discount: Boolean(effectiveSettings.include_lag_discount),
      cogs_per_unit: Number(effectiveSettings.cogs_per_unit || 0),
      cogs_per_size: {
        ...normalizeModelingCogsBySize(prev.cogs_per_size || {}),
        ...normalizeModelingCogsBySize(effectiveSettings.cogs_per_size || {}),
      },
      ...FIXED_STAGE3_SETTINGS,
    }))

    const baseFilters = lastCalculatedFilters || filters
    const payload = {
      run_id: runId || undefined,
      ...baseFilters,
      ...baseDepthConfig,
      time_aggregation: 'M',
      include_lag_discount: Boolean(effectiveSettings.include_lag_discount),
      l2_penalty: Number(FIXED_STAGE3_SETTINGS.l2_penalty),
      optimize_l2_penalty: Boolean(FIXED_STAGE3_SETTINGS.optimize_l2_penalty),
      constraint_residual_non_negative: Boolean(FIXED_STAGE3_SETTINGS.constraint_residual_non_negative),
      constraint_structural_non_negative: Boolean(FIXED_STAGE3_SETTINGS.constraint_structural_non_negative),
      constraint_tactical_non_negative: Boolean(FIXED_STAGE3_SETTINGS.constraint_tactical_non_negative),
      constraint_lag_non_positive: Boolean(FIXED_STAGE3_SETTINGS.constraint_lag_non_positive),
      cogs_per_unit: Number(effectiveSettings.cogs_per_unit || 0),
      cogs_per_size: normalizeModelingCogsBySize(effectiveSettings.cogs_per_size || {}),
      ...step2Filters,
    }
    setPlannerResult(null)
    setPlannerErrorMessage('')
    setPlannerDefaultByReference({})
    setForecastResult(null)
    setForecastErrorMessage('')
    setScenarioResult(null)
    setScenarioErrorMessage('')
    modelingMutation.mutate(payload)
  }

  const buildPlannerPayload = (overrides = {}) => {
    const baseFilters = lastCalculatedFilters || filters
    return {
      run_id: runId || undefined,
      ...baseFilters,
      ...baseDepthConfig,
      include_lag_discount: Boolean(modelingSettings.include_lag_discount),
      l2_penalty: Number(FIXED_STAGE3_SETTINGS.l2_penalty),
      optimize_l2_penalty: Boolean(FIXED_STAGE3_SETTINGS.optimize_l2_penalty),
      constraint_residual_non_negative: Boolean(FIXED_STAGE3_SETTINGS.constraint_residual_non_negative),
      constraint_structural_non_negative: Boolean(FIXED_STAGE3_SETTINGS.constraint_structural_non_negative),
      constraint_tactical_non_negative: Boolean(FIXED_STAGE3_SETTINGS.constraint_tactical_non_negative),
      constraint_lag_non_positive: Boolean(FIXED_STAGE3_SETTINGS.constraint_lag_non_positive),
      cogs_per_unit: Number(modelingSettings.cogs_per_unit || 0),
      cogs_per_size: normalizeModelingCogsBySize(modelingSettings.cogs_per_size || {}),
      forecast_months: 3,
      planner_mode: 'additive_only',
      reference_mode: 'ly_same_3m',
      ...step2Filters,
      ...overrides,
    }
  }

  const getStep5SlabOrderBySize = (plannerResponse) => {
    const defaultsMatrix = plannerResponse?.defaults_matrix || {}
    const sizeResults = Array.isArray(plannerResponse?.size_results) ? plannerResponse.size_results : []
    const out = {}
    ;['12-ML', '18-ML'].forEach((sizeKey) => {
      const fromResult = sizeResults.find((row) => String(row?.size || '') === sizeKey)
      const slabs = Array.isArray(fromResult?.slabs)
        ? fromResult.slabs.map((s) => String(s?.slab || '')).filter(Boolean)
        : Object.keys(defaultsMatrix?.[sizeKey] || {})
      out[sizeKey] = [...new Set(slabs)].sort((a, b) => {
        const ai = parseSlabIndex(a) ?? Number.MAX_SAFE_INTEGER
        const bi = parseSlabIndex(b) ?? Number.MAX_SAFE_INTEGER
        if (ai !== bi) return ai - bi
        return a.localeCompare(b)
      })
    })
    return out
  }

  const getStep5DefaultValue = (defaultsMatrix, sizeKey, slabKey, monthIdx) => {
    const defaultSeries = defaultsMatrix?.[sizeKey]?.[slabKey]
    const raw = Array.isArray(defaultSeries)
      ? defaultSeries?.[monthIdx] ?? defaultSeries?.[0] ?? 0
      : defaultSeries ?? 0
    const n = Number(raw)
    return Number.isFinite(n) ? Number(n.toFixed(2)) : 0
  }

  const buildStep5HistoricalDiscountContext = (plannerResponse) => {
    const defaultsMatrix = plannerResponse?.defaults_matrix || {}
    const slabOrderBySize = getStep5SlabOrderBySize(plannerResponse)
    const bySize = {}
    ;['12-ML', '18-ML'].forEach((sizeKey) => {
      bySize[sizeKey] = { ladders: {}, fullMonths: [], latestMonth: null }
    })

    const slabRows = Array.isArray(modelingResult?.slab_results) ? modelingResult.slab_results : []
    slabRows.forEach((row) => {
      const sizeKey = String(row?.size || '').trim()
      const slabKey = String(row?.slab || '').trim()
      if (!bySize[sizeKey] || !slabOrderBySize[sizeKey]?.includes(slabKey)) return
      const points = Array.isArray(row?.predicted_vs_actual) ? row.predicted_vs_actual : []
      points.forEach((pt) => {
        const periodKey = toPeriodKey(pt?.period)
        const baseDiscount = Number(pt?.base_discount_pct)
        if (!periodKey || !Number.isFinite(baseDiscount)) return
        if (!bySize[sizeKey].ladders[periodKey]) bySize[sizeKey].ladders[periodKey] = {}
        bySize[sizeKey].ladders[periodKey][slabKey] = Number(baseDiscount.toFixed(2))
      })
    })

    ;['12-ML', '18-ML'].forEach((sizeKey) => {
      const slabOrder = slabOrderBySize[sizeKey] || []
      const months = sortPeriodKeys(Object.keys(bySize[sizeKey].ladders))
      bySize[sizeKey].fullMonths = months.filter((periodKey) =>
        slabOrder.every((slabKey) => Number.isFinite(Number(bySize[sizeKey].ladders?.[periodKey]?.[slabKey])))
      )
      bySize[sizeKey].latestMonth = bySize[sizeKey].fullMonths.length
        ? bySize[sizeKey].fullMonths[bySize[sizeKey].fullMonths.length - 1]
        : null

      if (!bySize[sizeKey].latestMonth && slabOrder.length > 0) {
        const fallbackPeriod = String(plannerResponse?.periods?.[0] || 'fallback')
        bySize[sizeKey].ladders[fallbackPeriod] = {}
        slabOrder.forEach((slabKey, idx) => {
          bySize[sizeKey].ladders[fallbackPeriod][slabKey] = getStep5DefaultValue(defaultsMatrix, sizeKey, slabKey, idx)
        })
        bySize[sizeKey].fullMonths = [fallbackPeriod]
        bySize[sizeKey].latestMonth = fallbackPeriod
      }
    })

    return { bySize, slabOrderBySize, defaultsMatrix }
  }

  const pickStep5ScenarioLadder = ({
    context,
    sizeKey,
    scenarioId,
    periodKey,
    periodIdx,
    periods,
  }) => {
    const sizeCtx = context?.bySize?.[sizeKey]
    const slabOrder = context?.slabOrderBySize?.[sizeKey] || []
    const ladders = sizeCtx?.ladders || {}
    const fullMonths = sizeCtx?.fullMonths || []
    const latestMonth = sizeCtx?.latestMonth
    const firstPeriod = String(periods?.[0] || periodKey || '')
    const fallbackMonth = latestMonth || fullMonths[fullMonths.length - 1] || null
    const fallbackLadder = fallbackMonth ? ladders[fallbackMonth] : null

    const readLadder = (sourceMonth) => {
      const source = String(sourceMonth || '')
      if (!source || !ladders[source]) return null
      const out = {}
      slabOrder.forEach((slabKey) => {
        const v = Number(ladders[source]?.[slabKey])
        if (Number.isFinite(v)) out[slabKey] = Number(v.toFixed(2))
      })
      return Object.keys(out).length ? out : null
    }

    if (scenarioId === 'latest_month_discount') {
      const source = shiftPeriodKey(firstPeriod, -(3 - Number(periodIdx || 0)))
      return readLadder(source) || readLadder(fallbackMonth) || fallbackLadder || {}
    }

    if (scenarioId === 'last_3m_exact') {
      const source = shiftPeriodKey(firstPeriod, -(3 - Number(periodIdx || 0)))
      return readLadder(source) || readLadder(fallbackMonth) || fallbackLadder || {}
    }

    if (scenarioId === 'same_season_last_year') {
      const source = shiftPeriodKey(periodKey, -12)
      return readLadder(source) || readLadder(fallbackMonth) || fallbackLadder || {}
    }

    const scoreMonths = fullMonths.map((month) => {
      const ladder = readLadder(month) || {}
      const values = slabOrder
        .map((slabKey) => Number(ladder?.[slabKey]))
        .filter((v) => Number.isFinite(v))
      const avg = values.length ? values.reduce((acc, v) => acc + v, 0) / values.length : -Infinity
      const signature = slabOrder.map((slabKey) => Number(ladder?.[slabKey] || 0).toFixed(2)).join('|')
      return { month, ladder, avg, signature }
    })

    if (scenarioId === 'highest_historical_promo_ladder') {
      const pick = [...scoreMonths].sort((a, b) => (b.avg - a.avg) || String(b.month).localeCompare(String(a.month)))[0]
      return pick?.ladder || readLadder(fallbackMonth) || fallbackLadder || {}
    }

    if (scenarioId === 'lowest_historical_promo_ladder') {
      const pick = [...scoreMonths].sort((a, b) => (a.avg - b.avg) || String(b.month).localeCompare(String(a.month)))[0]
      return pick?.ladder || readLadder(fallbackMonth) || fallbackLadder || {}
    }

    if (scenarioId === 'most_common_historical_ladder') {
      const groups = {}
      scoreMonths.forEach((row) => {
        const sig = String(row.signature || '')
        if (!groups[sig]) groups[sig] = { count: 0, latest: row.month, ladder: row.ladder }
        groups[sig].count += 1
        if (String(row.month) > String(groups[sig].latest)) {
          groups[sig].latest = row.month
          groups[sig].ladder = row.ladder
        }
      })
      const pick = Object.values(groups).sort((a, b) => (b.count - a.count) || String(b.latest).localeCompare(String(a.latest)))[0]
      return pick?.ladder || readLadder(fallbackMonth) || fallbackLadder || {}
    }

    return readLadder(fallbackMonth) || fallbackLadder || {}
  }

  const clampStep5Discount = (value) => {
    const n = Number(value)
    if (!Number.isFinite(n)) return 5
    return Number(Math.min(30, Math.max(5, n)).toFixed(2))
  }

  const enforceStep5Ladder = (valuesBySlab = {}, slabOrder = []) => {
    if (!Array.isArray(slabOrder) || slabOrder.length === 0) return valuesBySlab
    const out = {}
    slabOrder.forEach((slabKey, idx) => {
      const desired = clampStep5Discount(valuesBySlab?.[slabKey])
      const lowerBound = idx === 0 ? 5 : Number(out[slabOrder[idx - 1]] || 5) + 1
      const upperBound = 30 - (slabOrder.length - 1 - idx)
      const next = Math.min(upperBound, Math.max(lowerBound, desired))
      out[slabKey] = Number(next.toFixed(2))
    })
    return out
  }

  const cloneStep5PeriodMap = (periodMap = {}) => JSON.parse(JSON.stringify(periodMap || {}))

  const applyStep5Movement = ({
    anchorMap,
    periods,
    slabOrderBySize,
    movementKind,
    movementMode,
    shiftPp = 0,
    shiftDirection = 'high',
    returnPattern = 'high_base_base',
  }) => {
    const out = cloneStep5PeriodMap(anchorMap)
    const orderedPeriods = Array.isArray(periods) ? periods.map((p) => String(p)) : []
    if (!orderedPeriods.length) return out
    const sign = shiftDirection === 'low' ? -1 : 1
    const shiftValue = Number(shiftPp || 0) * sign
    const patternFactor = (monthIdx) => {
      if (movementKind !== 'return_to_normal') return 1
      if (returnPattern === 'base_high_base') return monthIdx === 1 ? 1 : 0
      return monthIdx === 0 ? 1 : 0
    }

    ;['12-ML', '18-ML'].forEach((sizeKey) => {
      const slabOrder = slabOrderBySize?.[sizeKey] || []
      if (!slabOrder.length) return
      const firstPeriod = orderedPeriods[0]
      const firstLadder = out?.[firstPeriod]?.[sizeKey] || {}

      orderedPeriods.forEach((periodKey, monthIdx) => {
        if (!out?.[periodKey]?.[sizeKey]) out[periodKey] = { ...(out[periodKey] || {}), [sizeKey]: {} }
        const anchorLadder = out[periodKey][sizeKey] || {}
        let nextLadder = {}

        if (movementKind === 'same_all_3m') {
          slabOrder.forEach((slabKey) => {
            nextLadder[slabKey] = clampStep5Discount(firstLadder?.[slabKey])
          })
        } else if (movementMode === 'preserve_gaps') {
          const baseSlab = slabOrder[0]
          const anchorBase = Number(anchorLadder?.[baseSlab] || firstLadder?.[baseSlab] || 5)
          const factor = patternFactor(monthIdx)
          const targetBase = clampStep5Discount(anchorBase + shiftValue * factor)
          slabOrder.forEach((slabKey) => {
            const anchorVal = Number(anchorLadder?.[slabKey] || firstLadder?.[slabKey] || targetBase)
            const gap = anchorVal - anchorBase
            nextLadder[slabKey] = clampStep5Discount(targetBase + gap)
          })
        } else {
          const factor = movementKind === 'return_to_normal' ? patternFactor(monthIdx) : 1
          slabOrder.forEach((slabKey) => {
            const anchorVal = Number(anchorLadder?.[slabKey] || firstLadder?.[slabKey] || 5)
            nextLadder[slabKey] = clampStep5Discount(anchorVal + shiftValue * factor)
          })
        }

        out[periodKey][sizeKey] = enforceStep5Ladder(nextLadder, slabOrder)
      })
    })
    return out
  }

  const buildStep5AnchorScenarioMap = (plannerResponse, scenarioId, historicalContext = null) => {
    const periods = Array.isArray(plannerResponse?.periods) ? plannerResponse.periods.map((p) => String(p)) : []
    const defaultsMatrix = plannerResponse?.defaults_matrix || {}
    const context = historicalContext || buildStep5HistoricalDiscountContext(plannerResponse)
    const periodMap = {}

    periods.forEach((periodKey, periodIdx) => {
      periodMap[periodKey] = {}
      ;['12-ML', '18-ML'].forEach((sizeKey) => {
        const slabOrder = context?.slabOrderBySize?.[sizeKey] || Object.keys(defaultsMatrix?.[sizeKey] || {})
        const ladder = pickStep5ScenarioLadder({
          context,
          sizeKey,
          scenarioId,
          periodKey,
          periodIdx,
          periods,
        })
        periodMap[periodKey][sizeKey] = {}
        slabOrder.forEach((slabKey) => {
          const fallback = getStep5DefaultValue(defaultsMatrix, sizeKey, slabKey, periodIdx)
          const value = Number(ladder?.[slabKey])
          periodMap[periodKey][sizeKey][slabKey] = clampStep5Discount(Number.isFinite(value) ? value : fallback)
        })
        periodMap[periodKey][sizeKey] = enforceStep5Ladder(periodMap[periodKey][sizeKey], slabOrder)
      })
    })
    return periodMap
  }

  const buildStep5GeneratedScenarioMap = (plannerResponse, scenarioDef, historicalContext = null) => {
    const periods = Array.isArray(plannerResponse?.periods) ? plannerResponse.periods.map((p) => String(p)) : []
    const scenarioId = String(scenarioDef?.id || 'last_3m_exact')
    const context = historicalContext || buildStep5HistoricalDiscountContext(plannerResponse)
    const anchorMap = buildStep5AnchorScenarioMap(plannerResponse, scenarioId, context)
    if (String(scenarioDef?.type || 'anchor') !== 'layer2') return anchorMap
    return applyStep5Movement({
      anchorMap,
      periods,
      slabOrderBySize: context?.slabOrderBySize || {},
      movementKind: String(scenarioDef?.movement_kind || 'same_all_3m'),
      movementMode: String(scenarioDef?.movement_mode || 'all_slabs_together'),
      shiftPp: Number(scenarioDef?.shift_pp || 0),
      shiftDirection: String(scenarioDef?.shift_direction || 'high'),
      returnPattern: String(scenarioDef?.return_pattern || 'high_base_base'),
    })
  }

  const handleGeneratePlanner = (overrides = {}) => {
    if (!modelingResult?.success) {
      setPlannerErrorMessage('Run Step 3 modeling before Step 4 planner.')
      return
    }
    setPlannerErrorMessage('')
    plannerMutation.mutate(buildPlannerPayload(overrides))
  }

  const handleFetchPlannerReference = async (referenceMode) => {
    if (!modelingResult?.success) return
    const mode = String(referenceMode || '').trim()
    if (!mode) return
    if (plannerDefaultByReference?.[mode]) return
    try {
      const data = await calculateCrossSizePlanner(buildPlannerPayload({ reference_mode: mode }))
      if (data?.success) {
        setPlannerDefaultByReference((prev) => ({
          ...prev,
          [mode]: data,
        }))
      }
    } catch (_) {
      // Silent prefetch failure: Step 4 main flow remains unaffected.
    }
  }

  useEffect(() => {
    const lyCached = Boolean(plannerDefaultByReference?.ly_same_3m)
    if (activeStepTab !== 'step4') return
    if (!modelingResult?.success) return
    if (lyCached) return
    handleFetchPlannerReference('ly_same_3m')
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeStepTab, modelingResult?.success, Boolean(plannerDefaultByReference?.ly_same_3m)])

  const handleRunScenarioComparison = () => {
    if (!modelingResult?.success) {
      setScenarioErrorMessage('Run Step 3 modeling before Step 5 scenario generation.')
      return
    }
    setScenarioErrorMessage('')
    scenarioMutation.mutate({ mode: 'fixed' })
  }

  const handleOpenCreateScenario = () => {
    setStep5CreateScenarioRequestId((prev) => prev + 1)
  }

  const buildStep5PromptWithFilterContext = (basePromptRaw = '', filterContext = {}) => {
    const basePrompt = String(basePromptRaw || '').trim()
    const discountConstraints = Array.isArray(filterContext?.discount_constraints)
      ? filterContext.discount_constraints
      : []
    const thresholds = filterContext?.metric_thresholds || {}

    const constraintLines = []
    discountConstraints.forEach((c) => {
      const label = `${String(c?.period || '')} ${String(c?.size || '')} ${String(c?.slab || '')}`.trim()
      const minTxt = Number.isFinite(Number(c?.min)) ? `>= ${Number(c.min)}` : ''
      const maxTxt = Number.isFinite(Number(c?.max)) ? `<= ${Number(c.max)}` : ''
      const cond = [minTxt, maxTxt].filter(Boolean).join(' and ')
      if (label && cond) constraintLines.push(`- ${label}: ${cond}`)
    })

    const metricLines = []
    const pushMetricMin = (name, val) => {
      if (Number.isFinite(Number(val))) metricLines.push(`- ${name}: >= ${Number(val)}%`)
    }
    const pushMetricMax = (name, val) => {
      if (Number.isFinite(Number(val))) metricLines.push(`- ${name}: <= ${Number(val)}%`)
    }
    pushMetricMin('12-ML volume change', thresholds?.min_12_volume_pct)
    pushMetricMin('18-ML volume change', thresholds?.min_18_volume_pct)
    pushMetricMin('TOTAL volume change', thresholds?.min_total_volume_pct)
    pushMetricMin('TOTAL revenue change', thresholds?.min_revenue_pct)
    pushMetricMin('TOTAL gross margin change', thresholds?.min_gross_margin_pct ?? thresholds?.min_profit_pct)
    pushMetricMin('TOTAL investment change', thresholds?.min_investment_pct)
    pushMetricMax('TOTAL investment change', thresholds?.max_investment_pct)
    pushMetricMax('TOTAL CTS (investment/revenue)', thresholds?.max_cts_pct)

    const promptParts = [
      basePrompt,
      constraintLines.length > 0 ? `Hard slab-month discount constraints:\n${constraintLines.join('\n')}` : '',
      metricLines.length > 0 ? `Hard output constraints:\n${metricLines.join('\n')}` : '',
      (constraintLines.length > 0 || metricLines.length > 0)
        ? 'Generate scenarios that satisfy all constraints. If needed, prioritize valid scenarios over aggressive exploration.'
        : '',
    ].filter((s) => String(s || '').trim() !== '')

    return promptParts.join('\n\n')
  }

  const normalizeStep5FilterContextForAI = (filterContext = {}) => {
    const rawConstraints = Array.isArray(filterContext?.discount_constraints)
      ? filterContext.discount_constraints
      : []
    const discount_constraints = rawConstraints
      .map((c) => ({
        period: String(c?.period || '').trim(),
        size: String(c?.size || '').trim(),
        slab: String(c?.slab || '').trim(),
        min: Number.isFinite(Number(c?.min)) ? Number(c.min) : null,
        max: Number.isFinite(Number(c?.max)) ? Number(c.max) : null,
      }))
      .filter((c) => c.period && c.size && c.slab && (c.min !== null || c.max !== null))

    const rawThresholds = filterContext?.metric_thresholds || {}
    const metric_thresholds = {
      min_12_volume_pct: Number.isFinite(Number(rawThresholds?.min_12_volume_pct)) ? Number(rawThresholds.min_12_volume_pct) : null,
      min_18_volume_pct: Number.isFinite(Number(rawThresholds?.min_18_volume_pct)) ? Number(rawThresholds.min_18_volume_pct) : null,
      min_total_volume_pct: Number.isFinite(Number(rawThresholds?.min_total_volume_pct)) ? Number(rawThresholds.min_total_volume_pct) : null,
      min_revenue_pct: Number.isFinite(Number(rawThresholds?.min_revenue_pct)) ? Number(rawThresholds.min_revenue_pct) : null,
      min_gross_margin_pct: Number.isFinite(Number(rawThresholds?.min_gross_margin_pct))
        ? Number(rawThresholds.min_gross_margin_pct)
        : (Number.isFinite(Number(rawThresholds?.min_profit_pct)) ? Number(rawThresholds.min_profit_pct) : null),
      min_investment_pct: Number.isFinite(Number(rawThresholds?.min_investment_pct)) ? Number(rawThresholds.min_investment_pct) : null,
      max_investment_pct: Number.isFinite(Number(rawThresholds?.max_investment_pct)) ? Number(rawThresholds.max_investment_pct) : null,
      max_cts_pct: Number.isFinite(Number(rawThresholds?.max_cts_pct)) ? Number(rawThresholds.max_cts_pct) : null,
    }
    return { discount_constraints, metric_thresholds }
  }

  const handleGenerateAIScenarios = () => {
    if (!modelingResult?.success) {
      setScenarioErrorMessage('Run Step 3 modeling before AI scenario generation.')
      return
    }
    const promptWithConstraints = buildStep5PromptWithFilterContext(
      step5AISettings.prompt,
      step5CurrentFilterContext
    )
    const structuredFilters = normalizeStep5FilterContextForAI(step5CurrentFilterContext)
    setScenarioErrorMessage('')
    aiScenarioJobMutation.mutate({
      scenario_count: Math.min(10000, Math.max(1, Number(step5AISettings.scenario_count || 5))),
      prompt: promptWithConstraints,
      discount_constraints: structuredFilters.discount_constraints,
      metric_thresholds: structuredFilters.metric_thresholds,
    })
  }

  const handleGenerateAIScenariosForCurrentFilters = (filterContext = {}) => {
    if (!modelingResult?.success) {
      setScenarioErrorMessage('Run Step 3 modeling before AI scenario generation.')
      return
    }
    const promptWithConstraints = buildStep5PromptWithFilterContext(step5AISettings.prompt, filterContext)
    const structuredFilters = normalizeStep5FilterContextForAI(filterContext)

    setScenarioErrorMessage('')
    aiScenarioJobMutation.mutate({
      scenario_count: Math.min(10000, Math.max(1, Number(step5AISettings.scenario_count || 5))),
      prompt: promptWithConstraints,
      discount_constraints: structuredFilters.discount_constraints,
      metric_thresholds: structuredFilters.metric_thresholds,
    })
  }

  const isAIScenarioRow = (row = {}) => {
    const key = String(row?.key || '').toLowerCase()
    const id = String(row?.id || row?.scenario_id || '').toLowerCase()
    const name = String(row?.name || row?.scenario || '').toLowerCase()
    return key.startsWith('ai_') || id.startsWith('ai_') || name.startsWith('ai ')
  }

  const handleDeleteAIScenarios = () => {
    setScenarioResult((prev) => {
      const currentRows = Array.isArray(prev?.scenarios) ? prev.scenarios : []
      const keptRows = currentRows.filter((row) => !isAIScenarioRow(row))
      if (keptRows.length === currentRows.length) return prev
      return {
        ...(prev || {}),
        success: true,
        scenarios: keptRows,
        message: `Removed ${currentRows.length - keptRows.length} AI scenario(s).`,
      }
    })
    setScenarioErrorMessage('')
  }

  const aiScenarioCount = useMemo(() => {
    const rows = Array.isArray(scenarioResult?.scenarios) ? scenarioResult.scenarios : []
    return rows.reduce((acc, row) => (isAIScenarioRow(row) ? acc + 1 : acc), 0)
  }, [scenarioResult?.scenarios])

  const step5AIStatus = String(step5AIJob?.status || '').toLowerCase()
  const isStep5AIJobRunning = step5AIStatus === 'queued' || step5AIStatus === 'running'
  const isStep5AIApplyingResults = step5AIStatus === 'completed' && scenarioMutation.isPending
  const isStep5AIBusy = Boolean(aiScenarioJobMutation.isPending || isStep5AIJobRunning || isStep5AIApplyingResults)

  const handleRunBaselineForecast = () => {
    if (!modelingResult?.success) {
      setForecastErrorMessage('Run Step 3 modeling before baseline forecast.')
      return
    }
    const baseFilters = lastCalculatedFilters || filters
    forecastMutation.mutate({
      run_id: runId || undefined,
      ...baseFilters,
      ...baseDepthConfig,
      time_aggregation: 'M',
      include_lag_discount: Boolean(modelingSettings.include_lag_discount),
      l2_penalty: Number(FIXED_STAGE3_SETTINGS.l2_penalty),
      optimize_l2_penalty: Boolean(FIXED_STAGE3_SETTINGS.optimize_l2_penalty),
      constraint_residual_non_negative: Boolean(FIXED_STAGE3_SETTINGS.constraint_residual_non_negative),
      constraint_structural_non_negative: Boolean(FIXED_STAGE3_SETTINGS.constraint_structural_non_negative),
      constraint_tactical_non_negative: Boolean(FIXED_STAGE3_SETTINGS.constraint_tactical_non_negative),
      constraint_lag_non_positive: Boolean(FIXED_STAGE3_SETTINGS.constraint_lag_non_positive),
      cogs_per_unit: Number(modelingSettings.cogs_per_unit || 0),
      cogs_per_size: normalizeModelingCogsBySize(modelingSettings.cogs_per_size || {}),
      forecast_months: 3,
      ...step2Filters,
    })
  }

  const handleStep2FilterChange = (key, value) => {
    setModelingResult(null)
    setModelingErrorMessage('')
    setPlannerResult(null)
    setPlannerErrorMessage('')
    setPlannerDefaultByReference({})
    setForecastResult(null)
    setForecastErrorMessage('')
    setScenarioResult(null)
    setScenarioErrorMessage('')
    setSlabTrendResult(null)
    setSlabTrendErrorMessage('')
    setBaseDepthConfig((prev) => ({ ...prev, time_aggregation: 'M' }))
    setStep2Filters((prev) => {
      let normalizedValue = value
      if (key === 'outlet_classifications') {
        normalizedValue = normalizeStep2OutletClassifications(value || [])
      } else if (key === 'slabs') {
        normalizedValue = normalizeStep2Slabs(value || [], discountOptions?.slabs || [])
      } else if (key === 'defined_slab_count') {
        normalizedValue = normalizeDefinedSlabCount(value)
      } else if (key === 'defined_slab_thresholds') {
        normalizedValue = normalizeDefinedSlabThresholds(value || [], prev?.defined_slab_count)
      } else if (key === 'defined_slab_level') {
        normalizedValue = 'monthly_outlet'
      } else if (key === 'defined_slab_profiles') {
        normalizedValue = normalizeDefinedSlabProfiles(value || {})
      }
      const next = { ...prev, [key]: normalizedValue, slab_definition_mode: 'define' }
      if (key === 'defined_slab_count') {
        next.defined_slab_thresholds = normalizeDefinedSlabThresholds(
          prev?.defined_slab_thresholds || [],
          normalizedValue
        )
      }
      if (key === 'rfm_segments') {
        next.outlet_classifications = []
        next.slabs = []
      } else if (key === 'outlet_classifications') {
        next.slabs = []
      } else if (
        key === 'defined_slab_count' ||
        key === 'defined_slab_thresholds' ||
        key === 'defined_slab_level' ||
        key === 'defined_slab_profiles'
      ) {
        next.slabs = []
      }
      return next
    })
  }

  const buildStep5Payload = (overrides = {}) => {
    const merged = {
      ...step5Filters,
      ...overrides,
    }
    return {
      run_id: runId || undefined,
      states: merged.states || [],
      categories: merged.categories || [],
      subcategories: merged.subcategories || [],
      brands: merged.brands || [],
      sizes: merged.sizes || [],
      outlet_classifications: merged.outlet_classifications || [],
      product_codes: merged.product_codes || [],
      top_n_products: 300,
    }
  }

  const handleStep5FilterChange = (key, value) => {
    setEdaErrorMessage('')
    setEdaResult(null)
    setStep5Filters((prev) => ({ ...prev, [key]: value }))
  }

  const handleRunEDA = () => {
    edaMutation.mutate(buildStep5Payload())
  }

  const buildSlabTrendPayload = () => {
    const baseFilters = lastCalculatedFilters || filters
    return {
      run_id: runId || undefined,
      ...baseFilters,
      ...step2Filters,
    }
  }

  const handleRunSlabTrendEDA = () => {
    slabTrendMutation.mutate(buildSlabTrendPayload())
  }

  useEffect(() => {
    if (!lastCalculatedFilters || !rfmData?.success) return

    let isActive = true
    const timer = setTimeout(async () => {
      try {
        setIsDiscountOptionsLoading(true)
        const options = await getDiscountOptions({
          run_id: runId || undefined,
          ...lastCalculatedFilters,
          ...step2Filters,
        })
        if (isActive && options?.success) {
          setDiscountOptions(normalizeDiscountOptions(options))
        }
      } catch {
        if (isActive) {
          setDiscountOptions({
            rfm_segments: [],
            outlet_classifications: [],
            slabs: [],
            matching_outlets: 0,
          })
        }
      } finally {
        if (isActive) {
          setIsDiscountOptionsLoading(false)
        }
      }
    }, 350)

    return () => {
      isActive = false
      clearTimeout(timer)
    }
  }, [lastCalculatedFilters, rfmData?.success, step2Filters, runId])

  useEffect(() => {
    const validSegments = new Set((discountOptions?.rfm_segments || []).map((x) => String(x)))
    const validOutletTypes = new Set((discountOptions?.outlet_classifications || []).map((x) => String(x)))

    setStep2Filters((prev) => {
      const nextSegments = (prev?.rfm_segments || []).filter((x) => validSegments.has(String(x)))
      const nextOutletTypes = normalizeStep2OutletClassifications(prev?.outlet_classifications || [])
        .filter((x) => validOutletTypes.has(String(x)))

      const sameSegments =
        nextSegments.length === (prev?.rfm_segments || []).length &&
        nextSegments.every((v, i) => v === (prev?.rfm_segments || [])[i])
      const sameOutletTypes =
        nextOutletTypes.length === (prev?.outlet_classifications || []).length &&
        nextOutletTypes.every((v, i) => v === (prev?.outlet_classifications || [])[i])

      if (sameSegments && sameOutletTypes && (prev?.slabs || []).length === 0) return prev

      return {
        ...prev,
        rfm_segments: nextSegments,
        outlet_classifications: nextOutletTypes,
        slabs: [],
      }
    })
  }, [
    discountOptions?.rfm_segments,
    discountOptions?.outlet_classifications,
  ])

  useEffect(() => {
    if (activeStepTab !== 'step_eda') return

    let isActive = true
    const timer = setTimeout(async () => {
      try {
        setIsEdaOptionsLoading(true)
        const options = await getEDAOptions({
          run_id: runId || undefined,
          states: [],
          categories: [],
          subcategories: [],
          brands: [],
          sizes: [],
          outlet_classifications: [],
          product_codes: [],
          top_n_products: 300,
        })
        if (isActive) {
          setEdaOptions({
            product_options: options.product_options || [],
            outlet_classifications: options.outlet_classifications || [],
            matching_rows: options.matching_rows || 0,
          })
        }
      } catch {
        // Keep last successful options in UI on transient failures/timeouts.
      } finally {
        if (isActive) {
          setIsEdaOptionsLoading(false)
        }
      }
    }, 300)

    return () => {
      isActive = false
      clearTimeout(timer)
    }
  }, [
    activeStepTab,
    runId,
  ])

  useEffect(() => {
    const edaBaseOptions = availableFilters || EMPTY_FILTERS
    const validStates = new Set((edaBaseOptions.states || []).map((x) => String(x)))
    const validCategories = new Set((edaBaseOptions.categories || []).map((x) => String(x)))
    const validSubcategories = new Set((edaBaseOptions.subcategories || []).map((x) => String(x)))
    const validBrands = new Set((edaBaseOptions.brands || []).map((x) => String(x)))
    const validSizes = new Set((edaBaseOptions.sizes || []).map((x) => String(x)))
    const validProducts = new Set((edaOptions.product_options || []).map((p) => String(p.code)))
    const validClasses = new Set((edaOptions.outlet_classifications || []).map((x) => String(x)))
    setStep5Filters((prev) => {
      const nextStates = (prev.states || []).filter((x) => validStates.has(String(x)))
      const nextCategories = (prev.categories || []).filter((x) => validCategories.has(String(x)))
      const nextSubcategories = (prev.subcategories || []).filter((x) => validSubcategories.has(String(x)))
      const nextBrands = (prev.brands || []).filter((x) => validBrands.has(String(x)))
      const nextSizes = (prev.sizes || []).filter((x) => validSizes.has(String(x)))
      const nextProducts = (prev.product_codes || []).filter((x) => validProducts.has(String(x)))
      const nextClasses = (prev.outlet_classifications || []).filter((x) => validClasses.has(String(x)))
      if (
        nextStates.length === (prev.states || []).length &&
        nextCategories.length === (prev.categories || []).length &&
        nextSubcategories.length === (prev.subcategories || []).length &&
        nextBrands.length === (prev.brands || []).length &&
        nextSizes.length === (prev.sizes || []).length &&
        nextProducts.length === (prev.product_codes || []).length &&
        nextClasses.length === (prev.outlet_classifications || []).length
      ) {
        return prev
      }
      return {
        ...prev,
        states: nextStates,
        categories: nextCategories,
        subcategories: nextSubcategories,
        brands: nextBrands,
        sizes: nextSizes,
        product_codes: nextProducts,
        outlet_classifications: nextClasses,
      }
    })
  }, [edaOptions, availableFilters])

  useEffect(() => {
    if (activeStepTab !== 'step_eda') return
    if (!rfmData?.success) return
    if (slabTrendMutation.isPending) return
    if (slabTrendResult?.success) return
    handleRunSlabTrendEDA()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeStepTab, rfmData?.success, slabTrendMutation.isPending, slabTrendResult?.success])

  useEffect(() => {
    const validResults = (modelingResult?.slab_results || []).filter((x) => x?.valid)
    const firstValid = validResults[0]
    if (!firstValid) return
    const lagFlag = Number(firstValid?.model_coefficients?.include_lag_discount)
    const nextCogsBySize = { ...DEFAULT_MODELING_COGS_BY_SIZE }
    validResults.forEach((item) => {
      const sizeKey = normalizeStep2SizeKey(item?.size || item?.model_coefficients?.size_key)
      const cogs = Number(item?.summary?.cogs_per_unit ?? item?.model_coefficients?.cogs_per_unit)
      if (sizeKey && Number.isFinite(cogs)) {
        nextCogsBySize[sizeKey] = cogs
      }
    })
    const cogs = Number(firstValid?.summary?.cogs_per_unit ?? firstValid?.model_coefficients?.cogs_per_unit)
    setModelingSettings((prev) => ({
      include_lag_discount: Number.isFinite(lagFlag) ? lagFlag > 0 : prev.include_lag_discount,
      cogs_per_unit: Number.isFinite(cogs) ? cogs : prev.cogs_per_unit,
      cogs_per_size: {
        ...DEFAULT_MODELING_COGS_BY_SIZE,
        ...(prev.cogs_per_size || {}),
        ...nextCogsBySize,
      },
      ...FIXED_STAGE3_SETTINGS,
    }))
  }, [modelingResult])

  useEffect(() => {
    if (activeStepTab !== 'step4') return
    if (!modelingResult?.success) return
    if (plannerMutation.isPending) return
    // Avoid retry loop on stale/mismatched payloads.
    // Auto-fetch only when Step 4 has no result yet.
    if (plannerResult !== null) return
    handleGeneratePlanner()
  }, [
    activeStepTab,
    modelingResult?.success,
    plannerMutation.isPending,
    plannerResult,
  ])

  useEffect(() => {
    if (activeStepTab !== 'step4') return
    if (!modelingResult?.success) return
    if (forecastMutation.isPending) return
    if (forecastErrorMessage) return
    if (forecastResult?.success && Array.isArray(forecastResult?.points) && forecastResult.points.length > 0) return
    handleRunBaselineForecast()
  }, [activeStepTab, modelingResult?.success, forecastMutation.isPending, forecastResult?.success, forecastResult?.points, forecastErrorMessage])

  useEffect(() => {
    step5AutoScenarioInitRef.current = false
  }, [runId, modelingResult, step5ScenarioDefs.length])

  useEffect(() => {
    if (activeStepTab !== 'step5') return
    if (!modelingResult?.success) return
    if (scenarioMutation.isPending) return
    if (scenarioResult?.success) return
    if (step5AutoScenarioInitRef.current) return
    step5AutoScenarioInitRef.current = true
    setScenarioErrorMessage('')
    scenarioMutation.mutate({ mode: 'fixed' })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeStepTab, modelingResult?.success, scenarioMutation.isPending, scenarioResult?.success, step5ScenarioDefs.length])

  const handleFilterChange = (key, value) => {
    setSlabTrendResult(null)
    setSlabTrendErrorMessage('')
    setFilters((prev) => {
      const newFilters = { ...prev, [key]: value }
      
      // Clear dependent filters when parent changes
      if (key === 'states') {
        newFilters.categories = []
        newFilters.subcategories = []
        newFilters.brands = []
        newFilters.sizes = []
      } else if (key === 'categories') {
        newFilters.subcategories = []
        newFilters.brands = []
        newFilters.sizes = []
      } else if (key === 'subcategories') {
        newFilters.brands = []
        newFilters.sizes = []
      } else if (key === 'brands') {
        newFilters.sizes = []
      }
      
      return newFilters
    })
  }

  if (filtersLoading) {
    return (
      <Layout>
        <div className="flex items-center justify-center h-64">
          <Loader2 className="animate-spin text-primary" size={48} />
        </div>
      </Layout>
    )
  }

  // Use cascaded filters if available, otherwise use initial filters
  const displayFilters = cascadedFilters || availableFilters
  const edaDisplayFilters = availableFilters || EMPTY_FILTERS
  const step2ActiveSizes = Array.from(
    new Set(
      ((lastCalculatedFilters?.sizes && lastCalculatedFilters.sizes.length > 0
        ? lastCalculatedFilters.sizes
        : filters.sizes) || [])
        .map((v) => normalizeStep2SizeKey(v))
        .filter(Boolean)
    )
  )
  // Right Sidebar Content
  let rightSidebarContent
  if (activeStepTab === 'step2' && rfmData?.success) {
    rightSidebarContent = (
      <div className="space-y-4">
        <DiscountStepFilters
          filters={step2Filters}
          options={discountOptions}
          onChange={handleStep2FilterChange}
          activeSizes={step2ActiveSizes}
          matchingOutlets={discountOptions.matching_outlets || 0}
          isLoading={isDiscountOptionsLoading}
          title="Step 2: Discount Analysis Filters"
          description="Select RFM groups, outlet types, and define direct slab cutoffs for 12-ML and 18-ML."
          loadingLabel="Updating Step 2 options..."
          matchingLabel="Matching outlets after Step 2 filters"
        />
        <div className="bg-white rounded-lg shadow-md p-4">
          <button
            type="button"
            onClick={handleRunBaseDepth}
            disabled={baseDepthMutation.isPending}
            className="w-full px-4 py-2 rounded-md bg-white border border-primary text-body text-sm font-semibold disabled:opacity-50"
          >
            {baseDepthMutation.isPending ? 'Estimating...' : 'Run Base Depth Estimator'}
          </button>
        </div>
      </div>
    )
  } else if (activeStepTab === 'step3' && rfmData?.success) {
    rightSidebarContent = (
      <div className="space-y-4">
        <div className="bg-white rounded-lg shadow-md overflow-visible">
          <div className="bg-primary text-white p-4">
            <h3 className="text-lg font-semibold">Step 3: Modeling Settings</h3>
          </div>
          <div className="p-4 space-y-3">
            <label className="inline-flex items-center gap-2 text-sm text-body">
              <input
                type="checkbox"
                checked={Boolean(modelingSettings.include_lag_discount)}
                onChange={(e) => setModelingSettings((prev) => ({ ...prev, include_lag_discount: e.target.checked }))}
              />
              Include lag discount term
            </label>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">COGS Per Unit - 12-ML</label>
              <input
                type="number"
                min="0"
                step="0.5"
                value={modelingSettings.cogs_per_size?.['12-ML'] ?? DEFAULT_MODELING_COGS_BY_SIZE['12-ML']}
                onChange={(e) => setModelingSettings((prev) => ({
                  ...prev,
                  cogs_per_size: {
                    ...normalizeModelingCogsBySize(prev.cogs_per_size || {}),
                    '12-ML': parseFloat(e.target.value || '0'),
                  },
                }))}
                className="w-full px-3 py-2 text-sm border border-gray-300 rounded-lg"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">COGS Per Unit - 18-ML</label>
              <input
                type="number"
                min="0"
                step="0.5"
                value={modelingSettings.cogs_per_size?.['18-ML'] ?? DEFAULT_MODELING_COGS_BY_SIZE['18-ML']}
                onChange={(e) => setModelingSettings((prev) => ({
                  ...prev,
                  cogs_per_size: {
                    ...normalizeModelingCogsBySize(prev.cogs_per_size || {}),
                    '18-ML': parseFloat(e.target.value || '0'),
                  },
                }))}
                className="w-full px-3 py-2 text-sm border border-gray-300 rounded-lg"
              />
            </div>
            <button
              type="button"
              onClick={() => handleRunModeling(modelingSettings)}
              disabled={modelingMutation.isPending}
              className="w-full px-4 py-2 rounded-md bg-white border border-primary text-body text-sm font-semibold disabled:opacity-50"
            >
              {modelingMutation.isPending ? 'Running...' : 'Run Step 3 Modeling'}
            </button>
          </div>
        </div>
      </div>
    )
  } else if (activeStepTab === 'step4' && rfmData?.success) {
    rightSidebarContent = null
  } else if (activeStepTab === 'step5') {
    rightSidebarContent = (
      <div className="bg-white rounded-lg shadow-md overflow-visible">
        <div className="bg-primary text-white p-4">
          <h3 className="text-lg font-semibold">Step 5: Scenario Settings</h3>
        </div>
        <div className="p-4 space-y-3">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Reference For % Comparison</label>
            <select
              value={step4DisplayReferenceMode}
              onChange={(e) => setStep4DisplayReferenceMode(String(e.target.value || 'ly_same_3m'))}
              className="w-full px-3 py-2 text-sm border border-gray-300 rounded-lg"
            >
              <option value="ly_same_3m">Y-o-Y</option>
              <option value="last_3m_before_projection">Q-o-Q</option>
            </select>
          </div>
          <div className="rounded-lg border border-slate-200 bg-slate-50 p-3 space-y-2">
            <div className="text-xs font-semibold text-body">Scenario Engine</div>
            <div className="text-[11px] text-muted">
              Anchors: {STEP5_ANCHOR_SCENARIOS.length}
            </div>
            <div className="text-[11px] text-muted">
              Planned scenarios: {step5ScenarioDefs.length}
            </div>
          </div>
          <button
            type="button"
            onClick={handleOpenCreateScenario}
            disabled={!modelingResult?.success}
            className="w-full px-4 py-2 rounded-md bg-white border border-slate-300 text-body text-sm font-semibold disabled:opacity-50"
          >
            Create Scenario
          </button>
          <div className="rounded-lg border border-slate-200 bg-white p-3 space-y-2">
            <div className="text-xs font-semibold text-body">AI Scenario Generator (Append)</div>
            <div>
              <label className="block text-[11px] font-medium text-gray-700 mb-1">Scenario Count (1-10000)</label>
              <input
                type="number"
                min="1"
                max="10000"
                step="1"
                value={step5AISettings.scenario_count}
                onChange={(e) => setStep5AISettings((prev) => ({
                  ...prev,
                  scenario_count: Math.min(10000, Math.max(1, Number(e.target.value || 1))),
                }))}
                className="w-full px-3 py-2 text-sm border border-gray-300 rounded-lg"
              />
            </div>
            <div>
              <label className="block text-[11px] font-medium text-gray-700 mb-1">Prompt</label>
              <textarea
                rows={3}
                value={step5AISettings.prompt}
                onChange={(e) => setStep5AISettings((prev) => ({ ...prev, prompt: String(e.target.value || '') }))}
                className="w-full px-3 py-2 text-sm border border-gray-300 rounded-lg"
                placeholder="Example: Protect margin but keep 12-ML growth positive; avoid aggressive slab jumps."
              />
            </div>
            <button
              type="button"
              onClick={handleGenerateAIScenarios}
              disabled={!modelingResult?.success || isStep5AIBusy}
              className="w-full px-4 py-2 rounded-md bg-primary text-white text-sm font-semibold disabled:opacity-50"
            >
              {isStep5AIBusy ? 'Generating...' : 'Generate AI Scenarios (Add)'}
            </button>
            <button
              type="button"
              onClick={handleDeleteAIScenarios}
              disabled={aiScenarioCount <= 0 || isStep5AIBusy}
              className="w-full px-4 py-2 rounded-md bg-white border border-slate-300 text-body text-sm font-semibold disabled:opacity-50"
            >
              Delete AI Scenarios{aiScenarioCount > 0 ? ` (${aiScenarioCount})` : ''}
            </button>
            {step5AIJob?.jobId && (
              <div className="rounded border border-slate-200 bg-slate-50 px-2 py-2 text-[11px] text-muted space-y-1">
                <div className="flex items-center justify-between">
                  <span>Status</span>
                  <span className="font-semibold text-body">{String(step5AIJob.status || '').toUpperCase()}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span>Progress</span>
                  <span className="font-semibold text-body">
                    {Number(step5AIJob.progressCurrent || 0)} / {Number(step5AIJob.progressTotal || 0)}
                  </span>
                </div>
                {step5AIJob?.errorDetail ? (
                  <div className="text-danger">{String(step5AIJob.errorDetail)}</div>
                ) : null}
              </div>
            )}
            <p className="text-[11px] text-muted">
              AI scenarios are appended to the default scenario set.
            </p>
          </div>
        </div>
      </div>
    )
  } else if (activeStepTab === 'step_eda') {
    rightSidebarContent = (
      <div className="bg-white rounded-lg shadow-md overflow-visible">
        <div className="bg-primary text-white p-4">
          <h3 className="text-lg font-semibold">Slab Trend EDA</h3>
        </div>
        <div className="p-4 space-y-3">
          <p className="text-xs text-muted">
            Uses current Step 1 filters and Step 2 slab definition to plot month-wise slab discount and volume trends.
          </p>
          <button
            type="button"
            onClick={handleRunSlabTrendEDA}
            disabled={slabTrendMutation.isPending || !rfmData?.success}
            className="w-full px-4 py-2 rounded-md bg-white border border-primary text-body text-sm font-semibold disabled:opacity-50"
          >
            {slabTrendMutation.isPending ? 'Loading...' : 'Refresh EDA'}
          </button>
        </div>
      </div>
    )
  } else {
    rightSidebarContent = (
      <FilterPanel
        filters={filters}
        availableFilters={displayFilters}
        onFilterChange={handleFilterChange}
        onCalculate={handleCalculate}
        isCalculating={calculateMutation.isPending}
        isCascadeLoading={isCascadeLoading}
      />
    )
  }

  return (
    <Layout rightSidebar={rightSidebarContent}>
      <div className="space-y-6">
        {/* Error Message */}
        {calculateMutation.isError && (
          <div className="bg-brand-dangerLight border border-danger rounded-lg p-4 flex items-start space-x-3">
            <AlertCircle className="text-danger flex-shrink-0 mt-0.5" size={20} />
            <div>
              <h4 className="font-semibold text-body">Error</h4>
              <p className="text-muted text-sm">
                {calculateMutation.error?.message || 'Failed to calculate RFM'}
              </p>
            </div>
          </div>
        )}

        <div className="bg-white rounded-lg shadow-md p-3">
          <div className="flex gap-2">
            <button
              type="button"
              onClick={() => setStepTab('step1')}
              className={`px-4 py-2 rounded-md text-sm font-semibold border ${
                activeStepTab === 'step1'
                  ? 'bg-white text-body border-primary'
                  : 'bg-white text-muted border-gray-300'
              }`}
            >
              Step 1: Store Segmentation
            </button>
            <button
              type="button"
              onClick={() => setStepTab('step2')}
              className={`px-4 py-2 rounded-md text-sm font-semibold border ${
                activeStepTab === 'step2'
                  ? 'bg-white text-body border-primary'
                  : 'bg-white text-muted border-gray-300'
              }`}
            >
              Step 2: Discount Analysis
            </button>
            <button
              type="button"
              onClick={() => setStepTab('step3')}
              className={`px-4 py-2 rounded-md text-sm font-semibold border ${
                activeStepTab === 'step3'
                  ? 'bg-white text-body border-primary'
                  : 'bg-white text-muted border-gray-300'
              }`}
            >
              Step 3: Modeling & ROI
            </button>
            <button
              type="button"
              onClick={() => setStepTab('step4')}
              className={`px-4 py-2 rounded-md text-sm font-semibold border ${
                activeStepTab === 'step4'
                  ? 'bg-white text-body border-primary'
                  : 'bg-white text-muted border-gray-300'
              }`}
            >
              Step 4: Cross-Size Planner
            </button>
            <button
              type="button"
              onClick={() => setStepTab('step5')}
              className={`px-4 py-2 rounded-md text-sm font-semibold border ${
                activeStepTab === 'step5'
                  ? 'bg-white text-body border-primary'
                  : 'bg-white text-muted border-gray-300'
              }`}
            >
              Step 5: Scenario Comparison
            </button>
            <button
              type="button"
              onClick={() => setStepTab('step_eda')}
              className={`px-4 py-2 rounded-md text-sm font-semibold border ${
                activeStepTab === 'step_eda'
                  ? 'bg-white text-body border-primary'
                  : 'bg-white text-muted border-gray-300'
              }`}
            >
              Step 6: Slab Trend EDA
            </button>
          </div>
        </div>

        {activeStepTab === 'step1' && rfmData && rfmData.success && (
          <div className="space-y-6">
            <RFMSummary data={rfmData} />

            {rfmData.cluster_summary && (
              <ClusterSummary data={rfmData.cluster_summary} />
            )}

            {rfmData.segment_summary && (
              <SegmentGrid segments={rfmData.segment_summary} />
            )}

            {rfmData.rfm_data && (
              <OutletTable
                outlets={rfmData.rfm_data}
                totalOutlets={rfmData.total_filtered_outlets || rfmData.total_outlets || 0}
                page={rfmData.page || tableQuery.page}
                pageSize={rfmData.page_size || tableQuery.page_size}
                totalPages={rfmData.total_pages || 1}
                isLoading={calculateMutation.isPending}
                onQueryChange={handleTableQueryChange}
                onExport={handleExport}
              />
            )}

            <div className="bg-white rounded-lg shadow-md p-4 flex items-center justify-between">
              <div>
                <h4 className="text-base font-semibold text-body">Step 1 Completed</h4>
                <p className="text-sm text-muted">Proceed to Base Discount Estimator for discount analysis.</p>
              </div>
              <button
                type="button"
                onClick={() => setStepTab('step2')}
                className="px-4 py-2 rounded-md bg-white text-body border border-primary text-sm font-semibold"
              >
                Go To Base Discount Estimator
              </button>
            </div>
          </div>
        )}

        {activeStepTab === 'step2' && rfmData && rfmData.success && (
          <div className="space-y-6">
            <Step2SlabDefinitionPanel
              filters={step2Filters}
              onChange={handleStep2FilterChange}
              activeSizes={step2ActiveSizes}
              title="Step 2: Size-wise Slab Definition"
            />
            <BaseDepthEstimator
              config={baseDepthConfig}
              onConfigChange={handleBaseDepthConfigChange}
              onRun={handleRunBaseDepth}
              data={baseDepthResult}
              isLoading={baseDepthMutation.isPending}
              isError={baseDepthMutation.isError}
              errorMessage={baseDepthErrorMessage || baseDepthMutation.error?.message}
              showControls={false}
              definedSlabProfiles={step2Filters?.defined_slab_profiles || {}}
            />

            {baseDepthResult?.success && (
              <div className="bg-white rounded-lg shadow-md p-4 flex items-center justify-between">
                <div>
                  <h4 className="text-base font-semibold text-body">Step 2 Completed</h4>
                  <p className="text-sm text-muted">Run slab-wise modeling and ROI in Step 3.</p>
                </div>
                <button
                  type="button"
                  onClick={() => setStepTab('step3')}
                  className="px-4 py-2 rounded-md bg-white text-body border border-primary text-sm font-semibold"
                >
                  Go To Step 3 Modeling
                </button>
              </div>
            )}
          </div>
        )}

        {activeStepTab === 'step3' && rfmData && rfmData.success && (
          <div className="space-y-6">
            {!baseDepthResult?.success && (
              <div className="bg-white rounded-lg shadow-md p-8">
                <h3 className="text-xl font-semibold text-body mb-2">Step 3 Requires Step 2 Output</h3>
                <p className="text-muted mb-4">
                  Run Base Discount Estimator first, then continue to Modeling and ROI.
                </p>
                <button
                  type="button"
                  onClick={() => setStepTab('step2')}
                  className="px-4 py-2 rounded-md bg-white text-body border border-primary text-sm font-semibold"
                >
                  Go To Step 2
                </button>
              </div>
            )}

            <ModelingROI
              data={modelingResult}
              isLoading={modelingMutation.isPending}
              isError={modelingMutation.isError || Boolean(modelingErrorMessage)}
              errorMessage={modelingErrorMessage || modelingMutation.error?.message}
              onRun={handleRunModeling}
              settings={modelingSettings}
              showControls={false}
            />

            {modelingResult?.success && (
              <div className="bg-white rounded-lg shadow-md p-4 flex items-center justify-between">
                <div>
                  <h4 className="text-base font-semibold text-body">Step 3 Completed</h4>
                  <p className="text-sm text-muted">Proceed to the cross-size scenario planner for 12-ML and 18-ML planning.</p>
                </div>
                <button
                  type="button"
                  onClick={() => setStepTab('step4')}
                  className="px-4 py-2 rounded-md bg-white text-body border border-primary text-sm font-semibold"
                >
                  Go To Step 4 Planner
                </button>
              </div>
            )}
          </div>
        )}

        {activeStepTab === 'step4' && rfmData && rfmData.success && (
          <div className="space-y-6">
            {!modelingResult?.success && (
              <div className="bg-white rounded-lg shadow-md p-8">
                <h3 className="text-xl font-semibold text-body mb-2">Step 4 Requires Step 3 Output</h3>
                <p className="text-muted mb-4">
                  Run Modeling and ROI first, then continue to the cross-size scenario planner.
                </p>
                <button
                  type="button"
                  onClick={() => setStepTab('step3')}
                  className="px-4 py-2 rounded-md bg-white text-body border border-primary text-sm font-semibold"
                >
                  Go To Step 3
                </button>
              </div>
            )}

            <CrossSizePlanner
              data={plannerResult}
              isLoading={plannerMutation.isPending}
              isError={plannerMutation.isError || Boolean(plannerErrorMessage)}
              errorMessage={plannerErrorMessage || plannerMutation.error?.message}
              displayReferenceMode={step4DisplayReferenceMode}
              onDisplayReferenceModeChange={setStep4DisplayReferenceMode}
              referenceByMode={plannerDefaultByReference}
              onInitialize={() => {
                handleGeneratePlanner()
              }}
            />

            <BaselineForecast
              data={forecastResult}
              isLoading={forecastMutation.isPending}
              isError={forecastMutation.isError || Boolean(forecastErrorMessage)}
              errorMessage={forecastErrorMessage || forecastMutation.error?.message}
            />

            {plannerResult?.success && (
              <div className="bg-white rounded-lg shadow-md p-4 flex items-center justify-between">
                <div>
                  <h4 className="text-base font-semibold text-body">Step 4 Completed</h4>
                  <p className="text-sm text-muted">Cross-size planner and 3-month baseline forecast are ready. Proceed to scenario comparison next.</p>
                </div>
                <button
                  type="button"
                  onClick={() => setStepTab('step5')}
                  className="px-4 py-2 rounded-md bg-white text-body border border-primary text-sm font-semibold"
                >
                  Open Step 5
                </button>
              </div>
            )}
          </div>
        )}

        {activeStepTab === 'step5' && (
          <div className="space-y-6">
            {!modelingResult?.success && (
              <div className="bg-white rounded-lg shadow-md p-8">
                <h3 className="text-xl font-semibold text-body mb-2">Step 5 Requires Step 3 Output</h3>
                <p className="text-muted mb-4">
                  Run Modeling and ROI first, then generate Step 5 scenarios.
                </p>
                <button
                  type="button"
                  onClick={() => setStepTab('step3')}
                  className="px-4 py-2 rounded-md bg-white text-body border border-primary text-sm font-semibold"
                >
                  Go To Step 3
                </button>
              </div>
            )}
            <ScenarioComparison
              data={scenarioResult}
              plannerBase={plannerResult}
              isLoading={scenarioMutation.isPending}
              isError={scenarioMutation.isError || Boolean(scenarioErrorMessage)}
              errorMessage={scenarioErrorMessage || scenarioMutation.error?.message}
              createScenarioRequestId={step5CreateScenarioRequestId}
              onGenerateForCurrentFilters={handleGenerateAIScenariosForCurrentFilters}
              isAIGenerating={isStep5AIBusy}
              onFilterContextChange={setStep5CurrentFilterContext}
              onScenariosChange={(nextRows) => {
                setScenarioResult((prev) => ({
                  ...(prev || {}),
                  success: true,
                  scenarios: Array.isArray(nextRows) ? nextRows : [],
                  message: prev?.message || `Generated ${Array.isArray(nextRows) ? nextRows.length : 0} scenario(s).`,
                }))
              }}
            />
          </div>
        )}

        {activeStepTab === 'step_eda' && rfmData && rfmData.success && (
          <SlabTrendEDA
            data={slabTrendResult}
            isLoading={slabTrendMutation.isPending}
            isError={slabTrendMutation.isError || Boolean(slabTrendErrorMessage)}
            errorMessage={slabTrendErrorMessage || slabTrendMutation.error?.message}
          />
        )}

        {activeStepTab === 'step1' && !rfmData && !calculateMutation.isPending && (
          <div className="bg-white rounded-lg shadow-md p-12 text-center">
            <div className="max-w-md mx-auto">
              <div className="text-subtle mb-4">
                <BarChart3 size={64} className="mx-auto" />
              </div>
              <h3 className="text-xl font-semibold text-body mb-2">
                No Segmentation Analysis Yet
              </h3>
              <p className="text-muted mb-6">
                Configure your filters in the Settings panel on the right and click "Calculate RFM" to start the analysis
              </p>
            </div>
          </div>
        )}

        {activeStepTab === 'step2' && !rfmData && !calculateMutation.isPending && (
          <div className="bg-white rounded-lg shadow-md p-8">
            <h3 className="text-xl font-semibold text-body mb-2">Step 2 Requires Step 1 Output</h3>
            <p className="text-muted mb-4">
              Run Store Segmentation first, then continue to Base Discount Estimator.
            </p>
            <button
              type="button"
              onClick={() => setStepTab('step1')}
              className="px-4 py-2 rounded-md bg-white text-body border border-primary text-sm font-semibold"
            >
              Go To Step 1
            </button>
          </div>
        )}

        {activeStepTab === 'step3' && !rfmData && !calculateMutation.isPending && (
          <div className="bg-white rounded-lg shadow-md p-8">
            <h3 className="text-xl font-semibold text-body mb-2">Step 3 Requires Step 1 Output</h3>
            <p className="text-muted mb-4">
              Run Store Segmentation first, then proceed through Step 2 before modeling.
            </p>
            <button
              type="button"
              onClick={() => setStepTab('step1')}
              className="px-4 py-2 rounded-md bg-white text-body border border-primary text-sm font-semibold"
            >
              Go To Step 1
            </button>
          </div>
        )}

        {activeStepTab === 'step4' && !rfmData && !calculateMutation.isPending && (
          <div className="bg-white rounded-lg shadow-md p-8">
            <h3 className="text-xl font-semibold text-body mb-2">Step 4 Requires Step 1 Output</h3>
            <p className="text-muted mb-4">
              Run Store Segmentation first, then complete Steps 2 and 3 before planning.
            </p>
            <button
              type="button"
              onClick={() => setStepTab('step1')}
              className="px-4 py-2 rounded-md bg-white text-body border border-primary text-sm font-semibold"
            >
              Go To Step 1
            </button>
          </div>
        )}

        {activeStepTab === 'step5' && !rfmData && !calculateMutation.isPending && (
          <div className="bg-white rounded-lg shadow-md p-8">
            <h3 className="text-xl font-semibold text-body mb-2">Step 5 Requires Step 1 Output</h3>
            <p className="text-muted mb-4">
              Run Store Segmentation first, then complete Steps 2-4 before scenario comparison.
            </p>
            <button
              type="button"
              onClick={() => setStepTab('step1')}
              className="px-4 py-2 rounded-md bg-white text-body border border-primary text-sm font-semibold"
            >
              Go To Step 1
            </button>
          </div>
        )}

        {activeStepTab === 'step_eda' && !rfmData && !calculateMutation.isPending && (
          <div className="bg-white rounded-lg shadow-md p-8">
            <h3 className="text-xl font-semibold text-body mb-2">Step 6 Requires Step 1 Output</h3>
            <p className="text-muted mb-4">
              Run Store Segmentation first, then open Slab Trend EDA.
            </p>
            <button
              type="button"
              onClick={() => setStepTab('step1')}
              className="px-4 py-2 rounded-md bg-white text-body border border-primary text-sm font-semibold"
            >
              Go To Step 1
            </button>
          </div>
        )}

      </div>
    </Layout>
  )
}

export default RFMAnalysis
