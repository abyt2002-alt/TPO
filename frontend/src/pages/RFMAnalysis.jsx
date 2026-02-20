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
  calculate12MonthPlanner,
  comparePlannerScenarios,
  getEDAOptions,
  getEDAOverview,
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
import Planner12Month from '../components/rfm/Planner12Month'
import ScenarioComparison from '../components/rfm/ScenarioComparison'
import EDAInsights from '../components/rfm/EDAInsights'
import DiscountStepFilters from '../components/rfm/DiscountStepFilters'
import { Loader2, AlertCircle, BarChart3, ChevronDown, ChevronUp, Search } from 'lucide-react'

const DEFAULT_STEP2_FILTERS = {
  rfm_segments: [],
  outlet_classifications: [],
  slabs: [],
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
      return idx !== null && idx >= 1 && idx <= 4
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
  slabs: normalizeStep2Slabs(filters?.slabs || []),
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
  sizes: ['18-ML'],
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

const resolveStepTabFromQuery = (stepParam) => {
  if (stepParam === '2') return 'step2'
  if (stepParam === '3') return 'step3'
  if (stepParam === '4') return 'step4'
  if (stepParam === '5') return 'step5'
  return 'step1'
}

const resolveStepQueryFromTab = (stepTab) => {
  if (stepTab === 'step2') return '2'
  if (stepTab === 'step3') return '3'
  if (stepTab === 'step4') return '4'
  if (stepTab === 'step5') return '5'
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

const parseTrinityInsightBullets = (insightText) => {
  const lines = String(insightText || '')
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean)
    .filter((line) => !line.startsWith('###'))

  const bulletLines = lines
    .filter((line) => line.startsWith('-'))
    .map((line) => line.replace(/^-+\s*/, '').trim())

  if (bulletLines.length > 0) return bulletLines.slice(0, 6)
  if (lines.length > 0) return lines.slice(0, 6)
  return []
}

const splitInsightLabel = (line) => {
  const boldMatch = line.match(/^\*\*(.+?)\*\*\s*:?\s*(.*)$/)
  if (boldMatch) {
    return {
      label: String(boldMatch[1] || '').trim(),
      detail: String(boldMatch[2] || '').trim(),
    }
  }
  const idx = line.indexOf(':')
  if (idx > 0 && idx < 42) {
    return {
      label: line.slice(0, idx).trim(),
      detail: line.slice(idx + 1).trim(),
    }
  }
  return { label: '', detail: line }
}

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
  const [selectedPlannerSlab, setSelectedPlannerSlab] = useState('')
  const [scenarioResult, setScenarioResult] = useState(null)
  const [scenarioErrorMessage, setScenarioErrorMessage] = useState('')
  const [scenarioFile, setScenarioFile] = useState(null)
  const [step5Filters, setStep5Filters] = useState(DEFAULT_STEP5_FILTERS)
  const [edaOptions, setEdaOptions] = useState({
    product_options: [],
    outlet_classifications: [],
    matching_rows: 0,
  })
  const [isEdaOptionsLoading, setIsEdaOptionsLoading] = useState(false)
  const [edaResult, setEdaResult] = useState(null)
  const [edaErrorMessage, setEdaErrorMessage] = useState('')
  const [modelingSettings, setModelingSettings] = useState({
    include_lag_discount: true,
    cogs_per_unit: 0,
    ...FIXED_STAGE3_SETTINGS,
  })
  const [plannerInputs, setPlannerInputs] = useState({
    planned_structural_discounts: [],
    planned_base_prices: [],
  })

  // Fetch initial available filters
  const { data: availableFilters, isLoading: filtersLoading } = useQuery({
    queryKey: ['filters'],
    queryFn: getAvailableFilters,
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
      setBaseDepthErrorMessage(error?.message || 'Failed to estimate base depth')
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
    mutationFn: calculate12MonthPlanner,
    onSuccess: (data) => {
      setPlannerResult(data)
      setPlannerErrorMessage('')
      if (data?.slab) {
        setSelectedPlannerSlab(String(data.slab))
      }
    },
    onError: (error) => {
      setPlannerErrorMessage(error?.message || 'Failed to run Step 4 planner')
    },
  })

  const scenarioMutation = useMutation({
    mutationFn: comparePlannerScenarios,
    onSuccess: (data) => {
      setScenarioResult(data)
      setScenarioErrorMessage('')
    },
    onError: (error) => {
      setScenarioErrorMessage(error?.message || 'Failed to run Step 5 scenario comparison')
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
        const hasSavedStep1Selection = ['states', 'categories', 'subcategories', 'brands', 'sizes']
          .some((key) => Array.isArray(restoredFilters[key]) && restoredFilters[key].length > 0)
        const restoredTableQuery = state.table_query || {}
        const restoredStep2 = state.step2_filters || {}
        const restoredConfig = state.base_depth_config || {}
        const restoredUi = state.ui_state || {}
        const restoredStep5Filters = restoredUi.step5_filters || {}

        setFilters((prev) => ({
          ...prev,
          ...restoredFilters,
          states: restoredFilters.states || (hasSavedStep1Selection ? [] : DEFAULT_STEP1_FILTERS.states),
          categories: restoredFilters.categories || (hasSavedStep1Selection ? [] : DEFAULT_STEP1_FILTERS.categories),
          subcategories: restoredFilters.subcategories || (hasSavedStep1Selection ? [] : DEFAULT_STEP1_FILTERS.subcategories),
          brands: restoredFilters.brands || (hasSavedStep1Selection ? [] : DEFAULT_STEP1_FILTERS.brands),
          sizes: restoredFilters.sizes || (hasSavedStep1Selection ? [] : DEFAULT_STEP1_FILTERS.sizes),
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
          if (restored.step4_result?.slab) {
            setSelectedPlannerSlab(String(restored.step4_result.slab))
          }
        } else if (state?.step4_result) {
          setPlannerResult(state.step4_result)
          if (state.step4_result?.slab) {
            setSelectedPlannerSlab(String(state.step4_result.slab))
          }
        }
        if (restored?.step5_result) {
          setScenarioResult(restored.step5_result)
        } else if (state?.step5_result) {
          setScenarioResult(state.step5_result)
        }

        const stepFromUrl = resolveStepTabFromQuery(initialParams.get('step'))
        const restoredStep = ['step1', 'step2', 'step3', 'step4', 'step5'].includes(state.active_step) ? state.active_step : 'step1'
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

    let isActive = true
    const timer = setTimeout(async () => {
      try {
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
          },
        })
      } catch {
        if (!isActive) return
      }
    }, 500)

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
      ...FIXED_STAGE3_SETTINGS,
    })
    setPlannerResult(null)
    setPlannerErrorMessage('')
    setSelectedPlannerSlab('')
    setScenarioResult(null)
    setScenarioErrorMessage('')
    setScenarioFile(null)
    setPlannerInputs({
      planned_structural_discounts: [],
      planned_base_prices: [],
    })
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
    setScenarioResult(null)
    setScenarioErrorMessage('')
  }

  const handleRunBaseDepth = () => {
    const baseFilters = lastCalculatedFilters || filters
    setModelingResult(null)
    setModelingErrorMessage('')
    setPlannerResult(null)
    setPlannerErrorMessage('')
    setScenarioResult(null)
    setScenarioErrorMessage('')
    const payload = {
      run_id: runId || undefined,
      ...baseFilters,
      ...baseDepthConfig,
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
      ...FIXED_STAGE3_SETTINGS,
    }
    setModelingSettings((prev) => ({
      ...prev,
      include_lag_discount: Boolean(effectiveSettings.include_lag_discount),
      cogs_per_unit: Number(effectiveSettings.cogs_per_unit || 0),
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
      ...step2Filters,
    }
    setPlannerResult(null)
    setPlannerErrorMessage('')
    setScenarioResult(null)
    setScenarioErrorMessage('')
    modelingMutation.mutate(payload)
  }

  const availablePlannerSlabs = (modelingResult?.slab_results || [])
    .filter((x) => x?.valid)
    .map((x) => String(x.slab))

  const resolvePlannerSlab = () => {
    if (selectedPlannerSlab && availablePlannerSlabs.includes(selectedPlannerSlab)) {
      return selectedPlannerSlab
    }
    return availablePlannerSlabs[0] || ''
  }

  const sanitizeArray = (values, fallback = []) => {
    const source = Array.isArray(values) ? values : fallback
    return source.map((value, index) => {
      const parsed = Number(value)
      if (Number.isFinite(parsed)) return parsed
      const fallbackValue = Number((Array.isArray(fallback) ? fallback[index] : 0) || 0)
      return Number.isFinite(fallbackValue) ? fallbackValue : 0
    })
  }

  const handleGeneratePlanner = () => {
    if (!modelingResult?.success) {
      setPlannerErrorMessage('Run Step 3 modeling before Step 4 planner.')
      return
    }
    const slab = resolvePlannerSlab()
    if (!slab) {
      setPlannerErrorMessage('No valid slab available for planner.')
      return
    }
    const baseFilters = lastCalculatedFilters || filters
    plannerMutation.mutate({
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
      ...step2Filters,
      slab,
      slabs: [slab],
    })
  }

  const handleRecalculatePlanner = (inputs) => {
    const slab = resolvePlannerSlab()
    if (!slab) return
    const baseFilters = lastCalculatedFilters || filters
    const fallbackStruct = plannerResult?.planned_structural_discounts || []
    const fallbackBase = plannerResult?.planned_base_prices || []
    const nextStruct = sanitizeArray(inputs?.planned_structural_discounts, fallbackStruct)
    const nextBase = sanitizeArray(inputs?.planned_base_prices, fallbackBase)
    plannerMutation.mutate({
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
      ...step2Filters,
      slab,
      slabs: [slab],
      planned_structural_discounts: nextStruct,
      planned_base_prices: nextBase,
    })
  }

  const handleRunScenarioComparison = () => {
    if (!modelingResult?.success) {
      setScenarioErrorMessage('Run Step 3 modeling before Step 5 scenario comparison.')
      return
    }
    const slab = resolvePlannerSlab()
    if (!slab) {
      setScenarioErrorMessage('No valid slab available for scenario comparison.')
      return
    }
    if (!scenarioFile) {
      setScenarioErrorMessage('Upload a CSV/XLSX file with month-wise scenarios first.')
      return
    }
    const baseFilters = lastCalculatedFilters || filters
    const payload = {
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
      ...step2Filters,
      slab,
      slabs: [slab],
      disable_ai_insights: true,
    }
    scenarioMutation.mutate({ payload, file: scenarioFile })
  }

  const handleStep2FilterChange = (key, value) => {
    setModelingResult(null)
    setModelingErrorMessage('')
    setPlannerResult(null)
    setPlannerErrorMessage('')
    setScenarioResult(null)
    setScenarioErrorMessage('')
    setStep2Filters((prev) => {
      let normalizedValue = value
      if (key === 'outlet_classifications') {
        normalizedValue = normalizeStep2OutletClassifications(value || [])
      } else if (key === 'slabs') {
        normalizedValue = normalizeStep2Slabs(value || [], discountOptions?.slabs || [])
      }
      const next = { ...prev, [key]: normalizedValue }
      if (key === 'rfm_segments') {
        next.outlet_classifications = []
        next.slabs = []
      } else if (key === 'outlet_classifications') {
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
    const validSlabs = normalizeStep2Slabs(discountOptions?.slabs || [])

    setStep2Filters((prev) => {
      const nextSegments = (prev?.rfm_segments || []).filter((x) => validSegments.has(String(x)))
      const nextOutletTypes = normalizeStep2OutletClassifications(prev?.outlet_classifications || [])
        .filter((x) => validOutletTypes.has(String(x)))

      let nextSlabs = normalizeStep2Slabs(prev?.slabs || [], validSlabs)
      if (nextSlabs.length === 0 && validSlabs.length > 0) {
        // Default Step 2 slab selection: slab1-slab4 (slab0 hidden).
        nextSlabs = [...validSlabs]
      }

      const sameSegments =
        nextSegments.length === (prev?.rfm_segments || []).length &&
        nextSegments.every((v, i) => v === (prev?.rfm_segments || [])[i])
      const sameOutletTypes =
        nextOutletTypes.length === (prev?.outlet_classifications || []).length &&
        nextOutletTypes.every((v, i) => v === (prev?.outlet_classifications || [])[i])
      const sameSlabs =
        nextSlabs.length === (prev?.slabs || []).length &&
        nextSlabs.every((v, i) => v === (prev?.slabs || [])[i])

      if (sameSegments && sameOutletTypes && sameSlabs) return prev

      return {
        ...prev,
        rfm_segments: nextSegments,
        outlet_classifications: nextOutletTypes,
        slabs: nextSlabs,
      }
    })
  }, [
    discountOptions?.rfm_segments,
    discountOptions?.outlet_classifications,
    discountOptions?.slabs,
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
    const slabs = (modelingResult?.slab_results || []).filter((x) => x?.valid).map((x) => String(x.slab))
    if (!slabs.length) {
      setSelectedPlannerSlab('')
      return
    }
    setSelectedPlannerSlab((prev) => (prev && slabs.includes(prev) ? prev : slabs[0]))
  }, [modelingResult])

  useEffect(() => {
    const firstValid = (modelingResult?.slab_results || []).find((x) => x?.valid)
    if (!firstValid) return
    const lagFlag = Number(firstValid?.model_coefficients?.include_lag_discount)
    const cogs = Number(firstValid?.summary?.cogs_per_unit ?? firstValid?.model_coefficients?.cogs_per_unit)
    setModelingSettings((prev) => ({
      include_lag_discount: Number.isFinite(lagFlag) ? lagFlag > 0 : prev.include_lag_discount,
      cogs_per_unit: Number.isFinite(cogs) ? cogs : prev.cogs_per_unit,
      ...FIXED_STAGE3_SETTINGS,
    }))
  }, [modelingResult])

  useEffect(() => {
    if (!plannerResult?.success) return
    setPlannerInputs({
      planned_structural_discounts: plannerResult?.planned_structural_discounts || [],
      planned_base_prices: plannerResult?.planned_base_prices || [],
    })
  }, [plannerResult])

  const handleFilterChange = (key, value) => {
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
  const trinityStatus = String(plannerResult?.ai_insights_status || 'pending_recalculate')
  const trinityStatusLabel = trinityStatus === 'pending_recalculate'
    ? 'After Recalculate'
    : trinityStatus.replace(/_/g, ' ')
  const trinityBullets = parseTrinityInsightBullets(plannerResult?.ai_insights)
  const trinityStatusClass = trinityStatus === 'ready'
    ? 'text-green-700'
    : trinityStatus === 'disabled' || trinityStatus === 'error'
      ? 'text-red-600'
      : 'text-slate-500'

  // Right Sidebar Content
  let rightSidebarContent
  if (activeStepTab === 'step2' && rfmData?.success) {
    rightSidebarContent = (
      <div className="space-y-4">
        <DiscountStepFilters
          filters={step2Filters}
          options={discountOptions}
          onChange={handleStep2FilterChange}
          matchingOutlets={discountOptions.matching_outlets || 0}
          isLoading={isDiscountOptionsLoading}
          title="Step 2: Discount Analysis Filters"
          description="Select RFM groups, outlet types, and slabs for base discount estimation."
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
              <label className="block text-sm font-medium text-gray-700 mb-1">COGS Per Unit</label>
              <input
                type="number"
                min="0"
                step="0.5"
                value={modelingSettings.cogs_per_unit}
                onChange={(e) => setModelingSettings((prev) => ({ ...prev, cogs_per_unit: parseFloat(e.target.value || '0') }))}
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
    rightSidebarContent = (
      <div className="bg-white rounded-lg shadow-md overflow-visible">
        <div className="bg-primary text-white p-4">
          <h3 className="text-lg font-semibold">Step 4: Planner Settings</h3>
        </div>
        <div className="p-4 space-y-3">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Selected Slab</label>
            <select
              value={resolvePlannerSlab()}
              onChange={(e) => {
                setSelectedPlannerSlab(e.target.value)
                setPlannerResult(null)
                setPlannerErrorMessage('')
                setPlannerInputs({
                  planned_structural_discounts: [],
                  planned_base_prices: [],
                })
              }}
              className="w-full px-3 py-2 text-sm border border-gray-300 rounded-lg"
            >
              <option value="">Select slab</option>
              {availablePlannerSlabs.map((s) => (
                <option key={s} value={s}>{s}</option>
              ))}
            </select>
          </div>

          <button
            type="button"
            onClick={handleGeneratePlanner}
            disabled={plannerMutation.isPending || !resolvePlannerSlab()}
            className="w-full px-4 py-2 rounded-md bg-white border border-primary text-body text-sm font-semibold disabled:opacity-50"
          >
            {plannerMutation.isPending ? 'Loading Plan...' : 'Generate Step 4 Plan'}
          </button>
          {plannerMutation.isPending && (
            <p className="text-xs text-muted">
              Planner calculation is running. This can take around 30-60 seconds for large data.
            </p>
          )}

          <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
            <div className="flex items-center justify-between gap-2">
              <p className="text-sm font-semibold text-body">Trinity Insights</p>
              <span className={`text-[10px] uppercase tracking-wide font-semibold ${trinityStatusClass}`}>
                {trinityStatusLabel}
              </span>
            </div>
            {trinityStatus === 'pending_recalculate' ? (
              <p className="text-xs text-muted mt-2">
                Trinity insights are generated only after you click Recalculate Plan.
              </p>
            ) : trinityBullets.length > 0 ? (
              <ul className="mt-2 space-y-2">
                {trinityBullets.map((line, index) => {
                  const parts = splitInsightLabel(line)
                  return (
                    <li key={`${line}-${index}`} className="text-xs text-body leading-5">
                      <span className="font-semibold">{parts.label ? `${parts.label}:` : 'Insight:'}</span>{' '}
                      <span className="text-muted">{parts.detail || line}</span>
                    </li>
                  )
                })}
              </ul>
            ) : (
              <p className="text-xs text-muted mt-2">
                Trinity insights will appear after you click Recalculate Plan.
              </p>
            )}
          </div>
        </div>
      </div>
    )
  } else if (activeStepTab === 'step5') {
    rightSidebarContent = (
      <div className="bg-white rounded-lg shadow-md overflow-visible">
        <div className="bg-primary text-white p-4">
          <h3 className="text-lg font-semibold">Step 5: Scenario Settings</h3>
        </div>
        <div className="p-4 space-y-3">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Selected Slab</label>
            <select
              value={resolvePlannerSlab()}
              onChange={(e) => {
                setSelectedPlannerSlab(e.target.value)
                setScenarioResult(null)
                setScenarioErrorMessage('')
              }}
              className="w-full px-3 py-2 text-sm border border-gray-300 rounded-lg"
            >
              <option value="">Select slab</option>
              {availablePlannerSlabs.map((s) => (
                <option key={s} value={s}>{s}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Scenario File (CSV/XLSX)</label>
            <input
              type="file"
              accept=".csv,.xlsx,.xls"
              onChange={(e) => {
                const file = e.target.files?.[0] || null
                setScenarioFile(file)
                setScenarioResult(null)
                setScenarioErrorMessage('')
              }}
              className="w-full text-sm"
            />
          </div>
          <p className="text-xs text-muted">
            Format: first column = Month (April to March), other columns = Scenario 1..N with % values.
          </p>
          <button
            type="button"
            onClick={handleRunScenarioComparison}
            disabled={scenarioMutation.isPending || !resolvePlannerSlab() || !scenarioFile}
            className="w-full px-4 py-2 rounded-md bg-white border border-primary text-body text-sm font-semibold disabled:opacity-50"
          >
            {scenarioMutation.isPending ? 'Running Comparison...' : 'Run Comparison'}
          </button>
          {scenarioFile && (
            <p className="text-xs text-muted">
              Uploaded: <span className="font-semibold text-body">{scenarioFile.name}</span>
            </p>
          )}
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
              Step 4: 12-Month Planner
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
            <BaseDepthEstimator
              config={baseDepthConfig}
              onConfigChange={handleBaseDepthConfigChange}
              onRun={handleRunBaseDepth}
              data={baseDepthResult}
              isLoading={baseDepthMutation.isPending}
              isError={baseDepthMutation.isError}
              errorMessage={baseDepthErrorMessage || baseDepthMutation.error?.message}
              showControls={false}
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
                  <p className="text-sm text-muted">Proceed to 12-month planner for slab-level promo planning.</p>
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
                  Run Modeling and ROI first, then continue to 12-month planner.
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

            <Planner12Month
              data={plannerResult}
              isLoading={plannerMutation.isPending}
              isError={plannerMutation.isError || Boolean(plannerErrorMessage)}
              errorMessage={plannerErrorMessage || plannerMutation.error?.message}
              onRecalculate={handleRecalculatePlanner}
              showControls
              showSlabGenerateControls={false}
              showCogsInput={false}
              showMonthEditor
              showResetButton
              fixedCogsPerUnit={Number(modelingSettings.cogs_per_unit || 0)}
            />

            {plannerResult?.success && (
              <div className="bg-white rounded-lg shadow-md p-4 flex items-center justify-between">
                <div>
                  <h4 className="text-base font-semibold text-body">Step 4 Completed</h4>
                  <p className="text-sm text-muted">Proceed to Step 5 for scenario upload and comparison.</p>
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
                  Run Modeling and ROI first, then upload scenarios for comparison.
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
              isLoading={scenarioMutation.isPending}
              isError={scenarioMutation.isError || Boolean(scenarioErrorMessage)}
              errorMessage={scenarioErrorMessage || scenarioMutation.error?.message}
            />
          </div>
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

      </div>
    </Layout>
  )
}

export default RFMAnalysis
