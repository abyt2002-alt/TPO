import { useEffect, useMemo, useRef, useState } from 'react'
import { AlertCircle, Download, Loader2, Save, X } from 'lucide-react'
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  LabelList,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ReferenceLine,
} from 'recharts'
import { computeCrossSizePlannerData, normalizePlannerPeriodsFromData } from '../../utils/crossSizePlannerCompute'

const FRIENDLY_SCENARIO_NAMES = {
  latest_month_discount: 'Latest Month Discount',
  last_3m_exact: 'Last 3 Months Exact',
  same_season_last_year: 'Same Season Last Year',
  most_common_historical_ladder: 'Most Common Historical Ladder',
  highest_historical_promo_ladder: 'Deep Discount',
  lowest_historical_promo_ladder: 'Shallow Discount',
}

const FRIENDLY_SCENARIO_SHORT = {
  latest_month_discount: 'Latest Month',
  last_3m_exact: 'Last 3M Exact',
  same_season_last_year: 'LY Same Season',
  most_common_historical_ladder: 'Most Common Ladder',
  highest_historical_promo_ladder: 'Deep Discount',
  lowest_historical_promo_ladder: 'Shallow Discount',
}

const fmtPct = (value) => {
  const n = Number(value)
  if (!Number.isFinite(n)) return 'NA'
  const sign = n > 0 ? '+' : ''
  return `${sign}${n.toFixed(2)}%`
}

const fmtWhole = (value) => {
  const n = Number(value)
  if (!Number.isFinite(n)) return 'NA'
  return n.toLocaleString('en-IN', { maximumFractionDigits: 0 })
}

const formatPeriodLabel = (period) => {
  const raw = String(period || '')
  const match = raw.match(/^(\d{4})-(\d{2})$/)
  if (!match) return raw
  const y = Number(match[1])
  const m = Number(match[2]) - 1
  const dt = new Date(y, m, 1)
  return dt.toLocaleDateString('en-US', { month: 'short', year: 'numeric' })
}

const sortSlabEntries = (mapObj = {}) =>
  Object.entries(mapObj || {}).sort((a, b) => {
    const ai = Number(String(a[0]).replace(/\D/g, ''))
    const bi = Number(String(b[0]).replace(/\D/g, ''))
    if (Number.isFinite(ai) && Number.isFinite(bi)) return ai - bi
    return String(a[0]).localeCompare(String(b[0]))
  })

const clampDiscount = (value) => {
  const n = Number(value)
  if (!Number.isFinite(n)) return 5
  return Math.min(30, Math.max(5, Number(n.toFixed(2))))
}

const parseEditableDiscount = (raw, fallback = 5) => {
  const txt = String(raw ?? '').trim()
  if (!txt) return clampDiscount(fallback)
  const normalized = txt.replace(',', '.')
  const n = Number(normalized)
  if (!Number.isFinite(n)) return clampDiscount(fallback)
  return clampDiscount(n)
}

const fieldKey = (periodKey, sizeKey, slabKey) => `${String(periodKey)}|${String(sizeKey)}|${String(slabKey)}`

const enforceNonDecreasingLadder = (mapObj = {}) => {
  const entries = sortSlabEntries(mapObj)
  const out = {}
  let floor = 5
  entries.forEach(([slabKey, raw]) => {
    const current = clampDiscount(raw)
    const next = Math.max(floor, current)
    out[slabKey] = Number(next.toFixed(2))
    floor = out[slabKey]
  })
  return out
}

const normalizeScenarioDiscountsByPeriod = (scenario, periods = []) => {
  const periodMap = scenario?.scenario_discounts_by_period
  if (periodMap && Object.keys(periodMap).length > 0) return periodMap
  const bySize = scenario?.scenario_discounts_by_size
  if (!bySize || Object.keys(bySize).length === 0) return {}
  const periodKeys = Array.isArray(periods) && periods.length > 0 ? periods : ['M1', 'M2', 'M3']
  const out = {}
  periodKeys.forEach((period) => {
    out[String(period)] = {
      '12-ML': { ...(bySize?.['12-ML'] || {}) },
      '18-ML': { ...(bySize?.['18-ML'] || {}) },
    }
  })
  return out
}

const ScenarioBarTooltip = ({ active, payload, label }) => {
  if (!active || !payload || !payload.length) return null
  const scenarioName = payload?.[0]?.payload?.scenario_name || label
  const read = (key) => Number(payload.find((p) => p?.dataKey === key)?.value || 0)
  return (
    <div className="rounded-lg border border-slate-200 bg-white shadow-md px-3 py-2 text-xs min-w-[170px]">
      <div className="font-semibold text-body mb-1">{label}</div>
      <div className="text-[11px] text-muted mb-1">{scenarioName}</div>
      <div className="text-blue-800">Volume: {fmtPct(read('volume_pct'))}</div>
      <div className="text-green-700">Revenue: {fmtPct(read('revenue_pct'))}</div>
      <div className="text-orange-700">Profit: {fmtPct(read('profit_pct'))}</div>
    </div>
  )
}

const parseThreshold = (value) => {
  const n = Number(value)
  return Number.isFinite(n) ? n : -9999
}

const sortScenarioRowsByRevenue = (rows = []) =>
  [...rows].sort((a, b) => {
    const ar = Number(readSummary(a, 'TOTAL')?.revenue_pct || 0)
    const br = Number(readSummary(b, 'TOTAL')?.revenue_pct || 0)
    if (br !== ar) return br - ar
    const ap = Number(readSummary(a, 'TOTAL')?.profit_pct || 0)
    const bp = Number(readSummary(b, 'TOTAL')?.profit_pct || 0)
    if (bp !== ap) return bp - ap
    const av = Number(readSummary(a, 'TOTAL')?.volume_ml_pct || readSummary(a, 'TOTAL')?.volume_pct || 0)
    const bv = Number(readSummary(b, 'TOTAL')?.volume_ml_pct || readSummary(b, 'TOTAL')?.volume_pct || 0)
    return bv - av
  })

const resolveScenarioName = (row, fallbackIndex = 0) => {
  const custom = String(row?.custom_name || '').trim()
  if (custom) return custom

  const named = String(row?.scenario || '').trim()
  if (named && !/^s\d+$/i.test(named)) return named

  const byId = FRIENDLY_SCENARIO_NAMES[String(row?.scenario_id || '').trim()]
  if (byId) return byId

  return `Scenario ${fallbackIndex + 1}`
}

const resolveScenarioShortName = (row, fallbackIndex = 0) => {
  const custom = String(row?.custom_name || '').trim()
  if (custom) return custom.length > 24 ? `${custom.slice(0, 21)}...` : custom

  const byId = FRIENDLY_SCENARIO_SHORT[String(row?.scenario_id || '').trim()]
  if (byId) return byId

  const named = String(row?.scenario || '').trim()
  if (named && !/^s\d+$/i.test(named)) return named.length > 24 ? `${named.slice(0, 21)}...` : named

  return `Scenario ${fallbackIndex + 1}`
}

const readSummary = (row, sizeKey) => {
  const block = row?.summary?.[sizeKey] || {}
  return {
    volume: Number(block?.volume || 0),
    revenue: Number(block?.revenue || 0),
    profit: Number(block?.profit || 0),
    volume_pct: Number(block?.volume_pct || 0),
    revenue_pct: Number(block?.revenue_pct || 0),
    profit_pct: Number(block?.profit_pct || 0),
    volume_ml: Number(block?.volume_ml || 0),
    volume_ml_pct: Number(block?.volume_ml_pct || 0),
  }
}

const ScenarioComparison = ({
  data,
  plannerBase,
  isLoading,
  isError,
  errorMessage,
  createScenarioRequestId = 0,
  onScenariosChange = null,
}) => {
  const VISIBLE_SCENARIOS = 5
  const [minVolumePct, setMinVolumePct] = useState('')
  const [minRevenuePct, setMinRevenuePct] = useState('')
  const [minProfitPct, setMinProfitPct] = useState('')
  const [slidePage, setSlidePage] = useState(0)
  const [modalScenario, setModalScenario] = useState(null)
  const [modalDraft, setModalDraft] = useState(null)
  const [modalInputDraft, setModalInputDraft] = useState({})
  const [modalError, setModalError] = useState('')

  const [scenarios, setScenarios] = useState([])
  const [isCreateMode, setIsCreateMode] = useState(false)
  const lastCreateRequestRef = useRef(0)
  useEffect(() => {
    setScenarios(Array.isArray(data?.scenarios) ? data.scenarios : [])
  }, [data?.scenarios])

  const successfulRows = useMemo(() => scenarios.filter((row) => row?.success), [scenarios])
  const failedRows = useMemo(() => scenarios.filter((row) => !row?.success), [scenarios])

  const filteredRows = useMemo(() => {
    const minVol = parseThreshold(minVolumePct || -9999)
    const minRev = parseThreshold(minRevenuePct || -9999)
    const minProf = parseThreshold(minProfitPct || -9999)

    return successfulRows.filter((row) => {
      const total = readSummary(row, 'TOTAL')
      return (
        Number(total.volume_ml_pct || total.volume_pct) >= minVol &&
        Number(total.revenue_pct) >= minRev &&
        Number(total.profit_pct) >= minProf
      )
    })
  }, [successfulRows, minVolumePct, minRevenuePct, minProfitPct])

  const sortedFilteredRows = useMemo(
    () => sortScenarioRowsByRevenue(filteredRows),
    [filteredRows]
  )

  const chartData = useMemo(
    () =>
      sortedFilteredRows.map((row, idx) => {
        const total = readSummary(row, 'TOTAL')
        const scenarioKey = String(row?.key || row?.scenario || '')
        const display = resolveScenarioName(row, idx)
        const axisLabel = resolveScenarioShortName(row, idx)
        return {
          scenario_key: scenarioKey,
          axis_label: axisLabel,
          scenario_name: display,
          volume_pct: Number(total.volume_ml_pct || total.volume_pct || 0),
          revenue_pct: Number(total.revenue_pct || 0),
          profit_pct: Number(total.profit_pct || 0),
        }
      }),
    [sortedFilteredRows]
  )

  const revenueRankMap = useMemo(() => {
    const ranked = sortedFilteredRows
    const map = {}
    ranked.forEach((row, idx) => {
      map[String(row?.key || row?.scenario || '')] = idx + 1
    })
    return map
  }, [sortedFilteredRows])

  const handleDownloadFilteredCsv = () => {
    if (!sortedFilteredRows.length) return

    const esc = (v) => {
      const raw = v == null ? '' : String(v)
      const safe = raw.replace(/"/g, '""')
      return `"${safe}"`
    }

    const normalizedPeriods = Array.isArray(data?.periods) ? data.periods.map((p) => String(p)) : []
    const periodKeys = normalizedPeriods.length > 0
      ? normalizedPeriods
      : (() => {
        const found = new Set()
        sortedFilteredRows.forEach((row) => {
          const byPeriod = normalizeScenarioDiscountsByPeriod(row, normalizedPeriods)
          Object.keys(byPeriod || {}).forEach((k) => found.add(String(k)))
        })
        return Array.from(found).sort((a, b) => String(a).localeCompare(String(b)))
      })()

    const slabOrderBySize = {
      '12-ML': sortSlabEntries(plannerBase?.defaults_matrix?.['12-ML'] || {}).map(([k]) => k),
      '18-ML': sortSlabEntries(plannerBase?.defaults_matrix?.['18-ML'] || {}).map(([k]) => k),
    }

    if (!slabOrderBySize['12-ML'].length || !slabOrderBySize['18-ML'].length) {
      sortedFilteredRows.forEach((row) => {
        const byPeriod = normalizeScenarioDiscountsByPeriod(row, periodKeys)
        periodKeys.forEach((periodKey) => {
          const bySize = byPeriod?.[periodKey] || {}
          ;['12-ML', '18-ML'].forEach((sizeKey) => {
            sortSlabEntries(bySize?.[sizeKey] || {}).forEach(([slabKey]) => {
              if (!slabOrderBySize[sizeKey].includes(slabKey)) slabOrderBySize[sizeKey].push(slabKey)
            })
          })
        })
      })
      slabOrderBySize['12-ML'] = sortSlabEntries(
        Object.fromEntries((slabOrderBySize['12-ML'] || []).map((k) => [k, 1]))
      ).map(([k]) => k)
      slabOrderBySize['18-ML'] = sortSlabEntries(
        Object.fromEntries((slabOrderBySize['18-ML'] || []).map((k) => [k, 1]))
      ).map(([k]) => k)
    }

    const discountColumnDefs = []
    periodKeys.forEach((periodKey) => {
      ;['12-ML', '18-ML'].forEach((sizeKey) => {
        const slabs = slabOrderBySize[sizeKey] || []
        slabs.forEach((slabKey) => {
          discountColumnDefs.push({
            periodKey,
            sizeKey,
            slabKey,
            header: `${sizeKey} ${formatPeriodLabel(periodKey)} ${slabKey} Discount %`,
          })
        })
      })
    })

    const headers = [
      'Revenue Rank',
      'Scenario Name',
      'Scenario ID',
      '12-ML Volume',
      '12-ML Revenue',
      '12-ML Profit',
      '18-ML Volume',
      '18-ML Revenue',
      '18-ML Profit',
      'TOTAL Volume (ML)',
      'TOTAL Revenue',
      'TOTAL Profit',
      '12-ML Volume %',
      '12-ML Revenue %',
      '12-ML Profit %',
      '18-ML Volume %',
      '18-ML Revenue %',
      '18-ML Profit %',
      'TOTAL Volume %',
      'TOTAL Revenue %',
      'TOTAL Profit %',
      ...discountColumnDefs.map((d) => d.header),
    ]

    const lines = [headers.map(esc).join(',')]
    sortedFilteredRows.forEach((row, idx) => {
      const s12 = readSummary(row, '12-ML')
      const s18 = readSummary(row, '18-ML')
      const st = readSummary(row, 'TOTAL')
      const scenarioName = resolveScenarioName(row, idx)
      const scenarioId = String(row?.scenario_id || row?.key || '')
      const rank = idx + 1
      const byPeriod = normalizeScenarioDiscountsByPeriod(row, periodKeys)
      const discountValues = discountColumnDefs.map((def) => {
        const v = byPeriod?.[def.periodKey]?.[def.sizeKey]?.[def.slabKey]
        return Number.isFinite(Number(v)) ? Number(v).toFixed(2) : ''
      })
      const values = [
        rank,
        scenarioName,
        scenarioId,
        Number(s12.volume || 0).toFixed(2),
        Number(s12.revenue || 0).toFixed(2),
        Number(s12.profit || 0).toFixed(2),
        Number(s18.volume || 0).toFixed(2),
        Number(s18.revenue || 0).toFixed(2),
        Number(s18.profit || 0).toFixed(2),
        Number(st.volume_ml || st.volume || 0).toFixed(2),
        Number(st.revenue || 0).toFixed(2),
        Number(st.profit || 0).toFixed(2),
        Number(s12.volume_pct || 0).toFixed(2),
        Number(s12.revenue_pct || 0).toFixed(2),
        Number(s12.profit_pct || 0).toFixed(2),
        Number(s18.volume_pct || 0).toFixed(2),
        Number(s18.revenue_pct || 0).toFixed(2),
        Number(s18.profit_pct || 0).toFixed(2),
        Number(st.volume_ml_pct || st.volume_pct || 0).toFixed(2),
        Number(st.revenue_pct || 0).toFixed(2),
        Number(st.profit_pct || 0).toFixed(2),
        ...discountValues,
      ]
      lines.push(values.map(esc).join(','))
    })

    const csvText = lines.join('\n')
    const blob = new Blob([csvText], { type: 'text/csv;charset=utf-8;' })
    const url = URL.createObjectURL(blob)
    const ts = new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-')
    const a = document.createElement('a')
    a.href = url
    a.download = `step5_scenarios_filtered_${ts}.csv`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  const yDomain = useMemo(() => {
    if (!chartData.length) return [-5, 5]
    const values = chartData.flatMap((row) => [row.volume_pct, row.revenue_pct, row.profit_pct])
    const min = Math.min(...values)
    const max = Math.max(...values)
    const pad = Math.max(2, (max - min) * 0.15)
    return [Math.floor(min - pad), Math.ceil(max + pad)]
  }, [chartData])

  const maxPage = useMemo(
    () => Math.max(0, Math.ceil(chartData.length / VISIBLE_SCENARIOS) - 1),
    [chartData.length]
  )

  useEffect(() => {
    setSlidePage(0)
  }, [minVolumePct, minRevenuePct, minProfitPct])

  useEffect(() => {
    setSlidePage((prev) => Math.min(prev, maxPage))
  }, [maxPage])

  const visibleChartData = useMemo(
    () => {
      const start = slidePage * VISIBLE_SCENARIOS
      return chartData.slice(start, start + VISIBLE_SCENARIOS)
    },
    [chartData, slidePage]
  )

  const recomputeScenario = (scenarioLike) => {
    const baseForRecompute = (data?.planner_base?.success ? data.planner_base : null) || (plannerBase?.success ? plannerBase : null)
    if (!baseForRecompute?.success) {
      return { success: false, message: 'Planner base data missing. Run Step 4 once, then Save & Recalculate.' }
    }
    const periods = normalizePlannerPeriodsFromData(baseForRecompute)
    const scenarioByPeriod = scenarioLike?.scenario_discounts_by_period || {}
    const computed = computeCrossSizePlannerData({
      data: baseForRecompute,
      periods,
      scenarioDiscountsByPeriod: scenarioByPeriod,
    })
    if (!computed?.success) {
      return { success: false, message: computed?.message || 'Scenario recomputation failed.' }
    }
    const summary = computed?.summary_3m || {}
    const readMetric = (summaryBlock = {}) => ({
      volume: Number(summaryBlock?.final_qty ?? summaryBlock?.scenario_qty_additive ?? 0),
      revenue: Number(summaryBlock?.scenario_revenue ?? 0),
      profit: Number(summaryBlock?.scenario_profit ?? 0),
      volume_pct: Number(summaryBlock?.vs_reference_volume_pct ?? 0),
      revenue_pct: Number(summaryBlock?.vs_reference_revenue_pct ?? 0),
      profit_pct: Number(summaryBlock?.vs_reference_profit_pct ?? 0),
    })
    return {
      success: true,
      scenario: {
        ...scenarioLike,
        success: true,
        message: computed?.message || '',
        summary: {
          '12-ML': readMetric(summary?.['12-ML']),
          '18-ML': readMetric(summary?.['18-ML']),
          TOTAL: {
            ...readMetric(summary?.TOTAL),
            volume_ml: Number(summary?.TOTAL?.final_volume_ml ?? summary?.TOTAL?.scenario_volume_ml_additive ?? 0),
            volume_ml_pct: Number(summary?.TOTAL?.vs_reference_volume_ml_pct ?? 0),
          },
        },
      },
    }
  }

  const closeModal = () => {
    setModalScenario(null)
    setModalDraft(null)
    setModalInputDraft({})
    setModalError('')
    setIsCreateMode(false)
  }

  const modalPeriods = useMemo(() => {
    if (!modalScenario) return []
    const dataPeriods = Array.isArray(data?.periods) ? data.periods.map((p) => String(p)) : []
    if (dataPeriods.length) return dataPeriods
    return Object.keys(modalScenario?.scenario_discounts_by_period || {}).sort((a, b) => String(a).localeCompare(String(b)))
  }, [data?.periods, modalScenario])

  const openScenarioModalFromChart = (entry) => {
    const key = String(entry?.payload?.scenario_key || entry?.scenario_key || '')
    if (!key) return
    const found = filteredRows.find((row) => String(row?.key || row?.scenario || '') === key)
    if (!found) return
    const normalizedPeriods = Array.isArray(data?.periods) ? data.periods.map((p) => String(p)) : []
    const withNormalizedMap = {
      ...found,
      scenario_discounts_by_period: normalizeScenarioDiscountsByPeriod(found, normalizedPeriods),
    }
    setModalError('')
    setIsCreateMode(false)
    setModalScenario(withNormalizedMap)
    setModalDraft(JSON.parse(JSON.stringify(withNormalizedMap)))
  }

  const openCreateScenarioModal = () => {
    const baseForCreate = (data?.planner_base?.success ? data.planner_base : null) || (plannerBase?.success ? plannerBase : null)
    if (!baseForCreate?.success) {
      setModalError('Planner base data missing. Run Step 4 once before creating a scenario.')
      return
    }
    const periods = normalizePlannerPeriodsFromData(baseForCreate)
    const defaultsMatrix = baseForCreate?.defaults_matrix || {}
    const scenarioMap = {}
    periods.forEach((periodKey) => {
      scenarioMap[periodKey] = {}
      ;['12-ML', '18-ML'].forEach((sizeKey) => {
        const slabs = sortSlabEntries(defaultsMatrix?.[sizeKey] || {})
        const monthLadder = {}
        slabs.forEach(([slabKey, series]) => {
          const arr = Array.isArray(series) ? series : [series]
          const lastRaw = arr.length ? arr[arr.length - 1] : 0
          monthLadder[slabKey] = clampDiscount(lastRaw)
        })
        scenarioMap[periodKey][sizeKey] = enforceNonDecreasingLadder(monthLadder)
      })
    })
    const draft = {
      key: `custom-${Date.now()}`,
      scenario: 'Custom Scenario',
      scenario_id: 'custom_manual',
      custom_name: '',
      scenario_discounts_by_period: scenarioMap,
    }
    const recomputed = recomputeScenario(draft)
    if (!recomputed.success) {
      setModalError(recomputed.message || 'Scenario recomputation failed.')
      return
    }
    setIsCreateMode(true)
    setModalError('')
    setModalScenario(recomputed.scenario)
    setModalDraft(JSON.parse(JSON.stringify(recomputed.scenario)))
  }

  useEffect(() => {
    lastCreateRequestRef.current = Number(createScenarioRequestId || 0)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    const nextId = Number(createScenarioRequestId || 0)
    if (!Number.isFinite(nextId) || nextId <= 0) return
    if (nextId === lastCreateRequestRef.current) return
    lastCreateRequestRef.current = nextId
    openCreateScenarioModal()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [createScenarioRequestId])

  useEffect(() => {
    if (!modalDraft?.scenario_discounts_by_period) {
      setModalInputDraft({})
      return
    }
    const next = {}
    Object.entries(modalDraft.scenario_discounts_by_period || {}).forEach(([periodKey, bySize]) => {
      Object.entries(bySize || {}).forEach(([sizeKey, slabMap]) => {
        Object.entries(slabMap || {}).forEach(([slabKey, rawValue]) => {
          next[fieldKey(periodKey, sizeKey, slabKey)] = String(rawValue ?? '')
        })
      })
    })
    setModalInputDraft(next)
  }, [modalDraft])

  const updateDraftDiscount = (periodKey, sizeKey, slabKey, value) => {
    setModalInputDraft((prev) => ({
      ...prev,
      [fieldKey(periodKey, sizeKey, slabKey)]: String(value ?? ''),
    }))
  }

  const commitDraftDiscount = (periodKey, sizeKey, slabKey, rawValue) => {
    const nextValue = parseEditableDiscount(rawValue)
    setModalDraft((prev) => {
      if (!prev) return prev
      const next = JSON.parse(JSON.stringify(prev))
      if (!next.scenario_discounts_by_period) next.scenario_discounts_by_period = {}
      if (!next.scenario_discounts_by_period[periodKey]) next.scenario_discounts_by_period[periodKey] = {}
      if (!next.scenario_discounts_by_period[periodKey][sizeKey]) next.scenario_discounts_by_period[periodKey][sizeKey] = {}
      next.scenario_discounts_by_period[periodKey][sizeKey][slabKey] = nextValue
      next.scenario_discounts_by_period[periodKey][sizeKey] = enforceNonDecreasingLadder(
        next.scenario_discounts_by_period[periodKey][sizeKey]
      )
      const enforcedMap = next.scenario_discounts_by_period[periodKey][sizeKey] || {}
      setModalInputDraft((prevInput) => {
        const updated = { ...prevInput }
        Object.entries(enforcedMap).forEach(([k, v]) => {
          updated[fieldKey(periodKey, sizeKey, k)] = String(v ?? '')
        })
        return updated
      })
      return next
    })
  }

  const updateDraftName = (value) => {
    setModalDraft((prev) => {
      if (!prev) return prev
      return { ...prev, custom_name: value }
    })
  }

  const saveModalScenario = () => {
    if (!modalScenario || !modalDraft) return
    const draftForSave = JSON.parse(JSON.stringify(modalDraft))
    Object.entries(modalInputDraft || {}).forEach(([key, raw]) => {
      const [periodKey, sizeKey, slabKey] = key.split('|')
      if (!periodKey || !sizeKey || !slabKey) return
      if (!draftForSave.scenario_discounts_by_period?.[periodKey]?.[sizeKey]) return
      draftForSave.scenario_discounts_by_period[periodKey][sizeKey][slabKey] = parseEditableDiscount(
        raw,
        draftForSave.scenario_discounts_by_period[periodKey][sizeKey][slabKey]
      )
    })
    Object.entries(draftForSave.scenario_discounts_by_period || {}).forEach(([periodKey, bySize]) => {
      Object.entries(bySize || {}).forEach(([sizeKey, slabMap]) => {
        draftForSave.scenario_discounts_by_period[periodKey][sizeKey] = enforceNonDecreasingLadder(slabMap || {})
      })
    })

    const recomputed = recomputeScenario(draftForSave)
    if (!recomputed.success) {
      setModalError(recomputed.message || 'Scenario recomputation failed.')
      return
    }
    const updated = recomputed.scenario
    setScenarios((prev) => {
      const existingIdx = prev.findIndex(
        (row) => String(row?.key || row?.scenario || '') === String(updated?.key || updated?.scenario || '')
      )
      const nextRows = existingIdx < 0
        ? [...prev, updated]
        : (() => {
          const next = [...prev]
          next[existingIdx] = updated
          return next
        })()
      if (typeof onScenariosChange === 'function') {
        onScenariosChange(nextRows)
      }
      return nextRows
    })
    setModalScenario(updated)
    setModalDraft(JSON.parse(JSON.stringify(updated)))
    setModalError('')
    setIsCreateMode(false)
  }

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-xl font-bold text-body">Step 5: Scenario Comparison</h3>
        <p className="text-sm text-muted mt-1">Fixed scenario set (historical ladders) computed using Step 4 planner logic.</p>
      </div>

      {isError && (
        <div className="bg-brand-dangerLight border border-danger rounded-lg p-4 flex items-start space-x-3">
          <AlertCircle className="text-danger flex-shrink-0 mt-0.5" size={20} />
          <div>
            <h4 className="font-semibold text-body">Step 5 Error</h4>
            <p className="text-muted text-sm">{errorMessage || 'Failed to generate scenarios'}</p>
          </div>
        </div>
      )}

      {isLoading && (
        <div className="bg-white rounded-lg shadow-md p-8 flex items-center gap-3 text-muted">
          <Loader2 className="w-5 h-5 animate-spin" />
          <span className="text-sm">Generating scenarios from Step 4 planner...</span>
        </div>
      )}

      {!isLoading && data?.success && (
        <>
          {failedRows.length > 0 && (
            <div className="bg-white rounded-lg shadow-md p-4">
              <h4 className="text-base font-semibold text-body mb-3">Unsuccessful Scenario(s)</h4>
              <div className="space-y-2">
                {failedRows.map((row) => (
                  <div key={String(row?.key || row?.scenario || Math.random())} className="rounded-lg border border-red-200 bg-red-50 p-3">
                    <div className="text-sm font-semibold text-body">{row?.scenario || 'Scenario'}</div>
                    <div className="text-xs text-danger mt-1">{row?.message || 'Unknown error while computing scenario.'}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="bg-white rounded-lg shadow-md p-4">
            <h4 className="text-base font-semibold text-body mb-3">Scenario Filters (TOTAL %)</h4>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              <div>
                <label className="block text-xs font-medium text-muted mb-1">Min Volume % Increase</label>
                <input
                  type="number"
                  step="0.1"
                  value={minVolumePct}
                  onChange={(e) => setMinVolumePct(e.target.value)}
                  className="w-full px-3 py-2 text-sm border border-gray-300 rounded-lg"
                  placeholder="e.g. 0"
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-muted mb-1">Min Revenue % Increase</label>
                <input
                  type="number"
                  step="0.1"
                  value={minRevenuePct}
                  onChange={(e) => setMinRevenuePct(e.target.value)}
                  className="w-full px-3 py-2 text-sm border border-gray-300 rounded-lg"
                  placeholder="e.g. 0"
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-muted mb-1">Min Profit % Increase</label>
                <input
                  type="number"
                  step="0.1"
                  value={minProfitPct}
                  onChange={(e) => setMinProfitPct(e.target.value)}
                  className="w-full px-3 py-2 text-sm border border-gray-300 rounded-lg"
                  placeholder="e.g. 0"
                />
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-md p-4">
            <div className="flex items-center justify-between mb-3">
              <h4 className="text-base font-semibold text-body">TOTAL % Comparison (Volume / Revenue / Profit)</h4>
              <button
                type="button"
                onClick={handleDownloadFilteredCsv}
                disabled={sortedFilteredRows.length === 0}
                className="inline-flex items-center gap-2 px-3 py-2 rounded-md border border-slate-300 bg-white text-sm font-semibold text-body disabled:opacity-40"
              >
                <Download size={14} />
                Download CSV
              </button>
            </div>
            <div className="text-xs text-muted mb-3">{data?.message || `${successfulRows.length}/${scenarios.length} scenarios computed`}</div>
            {chartData.length > 0 ? (
              <div className="rounded-lg border border-slate-200 p-2">
                {chartData.length > VISIBLE_SCENARIOS && (
                  <div className="px-1 pb-2">
                    <div className="flex items-center justify-between text-xs text-muted mb-1">
                      <span>
                        Showing {(slidePage * VISIBLE_SCENARIOS) + 1}-{Math.min(((slidePage + 1) * VISIBLE_SCENARIOS), chartData.length)} of {chartData.length}
                      </span>
                      <div className="flex items-center gap-2">
                        <button
                          type="button"
                          onClick={() => setSlidePage((p) => Math.max(0, p - 1))}
                          disabled={slidePage <= 0}
                          className="px-2.5 py-1 rounded border border-slate-300 text-xs font-medium disabled:opacity-40"
                        >
                          Previous
                        </button>
                        <span>Page {slidePage + 1} / {maxPage + 1}</span>
                        <button
                          type="button"
                          onClick={() => setSlidePage((p) => Math.min(maxPage, p + 1))}
                          disabled={slidePage >= maxPage}
                          className="px-2.5 py-1 rounded border border-slate-300 text-xs font-medium disabled:opacity-40"
                        >
                          Next
                        </button>
                      </div>
                    </div>
                    <div className="flex items-center gap-1.5 mt-1">
                      {Array.from({ length: maxPage + 1 }).map((_, idx) => (
                        <button
                          key={`page-dot-${idx}`}
                          type="button"
                          onClick={() => setSlidePage(idx)}
                          className={`h-2.5 rounded-full transition-all ${
                            idx === slidePage ? 'w-6 bg-primary' : 'w-2.5 bg-slate-300 hover:bg-slate-400'
                          }`}
                          aria-label={`Go to page ${idx + 1}`}
                        />
                      ))}
                    </div>
                  </div>
                )}
                <div style={{ width: '100%', height: 360 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={visibleChartData} margin={{ top: 30, right: 16, left: 8, bottom: 8 }} barCategoryGap="30%" barGap={8}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                      <XAxis dataKey="axis_label" tick={{ fontSize: 12, fill: '#334155', fontWeight: 600 }} />
                      <YAxis domain={yDomain} tickFormatter={(value) => `${Number(value).toFixed(0)}%`} />
                      <Tooltip content={<ScenarioBarTooltip />} />
                      <ReferenceLine y={0} stroke="#94a3b8" strokeWidth={1.2} />
                      <Legend />
                      <Bar dataKey="volume_pct" name="Volume %" fill="#2563eb" radius={[6, 6, 0, 0]} cursor="pointer" onClick={openScenarioModalFromChart}>
                        <LabelList
                          dataKey="volume_pct"
                          position="top"
                          formatter={(v) => fmtPct(v)}
                          fontSize={12}
                          fontWeight={800}
                          fill="#0f172a"
                        />
                      </Bar>
                      <Bar dataKey="revenue_pct" name="Revenue %" fill="#16a34a" radius={[6, 6, 0, 0]} cursor="pointer" onClick={openScenarioModalFromChart}>
                        <LabelList
                          dataKey="revenue_pct"
                          position="top"
                          formatter={(v) => fmtPct(v)}
                          fontSize={12}
                          fontWeight={800}
                          fill="#0f172a"
                        />
                      </Bar>
                      <Bar dataKey="profit_pct" name="Profit %" fill="#f59e0b" radius={[6, 6, 0, 0]} cursor="pointer" onClick={openScenarioModalFromChart}>
                        <LabelList
                          dataKey="profit_pct"
                          position="top"
                          formatter={(v) => fmtPct(v)}
                          fontSize={12}
                          fontWeight={800}
                          fill="#0f172a"
                        />
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            ) : (
              <p className="text-sm text-muted">No scenario passed the filter thresholds.</p>
            )}
            <p className="text-xs text-muted mt-2">Click any bar to open scenario month-wise slab discounts.</p>
          </div>
        </>
      )}

      {!isLoading && !data?.success && !isError && (
        <div className="bg-white rounded-lg shadow-md p-12 text-center">
          <p className="text-sm text-muted">Click "Generate Scenarios" in Step 5 settings.</p>
        </div>
      )}

      {modalScenario && (
        <div className="fixed inset-0 z-[90] flex items-center justify-center p-4 md:p-6">
          <div className="absolute inset-0 bg-black/45" onClick={closeModal} />
          <div className="relative z-[91] w-full max-w-6xl max-h-[90vh] bg-white rounded-2xl shadow-2xl overflow-hidden flex flex-col">
            <div className="flex items-start justify-between px-5 py-4 border-b border-slate-200 bg-slate-50">
              <div>
                <h4 className="text-lg font-semibold text-body">
                  {String(modalDraft?.custom_name || '').trim() || modalScenario?.scenario || 'Scenario'}
                </h4>
                <p className="text-xs text-muted">Month-wise slab discount inputs for 12-ML and 18-ML</p>
                <div className="mt-2 inline-flex items-center gap-2 rounded-md border border-slate-200 bg-white px-2.5 py-1">
                  <span className="text-[11px] uppercase tracking-wide text-muted">Revenue Rank</span>
                  <span className="text-sm font-bold text-body">
                    #{revenueRankMap[String(modalScenario?.key || modalScenario?.scenario || '')] || '-'}
                  </span>
                  <span className="text-xs text-muted">/ {filteredRows.length || 0}</span>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={saveModalScenario}
                  className="inline-flex items-center gap-1 px-3 py-2 rounded-lg border border-primary text-primary hover:bg-primary hover:text-white text-sm"
                >
                  <Save size={14} />
                  {isCreateMode ? 'Create & Recalculate' : 'Save & Recalculate'}
                </button>
                <button
                  type="button"
                  onClick={closeModal}
                  className="p-2 rounded-lg hover:bg-slate-100 text-slate-600"
                >
                  <X size={18} />
                </button>
              </div>
            </div>

            <div className="px-5 py-3 border-b border-slate-200 bg-slate-50">
              <div className="mb-3">
                <label className="block text-[11px] uppercase text-muted mb-1">Scenario Name</label>
                <input
                  type="text"
                  value={String(modalDraft?.custom_name ?? '')}
                  onChange={(e) => updateDraftName(e.target.value)}
                  className="w-full md:w-[420px] px-3 py-2 text-sm border border-gray-300 rounded-lg"
                  placeholder="Enter custom scenario name"
                />
              </div>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                <div className="rounded-lg border border-slate-200 bg-white p-3">
                  <div className="text-[11px] uppercase text-muted">12-ML Volume %</div>
                  <div className="text-base font-semibold text-body">{fmtPct(readSummary(modalDraft || modalScenario, '12-ML').volume_pct)}</div>
                </div>
                <div className="rounded-lg border border-slate-200 bg-white p-3">
                  <div className="text-[11px] uppercase text-muted">18-ML Volume %</div>
                  <div className="text-base font-semibold text-body">{fmtPct(readSummary(modalDraft || modalScenario, '18-ML').volume_pct)}</div>
                </div>
                <div className="rounded-lg border border-slate-200 bg-white p-3">
                  <div className="text-[11px] uppercase text-muted">TOTAL Volume %</div>
                  <div className="text-base font-semibold text-body">{fmtPct(readSummary(modalDraft || modalScenario, 'TOTAL').volume_ml_pct || readSummary(modalDraft || modalScenario, 'TOTAL').volume_pct)}</div>
                </div>
              </div>
              {modalError && <div className="text-xs text-danger mt-2">{modalError}</div>}
            </div>

            <div className="p-5 overflow-auto">
              {modalPeriods.length === 0 && (
                <div className="rounded-lg border border-slate-200 bg-slate-50 p-4 text-sm text-muted">
                  Month-wise slab discount data not available for this scenario.
                </div>
              )}

              {modalPeriods.length > 0 && (
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                  {modalPeriods.map((periodKey) => {
                    const periodMap = modalDraft?.scenario_discounts_by_period?.[periodKey] || {}
                    const map12 = periodMap?.['12-ML'] || {}
                    const map18 = periodMap?.['18-ML'] || {}
                    const slabs12 = sortSlabEntries(map12)
                    const slabs18 = sortSlabEntries(map18)
                    return (
                      <div key={periodKey} className="rounded-xl border border-slate-200 p-4">
                        <h5 className="text-sm font-semibold text-body mb-3">{formatPeriodLabel(periodKey)}</h5>
                        <div className="space-y-3">
                          <div className="rounded-lg border border-slate-200 p-3">
                            <div className="text-[11px] uppercase tracking-wide text-muted mb-2">12-ML</div>
                            <div className="space-y-1.5">
                              {slabs12.length > 0 ? slabs12.map(([slab, value]) => (
                                <div key={`12-${periodKey}-${slab}`} className="flex items-center justify-between">
                                  <span className="text-xs font-medium text-body">{slab}</span>
                                  <input
                                    type="text"
                                    inputMode="decimal"
                                    value={modalInputDraft[fieldKey(periodKey, '12-ML', slab)] ?? String(value ?? '')}
                                    onChange={(e) => updateDraftDiscount(periodKey, '12-ML', slab, e.target.value)}
                                    onBlur={(e) => commitDraftDiscount(periodKey, '12-ML', slab, e.target.value)}
                                    onFocus={(e) => e.target.select()}
                                    className="w-24 px-2 py-1 text-xs border border-gray-300 rounded text-right [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
                                  />
                                </div>
                              )) : <div className="text-xs text-muted">No slab discounts</div>}
                            </div>
                          </div>
                          <div className="rounded-lg border border-slate-200 p-3">
                            <div className="text-[11px] uppercase tracking-wide text-muted mb-2">18-ML</div>
                            <div className="space-y-1.5">
                              {slabs18.length > 0 ? slabs18.map(([slab, value]) => (
                                <div key={`18-${periodKey}-${slab}`} className="flex items-center justify-between">
                                  <span className="text-xs font-medium text-body">{slab}</span>
                                  <input
                                    type="text"
                                    inputMode="decimal"
                                    value={modalInputDraft[fieldKey(periodKey, '18-ML', slab)] ?? String(value ?? '')}
                                    onChange={(e) => updateDraftDiscount(periodKey, '18-ML', slab, e.target.value)}
                                    onBlur={(e) => commitDraftDiscount(periodKey, '18-ML', slab, e.target.value)}
                                    onFocus={(e) => e.target.select()}
                                    className="w-24 px-2 py-1 text-xs border border-gray-300 rounded text-right [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
                                  />
                                </div>
                              )) : <div className="text-xs text-muted">No slab discounts</div>}
                            </div>
                          </div>
                        </div>
                      </div>
                    )
                  })}
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default ScenarioComparison
