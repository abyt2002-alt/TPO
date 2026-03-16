import { useEffect, useMemo, useRef, useState } from 'react'
import { AlertCircle, Loader2, RotateCcw } from 'lucide-react'

const normalizeSizeKey = (value) => String(value || '').toUpperCase().replace(/\s+/g, '').trim()
const REFERENCE_MODE_LABELS = {
  ly_same_3m: 'Y-o-Y',
  last_3m_before_projection: 'Q-o-Q',
}
const slabSortKey = (value) => {
  const text = String(value || '').toLowerCase()
  const match = text.match(/(\d+)/)
  return match ? Number(match[1]) : Number.MAX_SAFE_INTEGER
}

const formatSignedPct = (value) => {
  const n = Number(value)
  if (!Number.isFinite(n)) return 'NA'
  if (Math.abs(n) < 1e-9) return '+0.00%'
  return `${n >= 0 ? '+' : ''}${n.toFixed(2)}%`
}

const pctToneClass = (value) => {
  const n = Number(value)
  if (!Number.isFinite(n) || Math.abs(n) < 1e-9) return 'text-muted'
  return n > 0 ? 'text-success' : 'text-danger'
}

const formatWhole = (value) => {
  const n = Number(value)
  if (!Number.isFinite(n)) return 'NA'
  return n.toLocaleString(undefined, { maximumFractionDigits: 0 })
}

const formatCompact = (value) => {
  const n = Number(value)
  if (!Number.isFinite(n)) return 'NA'
  return new Intl.NumberFormat('en-US', {
    notation: 'compact',
    compactDisplay: 'short',
    maximumFractionDigits: 2,
  }).format(n)
}

const formatFixed = (value, digits = 4) => {
  const n = Number(value)
  if (!Number.isFinite(n)) return 'NA'
  return n.toFixed(digits)
}

const formatDiscountInput = (value) => {
  const n = Number(value)
  if (!Number.isFinite(n)) return ''
  return n % 1 === 0 ? String(n.toFixed(0)) : String(n.toFixed(1))
}

const normalizeDiscountValue = (raw, fallback) => {
  const text = String(raw ?? '').trim()
  if (!text) return Number(fallback || 0)
  const parsed = Number(text)
  if (!Number.isFinite(parsed)) return Number(fallback || 0)
  return Math.round(parsed * 2) / 2
}

const clampDiscountRange = (value, min = 5, max = 30) => {
  const n = Number(value)
  if (!Number.isFinite(n)) return min
  return Math.min(max, Math.max(min, n))
}

const enforceMonotonicSlabs = (slabMap, changedSlabKey, changedValue) => {
  const keys = Object.keys(slabMap || {}).sort((a, b) => slabSortKey(a) - slabSortKey(b))
  if (!keys.length) return {}
  const out = {}
  keys.forEach((k) => {
    out[k] = clampDiscountRange(slabMap?.[k])
  })
  const target = String(changedSlabKey)
  if (Object.prototype.hasOwnProperty.call(out, target)) {
    out[target] = clampDiscountRange(changedValue)
  }
  const idx = keys.indexOf(target)
  if (idx >= 0) {
    for (let i = idx + 1; i < keys.length; i += 1) {
      const prevKey = keys[i - 1]
      const currKey = keys[i]
      if (Number(out[currKey]) < Number(out[prevKey])) {
        out[currKey] = Number(out[prevKey])
      }
    }
    for (let i = idx - 1; i >= 0; i -= 1) {
      const nextKey = keys[i + 1]
      const currKey = keys[i]
      if (Number(out[currKey]) > Number(out[nextKey])) {
        out[currKey] = Number(out[nextKey])
      }
    }
  }
  return out
}

const formatPeriodLabel = (periodKey) => {
  const raw = String(periodKey || '').trim()
  if (!raw) return ''
  const [y, m] = raw.split('-')
  const yy = Number(y)
  const mm = Number(m)
  if (Number.isFinite(yy) && Number.isFinite(mm) && mm >= 1 && mm <= 12) {
    const d = new Date(Date.UTC(yy, mm - 1, 1))
    return d.toLocaleString('en-US', { month: 'short', timeZone: 'UTC' })
  }
  return raw
}

const normalizePeriodsFromData = (data) => {
  const explicit = Array.isArray(data?.periods) ? data.periods.map((p) => String(p)) : []
  if (explicit.length) return explicit
  const rawMonthly = data?.monthly_results
  if (Array.isArray(rawMonthly)) {
    const derived = rawMonthly
      .map((row) => String(row?.period || '').trim())
      .filter(Boolean)
    if (derived.length) return derived
  }
  if (rawMonthly && typeof rawMonthly === 'object') {
    const keys = Object.keys(rawMonthly).map((k) => String(k))
    if (keys.length) return keys
  }
  return []
}

const getSeriesValueAt = (seriesValue, monthIdx, periodKey) => {
  if (Array.isArray(seriesValue)) return Number(seriesValue?.[monthIdx] || 0)
  if (seriesValue && typeof seriesValue === 'object') return Number(seriesValue?.[periodKey] || 0)
  return Number(seriesValue || 0)
}

const buildScenarioStateFromResponse = (data, periods) => {
  const defaults = data?.defaults_matrix || {}
  const scenario = data?.scenario_matrix || defaults
  const next = {}
  const draft = {}

  periods.forEach((periodKey, idx) => {
    next[periodKey] = {}
    draft[periodKey] = {}
    Object.entries(scenario).forEach(([sizeKeyRaw, slabSeriesMap]) => {
      const sizeKey = normalizeSizeKey(sizeKeyRaw)
      next[periodKey][sizeKey] = {}
      draft[periodKey][sizeKey] = {}
      Object.entries(slabSeriesMap || {}).forEach(([slabKey, arr]) => {
        const v = getSeriesValueAt(arr, idx, periodKey)
        next[periodKey][sizeKey][String(slabKey)] = Number.isFinite(v) ? v : 0
        draft[periodKey][sizeKey][String(slabKey)] = formatDiscountInput(v)
      })
    })
  })

  return { scenarioByPeriod: next, draftByPeriod: draft }
}

const buildWeightMapForSize = (sizeResult, slabKeys) => {
  const keys = (Array.isArray(slabKeys) ? slabKeys : []).map((s) => String(s))
  if (!keys.length) return {}
  const raw = sizeResult?.modeled_weights && typeof sizeResult.modeled_weights === 'object'
    ? sizeResult.modeled_weights
    : {}
  const out = {}
  let sum = 0
  keys.forEach((k) => {
    const w = Number(raw?.[k] || 0)
    const safe = Number.isFinite(w) && w > 0 ? w : 0
    out[k] = safe
    sum += safe
  })
  if (sum <= 0) {
    const eq = 1 / keys.length
    keys.forEach((k) => { out[k] = eq })
    return out
  }
  keys.forEach((k) => { out[k] = out[k] / sum })
  return out
}

const computeOtherWeightedDiscount = (slabKey, discountMap, weightMap) => {
  const target = String(slabKey)
  let num = 0
  let den = 0
  Object.entries(weightMap || {}).forEach(([k, wRaw]) => {
    const kStr = String(k)
    if (kStr === target) return
    const w = Number(wRaw || 0)
    const d = Number(discountMap?.[kStr] || 0)
    if (!Number.isFinite(w) || w <= 0) return
    if (!Number.isFinite(d)) return
    num += w * d
    den += w
  })
  if (den <= 0) return 0
  return num / den
}

const CrossSizePlanner = ({
  data,
  isLoading,
  isError,
  errorMessage,
  displayReferenceMode = 'ly_same_3m',
  onDisplayReferenceModeChange,
  referenceByMode = {},
  onInitialize,
}) => {
  const [scenarioDiscountsByPeriod, setScenarioDiscountsByPeriod] = useState({})
  const [draftDiscountsByPeriod, setDraftDiscountsByPeriod] = useState({})
  const initRequestSentRef = useRef(false)

  const periods = useMemo(() => normalizePeriodsFromData(data), [data])
  const monthlyByPeriod = useMemo(() => {
    const map = {}
    const rawMonthly = data?.monthly_results
    if (Array.isArray(rawMonthly)) {
      rawMonthly.forEach((row) => {
        const key = String(row?.period || '').trim()
        if (key) map[key] = row
      })
    } else if (rawMonthly && typeof rawMonthly === 'object') {
      Object.entries(rawMonthly).forEach(([key, value]) => {
        const periodKey = String(value?.period || key || '').trim()
        if (periodKey) {
          map[periodKey] = { period: periodKey, ...(value || {}) }
        }
      })
    }
    return map
  }, [data?.monthly_results])

  const selectedReferenceMode = REFERENCE_MODE_LABELS?.[displayReferenceMode]
    ? String(displayReferenceMode)
    : 'ly_same_3m'
  const referenceLabel = REFERENCE_MODE_LABELS[selectedReferenceMode] || REFERENCE_MODE_LABELS.ly_same_3m
  const coeffBySizeSlab = useMemo(() => {
    const out = {}
    const rows = Array.isArray(data?.size_results) ? data.size_results : []
    rows.forEach((sizeRow) => {
      const sizeKey = normalizeSizeKey(sizeRow?.size)
      if (!sizeKey) return
      if (!out[sizeKey]) out[sizeKey] = {}
      const slabs = Array.isArray(sizeRow?.slabs) ? sizeRow.slabs : []
      slabs.forEach((slabRow) => {
        const slabKey = String(slabRow?.slab || '').trim()
        if (!slabKey) return
        out[sizeKey][slabKey] = {
          coef_base_discount_pct: Number(slabRow?.coef_base_discount_pct || 0),
          coef_lag1_base_discount_pct: Number(slabRow?.coef_lag1_base_discount_pct || 0),
          coef_other_slabs_weighted_base_discount_pct: Number(slabRow?.coef_other_slabs_weighted_base_discount_pct || 0),
          stage2_intercept: Number(slabRow?.stage2_intercept || 0),
          coef_residual_store: Number(slabRow?.coef_residual_store || 0),
          residual_store: Number(slabRow?.residual_store || 0),
        }
      })
    })
    return out
  }, [data?.size_results])

  useEffect(() => {
    if (!data?.success) return
    const { scenarioByPeriod, draftByPeriod } = buildScenarioStateFromResponse(data, periods)
    setScenarioDiscountsByPeriod(scenarioByPeriod)
    setDraftDiscountsByPeriod(draftByPeriod)
  }, [data, periods])

  useEffect(() => {
    const hasNewPayload = periods.length > 0 && Object.keys(monthlyByPeriod).length > 0
    if (hasNewPayload) {
      initRequestSentRef.current = false
      return
    }
    if (!data?.success) return
    if (isLoading || isError) return
    if (typeof onInitialize !== 'function') return
    if (initRequestSentRef.current) return
    initRequestSentRef.current = true
    onInitialize()
  }, [data?.success, periods.length, monthlyByPeriod, isLoading, isError, onInitialize])

  const handleScenarioDiscountChange = (periodKey, sizeKey, slabKey, rawValue) => {
    const currentMap = scenarioDiscountsByPeriod?.[periodKey]?.[sizeKey] || {}
    const fallback = currentMap?.[String(slabKey)] ?? 5
    const normalized = clampDiscountRange(normalizeDiscountValue(rawValue, fallback))
    const nextForSize = enforceMonotonicSlabs(currentMap, slabKey, normalized)

    setScenarioDiscountsByPeriod((prev) => ({
      ...prev,
      [periodKey]: {
        ...(prev?.[periodKey] || {}),
        [sizeKey]: {
          ...(prev?.[periodKey]?.[sizeKey] || {}),
          ...nextForSize,
        },
      },
    }))

    setDraftDiscountsByPeriod((prev) => {
      const nextDraftSize = {
        ...(prev?.[periodKey]?.[sizeKey] || {}),
      }
      Object.entries(nextForSize).forEach(([k, v]) => {
        nextDraftSize[String(k)] = formatDiscountInput(v)
      })
      return {
        ...prev,
        [periodKey]: {
          ...(prev?.[periodKey] || {}),
          [sizeKey]: nextDraftSize,
        },
      }
    })
  }

  const setDraftRawValue = (periodKey, sizeKey, slabKey, rawValue) => {
    setDraftDiscountsByPeriod((prev) => ({
      ...prev,
      [periodKey]: {
        ...(prev?.[periodKey] || {}),
        [sizeKey]: {
          ...(prev?.[periodKey]?.[sizeKey] || {}),
          [String(slabKey)]: String(rawValue ?? ''),
        },
      },
    }))
  }

  const handleReset = () => {
    const { scenarioByPeriod, draftByPeriod } = buildScenarioStateFromResponse({
      ...data,
      scenario_matrix: data?.defaults_matrix || {},
    }, periods)
    setScenarioDiscountsByPeriod(scenarioByPeriod)
    setDraftDiscountsByPeriod(draftByPeriod)
  }

  const computedPlannerData = useMemo(() => {
    if (!data?.success || !periods.length) return data

    const defaultsMatrix = data?.defaults_matrix || {}
    const baselineSlabMatrix = data?.baseline_slab_matrix || {}
    const baseSummary = data?.summary_3m || {}
    const sizeResults = Array.isArray(data?.size_results) ? data.size_results : []
    const sizeResultByKey = {}
    sizeResults.forEach((row) => {
      const k = normalizeSizeKey(row?.size)
      if (k) sizeResultByKey[k] = row
    })

    const ref12Qty = Number(baseSummary?.['12-ML']?.reference_qty || 0)
    const ref18Qty = Number(baseSummary?.['18-ML']?.reference_qty || 0)
    const ref12Rev = Number(baseSummary?.['12-ML']?.reference_revenue || 0)
    const ref18Rev = Number(baseSummary?.['18-ML']?.reference_revenue || 0)
    const ref12Profit = Number(baseSummary?.['12-ML']?.reference_profit || 0)
    const ref18Profit = Number(baseSummary?.['18-ML']?.reference_profit || 0)
    const e12From18 = Number(data?.cross_elasticity_12_from_18 || 0)
    const e18From12 = Number(data?.cross_elasticity_18_from_12 || 0)

    const monthlyResults = periods.map((periodKey, monthIdx) => {
      const sizes = {}
      ;['12-ML', '18-ML'].forEach((sizeKey) => {
        const sizeDefaults = defaultsMatrix?.[sizeKey] || {}
        const slabKeys = Object.keys(sizeDefaults).sort((a, b) => slabSortKey(a) - slabSortKey(b))
        if (!slabKeys.length) return
        const sizeResult = sizeResultByKey?.[normalizeSizeKey(sizeKey)] || {}
        const slabRowsModel = Array.isArray(sizeResult?.slabs) ? sizeResult.slabs : []
        const slabModelByKey = {}
        slabRowsModel.forEach((r) => {
          const k = String(r?.slab || '')
          if (k) slabModelByKey[k] = r
        })
        const weightMap = buildWeightMapForSize(sizeResult, slabKeys)
        const scenarioMapThisMonth = {}
        slabKeys.forEach((slabKey) => {
          const fallback = Number(sizeDefaults?.[slabKey]?.[monthIdx] || 0)
          scenarioMapThisMonth[slabKey] = Number(
            scenarioDiscountsByPeriod?.[periodKey]?.[sizeKey]?.[slabKey] ?? fallback
          )
        })

        const slabs = slabKeys.map((slabKey) => {
          const model = slabModelByKey?.[slabKey] || {}
          const defaultDiscount = Number(sizeDefaults?.[slabKey]?.[monthIdx] || model?.default_discount_pct || 0)
          const scenarioDiscount = Number(scenarioMapThisMonth?.[slabKey] ?? defaultDiscount)
          const lagUsed = monthIdx > 0
            ? Number(
              scenarioDiscountsByPeriod?.[periods?.[monthIdx - 1]]?.[sizeKey]?.[slabKey]
              ?? sizeDefaults?.[slabKey]?.[monthIdx - 1]
              ?? model?.default_discount_pct
              ?? defaultDiscount
            )
            : Number(model?.default_discount_pct ?? defaultDiscount)
          const otherWeighted = computeOtherWeightedDiscount(slabKey, scenarioMapThisMonth, weightMap)
          const coefBase = Number(model?.coef_base_discount_pct || 0)
          const coefLag = Number(model?.coef_lag1_base_discount_pct || 0)
          const coefOther = Number(model?.coef_other_slabs_weighted_base_discount_pct || 0)
          const basePrice = Number(model?.base_price || 0)
          const cogsPerUnit = Number(model?.cogs_per_unit || 0)
          const nonDiscountBaseline = Number(baselineSlabMatrix?.[sizeKey]?.[slabKey]?.[monthIdx] || 0)
          const discountComponentScenario = (coefBase * scenarioDiscount) + (coefLag * lagUsed) + (coefOther * otherWeighted)
          const preCrossQty = Math.max(nonDiscountBaseline + discountComponentScenario, 0)
          return {
            slab: slabKey,
            default_discount_pct: defaultDiscount,
            scenario_discount_pct: scenarioDiscount,
            default_lag_used_pct: defaultDiscount,
            lag_used_pct: lagUsed,
            other_weighted_default_pct: otherWeighted,
            other_weighted_scenario_pct: otherWeighted,
            discount_component_default_qty: discountComponentScenario,
            discount_component_scenario_qty: discountComponentScenario,
            non_discount_baseline_qty: nonDiscountBaseline,
            baseline_qty: nonDiscountBaseline,
            default_world_qty: preCrossQty,
            pre_cross_qty: preCrossQty,
            final_qty: preCrossQty,
            base_price: basePrice,
            cogs_per_unit: cogsPerUnit,
            baseline_revenue: 0,
            scenario_revenue: 0,
            baseline_profit: 0,
            scenario_profit: 0,
          }
        })

        const baselineTotal = slabs.reduce((s, x) => s + Number(x?.non_discount_baseline_qty || 0), 0)
        const preTotal = slabs.reduce((s, x) => s + Number(x?.pre_cross_qty || 0), 0)
        sizes[sizeKey] = {
          size: sizeKey,
          baseline_total_qty: baselineTotal,
          baseline_total_qty_default_world: preTotal,
          pre_cross_total_qty: preTotal,
          final_total_qty: preTotal,
          slabs,
        }
      })
      return {
        period: periodKey,
        sizes,
        impact: {
          prev12_qty: Number(sizes?.['12-ML']?.baseline_total_qty || 0),
          prev18_qty: Number(sizes?.['18-ML']?.baseline_total_qty || 0),
          pre12_qty: Number(sizes?.['12-ML']?.pre_cross_total_qty || 0),
          pre18_qty: Number(sizes?.['18-ML']?.pre_cross_total_qty || 0),
          final12_qty: Number(sizes?.['12-ML']?.final_total_qty || 0),
          final18_qty: Number(sizes?.['18-ML']?.final_total_qty || 0),
          own12_pct: 0,
          own18_pct: 0,
          overall12_pct: 0,
          overall18_pct: 0,
        },
      }
    })

    const pre12_3m = monthlyResults.reduce((s, row) => s + Number(row?.sizes?.['12-ML']?.pre_cross_total_qty || 0), 0)
    const pre18_3m = monthlyResults.reduce((s, row) => s + Number(row?.sizes?.['18-ML']?.pre_cross_total_qty || 0), 0)
    const own12 = ref12Qty > 0 ? ((pre12_3m - ref12Qty) / ref12Qty) * 100 : 0
    const own18 = ref18Qty > 0 ? ((pre18_3m - ref18Qty) / ref18Qty) * 100 : 0
    const adjusted12Pct = own12 + (e12From18 * own18)
    const adjusted18Pct = own18 + (e18From12 * own12)
    const final12_3m = ref12Qty > 0 ? Math.max(ref12Qty * (1 + adjusted12Pct / 100), 0) : Math.max(pre12_3m, 0)
    const final18_3m = ref18Qty > 0 ? Math.max(ref18Qty * (1 + adjusted18Pct / 100), 0) : Math.max(pre18_3m, 0)

    ;['12-ML', '18-ML'].forEach((sizeKey) => {
      const target = sizeKey === '12-ML' ? final12_3m : final18_3m
      const cells = []
      let sumPre = 0
      let sumBase = 0
      monthlyResults.forEach((row) => {
        const slabs = row?.sizes?.[sizeKey]?.slabs || []
        slabs.forEach((slab) => {
          const pre = Math.max(Number(slab?.pre_cross_qty || 0), 0)
          const base = Math.max(Number(slab?.non_discount_baseline_qty || 0), 0)
          sumPre += pre
          sumBase += base
          cells.push({ slab, pre, base })
        })
      })
      if (!cells.length) return
      let shares
      if (sumPre > 0) shares = cells.map((c) => c.pre / sumPre)
      else if (sumBase > 0) shares = cells.map((c) => c.base / sumBase)
      else shares = cells.map(() => 1 / cells.length)
      cells.forEach((c, idx) => {
        c.slab.final_qty = Math.max(target * shares[idx], 0)
      })
    })

    const summary = {
      '12-ML': {
        baseline_qty: 0, scenario_qty_additive: 0, final_qty: 0,
        baseline_revenue: 0, scenario_revenue: 0,
        baseline_profit: 0, scenario_profit: 0,
        baseline_investment: 0, scenario_investment: 0,
      },
      '18-ML': {
        baseline_qty: 0, scenario_qty_additive: 0, final_qty: 0,
        baseline_revenue: 0, scenario_revenue: 0,
        baseline_profit: 0, scenario_profit: 0,
        baseline_investment: 0, scenario_investment: 0,
      },
    }

    monthlyResults.forEach((row) => {
      ;['12-ML', '18-ML'].forEach((sizeKey) => {
        const block = row?.sizes?.[sizeKey]
        if (!block) return
        let baselineRevenueTotal = 0
        let scenarioRevenueTotal = 0
        let baselineProfitTotal = 0
        let scenarioProfitTotal = 0
        let baselineInvestmentTotal = 0
        let scenarioInvestmentTotal = 0
        const slabs = block?.slabs || []
        slabs.forEach((slab) => {
          const baseQty = Number(slab?.non_discount_baseline_qty || 0)
          const finalQty = Number(slab?.final_qty || 0)
          const basePrice = Number(slab?.base_price || 0)
          const defaultDiscount = Number(slab?.default_discount_pct || 0)
          const scenarioDiscount = Number(slab?.scenario_discount_pct || 0)
          const cogs = Number(slab?.cogs_per_unit || 0)
          const baselineRevenue = baseQty * basePrice
          const scenarioRevenue = finalQty * basePrice
          const baselineRevenueNet = baseQty * basePrice * (1 - (defaultDiscount / 100))
          const scenarioRevenueNet = finalQty * basePrice * (1 - (scenarioDiscount / 100))
          const baselineProfit = baselineRevenueNet - (baseQty * cogs)
          const scenarioProfit = scenarioRevenueNet - (finalQty * cogs)
          const baselineInvestment = baseQty * basePrice * (defaultDiscount / 100)
          const scenarioInvestment = finalQty * basePrice * (scenarioDiscount / 100)
          slab.baseline_revenue = baselineRevenue
          slab.scenario_revenue = scenarioRevenue
          slab.baseline_revenue_net = baselineRevenueNet
          slab.scenario_revenue_net = scenarioRevenueNet
          slab.baseline_profit = baselineProfit
          slab.scenario_profit = scenarioProfit
          slab.baseline_investment = baselineInvestment
          slab.scenario_investment = scenarioInvestment
          baselineRevenueTotal += baselineRevenue
          scenarioRevenueTotal += scenarioRevenue
          baselineProfitTotal += baselineProfit
          scenarioProfitTotal += scenarioProfit
          baselineInvestmentTotal += baselineInvestment
          scenarioInvestmentTotal += scenarioInvestment
        })
        block.final_total_qty = slabs.reduce((s, slab) => s + Number(slab?.final_qty || 0), 0)
        block.baseline_revenue_total = baselineRevenueTotal
        block.scenario_revenue_total = scenarioRevenueTotal
        block.baseline_profit_total = baselineProfitTotal
        block.scenario_profit_total = scenarioProfitTotal
        block.baseline_investment_total = baselineInvestmentTotal
        block.scenario_investment_total = scenarioInvestmentTotal

        summary[sizeKey].baseline_qty += Number(block?.baseline_total_qty || 0)
        summary[sizeKey].scenario_qty_additive += Number(block?.pre_cross_total_qty || 0)
        summary[sizeKey].final_qty += Number(block?.final_total_qty || 0)
        summary[sizeKey].baseline_revenue += baselineRevenueTotal
        summary[sizeKey].scenario_revenue += scenarioRevenueTotal
        summary[sizeKey].baseline_profit += baselineProfitTotal
        summary[sizeKey].scenario_profit += scenarioProfitTotal
        summary[sizeKey].baseline_investment += baselineInvestmentTotal
        summary[sizeKey].scenario_investment += scenarioInvestmentTotal
      })
    })

    const finalizeSizeSummary = (sizeKey) => {
      const s = summary[sizeKey]
      const refQty = Number(baseSummary?.[sizeKey]?.reference_qty || 0)
      const refRevNet = Number(baseSummary?.[sizeKey]?.reference_revenue || 0)
      const refProfit = Number(baseSummary?.[sizeKey]?.reference_profit || 0)
      const refInvestment = Number(baseSummary?.[sizeKey]?.reference_investment || 0)
      const refRev = refRevNet + refInvestment
      const refAvail = Number(baseSummary?.[sizeKey]?.reference_available || 0)
      return {
        baseline_qty: s.baseline_qty,
        scenario_qty_additive: s.scenario_qty_additive,
        discount_component_qty: s.scenario_qty_additive - s.baseline_qty,
        final_qty: s.final_qty,
        volume_delta_pct: s.baseline_qty > 0 ? ((s.final_qty - s.baseline_qty) / s.baseline_qty) * 100 : 0,
        volume_delta_additive_pct: s.baseline_qty > 0 ? ((s.scenario_qty_additive - s.baseline_qty) / s.baseline_qty) * 100 : 0,
        baseline_revenue: s.baseline_revenue,
        scenario_revenue: s.scenario_revenue,
        revenue_delta_pct: s.baseline_revenue > 0 ? ((s.scenario_revenue - s.baseline_revenue) / s.baseline_revenue) * 100 : 0,
        baseline_profit: s.baseline_profit,
        scenario_profit: s.scenario_profit,
        profit_delta_pct: Math.abs(s.baseline_profit) > 1e-9 ? ((s.scenario_profit - s.baseline_profit) / Math.abs(s.baseline_profit)) * 100 : 0,
        baseline_investment: s.baseline_investment,
        scenario_investment: s.scenario_investment,
        investment_delta_pct: s.baseline_investment > 0 ? ((s.scenario_investment - s.baseline_investment) / s.baseline_investment) * 100 : 0,
        reference_qty: refQty,
        reference_revenue_net: refRevNet,
        reference_revenue: refRev,
        reference_profit: refProfit,
        reference_investment: refInvestment,
        vs_reference_volume_pct: refQty > 0 ? ((s.final_qty - refQty) / refQty) * 100 : 0,
        vs_reference_revenue_pct: refRev > 0 ? ((s.scenario_revenue - refRev) / refRev) * 100 : 0,
        vs_reference_profit_pct: Math.abs(refProfit) > 1e-9 ? ((s.scenario_profit - refProfit) / Math.abs(refProfit)) * 100 : 0,
        vs_reference_investment_pct: refInvestment > 0 ? ((s.scenario_investment - refInvestment) / refInvestment) * 100 : 0,
        reference_available: refAvail,
      }
    }

    const s12 = finalizeSizeSummary('12-ML')
    const s18 = finalizeSizeSummary('18-ML')
    const totalBaselineQty = s12.baseline_qty + s18.baseline_qty
    const totalScenarioAddQty = s12.scenario_qty_additive + s18.scenario_qty_additive
    const totalFinalQty = s12.final_qty + s18.final_qty
    const totalBaselineRevenue = s12.baseline_revenue + s18.baseline_revenue
    const totalScenarioRevenue = s12.scenario_revenue + s18.scenario_revenue
    const totalBaselineProfit = s12.baseline_profit + s18.baseline_profit
    const totalScenarioProfit = s12.scenario_profit + s18.scenario_profit
    const totalBaselineInvestment = s12.baseline_investment + s18.baseline_investment
    const totalScenarioInvestment = s12.scenario_investment + s18.scenario_investment
    const refTotalQty = Number(baseSummary?.TOTAL?.reference_qty || 0)
    const refTotalRevNet = Number(baseSummary?.TOTAL?.reference_revenue || 0)
    const refTotalProfit = Number(baseSummary?.TOTAL?.reference_profit || 0)
    const ref12Investment = Number(baseSummary?.['12-ML']?.reference_investment || 0)
    const ref18Investment = Number(baseSummary?.['18-ML']?.reference_investment || 0)
    const refTotalInvestmentRaw = Number(baseSummary?.TOTAL?.reference_investment || 0)
    const refTotalInvestment = refTotalInvestmentRaw > 0 ? refTotalInvestmentRaw : (ref12Investment + ref18Investment)
    const refTotalRev = refTotalRevNet + refTotalInvestment
    const refTotalAvailRaw = Number(baseSummary?.TOTAL?.reference_available || 0)
    const ref12Avail = Number(baseSummary?.['12-ML']?.reference_available || 0)
    const ref18Avail = Number(baseSummary?.['18-ML']?.reference_available || 0)
    const refTotalAvail = refTotalAvailRaw > 0 ? refTotalAvailRaw : ((ref12Avail > 0 || ref18Avail > 0) ? 1 : 0)
    const baselineVolumeMl = (s12.baseline_qty * 12) + (s18.baseline_qty * 18)
    const scenarioVolumeMlAdd = (s12.scenario_qty_additive * 12) + (s18.scenario_qty_additive * 18)
    const finalVolumeMl = (s12.final_qty * 12) + (s18.final_qty * 18)
    const refVolumeMl = (Number(baseSummary?.['12-ML']?.reference_qty || 0) * 12) + (Number(baseSummary?.['18-ML']?.reference_qty || 0) * 18)

    const summary3m = {
      '12-ML': s12,
      '18-ML': s18,
      TOTAL: {
        baseline_qty: totalBaselineQty,
        scenario_qty_additive: totalScenarioAddQty,
        discount_component_qty: totalScenarioAddQty - totalBaselineQty,
        final_qty: totalFinalQty,
        volume_delta_pct: totalBaselineQty > 0 ? ((totalFinalQty - totalBaselineQty) / totalBaselineQty) * 100 : 0,
        volume_delta_additive_pct: totalBaselineQty > 0 ? ((totalScenarioAddQty - totalBaselineQty) / totalBaselineQty) * 100 : 0,
        baseline_revenue: totalBaselineRevenue,
        scenario_revenue: totalScenarioRevenue,
        revenue_delta_pct: totalBaselineRevenue > 0 ? ((totalScenarioRevenue - totalBaselineRevenue) / totalBaselineRevenue) * 100 : 0,
        baseline_profit: totalBaselineProfit,
        scenario_profit: totalScenarioProfit,
        profit_delta_pct: Math.abs(totalBaselineProfit) > 1e-9 ? ((totalScenarioProfit - totalBaselineProfit) / Math.abs(totalBaselineProfit)) * 100 : 0,
        baseline_investment: totalBaselineInvestment,
        scenario_investment: totalScenarioInvestment,
        investment_delta_pct: totalBaselineInvestment > 0 ? ((totalScenarioInvestment - totalBaselineInvestment) / totalBaselineInvestment) * 100 : 0,
        reference_qty: refTotalQty,
        reference_revenue_net: refTotalRevNet,
        reference_revenue: refTotalRev,
        reference_profit: refTotalProfit,
        reference_investment: refTotalInvestment,
        vs_reference_volume_pct: refTotalQty > 0 ? ((totalFinalQty - refTotalQty) / refTotalQty) * 100 : 0,
        vs_reference_revenue_pct: refTotalRev > 0 ? ((totalScenarioRevenue - refTotalRev) / refTotalRev) * 100 : 0,
        vs_reference_profit_pct: Math.abs(refTotalProfit) > 1e-9 ? ((totalScenarioProfit - refTotalProfit) / Math.abs(refTotalProfit)) * 100 : 0,
        vs_reference_investment_pct: refTotalInvestment > 0 ? ((totalScenarioInvestment - refTotalInvestment) / refTotalInvestment) * 100 : 0,
        reference_available: refTotalAvail,
        baseline_volume_ml: baselineVolumeMl,
        scenario_volume_ml_additive: scenarioVolumeMlAdd,
        final_volume_ml: finalVolumeMl,
        reference_volume_ml: refVolumeMl,
        volume_ml_delta_pct: baselineVolumeMl > 0 ? ((finalVolumeMl - baselineVolumeMl) / baselineVolumeMl) * 100 : 0,
        volume_ml_delta_additive_pct: baselineVolumeMl > 0 ? ((scenarioVolumeMlAdd - baselineVolumeMl) / baselineVolumeMl) * 100 : 0,
        vs_reference_volume_ml_pct: refVolumeMl > 0 ? ((finalVolumeMl - refVolumeMl) / refVolumeMl) * 100 : 0,
      },
    }

    monthlyResults.forEach((row) => {
      row.impact = {
        prev12_qty: Number(row?.sizes?.['12-ML']?.baseline_total_qty || 0),
        prev18_qty: Number(row?.sizes?.['18-ML']?.baseline_total_qty || 0),
        pre12_qty: Number(row?.sizes?.['12-ML']?.pre_cross_total_qty || 0),
        pre18_qty: Number(row?.sizes?.['18-ML']?.pre_cross_total_qty || 0),
        final12_qty: Number(row?.sizes?.['12-ML']?.final_total_qty || 0),
        final18_qty: Number(row?.sizes?.['18-ML']?.final_total_qty || 0),
        own12_pct: own12,
        own18_pct: own18,
        overall12_pct: adjusted12Pct,
        overall18_pct: adjusted18Pct,
      }
    })

    return {
      ...data,
      reference_mode: selectedReferenceMode,
      monthly_results: monthlyResults,
      summary_3m: summary3m,
    }
  }, [data, periods, scenarioDiscountsByPeriod, selectedReferenceMode])

  const summaryBySize = useMemo(() => computedPlannerData?.summary_3m || {}, [computedPlannerData?.summary_3m])
  const selectedReferenceSummary = useMemo(() => {
    const preferred = referenceByMode?.[selectedReferenceMode]?.summary_3m
    if (preferred && typeof preferred === 'object') return preferred
    const fallback = referenceByMode?.last_3m_before_projection?.summary_3m
    if (fallback && typeof fallback === 'object') return fallback
    return computedPlannerData?.summary_3m || {}
  }, [referenceByMode, selectedReferenceMode, computedPlannerData?.summary_3m])

  if (isLoading) {
    return (
      <div className="bg-white rounded-lg shadow-md border border-slate-200 p-10 flex flex-col items-center justify-center gap-3">
        <Loader2 className="w-8 h-8 animate-spin text-primary" />
        <div className="text-center">
          <p className="text-base font-semibold text-body">Updating cross-size planner</p>
          <p className="text-sm text-muted">Recomputing 3-month slab impact from backend model.</p>
        </div>
      </div>
    )
  }

  if (isError || data?.success === false) {
    return (
      <div className="bg-brand-dangerLight border border-danger rounded-lg p-4 flex items-start space-x-3">
        <AlertCircle className="text-danger flex-shrink-0 mt-0.5" size={20} />
        <div>
          <h4 className="font-semibold text-body">Step 4 Error</h4>
          <p className="text-muted text-sm">{errorMessage || data?.message || 'Failed to initialize cross-size planner.'}</p>
        </div>
      </div>
    )
  }

  if (!periods.length || !Object.keys(monthlyByPeriod).length) {
    return (
      <div className="bg-white rounded-lg shadow-md p-8">
        <h3 className="text-xl font-semibold text-body mb-2">Step 4 Requires Step 3 Output</h3>
        <p className="text-muted">Run Step 3 modeling first. Then Step 4 will initialize the cross-size planner.</p>
        <button
          type="button"
          onClick={() => {
            if (typeof onInitialize === 'function') onInitialize()
          }}
          className="mt-4 px-4 py-2 rounded-md bg-white text-body border border-primary text-sm font-semibold"
        >
          Refresh Step 4
        </button>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <div className="bg-white rounded-lg shadow-md p-4 sticky top-0 z-10">
        <div className="flex items-start justify-between gap-4">
          <h3 className="text-xl font-bold text-body">Step 4: Cross-Size Scenario Planner</h3>
          <div className="flex items-center gap-2">
            <select
              value={selectedReferenceMode}
              onChange={(e) => {
                if (typeof onDisplayReferenceModeChange === 'function') {
                  onDisplayReferenceModeChange(String(e.target.value || 'ly_same_3m'))
                }
              }}
              className="px-3 py-2 rounded-md border border-slate-300 bg-white text-sm text-body"
            >
              <option value="ly_same_3m">Y-o-Y</option>
              <option value="last_3m_before_projection">Q-o-Q</option>
            </select>
            <button
              type="button"
              onClick={handleReset}
              className="inline-flex items-center justify-center gap-2 px-4 py-2 rounded-md border border-gray-300 bg-white text-sm font-semibold text-body self-start"
            >
              <RotateCcw className="w-4 h-4" />
              Reset All Months
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-3 gap-3 mt-4">
          {[
            { key: '12-ML', title: '12-ML', isTotal: false },
            { key: 'TOTAL', title: 'TOTAL (12-ML + 18-ML)', isTotal: true },
            { key: '18-ML', title: '18-ML', isTotal: false },
          ].map((card) => {
            const summary = summaryBySize?.[card.key] || {}
            const refSummary = selectedReferenceSummary?.[card.key] || {}
            const refSummary12 = selectedReferenceSummary?.['12-ML'] || {}
            const refSummary18 = selectedReferenceSummary?.['18-ML'] || {}
            const volumeAbs = card.isTotal
              ? Number(summary.final_volume_ml ?? 0)
              : Number(summary.final_qty ?? summary.scenario_qty_additive ?? 0)
            const referenceQty = card.isTotal
              ? (Number(refSummary12?.reference_qty || 0) * 12) + (Number(refSummary18?.reference_qty || 0) * 18)
              : Number(refSummary?.reference_qty || 0)
            const referenceRevenueRaw = Number(refSummary?.reference_revenue || 0)
            const referenceProfit = Number(refSummary?.reference_profit || 0)
            const summary12 = summaryBySize?.['12-ML'] || {}
            const summary18 = summaryBySize?.['18-ML'] || {}
            const pickPositive = (...vals) => {
              for (const v of vals) {
                const n = Number(v)
                if (Number.isFinite(n) && n > 0) return n
              }
              return 0
            }
            const referenceInvestmentRaw = pickPositive(
              refSummary?.reference_investment,
              summaryBySize?.[card.key]?.reference_investment
            )
            const referenceInvestment = card.isTotal
              ? (referenceInvestmentRaw > 0
                ? referenceInvestmentRaw
                : (
                  pickPositive(refSummary12?.reference_investment, summary12?.reference_investment) +
                  pickPositive(refSummary18?.reference_investment, summary18?.reference_investment)
                ))
              : referenceInvestmentRaw
            const referenceRevenue = Object.prototype.hasOwnProperty.call(refSummary || {}, 'reference_revenue_net')
              ? referenceRevenueRaw
              : (referenceRevenueRaw + referenceInvestment)
            const hasReference = Number(refSummary?.reference_available || 0) > 0 && referenceQty > 0
            const volumePct = hasReference
              ? Number(((volumeAbs - referenceQty) / referenceQty) * 100)
              : Number(summary.volume_delta_pct ?? summary.volume_delta_additive_pct ?? 0)
            const revenueAbs = Number(summary.scenario_revenue || 0)
            const profitAbs = Number(summary.scenario_profit || 0)
            const investmentAbs = Number(summary.scenario_investment ?? 0)
            const revenuePct = hasReference
              ? Number(referenceRevenue > 0 ? ((revenueAbs - referenceRevenue) / referenceRevenue) * 100 : 0)
              : Number(summary.revenue_delta_pct || 0)
            const grossMarginAbs = revenueAbs > 0 ? ((profitAbs / revenueAbs) * 100) : 0
            const referenceGrossMargin = referenceRevenue > 0 ? ((referenceProfit / referenceRevenue) * 100) : 0
            const grossMarginPct = hasReference
              ? Number(grossMarginAbs - referenceGrossMargin)
              : 0
            const hasReferenceInvestment = referenceInvestment > 0
            const investmentPct = hasReferenceInvestment
              ? Number(((investmentAbs - referenceInvestment) / referenceInvestment) * 100)
              : 0
            const ctsAbs = Number(revenueAbs > 0 ? (investmentAbs / revenueAbs) * 100 : 0)
            const referenceCts = Number(referenceRevenue > 0 ? (referenceInvestment / referenceRevenue) * 100 : 0)
            const hasReferenceCts = referenceCts > 0
            const ctsPct = hasReferenceCts
              ? Number(((ctsAbs - referenceCts) / referenceCts) * 100)
              : 0
            return (
              <div key={card.key} className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                <p className="text-sm font-semibold text-body">{card.title}</p>
                <div className={`grid gap-3 mt-3 ${card.isTotal ? 'grid-cols-2 md:grid-cols-3 xl:grid-cols-5' : 'grid-cols-3'}`}>
                  <div>
                    <p className="text-[10px] uppercase tracking-wide text-muted">{card.isTotal ? 'Volume (ML)' : 'Volume'}</p>
                    <p className="text-2xl leading-none mt-2 font-semibold text-body">{formatCompact(volumeAbs)}</p>
                    <p className={`text-sm font-semibold mt-1 ${pctToneClass(volumePct)}`}>
                      {formatSignedPct(volumePct)}
                    </p>
                  </div>
                  <div>
                    <p className="text-[10px] uppercase tracking-wide text-muted">Revenue</p>
                    <p className="text-2xl leading-none mt-2 font-semibold text-body">{formatCompact(revenueAbs)}</p>
                    <p className={`text-sm font-semibold mt-1 ${pctToneClass(revenuePct)}`}>
                      {formatSignedPct(revenuePct)}
                    </p>
                  </div>
                  <div>
                    <p className="text-[10px] uppercase tracking-wide text-muted">Gross Margin %</p>
                    <p className="text-2xl leading-none mt-2 font-semibold text-body">{formatFixed(grossMarginAbs, 2)}%</p>
                    <p className={`text-sm font-semibold mt-1 ${pctToneClass(grossMarginPct)}`}>
                      {formatSignedPct(grossMarginPct)}
                    </p>
                  </div>
                  {card.isTotal ? (
                    <div>
                      <p className="text-[10px] uppercase tracking-wide text-muted">Investment</p>
                      <p className="text-2xl leading-none mt-2 font-semibold text-body">{formatCompact(investmentAbs)}</p>
                      <p className={`text-sm font-semibold mt-1 ${pctToneClass(investmentPct)}`}>
                        {formatSignedPct(investmentPct)}
                      </p>
                    </div>
                  ) : null}
                  {card.isTotal ? (
                    <div>
                      <p className="text-[10px] uppercase tracking-wide text-muted">CTS</p>
                      <p className="text-2xl leading-none mt-2 font-semibold text-body">{formatFixed(ctsAbs, 2)}%</p>
                      <p className={`text-sm font-semibold mt-1 ${pctToneClass(ctsPct)}`}>
                        {formatSignedPct(ctsPct)}
                      </p>
                    </div>
                  ) : null}
                </div>
              </div>
            )
          })}
        </div>
      </div>

      <div className="bg-white rounded-lg shadow-md p-4">
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
          {['12-ML', '18-ML'].map((sizeKey) => (
            <div key={sizeKey} className="rounded-lg border border-slate-200 bg-slate-50/70 p-4 space-y-4">
              <h4 className="text-lg font-semibold text-body">{sizeKey}</h4>
              <div className="grid grid-cols-1 xl:grid-cols-3 gap-3">
                {periods.map((periodKey) => {
                  const periodRow = computedPlannerData?.monthly_results?.find((r) => String(r?.period) === String(periodKey))
                  const sizeBlock = periodRow?.sizes?.[sizeKey]
                  const slabs = [...(sizeBlock?.slabs || [])].sort((a, b) => slabSortKey(a?.slab) - slabSortKey(b?.slab))
                  if (!sizeBlock) return null

                  return (
                    <div key={`${sizeKey}-${periodKey}`} className="rounded-lg border border-slate-200 bg-white p-3 space-y-3">
                      <div className="flex items-center justify-between border-b border-slate-100 pb-2">
                        <h5 className="text-base font-semibold text-body">{formatPeriodLabel(periodKey)}</h5>
                      </div>

                      {slabs.map((slab) => (
                        <div key={`${sizeKey}-${periodKey}-${slab.slab}`} className="rounded-lg border border-slate-200 bg-slate-50/60 p-2.5 space-y-2">
                          <div className="flex items-center justify-between gap-2">
                            <div className="text-sm font-semibold text-body">{slab.slab}</div>
                            <input
                              type="text"
                              inputMode="decimal"
                              value={draftDiscountsByPeriod?.[periodKey]?.[sizeKey]?.[String(slab.slab)] ?? formatDiscountInput(slab.scenario_discount_pct)}
                              onChange={(e) => {
                                setDraftRawValue(periodKey, sizeKey, String(slab.slab), e.target.value)
                              }}
                              onBlur={(e) => {
                                handleScenarioDiscountChange(periodKey, sizeKey, String(slab.slab), e.target.value)
                              }}
                              onKeyDown={(e) => {
                                if (e.key === 'Enter') {
                                  handleScenarioDiscountChange(periodKey, sizeKey, String(slab.slab), e.currentTarget.value)
                                  e.currentTarget.blur()
                                }
                              }}
                              className="w-24 px-3 py-2 bg-white border border-gray-300 rounded-lg text-sm font-semibold tabular-nums"
                            />
                          </div>
                        </div>
                      ))}
                    </div>
                  )
                })}
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="bg-white rounded-lg shadow-md p-4">
        <details>
          <summary className="cursor-pointer text-sm font-semibold text-body">
            Step 4 Calculation Trace (Equation + Values)
          </summary>
          <div className="mt-4 space-y-4">
            {periods.map((periodKey, monthIdx) => {
              const periodRow = computedPlannerData?.monthly_results?.find((r) => String(r?.period) === String(periodKey)) || {}
              const impact = periodRow?.impact || {}
                  return (
                <div key={`trace-${periodKey}`} className="rounded-lg border border-slate-200 bg-slate-50 p-3 space-y-3">
                  <div className="flex flex-wrap items-center gap-4">
                    <p className="text-sm font-semibold text-body">{formatPeriodLabel(periodKey)}</p>
                    <p className="text-xs text-muted">
                      Prev: 12={formatWhole(impact?.prev12_qty)} | 18={formatWhole(impact?.prev18_qty)}
                    </p>
                    <p className="text-xs text-muted">
                      Pre: 12={formatWhole(impact?.pre12_qty)} | 18={formatWhole(impact?.pre18_qty)}
                    </p>
                    <p className="text-xs text-muted">
                      Final: 12={formatWhole(impact?.final12_qty)} | 18={formatWhole(impact?.final18_qty)}
                    </p>
                    <p className="text-xs text-muted">Own12: {formatSignedPct(impact?.own12_pct)}</p>
                    <p className="text-xs text-muted">Own18: {formatSignedPct(impact?.own18_pct)}</p>
                    <p className="text-xs text-muted">Overall12: {formatSignedPct(impact?.overall12_pct)}</p>
                    <p className="text-xs text-muted">Overall18: {formatSignedPct(impact?.overall18_pct)}</p>
                  </div>

                  <div className="grid grid-cols-1 xl:grid-cols-2 gap-3">
                    {['12-ML', '18-ML'].map((sizeKey) => {
                      const sizeBlock = periodRow?.sizes?.[sizeKey]
                      if (!sizeBlock) return null
                      const slabs = [...(sizeBlock?.slabs || [])].sort((a, b) => slabSortKey(a?.slab) - slabSortKey(b?.slab))
                      return (
                        <div key={`trace-${periodKey}-${sizeKey}`} className="rounded-lg border border-slate-200 bg-white p-3 space-y-2">
                          <p className="text-sm font-semibold text-body">
                            {sizeKey} | Baseline(non-discount): {formatWhole(sizeBlock?.baseline_total_qty)} | Baseline(default-world): {formatWhole(sizeBlock?.baseline_total_qty_default_world)} | Scenario Qty: {formatWhole(sizeBlock?.pre_cross_total_qty)} | Final Qty: {formatWhole(sizeBlock?.final_total_qty)}
                          </p>
                          {slabs.map((slab) => {
                            const slabKey = String(slab?.slab || '')
                            const c = coeffBySizeSlab?.[sizeKey]?.[slabKey] || {}
                            const bBase = Number(c?.coef_base_discount_pct || 0)
                            const bLag = Number(c?.coef_lag1_base_discount_pct || 0)
                            const bOther = Number(c?.coef_other_slabs_weighted_base_discount_pct || 0)
                            const defaultDiscount = Number(slab?.default_discount_pct || 0)
                            const scenarioDiscount = Number(slab?.scenario_discount_pct || 0)
                            const defaultLag = Number(slab?.default_lag_used_pct ?? defaultDiscount)
                            const scenarioLag = Number(slab?.lag_used_pct || 0)
                            const otherDefault = Number(slab?.other_weighted_default_pct || 0)
                            const otherScenario = Number(slab?.other_weighted_scenario_pct || 0)
                            const nonDiscountBaseline = Number(slab?.non_discount_baseline_qty ?? slab?.baseline_qty ?? 0)
                            const discountComponentDefaultBackend = Number(
                              slab?.discount_component_default_qty ?? ((bBase * defaultDiscount) + (bLag * defaultLag) + (bOther * otherDefault))
                            )
                            const discountComponentScenarioBackend = Number(
                              slab?.discount_component_scenario_qty ?? ((bBase * scenarioDiscount) + (bLag * scenarioLag) + (bOther * otherScenario))
                            )
                            const discountComponentDefaultRecomputed = (bBase * defaultDiscount) + (bLag * defaultLag) + (bOther * otherDefault)
                            const discountComponentScenarioRecomputed = (bBase * scenarioDiscount) + (bLag * scenarioLag) + (bOther * otherScenario)
                            const defaultWorldQty = Number(slab?.default_world_qty ?? Math.max(nonDiscountBaseline + discountComponentDefaultBackend, 0))
                            const recomputedPreCrossQty = Number(Math.max(nonDiscountBaseline + discountComponentScenarioBackend, 0))
                            const preCrossQty = Number(slab?.pre_cross_qty ?? recomputedPreCrossQty)
                            const finalQty = Number(slab?.final_qty || 0)
                            const baselineRevenue = Number(slab?.baseline_revenue || 0)
                            const scenarioRevenue = Number(slab?.scenario_revenue || 0)
                            const baselineProfit = Number(slab?.baseline_profit || 0)
                            const scenarioProfit = Number(slab?.scenario_profit || 0)
                            const mismatchDefault = Math.abs(defaultWorldQty - Math.max(nonDiscountBaseline + discountComponentDefaultBackend, 0))
                            const mismatchPre = Math.abs(preCrossQty - recomputedPreCrossQty)
                            const mismatchFormulaDefault = Math.abs(discountComponentDefaultBackend - discountComponentDefaultRecomputed)
                            const mismatchFormulaScenario = Math.abs(discountComponentScenarioBackend - discountComponentScenarioRecomputed)
                            const hasMismatch = (mismatchDefault > 1e-3) || (mismatchPre > 1e-3) || (mismatchFormulaDefault > 1e-3) || (mismatchFormulaScenario > 1e-3)
                            return (
                              <div key={`trace-${periodKey}-${sizeKey}-${slabKey}`} className="rounded border border-slate-200 bg-slate-50/70 p-2">
                                <p className="text-xs font-semibold text-body mb-1">{slabKey}</p>
                                {hasMismatch && (
                                  <p className="text-[11px] text-danger font-semibold mb-1">
                                    Mismatch detected: backend qty fields do not match equation output for this slab/month.
                                  </p>
                                )}
                                <p className="text-[11px] font-mono text-muted">
                                  default_component = ({formatFixed(bBase, 4)} x {formatFixed(defaultDiscount, 2)}) + ({formatFixed(bLag, 4)} x {formatFixed(defaultLag, 2)}) + ({formatFixed(bOther, 4)} x {formatFixed(otherDefault, 2)}) = {formatFixed(discountComponentDefaultBackend, 2)}
                                </p>
                                <p className="text-[11px] font-mono text-muted">
                                  scenario_component = ({formatFixed(bBase, 4)} x {formatFixed(scenarioDiscount, 2)}) + ({formatFixed(bLag, 4)} x {formatFixed(scenarioLag, 2)}) + ({formatFixed(bOther, 4)} x {formatFixed(otherScenario, 2)}) = {formatFixed(discountComponentScenarioBackend, 2)}
                                </p>
                                <p className="text-[11px] font-mono text-muted">
                                  non_discount_baseline = {formatFixed(nonDiscountBaseline, 2)} | default_world_qty = baseline + default_component = {formatFixed(defaultWorldQty, 2)} | pre_cross_qty = baseline + scenario_component = {formatFixed(preCrossQty, 2)} | final_qty = {formatFixed(finalQty, 2)}
                                </p>
                                <p className="text-[11px] font-mono text-muted">
                                  recomputed: default_component={formatFixed(discountComponentDefaultRecomputed, 2)} | scenario_component={formatFixed(discountComponentScenarioRecomputed, 2)} | default_world={formatFixed(Math.max(nonDiscountBaseline + discountComponentDefaultBackend, 0), 2)} | pre_cross={formatFixed(recomputedPreCrossQty, 2)}
                                </p>
                                <p className="text-[11px] font-mono text-muted">
                                  baseline_revenue = {formatFixed(baselineRevenue, 2)} | scenario_revenue = {formatFixed(scenarioRevenue, 2)} | baseline_profit = {formatFixed(baselineProfit, 2)} | scenario_profit = {formatFixed(scenarioProfit, 2)}
                                </p>
                              </div>
                            )
                          })}
                        </div>
                      )
                    })}
                  </div>
                </div>
              )
            })}
          </div>
        </details>
      </div>
    </div>
  )
}

export default CrossSizePlanner
