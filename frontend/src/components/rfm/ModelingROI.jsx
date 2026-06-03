import { useEffect, useMemo, useState } from 'react'
import { AlertCircle, Loader2, Play, X } from 'lucide-react'
import {
  Area,
  Bar,
  LabelList,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

const formatDate = (value) => {
  try {
    return new Date(value).toLocaleDateString()
  } catch {
    return String(value)
  }
}

const formatMonth = (value) => {
  try {
    return new Date(value).toLocaleDateString(undefined, { month: 'short' })
  } catch {
    return String(value)
  }
}

const fmt = (value) => Number(value || 0).toFixed(2)
const formatWhole = (value) => Number(value || 0).toLocaleString(undefined, { maximumFractionDigits: 0 })
const formatMetric = (value) => Number(value || 0).toLocaleString(undefined, { maximumFractionDigits: 2 })
const formatRoiAxisTick = (value) => {
  const n = Number(value)
  if (!Number.isFinite(n)) return ''
  if (Math.abs(n) >= 1000) {
    return n.toLocaleString(undefined, { maximumFractionDigits: 0 })
  }
  return n.toFixed(2).replace(/\.?0+$/, '')
}

const getAdjustedR2 = (r2, sampleSize, predictorCount) => {
  const n = Number(sampleSize)
  const p = Number(predictorCount)
  const score = Number(r2)
  if (!Number.isFinite(n) || !Number.isFinite(p) || !Number.isFinite(score)) return null
  if (n <= p + 1 || p < 0) return null
  return 1 - ((1 - score) * (n - 1)) / (n - p - 1)
}
const toDayKey = (value) => {
  const raw = String(value || '')
  const match = raw.match(/^(\d{4})-(\d{2})-(\d{2})/)
  if (match) return `${match[1]}-${match[2]}-${match[3]}`
  const d = new Date(value)
  if (Number.isNaN(d.getTime())) return raw
  return d.toISOString().slice(0, 10)
}
const slabSortKey = (slab) => {
  const text = String(slab || '').toLowerCase()
  const m = text.match(/\d+/)
  if (m) return [0, parseInt(m[0], 10), text]
  return [1, Number.MAX_SAFE_INTEGER, text]
}
const isCombinedSlabKey = (slab) => String(slab || '').trim().toLowerCase() === 'combined_all_slabs'

const ModelingROI = ({
  data,
  isLoading,
  isError,
  errorMessage,
  onRun,
  settings,
  showControls = true,
}) => {
  const [activeSize, setActiveSize] = useState('')
  const [activeSlab, setActiveSlab] = useState('')
  const [isModelModalOpen, setIsModelModalOpen] = useState(false)
  const [includeLagDiscount, setIncludeLagDiscount] = useState(true)
  const [cogsPerUnit, setCogsPerUnit] = useState(0)
  const [roiMode, setRoiMode] = useState('both')

  useEffect(() => {
    if (!settings) return
    if (typeof settings.include_lag_discount === 'boolean') {
      setIncludeLagDiscount(settings.include_lag_discount)
    }
    const cogs = Number(settings.cogs_per_unit)
    if (Number.isFinite(cogs) && cogs >= 0) {
      setCogsPerUnit(cogs)
    }
    if (typeof settings.roi_mode === 'string') {
      setRoiMode(settings.roi_mode)
    }
  }, [settings])

  const validResults = useMemo(() => {
    const slabs = (data?.slab_results || []).filter((s) => s.valid)
    return slabs.sort((a, b) => {
      const sa = String(a?.size || a?.model_coefficients?.size_key || '')
      const sb = String(b?.size || b?.model_coefficients?.size_key || '')
      if (sa !== sb) return sa.localeCompare(sb)
      const ka = slabSortKey(a?.slab)
      const kb = slabSortKey(b?.slab)
      if (ka[0] !== kb[0]) return ka[0] - kb[0]
      if (ka[1] !== kb[1]) return ka[1] - kb[1]
      return ka[2].localeCompare(kb[2])
    })
  }, [data])

  const sizeOptions = useMemo(() => (
    Array.from(
      new Set(
        validResults
          .map((item) => String(item?.size || item?.model_coefficients?.size_key || '').trim())
          .filter(Boolean)
      )
    )
  ), [validResults])

  const validSlabs = useMemo(() => {
    if (!activeSize) return []
    return validResults.filter((item) => String(item?.size || item?.model_coefficients?.size_key || '') === activeSize)
  }, [validResults, activeSize])

  const slabOnlyResults = useMemo(
    () => validSlabs.filter((item) => !isCombinedSlabKey(item?.slab)),
    [validSlabs]
  )

  useEffect(() => {
    if (!sizeOptions.length) {
      setActiveSize('')
      return
    }
    if (!sizeOptions.includes(activeSize)) {
      setActiveSize(sizeOptions[0])
    }
  }, [sizeOptions, activeSize])

  useEffect(() => {
    if (!slabOnlyResults.length) {
      setActiveSlab('')
      return
    }
    if (!slabOnlyResults.find((s) => s.slab === activeSlab)) {
      setActiveSlab(slabOnlyResults[0].slab)
    }
  }, [slabOnlyResults, activeSlab])

  const slabData = slabOnlyResults.find((s) => s.slab === activeSlab) || slabOnlyResults[0]
  const slabSizeKey = String(slabData?.size || slabData?.model_coefficients?.size_key || activeSize || '').trim()
  const selectedSizeCombinedRoiSummary = useMemo(() => {
    if (!slabOnlyResults.length) return null
    const totalSpend = slabOnlyResults.reduce((acc, item) => acc + Number(item?.summary?.total_spend || 0), 0)
    const totalIncrementalRevenue = slabOnlyResults.reduce(
      (acc, item) => acc + Number(item?.summary?.total_incremental_revenue || 0),
      0
    )
    const totalIncrementalProfit = slabOnlyResults.reduce(
      (acc, item) => acc + Number(item?.summary?.total_incremental_profit || 0),
      0
    )
    return {
      structural_roi_1mo: totalSpend > 0 ? totalIncrementalRevenue / totalSpend : 0,
      structural_profit_roi_1mo: totalSpend > 0 ? totalIncrementalProfit / totalSpend : 0,
      slab_count: slabOnlyResults.length,
    }
  }, [slabOnlyResults])
  const overallCombinedRoiSummary = useMemo(() => {
    if (!validResults.length) return null
    const allRealSlabs = validResults.filter((item) => !isCombinedSlabKey(item?.slab))
    if (!allRealSlabs.length) return null
    const totalSpend = allRealSlabs.reduce((acc, item) => acc + Number(item?.summary?.total_spend || 0), 0)
    const totalIncrementalRevenue = allRealSlabs.reduce(
      (acc, item) => acc + Number(item?.summary?.total_incremental_revenue || 0),
      0
    )
    const totalIncrementalProfit = allRealSlabs.reduce(
      (acc, item) => acc + Number(item?.summary?.total_incremental_profit || 0),
      0
    )
    return {
      structural_roi_1mo: totalSpend > 0 ? totalIncrementalRevenue / totalSpend : 0,
      structural_profit_roi_1mo: totalSpend > 0 ? totalIncrementalProfit / totalSpend : 0,
      slab_count: allRealSlabs.length,
    }
  }, [validResults])
  const summaryBySlab = useMemo(
    () => (Array.isArray(data?.summary_by_slab) ? data.summary_by_slab : []),
    [data]
  )
  const step3SummaryRows = useMemo(() => {
    const rows = (summaryBySlab || [])
      .map((row) => {
        const raw = String(row?.Slab || '').trim()
        const m = raw.toLowerCase().match(/^slab\d+/)
        const slabKey = m ? m[0] : raw
        return {
          ...row,
          Slab: slabKey || raw,
        }
      })
      .filter((row) => {
        const slab = String(row?.Slab || '').toLowerCase()
        return slab !== 'slab0'
      })
      .filter((row) => {
        const sizeKey = String(row?.Size_Key || '').trim()
        return !activeSize || !sizeKey || sizeKey === activeSize
      })

    return rows.sort((a, b) => {
      const ai = slabSortKey(a?.Slab)
      const bi = slabSortKey(b?.Slab)
      if (ai[0] !== bi[0]) return ai[0] - bi[0]
      if (ai[1] !== bi[1]) return ai[1] - bi[1]
      return ai[2].localeCompare(bi[2])
    })
  }, [summaryBySlab, activeSize])

  useEffect(() => {
    const fromModel = Number(slabData?.summary?.cogs_per_unit)
    if (Number.isFinite(fromModel) && fromModel >= 0) {
      setCogsPerUnit(fromModel)
    }
  }, [slabData?.summary?.cogs_per_unit])

  const roiChartData = useMemo(() => {
    if (!slabData) return []

    const fullTimeline = [...(slabData.predicted_vs_actual || [])]
      .sort((a, b) => new Date(a.period).getTime() - new Date(b.period).getTime())

    const roiByPeriod = new Map(
      (slabData.roi_points || []).map((p) => [toDayKey(p.period), p])
    )

    return fullTimeline.map((p) => {
      const key = toDayKey(p.period)
      const roiPoint = roiByPeriod.get(key)
      return {
        ...p,
        roi_1mo: roiPoint?.roi_1mo ?? null,
        profit_roi_1mo: roiPoint?.profit_roi_1mo ?? null,
      }
    })
  }, [slabData])

  const ridgePredictorCount = Number(slabData?.model_coefficients?.include_lag_discount) > 0 ? 4 : 3
  const ridgeAdjustedR2 = useMemo(
    () => getAdjustedR2(slabData?.model_coefficients?.stage2_r2, slabData?.predicted_vs_actual?.length, ridgePredictorCount),
    [slabData, ridgePredictorCount]
  )

  const modelMape = useMemo(() => {
    const pts = slabData?.predicted_vs_actual || []
    const valid = pts.filter(p => Number(p.actual_quantity) > 0)
    if (!valid.length) return null
    const mape = valid.reduce((sum, p) => {
      return sum + Math.abs(Number(p.actual_quantity) - Number(p.predicted_quantity)) / Number(p.actual_quantity)
    }, 0) / valid.length * 100
    return mape
  }, [slabData])

  const renderModelContent = () => {
    if (!slabData) return null
    const r2 = Number(slabData?.model_coefficients?.stage2_r2 || 0)
    const adjR2 = ridgeAdjustedR2
    const mape = modelMape

    const r2Color = r2 >= 0.85 ? '#16a34a' : r2 >= 0.65 ? '#d97706' : '#dc2626'
    const r2Bg   = r2 >= 0.85 ? '#f0fdf4' : r2 >= 0.65 ? '#fffbeb' : '#fef2f2'
    const mapeColor = mape == null ? '#64748b' : mape <= 10 ? '#16a34a' : mape <= 20 ? '#d97706' : '#dc2626'
    const mapeBg   = mape == null ? '#f8fafc' : mape <= 10 ? '#f0fdf4' : mape <= 20 ? '#fffbeb' : '#fef2f2'

    const CoefRow = ({ label, value, hint }) => {
      const v = Number(value || 0)
      const isPos = v > 0
      const isZero = Math.abs(v) < 0.01
      return (
        <div className="flex items-center justify-between py-2 border-b border-slate-100 last:border-0">
          <div>
            <span className="text-sm text-body font-medium">{label}</span>
            {hint && <span className="block text-xs text-muted">{hint}</span>}
          </div>
          <div className="flex items-center gap-2">
            {!isZero && (
              <span className={`text-xs font-semibold px-1.5 py-0.5 rounded ${isPos ? 'bg-green-50 text-green-700' : 'bg-red-50 text-red-600'}`}>
                {isPos ? '▲' : '▼'}
              </span>
            )}
            <span className="text-sm font-bold text-body tabular-nums">{Number(value || 0).toLocaleString(undefined, { maximumFractionDigits: 1 })}</span>
          </div>
        </div>
      )
    }

    return (
      <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1.6fr)_340px] gap-5">
        {/* ── Chart ── */}
        <div className="bg-white rounded-xl border border-slate-200 p-5" style={{ boxShadow: '0 1px 8px 0 rgba(30,64,175,0.07)' }}>
          <div className="flex items-start justify-between mb-1">
            <div>
              <h4 className="text-base font-semibold text-body">Actual vs Predicted</h4>
              <p className="text-xs text-muted mt-0.5">{slabSizeKey} · {slabData.slab} · monthly quantity</p>
            </div>
            <div className="flex gap-2">
              <span className="text-xs px-2 py-1 rounded-full bg-blue-50 text-blue-700 font-medium">● Actual</span>
              <span className="text-xs px-2 py-1 rounded-full bg-orange-50 text-orange-600 font-medium">-- Predicted</span>
            </div>
          </div>
          <div style={{ width: '100%', height: 380 }} className="mt-3">
            <ResponsiveContainer>
              <ComposedChart data={slabData.predicted_vs_actual || []} margin={{ top: 8, right: 16, left: 8, bottom: 4 }}>
                <defs>
                  <linearGradient id="actualGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#2563eb" stopOpacity={0.15} />
                    <stop offset="95%" stopColor="#2563eb" stopOpacity={0.01} />
                  </linearGradient>
                  <linearGradient id="predictedGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#f97316" stopOpacity={0.10} />
                    <stop offset="95%" stopColor="#f97316" stopOpacity={0.01} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
                <XAxis dataKey="period" tickFormatter={formatMonth} minTickGap={28} tick={{ fontSize: 11, fill: '#94a3b8' }} axisLine={false} tickLine={false} />
                <YAxis tickFormatter={(v) => `${(v / 1000).toFixed(0)}K`} tick={{ fontSize: 11, fill: '#94a3b8' }} axisLine={false} tickLine={false} width={48} />
                <Tooltip
                  labelFormatter={formatDate}
                  formatter={(value, name) => [`${Number(value).toLocaleString()}`, name]}
                  contentStyle={{ borderRadius: 10, border: '1px solid #e2e8f0', fontSize: 12, boxShadow: '0 4px 12px rgba(0,0,0,0.08)' }}
                />
                <Area type="monotone" dataKey="actual_quantity" stroke="none" fill="url(#actualGrad)" legendType="none" />
                <Area type="monotone" dataKey="predicted_quantity" stroke="none" fill="url(#predictedGrad)" legendType="none" />
                <Line type="monotone" dataKey="actual_quantity" name="Actual" stroke="#2563eb" strokeWidth={2.5} dot={{ r: 3, fill: '#2563eb', strokeWidth: 0 }} activeDot={{ r: 5, strokeWidth: 0 }} legendType="none" />
                <Line type="monotone" dataKey="predicted_quantity" name="Predicted" stroke="#f97316" strokeWidth={2} strokeDasharray="5 3" dot={{ r: 3, fill: '#f97316', strokeWidth: 0 }} activeDot={{ r: 5, strokeWidth: 0 }} legendType="none" />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* ── Right panel ── */}
        <div className="space-y-4">
          {/* Fit metrics */}
          <div className="bg-white rounded-xl border border-slate-200 p-4" style={{ boxShadow: '0 1px 6px rgba(0,0,0,0.05)' }}>
            <p className="text-xs font-semibold text-muted uppercase tracking-wide mb-3">Fit Quality</p>
            <div className="grid grid-cols-3 gap-2">
              <div className="rounded-lg p-3 text-center" style={{ background: r2Bg }}>
                <p className="text-xs text-muted mb-1">R²</p>
                <p className="text-lg font-bold" style={{ color: r2Color }}>{r2.toFixed(2)}</p>
              </div>
              <div className="rounded-lg p-3 text-center" style={{ background: adjR2 != null && adjR2 >= 0.65 ? '#f0fdf4' : '#f8fafc' }}>
                <p className="text-xs text-muted mb-1">Adj R²</p>
                <p className="text-lg font-bold text-body">{adjR2 == null ? '—' : adjR2.toFixed(2)}</p>
              </div>
              <div className="rounded-lg p-3 text-center" style={{ background: mapeBg }}>
                <p className="text-xs text-muted mb-1">MAPE</p>
                <p className="text-lg font-bold" style={{ color: mapeColor }}>{mape == null ? '—' : `${mape.toFixed(1)}%`}</p>
              </div>
            </div>
          </div>

          {/* Coefficients */}
          <div className="bg-white rounded-xl border border-slate-200 p-4" style={{ boxShadow: '0 1px 6px rgba(0,0,0,0.05)' }}>
            <p className="text-xs font-semibold text-muted uppercase tracking-wide mb-3">Stage 2 Coefficients</p>
            <CoefRow label="Intercept" value={slabData?.model_coefficients?.stage2_intercept} />
            <CoefRow label="Residual Store" value={slabData?.model_coefficients?.coef_residual_store} hint="outlet count signal" />
            <CoefRow label="Base Discount" value={slabData?.model_coefficients?.coef_structural_discount} hint="structural level effect" />
            <CoefRow label="Lag Discount" value={slabData?.model_coefficients?.coef_lag1_structural_discount} hint="prior month base" />
            <CoefRow label="Other Slabs" value={slabData?.model_coefficients?.coef_other_slabs_weighted_base_discount_pct} hint="cross-slab weighted base" />
          </div>

          {/* Regularisation */}
          <div className="bg-slate-50 rounded-xl border border-slate-200 p-3">
            <p className="text-xs font-semibold text-muted uppercase tracking-wide mb-2">Regularisation</p>
            <div className="flex items-center justify-between">
              <span className="text-sm text-body">L2 Penalty</span>
              <span className="text-sm font-bold text-body">{fmt(slabData?.model_coefficients?.l2_penalty)}</span>
            </div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {showControls && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <div className="flex items-center justify-between flex-wrap gap-3 mb-4">
            <div>
              <h3 className="text-xl font-bold text-body">Step 3: Modeling</h3>
              <p className="text-sm text-muted mt-1">
                Build slab-wise monthly models, compare predicted vs actual quantity, and evaluate structural/profit ROI.
              </p>
            </div>
            <button
              type="button"
              onClick={() => onRun({
                include_lag_discount: includeLagDiscount,
                cogs_per_unit: Number(cogsPerUnit || 0),
              })}
              disabled={isLoading}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-primary text-white text-sm font-semibold disabled:opacity-60"
            >
              {isLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
              {isLoading ? 'Running Modeling...' : 'Run Step 3 Modeling'}
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-accent-light rounded-md p-3">
              <p className="text-sm text-body font-semibold mb-2">Lag Discount In Model</p>
              <label className="inline-flex items-center gap-2 text-sm text-body">
                <input
                  type="checkbox"
                  checked={includeLagDiscount}
                  onChange={(e) => setIncludeLagDiscount(e.target.checked)}
                />
                Include lag discount term
              </label>
            </div>
            <div className="bg-accent-light rounded-md p-3">
              <p className="text-sm text-body font-semibold mb-2">COGS Per Unit</p>
              <input
                type="number"
                min="0"
                step="0.5"
                value={cogsPerUnit}
                onChange={(e) => setCogsPerUnit(parseFloat(e.target.value || '0'))}
                className="w-full px-3 py-2 text-sm border border-gray-300 rounded-lg bg-white"
              />
            </div>
            <div className="bg-accent-light rounded-md p-3">
              <p className="text-sm text-body font-semibold mb-2">ROI Display Filter</p>
              <select
                value={roiMode}
                onChange={(e) => setRoiMode(e.target.value)}
                className="w-full px-3 py-2 text-sm border border-gray-300 rounded-lg bg-white"
              >
                <option value="structural">Topline ROI</option>
                <option value="profit">Gross Margin ROI</option>
                <option value="both">Both</option>
              </select>
            </div>
          </div>
        </div>
      )}

      {isError && (
        <div className="bg-brand-dangerLight border border-danger rounded-lg p-4 flex items-start space-x-3">
          <AlertCircle className="text-danger flex-shrink-0 mt-0.5" size={20} />
          <div>
            <h4 className="font-semibold text-body">Modeling Error</h4>
            <p className="text-muted text-sm">{errorMessage || 'Failed to run modeling'}</p>
          </div>
        </div>
      )}

      {data?.success === false && (
        <div className="bg-brand-dangerLight border border-danger rounded-lg p-4">
          <p className="text-sm text-body font-semibold">Modeling could not run</p>
          <p className="text-sm text-muted mt-1">{data.message || 'No data available.'}</p>
        </div>
      )}

      {data?.success && (
        <>
          {(overallCombinedRoiSummary || sizeOptions.length > 0 || slabOnlyResults.length > 0) && (
            <>
              {overallCombinedRoiSummary && (
                <div className="bg-white rounded-lg shadow-md p-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    <div className="bg-accent-light rounded-md p-3">
                      <p className="text-muted">Subcategory Topline ROI</p>
                      <p className="font-bold text-body">{fmt(overallCombinedRoiSummary.structural_roi_1mo)}</p>
                    </div>
                    <div className="bg-accent-light rounded-md p-3">
                      <p className="text-muted">Subcategory Gross Margin ROI</p>
                      <p className="font-bold text-body">{fmt(overallCombinedRoiSummary.structural_profit_roi_1mo)}</p>
                    </div>
                  </div>
                </div>
              )}

              <div className="bg-white rounded-lg shadow-md p-4">
                <div className="grid grid-cols-1 xl:grid-cols-[240px_minmax(0,1fr)] gap-4 items-start">
                  <div>
                    <p className="text-sm font-semibold text-body mb-2">Pack Level</p>
                    {sizeOptions.length > 0 && (
                      <div className="flex flex-wrap gap-2">
                        {sizeOptions.map((sizeKey) => (
                          <button
                            key={sizeKey}
                            type="button"
                            onClick={() => setActiveSize(sizeKey)}
                            className={`px-3 py-1.5 text-sm rounded-md border ${
                              activeSize === sizeKey
                                ? 'bg-primary text-white border-primary'
                                : 'bg-white text-body border-gray-300'
                            }`}
                          >
                            {sizeKey}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                  {selectedSizeCombinedRoiSummary && (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                      <div className="bg-accent-light rounded-md p-3">
                        <p className="text-muted">{activeSize || slabSizeKey} Topline ROI</p>
                        <p className="font-bold text-body">{fmt(selectedSizeCombinedRoiSummary.structural_roi_1mo)}</p>
                      </div>
                      <div className="bg-accent-light rounded-md p-3">
                        <p className="text-muted">{activeSize || slabSizeKey} Gross Margin ROI</p>
                        <p className="font-bold text-body">{fmt(selectedSizeCombinedRoiSummary.structural_profit_roi_1mo)}</p>
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {slabOnlyResults.length > 0 && (
                <div className="bg-white rounded-lg shadow-md p-4">
                  <p className="text-sm font-semibold text-body mb-2">ROI Slab Selection</p>
                  <div className="flex flex-wrap gap-2">
                    {slabOnlyResults.map((slab) => (
                      <button
                        key={slab.slab}
                        type="button"
                        onClick={() => setActiveSlab(slab.slab)}
                        className={`px-3 py-1.5 text-sm rounded-md border ${
                          activeSlab === slab.slab
                            ? 'bg-primary text-white border-primary'
                            : 'bg-white text-body border-gray-300'
                        }`}
                      >
                        {slab.slab}
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </>
          )}

          <details className="bg-white rounded-lg shadow-md p-4">
            <summary className="cursor-pointer text-base font-semibold text-body">
              Summary by Slab
            </summary>
            <div className="mt-4">
              {step3SummaryRows.length > 0 ? (
                <div className="overflow-x-auto border border-gray-200 rounded-lg">
                  <table className="w-full text-sm">
                    <thead className="bg-gray-50">
                      <tr>
                        <th className="text-left px-3 py-2">Slab</th>
                        <th className="text-right px-3 py-2">Invoices</th>
                        <th className="text-right px-3 py-2">Invoice Contribution %</th>
                        <th className="text-right px-3 py-2">Quantity</th>
                        <th className="text-right px-3 py-2">AOQ</th>
                        <th className="text-right px-3 py-2">AOV</th>
                        <th className="text-right px-3 py-2">Sales Value</th>
                        <th className="text-right px-3 py-2">Sales Contribution %</th>
                        <th className="text-right px-3 py-2">Total Discount</th>
                        <th className="text-right px-3 py-2">Discount %</th>
                      </tr>
                    </thead>
                    <tbody>
                      {step3SummaryRows.map((row, idx) => (
                        <tr key={`${row?.Slab || 'slab'}-${idx}`} className="border-t border-gray-100">
                          <td className="px-3 py-2">{row?.Slab || '-'}</td>
                          <td className="px-3 py-2 text-right">{formatWhole(row?.Invoices)}</td>
                          <td className="px-3 py-2 text-right">{fmt(row?.['Invoice_Contribution_%'])}%</td>
                          <td className="px-3 py-2 text-right">{formatWhole(row?.Quantity)}</td>
                          <td className="px-3 py-2 text-right">{fmt(row?.AOQ)}</td>
                          <td className="px-3 py-2 text-right">{formatMetric(row?.AOV)}</td>
                          <td className="px-3 py-2 text-right">{formatMetric(row?.Sales_Value)}</td>
                          <td className="px-3 py-2 text-right">{fmt(row?.['Sales_Contribution_%'])}%</td>
                          <td className="px-3 py-2 text-right">{formatMetric(row?.Total_Discount)}</td>
                          <td className="px-3 py-2 text-right">{fmt(row?.Discount_Pct)}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <p className="text-sm text-muted">No slab summary data available for current filters.</p>
              )}
            </div>
          </details>

          {slabData && (
            <div className="bg-white rounded-lg shadow-md p-6">
              <div className="flex items-center justify-between mb-4">
                <h4 className="text-lg font-semibold text-body">Topline ROI View - {slabSizeKey} {slabData.slab}</h4>
                <button
                  type="button"
                  onClick={() => setIsModelModalOpen(true)}
                  className="px-3 py-1.5 text-sm rounded-md border border-primary text-body bg-white"
                >
                  View Model
                </button>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-sm mb-4">
                <div className="bg-accent-light rounded-md p-3">
                  <p className="text-muted">Topline ROI (Sum Num / Sum Den)</p>
                  <p className="font-bold text-body">{fmt(slabData?.summary?.structural_roi_1mo)}</p>
                </div>
                <div className="bg-accent-light rounded-md p-3">
                  <p className="text-muted">Gross Margin ROI (Sum Num / Sum Den)</p>
                  <p className="font-bold text-body">{fmt(slabData?.summary?.structural_profit_roi_1mo)}</p>
                </div>
                <div className="bg-accent-light rounded-md p-3">
                  <p className="text-muted">Instances of discount depth increase</p>
                  <p className="font-bold text-body">{Math.round(Number(slabData?.summary?.structural_episodes || 0))}</p>
                </div>
              </div>
              <div className="rounded-lg border border-slate-200 bg-gradient-to-br from-amber-50 via-white to-sky-50 p-3">
                <div style={{ width: '100%', height: 430 }}>
                  <ResponsiveContainer>
                    <ComposedChart
                      data={roiChartData}
                      margin={{ top: 24, right: 8, left: 0, bottom: 0 }}
                    >
                      <CartesianGrid strokeDasharray="4 4" stroke="#d8dde6" />
                      <XAxis dataKey="period" tickFormatter={formatMonth} minTickGap={24} />
                      <YAxis
                        yAxisId="left"
                        domain={[
                          (dataMin) => {
                            const n = Number(dataMin)
                            if (!Number.isFinite(n)) return 0
                            const padded = Math.min(0, n - Math.abs(n) * 0.1)
                            return Math.floor(padded * 10) / 10
                          },
                          (dataMax) => {
                            const n = Number(dataMax)
                            if (!Number.isFinite(n)) return 1
                            if (Math.abs(n) < 1e-9) return 1
                            const padded = n > 0 ? n * 1.2 : n * 0.8
                            return Math.ceil(padded * 10) / 10
                          },
                        ]}
                        tickFormatter={formatRoiAxisTick}
                      />
                      <YAxis yAxisId="right" orientation="right" />
                      <Tooltip labelFormatter={formatDate} formatter={(value) => [fmt(value), '']} />
                      <Legend />
                      {(roiMode === 'structural' || roiMode === 'both') && (
                        <Bar yAxisId="left" dataKey="roi_1mo" name="Topline ROI" fill="#3B82F6">
                          <LabelList
                            dataKey="roi_1mo"
                            position="top"
                            formatter={(value) => (value == null || Number.isNaN(Number(value)) ? '' : Number(value).toFixed(2))}
                            style={{ fill: '#1F2937', fontSize: 11, fontWeight: 600 }}
                          />
                        </Bar>
                      )}
                      {(roiMode === 'profit' || roiMode === 'both') && (
                        <Bar yAxisId="left" dataKey="profit_roi_1mo" name="Gross Margin ROI" fill="#F59E0B">
                          <LabelList
                            dataKey="profit_roi_1mo"
                            position="top"
                            formatter={(value) => (value == null || Number.isNaN(Number(value)) ? '' : Number(value).toFixed(2))}
                            style={{ fill: '#1F2937', fontSize: 11, fontWeight: 600 }}
                          />
                        </Bar>
                      )}
                      <Line
                        yAxisId="right"
                        type="stepAfter"
                        dataKey="base_discount_pct"
                        name="Base Discount %"
                        stroke="#0F766E"
                        strokeWidth={3}
                        dot={{ r: 2 }}
                      />
                    </ComposedChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          )}
        </>
      )}

      {isModelModalOpen && slabData && (
        <div className="fixed inset-0 z-50 bg-black/40 flex items-start justify-center p-4 overflow-y-auto">
          <div className="w-full max-w-6xl bg-white rounded-lg shadow-xl">
            <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200">
              <h3 className="text-lg font-semibold text-body">Model View - {slabSizeKey} {slabData.slab}</h3>
              <button
                type="button"
                onClick={() => setIsModelModalOpen(false)}
                className="p-1 rounded border border-gray-300 text-body"
              >
                <X size={16} />
              </button>
            </div>
            <div className="p-6 space-y-6">
              {renderModelContent()}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default ModelingROI
