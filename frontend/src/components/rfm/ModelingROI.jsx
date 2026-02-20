import { useEffect, useMemo, useState } from 'react'
import { AlertCircle, Loader2, Play, X } from 'lucide-react'
import {
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

const ModelingROI = ({
  data,
  isLoading,
  isError,
  errorMessage,
  onRun,
  settings,
  showControls = true,
}) => {
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

  const validSlabs = useMemo(() => {
    const slabs = (data?.slab_results || []).filter((s) => s.valid)
    return slabs.sort((a, b) => {
      const ka = slabSortKey(a?.slab)
      const kb = slabSortKey(b?.slab)
      if (ka[0] !== kb[0]) return ka[0] - kb[0]
      if (ka[1] !== kb[1]) return ka[1] - kb[1]
      return ka[2].localeCompare(kb[2])
    })
  }, [data])

  useEffect(() => {
    if (!validSlabs.length) {
      setActiveSlab('')
      return
    }
    if (!validSlabs.find((s) => s.slab === activeSlab)) {
      setActiveSlab(validSlabs[0].slab)
    }
  }, [validSlabs, activeSlab])

  const slabData = validSlabs.find((s) => s.slab === activeSlab) || validSlabs[0]
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

    return rows.sort((a, b) => {
      const ai = slabSortKey(a?.Slab)
      const bi = slabSortKey(b?.Slab)
      if (ai[0] !== bi[0]) return ai[0] - bi[0]
      if (ai[1] !== bi[1]) return ai[1] - bi[1]
      return ai[2].localeCompare(bi[2])
    })
  }, [summaryBySlab])

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

  const renderModelContent = () => {
    if (!slabData) return null
    return (
      <>
        <div className="bg-white rounded-lg shadow-md p-6">
          <h4 className="text-lg font-semibold text-body mb-4">Slab {slabData.slab} - Monthly Model</h4>
          <div className="grid grid-cols-1 md:grid-cols-6 gap-3 text-sm mb-4">
            <div className="bg-accent-light rounded-md p-3">
              <p className="text-muted">Constrained Ridge R2</p>
              <p className="font-bold text-body">{fmt(slabData?.model_coefficients?.stage2_r2)}</p>
            </div>
            <div className="bg-accent-light rounded-md p-3">
              <p className="text-muted">OLS Reference R2</p>
              <p className="font-bold text-body">{fmt(slabData?.model_coefficients?.stage2_ols_r2)}</p>
            </div>
            <div className="bg-accent-light rounded-md p-3">
              <p className="text-muted">L2 Used</p>
              <p className="font-bold text-body">{fmt(slabData?.model_coefficients?.l2_penalty)}</p>
            </div>
            <div className="bg-accent-light rounded-md p-3">
              <p className="text-muted">CV R2</p>
              <p className="font-bold text-body">{fmt(slabData?.model_coefficients?.stage2_cv_r2)}</p>
            </div>
            <div className="bg-accent-light rounded-md p-3">
              <p className="text-muted">L2 Auto-Tune</p>
              <p className="font-bold text-body">
                {Number(slabData?.model_coefficients?.optimize_l2_penalty) > 0 ? 'Yes' : 'No'}
              </p>
            </div>
            <div className="bg-accent-light rounded-md p-3">
              <p className="text-muted">Lag Discount In Model</p>
              <p className="font-bold text-body">
                {Number(slabData?.model_coefficients?.include_lag_discount) > 0 ? 'Yes' : 'No'}
              </p>
            </div>
          </div>
          <p className="text-xs text-muted mb-3">OLS is shown for comparison only. Planner/ROI still uses constrained ridge.</p>
          <h5 className="text-base font-semibold text-body mb-3">Coefficient Comparison</h5>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
            <div className="rounded-md border border-gray-200 p-3">
              <p className="text-xs font-semibold text-muted uppercase mb-2">Constrained Ridge</p>
              <div className="space-y-2">
                <div className="bg-accent-light rounded-md p-3">
                  <p className="text-muted">Intercept</p>
                  <p className="font-bold text-body">{fmt(slabData?.model_coefficients?.stage2_intercept)}</p>
                </div>
                <div className="bg-accent-light rounded-md p-3">
                  <p className="text-muted">Residual (Store)</p>
                  <p className="font-bold text-body">{fmt(slabData?.model_coefficients?.coef_residual_store)}</p>
                </div>
                <div className="bg-accent-light rounded-md p-3">
                  <p className="text-muted">Structural Discount</p>
                  <p className="font-bold text-body">{fmt(slabData?.model_coefficients?.coef_structural_discount)}</p>
                </div>
                <div className="bg-accent-light rounded-md p-3">
                  <p className="text-muted">Tactical Discount</p>
                  <p className="font-bold text-body">{fmt(slabData?.model_coefficients?.coef_tactical_discount)}</p>
                </div>
                <div className="bg-accent-light rounded-md p-3">
                  <p className="text-muted">Lag1 Structural Discount</p>
                  <p className="font-bold text-body">{fmt(slabData?.model_coefficients?.coef_lag1_structural_discount)}</p>
                </div>
              </div>
            </div>
            <div className="rounded-md border border-gray-200 p-3">
              <p className="text-xs font-semibold text-muted uppercase mb-2">OLS Reference</p>
              <div className="space-y-2">
                <div className="bg-accent-light rounded-md p-3">
                  <p className="text-muted">Intercept</p>
                  <p className="font-bold text-body">{fmt(slabData?.model_coefficients?.stage2_ols_intercept)}</p>
                </div>
                <div className="bg-accent-light rounded-md p-3">
                  <p className="text-muted">Residual (Store)</p>
                  <p className="font-bold text-body">{fmt(slabData?.model_coefficients?.stage2_ols_coef_residual_store)}</p>
                </div>
                <div className="bg-accent-light rounded-md p-3">
                  <p className="text-muted">Structural Discount</p>
                  <p className="font-bold text-body">{fmt(slabData?.model_coefficients?.stage2_ols_coef_structural_discount)}</p>
                </div>
                <div className="bg-accent-light rounded-md p-3">
                  <p className="text-muted">Tactical Discount</p>
                  <p className="font-bold text-body">{fmt(slabData?.model_coefficients?.stage2_ols_coef_tactical_discount)}</p>
                </div>
                <div className="bg-accent-light rounded-md p-3">
                  <p className="text-muted">Lag1 Structural Discount</p>
                  <p className="font-bold text-body">{fmt(slabData?.model_coefficients?.stage2_ols_coef_lag1_structural_discount)}</p>
                </div>
              </div>
            </div>
          </div>
        </div>
        <div className="bg-white rounded-lg shadow-md p-6">
          <h4 className="text-lg font-semibold text-body mb-4">Actual vs Predicted Quantity - {slabData.slab}</h4>
          <div className="rounded-lg border border-slate-200 bg-gradient-to-br from-sky-50 via-white to-teal-50 p-3">
            <div style={{ width: '100%', height: 380 }}>
              <ResponsiveContainer>
                <LineChart data={slabData.predicted_vs_actual || []}>
                  <CartesianGrid strokeDasharray="4 4" stroke="#cfd8e3" />
                  <XAxis dataKey="period" tickFormatter={formatMonth} minTickGap={28} />
                  <YAxis />
                  <Tooltip labelFormatter={formatDate} formatter={(value) => [fmt(value), '']} />
                  <Legend />
                  <Line type="monotone" dataKey="actual_quantity" name="Actual Quantity" stroke="#111827" strokeWidth={3} dot={false} />
                  <Line type="monotone" dataKey="predicted_quantity" name="Predicted Quantity" stroke="#059669" strokeWidth={3} dot={false} />
                  <Line type="monotone" dataKey="predicted_quantity_ols" name="OLS Predicted (ref)" stroke="#F59E0B" strokeWidth={2} strokeDasharray="6 4" dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </>
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
          {validSlabs.length > 0 && (
            <div className="bg-white rounded-lg shadow-md p-4">
              <p className="text-sm font-semibold text-body mb-3">ROI Slab Selection</p>
              <div className="flex flex-wrap gap-2 mb-3">
                {validSlabs.map((slab) => (
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
                <h4 className="text-lg font-semibold text-body">Topline ROI View - {slabData.slab}</h4>
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
                  <p className="font-bold text-body">{fmt(slabData?.summary?.structural_roi_1mo)}x</p>
                </div>
                <div className="bg-accent-light rounded-md p-3">
                  <p className="text-muted">Gross Margin ROI (Sum Num / Sum Den)</p>
                  <p className="font-bold text-body">{fmt(slabData?.summary?.structural_profit_roi_1mo)}x</p>
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
              <h3 className="text-lg font-semibold text-body">Model View - {slabData.slab}</h3>
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
