import { useEffect, useMemo, useState } from 'react'
import { AlertCircle, Loader2, Play } from 'lucide-react'
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

const fmt = (value) => Number(value || 0).toFixed(2)

const formatPctChange = (value) => {
  const n = Number(value)
  if (!Number.isFinite(n)) return 'NA'
  if (Math.abs(n) < 1e-9) return 'No Change 0.0%'
  return n > 0 ? `Increase +${n.toFixed(1)}%` : `Decrease ${n.toFixed(1)}%`
}

const toMonthLabel = (monthKey) => {
  if (!monthKey) return ''
  const d = new Date(`${monthKey}-01`)
  if (Number.isNaN(d.getTime())) return String(monthKey)
  return d.toLocaleDateString(undefined, { month: 'short', year: 'numeric' })
}

const Planner12Month = ({
  data,
  isLoading,
  isError,
  errorMessage,
  availableSlabs = [],
  selectedSlab = '',
  onSelectSlab,
  onGenerate,
  onRecalculate,
  showControls = true,
  showSlabGenerateControls = true,
  showCogsInput = true,
  showMonthEditor = true,
  showResetButton = true,
  fixedCogsPerUnit = null,
}) => {
  const [plannedStruct, setPlannedStruct] = useState([])
  const [plannedBasePrice, setPlannedBasePrice] = useState([])
  const [cogsPerUnit, setCogsPerUnit] = useState(0)

  useEffect(() => {
    setPlannedStruct(data?.planned_structural_discounts || [])
    setPlannedBasePrice(data?.planned_base_prices || [])
    setCogsPerUnit(Number(data?.cogs_per_unit || 0))
  }, [data])

  useEffect(() => {
    const fixed = Number(fixedCogsPerUnit)
    if (Number.isFinite(fixed) && fixed >= 0) {
      setCogsPerUnit(fixed)
    }
  }, [fixedCogsPerUnit])

  const handleResetInputs = () => {
    setPlannedStruct(data?.planned_structural_discounts || [])
    setPlannedBasePrice(data?.planned_base_prices || [])
    const fixed = Number(fixedCogsPerUnit)
    if (Number.isFinite(fixed) && fixed >= 0) {
      setCogsPerUnit(fixed)
    } else {
      setCogsPerUnit(Number(data?.cogs_per_unit || 0))
    }
  }

  const months = data?.months || []
  const series = data?.series || []
  const chartData = useMemo(() => {
    return series.map((row, idx) => ({
      period: row.period,
      month: toMonthLabel(months[idx] || ''),
      current_promo_pct: row.current_promo_pct,
      planned_promo_pct: row.planned_promo_pct,
    }))
  }, [series, months])

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex items-center justify-between flex-wrap gap-3">
          <div>
            <h3 className="text-xl font-bold text-body">Step 4: 12-Month Planner</h3>
            <p className="text-sm text-muted mt-1">
              Plan month-wise structural promo for a selected slab and compare Current vs Planned outcomes.
            </p>
          </div>
        </div>
      </div>

      {isError && (
        <div className="bg-brand-dangerLight border border-danger rounded-lg p-4 flex items-start space-x-3">
          <AlertCircle className="text-danger flex-shrink-0 mt-0.5" size={20} />
          <div>
            <h4 className="font-semibold text-body">Planner Error</h4>
            <p className="text-muted text-sm">{errorMessage || 'Failed to run planner'}</p>
          </div>
        </div>
      )}

      {data?.success === false && (
        <div className="bg-brand-dangerLight border border-danger rounded-lg p-4">
          <p className="text-sm text-body font-semibold">Planner could not run</p>
          <p className="text-sm text-muted mt-1">{data.message || 'No planner data available.'}</p>
        </div>
      )}

      <div className="grid grid-cols-1 xl:grid-cols-12 gap-6">
        {showControls && (showSlabGenerateControls || showMonthEditor) && (
          <div className="xl:col-span-4 space-y-4">
            {showSlabGenerateControls && (
              <div className="bg-white rounded-lg shadow-md p-4">
                <label className="block text-sm font-medium text-gray-700 mb-1">Selected Slab</label>
                <select
                  value={selectedSlab}
                  onChange={(e) => onSelectSlab(e.target.value)}
                  className="w-full px-3 py-2 text-sm border border-gray-300 rounded-lg"
                >
                  <option value="">Select slab</option>
                  {availableSlabs.map((s) => (
                    <option key={s} value={s}>{s}</option>
                  ))}
                </select>
                <button
                  type="button"
                  onClick={onGenerate}
                  disabled={isLoading || !selectedSlab}
                  className="mt-3 inline-flex items-center gap-2 px-4 py-2 rounded-md bg-primary text-white text-sm font-semibold disabled:opacity-60"
                >
                  {isLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
                  {isLoading ? 'Loading Plan...' : 'Generate Step 4 Plan'}
                </button>
              </div>
            )}

            {showMonthEditor && data?.success && months.length > 0 && (
              <div className="bg-white rounded-lg shadow-md p-4">
                {showCogsInput && (
                  <>
                    <label className="block text-sm font-medium text-gray-700 mb-2">COGS Per Unit</label>
                    <input
                      type="number"
                      min="0"
                      step="0.5"
                      value={cogsPerUnit}
                      onChange={(e) => setCogsPerUnit(parseFloat(e.target.value || '0'))}
                      className="w-full px-3 py-2 text-sm border border-gray-300 rounded-lg mb-3"
                    />
                  </>
                )}

                <div className="max-h-[520px] overflow-auto border border-gray-200 rounded-lg">
                  <table className="w-full text-sm">
                    <thead className="bg-gray-50 sticky top-0">
                      <tr>
                        <th className="text-left px-3 py-2">Month</th>
                        <th className="text-left px-3 py-2">Current %</th>
                        <th className="text-left px-3 py-2">Planned %</th>
                        <th className="text-left px-3 py-2">Base Price</th>
                      </tr>
                    </thead>
                    <tbody>
                      {months.map((m, i) => (
                        <tr key={m} className="border-t border-gray-100">
                          <td className="px-3 py-2">{toMonthLabel(m)}</td>
                          <td className="px-3 py-2">{fmt((data?.current_structural_discounts || [])[i])}</td>
                          <td className="px-3 py-2">
                            <input
                              type="number"
                              min="0"
                              max="60"
                              step="0.5"
                              value={plannedStruct[i] ?? 0}
                              onChange={(e) => {
                                const next = [...plannedStruct]
                                next[i] = parseFloat(e.target.value || '0')
                                setPlannedStruct(next)
                              }}
                              className="w-24 px-2 py-1 border border-gray-300 rounded"
                            />
                          </td>
                          <td className="px-3 py-2">
                            <input
                              type="number"
                              min="0"
                              step="0.5"
                              value={plannedBasePrice[i] ?? 0}
                              onChange={(e) => {
                                const next = [...plannedBasePrice]
                                next[i] = parseFloat(e.target.value || '0')
                                setPlannedBasePrice(next)
                              }}
                              className="w-24 px-2 py-1 border border-gray-300 rounded"
                            />
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                <button
                  type="button"
                  onClick={() => onRecalculate({
                    planned_structural_discounts: plannedStruct,
                    planned_base_prices: plannedBasePrice,
                    cogs_per_unit: Number.isFinite(Number(fixedCogsPerUnit))
                      ? Number(fixedCogsPerUnit)
                      : cogsPerUnit,
                  })}
                  disabled={isLoading}
                  className="mt-3 w-full px-4 py-2 rounded-md bg-white border border-primary text-body text-sm font-semibold disabled:opacity-50"
                >
                  Recalculate Plan
                </button>
                {showResetButton && (
                  <button
                    type="button"
                    onClick={handleResetInputs}
                    disabled={isLoading}
                    className="mt-2 w-full px-4 py-2 rounded-md bg-white border border-gray-300 text-body text-sm font-semibold disabled:opacity-50"
                  >
                    Reset Inputs
                  </button>
                )}
              </div>
            )}
          </div>
        )}

        <div className={`${showControls && (showSlabGenerateControls || showMonthEditor) ? 'xl:col-span-8' : 'xl:col-span-12'} space-y-4`}>
          {data?.success && months.length > 0 && (
            <>
              <div className="bg-white rounded-lg shadow-md p-4">
                <div className="grid grid-cols-1 md:grid-cols-3 xl:grid-cols-6 gap-3 text-sm">
                  <div className="bg-accent-light rounded-md p-3">
                    <p className="text-muted">Volume Change</p>
                    <p className="font-bold text-body">{formatPctChange(data?.metrics?.volume_change_pct)}</p>
                  </div>
                  <div className="bg-accent-light rounded-md p-3">
                    <p className="text-muted">Revenue Change</p>
                    <p className="font-bold text-body">{formatPctChange(data?.metrics?.revenue_change_pct)}</p>
                  </div>
                  <div className="bg-accent-light rounded-md p-3">
                    <p className="text-muted">Profit Change</p>
                    <p className="font-bold text-body">{formatPctChange(data?.metrics?.profit_change_pct)}</p>
                  </div>
                  <div className="bg-accent-light rounded-md p-3">
                    <p className="text-muted">Promo Change</p>
                    <p className="font-bold text-body">{formatPctChange(data?.metrics?.promo_change_pct)}</p>
                  </div>
                  <div className="bg-accent-light rounded-md p-3">
                    <p className="text-muted">ROI Change</p>
                    <p className="font-bold text-body">{formatPctChange(data?.metrics?.roi_change_pct)}</p>
                  </div>
                  <div className="bg-accent-light rounded-md p-3">
                    <p className="text-muted">ROI (x)</p>
                    <p className="font-bold text-body">{fmt(data?.metrics?.roi_revenue_x)}x</p>
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-lg shadow-md p-6">
                <h4 className="text-lg font-semibold text-body mb-4">Promo Calendar Path</h4>
                <div className="rounded-lg border border-slate-200 bg-gradient-to-br from-slate-50 via-white to-sky-50 p-3">
                  <div style={{ width: '100%', height: 420 }}>
                    <ResponsiveContainer>
                      <LineChart data={chartData}>
                        <CartesianGrid strokeDasharray="4 4" stroke="#d8dde6" />
                        <XAxis dataKey="month" minTickGap={20} />
                        <YAxis />
                        <Tooltip formatter={(value) => [fmt(value), '']} />
                        <Legend />
                        <Line type="stepAfter" dataKey="current_promo_pct" name="Current Promo %" stroke="#6B7280" strokeWidth={2} dot={false} />
                        <Line type="stepAfter" dataKey="planned_promo_pct" name="Planned Promo %" stroke="#1D4E89" strokeWidth={3} dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>

            </>
          )}
        </div>
      </div>
    </div>
  )
}

export default Planner12Month
