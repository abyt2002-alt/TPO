import { useEffect, useMemo, useState } from 'react'
import { AlertCircle, Loader2, Upload } from 'lucide-react'
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

const fmtPct = (value) => {
  const n = Number(value)
  if (!Number.isFinite(n)) return 'NA'
  const sign = n > 0 ? '+' : ''
  return `${sign}${n.toFixed(2)}%`
}

const fmtX = (value) => {
  const n = Number(value)
  if (!Number.isFinite(n)) return 'NA'
  return `${n.toFixed(2)}x`
}

const toMonthLabel = (monthKey) => {
  if (!monthKey) return ''
  const d = new Date(`${monthKey}-01`)
  if (Number.isNaN(d.getTime())) return String(monthKey)
  return d.toLocaleDateString(undefined, { month: 'short' })
}

const ScenarioComparison = ({
  data,
  isLoading,
  isError,
  errorMessage,
}) => {
  const [selectedScenario, setSelectedScenario] = useState('')
  const [loadingTick, setLoadingTick] = useState(0)

  const loadingChartData = useMemo(() => {
    const labels = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar']
    const base = [17, 19, 17, 17, 19, 17, 17, 17, 17, 17, 17, 17]
    const t = loadingTick
    return labels.map((label, idx) => {
      const wave = Math.sin((idx + t) * 0.9) * 1.1 + Math.cos((idx * 0.6 + t) * 0.7) * 0.6
      const scenario = Math.max(8, Math.min(30, base[idx] + wave))
      return {
        month: label,
        current_promo_pct: base[idx],
        planned_promo_pct: Number(scenario.toFixed(2)),
      }
    })
  }, [loadingTick])

  const loadingDiffSummary = useMemo(() => {
    const byMonth = loadingChartData.map((row) => {
      const diff = Number((row.planned_promo_pct - row.current_promo_pct).toFixed(2))
      const trend = diff > 0.05 ? 'up' : diff < -0.05 ? 'down' : 'flat'
      return {
        month: row.month,
        diff,
        trend,
      }
    })
    const up = byMonth.filter((x) => x.trend === 'up').length
    const down = byMonth.filter((x) => x.trend === 'down').length
    const flat = byMonth.filter((x) => x.trend === 'flat').length
    return { byMonth, up, down, flat }
  }, [loadingChartData])

  useEffect(() => {
    if (!isLoading) return
    const id = window.setInterval(() => {
      setLoadingTick((v) => (v + 1) % 10000)
    }, 350)
    return () => window.clearInterval(id)
  }, [isLoading])

  const successfulRows = useMemo(
    () => (data?.scenarios || []).filter((row) => row?.success),
    [data]
  )

  const selectedScenarioName = useMemo(() => {
    if (!successfulRows.length) return ''
    if (selectedScenario && successfulRows.some((x) => x.scenario === selectedScenario)) return selectedScenario
    return successfulRows[0].scenario
  }, [successfulRows, selectedScenario])

  const selectedRow = useMemo(
    () => successfulRows.find((x) => x.scenario === selectedScenarioName) || null,
    [successfulRows, selectedScenarioName]
  )

  const chartData = useMemo(() => {
    const months = data?.months || []
    const current = data?.default_structural_discounts || []
    const planned = selectedRow?.planned_structural_discounts || []
    return months.map((month, i) => ({
      month: toMonthLabel(month),
      current_promo_pct: Number(current[i] || 0),
      planned_promo_pct: Number(planned[i] || 0),
    }))
  }, [data, selectedRow])

  const selectedCompareSummary = useMemo(() => {
    if (!selectedRow) return { up: 0, down: 0, same: 0 }
    const current = data?.default_structural_discounts || []
    const planned = selectedRow?.planned_structural_discounts || []
    let up = 0
    let down = 0
    let same = 0
    for (let i = 0; i < Math.min(current.length, planned.length); i += 1) {
      const c = Number(current[i] || 0)
      const p = Number(planned[i] || 0)
      if (p > c + 1e-9) up += 1
      else if (p < c - 1e-9) down += 1
      else same += 1
    }
    return { up, down, same }
  }, [data, selectedRow])

  const formatPlannedRoi = (row) => {
    const roi = Number(row?.roi_planned_x)
    const invest = Number(row?.investment_change_pct)
    const promo = Number(row?.promo_change_pct)
    if (Number.isFinite(roi) && Math.abs(roi) < 1e-9 && Number.isFinite(invest) && Number.isFinite(promo) && invest <= 0 && promo <= 0) {
      return 'NA'
    }
    return fmtX(roi)
  }

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-xl font-bold text-body">Step 5: Scenario Comparison</h3>
        <p className="text-sm text-muted mt-1">
          Upload a scenario file (CSV/XLSX) and compare all scenarios against the default plan.
        </p>
      </div>

      {isError && (
        <div className="bg-brand-dangerLight border border-danger rounded-lg p-4 flex items-start space-x-3">
          <AlertCircle className="text-danger flex-shrink-0 mt-0.5" size={20} />
          <div>
            <h4 className="font-semibold text-body">Scenario Comparison Error</h4>
            <p className="text-muted text-sm">{errorMessage || 'Failed to run scenario comparison'}</p>
          </div>
        </div>
      )}

      {isLoading && (
        <div className="bg-white rounded-lg shadow-md p-6 space-y-3">
          <div className="flex items-center justify-center gap-3 text-muted">
            <Loader2 className="w-5 h-5 animate-spin" />
            <span className="text-sm">Running scenarios. Simulating scenario paths...</span>
          </div>
          <div className="grid grid-cols-3 gap-2 text-xs">
            <div className="rounded-md border border-emerald-200 bg-emerald-50 px-2 py-1 text-emerald-700 font-semibold">
              + Higher: {loadingDiffSummary.up}
            </div>
            <div className="rounded-md border border-rose-200 bg-rose-50 px-2 py-1 text-rose-700 font-semibold">
              - Lower: {loadingDiffSummary.down}
            </div>
            <div className="rounded-md border border-slate-200 bg-slate-50 px-2 py-1 text-slate-600 font-semibold">
              = Flat: {loadingDiffSummary.flat}
            </div>
          </div>
          <div className="rounded-lg border border-slate-200 bg-gradient-to-br from-slate-50 via-white to-sky-50 p-3">
            <div style={{ width: '100%', height: 260 }}>
              <ResponsiveContainer>
                <LineChart data={loadingChartData}>
                  <CartesianGrid strokeDasharray="4 4" stroke="#d8dde6" />
                  <XAxis dataKey="month" minTickGap={20} />
                  <YAxis />
                  <Tooltip formatter={(value) => [Number(value || 0).toFixed(2), '']} />
                  <Legend />
                  <Line type="stepAfter" dataKey="current_promo_pct" name="Default Promo %" stroke="#6B7280" strokeWidth={2} dot={false} strokeDasharray="6 4" />
                  <Line type="stepAfter" dataKey="planned_promo_pct" name="Scenario Promo %" stroke="#1D4E89" strokeWidth={3} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
          <div className="overflow-x-auto pb-1">
            <div className="flex items-center gap-2 min-w-max">
              {loadingDiffSummary.byMonth.map((m) => (
                <div
                  key={`loading-trend-${m.month}`}
                  className={`rounded-full px-2.5 py-1 text-[11px] font-semibold ${
                    m.trend === 'up'
                      ? 'bg-emerald-100 text-emerald-700'
                      : m.trend === 'down'
                        ? 'bg-rose-100 text-rose-700'
                        : 'bg-slate-100 text-slate-600'
                  }`}
                >
                  {m.month} {m.trend === 'up' ? '+' : m.trend === 'down' ? '-' : '='}
                  {Math.abs(m.diff).toFixed(1)}pp
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {!isLoading && data?.success && (
        <div className="bg-white rounded-lg shadow-md p-4 space-y-4">
          <div className="text-sm text-muted">{data?.message}</div>
          <div className="overflow-x-auto border border-gray-200 rounded-lg">
            <table className="w-full text-sm">
              <thead className="bg-gray-50">
                <tr>
                  <th className="text-left px-3 py-2">Scenario</th>
                  <th className="text-left px-3 py-2">Volume Change</th>
                  <th className="text-left px-3 py-2">Revenue Change</th>
                  <th className="text-left px-3 py-2">Profit Change</th>
                  <th className="text-left px-3 py-2">Investment Change</th>
                  <th className="text-left px-3 py-2">Topline ROI @ Default</th>
                  <th className="text-left px-3 py-2">Topline ROI @ Planned</th>
                  <th className="text-left px-3 py-2">Gross Margin ROI @ Default</th>
                  <th className="text-left px-3 py-2">Gross Margin ROI @ Planned</th>
                </tr>
              </thead>
              <tbody>
                {(data?.scenarios || []).map((row, idx) => (
                  <tr
                    key={`${row.scenario}-${idx}`}
                    className={`border-t border-gray-100 ${row.success ? 'cursor-pointer hover:bg-slate-50' : ''} ${
                      row.success && row.scenario === selectedScenarioName ? 'bg-blue-50/40' : ''
                    }`}
                    onClick={() => {
                      if (row.success) setSelectedScenario(row.scenario)
                    }}
                  >
                    <td className="px-3 py-2 font-medium text-body">
                      {row.scenario}
                      {!row.success && (
                        <span className="block text-xs text-red-600 mt-0.5">{row.message || 'Failed'}</span>
                      )}
                    </td>
                    <td className="px-3 py-2">{fmtPct(row.volume_change_pct)}</td>
                    <td className="px-3 py-2">{fmtPct(row.revenue_change_pct)}</td>
                    <td className="px-3 py-2">{fmtPct(row.profit_change_pct)}</td>
                    <td className="px-3 py-2">{fmtPct(row.investment_change_pct)}</td>
                    <td className="px-3 py-2">{fmtX(row.roi_default_x)}</td>
                    <td className="px-3 py-2">{formatPlannedRoi(row)}</td>
                    <td className="px-3 py-2">{fmtX(row.gross_margin_roi_default_x)}</td>
                    <td className="px-3 py-2">{fmtX(row.gross_margin_roi_planned_x)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {!isLoading && data?.success && selectedRow && chartData.length > 0 && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <h4 className="text-lg font-semibold text-body mb-3">
            Scenario Deep Dive: {selectedScenarioName}
          </h4>
          <p className="text-xs text-muted mb-3">
            Vs default: Higher in {selectedCompareSummary.up} month(s), Lower in {selectedCompareSummary.down} month(s), Same in {selectedCompareSummary.same} month(s).
          </p>
          <div className="rounded-lg border border-slate-200 bg-gradient-to-br from-slate-50 via-white to-sky-50 p-3">
            <div style={{ width: '100%', height: 360 }}>
              <ResponsiveContainer>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="4 4" stroke="#d8dde6" />
                  <XAxis dataKey="month" minTickGap={20} />
                  <YAxis />
                  <Tooltip formatter={(value) => [Number(value || 0).toFixed(2), '']} />
                  <Legend />
                  <Line type="stepAfter" dataKey="current_promo_pct" name="Default Promo %" stroke="#6B7280" strokeWidth={2} dot={false} strokeDasharray="6 4" />
                  <Line type="stepAfter" dataKey="planned_promo_pct" name="Scenario Promo %" stroke="#1D4E89" strokeWidth={3} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}

      {!isLoading && !data?.success && !isError && (
        <div className="bg-white rounded-lg shadow-md p-12 text-center">
          <Upload className="mx-auto text-subtle mb-3" size={40} />
          <p className="text-sm text-muted">Upload a scenario file and click Run Comparison.</p>
        </div>
      )}
    </div>
  )
}

export default ScenarioComparison
