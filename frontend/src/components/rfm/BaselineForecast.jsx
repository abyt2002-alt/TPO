import { useEffect, useState } from 'react'
import { AlertCircle, Loader2 } from 'lucide-react'
import {
  Area,
  CartesianGrid,
  ReferenceArea,
  Legend,
  Line,
  ComposedChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

const formatQty = (value) => Number(value || 0).toLocaleString(undefined, { maximumFractionDigits: 0 })
const formatAxisTick = (value) => {
  const n = Number(value || 0)
  if (!Number.isFinite(n)) return '0'
  const abs = Math.abs(n)
  if (abs >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (abs >= 1_000) return `${(n / 1_000).toFixed(0)}K`
  return String(Math.round(n))
}

const toMonthLabel = (period) => {
  if (!period) return ''
  const d = new Date(`${period}-01`)
  if (Number.isNaN(d.getTime())) return String(period)
  return d.toLocaleDateString(undefined, { month: 'short', year: '2-digit' })
}

const BaselineForecast = ({
  data,
  plannerData,
  isLoading,
  isError,
  errorMessage,
  actualVolumeByPeriod = {},
}) => {
  const [showIntroLoading, setShowIntroLoading] = useState(() => {
    if (typeof window === 'undefined') return true
    return window.sessionStorage.getItem('baseline-forecast-loading-seen') !== '1'
  })

  useEffect(() => {
    if (isLoading || !showIntroLoading) return
    if (typeof window !== 'undefined') {
      window.sessionStorage.setItem('baseline-forecast-loading-seen', '1')
    }
    setShowIntroLoading(false)
  }, [isLoading, showIntroLoading])

  if (isLoading) {
    if (!showIntroLoading) {
      return (
        <div className="bg-white rounded-lg shadow-md border border-slate-200 p-10 flex flex-col items-center justify-center gap-3">
          <Loader2 className="w-8 h-8 animate-spin text-primary" />
          <div className="text-center">
            <p className="text-base font-semibold text-body">Updating baseline forecast</p>
            <p className="text-sm text-muted">Projecting 12-ML, 18-ML, and total baseline.</p>
          </div>
        </div>
      )
    }
    return (
      <div className="planner-loading-shell p-6 space-y-5">
        <div className="flex items-start justify-between gap-4">
          <div>
            <div className="h-7 w-64 rounded-lg bg-slate-200" />
            <div className="h-4 w-80 rounded-lg bg-slate-100 mt-3" />
          </div>
          <div className="h-16 w-80 rounded-xl bg-sky-50 border border-sky-200" />
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
          <div className="planner-loading-kpi" />
          <div className="planner-loading-kpi" />
          <div className="planner-loading-kpi" />
        </div>

        <div className="planner-loading-canvas h-[180px]">
          <div className="planner-loading-line green" />
          <div className="planner-loading-line blue" />
          <div className="planner-loading-line amber" />
          <div className="planner-loading-dot up" />
          <div className="planner-loading-dot down" />
          <div className="planner-loading-dot neutral" />
        </div>

        <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
          <div className="h-6 w-72 rounded-lg bg-slate-200 mb-4" />
          <div className="rounded-lg border border-slate-200 bg-white/80 p-3">
            <div className="planner-loading-canvas h-[340px]">
              <div className="planner-loading-line green" />
              <div className="planner-loading-line blue" />
              <div className="planner-loading-line amber" />
            </div>
          </div>
        </div>
      </div>
    )
  }

  if (isError || data?.success === false) {
    return (
      <div className="bg-brand-dangerLight border border-danger rounded-lg p-4 flex items-start space-x-3">
        <AlertCircle className="text-danger flex-shrink-0 mt-0.5" size={20} />
        <div>
          <h4 className="font-semibold text-body">Baseline Forecast Error</h4>
          <p className="text-muted text-sm">{errorMessage || data?.message || 'Failed to run baseline forecast.'}</p>
        </div>
      </div>
    )
  }

  const points = Array.isArray(data?.points) ? data.points : []
  if (!points.length) {
    return (
      <div className="bg-white rounded-lg shadow-md p-8">
        <h3 className="text-xl font-semibold text-body mb-2">Baseline Forecast Requires Step 3 Output</h3>
        <p className="text-muted">Run Step 3 modeling first. Then the 12-ML, 18-ML, and total baseline forecast will appear here.</p>
      </div>
    )
  }

  const parsedPoints = points.map((row) => {
    const baseline12 = Number(row.baseline_12_ml || 0)
    const discount12 = Number(row.discount_component_12_ml || 0)
    const baseline18 = Number(row.baseline_18_ml || 0)
    const discount18 = Number(row.discount_component_18_ml || 0)
    return {
      ...row,
      baseline12,
      discount12,
      baseline18,
      discount18,
      total12: baseline12 + discount12,
      total18: baseline18 + discount18,
    }
  })

  const plannerMonthlyResults = Array.isArray(plannerData?.monthly_results) ? plannerData.monthly_results : []
  const plannerForecastPoints = plannerMonthlyResults.map((row) => {
    const size12 = row?.sizes?.['12-ML'] || {}
    const size18 = row?.sizes?.['18-ML'] || {}
    const baseline12 = Number(size12?.baseline_total_qty || 0)
    const baseline18 = Number(size18?.baseline_total_qty || 0)
    const total12 = Number(size12?.final_total_qty ?? size12?.pre_cross_total_qty ?? 0)
    const total18 = Number(size18?.final_total_qty ?? size18?.pre_cross_total_qty ?? 0)
    return {
      period: String(row?.period || ''),
      is_forecast: true,
      baseline12,
      discount12: total12 - baseline12,
      baseline18,
      discount18: total18 - baseline18,
      total12,
      total18,
    }
  }).filter((row) => row.period)

  const historicalPoints = parsedPoints
    .filter((row) => !row.is_forecast)
    .map((row) => {
      const periodKey = String(row.period || '')
      const actualPair = actualVolumeByPeriod?.[periodKey] || {}
      const actual12 = Number(actualPair?.['12-ML'])
      const actual18 = Number(actualPair?.['18-ML'])
      const total12 = Number.isFinite(actual12) && actual12 > 0 ? actual12 : Number(row.total12 || 0)
      const total18 = Number.isFinite(actual18) && actual18 > 0 ? actual18 : Number(row.total18 || 0)
      const baseline12 = Math.min(Math.max(Number(row.baseline12 || 0), 0), total12)
      const baseline18 = Math.min(Math.max(Number(row.baseline18 || 0), 0), total18)
      return {
        period: periodKey,
        is_forecast: false,
        baseline12,
        discount12: Math.max(total12 - baseline12, 0),
        baseline18,
        discount18: Math.max(total18 - baseline18, 0),
        total12,
        total18,
      }
    })

  const combinedPoints = [
    ...historicalPoints,
    ...(plannerForecastPoints.length > 0 ? plannerForecastPoints : parsedPoints.filter((row) => row.is_forecast)),
  ]

  const renderForecastDot = ({ cx, cy, payload }) => {
    if (!payload?.is_forecast) return null
    return <circle cx={cx} cy={cy} r={3} strokeWidth={1.5} stroke="currentColor" fill="#ffffff" />
  }

  const chartData = combinedPoints.map((row, index) => {
    const next = combinedPoints[index + 1]
    const isLastHistoryPoint = !row.is_forecast && next?.is_forecast
    const total12 = Number(row.total12 || 0)
    const total18 = Number(row.total18 || 0)

    return {
      ...row,
      month: toMonthLabel(row.period),
      baseline_12_ml: Number(row.baseline12 || 0),
      baseline_18_ml: Number(row.baseline18 || 0),
      discount_component_12_ml: Number(row.discount12 || 0),
      discount_component_18_ml: Number(row.discount18 || 0),
      total_baseline: Number(row.baseline12 || 0) + Number(row.baseline18 || 0),
      total_12_ml: total12,
      total_18_ml: total18,
      total_12_ml_history: row.is_forecast ? null : total12,
      total_12_ml_forecast_path: row.is_forecast || isLastHistoryPoint
        ? total12
        : null,
      total_18_ml_history: row.is_forecast ? null : total18,
      total_18_ml_forecast_path: row.is_forecast || isLastHistoryPoint
        ? total18
        : null,
    }
  })
  const firstForecastIndex = chartData.findIndex((row) => row.is_forecast)
  const forecastStartLabel = firstForecastIndex >= 0 ? chartData[firstForecastIndex]?.month : null
  const forecastEndLabel = firstForecastIndex >= 0 ? chartData[chartData.length - 1]?.month : null
  const latestHistory = [...chartData].reverse().find((row) => !row.is_forecast)
  const forecastOnly = chartData.filter((row) => row.is_forecast)
  const nextForecast = forecastOnly[0] || null

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-xl shadow-md border border-slate-200 p-6">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
          <div>
            <h3 className="text-2xl font-bold text-body">3-Month Baseline Forecast</h3>
            <p className="text-sm text-muted mt-1">
              Historical baseline stays separate from the next {data?.forecast_months || 3} month projection.
            </p>
          </div>
          <div className="rounded-lg border border-sky-200 bg-sky-50 px-4 py-3 text-sm text-sky-900 max-w-md">
            <p className="font-semibold">Method</p>
            <p className="mt-1">
              Historical months use backend baseline history. Forecast months use the current Step 4 planner scenario when available. Forecast zone starts after{' '}
              <span className="font-semibold">{latestHistory ? toMonthLabel(latestHistory.period) : '-'}</span>.
            </p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
        <div className="bg-white rounded-xl shadow-md border border-slate-200 p-4">
          <p className="text-xs font-semibold tracking-wide text-muted uppercase">Last Historical Baseline</p>
          <p className="text-sm text-muted mt-1">{latestHistory ? toMonthLabel(latestHistory.period) : '-'}</p>
          <div className="grid grid-cols-3 gap-3 mt-4">
            <div className="rounded-lg bg-slate-50 border border-slate-200 p-3">
              <p className="text-xs text-muted">12-ML</p>
              <p className="font-bold text-body text-xl mt-1">{formatQty(latestHistory?.baseline_12_ml)}</p>
            </div>
            <div className="rounded-lg bg-slate-50 border border-slate-200 p-3">
              <p className="text-xs text-muted">18-ML</p>
              <p className="font-bold text-body text-xl mt-1">{formatQty(latestHistory?.baseline_18_ml)}</p>
            </div>
            <div className="rounded-lg bg-slate-50 border border-slate-200 p-3">
              <p className="text-xs text-muted">Total</p>
              <p className="font-bold text-body text-xl mt-1">{formatQty(latestHistory?.total_baseline)}</p>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-xl shadow-md border border-primary/20 p-4">
          <p className="text-xs font-semibold tracking-wide text-primary uppercase">Next Month Projection</p>
          <p className="text-sm text-muted mt-1">{nextForecast ? toMonthLabel(nextForecast.period) : '-'}</p>
          <div className="grid grid-cols-3 gap-3 mt-4">
            <div className="rounded-lg bg-blue-50 border border-blue-200 p-3">
              <p className="text-xs text-muted">12-ML</p>
              <p className="font-bold text-body text-xl mt-1">{formatQty(nextForecast?.total12)}</p>
            </div>
            <div className="rounded-lg bg-rose-50 border border-rose-200 p-3">
              <p className="text-xs text-muted">18-ML</p>
              <p className="font-bold text-body text-xl mt-1">{formatQty(nextForecast?.total18)}</p>
            </div>
            <div className="rounded-lg bg-teal-50 border border-teal-200 p-3">
              <p className="text-xs text-muted">Total</p>
              <p className="font-bold text-body text-xl mt-1">{formatQty((Number(nextForecast?.total12 || 0) + Number(nextForecast?.total18 || 0)))}</p>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-xl shadow-md border border-amber-200 p-4">
          <p className="text-xs font-semibold tracking-wide text-amber-700 uppercase">Forecast Window</p>
          <p className="text-sm text-muted mt-1">
            {forecastStartLabel && forecastEndLabel ? `${forecastStartLabel} to ${forecastEndLabel}` : '-'}
          </p>
          <div className="grid grid-cols-3 gap-3 mt-4">
            <div className="rounded-lg bg-amber-50 border border-amber-200 p-3">
              <p className="text-xs text-muted">Months</p>
              <p className="font-bold text-body text-xl mt-1">{data?.forecast_months || 3}</p>
            </div>
            <div className="rounded-lg bg-amber-50 border border-amber-200 p-3">
              <p className="text-xs text-muted">Start</p>
              <p className="font-bold text-body text-lg mt-1">{forecastStartLabel || '-'}</p>
            </div>
            <div className="rounded-lg bg-amber-50 border border-amber-200 p-3">
              <p className="text-xs text-muted">End</p>
              <p className="font-bold text-body text-lg mt-1">{forecastEndLabel || '-'}</p>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-white rounded-xl shadow-md border border-slate-200 p-6">
        <div className="flex flex-col gap-1 mb-4">
          <h4 className="text-lg font-semibold text-body">Historical Baseline + Discount Component (Size-wise)</h4>
          <p className="text-sm text-muted">
            Each size chart shows baseline area with discount component stacked on top. Dashed top-line inside the shaded zone is forecast.
          </p>
        </div>
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
          <div className="rounded-lg border border-slate-200 bg-gradient-to-br from-blue-50 via-white to-slate-50 p-3">
            <p className="text-sm font-semibold text-body mb-2">12-ML</p>
            <div style={{ width: '100%', height: 360 }}>
              <ResponsiveContainer>
                <ComposedChart data={chartData}>
                  <CartesianGrid strokeDasharray="4 4" stroke="#d8dde6" />
                  <XAxis dataKey="month" minTickGap={20} />
                  <YAxis width={72} tickFormatter={formatAxisTick} />
                  <Tooltip formatter={(value, name) => [formatQty(value), name]} />
                  <Legend />
                  {forecastStartLabel && forecastEndLabel && (
                    <ReferenceArea
                      x1={forecastStartLabel}
                      x2={forecastEndLabel}
                      fill="#F8FAFC"
                      fillOpacity={0.85}
                      strokeOpacity={0}
                    />
                  )}
                  <Area
                    type="monotone"
                    dataKey="baseline_12_ml"
                    name="12-ML Baseline"
                    stackId="12ml-stack"
                    stroke="#1D4ED8"
                    fill="#93C5FD"
                    fillOpacity={0.55}
                    dot={false}
                  />
                  <Area
                    type="monotone"
                    dataKey="discount_component_12_ml"
                    name="12-ML Discount Component"
                    stackId="12ml-stack"
                    stroke="#0F766E"
                    fill="#5EEAD4"
                    fillOpacity={0.45}
                    dot={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="baseline_12_ml"
                    name="12-ML Baseline Line"
                    stroke="#1D4ED8"
                    strokeWidth={1.8}
                    dot={false}
                    connectNulls
                  />
                  <Line
                    type="monotone"
                    dataKey="total_12_ml_history"
                    name="12-ML Discount-Applied (History)"
                    stroke="#0F766E"
                    strokeWidth={2.5}
                    dot={false}
                    connectNulls
                  />
                  <Line
                    type="monotone"
                    dataKey="total_12_ml_forecast_path"
                    name="12-ML Discount-Applied (Forecast)"
                    stroke="#0F766E"
                    strokeWidth={2.5}
                    strokeDasharray="7 5"
                    dot={renderForecastDot}
                    connectNulls
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="rounded-lg border border-slate-200 bg-gradient-to-br from-rose-50 via-white to-slate-50 p-3">
            <p className="text-sm font-semibold text-body mb-2">18-ML</p>
            <div style={{ width: '100%', height: 360 }}>
              <ResponsiveContainer>
                <ComposedChart data={chartData}>
                  <CartesianGrid strokeDasharray="4 4" stroke="#d8dde6" />
                  <XAxis dataKey="month" minTickGap={20} />
                  <YAxis width={72} tickFormatter={formatAxisTick} />
                  <Tooltip formatter={(value, name) => [formatQty(value), name]} />
                  <Legend />
                  {forecastStartLabel && forecastEndLabel && (
                    <ReferenceArea
                      x1={forecastStartLabel}
                      x2={forecastEndLabel}
                      fill="#F8FAFC"
                      fillOpacity={0.85}
                      strokeOpacity={0}
                    />
                  )}
                  <Area
                    type="monotone"
                    dataKey="baseline_18_ml"
                    name="18-ML Baseline"
                    stackId="18ml-stack"
                    stroke="#DC2626"
                    fill="#FCA5A5"
                    fillOpacity={0.55}
                    dot={false}
                  />
                  <Area
                    type="monotone"
                    dataKey="discount_component_18_ml"
                    name="18-ML Discount Component"
                    stackId="18ml-stack"
                    stroke="#F59E0B"
                    fill="#FCD34D"
                    fillOpacity={0.45}
                    dot={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="baseline_18_ml"
                    name="18-ML Baseline Line"
                    stroke="#DC2626"
                    strokeWidth={1.8}
                    dot={false}
                    connectNulls
                  />
                  <Line
                    type="monotone"
                    dataKey="total_18_ml_history"
                    name="18-ML Discount-Applied (History)"
                    stroke="#B91C1C"
                    strokeWidth={2.5}
                    dot={false}
                    connectNulls
                  />
                  <Line
                    type="monotone"
                    dataKey="total_18_ml_forecast_path"
                    name="18-ML Discount-Applied (Forecast)"
                    stroke="#B91C1C"
                    strokeWidth={2.5}
                    strokeDasharray="7 5"
                    dot={renderForecastDot}
                    connectNulls
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default BaselineForecast
