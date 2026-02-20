import { useEffect, useMemo, useState } from 'react'
import { AlertCircle, Loader2, Maximize2, X } from 'lucide-react'
import {
  CartesianGrid,
  ComposedChart,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  Legend,
} from 'recharts'

const formatDate = (value) => {
  try {
    return new Date(value).toLocaleDateString()
  } catch {
    return String(value)
  }
}

const formatPct = (value) => Number(value || 0).toFixed(2)
const formatMetric = (value) => Number(value || 0).toLocaleString(undefined, { maximumFractionDigits: 2 })
const formatWhole = (value) => Number(value || 0).toLocaleString(undefined, { maximumFractionDigits: 0 })

const BaseDepthEstimator = ({
  config,
  onConfigChange,
  onRun,
  data,
  isLoading,
  isError,
  errorMessage,
  showControls = true,
}) => {
  const [isChartModalOpen, setIsChartModalOpen] = useState(false)
  const [activeTabId, setActiveTabId] = useState('all')
  const [showActualDiscount, setShowActualDiscount] = useState(true)

  const tabItems = useMemo(() => {
    const allTab = {
      id: 'all',
      label: 'All Slabs',
      data: {
        success: data?.success,
        message: data?.message,
        points: data?.points || [],
        summary: data?.summary || {},
      },
    }

    const slabTabs = (data?.slab_results || []).map((item, idx) => ({
      id: `slab-${idx}-${item?.slab || 'unknown'}`,
      label: String(item?.slab || `Slab ${idx + 1}`),
      data: {
        success: item?.success,
        message: item?.message,
        points: item?.points || [],
        summary: item?.summary || {},
      },
    }))

    return slabTabs.length > 0 ? [allTab, ...slabTabs] : [allTab]
  }, [data])

  useEffect(() => {
    if (!tabItems.find((t) => t.id === activeTabId)) {
      setActiveTabId(tabItems[0]?.id || 'all')
    }
  }, [tabItems, activeTabId])

  const activeTab = tabItems.find((t) => t.id === activeTabId) || tabItems[0]
  const points = activeTab?.data?.points || []
  const summaryBySlab = Array.isArray(data?.summary_by_slab) ? data.summary_by_slab : []
  const visibleSummaryRows = useMemo(() => {
    if (Array.isArray(summaryBySlab) && summaryBySlab.length > 0) return summaryBySlab
    const slabs = (data?.slab_results || []).filter((item) => item?.success && Array.isArray(item?.points) && item.points.length > 0)
    if (!slabs.length) return []

    const rows = slabs.map((item) => {
      const pts = item.points || []
      const invoices = pts.reduce((acc, p) => acc + Number(p?.orders || 0), 0)
      const quantity = pts.reduce((acc, p) => acc + Number(p?.quantity || 0), 0)
      const sales = pts.reduce((acc, p) => acc + Number(p?.sales_value || 0), 0)
      const totalDiscount = pts.reduce((acc, p) => acc + (Number(p?.actual_discount_pct || 0) * Number(p?.sales_value || 0) / 100), 0)
      return {
        Slab: String(item?.slab || ''),
        Outlets: null,
        Invoices: invoices,
        Quantity: quantity,
        AOQ: invoices > 0 ? quantity / invoices : 0,
        AOV: invoices > 0 ? sales / invoices : 0,
        Sales_Value: sales,
        Total_Discount: totalDiscount,
        Discount_Pct: sales > 0 ? (totalDiscount / sales) * 100 : 0,
      }
    })

    const totalInvoices = rows.reduce((acc, r) => acc + Number(r?.Invoices || 0), 0)
    const totalSales = rows.reduce((acc, r) => acc + Number(r?.Sales_Value || 0), 0)
    return rows.map((r) => ({
      ...r,
      'Invoice_Contribution_%': totalInvoices > 0 ? (Number(r.Invoices || 0) / totalInvoices) * 100 : 0,
      'Sales_Contribution_%': totalSales > 0 ? (Number(r.Sales_Value || 0) / totalSales) * 100 : 0,
    }))
  }, [summaryBySlab, data])
  const hasSlabTabs = (data?.slab_results || []).length > 0
  const showSlabOnlyCharts = hasSlabTabs && activeTabId !== 'all' && points.length > 0

  const renderChart = (height = 380) => (
    <div style={{ width: '100%', height }}>
      <ResponsiveContainer>
        <LineChart data={points} margin={{ top: 10, right: 20, left: 8, bottom: 0 }}>
          <CartesianGrid strokeDasharray="4 4" stroke="#cfd8e3" />
          <XAxis dataKey="period" tickFormatter={formatDate} minTickGap={28} tick={{ fill: '#4a5568', fontSize: 12 }} />
          <YAxis tick={{ fill: '#4a5568', fontSize: 12 }} />
          <Tooltip
            labelFormatter={formatDate}
            formatter={(value, name) => [formatPct(value), name]}
            contentStyle={{ borderRadius: 10, border: '1px solid #d6dee8', boxShadow: '0 8px 24px rgba(15, 23, 42, 0.12)' }}
          />
          <Legend wrapperStyle={{ paddingTop: 8 }} />
          {showActualDiscount && (
            <Line
              type="linear"
              dataKey="actual_discount_pct"
              name="Actual Discount %"
              stroke="#93C5FD"
              strokeWidth={1.5}
              dot={false}
              activeDot={{ r: 3, fill: '#60A5FA', stroke: '#60A5FA' }}
              connectNulls={false}
            />
          )}
          <Line
            type="monotone"
            dataKey="base_discount_pct"
            name="Estimated Base Discount %"
            stroke="#0F766E"
            strokeWidth={4}
            dot={false}
            activeDot={{ r: 5 }}
            connectNulls
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )

  const renderComposedChart = ({ dataKey, metricLabel, lineColor, height = 360 }) => (
    <div style={{ width: '100%', height }}>
      <ResponsiveContainer>
        <ComposedChart data={points} margin={{ top: 10, right: 20, left: 8, bottom: 0 }}>
          <CartesianGrid strokeDasharray="4 4" stroke="#d8e0ea" />
          <XAxis dataKey="period" tickFormatter={formatDate} minTickGap={28} tick={{ fill: '#4a5568', fontSize: 12 }} />
          <YAxis
            yAxisId="left"
            tick={{ fill: '#4a5568', fontSize: 12 }}
            tickFormatter={(value) => formatMetric(value)}
          />
          <YAxis
            yAxisId="right"
            orientation="right"
            tick={{ fill: '#4a5568', fontSize: 12 }}
            tickFormatter={(value) => `${formatPct(value)}%`}
          />
          <Tooltip
            labelFormatter={formatDate}
            formatter={(value, name) => {
              if (name === metricLabel) return [formatMetric(value), name]
              return [`${formatPct(value)}%`, name]
            }}
            contentStyle={{ borderRadius: 10, border: '1px solid #d6dee8', boxShadow: '0 8px 24px rgba(15, 23, 42, 0.12)' }}
          />
          <Legend wrapperStyle={{ paddingTop: 8 }} />
          <Line
            yAxisId="left"
            dataKey={dataKey}
            name={metricLabel}
            type="monotone"
            stroke={lineColor}
            strokeWidth={2.5}
            dot={false}
            activeDot={{ r: 4 }}
            connectNulls
          />
          {showActualDiscount && (
            <Line
              yAxisId="right"
              type="linear"
              dataKey="actual_discount_pct"
              name="Actual Discount %"
              stroke="#93C5FD"
              strokeWidth={1.5}
              dot={false}
              activeDot={{ r: 3, fill: '#60A5FA', stroke: '#60A5FA' }}
              connectNulls={false}
            />
          )}
          <Line
            yAxisId="right"
            type="monotone"
            dataKey="base_discount_pct"
            name="Estimated Base Discount %"
            stroke="#0F766E"
            strokeWidth={4}
            dot={false}
            activeDot={{ r: 4 }}
            connectNulls
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )

  return (
    <div className="space-y-6">
      {showControls && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-xl font-bold text-body mb-2">Step 2: Base Depth Estimator</h3>
          <p className="text-sm text-muted mb-4">
            Estimate base discount depth using monthly block logic from daily discount behavior.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Time Aggregation</label>
              <select
                value={config.time_aggregation}
                onChange={(e) => onConfigChange('time_aggregation', e.target.value)}
                className="w-full px-3 py-2 text-sm border border-gray-300 rounded-lg"
              >
                <option value="D">Daily</option>
                <option value="W">Weekly</option>
                <option value="M">Monthly</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Min Monthly Step-Up (pp)</label>
              <input
                type="number"
                min="0"
                max="5"
                step="0.1"
                value={config.min_upward_jump_pp}
                onChange={(e) => onConfigChange('min_upward_jump_pp', parseFloat(e.target.value || '0'))}
                className="w-full px-3 py-2 text-sm border border-gray-300 rounded-lg"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Min Monthly Step-Down (pp)</label>
              <input
                type="number"
                min="0"
                max="5"
                step="0.1"
                value={config.min_downward_drop_pp}
                onChange={(e) => onConfigChange('min_downward_drop_pp', parseFloat(e.target.value || '0'))}
                className="w-full px-3 py-2 text-sm border border-gray-300 rounded-lg"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Round Step</label>
              <input
                type="number"
                min="0.1"
                max="10"
                step="0.1"
                value={config.round_step}
                onChange={(e) => onConfigChange('round_step', parseFloat(e.target.value || '0.5'))}
                className="w-full px-3 py-2 text-sm border border-gray-300 rounded-lg"
              />
            </div>
          </div>

          <div className="mt-4">
            <button
              onClick={onRun}
              disabled={isLoading}
              className="px-5 py-2 rounded-lg bg-primary text-white font-semibold disabled:opacity-60"
            >
              {isLoading ? 'Estimating...' : 'Run Base Depth Estimator'}
            </button>
          </div>
        </div>
      )}

      {isError && (
        <div className="bg-brand-dangerLight border border-danger rounded-lg p-4 flex items-start space-x-3">
          <AlertCircle className="text-danger flex-shrink-0 mt-0.5" size={20} />
          <div>
            <h4 className="font-semibold text-body">Estimator Error</h4>
            <p className="text-muted text-sm">{errorMessage || 'Failed to estimate base depth'}</p>
          </div>
        </div>
      )}

      {data && data.success === false && (
        <div className="bg-brand-dangerLight border border-danger rounded-lg p-4">
          <p className="text-sm text-body font-semibold">Estimator could not run</p>
          <p className="text-sm text-muted mt-1">{data.message || 'No data available for selected filters.'}</p>
        </div>
      )}

      {isLoading && (
        <div className="bg-white rounded-lg shadow-md p-8 flex items-center justify-center">
          <Loader2 className="w-6 h-6 animate-spin text-primary" />
        </div>
      )}

      {data?.success && tabItems.length > 1 && (
        <div className="bg-white rounded-lg shadow-md p-4">
          <h4 className="text-base font-semibold text-body mb-3">Slab Tabs</h4>
          <div className="flex flex-wrap gap-2">
            {tabItems.map((tab) => (
              <button
                key={tab.id}
                type="button"
                onClick={() => setActiveTabId(tab.id)}
                className={`px-3 py-1.5 text-sm rounded-md border ${
                  activeTabId === tab.id
                    ? 'bg-primary text-white border-primary'
                    : 'bg-white text-body border-gray-300'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>
      )}

      {data?.success && (
        <>
          <div className="space-y-4">
            <details className="bg-white rounded-lg shadow-md p-4">
              <summary className="cursor-pointer text-base font-semibold text-body">
                Summary by Slab
              </summary>
              <div className="mt-4">
                {visibleSummaryRows.length > 0 ? (
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
                        {visibleSummaryRows.map((row, idx) => (
                          <tr key={`${row?.Slab || 'slab'}-${idx}`} className="border-t border-gray-100">
                            <td className="px-3 py-2">{row?.Slab || '-'}</td>
                            <td className="px-3 py-2 text-right">{formatWhole(row?.Invoices)}</td>
                            <td className="px-3 py-2 text-right">{formatPct(row?.['Invoice_Contribution_%'])}%</td>
                            <td className="px-3 py-2 text-right">{formatWhole(row?.Quantity)}</td>
                            <td className="px-3 py-2 text-right">{formatPct(row?.AOQ)}</td>
                            <td className="px-3 py-2 text-right">{formatMetric(row?.AOV)}</td>
                            <td className="px-3 py-2 text-right">{formatMetric(row?.Sales_Value)}</td>
                            <td className="px-3 py-2 text-right">{formatPct(row?.['Sales_Contribution_%'])}%</td>
                            <td className="px-3 py-2 text-right">{formatMetric(row?.Total_Discount)}</td>
                            <td className="px-3 py-2 text-right">{formatPct(row?.Discount_Pct)}%</td>
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

            {points.length > 0 ? (
              <div className="space-y-6">
                  <div className="bg-white rounded-lg shadow-md p-6">
                    <div className="flex items-center justify-between mb-4">
                      <h4 className="text-lg font-semibold text-body">
                        Actual vs Estimated Base Discount {activeTab?.label ? `- ${activeTab.label}` : ''}
                      </h4>
                      <div className="flex items-center gap-3">
                        <label className="inline-flex items-center gap-2 text-sm text-body">
                          <input
                            type="checkbox"
                            checked={showActualDiscount}
                            onChange={(e) => setShowActualDiscount(e.target.checked)}
                            className="h-4 w-4 rounded border-gray-300 text-primary focus:ring-primary"
                          />
                          Show Actual Discount
                        </label>
                        <button
                          type="button"
                          onClick={() => setIsChartModalOpen(true)}
                          className="inline-flex items-center gap-2 px-3 py-1.5 text-sm border border-gray-300 rounded-md bg-white hover:bg-gray-50"
                        >
                          <Maximize2 size={16} />
                          Open In Modal
                        </button>
                      </div>
                    </div>
                    <div className="rounded-lg border border-slate-200 bg-gradient-to-br from-sky-50 via-white to-teal-50 p-3">
                      {renderChart(380)}
                    </div>
                  </div>

                  {showSlabOnlyCharts && (
                    <>
                      <div className="bg-white rounded-lg shadow-md p-6">
                        <h4 className="text-lg font-semibold text-body mb-4">
                          Quantity vs Discount % - {activeTab?.label}
                        </h4>
                        <div className="rounded-lg border border-slate-200 bg-gradient-to-br from-indigo-50 via-white to-cyan-50 p-3">
                          {renderComposedChart({
                            dataKey: 'quantity',
                            metricLabel: 'Quantity',
                            lineColor: '#6366F1',
                            height: 360,
                          })}
                        </div>
                      </div>

                      <div className="bg-white rounded-lg shadow-md p-6">
                        <h4 className="text-lg font-semibold text-body mb-4">
                          Sales Value vs Discount % - {activeTab?.label}
                        </h4>
                        <div className="rounded-lg border border-slate-200 bg-gradient-to-br from-amber-50 via-white to-orange-50 p-3">
                          {renderComposedChart({
                            dataKey: 'sales_value',
                            metricLabel: 'Sales Value',
                            lineColor: '#F59E0B',
                            height: 360,
                          })}
                        </div>
                      </div>
                    </>
                  )}
              </div>
            ) : (
              <div className="bg-white rounded-lg shadow-md p-6">
                <p className="text-sm text-muted">
                  {activeTab?.data?.message || 'Estimation completed but no time-series points were produced for the selected filters.'}
                </p>
              </div>
            )}
          </div>
        </>
      )}

      {isChartModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          <div
            className="absolute inset-0 bg-black/50"
            onClick={() => setIsChartModalOpen(false)}
          />
          <div className="relative w-[94vw] max-w-6xl h-[85vh] bg-white rounded-lg shadow-2xl p-4">
            <div className="flex items-center justify-between mb-3">
              <h4 className="text-lg font-semibold text-body">
                Actual vs Estimated Base Discount {activeTab?.label ? `- ${activeTab.label}` : ''}
              </h4>
              <button
                type="button"
                onClick={() => setIsChartModalOpen(false)}
                className="p-2 rounded-md hover:bg-gray-100"
              >
                <X size={18} />
              </button>
            </div>
            <div className="rounded-lg border border-slate-200 bg-gradient-to-br from-sky-50 via-white to-teal-50 p-3 h-[88%]">
              {renderChart(640)}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default BaseDepthEstimator
