import { useMemo, useState } from 'react'
import {
  AlertCircle,
  Loader2,
} from 'lucide-react'
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

const COLORS = ['#2563eb', '#16a34a', '#f59e0b', '#dc2626', '#7c3aed', '#0891b2']

const monthLabel = (period) => {
  const raw = String(period || '')
  const m = raw.match(/^(\d{4})-(\d{2})$/)
  if (!m) return raw
  const dt = new Date(Number(m[1]), Number(m[2]) - 1, 1)
  if (Number.isNaN(dt.getTime())) return raw
  return dt.toLocaleString(undefined, { month: 'short', year: '2-digit' })
}

const numFmt = (value) => {
  const n = Number(value || 0)
  if (!Number.isFinite(n)) return '0'
  return n.toLocaleString(undefined, { maximumFractionDigits: 0 })
}

const compactFmt = (value) => {
  const n = Number(value || 0)
  if (!Number.isFinite(n)) return '0'
  const abs = Math.abs(n)
  if (abs >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (abs >= 1_000) return `${(n / 1_000).toFixed(1)}K`
  return `${Math.round(n)}`
}

const pctFmt = (value) => {
  const n = Number(value || 0)
  if (!Number.isFinite(n)) return '0.00%'
  return `${n.toFixed(2)}%`
}

const slabSort = (a, b) => {
  const ai = Number(String(a || '').replace(/\D/g, ''))
  const bi = Number(String(b || '').replace(/\D/g, ''))
  if (Number.isFinite(ai) && Number.isFinite(bi) && ai !== bi) return ai - bi
  return String(a || '').localeCompare(String(b || ''))
}

const mapSizeChartData = (series = [], valueKey) => {
  const periodMap = {}
  const slabList = []
  series.forEach((row) => {
    const slab = String(row?.slab || '')
    if (!slab) return
    slabList.push(slab)
    ;(row?.points || []).forEach((pt) => {
      const period = String(pt?.period || '')
      if (!period) return
      if (!periodMap[period]) periodMap[period] = { period }
      periodMap[period][slab] = Number(pt?.[valueKey] || 0)
    })
  })
  const slabs = [...new Set(slabList)].sort(slabSort)
  const rows = Object.values(periodMap).sort((a, b) => String(a.period).localeCompare(String(b.period)))
  return { rows, slabs }
}

const buildPackCombinedData = (series = []) => {
  const periodSizeMap = {}
  series.forEach((row) => {
    const size = String(row?.size || '')
    if (!size) return
    ;(row?.points || []).forEach((pt) => {
      const period = String(pt?.period || '')
      if (!period) return
      if (!periodSizeMap[period]) periodSizeMap[period] = {}
      if (!periodSizeMap[period][size]) {
        periodSizeMap[period][size] = {
          sales: 0,
          volume: 0,
          discountWeightedNum: 0,
          discountWeightedDen: 0,
          mrpWeightedNum: 0,
        }
      }
      const sales = Number(pt?.revenue || 0)
      const volume = Number(pt?.volume || 0)
      const discountPct = Number(pt?.discount_pct || 0)
      const mrp = Number(pt?.mrp || 0)
      const cell = periodSizeMap[period][size]
      cell.sales += Number.isFinite(sales) ? sales : 0
      cell.volume += Number.isFinite(volume) ? volume : 0
      if (Number.isFinite(sales) && sales > 0 && Number.isFinite(discountPct)) {
        cell.discountWeightedNum += (sales * discountPct)
        cell.discountWeightedDen += sales
      }
      if (Number.isFinite(mrp) && Number.isFinite(volume) && volume > 0) {
        cell.mrpWeightedNum += (mrp * volume)
      }
    })
  })

  return Object.keys(periodSizeMap)
    .sort((a, b) => String(a).localeCompare(String(b)))
    .map((period) => {
      const p = periodSizeMap[period] || {}
      const s12 = p['12-ML'] || {}
      const s18 = p['18-ML'] || {}
      const d12 = Number(s12.discountWeightedDen || 0)
      const d18 = Number(s18.discountWeightedDen || 0)
      const v12 = Number(s12.volume || 0)
      const v18 = Number(s18.volume || 0)
      const mrpNum12 = Number(s12.mrpWeightedNum || 0)
      const mrpNum18 = Number(s18.mrpWeightedNum || 0)
      return {
        period,
        sales_12: Number(s12.sales || 0),
        sales_18: Number(s18.sales || 0),
        volume_12: v12,
        volume_18: v18,
        discount_12: d12 > 0 ? Number(s12.discountWeightedNum || 0) / d12 : 0,
        discount_18: d18 > 0 ? Number(s18.discountWeightedNum || 0) / d18 : 0,
        mrp_12: v12 > 0 ? mrpNum12 / v12 : 0,
        mrp_18: v18 > 0 ? mrpNum18 / v18 : 0,
      }
    })
}

const SlabLegend = ({ slabs, hiddenSlabs, onToggle }) => (
  <div className="flex flex-wrap gap-2 mb-2">
    {slabs.map((slab, idx) => {
      const hidden = hiddenSlabs.has(slab)
      return (
        <button
          key={`legend-${slab}`}
          type="button"
          onClick={() => onToggle(slab)}
          className={`px-2 py-1 rounded-md border text-xs font-semibold ${
            hidden ? 'bg-white text-muted border-gray-300 opacity-70' : 'bg-white text-body border-gray-300'
          }`}
          title={hidden ? `Show ${slab}` : `Hide ${slab}`}
        >
          <span className="inline-block w-2 h-2 rounded-full mr-1.5 align-middle" style={{ backgroundColor: COLORS[idx % COLORS.length] }} />
          {slab}
        </button>
      )
    })}
  </div>
)

const SeriesToggleLegend = ({ items, hiddenKeys, onToggle }) => (
  <div className="flex flex-wrap gap-2 mb-2">
    {items.map((item) => {
      const hidden = hiddenKeys.has(item.key)
      return (
        <button
          key={`legend-series-${item.key}`}
          type="button"
          onClick={() => onToggle(item.key)}
          className={`px-2 py-1 rounded-md border text-xs font-semibold ${
            hidden ? 'bg-white text-muted border-gray-300 opacity-70' : 'bg-white text-body border-gray-300'
          }`}
          title={hidden ? `Show ${item.label}` : `Hide ${item.label}`}
        >
          <span className="inline-block w-2 h-2 rounded-full mr-1.5 align-middle" style={{ backgroundColor: item.color }} />
          {item.label}
        </button>
      )
    })}
  </div>
)

const SizePanel = ({ title, series }) => {
  const discountData = mapSizeChartData(series, 'discount_pct')
  const volumeChangeData = mapSizeChartData(series, 'volume_change_pct')
  const [hiddenSlabs, setHiddenSlabs] = useState(new Set())

  const handleToggleSlab = (slab) => {
    setHiddenSlabs((prev) => {
      const next = new Set(prev)
      if (next.has(slab)) next.delete(slab)
      else next.add(slab)
      return next
    })
  }

  return (
    <div className="bg-white rounded-lg shadow-md p-4 space-y-4">
      <h4 className="text-lg font-semibold text-body">{title}</h4>

      <div className="bg-slate-50 rounded-md p-3">
        <h5 className="text-sm font-semibold text-body mb-2">Slab Discount Level (%)</h5>
        <p className="text-xs text-muted mb-2">Click legend slab to hide/show chart lines.</p>
        <SlabLegend slabs={discountData.slabs} hiddenSlabs={hiddenSlabs} onToggle={handleToggleSlab} />
        <div className="h-72">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={discountData.rows} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="period" tickFormatter={monthLabel} />
              <YAxis tickFormatter={(v) => `${Number(v || 0).toFixed(0)}%`} />
              <Tooltip
                formatter={(value) => pctFmt(value)}
                labelFormatter={(label) => monthLabel(label)}
              />
              {discountData.slabs.map((slab, idx) => (
                <Line
                  key={`discount-${slab}`}
                  type="monotone"
                  dataKey={slab}
                  name={slab}
                  stroke={COLORS[idx % COLORS.length]}
                  strokeWidth={2}
                  dot={{ r: 2 }}
                  connectNulls
                  hide={hiddenSlabs.has(slab)}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="bg-slate-50 rounded-md p-3">
        <h5 className="text-sm font-semibold text-body mb-2">Slab Volume Change (%)</h5>
        <div className="h-72">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={volumeChangeData.rows} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="period" tickFormatter={monthLabel} />
              <YAxis tickFormatter={(v) => `${Number(v || 0).toFixed(0)}%`} />
              <Tooltip
                formatter={(value) => pctFmt(value)}
                labelFormatter={(label) => monthLabel(label)}
              />
              {volumeChangeData.slabs.map((slab, idx) => (
                <Line
                  key={`volume-change-${slab}`}
                  type="monotone"
                  dataKey={slab}
                  name={slab}
                  stroke={COLORS[idx % COLORS.length]}
                  strokeWidth={2}
                  dot={{ r: 2 }}
                  connectNulls
                  hide={hiddenSlabs.has(slab)}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  )
}

const SlabTrendEDA = ({ data, isLoading, isError, errorMessage }) => {
  const series = Array.isArray(data?.series) ? data.series : []
  const series12 = series.filter((row) => String(row?.size || '') === '12-ML')
  const series18 = series.filter((row) => String(row?.size || '') === '18-ML')
  const packCombinedRows = useMemo(() => buildPackCombinedData(series), [series])
  const periodCount = Array.isArray(data?.periods) ? data.periods.length : 0
  const [hiddenPackSales, setHiddenPackSales] = useState(new Set())
  const [hiddenPackDiscount, setHiddenPackDiscount] = useState(new Set())
  const [hiddenPackMrpVol, setHiddenPackMrpVol] = useState(new Set())
  const [hiddenPackVolumeOnly, setHiddenPackVolumeOnly] = useState(new Set())

  const toggleSetKey = (setter, key) => {
    setter((prev) => {
      const next = new Set(prev)
      if (next.has(key)) next.delete(key)
      else next.add(key)
      return next
    })
  }

  return (
    <div className="space-y-4">
      <div className="bg-white rounded-lg shadow-md p-4">
        <h3 className="text-xl font-bold text-body">Slab Discount & Volume EDA</h3>
        <p className="text-sm text-muted mt-1">
          Month-wise slab discount levels and volume change trends from actual data.
        </p>
      </div>

      {isLoading && (
        <div className="bg-white rounded-lg shadow-md p-6 flex items-center gap-2">
          <Loader2 className="animate-spin text-primary" size={18} />
          <p className="text-sm text-muted">Loading slab trend EDA...</p>
        </div>
      )}

      {isError && (
        <div className="bg-brand-dangerLight border border-danger rounded-lg p-4 flex items-start gap-2">
          <AlertCircle className="text-danger mt-0.5" size={18} />
          <div>
            <h4 className="font-semibold text-body">EDA Error</h4>
            <p className="text-sm text-muted">{errorMessage || 'Failed to load slab trends.'}</p>
          </div>
        </div>
      )}

      {!isLoading && !isError && data?.success === false && (
        <div className="bg-brand-dangerLight border border-danger rounded-lg p-4">
          <p className="text-sm font-semibold text-body">EDA could not run</p>
          <p className="text-sm text-muted mt-1">{data?.message || 'No data available.'}</p>
        </div>
      )}

      {!isLoading && !isError && data?.success && (
        <>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            <div className="bg-white rounded-lg shadow-md p-3">
              <p className="text-xs text-muted uppercase">Months</p>
              <p className="text-2xl font-bold text-body">{numFmt(periodCount)}</p>
            </div>
            <div className="bg-white rounded-lg shadow-md p-3">
              <p className="text-xs text-muted uppercase">12-ML Slabs</p>
              <p className="text-2xl font-bold text-body">{numFmt(series12.length)}</p>
            </div>
            <div className="bg-white rounded-lg shadow-md p-3">
              <p className="text-xs text-muted uppercase">18-ML Slabs</p>
              <p className="text-2xl font-bold text-body">{numFmt(series18.length)}</p>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-md p-4 space-y-4">
            <h4 className="text-lg font-semibold text-body">Pack Size Combined Trends</h4>
            <p className="text-xs text-muted">12-ML and 18-ML shown together for quick comparison.</p>

            <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
              <div className="bg-slate-50 rounded-md p-3">
                <h5 className="text-sm font-semibold text-body mb-2">Sales by Pack Size</h5>
                <SeriesToggleLegend
                  items={[
                    { key: 'sales_12', label: '12-ML', color: '#2563eb' },
                    { key: 'sales_18', label: '18-ML', color: '#f97316' },
                  ]}
                  hiddenKeys={hiddenPackSales}
                  onToggle={(key) => toggleSetKey(setHiddenPackSales, key)}
                />
                <div className="h-72">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={packCombinedRows} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="period" tickFormatter={monthLabel} />
                      <YAxis width={64} tickFormatter={compactFmt} />
                      <Tooltip
                        formatter={(value) => numFmt(value)}
                        labelFormatter={(label) => monthLabel(label)}
                      />
                      <Line type="monotone" dataKey="sales_12" name="12-ML" stroke="#2563eb" strokeWidth={2} dot={{ r: 2 }} connectNulls hide={hiddenPackSales.has('sales_12')} />
                      <Line type="monotone" dataKey="sales_18" name="18-ML" stroke="#f97316" strokeWidth={2} dot={{ r: 2 }} connectNulls hide={hiddenPackSales.has('sales_18')} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="bg-slate-50 rounded-md p-3">
                <h5 className="text-sm font-semibold text-body mb-2">Discount Level % by Pack Size</h5>
                <SeriesToggleLegend
                  items={[
                    { key: 'discount_12', label: '12-ML', color: '#2563eb' },
                    { key: 'discount_18', label: '18-ML', color: '#f97316' },
                  ]}
                  hiddenKeys={hiddenPackDiscount}
                  onToggle={(key) => toggleSetKey(setHiddenPackDiscount, key)}
                />
                <div className="h-72">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={packCombinedRows} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="period" tickFormatter={monthLabel} />
                      <YAxis tickFormatter={(v) => `${Number(v || 0).toFixed(0)}%`} />
                      <Tooltip
                        formatter={(value) => pctFmt(value)}
                        labelFormatter={(label) => monthLabel(label)}
                      />
                      <Line type="monotone" dataKey="discount_12" name="12-ML" stroke="#2563eb" strokeWidth={2} dot={{ r: 2 }} connectNulls hide={hiddenPackDiscount.has('discount_12')} />
                      <Line type="monotone" dataKey="discount_18" name="18-ML" stroke="#f97316" strokeWidth={2} dot={{ r: 2 }} connectNulls hide={hiddenPackDiscount.has('discount_18')} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>

            <div className="bg-slate-50 rounded-md p-3">
              <h5 className="text-sm font-semibold text-body mb-2">MRP and Volume by Pack Size</h5>
              <SeriesToggleLegend
                items={[
                  { key: 'mrp_12', label: '12-ML MRP', color: '#1d4ed8' },
                  { key: 'mrp_18', label: '18-ML MRP', color: '#ea580c' },
                  { key: 'volume_12', label: '12-ML Volume', color: '#2563eb' },
                  { key: 'volume_18', label: '18-ML Volume', color: '#f97316' },
                ]}
                hiddenKeys={hiddenPackMrpVol}
                onToggle={(key) => toggleSetKey(setHiddenPackMrpVol, key)}
              />
              <div className="h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={packCombinedRows} margin={{ top: 8, right: 20, left: 0, bottom: 8 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="period" tickFormatter={monthLabel} />
                    <YAxis yAxisId="mrp" width={56} tickFormatter={(v) => Number(v || 0).toFixed(1)} />
                    <YAxis yAxisId="vol" orientation="right" width={64} tickFormatter={compactFmt} />
                    <Tooltip
                      labelFormatter={(label) => monthLabel(label)}
                      formatter={(value, name) => {
                        if (String(name).includes('MRP')) return Number(value || 0).toFixed(2)
                        return numFmt(value)
                      }}
                    />
                    <Line yAxisId="mrp" type="monotone" dataKey="mrp_12" name="12-ML MRP" stroke="#1d4ed8" strokeWidth={2} dot={{ r: 2 }} connectNulls hide={hiddenPackMrpVol.has('mrp_12')} />
                    <Line yAxisId="mrp" type="monotone" dataKey="mrp_18" name="18-ML MRP" stroke="#ea580c" strokeWidth={2} dot={{ r: 2 }} connectNulls hide={hiddenPackMrpVol.has('mrp_18')} />
                    <Line yAxisId="vol" type="monotone" dataKey="volume_12" name="12-ML Volume" stroke="#2563eb" strokeDasharray="5 3" strokeWidth={2} dot={{ r: 2 }} connectNulls hide={hiddenPackMrpVol.has('volume_12')} />
                    <Line yAxisId="vol" type="monotone" dataKey="volume_18" name="18-ML Volume" stroke="#f97316" strokeDasharray="5 3" strokeWidth={2} dot={{ r: 2 }} connectNulls hide={hiddenPackMrpVol.has('volume_18')} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="bg-slate-50 rounded-md p-3">
              <h5 className="text-sm font-semibold text-body mb-2">Volume by Pack Size</h5>
              <SeriesToggleLegend
                items={[
                  { key: 'volume_12_only', label: '12-ML Volume', color: '#2563eb' },
                  { key: 'volume_18_only', label: '18-ML Volume', color: '#f97316' },
                ]}
                hiddenKeys={hiddenPackVolumeOnly}
                onToggle={(key) => toggleSetKey(setHiddenPackVolumeOnly, key)}
              />
              <div className="h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={packCombinedRows} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="period" tickFormatter={monthLabel} />
                    <YAxis width={64} tickFormatter={compactFmt} />
                    <Tooltip
                      formatter={(value) => numFmt(value)}
                      labelFormatter={(label) => monthLabel(label)}
                    />
                    <Line
                      type="monotone"
                      dataKey="volume_12"
                      name="12-ML Volume"
                      stroke="#2563eb"
                      strokeWidth={2}
                      dot={{ r: 2 }}
                      connectNulls
                      hide={hiddenPackVolumeOnly.has('volume_12_only')}
                    />
                    <Line
                      type="monotone"
                      dataKey="volume_18"
                      name="18-ML Volume"
                      stroke="#f97316"
                      strokeWidth={2}
                      dot={{ r: 2 }}
                      connectNulls
                      hide={hiddenPackVolumeOnly.has('volume_18_only')}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
            <SizePanel title="12-ML" series={series12} />
            <SizePanel title="18-ML" series={series18} />
          </div>
        </>
      )}
    </div>
  )
}

export default SlabTrendEDA
