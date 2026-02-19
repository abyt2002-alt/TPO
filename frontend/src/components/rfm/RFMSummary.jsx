import { Users, TrendingUp, DollarSign, Calendar } from 'lucide-react'

const RFMSummary = ({ data }) => {
  const segments = data.segment_summary || []
  const totalOutlets = data.total_outlets || 0

  const recentCount = segments
    .filter((s) => s.segment?.startsWith('Recent-'))
    .reduce((sum, s) => sum + (s.total_outlets || 0), 0)
  const highFreqCount = segments
    .filter((s) => s.segment?.includes('-High-'))
    .reduce((sum, s) => sum + (s.total_outlets || 0), 0)
  const highMonetaryCount = segments
    .filter((s) => s.segment?.endsWith('-High'))
    .reduce((sum, s) => sum + (s.total_outlets || 0), 0)

  const recentPct = totalOutlets > 0 ? (recentCount / totalOutlets) * 100 : 0
  const highFreqPct = totalOutlets > 0 ? (highFreqCount / totalOutlets) * 100 : 0
  const highMonetaryPct = totalOutlets > 0 ? (highMonetaryCount / totalOutlets) * 100 : 0

  const weightedSegmentMetric = (key) => {
    const denom = segments.reduce((sum, s) => sum + (s.total_outlets || 0), 0)
    if (denom === 0) return 0
    const num = segments.reduce((sum, s) => sum + ((s[key] || 0) * (s.total_outlets || 0)), 0)
    return num / denom
  }

  const avgAOV = weightedSegmentMetric('avg_aov')
  const avgOrderDays = weightedSegmentMetric('avg_order_days')
  const avgOrdersPerDayAgg = (data.cluster_summary?.frequency || []).reduce(
    (acc, row) => {
      const outlets = row.Outlets || 0
      return {
        num: acc.num + ((row.Mean_Orders_Per_Day || 0) * outlets),
        den: acc.den + outlets
      }
    },
    { num: 0, den: 0 }
  )
  const avgOrdersPerDay = avgOrdersPerDayAgg.den > 0 ? avgOrdersPerDayAgg.num / avgOrdersPerDayAgg.den : 0

  const metrics = [
    {
      label: 'Total Outlets',
      value: data.total_outlets.toLocaleString(),
      icon: Users,
      color: 'text-primary',
      bgColor: 'bg-accent-light',
    },
    {
      label: 'Recent Outlets',
      value: `${recentPct.toFixed(1)}%`,
      icon: Calendar,
      color: 'text-secondary',
      bgColor: 'bg-accent-light',
    },
    {
      label: 'High Frequency',
      value: `${highFreqPct.toFixed(1)}%`,
      icon: TrendingUp,
      color: 'text-primary',
      bgColor: 'bg-accent-soft',
    },
    {
      label: 'High Monetary',
      value: `${highMonetaryPct.toFixed(1)}%`,
      icon: DollarSign,
      color: 'text-accent',
      bgColor: 'bg-accent-light',
    },
  ]

  const stats = [
    { label: 'Avg Order Days', value: avgOrderDays.toFixed(1) },
    { label: 'Avg Orders/Day', value: avgOrdersPerDay.toFixed(3) },
    { label: 'Avg AOV', value: avgAOV.toFixed(2) },
    { label: 'RFM Segments', value: '8' },
  ]

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h3 className="text-xl font-bold text-body mb-6">RFM Summary</h3>

      {/* Main Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        {metrics.map((metric) => {
          const Icon = metric.icon
          return (
            <div key={metric.label} className={`${metric.bgColor} rounded-lg p-4`}>
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium text-muted">{metric.label}</span>
                <Icon className={metric.color} size={24} />
              </div>
              <p className={`text-2xl font-bold ${metric.color}`}>{metric.value}</p>
            </div>
          )
        })}
      </div>

      {/* Additional Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 pt-4 border-t">
        {stats.map((stat) => (
          <div key={stat.label} className="text-center">
            <p className="text-sm text-muted mb-1">{stat.label}</p>
            <p className="text-xl font-bold text-body">{stat.value}</p>
          </div>
        ))}
      </div>

      {/* Data Info */}
      <div className="mt-4 pt-4 border-t">
        <div className="flex items-center justify-between text-sm text-muted">
          <span>Input Rows: {data.input_rows.toLocaleString()}</span>
          <span>Input Outlets: {data.input_outlets.toLocaleString()}</span>
          {data.max_date && (
            <span>Analysis Date: {new Date(data.max_date).toLocaleDateString()}</span>
          )}
        </div>
      </div>
    </div>
  )
}

export default RFMSummary
