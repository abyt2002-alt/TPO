const SegmentGrid = ({ segments }) => {
  const getSegmentColor = (segment) => {
    if (segment.includes('Recent-High-High')) return { bg: 'bg-brand-yellowPale', border: 'border-secondary', text: 'text-secondary', badge: 'Best' }
    if (segment.includes('Stale-Low-Low')) return { bg: 'bg-brand-dangerLight', border: 'border-danger', text: 'text-danger', badge: 'Risk' }
    if (segment.includes('Recent')) return { bg: 'bg-accent-light', border: 'border-primary', text: 'text-primary', badge: 'Recent' }
    return { bg: 'bg-brand-yellowSoft', border: 'border-accent', text: 'text-body', badge: 'Stale' }
  }

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h3 className="text-xl font-bold text-body mb-6">RFM Segment Grid View</h3>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {segments.map((seg) => {
          const colors = getSegmentColor(seg.segment)
          
          return (
            <div
              key={seg.segment}
              className={`${colors.bg} border-2 ${colors.border} rounded-lg p-4 transition-all hover:shadow-lg`}
            >
              <div className="text-center mb-3">
                <span className={`inline-flex items-center px-2.5 py-1 rounded-full text-xs font-semibold ${colors.bg} ${colors.text}`}>
                  {colors.badge}
                </span>
              </div>
              
              {/* Segment Name */}
              <h4 className={`text-center font-bold text-sm mb-3 ${colors.text}`}>
                {seg.segment}
              </h4>
              
              {/* Main Metrics */}
              <div className="text-center mb-3">
                <p className={`text-4xl font-bold ${colors.text}`}>
                  {seg.total_outlets.toLocaleString()}
                </p>
                <p className={`text-lg ${colors.text}`}>{seg.percentage.toFixed(1)}%</p>
                <p className="text-xs text-muted font-semibold mt-1">
                  Market Share: {seg.market_share.toFixed(1)}%
                </p>
              </div>
              
              {/* Stats */}
              <div className={`${colors.bg} bg-opacity-50 rounded p-2 space-y-1 mb-3`}>
                <div className="text-xs text-muted">
                  <span className="font-semibold">Avg Order Days:</span> {seg.avg_order_days.toFixed(1)}
                </div>
                <div className="text-xs text-muted">
                  <span className="font-semibold">Avg AOV:</span> {seg.avg_aov.toFixed(0)}
                </div>
                <div className="text-xs text-muted">
                  <span className="font-semibold">Avg Recency:</span> {seg.avg_recency.toFixed(0)} days
                </div>
              </div>
              
              {/* State Breakdown */}
              <div className="grid grid-cols-2 gap-2 pt-2 border-t border-gray-300">
                <div className="text-center">
                  <p className="text-xs font-semibold text-muted">MAH</p>
                  <p className={`text-lg font-bold ${colors.text}`}>{seg.mah_count.toLocaleString()}</p>
                </div>
                <div className="text-center">
                  <p className="text-xs font-semibold text-muted">UP</p>
                  <p className={`text-lg font-bold ${colors.text}`}>{seg.up_count.toLocaleString()}</p>
                </div>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

export default SegmentGrid
