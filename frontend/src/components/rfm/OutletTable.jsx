import { useState, useEffect, useRef, useCallback } from 'react'
import { Search, Download, ChevronDown, ChevronUp, Loader2 } from 'lucide-react'

const OutletTable = ({
  outlets = [],
  totalOutlets = 0,
  page = 1,
  pageSize = 20,
  totalPages = 1,
  isLoading = false,
  onQueryChange,
  onExport,
}) => {
  const [searchTerm, setSearchTerm] = useState('')
  const [sortConfig, setSortConfig] = useState({ key: 'total_net_amt', direction: 'desc' })
  const [currentPage, setCurrentPage] = useState(page)
  const [rowsPerPage, setRowsPerPage] = useState(pageSize)
  const firstSyncDone = useRef(false)
  const onQueryChangeRef = useRef(onQueryChange)
  const lastSentQueryRef = useRef('')

  useEffect(() => {
    onQueryChangeRef.current = onQueryChange
  }, [onQueryChange])

  const emitQuery = useCallback((query) => {
    const queryKey = JSON.stringify(query)
    if (lastSentQueryRef.current === queryKey) return
    lastSentQueryRef.current = queryKey
    onQueryChangeRef.current?.(query)
  }, [])

  useEffect(() => {
    setCurrentPage(page)
  }, [page])

  useEffect(() => {
    setRowsPerPage(pageSize)
  }, [pageSize])

  useEffect(() => {
    if (!firstSyncDone.current) {
      firstSyncDone.current = true
      return
    }
    emitQuery({
      page: currentPage,
      page_size: rowsPerPage,
      search: searchTerm,
      sort_key: sortConfig.key,
      sort_direction: sortConfig.direction,
    })
  }, [currentPage, emitQuery, rowsPerPage, sortConfig.direction, sortConfig.key])

  useEffect(() => {
    const handle = setTimeout(() => {
      if (!firstSyncDone.current) return
      emitQuery({
        page: 1,
        page_size: rowsPerPage,
        search: searchTerm,
        sort_key: sortConfig.key,
        sort_direction: sortConfig.direction,
      })
      setCurrentPage(1)
    }, 350)
    return () => clearTimeout(handle)
  }, [emitQuery, rowsPerPage, searchTerm, sortConfig.direction, sortConfig.key])

  const handleSort = (key) => {
    setSortConfig((prev) => ({
      key,
      direction: prev.key === key && prev.direction === 'desc' ? 'asc' : 'desc'
    }))
    setCurrentPage(1)
  }

  const SortIcon = ({ columnKey }) => {
    if (sortConfig.key !== columnKey) return <ChevronDown className="w-4 h-4 opacity-30" />
    return sortConfig.direction === 'desc'
      ? <ChevronDown className="w-4 h-4" />
      : <ChevronUp className="w-4 h-4" />
  }

  const startIndex = totalOutlets === 0 ? 0 : ((currentPage - 1) * rowsPerPage) + 1
  const endIndex = Math.min(currentPage * rowsPerPage, totalOutlets)

  return (
    <div className="bg-white rounded-lg shadow-md overflow-hidden flex flex-col" style={{ height: '600px' }}>
      <div className="flex-shrink-0 border-b border-gray-200 bg-gradient-to-r from-primary to-secondary p-4">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-xl font-bold text-white">
            Outlet Details ({totalOutlets.toLocaleString()} outlets)
          </h3>
          <button
            onClick={onExport}
            disabled={isLoading}
            className="flex items-center gap-2 px-4 py-2 bg-white text-primary rounded-lg hover:bg-gray-100 transition-colors font-medium disabled:opacity-60"
          >
            <Download className="w-4 h-4" />
            Download Full CSV
          </button>
        </div>
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-subtle w-5 h-5" />
          <input
            type="text"
            placeholder="Search by outlet ID, segment, or state..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-white"
          />
        </div>
      </div>

      <div className="px-4 py-2 bg-accent-light text-xs text-muted border-b border-gray-200">
        Server-side mode: only the current page is sent to browser.
      </div>

      <div className="flex-1 overflow-auto relative">
        {isLoading && (
          <div className="absolute inset-0 bg-white/70 z-20 flex items-center justify-center">
            <Loader2 className="w-6 h-6 animate-spin text-primary" />
          </div>
        )}
        <table className="w-full">
          <thead className="bg-gray-50 sticky top-0 z-10 shadow-sm">
            <tr>
              <th onClick={() => handleSort('outlet_id')} className="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider cursor-pointer hover:bg-gray-100 bg-gray-50">
                <div className="flex items-center gap-1">Outlet ID <SortIcon columnKey="outlet_id" /></div>
              </th>
              <th onClick={() => handleSort('final_state')} className="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider cursor-pointer hover:bg-gray-100 bg-gray-50">
                <div className="flex items-center gap-1">State <SortIcon columnKey="final_state" /></div>
              </th>
              <th onClick={() => handleSort('rfm_segment')} className="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider cursor-pointer hover:bg-gray-100 bg-gray-50">
                <div className="flex items-center gap-1">RFM Segment <SortIcon columnKey="rfm_segment" /></div>
              </th>
              <th onClick={() => handleSort('total_net_amt')} className="px-4 py-3 text-right text-xs font-semibold text-gray-700 uppercase tracking-wider cursor-pointer hover:bg-gray-100 bg-gray-50">
                <div className="flex items-center justify-end gap-1">Total Net Amt <SortIcon columnKey="total_net_amt" /></div>
              </th>
              <th onClick={() => handleSort('orders_count')} className="px-4 py-3 text-right text-xs font-semibold text-gray-700 uppercase tracking-wider cursor-pointer hover:bg-gray-100 bg-gray-50">
                <div className="flex items-center justify-end gap-1">Orders <SortIcon columnKey="orders_count" /></div>
              </th>
              <th onClick={() => handleSort('aov')} className="px-4 py-3 text-right text-xs font-semibold text-gray-700 uppercase tracking-wider cursor-pointer hover:bg-gray-100 bg-gray-50">
                <div className="flex items-center justify-end gap-1">AOV <SortIcon columnKey="aov" /></div>
              </th>
              <th onClick={() => handleSort('recency_days')} className="px-4 py-3 text-right text-xs font-semibold text-gray-700 uppercase tracking-wider cursor-pointer hover:bg-gray-100 bg-gray-50">
                <div className="flex items-center justify-end gap-1">Recency <SortIcon columnKey="recency_days" /></div>
              </th>
              <th onClick={() => handleSort('unique_order_days')} className="px-4 py-3 text-right text-xs font-semibold text-gray-700 uppercase tracking-wider cursor-pointer hover:bg-gray-100 bg-gray-50">
                <div className="flex items-center justify-end gap-1">Order Days <SortIcon columnKey="unique_order_days" /></div>
              </th>
              <th onClick={() => handleSort('orders_per_day')} className="px-4 py-3 text-right text-xs font-semibold text-gray-700 uppercase tracking-wider cursor-pointer hover:bg-gray-100 bg-gray-50">
                <div className="flex items-center justify-end gap-1">Orders/Day <SortIcon columnKey="orders_per_day" /></div>
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {outlets.map((outlet, index) => (
              <tr key={outlet.outlet_id || index} className="hover:bg-gray-50 transition-colors">
                <td className="px-4 py-3 text-sm font-medium text-gray-900">{outlet.outlet_id}</td>
                <td className="px-4 py-3 text-sm text-gray-700">{outlet.final_state}</td>
                <td className="px-4 py-3">
                  <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-accent-light text-primary">
                    {outlet.rfm_segment}
                  </span>
                </td>
                <td className="px-4 py-3 text-sm text-right font-semibold text-gray-900">
                  Rs.{outlet.total_net_amt?.toLocaleString('en-IN', { maximumFractionDigits: 2 })}
                </td>
                <td className="px-4 py-3 text-sm text-right text-gray-700">{outlet.orders_count}</td>
                <td className="px-4 py-3 text-sm text-right text-gray-700">
                  Rs.{outlet.aov?.toLocaleString('en-IN', { maximumFractionDigits: 2 })}
                </td>
                <td className="px-4 py-3 text-sm text-right text-gray-700">{outlet.recency_days} days</td>
                <td className="px-4 py-3 text-sm text-right text-gray-700">{outlet.unique_order_days}</td>
                <td className="px-4 py-3 text-sm text-right text-gray-700">{outlet.orders_per_day?.toFixed(4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="flex items-center justify-between gap-3 px-4 py-3 border-t border-gray-200 bg-white">
        <div className="text-xs text-muted">
          Showing {startIndex}-{endIndex} of {totalOutlets.toLocaleString()}
        </div>
        <div className="flex items-center gap-2">
          <label className="text-xs text-muted">Rows</label>
          <select
            value={rowsPerPage}
            onChange={(e) => {
              setRowsPerPage(parseInt(e.target.value, 10))
              setCurrentPage(1)
            }}
            className="text-xs border border-gray-300 rounded px-2 py-1"
          >
            <option value={20}>20</option>
            <option value={50}>50</option>
            <option value={100}>100</option>
          </select>
          <button
            type="button"
            onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
            disabled={currentPage === 1 || isLoading}
            className="px-3 py-1 text-xs rounded border border-gray-300 disabled:opacity-50"
          >
            Prev
          </button>
          <span className="text-xs text-body">
            Page {currentPage} / {Math.max(1, totalPages)}
          </span>
          <button
            type="button"
            onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
            disabled={currentPage >= totalPages || isLoading}
            className="px-3 py-1 text-xs rounded border border-gray-300 disabled:opacity-50"
          >
            Next
          </button>
        </div>
      </div>
    </div>
  )
}

export default OutletTable
