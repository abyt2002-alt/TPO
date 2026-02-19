import { useMemo, useState } from 'react'
import { Filter, ChevronDown, ChevronUp, Search, X } from 'lucide-react'

const StepMultiSelect = ({ label, options = [], selectedValues = [], onChange, placeholder = 'All' }) => {
  const [isOpen, setIsOpen] = useState(false)
  const [search, setSearch] = useState('')

  const filtered = useMemo(() => {
    if (!search.trim()) return options
    return options.filter((opt) => opt.toLowerCase().includes(search.toLowerCase()))
  }, [options, search])

  const toggle = (item) => {
    if (selectedValues.includes(item)) {
      onChange(selectedValues.filter((v) => v !== item))
    } else {
      onChange([...selectedValues, item])
    }
  }

  const handleSelectAll = () => {
    onChange(filtered)
  }

  const handleClear = () => {
    onChange([])
  }

  const text =
    selectedValues.length === 0
      ? placeholder
      : selectedValues.length === 1
        ? selectedValues[0]
        : `${selectedValues.length} selected`

  return (
    <div className="relative">
      <label className="block text-sm font-medium text-gray-700 mb-1">{label}</label>
      <button
        type="button"
        onClick={() => setIsOpen((v) => !v)}
        className="w-full px-3 py-2 text-left border border-gray-300 rounded-lg bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-primary flex items-center justify-between text-sm"
      >
        <span className={selectedValues.length === 0 ? 'text-gray-400' : 'text-gray-900'}>{text}</span>
        {isOpen ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
      </button>

      {isOpen && (
        <>
          <div className="fixed inset-0 z-40" onClick={() => setIsOpen(false)} />
          <div className="absolute z-50 w-full mt-1 bg-white border border-gray-300 rounded-lg shadow-lg max-h-80 overflow-hidden">
            <div className="p-2 border-b border-gray-200 bg-gray-50">
              <div className="relative">
                <Search className="absolute left-2 top-2.5 w-4 h-4 text-gray-400" />
                <input
                  type="text"
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  placeholder="Search options..."
                  className="w-full pl-8 pr-3 py-2 text-sm border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-primary"
                />
              </div>
              <div className="flex gap-2 mt-2">
                <button
                  type="button"
                  onClick={handleSelectAll}
                  className="flex-1 text-xs px-2 py-1 bg-primary text-white rounded hover:bg-opacity-90"
                >
                  Select All
                </button>
                <button
                  type="button"
                  onClick={handleClear}
                  className="flex-1 text-xs px-2 py-1 bg-gray-200 text-gray-700 rounded hover:bg-gray-300"
                >
                  Clear
                </button>
              </div>
            </div>
            <div className="max-h-60 overflow-y-auto">
              {filtered.length === 0 ? (
                <div className="px-3 py-2 text-sm text-gray-500">No options found</div>
              ) : (
                filtered.map((opt) => (
                  <label key={opt} className="flex items-center px-3 py-2 hover:bg-gray-50 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={selectedValues.includes(opt)}
                      onChange={() => toggle(opt)}
                      className="w-4 h-4 text-primary border-gray-300 rounded focus:ring-primary"
                    />
                    <span className="ml-2 text-sm text-gray-700">{opt}</span>
                  </label>
                ))
              )}
            </div>
          </div>
        </>
      )}
    </div>
  )
}

const DiscountStepFilters = ({
  filters,
  options,
  onChange,
  matchingOutlets = 0,
  isLoading = false,
  title = 'Step 2: Discount Analysis Filters',
  description = 'Select RFM groups, outlet types, and slabs for base discount estimation.',
  loadingLabel = 'Updating Step 2 options...',
  matchingLabel = 'Matching outlets after Step 2 filters',
}) => {
  const [isExpanded, setIsExpanded] = useState(true)

  const hasActiveFilters =
    (filters?.rfm_segments?.length || 0) > 0 ||
    (filters?.outlet_classifications?.length || 0) > 0 ||
    (filters?.slabs?.length || 0) > 0

  const clearAll = () => {
    onChange('rfm_segments', [])
    onChange('outlet_classifications', [])
    onChange('slabs', [])
  }

  return (
    <div className="bg-white rounded-lg shadow-md overflow-visible">
      <div
        className="bg-primary text-white p-4 flex items-center justify-between cursor-pointer"
        onClick={() => setIsExpanded((v) => !v)}
      >
        <div className="flex items-center space-x-3">
          <Filter size={20} />
          <div>
            <h3 className="text-lg font-semibold">{title}</h3>
            {hasActiveFilters && (
              <p className="text-xs text-white/80 mt-0.5">
                {[
                  filters?.rfm_segments?.length ? `${filters.rfm_segments.length} RFM group(s)` : '',
                  filters?.outlet_classifications?.length ? `${filters.outlet_classifications.length} outlet type(s)` : '',
                  filters?.slabs?.length ? `${filters.slabs.length} slab(s)` : '',
                ].filter(Boolean).join(', ')}
              </p>
            )}
          </div>
        </div>
        <div className="flex items-center gap-2">
          {hasActiveFilters && (
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation()
                clearAll()
              }}
              className="flex items-center gap-1 px-2 py-1 text-xs bg-white/20 hover:bg-white/30 rounded"
            >
              <X size={14} />
              Clear All
            </button>
          )}
          {isExpanded ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
        </div>
      </div>

      {isExpanded && (
        <div className="p-4 space-y-3">
          <p className="text-sm text-muted">{description}</p>
          <p className="text-xs text-muted bg-accent-light px-3 py-2 rounded-md">
            Cascade order: RFM Group -&gt; Outlet Type -&gt; Slab. Options update automatically after a short pause.
          </p>

          <StepMultiSelect
            label="RFM Group(s)"
            options={options?.rfm_segments || []}
            selectedValues={filters?.rfm_segments || []}
            onChange={(v) => onChange('rfm_segments', v)}
            placeholder="All RFM Groups"
          />
          <StepMultiSelect
            label="Outlet Type(s)"
            options={options?.outlet_classifications || []}
            selectedValues={filters?.outlet_classifications || []}
            onChange={(v) => onChange('outlet_classifications', v)}
            placeholder="All Outlet Types"
          />
          <StepMultiSelect
            label="Slab(s)"
            options={options?.slabs || []}
            selectedValues={filters?.slabs || []}
            onChange={(v) => onChange('slabs', v)}
            placeholder="All Slabs"
          />

          <div className="text-sm text-muted pt-1">
            {isLoading ? (
              <span>{loadingLabel}</span>
            ) : (
              <>
                {matchingLabel}:{' '}
                <span className="font-semibold text-body">{matchingOutlets.toLocaleString()}</span>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

export default DiscountStepFilters
