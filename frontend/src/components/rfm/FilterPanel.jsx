import { useState, useMemo } from 'react'
import { Filter, Play, ChevronDown, ChevronUp, Search, X } from 'lucide-react'

const MultiSelectDropdown = ({ label, options, selectedValues, onChange, placeholder, disabled = false }) => {
  const [isOpen, setIsOpen] = useState(false)
  const [searchTerm, setSearchTerm] = useState('')

  const filteredOptions = useMemo(() => {
    if (!searchTerm) return options || []
    return (options || []).filter(option => 
      option.toLowerCase().includes(searchTerm.toLowerCase())
    )
  }, [options, searchTerm])

  const handleToggle = (value) => {
    if (disabled) return
    const newValues = selectedValues.includes(value)
      ? selectedValues.filter(v => v !== value)
      : [...selectedValues, value]
    onChange(newValues)
  }

  const handleSelectAll = () => {
    if (disabled) return
    onChange(filteredOptions)
  }

  const handleClearAll = () => {
    if (disabled) return
    onChange([])
  }

  const displayText = selectedValues.length === 0 
    ? placeholder 
    : selectedValues.length === 1 
    ? selectedValues[0] 
    : `${selectedValues.length} selected`

  return (
    <div className="relative">
      <label className="block text-sm font-medium text-gray-700 mb-1">
        {label}
      </label>
      <button
        type="button"
        onClick={() => !disabled && setIsOpen(!isOpen)}
        disabled={disabled}
        className={`w-full px-3 py-2 text-left border border-gray-300 rounded-lg bg-white focus:outline-none focus:ring-2 focus:ring-primary flex items-center justify-between text-sm ${
          disabled ? 'opacity-60 cursor-not-allowed' : 'hover:bg-gray-50'
        }`}
      >
        <span className={selectedValues.length === 0 ? 'text-gray-400' : 'text-gray-900'}>
          {displayText}
        </span>
        {isOpen ? <ChevronUp className="w-4 h-4 flex-shrink-0" /> : <ChevronDown className="w-4 h-4 flex-shrink-0" />}
      </button>

      {isOpen && (
        <>
          <div 
            className="fixed inset-0 z-40" 
            onClick={() => setIsOpen(false)}
          />
          <div className="absolute z-50 w-full mt-1 bg-white border border-gray-300 rounded-lg shadow-lg max-h-80 overflow-hidden">
            <div className="p-2 border-b border-gray-200 bg-gray-50">
              <div className="relative">
                <Search className="absolute left-2 top-2.5 w-4 h-4 text-gray-400" />
                <input
                  type="text"
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  placeholder="Search options..."
                  className="w-full pl-8 pr-3 py-2 text-sm border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-primary"
                  onClick={(e) => e.stopPropagation()}
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
                  onClick={handleClearAll}
                  className="flex-1 text-xs px-2 py-1 bg-gray-200 text-gray-700 rounded hover:bg-gray-300"
                >
                  Clear
                </button>
              </div>
            </div>
            <div className="max-h-60 overflow-y-auto">
              {filteredOptions.length === 0 ? (
                <div className="px-3 py-2 text-sm text-gray-500">No options found</div>
              ) : (
                filteredOptions.map((option) => (
                  <label
                    key={option}
                    className="flex items-center px-3 py-2 hover:bg-gray-50 cursor-pointer"
                  >
                    <input
                      type="checkbox"
                      checked={selectedValues.includes(option)}
                      onChange={() => handleToggle(option)}
                      className="w-4 h-4 text-primary border-gray-300 rounded focus:ring-primary"
                    />
                    <span className="ml-2 text-sm text-gray-700">{option}</span>
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

const FilterPanel = ({ filters, availableFilters, onFilterChange, onCalculate, isCalculating, isCascadeLoading = false }) => {
  const [isExpanded, setIsExpanded] = useState(true)
  const [expandedSections, setExpandedSections] = useState({
    products: true,
    location: true,
    thresholds: true
  })

  const toggleSection = (section) => {
    setExpandedSections(prev => ({
      ...prev,
      [section]: !prev[section]
    }))
  }

  const hasActiveFilters = () => {
    return filters.states.length > 0 || 
           filters.categories.length > 0 || 
           filters.subcategories.length > 0 || 
           filters.brands.length > 0 || 
           filters.sizes.length > 0
  }

  const clearAllFilters = () => {
    onFilterChange('states', [])
    onFilterChange('categories', [])
    onFilterChange('subcategories', [])
    onFilterChange('brands', [])
    onFilterChange('sizes', [])
  }

  return (
    <div className="bg-white rounded-lg shadow-md overflow-visible">
      {/* Header */}
      <div
        className="bg-primary text-white p-4 flex items-center justify-between cursor-pointer"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center space-x-3">
          <Filter size={24} />
          <div>
            <h3 className="text-lg font-semibold">Step 1: RFM Calculation Filters</h3>
            {hasActiveFilters() && (
              <p className="text-xs text-white/80 mt-0.5">
                {[
                  filters.states.length && `${filters.states.length} state(s)`,
                  filters.categories.length && `${filters.categories.length} category(ies)`,
                  filters.subcategories.length && `${filters.subcategories.length} subcategory(ies)`,
                  filters.brands.length && `${filters.brands.length} brand(s)`,
                  filters.sizes.length && `${filters.sizes.length} size(s)`
                ].filter(Boolean).join(', ')}
              </p>
            )}
          </div>
        </div>
        <div className="flex items-center gap-2">
          {hasActiveFilters() && (
            <button
              onClick={(e) => {
                e.stopPropagation()
                clearAllFilters()
              }}
              className="flex items-center gap-1 px-2 py-1 text-xs bg-white/20 hover:bg-white/30 rounded"
            >
              <X size={14} />
              Clear All
            </button>
          )}
          {isExpanded ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
        </div>
      </div>

      {/* Content */}
      {isExpanded && (
        <div className="p-4 space-y-3">
          <p className="text-gray-600 text-sm">
            Select filters to define the dataset for RFM analysis
          </p>
          <p className="text-xs text-muted bg-accent-light px-3 py-2 rounded-md">
            Cascade order: State -&gt; Category -&gt; Subcategory -&gt; Brand -&gt; Size. Changing upper filters clears dependent lower filters.
            {isCascadeLoading ? ' Updating options after your last selection...' : ' Options are updated automatically after a short pause.'}
          </p>
          <p className="text-xs text-muted">
            Dropdown search filters option lists only. Data updates when you click Calculate RFM.
          </p>

          {/* Location Section */}
          <div className="border border-gray-200 rounded-lg">
            <button
              onClick={() => toggleSection('location')}
              className="w-full flex items-center justify-between p-3 hover:bg-gray-50 rounded-t-lg"
            >
              <span className="font-medium text-sm">Location</span>
              <div className="flex items-center gap-2">
                {filters.states.length > 0 && (
                  <span className="text-xs bg-primary text-white px-2 py-0.5 rounded-full">
                    {filters.states.length}
                  </span>
                )}
                {expandedSections.location ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
              </div>
            </button>
            {expandedSections.location && (
              <div className="p-3 pt-0 border-t border-gray-200">
                <MultiSelectDropdown
                  label="State(s)"
                  options={availableFilters?.states || []}
                  selectedValues={filters.states}
                  onChange={(values) => onFilterChange('states', values)}
                  placeholder="All States"
                />
              </div>
            )}
          </div>

          {/* Products Section */}
          <div className="border border-gray-200 rounded-lg">
            <button
              onClick={() => toggleSection('products')}
              className="w-full flex items-center justify-between p-3 hover:bg-gray-50 rounded-t-lg"
            >
              <span className="font-medium text-sm">Products</span>
              <div className="flex items-center gap-2">
                {(filters.categories.length + filters.subcategories.length + filters.brands.length + filters.sizes.length) > 0 && (
                  <span className="text-xs bg-primary text-white px-2 py-0.5 rounded-full">
                    {filters.categories.length + filters.subcategories.length + filters.brands.length + filters.sizes.length}
                  </span>
                )}
                {expandedSections.products ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
              </div>
            </button>
            {expandedSections.products && (
              <div className="p-3 pt-0 space-y-3 border-t border-gray-200">
                <MultiSelectDropdown
                  label="Category(ies)"
                  options={availableFilters?.categories || []}
                  selectedValues={filters.categories}
                  onChange={(values) => onFilterChange('categories', values)}
                  placeholder="All Categories"
                />
                <MultiSelectDropdown
                  label="Subcategory(ies)"
                  options={availableFilters?.subcategories || []}
                  selectedValues={filters.subcategories}
                  onChange={(values) => onFilterChange('subcategories', values)}
                  placeholder="All Subcategories"
                />
                <MultiSelectDropdown
                  label="Brand(s)"
                  options={availableFilters?.brands || []}
                  selectedValues={filters.brands}
                  onChange={(values) => onFilterChange('brands', values)}
                  placeholder="All Brands"
                />
                <MultiSelectDropdown
                  label="Size(s)"
                  options={availableFilters?.sizes || []}
                  selectedValues={filters.sizes}
                  onChange={(values) => onFilterChange('sizes', values)}
                  placeholder="All Sizes"
                />
              </div>
            )}
          </div>

          {/* RFM Thresholds Section */}
          <div className="border border-gray-200 rounded-lg">
            <button
              onClick={() => toggleSection('thresholds')}
              className="w-full flex items-center justify-between p-3 hover:bg-gray-50 rounded-t-lg"
            >
              <span className="font-medium text-sm">RFM Configuration</span>
              {expandedSections.thresholds ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            </button>
            {expandedSections.thresholds && (
              <div className="p-3 pt-0 space-y-3 border-t border-gray-200">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Recency Threshold (days)
                  </label>
                  <input
                    type="number"
                    min="30"
                    max="180"
                    step="10"
                    value={filters.recency_threshold}
                    onChange={(e) => onFilterChange('recency_threshold', parseInt(e.target.value))}
                    className="w-full px-3 py-2 text-sm border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary focus:border-transparent"
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    Outlets with last order within this many days are considered 'Recent'
                  </p>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Frequency Threshold (order days)
                  </label>
                  <input
                    type="number"
                    min="1"
                    max="100"
                    step="1"
                    value={filters.frequency_threshold}
                    onChange={(e) => onFilterChange('frequency_threshold', parseInt(e.target.value))}
                    className="w-full px-3 py-2 text-sm border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary focus:border-transparent"
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    Outlets with this many or more unique order days are considered 'High' frequency
                  </p>
                </div>
              </div>
            )}
          </div>

          {/* Calculate Button */}
          <div className="pt-2">
            <button
              onClick={onCalculate}
              disabled={isCalculating}
              className="w-full flex items-center justify-center space-x-2 bg-primary text-white px-6 py-3 rounded-lg hover:bg-opacity-90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed font-semibold"
            >
              {isCalculating ? (
                <>
                  <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                  <span>Calculating...</span>
                </>
              ) : (
                <>
                  <Play size={20} />
                  <span>Calculate RFM</span>
                </>
              )}
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

export default FilterPanel
