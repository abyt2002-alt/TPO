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

const buildProfileThresholds = (values, slabCount, fallbackCount = 5) => {
  const defaultThresholds = [8, 32, 576, 960]
  const safeCount = Math.min(20, Math.max(2, Number(slabCount || fallbackCount || 5)))
  const expected = Math.max(1, safeCount - 1)
  const base = [...defaultThresholds]
  while (base.length < expected) {
    const last = base[base.length - 1] ?? 8
    base.push(last + Math.max(1, Math.abs(last) * 0.1))
  }
  const parsed = (values || [])
    .map((v) => Number(v))
    .filter((v) => Number.isFinite(v))
    .slice(0, expected)
  while (parsed.length < expected) {
    parsed.push(base[parsed.length] ?? (parsed.length === 0 ? 8 : parsed[parsed.length - 1] + 1))
  }
  return parsed
}

export const Step2SlabDefinitionPanel = ({
  filters,
  onChange,
  activeSizes = [],
  title = 'Step 2: Slab Definition',
}) => {
  const definedSlabCount = Math.min(20, Math.max(2, Number(filters?.defined_slab_count || 5)))
  const profileSizeKeys = useMemo(() => {
    const keys = (activeSizes || [])
      .map((v) => String(v || '').toUpperCase().replace(/\s+/g, '').trim())
      .filter(Boolean)
    if (keys.length > 0) return Array.from(new Set(keys))
    return ['12-ML', '18-ML']
  }, [activeSizes])

  const normalizeProfileCount = (value) => {
    const parsed = Number(value)
    if (!Number.isFinite(parsed)) return definedSlabCount
    return Math.max(2, Math.min(20, Math.round(parsed)))
  }

  const buildDefinedProfiles = () => {
    const existing = filters?.defined_slab_profiles || {}
    const next = {}
    profileSizeKeys.forEach((sizeKey) => {
      const cfg = existing[sizeKey] || {}
      const count = normalizeProfileCount(cfg?.defined_slab_count)
      next[sizeKey] = {
        defined_slab_count: count,
        defined_slab_thresholds: buildProfileThresholds(cfg?.defined_slab_thresholds || [], count, definedSlabCount),
      }
    })
    return next
  }

  const getProfileConfig = (sizeKey) => {
    const profiles = buildDefinedProfiles()
    return profiles[sizeKey] || {
      defined_slab_count: definedSlabCount,
      defined_slab_thresholds: buildProfileThresholds([], definedSlabCount, definedSlabCount),
    }
  }

  const handleProfileCountChange = (sizeKey, value) => {
    const profiles = buildDefinedProfiles()
    const current = profiles?.[sizeKey] || {}
    const count = normalizeProfileCount(value)
    profiles[sizeKey] = {
      defined_slab_count: count,
      defined_slab_thresholds: buildProfileThresholds(current?.defined_slab_thresholds || [], count, definedSlabCount),
    }
    onChange('defined_slab_profiles', profiles)
  }

  const handleProfileThresholdChange = (sizeKey, idx, value) => {
    const profiles = buildDefinedProfiles()
    const current = profiles?.[sizeKey] || { defined_slab_count: definedSlabCount, defined_slab_thresholds: [] }
    const count = normalizeProfileCount(current.defined_slab_count)
    const thresholds = buildProfileThresholds(current.defined_slab_thresholds || [], count, definedSlabCount)
    const parsed = Number(value)
    thresholds[idx] = Number.isFinite(parsed) ? parsed : 0
    profiles[sizeKey] = {
      defined_slab_count: count,
      defined_slab_thresholds: thresholds,
    }
    onChange('defined_slab_profiles', profiles)
  }

  return (
    <div className="bg-white rounded-lg shadow-md p-5 space-y-4">
      <div>
        <h3 className="text-xl font-semibold text-body">{title}</h3>
        <p className="text-sm text-muted mt-1">
          Define direct slab cutoffs for `12-ML` and `18-ML`. `slab0` always stays below the first cutoff.
        </p>
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-1">Defined Slab Level</label>
        <div className="w-full px-3 py-2 text-sm border border-gray-300 rounded-lg bg-gray-50 text-gray-700">
          Monthly Quantity per Outlet
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        {profileSizeKeys.map((sizeKey) => {
          const profile = getProfileConfig(sizeKey)
          return (
            <div key={sizeKey} className="rounded-lg border border-gray-200 bg-gray-50 p-4 space-y-3">
              <div className="flex items-center justify-between gap-4">
                <h4 className="text-base font-semibold text-body">{sizeKey}</h4>
                <div className="w-32">
                  <label className="block text-xs font-medium text-gray-600 mb-1">Slabs</label>
                  <input
                    type="number"
                    min="2"
                    max="20"
                    step="1"
                    value={profile.defined_slab_count}
                    onChange={(e) => handleProfileCountChange(sizeKey, e.target.value)}
                    className="w-full px-3 py-2 text-sm border border-gray-300 rounded-lg bg-white"
                  />
                </div>
              </div>
              <div className="grid grid-cols-1 gap-2">
                {profile.defined_slab_thresholds.map((threshold, idx) => (
                  <div key={`${sizeKey}-threshold-${idx}`} className="grid grid-cols-[100px_1fr] items-center gap-3">
                    <span className="text-sm text-gray-700">Cutoff {idx + 1}</span>
                    <input
                      type="number"
                      step="0.01"
                      value={threshold}
                      onChange={(e) => handleProfileThresholdChange(sizeKey, idx, e.target.value)}
                      className="w-full px-3 py-2 text-sm border border-gray-300 rounded-lg bg-white"
                    />
                  </div>
                ))}
              </div>
              <p className="text-xs text-muted">
                Example: {profile.defined_slab_thresholds.join(', ')} creates `slab0` to `slab{profile.defined_slab_count - 1}`.
              </p>
            </div>
          )
        })}
      </div>
    </div>
  )
}

const DiscountStepFilters = ({
  filters,
  options,
  onChange,
  activeSizes = [],
  matchingOutlets = 0,
  isLoading = false,
  title = 'Step 2: Discount Analysis Filters',
  description = 'Select RFM groups and outlet types for base discount estimation.',
  loadingLabel = 'Updating Step 2 options...',
  matchingLabel = 'Matching outlets after Step 2 filters',
}) => {
  const [isExpanded, setIsExpanded] = useState(true)
  const hasActiveFilters =
    (filters?.rfm_segments?.length || 0) > 0 ||
    (filters?.outlet_classifications?.length || 0) > 0

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
            Cascade order: RFM Group -&gt; Outlet Type. Slab definition is set in the main Step 2 screen.
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
