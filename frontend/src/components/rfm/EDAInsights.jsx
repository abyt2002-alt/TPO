import { useState } from 'react'
import { AlertCircle, Loader2 } from 'lucide-react'

const fmtNumber = (value) => {
  const n = Number(value || 0)
  return Number.isFinite(n) ? n.toLocaleString(undefined, { maximumFractionDigits: 0 }) : '0'
}

const fmtCurrency = (value) => {
  const n = Number(value || 0)
  if (!Number.isFinite(n)) return '0'
  return n.toLocaleString(undefined, { maximumFractionDigits: 2 })
}

const fmtPct = (value) => {
  const n = Number(value || 0)
  if (!Number.isFinite(n)) return '0.00%'
  return `${n.toFixed(2)}%`
}

const MixTable = ({ title, rows }) => {
  return (
    <div className="bg-white rounded-lg shadow-md p-4">
      <h4 className="text-base font-semibold text-body mb-3">{title}</h4>
      <div className="overflow-x-auto">
        <table className="w-full text-sm min-w-[520px]">
          <thead className="bg-gray-50">
            <tr>
              <th className="text-left px-3 py-2">Group</th>
              <th className="text-right px-3 py-2">Sales Value</th>
              <th className="text-right px-3 py-2">Quantity</th>
              <th className="text-right px-3 py-2">Value %</th>
              <th className="text-right px-3 py-2">Volume %</th>
            </tr>
          </thead>
          <tbody>
            {(rows || []).map((row) => (
              <tr key={row.key} className="border-t border-gray-100">
                <td className="px-3 py-2">{row.label}</td>
                <td className="px-3 py-2 text-right">{fmtCurrency(row.sales_value)}</td>
                <td className="px-3 py-2 text-right">{fmtNumber(row.quantity)}</td>
                <td className="px-3 py-2 text-right">{fmtPct(row.value_pct)}</td>
                <td className="px-3 py-2 text-right">{fmtPct(row.volume_pct)}</td>
              </tr>
            ))}
            {(!rows || rows.length === 0) && (
              <tr>
                <td className="px-3 py-3 text-muted" colSpan={5}>No rows available</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}

const EDAInsights = ({ data, isLoading, isError, errorMessage }) => {
  const [activeTab, setActiveTab] = useState('product')

  const renderProductTab = () => (
    <div className="space-y-4">
      <div className="bg-white rounded-lg shadow-md p-4">
        <h4 className="text-base font-semibold text-body mb-3">Product Contribution vs Brand Total</h4>
        <div className="overflow-auto max-h-[860px] border border-gray-100 rounded-md">
          <table className="w-full text-sm min-w-[1000px]">
            <thead className="bg-gray-50 sticky top-0 z-10">
              <tr>
                <th className="text-left px-3 py-2">Product</th>
                <th className="text-left px-3 py-2">Brand</th>
                <th className="text-left px-3 py-2">Category</th>
                <th className="text-left px-3 py-2">Subcategory</th>
                <th className="text-left px-3 py-2">Size</th>
                <th className="text-right px-3 py-2">Sales Value</th>
                <th className="text-right px-3 py-2">Quantity</th>
                <th className="text-right px-3 py-2">Brand Sales</th>
                <th className="text-right px-3 py-2">Brand Quantity</th>
                <th className="text-right px-3 py-2">Value Contribution</th>
                <th className="text-right px-3 py-2">Volume Contribution</th>
              </tr>
            </thead>
            <tbody>
              {(data?.product_contributions || []).map((row) => (
                <tr key={`${row.code}-${row.size}`} className="border-t border-gray-100">
                  <td className="px-3 py-2">{row.name || row.code}</td>
                  <td className="px-3 py-2">{row.brand}</td>
                  <td className="px-3 py-2">{row.category || '-'}</td>
                  <td className="px-3 py-2">{row.subcategory || '-'}</td>
                  <td className="px-3 py-2">{row.size}</td>
                  <td className="px-3 py-2 text-right">{fmtCurrency(row.sales_value)}</td>
                  <td className="px-3 py-2 text-right">{fmtNumber(row.quantity)}</td>
                  <td className="px-3 py-2 text-right">{fmtCurrency(row.brand_sales_value)}</td>
                  <td className="px-3 py-2 text-right">{fmtNumber(row.brand_quantity)}</td>
                  <td className="px-3 py-2 text-right font-semibold">{fmtPct(row.value_contribution_pct)}</td>
                  <td className="px-3 py-2 text-right font-semibold">{fmtPct(row.volume_contribution_pct)}</td>
                </tr>
              ))}
              {(data?.product_contributions || []).length === 0 && (
                <tr>
                  <td className="px-3 py-3 text-muted" colSpan={11}>No product rows available</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
      <MixTable title="Category Level Mix" rows={data?.category_level_mix || data?.category_mix || []} />
      {(data?.subcategory_within_category_sections || []).length > 0 ? (
        (data?.subcategory_within_category_sections || []).map((section, idx) => (
          <MixTable
            key={`${section?.category || 'category'}-${idx}`}
            title={`Within Category: ${section?.category || 'Unknown'} (Subcategory Mix)`}
            rows={section?.rows || []}
          />
        ))
      ) : (
        <MixTable
          title="Within Selected Category (Subcategory Mix)"
          rows={data?.subcategory_within_category_mix || []}
        />
      )}
    </div>
  )

  const renderBrandTab = () => (
    <div className="space-y-4">
      <MixTable title="Brand Mix" rows={data?.brand_mix || []} />
    </div>
  )

  const renderStateTab = () => (
    <div className="space-y-4">
      <MixTable title="State Mix" rows={data?.state_mix || []} />
      <MixTable title="Final Outlet Classification Mix" rows={data?.outlet_class_mix || []} />
    </div>
  )

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-xl font-bold text-body">EDA Explorer</h3>
        <p className="text-sm text-muted mt-1">
          Use product, state, and outlet filters to evaluate contribution, channel mix, and scale.
        </p>
      </div>

      {isLoading && (
        <div className="bg-white rounded-lg shadow-md p-8 flex items-center gap-3">
          <Loader2 className="animate-spin text-primary" size={20} />
          <p className="text-sm text-muted">Loading EDA insights...</p>
        </div>
      )}

      {isError && (
        <div className="bg-brand-dangerLight border border-danger rounded-lg p-4 flex items-start space-x-3">
          <AlertCircle className="text-danger flex-shrink-0 mt-0.5" size={20} />
          <div>
            <h4 className="font-semibold text-body">EDA Error</h4>
            <p className="text-muted text-sm">{errorMessage || 'Failed to load EDA insights'}</p>
          </div>
        </div>
      )}

      {data?.success === false && !isLoading && (
        <div className="bg-brand-dangerLight border border-danger rounded-lg p-4">
          <p className="text-sm text-body font-semibold">EDA could not run</p>
          <p className="text-sm text-muted mt-1">{data.message || 'No data available.'}</p>
        </div>
      )}

      {data?.success && !isLoading && (
        <>
          <div className="bg-white rounded-lg shadow-md p-4">
            <div className="grid grid-cols-1 md:grid-cols-3 xl:grid-cols-6 gap-3 text-sm">
              <div className="bg-accent-light rounded-md p-3">
                <p className="text-muted">Sales Value</p>
                <p className="font-bold text-body">{fmtCurrency(data?.summary?.total_sales_value)}</p>
              </div>
              <div className="bg-accent-light rounded-md p-3">
                <p className="text-muted">Quantity</p>
                <p className="font-bold text-body">{fmtNumber(data?.summary?.total_quantity)}</p>
              </div>
              <div className="bg-accent-light rounded-md p-3">
                <p className="text-muted">Outlets</p>
                <p className="font-bold text-body">{fmtNumber(data?.summary?.total_outlets)}</p>
              </div>
              <div className="bg-accent-light rounded-md p-3">
                <p className="text-muted">Rows</p>
                <p className="font-bold text-body">{fmtNumber(data?.summary?.total_rows)}</p>
              </div>
              <div className="bg-accent-light rounded-md p-3">
                <p className="text-muted">Products</p>
                <p className="font-bold text-body">{fmtNumber(data?.summary?.distinct_products)}</p>
              </div>
              <div className="bg-accent-light rounded-md p-3">
                <p className="text-muted">Brands</p>
                <p className="font-bold text-body">{fmtNumber(data?.summary?.distinct_brands)}</p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-md p-4">
            <div className="flex flex-wrap gap-2">
              <button
                type="button"
                onClick={() => setActiveTab('product')}
                className={`px-3 py-1.5 rounded-md text-sm font-semibold border ${
                  activeTab === 'product'
                    ? 'bg-white text-body border-primary'
                    : 'bg-white text-muted border-gray-300'
                }`}
              >
                Product Level
              </button>
              <button
                type="button"
                onClick={() => setActiveTab('brand')}
                className={`px-3 py-1.5 rounded-md text-sm font-semibold border ${
                  activeTab === 'brand'
                    ? 'bg-white text-body border-primary'
                    : 'bg-white text-muted border-gray-300'
                }`}
              >
                Brand Level
              </button>
              <button
                type="button"
                onClick={() => setActiveTab('state')}
                className={`px-3 py-1.5 rounded-md text-sm font-semibold border ${
                  activeTab === 'state'
                    ? 'bg-white text-body border-primary'
                    : 'bg-white text-muted border-gray-300'
                }`}
              >
                State Level
              </button>
            </div>
          </div>

          {activeTab === 'product' && renderProductTab()}
          {activeTab === 'brand' && renderBrandTab()}
          {activeTab === 'state' && renderStateTab()}
        </>
      )}
    </div>
  )
}

export default EDAInsights
