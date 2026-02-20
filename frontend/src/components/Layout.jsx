import { Link, useLocation } from 'react-router-dom'
import { BarChart3, Home, Menu, X, Settings, Percent, LineChart, CalendarDays, Upload } from 'lucide-react'
import { useState } from 'react'

const Layout = ({ children, rightSidebar }) => {
  const location = useLocation()
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false)
  const [isMobileSidebarOpen, setIsMobileSidebarOpen] = useState(false)
  const stepParam = new URLSearchParams(location.search).get('step')

  const navigation = [
    { name: 'Dashboard', href: '/', icon: Home },
    { name: 'Store Segmentation', href: '/rfm', icon: BarChart3, step: '1' },
    { name: 'Base Discount Estimator', href: '/rfm?step=2', icon: Percent, step: '2' },
    { name: 'Modeling & ROI', href: '/rfm?step=3', icon: LineChart, step: '3' },
    { name: '12-Month Planner', href: '/rfm?step=4', icon: CalendarDays, step: '4' },
    { name: 'Scenario Comparison', href: '/rfm?step=5', icon: Upload, step: '5' },
  ]

  const isNavActive = (item) => {
    if (item.href === '/') {
      return location.pathname === '/'
    }
    if (item.step === '2') {
      return location.pathname === '/rfm' && stepParam === '2'
    }
    if (item.step === '3') {
      return location.pathname === '/rfm' && stepParam === '3'
    }
    if (item.step === '4') {
      return location.pathname === '/rfm' && stepParam === '4'
    }
    if (item.step === '5') {
      return location.pathname === '/rfm' && stepParam === '5'
    }
    if (item.step === '1') {
      return location.pathname === '/rfm' && stepParam !== '2' && stepParam !== '3' && stepParam !== '4' && stepParam !== '5'
    }
    return location.pathname === item.href
  }

  return (
    <div className="h-screen flex flex-col overflow-hidden bg-canvas">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200 flex-shrink-0 z-30">
        <div className="px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <BarChart3 className="w-8 h-8 text-primary" />
              <h1 className="ml-3 text-xl font-bold text-body">
                Trade Promo Optimization Tool
              </h1>
            </div>
            
            {/* Mobile buttons */}
            <div className="flex items-center gap-2 lg:hidden">
              {rightSidebar && (
                <button
                  onClick={() => setIsMobileSidebarOpen(!isMobileSidebarOpen)}
                  className="p-2 rounded-md text-muted hover:text-body hover:bg-accent-light"
                >
                  <Settings className="w-6 h-6" />
                </button>
              )}
              <button
                onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
                className="p-2 rounded-md text-muted hover:text-body hover:bg-accent-light"
              >
                {isMobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
              </button>
            </div>
          </div>
        </div>

        {/* Mobile Navigation */}
        {isMobileMenuOpen && (
          <div className="lg:hidden border-t border-gray-200">
            <nav className="px-2 pt-2 pb-3 space-y-1">
              {navigation.map((item) => {
                const Icon = item.icon
                const isActive = isNavActive(item)
                return (
                  <Link
                    key={item.name}
                    to={item.href}
                    onClick={() => setIsMobileMenuOpen(false)}
                    className={`flex items-center px-3 py-2 rounded-md text-base font-medium ${
                      isActive
                        ? 'bg-white text-body border border-primary'
                        : 'text-body border border-transparent hover:bg-white hover:border-gray-300'
                    }`}
                  >
                    <Icon className="w-5 h-5 mr-3" />
                    {item.name}
                  </Link>
                )
              })}
            </nav>
          </div>
        )}
      </header>

      {/* Main Layout - Fixed height with flex */}
      <div className="flex-1 flex min-h-0">
        {/* Left Sidebar - Navigation */}
        <aside className="hidden lg:block lg:flex-shrink-0 w-64 bg-white border-r border-gray-200">
          <div className="h-full overflow-y-auto">
            <nav className="p-4 space-y-1">
              {navigation.map((item) => {
                const Icon = item.icon
                const isActive = isNavActive(item)
                return (
                  <Link
                    key={item.name}
                    to={item.href}
                    className={`flex items-center px-4 py-3 rounded-lg text-sm font-medium transition-colors ${
                      isActive
                        ? 'bg-white text-body border border-primary'
                        : 'text-body border border-transparent hover:bg-white hover:border-gray-300'
                    }`}
                  >
                    <Icon className="w-5 h-5 mr-3" />
                    {item.name}
                  </Link>
                )
              })}
            </nav>
          </div>
        </aside>

        {/* Center Content */}
        <main className="flex-1 overflow-y-auto min-w-0">
          <div className="p-4 sm:p-6 lg:p-8">
            {children}
          </div>
        </main>

        {/* Right Sidebar - Settings/Filters */}
        {rightSidebar && (
          <>
            {/* Desktop Right Sidebar */}
            <aside className="hidden lg:block lg:flex-shrink-0 w-80 bg-white border-l border-gray-200">
              <div className="h-full flex flex-col">
                {/* Fixed Header */}
                <div className="flex-shrink-0 flex items-center gap-2 p-4 border-b border-gray-200 bg-white">
                  <Settings className="w-5 h-5 text-primary" />
                  <h2 className="text-lg font-semibold text-body">Settings</h2>
                </div>
                {/* Scrollable Content */}
                <div className="flex-1 overflow-y-auto p-4">
                  {rightSidebar}
                </div>
              </div>
            </aside>

            {/* Mobile Right Sidebar */}
            {isMobileSidebarOpen && (
              <div className="fixed inset-0 z-40 lg:hidden">
                <div 
                  className="absolute inset-0 bg-body/50"
                  onClick={() => setIsMobileSidebarOpen(false)}
                />
                <div className="absolute right-0 top-0 bottom-0 w-full max-w-sm bg-white shadow-xl flex flex-col">
                  <div className="flex-shrink-0 flex items-center justify-between p-4 border-b border-gray-200">
                    <div className="flex items-center gap-2">
                      <Settings className="w-5 h-5 text-primary" />
                      <h2 className="text-lg font-semibold text-body">Settings</h2>
                    </div>
                    <button
                      onClick={() => setIsMobileSidebarOpen(false)}
                      className="p-2 rounded-md text-muted hover:text-body hover:bg-accent-light"
                    >
                      <X className="w-5 h-5" />
                    </button>
                  </div>
                  <div className="flex-1 overflow-y-auto p-4">
                    {rightSidebar}
                  </div>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}

export default Layout
