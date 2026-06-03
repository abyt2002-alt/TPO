import { BrowserRouter as Router, useLocation } from 'react-router-dom'
import RFMAnalysis from './pages/RFMAnalysis'
import Dashboard from './pages/Dashboard'

const AppRoutes = () => {
  const location = useLocation()
  const isDashboard = location.pathname === '/'
  const isRFM = location.pathname === '/rfm'

  return (
    <>
      {/* Dashboard — shown only on / */}
      {isDashboard && <Dashboard />}

      {/* RFMAnalysis — always mounted, hidden when not on /rfm */}
      <div style={{ display: isRFM ? 'contents' : 'none' }}>
        <RFMAnalysis />
      </div>
    </>
  )
}

function App() {
  return (
    <Router>
      <AppRoutes />
    </Router>
  )
}

export default App
