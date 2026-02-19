import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import RFMAnalysis from './pages/RFMAnalysis'
import Dashboard from './pages/Dashboard'

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/rfm" element={<RFMAnalysis />} />
      </Routes>
    </Router>
  )
}

export default App
