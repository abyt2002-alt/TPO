import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom'
import RFMAnalysis from './pages/RFMAnalysis'

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Navigate to="/rfm" replace />} />
        <Route path="/rfm" element={<RFMAnalysis />} />
      </Routes>
    </Router>
  )
}

export default App
