import axios from 'axios'

const resolveDefaultApiBaseUrl = () => {
  if (import.meta.env.DEV) return ''
  if (typeof window === 'undefined') return ''
  // For preview/static network sharing, fall back to backend on same host:8000.
  return `${window.location.protocol}//${window.location.hostname}:8000`
}

const API_BASE_URL = (import.meta.env.VITE_API_URL || '').trim() || resolveDefaultApiBaseUrl()

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// RFM API calls
export const getAvailableFilters = async () => {
  const response = await api.get('/api/rfm/filters', {
    timeout: 120000,
  })
  return response.data
}

export const getCascadingFilters = async (currentFilters) => {
  const response = await api.post('/api/rfm/filters/cascade', currentFilters, {
    timeout: 120000,
  })
  return response.data
}

export const calculateRFM = async (filters) => {
  const response = await api.post('/api/rfm/calculate', filters, {
    timeout: 300000,
  })
  return response.data
}

export const exportRFMOutlets = async (filters) => {
  const response = await api.post('/api/rfm/export', filters, {
    responseType: 'blob',
  })
  return response
}

export const calculateBaseDepth = async (payload) => {
  const response = await api.post('/api/discount/base-depth', payload)
  return response.data
}

export const getDiscountOptions = async (payload) => {
  const response = await api.post('/api/discount/options', payload)
  return response.data
}

export const calculateModeling = async (payload) => {
  const response = await api.post('/api/discount/modeling', payload)
  return response.data
}

export const calculate12MonthPlanner = async (payload) => {
  const response = await api.post('/api/planner/12-month', payload, {
    timeout: 180000,
  })
  return response.data
}

export const comparePlannerScenarios = async ({ payload, file }) => {
  const formData = new FormData()
  formData.append('payload_json', JSON.stringify(payload || {}))
  formData.append('file', file)
  const response = await api.post('/api/planner/scenario-compare', formData, {
    timeout: 600000,
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  })
  return response.data
}

export const getEDAOptions = async (payload) => {
  const response = await api.post('/api/eda/options', payload, {
    timeout: 120000,
  })
  return response.data
}

export const getEDAOverview = async (payload) => {
  const response = await api.post('/api/eda/overview', payload, {
    timeout: 120000,
  })
  return response.data
}

export const createRun = async (runId = null) => {
  const response = await api.post('/api/runs/create', runId ? { run_id: runId } : {})
  return response.data
}

export const getRunState = async (runId) => {
  const response = await api.get(`/api/runs/${runId}/state`)
  return response.data
}

export const saveRunState = async (runId, payload) => {
  const response = await api.post(`/api/runs/${runId}/state`, payload)
  return response.data
}

export const getRFMSegments = async () => {
  const response = await api.get('/api/rfm/segments')
  return response.data
}

// Health check
export const healthCheck = async () => {
  const response = await api.get('/health')
  return response.data
}

export default api
