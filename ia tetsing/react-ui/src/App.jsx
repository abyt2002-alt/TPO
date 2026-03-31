import { useEffect, useMemo, useState } from 'react'
import './App.css'

const MONTHS = ['Dec', 'Jan', 'Feb']
const SIZES = {
  '12-ML': ['slab1', 'slab2'],
  '18-ML': ['slab1', 'slab2', 'slab3', 'slab4'],
}
const DEFAULTS = {
  Dec: {
    '12-ML': { slab1: 14, slab2: 21 },
    '18-ML': { slab1: 11.5, slab2: 15.5, slab3: 16.5, slab4: 17 },
  },
  Jan: {
    '12-ML': { slab1: 14, slab2: 21 },
    '18-ML': { slab1: 11.5, slab2: 15.5, slab3: 16.5, slab4: 17 },
  },
  Feb: {
    '12-ML': { slab1: 14, slab2: 21 },
    '18-ML': { slab1: 11.5, slab2: 15.5, slab3: 16.5, slab4: 17 },
  },
}
const DEFAULT_PROMPT =
  'Generate realistic scenarios with focus on higher revenue but controlled discount depth.'
const DEFAULT_NOTES = [
  'Scenario planner context (important):',
  '1) There is a negative relationship between 12-ML and 18-ML. They can eat into each other volume.',
  '2) Slab volume uses beta_own * current slab discount, beta_lag * previous month discount, beta_other * weighted discount of other slabs in same size.',
  '3) Slab volumes are summed to get total 12-ML volume and 18-ML volume.',
  '4) Cross-elasticity adjustment is applied on month-on-month change from previous month.',
  '5) Keep scenario ladders valid, realistic, and business-applicable.',
].join('\n')

function clampDiscount(value) {
  const parsed = Math.round(Number(value))
  if (Number.isNaN(parsed)) return 1
  return Math.max(1, Math.min(30, parsed))
}

function enforceLadder(values) {
  const fixed = []
  let rangeOrRoundingFixes = 0
  let ladderFixes = 0

  values.forEach((raw, idx) => {
    const floatVal = Number(raw)
    const rounded = Number.isNaN(floatVal) ? 1 : Math.round(floatVal)
    let val = Math.max(1, Math.min(30, rounded))
    if (rounded !== val || Math.abs(floatVal - rounded) > 1e-9) rangeOrRoundingFixes += 1
    if (idx > 0 && val < fixed[idx - 1]) {
      val = fixed[idx - 1]
      ladderFixes += 1
    }
    fixed.push(val)
  })

  return { fixed, rangeOrRoundingFixes, ladderFixes }
}

function buildEmptyScenario() {
  const out = {}
  MONTHS.forEach((month) => {
    out[month] = {}
    Object.entries(SIZES).forEach(([size, slabs]) => {
      out[month][size] = {}
      slabs.forEach((slab) => {
        out[month][size][slab] = DEFAULTS[month][size][slab]
      })
    })
  })
  return out
}

function normalizeScenario(item, index) {
  const monthsData = item?.months ?? {}
  const out = buildEmptyScenario()
  const corrections = {
    missing_filled: 0,
    range_or_rounding_fixes: 0,
    ladder_fixes: 0,
  }

  MONTHS.forEach((month) => {
    Object.entries(SIZES).forEach(([size, slabs]) => {
      const incoming = monthsData?.[month]?.[size] ?? {}
      const ladderValues = slabs.map((slab) => {
        if (!(slab in incoming)) corrections.missing_filled += 1
        return incoming[slab] ?? out[month][size][slab]
      })
      const { fixed, rangeOrRoundingFixes, ladderFixes } = enforceLadder(ladderValues)
      corrections.range_or_rounding_fixes += rangeOrRoundingFixes
      corrections.ladder_fixes += ladderFixes
      slabs.forEach((slab, i) => {
        out[month][size][slab] = fixed[i]
      })
    })
  })

  return {
    name: `SCN-${String(index + 1).padStart(3, '0')}`,
    months: out,
    corrections,
  }
}

function extractJSONObject(text) {
  const trimmed = String(text ?? '').trim()
  if (!trimmed) throw new Error('Gemini returned empty response')
  try {
    const parsed = JSON.parse(trimmed)
    if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) return parsed
  } catch {
    // continue
  }
  const match = trimmed.match(/\{[\s\S]*\}/)
  if (!match) throw new Error('No JSON object found in Gemini response')
  return JSON.parse(match[0])
}

function strategyPrompt(userPrompt, plannerContext, sampleIndex = 1, sampleTotal = 1) {
  const strategySchema = {
    base_min: 10,
    base_max: 18,
    gap_min: 1,
    gap_max: 4,
    month_pattern: 'flat',
    month_shift_strength: 2,
    size_bias_12: 0,
    size_bias_18: 0,
    volatility: 1,
  }
  return [
    'Return ONE JSON OBJECT only (no markdown).',
    'Task: pick a strategy to generate many discount scenarios in JavaScript.',
    'The strategy object controls generator ranges and movement behavior.',
    'Constraints:',
    '- Integers only.',
    '- Discounts must remain in range [1,30].',
    '- Slab ladder per month/size must be non-decreasing.',
    'Allowed month_pattern values: flat, up, down, wave, pulse.',
    `Strategy schema: ${JSON.stringify(strategySchema)}`,
    `Default ladder reference: ${JSON.stringify(DEFAULTS)}`,
    `Planner/model context (must consider while proposing scenarios): ${JSON.stringify(plannerContext)}`,
    `Sampling run: ${sampleIndex}/${sampleTotal}. You can explore a slightly different feasible strategy point than prior runs.`,
    `Goal: ${userPrompt}`,
  ].join('\n')
}

function sanitizeStrategy(strategy) {
  const pattern = String(strategy?.month_pattern ?? 'flat').toLowerCase().trim()
  const safePattern = ['flat', 'up', 'down', 'wave', 'pulse'].includes(pattern) ? pattern : 'flat'
  let baseMin = clampDiscount(strategy?.base_min ?? 10)
  let baseMax = clampDiscount(strategy?.base_max ?? 18)
  if (baseMax < baseMin) [baseMin, baseMax] = [baseMax, baseMin]

  let gapMin = Math.max(1, Math.min(10, Math.round(Number(strategy?.gap_min ?? 1))))
  let gapMax = Math.max(1, Math.min(10, Math.round(Number(strategy?.gap_max ?? 4))))
  if (gapMax < gapMin) [gapMin, gapMax] = [gapMax, gapMin]

  const monthShiftStrength = Math.max(
    0,
    Math.min(6, Math.round(Number(strategy?.month_shift_strength ?? 2))),
  )
  const volatility = Math.max(0, Math.min(4, Math.round(Number(strategy?.volatility ?? 1))))
  const sizeBias12 = Math.max(-5, Math.min(5, Math.round(Number(strategy?.size_bias_12 ?? 0))))
  const sizeBias18 = Math.max(-5, Math.min(5, Math.round(Number(strategy?.size_bias_18 ?? 0))))

  return {
    base_min: baseMin,
    base_max: baseMax,
    gap_min: gapMin,
    gap_max: gapMax,
    month_pattern: safePattern,
    month_shift_strength: monthShiftStrength,
    volatility,
    size_bias_12: sizeBias12,
    size_bias_18: sizeBias18,
  }
}

function monthShift(pattern, idx, strength) {
  if (pattern === 'up') return idx * strength
  if (pattern === 'down') return -idx * strength
  if (pattern === 'wave') return [0, strength, -strength][idx % 3]
  if (pattern === 'pulse') return [strength, 0, strength][idx % 3]
  return 0
}

function hashString(input) {
  let hash = 2166136261
  for (let i = 0; i < input.length; i += 1) {
    hash ^= input.charCodeAt(i)
    hash +=
      (hash << 1) + (hash << 4) + (hash << 7) + (hash << 8) + (hash << 24)
  }
  return hash >>> 0
}

function mulberry32(seed) {
  let t = seed >>> 0
  return function rand() {
    t += 0x6d2b79f5
    let v = Math.imul(t ^ (t >>> 15), t | 1)
    v ^= v + Math.imul(v ^ (v >>> 7), v | 61)
    return ((v ^ (v >>> 14)) >>> 0) / 4294967296
  }
}

function randomInt(rand, min, max) {
  return Math.floor(rand() * (max - min + 1)) + min
}

function scenarioSignature(item) {
  const flat = []
  MONTHS.forEach((month) => {
    Object.entries(SIZES).forEach(([size, slabs]) => {
      slabs.forEach((slab) => flat.push(Number(item.months?.[month]?.[size]?.[slab] ?? 0)))
    })
  })
  return flat.join(',')
}

async function buildScenariosFromStrategy(strategy, count, userPrompt, onProgress) {
  const s = sanitizeStrategy(strategy)
  const seedBase = hashString(`${JSON.stringify(s)}|${String(userPrompt ?? '').trim().toLowerCase()}`)
  const scenarios = []
  const seen = new Set()

  for (let i = 0; i < count; i += 1) {
    const rand = mulberry32((seedBase + i * 9973) >>> 0)
    const months = buildEmptyScenario()

    const anchorBase = randomInt(rand, s.base_min, s.base_max)
    const gapAnchor = randomInt(rand, s.gap_min, s.gap_max)

    MONTHS.forEach((month, mIdx) => {
      const shift = monthShift(s.month_pattern, mIdx, s.month_shift_strength)
      const noise = s.volatility > 0 ? randomInt(rand, -s.volatility, s.volatility) : 0

      const base12 = anchorBase + s.size_bias_12 + shift + noise
      const g12 = Math.max(1, gapAnchor + randomInt(rand, -1, 1))
      const v12 = enforceLadder([base12, base12 + g12]).fixed
      months[month]['12-ML'].slab1 = v12[0]
      months[month]['12-ML'].slab2 = v12[1]

      const base18 = anchorBase + s.size_bias_18 + shift + noise - 1
      const g18a = Math.max(1, gapAnchor + randomInt(rand, -1, 1))
      const g18b = Math.max(1, gapAnchor + randomInt(rand, -1, 1))
      const g18c = Math.max(1, gapAnchor + randomInt(rand, -1, 1))
      const v18 = enforceLadder([base18, base18 + g18a, base18 + g18a + g18b, base18 + g18a + g18b + g18c]).fixed
      SIZES['18-ML'].forEach((slab, idx) => {
        months[month]['18-ML'][slab] = v18[idx]
      })
    })

    const normalized = normalizeScenario({ months }, i)
    const sig = scenarioSignature(normalized)
    if (!seen.has(sig)) {
      seen.add(sig)
      scenarios.push(normalized)
    }

    if ((i + 1) % 100 === 0 || i + 1 === count) {
      onProgress?.(i + 1, count)
      await new Promise((resolve) => setTimeout(resolve, 0))
    }
  }

  if (scenarios.length < count) {
    throw new Error(
      `Generated ${scenarios.length} unique scenarios out of requested ${count}. Increase variability in prompt/context.`,
    )
  }
  return scenarios.slice(0, count)
}

function flattenScenarios(scenarios) {
  return scenarios.map((item) => {
    const row = {
      scenario_name: item.name,
      missing_filled: Number(item.corrections?.missing_filled ?? 0),
      range_or_rounding_fixes: Number(item.corrections?.range_or_rounding_fixes ?? 0),
      ladder_fixes: Number(item.corrections?.ladder_fixes ?? 0),
    }
    MONTHS.forEach((month) => {
      Object.entries(SIZES).forEach(([size, slabs]) => {
        slabs.forEach((slab) => {
          const col = `${month}_${size}_${slab}`.replaceAll('-', '')
          row[col] = item.months[month][size][slab]
        })
      })
    })
    return row
  })
}

function toCSV(rows) {
  if (!rows.length) return ''
  const headers = Object.keys(rows[0])
  const escapeCell = (value) => {
    const str = String(value ?? '')
    if (str.includes('"') || str.includes(',') || str.includes('\n')) {
      return `"${str.replaceAll('"', '""')}"`
    }
    return str
  }
  const lines = [headers.join(',')]
  rows.forEach((row) => {
    lines.push(headers.map((h) => escapeCell(row[h])).join(','))
  })
  return lines.join('\n')
}

function triggerDownload(filename, text, mimeType) {
  const blob = new Blob([text], { type: mimeType })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}

async function fetchGeminiStrategy({
  apiKey,
  prompt,
  plannerContext,
  temperature,
  setStage,
  sampleIndex = 1,
  sampleTotal = 1,
}) {
  const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=${encodeURIComponent(apiKey)}`
  const payload = {
    contents: [{ parts: [{ text: strategyPrompt(prompt, plannerContext) }] }],
    generationConfig: {
      temperature,
      responseMimeType: 'application/json',
    },
  }

  let lastError = null
  for (let attempt = 1; attempt <= 4; attempt += 1) {
    try {
      setStage(`Gemini thinking... strategy build (attempt ${attempt}/4)`)
      const controller = new AbortController()
      const timeout = setTimeout(() => controller.abort(), 180000)
      const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...payload,
          contents: [
            {
              parts: [
                {
                  text: strategyPrompt(prompt, plannerContext, sampleIndex, sampleTotal),
                },
              ],
            },
          ],
        }),
        signal: controller.signal,
      })
      clearTimeout(timeout)
      if (!res.ok) {
        const txt = await res.text()
        throw new Error(`HTTP ${res.status}: ${txt}`)
      }
      const data = await res.json()
      const text = data?.candidates?.[0]?.content?.parts?.[0]?.text ?? ''
      const strategy = extractJSONObject(text)
      return { strategy, raw: text }
    } catch (err) {
      lastError = err
      if (attempt < 4) await new Promise((resolve) => setTimeout(resolve, Math.min(2 ** (attempt - 1), 8) * 1000))
    }
  }
  throw new Error(`Gemini strategy failed after retries: ${String(lastError)}`)
}

function medianInt(values, fallback) {
  if (!values.length) return fallback
  const arr = [...values].sort((a, b) => a - b)
  const mid = Math.floor(arr.length / 2)
  const v = arr.length % 2 === 0 ? Math.round((arr[mid - 1] + arr[mid]) / 2) : arr[mid]
  return Number.isFinite(v) ? v : fallback
}

function aggregateStrategies(samples) {
  if (!samples.length) throw new Error('No strategy samples available to aggregate')
  const sanitized = samples.map((s) => sanitizeStrategy(s))
  const patternCounts = sanitized.reduce((acc, s) => {
    const key = s.month_pattern
    acc[key] = (acc[key] || 0) + 1
    return acc
  }, {})
  const monthPattern = Object.entries(patternCounts).sort((a, b) => b[1] - a[1])[0][0]

  let baseMin = clampDiscount(medianInt(sanitized.map((s) => s.base_min), 10))
  let baseMax = clampDiscount(medianInt(sanitized.map((s) => s.base_max), 18))
  if (baseMax < baseMin) [baseMin, baseMax] = [baseMax, baseMin]
  let gapMin = Math.max(1, Math.min(10, medianInt(sanitized.map((s) => s.gap_min), 1)))
  let gapMax = Math.max(1, Math.min(10, medianInt(sanitized.map((s) => s.gap_max), 4)))
  if (gapMax < gapMin) [gapMin, gapMax] = [gapMax, gapMin]

  return sanitizeStrategy({
    base_min: baseMin,
    base_max: baseMax,
    gap_min: gapMin,
    gap_max: gapMax,
    month_pattern: monthPattern,
    month_shift_strength: medianInt(
      sanitized.map((s) => s.month_shift_strength),
      2,
    ),
    volatility: medianInt(
      sanitized.map((s) => s.volatility),
      1,
    ),
    size_bias_12: medianInt(
      sanitized.map((s) => s.size_bias_12),
      0,
    ),
    size_bias_18: medianInt(
      sanitized.map((s) => s.size_bias_18),
      0,
    ),
  })
}

function App() {
  const [apiKey, setApiKey] = useState('')
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT)
  const [scenarioCount, setScenarioCount] = useState(20)
  const [temperature, setTemperature] = useState(0.4)
  const [strategySamples, setStrategySamples] = useState(30)
  const [contextOpen, setContextOpen] = useState(false)
  const [e12, setE12] = useState(-1.02)
  const [e18, setE18] = useState(-0.305)
  const [referenceWindow, setReferenceWindow] = useState('LY same 3M')
  const [modelNotes, setModelNotes] = useState(DEFAULT_NOTES)

  const [isGenerating, setIsGenerating] = useState(false)
  const [stageMessage, setStageMessage] = useState('AI idle')
  const [progressDone, setProgressDone] = useState(0)
  const [progressTotal, setProgressTotal] = useState(0)
  const [error, setError] = useState('')
  const [success, setSuccess] = useState('')

  const [scenarios, setScenarios] = useState([])
  const [rawBatches, setRawBatches] = useState([])
  const [selectedBatchIndex, setSelectedBatchIndex] = useState(0)
  const [selectedScenarioIndex, setSelectedScenarioIndex] = useState(0)
  const [rowsPerPage, setRowsPerPage] = useState(10)
  const [startRow, setStartRow] = useState(0)

  useEffect(() => {
    const stored = window.localStorage.getItem('ai_testing_gemini_key')
    if (stored) setApiKey(stored)
  }, [])

  const flatRows = useMemo(() => flattenScenarios(scenarios), [scenarios])
  const totalRows = flatRows.length
  const maxStart = Math.max(0, totalRows - rowsPerPage)
  const pagedRows = useMemo(
    () => flatRows.slice(startRow, startRow + rowsPerPage),
    [flatRows, startRow, rowsPerPage],
  )
  const selectedScenario = scenarios[selectedScenarioIndex] ?? null

  useEffect(() => {
    setStartRow((prev) => Math.min(prev, maxStart))
  }, [maxStart])

  useEffect(() => {
    setSelectedScenarioIndex((prev) => Math.min(prev, Math.max(0, scenarios.length - 1)))
  }, [scenarios.length])

  const progressPct = progressTotal > 0 ? Math.round((progressDone / progressTotal) * 100) : 0

  const plannerContext = useMemo(
    () => ({
      cross_elasticity_12_wrt_18: Number(e12),
      cross_elasticity_18_wrt_12: Number(e18),
      reference_window: referenceWindow,
      model_build_notes: modelNotes.trim(),
    }),
    [e12, e18, referenceWindow, modelNotes],
  )

  const correctionSummary = useMemo(() => {
    if (!flatRows.length) return null
    return flatRows.reduce(
      (acc, row) => {
        acc.missing += Number(row.missing_filled ?? 0)
        acc.range += Number(row.range_or_rounding_fixes ?? 0)
        acc.ladder += Number(row.ladder_fixes ?? 0)
        return acc
      },
      { missing: 0, range: 0, ladder: 0 },
    )
  }, [flatRows])

  async function handleGenerate() {
    setError('')
    setSuccess('')

    const safeCount = Math.max(1, Math.min(10000, Number(scenarioCount) || 1))
    const safeSamples = Math.max(1, Math.min(30, Number(strategySamples) || 1))
    if (!apiKey.trim()) {
      setError('Gemini API key is required.')
      return
    }
    if (!prompt.trim()) {
      setError('Prompt is required.')
      return
    }

    setIsGenerating(true)
    setProgressDone(0)
    setProgressTotal(safeSamples + safeCount)
    setStageMessage('Sending prompt to Gemini...')

    try {
      window.localStorage.setItem('ai_testing_gemini_key', apiKey.trim())
      const strategySampleList = []
      const rawList = []
      for (let sampleIdx = 1; sampleIdx <= safeSamples; sampleIdx += 1) {
      const { strategy, raw } = await fetchGeminiStrategy({
          apiKey: apiKey.trim(),
          prompt: prompt.trim(),
          plannerContext,
          temperature: Number(temperature),
          setStage: (msg) => setStageMessage(`${msg} | sample ${sampleIdx}/${safeSamples}`),
          sampleIndex: sampleIdx,
          sampleTotal: safeSamples,
        })
        strategySampleList.push(sanitizeStrategy(strategy))
        rawList.push({
          batch: sampleIdx,
          requested: 1,
          accepted: 1,
          type: 'strategy',
          raw,
        })
        setProgressDone(sampleIdx)
      }

      const aggregatedStrategy = aggregateStrategies(strategySampleList)
      rawList.push({
        batch: 'aggregated',
        requested: safeSamples,
        accepted: safeSamples,
        type: 'aggregated_strategy',
        raw: JSON.stringify(aggregatedStrategy, null, 2),
      })

      setStageMessage('AI strategy aggregation complete. Generating scenarios in JavaScript engine...')
      const generated = await buildScenariosFromStrategy(
        aggregatedStrategy,
        safeCount,
        prompt.trim(),
        (done, total) => {
          setProgressDone(safeSamples + done)
          setProgressTotal(safeSamples + total)
        },
      )

      setScenarios(generated)
      setRawBatches(rawList)
      setSelectedBatchIndex(0)
      setSelectedScenarioIndex(0)
      setStartRow(0)
      setStageMessage('Generation complete.')
      setProgressDone(safeSamples + safeCount)
      setProgressTotal(safeSamples + safeCount)
      setSuccess(
        `Generated ${generated.length} scenario(s) using ${safeSamples} AI strategy samples + aggregated strategy.`,
      )
    } catch (err) {
      setScenarios([])
      setRawBatches([])
      setSelectedBatchIndex(0)
      setSelectedScenarioIndex(0)
      setStageMessage('Generation failed.')
      setProgressDone(0)
      setProgressTotal(0)
      setError(`Gemini generation failed: ${String(err?.message ?? err)}`)
    } finally {
      setIsGenerating(false)
    }
  }

  return (
    <div className="page">
      <header className="hero">
        <div>
          <h1>AI Scenario Testing</h1>
          <p>Prompt-driven 12-ML / 18-ML slab scenario generator with validation and export.</p>
        </div>
        <div className="status-chip">{isGenerating ? 'Running' : 'Ready'}</div>
      </header>

      <section className="panel controls">
        <div className="field full">
          <label>Gemini API Key</label>
          <input
            type="password"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            placeholder="Paste GEMINI_API_KEY"
            autoComplete="off"
          />
        </div>

        <div className="field full">
          <label>Prompt</label>
          <textarea value={prompt} onChange={(e) => setPrompt(e.target.value)} rows={5} />
        </div>

        <div className="field">
          <label>Scenarios</label>
          <input
            type="number"
            min={1}
            max={10000}
            step={1}
            value={scenarioCount}
            onChange={(e) => setScenarioCount(Number(e.target.value))}
          />
        </div>
        <div className="field">
          <label>AI Strategy Samples</label>
          <input
            type="number"
            min={1}
            max={30}
            step={1}
            value={strategySamples}
            onChange={(e) => setStrategySamples(Number(e.target.value))}
          />
        </div>
        <div className="field">
          <label>AI Temperature</label>
          <input
            type="range"
            min={0}
            max={1}
            step={0.1}
            value={temperature}
            onChange={(e) => setTemperature(Number(e.target.value))}
          />
          <span className="range-value">{temperature.toFixed(1)}</span>
        </div>

        <div className="field full">
          <button type="button" className="toggle" onClick={() => setContextOpen((v) => !v)}>
            {contextOpen ? 'Hide Planner Context' : 'Show Planner Context'}
          </button>
        </div>

        {contextOpen && (
          <div className="context-grid">
            <div className="field">
              <label>Cross Elasticity (12 wrt 18)</label>
              <input type="number" step={0.001} value={e12} onChange={(e) => setE12(Number(e.target.value))} />
            </div>
            <div className="field">
              <label>Cross Elasticity (18 wrt 12)</label>
              <input type="number" step={0.001} value={e18} onChange={(e) => setE18(Number(e.target.value))} />
            </div>
            <div className="field full">
              <label>Reference Window</label>
              <select value={referenceWindow} onChange={(e) => setReferenceWindow(e.target.value)}>
                <option value="LY same 3M">LY same 3M</option>
                <option value="Last 3M">Last 3M</option>
              </select>
            </div>
            <div className="field full">
              <label>Model Build Notes</label>
              <textarea value={modelNotes} onChange={(e) => setModelNotes(e.target.value)} rows={7} />
            </div>
          </div>
        )}

        <div className="actions full">
          <button type="button" className="primary" onClick={handleGenerate} disabled={isGenerating}>
            {isGenerating ? 'Generating...' : 'Generate'}
          </button>
        </div>
      </section>

      <section className="panel status-panel">
        <div className="status-line">
          <strong>Stage:</strong> {stageMessage}
        </div>
        <div className="progress">
          <div className="progress-fill" style={{ width: `${progressPct}%` }} />
        </div>
        <div className="progress-meta">
          <span>{progressDone}</span>
          <span>/</span>
          <span>{progressTotal}</span>
          <span>({progressPct}%)</span>
        </div>
        {error && <div className="banner error">{error}</div>}
        {success && <div className="banner success">{success}</div>}
      </section>

      {!scenarios.length ? (
        <section className="panel empty">No scenarios yet. Click Generate to create a scenario set.</section>
      ) : (
        <>
          {correctionSummary && (
            <section className="panel corrections">
              <h2>Validation Summary</h2>
              <div className="stats">
                <div className="stat">
                  <span>Missing Filled</span>
                  <strong>{correctionSummary.missing}</strong>
                </div>
                <div className="stat">
                  <span>Range/Rounding Fixes</span>
                  <strong>{correctionSummary.range}</strong>
                </div>
                <div className="stat">
                  <span>Ladder Fixes</span>
                  <strong>{correctionSummary.ladder}</strong>
                </div>
              </div>
            </section>
          )}

          <section className="panel">
            <div className="panel-head">
              <h2>Preview</h2>
              <div className="inline-controls">
                <label>
                  Rows per page
                  <select value={rowsPerPage} onChange={(e) => setRowsPerPage(Number(e.target.value))}>
                    {[5, 10, 20, 50, 100].map((n) => (
                      <option key={n} value={n}>
                        {n}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  Start row
                  <input
                    type="number"
                    min={0}
                    max={maxStart}
                    value={startRow}
                    onChange={(e) => {
                      const next = Number(e.target.value)
                      if (Number.isNaN(next)) return
                      setStartRow(Math.max(0, Math.min(maxStart, next)))
                    }}
                  />
                </label>
              </div>
            </div>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    {Object.keys(flatRows[0] ?? {}).map((key) => (
                      <th key={key}>{key}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {pagedRows.map((row, idx) => (
                    <tr key={`${row.scenario_name}-${idx}`}>
                      {Object.keys(flatRows[0] ?? {}).map((key) => (
                        <td key={key}>{row[key]}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          <section className="panel">
            <div className="panel-head">
              <h2>Scenario Detail</h2>
              <label>
                Select scenario
                <select
                  value={selectedScenarioIndex}
                  onChange={(e) => setSelectedScenarioIndex(Number(e.target.value))}
                >
                  {scenarios.map((s, idx) => (
                    <option key={s.name} value={idx}>
                      {s.name}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            {selectedScenario && (
              <div className="months-grid">
                {MONTHS.map((month) => (
                  <div className="month-card" key={month}>
                    <h3>{month}</h3>
                    {Object.entries(SIZES).map(([size, slabs]) => (
                      <div className="size-block" key={`${month}-${size}`}>
                        <h4>{size}</h4>
                        <ul>
                          {slabs.map((slab) => (
                            <li key={`${month}-${size}-${slab}`}>
                              <span>{slab}</span>
                              <strong>{selectedScenario.months[month][size][slab]}%</strong>
                            </li>
                          ))}
                        </ul>
                      </div>
                    ))}
                  </div>
                ))}
              </div>
            )}
          </section>

          <section className="panel">
            <h2>Downloads</h2>
            <div className="actions">
              <button
                type="button"
                onClick={() => triggerDownload('generated_scenarios.csv', toCSV(flatRows), 'text/csv')}
              >
                Download CSV
              </button>
              <button
                type="button"
                onClick={() =>
                  triggerDownload('generated_scenarios.json', JSON.stringify(scenarios, null, 2), 'application/json')
                }
              >
                Download JSON
              </button>
            </div>
          </section>

          <section className="panel">
            <div className="panel-head">
              <h2>Raw Gemini Outputs (Used by App)</h2>
              <label>
                Batch
                <select value={selectedBatchIndex} onChange={(e) => setSelectedBatchIndex(Number(e.target.value))}>
                  {rawBatches.map((b, idx) => (
                    <option key={`${b.batch}-${idx}`} value={idx}>
                      Batch {b.batch} | requested={b.requested} | accepted={b.accepted}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            <pre className="raw-box">{rawBatches[selectedBatchIndex]?.raw ?? ''}</pre>
            <div className="actions">
              <button
                type="button"
                onClick={() => {
                  const jsonl = rawBatches
                    .map((b) =>
                      JSON.stringify({
                        batch: b.batch,
                        requested: b.requested,
                        accepted: b.accepted,
                        raw: b.raw,
                      }),
                    )
                    .join('\n')
                  triggerDownload('raw_gemini_batches.jsonl', jsonl, 'application/json')
                }}
              >
                Download Raw Gemini Batches (JSONL)
              </button>
            </div>
          </section>
        </>
      )}
    </div>
  )
}

export default App
