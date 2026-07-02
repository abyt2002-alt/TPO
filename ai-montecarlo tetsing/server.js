const fs = require('fs')
const http = require('http')
const https = require('https')
const path = require('path')
const { URL } = require('url')

const ROOT = __dirname
const PUBLIC_DIR = path.join(ROOT, 'public')
const OUTPUT_DIR = path.join(ROOT, 'outputs')
const CONTEXT_PATH = path.join(ROOT, 'sample_planner_context.json')
const PORT = Number(process.env.PORT || 8510)

const GEMINI_MODEL = String(process.env.GEMINI_MODEL || 'gemini-2.5-flash').trim()
const GEMINI_TEMPERATURE = 0.35
const GEMINI_TOP_P = 0.9
const GEMINI_MAX_OUTPUT_TOKENS = 9000

const MIME_TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8'))
}

function sendJson(res, statusCode, payload) {
  const body = JSON.stringify(payload, null, 2)
  res.writeHead(statusCode, {
    'Content-Type': 'application/json; charset=utf-8',
    'Content-Length': Buffer.byteLength(body),
  })
  res.end(body)
}

function sendText(res, statusCode, text, contentType = 'text/plain; charset=utf-8') {
  res.writeHead(statusCode, {
    'Content-Type': contentType,
    'Content-Length': Buffer.byteLength(text),
  })
  res.end(text)
}

function readRequestBody(req) {
  return new Promise((resolve, reject) => {
    let body = ''
    req.on('data', (chunk) => {
      body += chunk
      if (body.length > 5_000_000) {
        req.destroy()
        reject(new Error('Request body too large'))
      }
    })
    req.on('end', () => resolve(body))
    req.on('error', reject)
  })
}

function slabSortKey(value) {
  const match = String(value || '').toLowerCase().match(/(\d+)/)
  return match ? Number(match[1]) : Number.MAX_SAFE_INTEGER
}

function buildPromptWithFilterContext(basePrompt, discountConstraints, metricThresholds) {
  const constraintLines = []
  ;(Array.isArray(discountConstraints) ? discountConstraints : []).forEach((item) => {
    const period = String(item?.period || '').trim()
    const size = String(item?.size || '').trim()
    const slab = String(item?.slab || '').trim()
    const label = [period, size, slab].filter(Boolean).join(' ')
    const parts = []
    if (item?.min !== null && item?.min !== undefined && String(item.min).trim() !== '') parts.push(`>= ${item.min}`)
    if (item?.max !== null && item?.max !== undefined && String(item.max).trim() !== '') parts.push(`<= ${item.max}`)
    if (label && parts.length) constraintLines.push(`- ${label}: ${parts.join(' and ')}`)
  })

  const thresholds = metricThresholds && typeof metricThresholds === 'object' ? metricThresholds : {}
  const metricLines = []
  const pushMin = (name, key) => {
    const val = thresholds[key]
    if (val !== null && val !== undefined && String(val).trim() !== '') metricLines.push(`- ${name}: >= ${val}%`)
  }
  const pushMax = (name, key) => {
    const val = thresholds[key]
    if (val !== null && val !== undefined && String(val).trim() !== '') metricLines.push(`- ${name}: <= ${val}%`)
  }
  pushMin('12-ML volume change', 'min_12_volume_pct')
  pushMin('18-ML volume change', 'min_18_volume_pct')
  pushMin('TOTAL volume change', 'min_total_volume_pct')
  pushMin('TOTAL revenue change', 'min_revenue_pct')
  pushMin('TOTAL net margin change', 'min_gross_margin_pct')
  pushMin('TOTAL investment change', 'min_investment_pct')
  pushMax('TOTAL investment change', 'max_investment_pct')
  pushMax('TOTAL CTS (investment/revenue)', 'max_cts_pct')

  const parts = [String(basePrompt || '').trim()]
  if (constraintLines.length) parts.push(`Hard slab-month discount constraints:\n${constraintLines.join('\n')}`)
  if (metricLines.length) parts.push(`Hard output constraints:\n${metricLines.join('\n')}`)
  if (constraintLines.length || metricLines.length) {
    parts.push('Generate scenarios that satisfy all constraints. If needed, prioritize valid scenarios over aggressive exploration.')
  }
  return parts.filter((part) => String(part || '').trim()).join('\n\n')
}

function buildStep5AiPrompt({ scenarioCount, goal, userPrompt, periods, defaultsMatrix, plannerContext }) {
  const slabOrder = {}
  ;['12-ML', '18-ML'].forEach((sizeKey) => {
    slabOrder[sizeKey] = Object.keys(defaultsMatrix?.[sizeKey] || {}).sort((a, b) => slabSortKey(a) - slabSortKey(b))
  })

  const periodDefaults = {}
  ;(periods || []).forEach((period, idx) => {
    const pkey = String(period)
    periodDefaults[pkey] = {}
    ;['12-ML', '18-ML'].forEach((sizeKey) => {
      periodDefaults[pkey][sizeKey] = {}
      ;(slabOrder[sizeKey] || []).forEach((slabKey) => {
        const series = defaultsMatrix?.[sizeKey]?.[slabKey] || []
        const value = idx < series.length ? Number(series[idx]) : 10
        periodDefaults[pkey][sizeKey][slabKey] = Math.round(value * 100) / 100
      })
    })
  })

  const schema = {
    families: [
      {
        name: 'Balanced realistic',
        priority_weight: 0.4,
        base_min: 10,
        base_max: 17,
        gap_min: 1,
        gap_max: 3,
        month_pattern: 'flat',
        month_shift_strength: 2,
        size_bias_12: 0,
        size_bias_18: 0,
        volatility: 1,
        anchor_weights: {
          latest_month: 0.4,
          last_3m_avg: 0.3,
          ly_same_3m: 0.2,
          stress_explore: 0.1,
        },
      },
    ],
  }

  const promptText = String(userPrompt || '').trim()
  const promptLower = promptText.toLowerCase()
  const intentHints = []
  if (
    (promptLower.includes('12') || promptLower.includes('12ml') || promptLower.includes('12-ml')) &&
    ['increase', 'grow', 'growth', 'up', 'volume'].some((k) => promptLower.includes(k))
  ) {
    intentHints.push('If user asks 12-ML volume growth, prioritize stronger 12-ML discount moves (within constraints) before changing 18-ML.')
  }
  if (
    (promptLower.includes('18') || promptLower.includes('18ml') || promptLower.includes('18-ml')) &&
    ['increase', 'grow', 'growth', 'up', 'volume'].some((k) => promptLower.includes(k))
  ) {
    intentHints.push('If user asks 18-ML volume growth, prioritize stronger 18-ML discount moves (within constraints).')
  }
  if (['profit', 'margin', 'gm'].some((k) => promptLower.includes(k))) {
    intentHints.push('For profit/margin goals, prefer shallower discount ladders, especially for 18-ML (higher margin pack), unless prompt explicitly asks aggressive 18-ML discounting.')
  }
  if (['no deep', 'not deep', 'avoid deep', 'shallow'].some((k) => promptLower.includes(k)) && (promptLower.includes('18') || promptLower.includes('18ml') || promptLower.includes('18-ml'))) {
    intentHints.push('If prompt says no deep 18-ML discount, keep 18-ML ladder in shallow-to-moderate range and avoid high deep-discount tops.')
  }
  if (['revenue', 'sales'].some((k) => promptLower.includes(k))) {
    intentHints.push('For revenue goals, allow moderate-to-high discount ladders but keep family diversity and realism.')
  }
  if (!intentHints.length) {
    intentHints.push('Map user intent explicitly by pack: 12-ML actions should primarily come from 12-ML ladders, 18-ML actions from 18-ML ladders.')
  }

  const promptPayload = {
    scenario_count: Number(scenarioCount || 5),
    goal: String(goal || '').trim() || 'maximize_revenue',
    user_prompt: promptText,
    periods: (periods || []).map(String),
    default_discounts_by_period: periodDefaults,
    slab_order: slabOrder,
    planner_context: plannerContext || {},
    intent_hints: intentHints,
    intent_resolution_framework: {
      step_1: 'Infer pack target from prompt (12-ML, 18-ML, both).',
      step_2: 'Infer objective priority (volume, revenue, profit, balanced).',
      step_3: 'Use coefficient signs/magnitudes to set family direction by slab.',
      step_4: 'Use margin profile to avoid unnecessary deep discounting when objective is profit.',
      step_5: 'Generate diverse but business-realistic families aligned to inferred intent.',
    },
    constraints: [
      'return EXACTLY 3 families',
      'all discounts are later repaired to integer [5,30]',
      'ladder is repaired monotonic later; still keep realistic family ranges',
      'avoid degenerate families (all ones / all flat tiny values)',
    ],
    schema,
  }

  return [
    'Return ONE JSON object only (no markdown, no code fences).',
    'You are generating scenario families for discount ladders.',
    'Create exactly 3 families with this schema:',
    JSON.stringify(schema),
    'Use planner_context deeply (do not ignore it):',
    '- size_slab_coefficients gives per-slab model coefficients.',
    '- pack_margin_profile gives relative pricing/margin context by pack.',
    '- cross elasticities indicate cross-pack relationship direction/strength.',
    'How to interpret coefficients:',
    '- coef_base_discount_pct: own discount sensitivity for that slab (stronger absolute magnitude => stronger response).',
    '- coef_lag1_base_discount_pct: carryover/drag from prior month discount.',
    '- coef_other_slabs_weighted_base_discount_pct: coupling/cannibalization signal within same pack.',
    'Pack-aware decision rules (must follow):',
    '- If user asks 12-ML growth, move 12-ML ladders up first; do not rely mainly on 18-ML changes.',
    '- If user asks 18-ML growth, move 18-ML ladders up first.',
    '- If user asks profit/margin increase, avoid deep discounting (especially in 18-ML unless explicitly asked).',
    '- Respect pack-specific constraints mentioned in prompt (e.g., keep 18-ML shallow).',
    '- If prompt is ambiguous, choose balanced families and explicitly hedge with one conservative and one growth family.',
    'Hard constraints:',
    '- month_pattern must be one of: flat, up, down, wave, pulse.',
    '- base_min/base_max/gap_min/gap_max must define meaningful spread.',
    '- priority_weight and anchor_weights can be floats; they will be normalized.',
    '- focus on realistic business scenarios, but keep diversity across families.',
    `Input:\n${JSON.stringify(promptPayload)}`,
  ].join('\n')
}

function callGemini(fullPrompt) {
  const apiKey = String(process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY || '').trim()
  if (!apiKey) {
    return Promise.reject(new Error('Missing GEMINI_API_KEY or GOOGLE_API_KEY in the server environment.'))
  }
  const requestBody = {
    contents: [{ role: 'user', parts: [{ text: fullPrompt }] }],
    generationConfig: {
      temperature: GEMINI_TEMPERATURE,
      topP: GEMINI_TOP_P,
      maxOutputTokens: GEMINI_MAX_OUTPUT_TOKENS,
      responseMimeType: 'application/json',
    },
  }
  const body = JSON.stringify(requestBody)
  const endpoint = new URL(`https://generativelanguage.googleapis.com/v1beta/models/${GEMINI_MODEL}:generateContent?key=${apiKey}`)

  return new Promise((resolve, reject) => {
    const req = https.request(
      {
        hostname: endpoint.hostname,
        path: endpoint.pathname + endpoint.search,
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Content-Length': Buffer.byteLength(body),
        },
        timeout: 75000,
      },
      (res) => {
        let responseBody = ''
        res.on('data', (chunk) => { responseBody += chunk })
        res.on('end', () => {
          if (res.statusCode < 200 || res.statusCode >= 300) {
            reject(new Error(`Gemini HTTP ${res.statusCode}: ${responseBody.slice(0, 500)}`))
            return
          }
          try {
            resolve({ requestBody, rawEnvelope: JSON.parse(responseBody) })
          } catch (err) {
            reject(new Error(`Invalid Gemini response JSON: ${err.message}`))
          }
        })
      }
    )
    req.on('timeout', () => {
      req.destroy(new Error('Gemini request timed out.'))
    })
    req.on('error', reject)
    req.write(body)
    req.end()
  })
}

function extractTextParts(envelope) {
  const out = []
  ;(envelope?.candidates || []).forEach((candidate) => {
    ;(candidate?.content?.parts || []).forEach((part) => {
      if (typeof part?.text === 'string' && part.text.trim()) out.push(part.text.trim())
    })
  })
  return out
}

function parseFamilyPayload(textParts) {
  for (const text of textParts || []) {
    try {
      const parsed = JSON.parse(text)
      if (parsed && Array.isArray(parsed.families)) return parsed
    } catch (_) {
      const match = String(text).match(/\{[\s\S]*\}/)
      if (!match) continue
      try {
        const parsed = JSON.parse(match[0])
        if (parsed && Array.isArray(parsed.families)) return parsed
      } catch (_) {
        continue
      }
    }
  }
  return null
}

function saveRun(record) {
  fs.mkdirSync(OUTPUT_DIR, { recursive: true })
  const stamp = new Date().toISOString().replace(/[-:]/g, '').replace(/\.\d{3}Z$/, 'Z')
  const filePath = path.join(OUTPUT_DIR, `gemini_family_run_${stamp}.json`)
  fs.writeFileSync(filePath, JSON.stringify(record, null, 2), 'utf8')
  return filePath
}

function listSavedRuns() {
  if (!fs.existsSync(OUTPUT_DIR)) return []
  return fs.readdirSync(OUTPUT_DIR)
    .filter((name) => /^gemini_family_run_.*\.json$/.test(name))
    .map((name) => {
      const filePath = path.join(OUTPUT_DIR, name)
      try {
        const record = readJson(filePath)
        const families = Array.isArray(record?.parsedFamilies?.families) ? record.parsedFamilies.families : []
        return {
          file: name,
          created_at: record?.created_at || '',
          prompt: record?.inputs?.userPrompt || '',
          prompt_with_constraints: record?.inputs?.promptWithConstraints || '',
          scenario_count: record?.inputs?.scenarioCount ?? null,
          constraint_count: Array.isArray(record?.inputs?.discountConstraints) ? record.inputs.discountConstraints.length : 0,
          family_count: families.length,
          family_names: families.map((family) => String(family?.name || '')).filter(Boolean),
        }
      } catch (err) {
        return {
          file: name,
          created_at: '',
          prompt: '',
          prompt_with_constraints: '',
          scenario_count: null,
          constraint_count: 0,
          family_count: 0,
          family_names: [],
          error: err.message || String(err),
        }
      }
    })
    .sort((a, b) => String(b.file).localeCompare(String(a.file)))
}

function readSavedRun(fileName) {
  const safeName = path.basename(String(fileName || ''))
  if (!/^gemini_family_run_.*\.json$/.test(safeName)) {
    throw new Error('Invalid saved run name.')
  }
  const filePath = path.join(OUTPUT_DIR, safeName)
  const normalized = path.normalize(filePath)
  if (!normalized.startsWith(OUTPUT_DIR)) {
    throw new Error('Invalid saved run path.')
  }
  if (!fs.existsSync(normalized)) {
    throw new Error('Saved run not found.')
  }
  return readJson(normalized)
}

async function handleRun(req, res) {
  const rawBody = await readRequestBody(req)
  const input = rawBody ? JSON.parse(rawBody) : {}
  const baseContext = readJson(CONTEXT_PATH)
  const context = input.context && typeof input.context === 'object' ? input.context : baseContext
  const discountConstraints = Array.isArray(input.discountConstraints) ? input.discountConstraints : []
  const metricThresholds = input.metricThresholds && typeof input.metricThresholds === 'object' ? input.metricThresholds : {}
  const plannerContext = JSON.parse(JSON.stringify(context.planner_context || {}))
  plannerContext.hard_discount_constraints = discountConstraints

  const promptWithConstraints = buildPromptWithFilterContext(input.userPrompt, discountConstraints, metricThresholds)
  const fullPrompt = buildStep5AiPrompt({
    scenarioCount: Number(input.scenarioCount || 5),
    goal: String(input.goal || 'maximize_revenue'),
    userPrompt: promptWithConstraints,
    periods: (context.periods || []).map(String),
    defaultsMatrix: context.defaults_matrix || {},
    plannerContext,
  })

  const started = Date.now()
  const gemini = await callGemini(fullPrompt)
  const textParts = extractTextParts(gemini.rawEnvelope)
  const parsedFamilies = parseFamilyPayload(textParts)
  const record = {
    created_at: new Date().toISOString(),
    stage: 'stage_1_gemini_family_creation',
    inputs: {
      userPrompt: input.userPrompt,
      goal: input.goal,
      scenarioCount: Number(input.scenarioCount || 5),
      discountConstraints,
      metricThresholds,
      context,
      promptWithConstraints,
    },
    fullPrompt,
    gemini,
    textParts,
    parsedFamilies,
    elapsedSeconds: Number(((Date.now() - started) / 1000).toFixed(3)),
  }
  const savedPath = saveRun(record)
  sendJson(res, 200, {
    success: true,
    elapsedSeconds: record.elapsedSeconds,
    savedPath,
    fullPrompt,
    textParts,
    rawEnvelope: gemini.rawEnvelope,
    parsedFamilies,
  })
}

function serveStatic(req, res, pathname) {
  let filePath = pathname === '/' ? path.join(PUBLIC_DIR, 'index.html') : path.join(PUBLIC_DIR, pathname)
  const normalized = path.normalize(filePath)
  if (!normalized.startsWith(PUBLIC_DIR)) {
    sendText(res, 403, 'Forbidden')
    return
  }
  fs.readFile(normalized, (err, data) => {
    if (err) {
      sendText(res, 404, 'Not found')
      return
    }
    const ext = path.extname(normalized).toLowerCase()
    res.writeHead(200, { 'Content-Type': MIME_TYPES[ext] || 'application/octet-stream' })
    res.end(data)
  })
}

const server = http.createServer(async (req, res) => {
  try {
    const url = new URL(req.url, `http://${req.headers.host}`)
    if (req.method === 'GET' && url.pathname === '/api/context') {
      sendJson(res, 200, readJson(CONTEXT_PATH))
      return
    }
    if (req.method === 'GET' && url.pathname === '/api/runs') {
      sendJson(res, 200, { success: true, runs: listSavedRuns() })
      return
    }
    if (req.method === 'GET' && url.pathname === '/api/run-detail') {
      sendJson(res, 200, { success: true, run: readSavedRun(url.searchParams.get('file')) })
      return
    }
    if (req.method === 'POST' && url.pathname === '/api/run') {
      await handleRun(req, res)
      return
    }
    if (req.method === 'GET') {
      serveStatic(req, res, url.pathname)
      return
    }
    sendText(res, 405, 'Method not allowed')
  } catch (err) {
    sendJson(res, 500, { success: false, message: err.message || String(err) })
  }
})

server.listen(PORT, '127.0.0.1', () => {
  console.log(`Gemini family tester running at http://127.0.0.1:${PORT}`)
})
