(function () {
  const { useEffect, useMemo, useState } = React

  const defaultMetricThresholds = {
    min_12_volume_pct: null,
    min_18_volume_pct: null,
    min_total_volume_pct: null,
    min_revenue_pct: null,
    min_gross_margin_pct: null,
    min_investment_pct: null,
    max_investment_pct: null,
    max_cts_pct: null,
  }

  function pretty(value) {
    return JSON.stringify(value, null, 2)
  }

  function parseJson(text, label) {
    try {
      return JSON.parse(text)
    } catch (err) {
      throw new Error(`${label} is not valid JSON: ${err.message}`)
    }
  }

  function slabSortKey(value) {
    const match = String(value || '').toLowerCase().match(/(\d+)/)
    return match ? Number(match[1]) : Number.MAX_SAFE_INTEGER
  }

  function fieldKey(period, size, slab, bound) {
    return `${period}||${size}||${slab}||${bound}`
  }

  function getSlabsForSize(context, sizeKey) {
    return Object.keys(context?.defaults_matrix?.[sizeKey] || {}).sort((a, b) => slabSortKey(a) - slabSortKey(b))
  }

  function toNullableNumber(value) {
    if (value === null || value === undefined) return null
    const text = String(value).trim()
    if (!text) return null
    const parsed = Number(text)
    return Number.isFinite(parsed) ? parsed : null
  }

  function buildDiscountConstraintsFromGrid(context, gridValues) {
    const periods = Array.isArray(context?.periods) ? context.periods.map(String) : []
    const constraints = []
    ;['12-ML', '18-ML'].forEach((sizeKey) => {
      getSlabsForSize(context, sizeKey).forEach((slabKey) => {
        periods.forEach((period) => {
          const min = toNullableNumber(gridValues[fieldKey(period, sizeKey, slabKey, 'min')])
          const max = toNullableNumber(gridValues[fieldKey(period, sizeKey, slabKey, 'max')])
          if (min === null && max === null) return
          constraints.push({ period, size: sizeKey, slab: slabKey, min, max })
        })
      })
    })
    return constraints
  }

  function FamilyCard({ family, index }) {
    const metrics = [
      ['base_min', family.base_min],
      ['base_max', family.base_max],
      ['gap_min', family.gap_min],
      ['gap_max', family.gap_max],
      ['pattern', family.month_pattern],
      ['shift', family.month_shift_strength],
      ['bias 12', family.size_bias_12],
      ['bias 18', family.size_bias_18],
      ['volatility', family.volatility],
      ['priority', family.priority_weight],
    ]
    return React.createElement(
      'div',
      { className: 'family-card' },
      React.createElement(
        'div',
        { className: 'family-head' },
        React.createElement('div', { className: 'family-name' }, `${index + 1}. ${family.name || 'Unnamed family'}`),
        React.createElement('div', { className: 'family-weight' }, `priority ${family.priority_weight ?? '-'}`)
      ),
      React.createElement(
        'div',
        { className: 'metric-grid' },
        metrics.map(([label, value]) =>
          React.createElement(
            'div',
            { className: 'metric', key: label },
            React.createElement('span', null, label),
            React.createElement('strong', null, value === undefined || value === null ? '-' : String(value))
          )
        )
      ),
      family.anchor_weights
        ? React.createElement('pre', { className: 'mini-pre' }, pretty({ anchor_weights: family.anchor_weights }))
        : null
    )
  }

  function ConstraintGrid({ context, gridValues, onChange }) {
    const periods = Array.isArray(context?.periods) ? context.periods.map(String) : []
    if (!periods.length) {
      return React.createElement('div', { className: 'hint' }, 'No periods found in planner context.')
    }

    return React.createElement(
      'div',
      { className: 'constraint-stack' },
      ['12-ML', '18-ML'].map((sizeKey) => {
        const slabs = getSlabsForSize(context, sizeKey)
        return React.createElement(
          'div',
          { className: 'constraint-block', key: sizeKey },
          React.createElement(
            'div',
            { className: 'constraint-title' },
            React.createElement('strong', null, sizeKey),
            React.createElement('span', null, `${slabs.length} slabs x ${periods.length} months`)
          ),
          React.createElement(
            'div',
            { className: 'constraint-table-wrap' },
            React.createElement(
              'table',
              { className: 'constraint-table' },
              React.createElement(
                'thead',
                null,
                React.createElement(
                  'tr',
                  null,
                  React.createElement('th', null, 'Slab'),
                  periods.map((period) => React.createElement('th', { key: period }, period))
                )
              ),
              React.createElement(
                'tbody',
                null,
                slabs.map((slabKey) =>
                  React.createElement(
                    'tr',
                    { key: slabKey },
                    React.createElement('td', { className: 'slab-cell' }, slabKey),
                    periods.map((period) =>
                      React.createElement(
                        'td',
                        { key: period },
                        React.createElement(
                          'div',
                          { className: 'mini-inputs' },
                          React.createElement('input', {
                            value: gridValues[fieldKey(period, sizeKey, slabKey, 'min')] || '',
                            placeholder: 'min',
                            inputMode: 'decimal',
                            onChange: (event) => onChange(fieldKey(period, sizeKey, slabKey, 'min'), event.target.value),
                          }),
                          React.createElement('input', {
                            value: gridValues[fieldKey(period, sizeKey, slabKey, 'max')] || '',
                            placeholder: 'max',
                            inputMode: 'decimal',
                            onChange: (event) => onChange(fieldKey(period, sizeKey, slabKey, 'max'), event.target.value),
                          })
                        )
                      )
                    )
                  )
                )
              )
            )
          )
        )
      })
    )
  }

  function App() {
    const [context, setContext] = useState(null)
    const [contextText, setContextText] = useState('')
    const [goal] = useState('maximize_revenue')
    const [scenarioCount, setScenarioCount] = useState(5)
    const [userPrompt, setUserPrompt] = useState('Grow 12-ML volume while keeping 18-ML discounts shallow and protecting margin.')
    const [constraintGrid, setConstraintGrid] = useState({})
    const [metricThresholdsText, setMetricThresholdsText] = useState(pretty(defaultMetricThresholds))
    const [isRunning, setIsRunning] = useState(false)
    const [error, setError] = useState('')
    const [result, setResult] = useState(null)
    const [activeTab, setActiveTab] = useState('families')
    const [savedRuns, setSavedRuns] = useState([])
    const [savedRunsError, setSavedRunsError] = useState('')

    useEffect(() => {
      fetch('/api/context')
        .then((res) => res.json())
        .then((data) => {
          setContext(data)
          setContextText(pretty(data))
        })
        .catch((err) => setError(err.message || 'Failed to load context'))
    }, [])

    function refreshSavedRuns() {
      setSavedRunsError('')
      fetch('/api/runs')
        .then((res) => res.json())
        .then((data) => {
          if (!data.success) throw new Error(data.message || 'Failed to load saved runs')
          setSavedRuns(Array.isArray(data.runs) ? data.runs : [])
        })
        .catch((err) => setSavedRunsError(err.message || 'Failed to load saved runs'))
    }

    useEffect(() => {
      refreshSavedRuns()
    }, [])

    const effectiveContext = useMemo(() => {
      try {
        return parseJson(contextText, 'Planner context')
      } catch (_) {
        return context
      }
    }, [contextText, context])

    const discountConstraints = useMemo(() => {
      return buildDiscountConstraintsFromGrid(effectiveContext, constraintGrid)
    }, [effectiveContext, constraintGrid])

    const families = useMemo(() => {
      return Array.isArray(result?.parsedFamilies?.families) ? result.parsedFamilies.families : []
    }, [result])

    function updateConstraint(key, value) {
      setConstraintGrid((prev) => ({ ...prev, [key]: value }))
    }

    function clearConstraints() {
      setConstraintGrid({})
    }

    async function runGemini() {
      setError('')
      setResult(null)
      let parsedContext
      let metricThresholds
      try {
        parsedContext = parseJson(contextText, 'Planner context')
        metricThresholds = parseJson(metricThresholdsText, 'Metric constraints')
        if (!metricThresholds || typeof metricThresholds !== 'object' || Array.isArray(metricThresholds)) {
          throw new Error('Metric constraints must be a JSON object.')
        }
        for (const item of discountConstraints) {
          if (item.min !== null && item.max !== null && Number(item.min) > Number(item.max)) {
            throw new Error(`Invalid constraint for ${item.period} ${item.size} ${item.slab}: min is greater than max.`)
          }
        }
      } catch (err) {
        setError(err.message)
        return
      }

      setIsRunning(true)
      try {
        const res = await fetch('/api/run', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            userPrompt,
            goal,
            scenarioCount: Number(scenarioCount),
            discountConstraints,
            metricThresholds,
            context: parsedContext,
          }),
        })
        const data = await res.json()
        if (!res.ok || !data.success) throw new Error(data.message || 'Gemini run failed')
        setResult(data)
        setActiveTab('families')
        refreshSavedRuns()
      } catch (err) {
        setError(err.message || 'Gemini run failed')
      } finally {
        setIsRunning(false)
      }
    }

    function resetContext() {
      if (!context) return
      setContextText(pretty(context))
    }

    async function loadSavedRun(fileName) {
      setError('')
      try {
        const res = await fetch(`/api/run-detail?file=${encodeURIComponent(fileName)}`)
        const data = await res.json()
        if (!res.ok || !data.success) throw new Error(data.message || 'Failed to load saved run')
        const run = data.run || {}
        setResult({
          elapsedSeconds: run.elapsedSeconds,
          savedPath: fileName,
          fullPrompt: run.fullPrompt,
          textParts: run.textParts || [],
          rawEnvelope: run?.gemini?.rawEnvelope || run.rawEnvelope || {},
          parsedFamilies: run.parsedFamilies || null,
        })
        if (run?.inputs?.userPrompt) setUserPrompt(run.inputs.userPrompt)
        if (run?.inputs?.scenarioCount) setScenarioCount(run.inputs.scenarioCount)
        setActiveTab('families')
        window.scrollTo({ top: 0, behavior: 'smooth' })
      } catch (err) {
        setError(err.message || 'Failed to load saved run')
      }
    }

    return React.createElement(
      'div',
      { className: 'app' },
      React.createElement(
        'header',
        { className: 'topbar' },
        React.createElement(
          'div',
          null,
          React.createElement('h1', { className: 'title' }, 'Gemini Scenario Family Tester'),
          React.createElement('p', { className: 'subtitle' }, 'Stage 1: business intent and constraints to Gemini-created family parameters')
        ),
        React.createElement('div', { className: 'status-pill' }, 'React test environment · app untouched')
      ),
      React.createElement(
        'main',
        { className: 'shell' },
        React.createElement(
          'section',
          { className: 'panel' },
          React.createElement('h2', null, 'Business Input'),
          React.createElement(
            'div',
            { className: 'row' },
            React.createElement(
              'div',
              { className: 'field' },
              React.createElement('label', null, 'Business prompt'),
              React.createElement('textarea', {
                value: userPrompt,
                onChange: (event) => setUserPrompt(event.target.value),
                placeholder: 'Example: Grow 12-ML volume while keeping 18-ML discounts shallow.',
              })
            ),
            React.createElement(
              'div',
              { className: 'field compact-field' },
              React.createElement('label', null, 'Number of scenarios'),
              React.createElement('input', {
                type: 'number',
                min: 1,
                max: 10000,
                value: scenarioCount,
                onChange: (event) => setScenarioCount(event.target.value),
              }),
              React.createElement('div', { className: 'hint' }, 'Used by Gemini as context for family allocation.')
            )
          )
        ),
        React.createElement(
          'section',
          { className: 'panel', style: { marginTop: 18 } },
          React.createElement(
            'div',
            { className: 'section-head' },
            React.createElement(
              'div',
              null,
              React.createElement('h2', null, 'Discount Constraints'),
              React.createElement('p', { className: 'subtitle inline-subtitle' }, 'Enter optional min/max discount bounds by pack, slab, and month.')
            ),
            React.createElement('button', { className: 'secondary', onClick: clearConstraints }, 'Clear constraints')
          ),
          React.createElement(ConstraintGrid, {
            context: effectiveContext,
            gridValues: constraintGrid,
            onChange: updateConstraint,
          }),
          React.createElement(
            'details',
            { className: 'dev-details' },
            React.createElement('summary', null, `Developer: structured constraints payload (${discountConstraints.length})`),
            React.createElement('pre', null, pretty(discountConstraints))
          )
        ),
        React.createElement(
          'section',
          { className: 'panel', style: { marginTop: 18 } },
          React.createElement(
            'div',
            { className: 'button-row' },
            React.createElement(
              'button',
              { className: 'primary', onClick: runGemini, disabled: isRunning },
              isRunning ? 'Running Gemini...' : 'Play / Run Gemini'
            ),
            React.createElement('span', { className: 'hint' }, 'Each run saves a JSON record in outputs/.')
          ),
          error ? React.createElement('div', { className: 'error' }, error) : null,
          result ? React.createElement('div', { className: 'success' }, `Returned in ${result.elapsedSeconds}s. Saved to ${result.savedPath}`) : null
        ),
        React.createElement(
          'section',
          { className: 'panel', style: { marginTop: 18 } },
          React.createElement('h2', null, 'Gemini Response'),
          React.createElement(
            'div',
            { className: 'tabs' },
            [
              ['families', 'Parsed Families'],
              ['developer', 'Developer Details'],
            ].map(([key, label]) =>
              React.createElement(
                'button',
                {
                  key,
                  className: `tab ${activeTab === key ? 'active' : ''}`,
                  onClick: () => setActiveTab(key),
                },
                label
              )
            )
          ),
          !result
            ? React.createElement('div', { className: 'hint' }, 'Run Gemini to see the family response.')
            : activeTab === 'families'
              ? families.length
                ? React.createElement('div', { className: 'families' }, families.map((family, index) => React.createElement(FamilyCard, { family, index, key: index })))
                : React.createElement('pre', null, pretty(result.parsedFamilies || { message: 'No parsed families found.' }))
              : React.createElement(
                  'div',
                  { className: 'developer-stack' },
                  React.createElement(
                    'details',
                    { className: 'dev-details', open: false },
                    React.createElement('summary', null, 'Raw Gemini text'),
                    React.createElement('pre', null, (result.textParts || []).join('\n\n---\n\n'))
                  ),
                  React.createElement(
                    'details',
                    { className: 'dev-details', open: false },
                    React.createElement('summary', null, 'Raw Gemini envelope'),
                    React.createElement('pre', null, pretty(result.rawEnvelope))
                  ),
                  React.createElement(
                    'details',
                    { className: 'dev-details', open: false },
                    React.createElement('summary', null, 'Full prompt sent to Gemini'),
                    React.createElement('pre', null, result.fullPrompt)
                  )
                )
        ),
        React.createElement(
          'section',
          { className: 'panel', style: { marginTop: 18 } },
          React.createElement(
            'div',
            { className: 'section-head' },
            React.createElement(
              'div',
              null,
              React.createElement('h2', null, 'Saved Results'),
              React.createElement('p', { className: 'subtitle inline-subtitle' }, 'Every Gemini run is saved here with the prompt and returned family names.')
            ),
            React.createElement('button', { className: 'secondary', onClick: refreshSavedRuns }, 'Refresh')
          ),
          savedRunsError ? React.createElement('div', { className: 'error' }, savedRunsError) : null,
          savedRuns.length
            ? React.createElement(
                'div',
                { className: 'saved-table-wrap' },
                React.createElement(
                  'table',
                  { className: 'saved-table' },
                  React.createElement(
                    'thead',
                    null,
                    React.createElement(
                      'tr',
                      null,
                      React.createElement('th', null, 'Run'),
                      React.createElement('th', null, 'Prompt'),
                      React.createElement('th', null, 'Families'),
                      React.createElement('th', null, 'Constraints'),
                      React.createElement('th', null, '')
                    )
                  ),
                  React.createElement(
                    'tbody',
                    null,
                    savedRuns.map((run) =>
                      React.createElement(
                        'tr',
                        { key: run.file },
                        React.createElement(
                          'td',
                          null,
                          React.createElement('div', { className: 'saved-file' }, run.file),
                          React.createElement('div', { className: 'saved-date' }, run.created_at || '-')
                        ),
                        React.createElement('td', { className: 'saved-prompt' }, run.prompt || '-'),
                        React.createElement(
                          'td',
                          null,
                          run.family_names && run.family_names.length
                            ? React.createElement(
                                'div',
                                { className: 'family-chip-row' },
                                run.family_names.map((name, idx) => React.createElement('span', { className: 'family-chip', key: `${name}_${idx}` }, name))
                              )
                            : '-'
                        ),
                        React.createElement('td', null, String(run.constraint_count || 0)),
                        React.createElement(
                          'td',
                          null,
                          React.createElement('button', { className: 'secondary small-button', onClick: () => loadSavedRun(run.file) }, 'Open')
                        )
                      )
                    )
                  )
                )
              )
            : React.createElement('div', { className: 'hint' }, 'No saved runs yet.')
        ),
        React.createElement(
          'section',
          { className: 'panel developer-panel', style: { marginTop: 18 } },
          React.createElement(
            'details',
            { className: 'dev-details' },
            React.createElement('summary', null, 'Developer: planner context and metric constraints'),
            React.createElement(
              'div',
              { className: 'dev-grid' },
              React.createElement(
                'div',
                { className: 'field' },
                React.createElement('label', null, 'Planner context JSON'),
                React.createElement('textarea', {
                  className: 'context-area',
                  value: contextText,
                  onChange: (event) => setContextText(event.target.value),
                }),
                React.createElement('button', { className: 'secondary', onClick: resetContext }, 'Reset context')
              ),
              React.createElement(
                'div',
                { className: 'field' },
                React.createElement('label', null, 'Metric constraints JSON'),
                React.createElement('textarea', {
                  className: 'json-area',
                  value: metricThresholdsText,
                  onChange: (event) => setMetricThresholdsText(event.target.value),
                })
              )
            )
          )
        )
      )
    )
  }

  ReactDOM.createRoot(document.getElementById('root')).render(React.createElement(App))
})()
