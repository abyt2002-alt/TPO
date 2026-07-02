# AI Monte Carlo Gemini Family Tester

This is a standalone React testing environment for Stage 1 only.

It does not change or run the production app logic. It only builds the Gemini family-generation prompt, sends it to Gemini through the local test server, displays the raw response, and saves each run under `outputs/`.

## Run

```powershell
cd "C:\Users\abqua\Desktop\HRI app\ai-montecarlo tetsing"
npm start
```

Then open:

```text
http://127.0.0.1:8510
```

Set an API key before running. The key and Gemini generation settings are intentionally hidden from the UI:

```powershell
$env:GEMINI_API_KEY="your-key"
```

Optional:

```powershell
$env:GEMINI_MODEL="gemini-2.5-flash"
```

## Current Scope

Current stage: Gemini intent understanding and family creation.

This tool validates:

```text
User prompt + constraints + planner context
-> Gemini
-> 3 scenario families
```

It does not validate Monte Carlo sampling, final scenario business metrics, backend repair logic, or frontend behavior.
