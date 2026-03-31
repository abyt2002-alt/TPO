# AI Testing Streamlit Tool

## Folder
`ia tetsing`

## Run
```powershell
cd "ia tetsing"
pip install -r requirements.txt
streamlit run app.py
```

## Optional Gemini
Set API key before run:
```powershell
$env:GEMINI_API_KEY="YOUR_KEY"
```

If Gemini is unreachable, app automatically uses deterministic fallback generation.
