# 🔴 RESTART REQUIRED - White Screen Fix

## The Problem

You're seeing a white screen with errors because:
1. ✅ Frontend code is updated (BarChart3 import fixed)
2. ❌ Backend hasn't been restarted yet
3. ❌ Backend is still looking for old data path
4. ❌ API endpoint returns 404 because data isn't loaded

## The Solution - Restart Both Servers

### Quick Method (Recommended):

1. **Find the two terminal windows** running your servers
2. **Press `Ctrl+C`** in BOTH windows to stop them
3. **Run this command** in your project folder:
   ```bash
   start.bat
   ```

### Manual Method:

**Terminal 1 - Backend:**
```bash
# Press Ctrl+C to stop current backend
cd backend
venv\Scripts\activate
python main.py
```

**Terminal 2 - Frontend:**
```bash
# Press Ctrl+C to stop current frontend
cd frontend
npm run dev
```

## What You Should See After Restart

### Backend Terminal:
```
Loaded 1,234,567 rows from 20 files
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Frontend Terminal:
```
VITE v5.0.11  ready in 500 ms
➜  Local:   http://localhost:3000/
```

### Browser Console (F12):
- ✅ No more 404 errors
- ✅ No more BarChart3 errors
- ✅ RFM page loads correctly

## After Restart - Test It

1. Open: http://localhost:3000
2. Click "RFM Analysis"
3. Page should load with filter panel
4. Click "Calculate RFM"
5. Should work! 🎉

## Still Having Issues?

Check backend terminal for:
- "Loaded X rows from 20 files" ← Should see this
- Any error messages about DATA folder

If you see "Warning: Could not find DATA folder":
- Make sure DATA folder is in project root
- Check that it contains .parquet files

---

**Action Required**: Stop both servers and run `start.bat` now!
