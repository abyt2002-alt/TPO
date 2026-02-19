# Troubleshooting Guide

## Common Issues and Solutions

### Setup Issues

#### Issue: `python: command not found`
**Solution:**
- Install Python 3.9 or higher from https://www.python.org/
- Make sure to check "Add Python to PATH" during installation
- Restart your terminal after installation

#### Issue: `npm: command not found`
**Solution:**
- Install Node.js 18+ from https://nodejs.org/
- Restart your terminal after installation
- Verify: `node --version` and `npm --version`

#### Issue: Virtual environment won't activate
**Solution:**
```bash
# Windows
cd backend
python -m venv venv
venv\Scripts\activate

# If you get execution policy error:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Backend Issues

#### Issue: Port 8000 already in use
**Solution:**
```bash
# Find and kill the process using port 8000
# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Or change the port in main.py:
uvicorn.run(app, host="0.0.0.0", port=8001)
```

#### Issue: `ModuleNotFoundError: No module named 'fastapi'`
**Solution:**
```bash
cd backend
venv\Scripts\activate
pip install -r requirements.txt
```

#### Issue: Data files not found
**Solution:**
- Ensure parquet files are in `backend/step3_filtered_engineered/`
- Check file permissions
- Verify file names end with `.parquet`
- Check backend console for exact path being searched

#### Issue: Memory error when loading data
**Solution:**
- Reduce the number of parquet files
- Increase system RAM
- Use data sampling for testing:
```python
# In rfm_service.py, modify load_data():
df = pd.read_parquet(folder_path / file)
df = df.sample(frac=0.1)  # Use 10% of data
```

### Frontend Issues

#### Issue: Port 3000 already in use
**Solution:**
```bash
# Change port in vite.config.js:
server: {
  port: 3001,
  ...
}
```

#### Issue: `npm install` fails
**Solution:**
```bash
# Clear npm cache
npm cache clean --force

# Delete node_modules and package-lock.json
rm -rf node_modules package-lock.json

# Reinstall
npm install
```

#### Issue: Blank page after `npm run dev`
**Solution:**
- Check browser console for errors (F12)
- Verify backend is running at http://localhost:8000
- Check CORS settings in backend
- Clear browser cache

#### Issue: API calls failing (CORS errors)
**Solution:**
- Verify backend CORS settings in `main.py`:
```python
allow_origins=["http://localhost:3000", "http://localhost:5173"]
```
- Check if backend is running
- Verify API URL in frontend `.env` or `api.js`

### Data Issues

#### Issue: No data showing after RFM calculation
**Solution:**
1. Check backend console for errors
2. Verify filters aren't too restrictive
3. Check API response in browser DevTools (Network tab)
4. Verify data format matches expected schema

#### Issue: RFM calculation takes too long
**Solution:**
- First calculation loads data (2-3 seconds)
- Subsequent calculations use cache (<1 second)
- If still slow, check data size and system resources

#### Issue: Incorrect RFM segments
**Solution:**
- Verify recency threshold (default: 90 days)
- Verify frequency threshold (default: 20 order days)
- Check date format in data
- Ensure `Date` column is datetime type

### UI Issues

#### Issue: Table not displaying correctly
**Solution:**
- Check browser console for errors
- Verify data structure matches expected format
- Try different browser
- Clear browser cache

#### Issue: Filters not working
**Solution:**
- Check if backend returns filter options
- Verify API endpoint: http://localhost:8000/api/rfm/filters
- Check browser console for errors
- Ensure data has the filter columns

#### Issue: Export CSV not working
**Solution:**
- Check browser's download settings
- Verify popup blocker isn't blocking download
- Check browser console for errors
- Try different browser

### Performance Issues

#### Issue: Slow initial load
**Solution:**
- First load reads parquet files (normal)
- Subsequent loads use cache (fast)
- Reduce data size for testing
- Check system resources (RAM, CPU)

#### Issue: Slow table rendering
**Solution:**
- Table is paginated (20 items per page)
- Use search to filter results
- Check browser performance
- Close other tabs/applications

#### Issue: High memory usage
**Solution:**
- Backend caches data in memory (expected)
- Restart backend to clear cache
- Reduce data size
- Increase system RAM

## Debugging Tips

### Backend Debugging

1. **Check logs in terminal**
```bash
cd backend
venv\Scripts\activate
python main.py
# Watch for error messages
```

2. **Test API directly**
- Visit http://localhost:8000/docs
- Try endpoints in Swagger UI
- Check response data

3. **Add debug prints**
```python
# In rfm_service.py
print(f"Loaded {len(df)} rows")
print(f"Filters applied: {request}")
```

### Frontend Debugging

1. **Check browser console** (F12)
- Look for red error messages
- Check Network tab for failed requests
- Verify API responses

2. **Check React Query DevTools**
```jsx
// Add to App.jsx
import { ReactQueryDevtools } from '@tanstack/react-query-devtools'

<ReactQueryDevtools initialIsOpen={false} />
```

3. **Add console logs**
```javascript
console.log('Filters:', filters)
console.log('API Response:', data)
```

## Getting Help

### Before Asking for Help

1. ✅ Check this troubleshooting guide
2. ✅ Read error messages carefully
3. ✅ Check browser console (F12)
4. ✅ Check backend terminal output
5. ✅ Try restarting both servers
6. ✅ Verify data files are present

### Information to Provide

When reporting issues, include:
- Operating system and version
- Python version (`python --version`)
- Node.js version (`node --version`)
- Error messages (full text)
- Steps to reproduce
- Screenshots if relevant
- Browser and version (for frontend issues)

### Useful Commands

```bash
# Check Python version
python --version

# Check Node version
node --version
npm --version

# Check if ports are in use
netstat -ano | findstr :8000
netstat -ano | findstr :3000

# Check backend health
curl http://localhost:8000/health

# Check available filters
curl http://localhost:8000/api/rfm/filters

# Reinstall backend dependencies
cd backend
venv\Scripts\activate
pip install --upgrade -r requirements.txt

# Reinstall frontend dependencies
cd frontend
rm -rf node_modules package-lock.json
npm install
```

## Quick Fixes

### "Nothing works!"
```bash
# Nuclear option - restart everything
1. Close all terminals
2. Delete backend/venv
3. Delete frontend/node_modules
4. Run setup.bat
5. Run start.bat
```

### "Backend won't start"
```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

### "Frontend won't start"
```bash
cd frontend
npm install
npm run dev
```

### "Data not loading"
1. Check `backend/step3_filtered_engineered/` exists
2. Verify `.parquet` files are present
3. Check file permissions
4. Restart backend

## Still Having Issues?

1. Check the main `README.md`
2. Review `QUICKSTART.md`
3. Check `ARCHITECTURE.md` for technical details
4. Visit API docs: http://localhost:8000/docs
5. Check GitHub issues (if applicable)
6. Contact the development team

## Prevention Tips

1. **Always activate virtual environment** before running backend
2. **Keep dependencies updated** regularly
3. **Use version control** (git) to track changes
4. **Test after each change** to catch issues early
5. **Keep data backups** before major changes
6. **Document custom changes** for future reference

---

**Remember**: Most issues are simple configuration or setup problems. Take a deep breath, read the error message, and follow the steps above!
