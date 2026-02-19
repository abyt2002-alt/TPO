# Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Run Setup
```bash
setup.bat
```
This will:
- Create Python virtual environment
- Install all Python dependencies
- Install all Node.js dependencies

### Step 2: Add Your Data
Place your parquet files in:
```
backend/step3_filtered_engineered/
```

### Step 3: Start the Application
```bash
start.bat
```

The application will open:
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## 📊 Using the RFM Analysis

1. **Navigate to RFM Analysis** from the dashboard
2. **Select Filters**:
   - Choose states (MAH, UP)
   - Select categories, subcategories, brands, sizes (optional)
   - Adjust recency threshold (default: 90 days)
   - Adjust frequency threshold (default: 20 order days)
3. **Click "Calculate RFM"**
4. **View Results**:
   - Summary metrics
   - 8-segment grid visualization
   - Cluster summaries
   - Detailed outlet table
5. **Export Data**: Click "Export CSV" to download results

## 🎯 RFM Segments Explained

### Best Customers
- **Recent-High-High** 🌟: Active, frequent, high-value customers

### Good Customers
- **Recent-High-Low** ✨: Active and frequent, but lower value
- **Recent-Low-High** ✨: Active and high-value, but less frequent

### At-Risk Customers
- **Stale-High-High** ⏰: Previously great, now inactive
- **Stale-High-Low** ⏰: Previously frequent, now inactive

### Lost Customers
- **Stale-Low-Low** ⚠️: Inactive, infrequent, low-value

## 🔧 Troubleshooting

### Backend won't start
- Check if Python 3.9+ is installed: `python --version`
- Ensure virtual environment is activated
- Check if port 8000 is available

### Frontend won't start
- Check if Node.js 18+ is installed: `node --version`
- Delete `node_modules` and run `npm install` again
- Check if port 3000 is available

### No data showing
- Verify parquet files are in `backend/step3_filtered_engineered/`
- Check backend console for error messages
- Verify data format matches expected schema

## 📝 Manual Setup (Alternative)

### Backend
```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

## 🌐 API Testing

Visit http://localhost:8000/docs for interactive API documentation (Swagger UI)

Test endpoints:
- GET `/health` - Check if API is running
- GET `/api/rfm/filters` - Get available filter options
- POST `/api/rfm/calculate` - Calculate RFM metrics

## 📦 What's Included

### Phase 1 (Current)
✅ RFM Analysis
✅ Multi-filter selection
✅ 8-segment visualization
✅ Cluster analysis
✅ Outlet details table
✅ CSV export

### Coming Soon
🔜 Discount Analysis
🔜 ROI Calculator
🔜 Promo Planner

## 💡 Tips

1. **Start with all states selected** to see the full picture
2. **Use filters to drill down** into specific segments
3. **Export data** for further analysis in Excel
4. **Check cluster summaries** to understand segment boundaries
5. **Sort outlet table** by different columns to find insights

## 🆘 Need Help?

- Check the main README.md for detailed documentation
- Review API docs at http://localhost:8000/docs
- Check browser console for frontend errors
- Check terminal/command prompt for backend errors

## 🎉 You're Ready!

Start exploring your outlet data with powerful RFM segmentation!
