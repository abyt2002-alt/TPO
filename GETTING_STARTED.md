# Getting Started - Your First Steps

## 🎉 Welcome!

You now have a complete, scalable FastAPI + React application for outlet analysis! This guide will help you get started quickly.

## ⚡ Quick Start (3 Steps)

### Step 1: Run Setup
Open Command Prompt in the project folder and run:
```bash
setup.bat
```
This installs all dependencies (takes 2-3 minutes).

### Step 2: Add Your Data
Copy your parquet files to:
```
backend/step3_filtered_engineered/
```

### Step 3: Start the App
```bash
start.bat
```

That's it! The app will open at:
- Frontend: http://localhost:3000
- Backend: http://localhost:8000/docs

## 📚 What to Read Next

### For Quick Usage
1. **QUICKSTART.md** - How to use the RFM analysis
2. **TROUBLESHOOTING.md** - If something goes wrong

### For Understanding
1. **README.md** - Complete documentation
2. **ARCHITECTURE.md** - How it works
3. **PROJECT_SUMMARY.md** - What was built

### For Development
1. **FILE_STRUCTURE.md** - Where everything is
2. **ARCHITECTURE.md** - System design

## 🎯 Your First RFM Analysis

1. Open http://localhost:3000
2. Click "RFM Analysis" in the navigation
3. Select filters (or leave default)
4. Click "Calculate RFM"
5. Explore the results!

## 🔍 What You'll See

### Dashboard
- Overview of features
- Quick stats
- Navigation to modules

### RFM Analysis Page
- **Filter Panel**: Select your data filters
- **Summary Metrics**: Key statistics
- **Segment Grid**: 8 RFM segments with color coding
- **Cluster Summary**: Frequency and monetary ranges
- **Outlet Table**: Detailed data with search, sort, export

## 🎨 Understanding RFM Segments

The 8 segments are color-coded:

🌟 **Green** - Best customers (Recent-High-High)
- Active, frequent, high-value
- Focus on retention

✨ **Blue** - Good customers (Recent-*)
- Active but varying frequency/value
- Opportunities for upselling

⏰ **Yellow** - At-risk customers (Stale-High-*)
- Previously good, now inactive
- Re-engagement campaigns

⚠️ **Red** - Lost customers (Stale-Low-Low)
- Inactive, infrequent, low-value
- Win-back or deprioritize

## 🛠️ Common Tasks

### Export Data
Click "Export CSV" button in the outlet table

### Search Outlets
Use the search box above the outlet table

### Sort Data
Click column headers in the outlet table

### Filter Results
Use the multi-select dropdowns in the filter panel

### Adjust Thresholds
- Recency: Days to consider "Recent" (default: 90)
- Frequency: Order days for "High" frequency (default: 20)

## 🚀 Next Steps

### Explore the Data
- Try different filter combinations
- Compare segments
- Identify patterns

### Understand the API
- Visit http://localhost:8000/docs
- Try the interactive API documentation
- Test endpoints directly

### Plan Your Strategy
- Identify high-value segments
- Plan targeted campaigns
- Track segment changes over time

## 📊 Key Metrics Explained

### Recency
- Days since last order
- Lower is better
- Threshold: 90 days (configurable)

### Frequency
- Unique order days
- Higher is better
- Threshold: 20 days (configurable)

### Monetary
- Average Order Value (AOV)
- Higher is better
- Clustered automatically

### Market Share
- Percentage of total sales value
- Shows segment importance

## 💡 Pro Tips

1. **Start broad, then narrow**: Begin with all states, then filter down
2. **Compare segments**: Look at market share vs outlet count
3. **Check cluster ranges**: Understand what "High" and "Low" mean
4. **Export for analysis**: Use CSV export for deeper analysis in Excel
5. **Monitor trends**: Run analysis regularly to track changes

## 🔧 Development Tips

### Backend Development
```bash
cd backend
venv\Scripts\activate
python main.py
```
- Edit files in `backend/`
- Server auto-reloads on changes
- Check terminal for errors

### Frontend Development
```bash
cd frontend
npm run dev
```
- Edit files in `frontend/src/`
- Browser auto-refreshes on changes
- Check browser console for errors

### API Testing
- Visit http://localhost:8000/docs
- Use Swagger UI to test endpoints
- Check request/response formats

## 📖 Learning Path

### Day 1: Get Familiar
- Run the app
- Try RFM analysis
- Explore different filters
- Export some data

### Day 2: Understand the Code
- Read ARCHITECTURE.md
- Explore backend code
- Explore frontend components
- Check API documentation

### Day 3: Customize
- Adjust thresholds
- Try different data
- Modify UI colors
- Add custom filters

### Week 2: Extend
- Plan Phase 2 (Discount Analysis)
- Design new features
- Add custom reports
- Integrate with other systems

## 🎓 Resources

### Documentation
- All `.md` files in project root
- Inline code comments
- API documentation at /docs

### Technologies
- FastAPI: https://fastapi.tiangolo.com/
- React: https://react.dev/
- TailwindCSS: https://tailwindcss.com/
- Pandas: https://pandas.pydata.org/

### Community
- FastAPI Discord
- React Community
- Stack Overflow

## ✅ Checklist

Before you start:
- [ ] Python 3.9+ installed
- [ ] Node.js 18+ installed
- [ ] Data files ready
- [ ] Ran setup.bat
- [ ] Both servers running

First analysis:
- [ ] Opened http://localhost:3000
- [ ] Navigated to RFM Analysis
- [ ] Selected filters
- [ ] Clicked Calculate RFM
- [ ] Explored results
- [ ] Exported CSV

Understanding:
- [ ] Read QUICKSTART.md
- [ ] Checked API docs
- [ ] Explored code structure
- [ ] Tried different filters

## 🆘 Need Help?

1. **Check TROUBLESHOOTING.md** for common issues
2. **Check browser console** (F12) for frontend errors
3. **Check terminal** for backend errors
4. **Review documentation** in project root
5. **Test API directly** at /docs endpoint

## 🎉 You're Ready!

You now have:
✅ A working FastAPI backend
✅ A modern React frontend
✅ Complete RFM analysis
✅ Comprehensive documentation
✅ A scalable architecture

Start exploring your outlet data and discover insights!

---

**Remember**: This is Phase 1 (RFM Analysis). More features coming in future phases!

**Have fun analyzing! 🚀**
