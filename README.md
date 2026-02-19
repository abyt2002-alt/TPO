# Outlet Analysis Tool - FastAPI + React

A scalable, modern web application for retail outlet performance analysis using RFM (Recency, Frequency, Monetary) segmentation.

## Architecture

- **Backend**: FastAPI (Python)
- **Frontend**: React + Vite + TailwindCSS
- **Data Processing**: Pandas, NumPy, Scikit-learn

## Project Structure

```
.
├── backend/
│   ├── main.py                 # FastAPI application entry point
│   ├── models/
│   │   └── rfm_models.py      # Pydantic models for RFM
│   ├── services/
│   │   └── rfm_service.py     # RFM business logic
│   └── requirements.txt        # Python dependencies
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Layout.jsx
│   │   │   └── rfm/           # RFM-specific components
│   │   ├── pages/
│   │   │   ├── Dashboard.jsx
│   │   │   └── RFMAnalysis.jsx
│   │   ├── services/
│   │   │   └── api.js         # API client
│   │   ├── App.jsx
│   │   └── main.jsx
│   ├── package.json
│   └── vite.config.js
│
└── outlet_analysis_tool.py    # Original Streamlit app (reference)
```

## Setup Instructions

### Prerequisites

- Python 3.9+
- Node.js 18+
- npm or yarn

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:
```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Place your data files in the `step3_filtered_engineered` folder (same location as original Streamlit app)

6. Run the backend server:
```bash
python main.py
```

The API will be available at `http://localhost:8000`

API Documentation: `http://localhost:8000/docs`

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Create a `.env` file (optional):
```env
VITE_API_URL=http://localhost:8000
```

4. Run the development server:
```bash
npm run dev
```

The app will be available at `http://localhost:3000`

## Features Implemented (Phase 1: RFM)

### Backend
- ✅ FastAPI REST API with automatic documentation
- ✅ RFM calculation endpoint with filtering
- ✅ Available filters endpoint
- ✅ Pydantic models for request/response validation
- ✅ Service layer for business logic
- ✅ CORS middleware for frontend communication

### Frontend
- ✅ Modern React with Vite
- ✅ TailwindCSS for styling
- ✅ React Query for data fetching
- ✅ React Router for navigation
- ✅ Responsive design
- ✅ RFM filter panel with multi-select
- ✅ RFM summary metrics
- ✅ 8-segment grid visualization
- ✅ Cluster summary tables
- ✅ Outlet data table with search, sort, and pagination
- ✅ CSV export functionality

## API Endpoints

### RFM Endpoints

#### POST `/api/rfm/calculate`
Calculate RFM metrics for outlets

**Request Body:**
```json
{
  "states": ["MAH", "UP"],
  "categories": [],
  "subcategories": [],
  "brands": [],
  "sizes": [],
  "recency_threshold": 90,
  "frequency_threshold": 20
}
```

**Response:**
```json
{
  "success": true,
  "message": "RFM calculation completed successfully",
  "total_outlets": 150,
  "input_rows": 50000,
  "input_outlets": 150,
  "rfm_data": [...],
  "segment_summary": [...],
  "cluster_summary": {...},
  "max_date": "2024-01-01T00:00:00"
}
```

#### GET `/api/rfm/filters`
Get available filter options

**Response:**
```json
{
  "states": ["MAH", "UP"],
  "categories": ["Category1", "Category2"],
  "subcategories": [...],
  "brands": [...],
  "sizes": [...]
}
```

#### GET `/api/rfm/segments`
Get all RFM segment definitions

## Next Steps (Future Phases)

### Phase 2: Discount Analysis
- Base discount estimation
- Two-stage OLS regression
- Discount effectiveness visualization

### Phase 3: ROI Calculation
- Structural ROI
- Tactical ROI
- Profit ROI with COGS

### Phase 4: Promo Planner
- 12-month calendar planner
- Interactive scenario planning
- Impact calculations

## Development

### Backend Development
```bash
cd backend
python main.py
```

### Frontend Development
```bash
cd frontend
npm run dev
```

### Build for Production

**Backend:**
```bash
# Use gunicorn or uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000
```

**Frontend:**
```bash
cd frontend
npm run build
# Output will be in frontend/dist
```

## Technology Stack

### Backend
- FastAPI - Modern Python web framework
- Pandas - Data manipulation
- NumPy - Numerical computing
- Scikit-learn - Machine learning (K-means clustering)
- Pydantic - Data validation
- Uvicorn - ASGI server

### Frontend
- React 18 - UI library
- Vite - Build tool
- TailwindCSS - Utility-first CSS
- React Query - Data fetching
- React Router - Routing
- Axios - HTTP client
- Lucide React - Icons

## License

Proprietary - All rights reserved

## Support

For issues or questions, please contact the development team.
