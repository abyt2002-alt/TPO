# Complete File Structure

```
outlet-analysis-tool/
│
├── 📁 backend/                          # FastAPI Backend
│   ├── 📁 models/                       # Pydantic data models
│   │   ├── __init__.py
│   │   └── rfm_models.py               # RFM request/response models
│   │
│   ├── 📁 services/                     # Business logic layer
│   │   ├── __init__.py
│   │   └── rfm_service.py              # RFM calculation service
│   │
│   ├── 📁 step3_filtered_engineered/   # Data folder (create this)
│   │   ├── file1.parquet               # Your data files
│   │   ├── file2.parquet
│   │   └── ...
│   │
│   ├── 📁 venv/                         # Python virtual environment (created by setup)
│   │
│   ├── __init__.py
│   ├── main.py                          # FastAPI app entry point
│   └── requirements.txt                 # Python dependencies
│
├── 📁 frontend/                         # React Frontend
│   ├── 📁 public/                       # Static assets
│   │
│   ├── 📁 src/                          # Source code
│   │   ├── 📁 components/              # Reusable components
│   │   │   ├── 📁 rfm/                 # RFM-specific components
│   │   │   │   ├── ClusterSummary.jsx  # Cluster tables
│   │   │   │   ├── FilterPanel.jsx     # Filter controls
│   │   │   │   ├── OutletTable.jsx     # Outlet data table
│   │   │   │   ├── RFMSummary.jsx      # Summary metrics
│   │   │   │   └── SegmentGrid.jsx     # 8-segment grid
│   │   │   │
│   │   │   └── Layout.jsx              # App layout wrapper
│   │   │
│   │   ├── 📁 pages/                   # Page components
│   │   │   ├── Dashboard.jsx           # Home/dashboard page
│   │   │   └── RFMAnalysis.jsx         # RFM analysis page
│   │   │
│   │   ├── 📁 services/                # API services
│   │   │   └── api.js                  # Axios API client
│   │   │
│   │   ├── App.jsx                     # Main app component
│   │   ├── index.css                   # Global styles
│   │   └── main.jsx                    # React entry point
│   │
│   ├── 📁 node_modules/                # Node dependencies (created by npm install)
│   │
│   ├── .env.example                    # Environment variables template
│   ├── index.html                      # HTML entry point
│   ├── package.json                    # Node dependencies
│   ├── postcss.config.js               # PostCSS config
│   ├── tailwind.config.js              # TailwindCSS config
│   └── vite.config.js                  # Vite build config
│
├── 📄 .gitignore                       # Git ignore rules
├── 📄 ARCHITECTURE.md                  # System architecture docs
├── 📄 FILE_STRUCTURE.md                # This file
├── 📄 PROJECT_SUMMARY.md               # Project overview
├── 📄 QUICKSTART.md                    # Quick start guide
├── 📄 README.md                        # Main documentation
├── 📄 TROUBLESHOOTING.md               # Troubleshooting guide
├── 📄 outlet_analysis_tool.py          # Original Streamlit app (reference)
├── 📄 setup.bat                        # Setup script (Windows)
└── 📄 start.bat                        # Start script (Windows)
```

## File Descriptions

### Backend Files

#### `backend/main.py`
- FastAPI application setup
- API endpoint definitions
- CORS middleware configuration
- Server startup code

#### `backend/models/rfm_models.py`
- `RFMRequest` - Input validation model
- `RFMResponse` - API response model
- `OutletRFM` - Individual outlet data model
- `SegmentSummary` - Segment statistics model
- `ClusterSummary` - Cluster range model

#### `backend/services/rfm_service.py`
- `RFMService` compatibility facade (thin orchestrator)
- Keeps public service method names used by `backend/main.py`
- Delegates implementation to modular step/core mixins

#### `backend/services/core/`
- `context.py`: shared runtime context and dependency initialization
- `state_store.py`: run state DB create/read/write and JSON-safe merge
- `data_loader.py`: parquet loading and cache bootstrap
- `scope_builder.py`: common scope/filter/slab normalization utilities
- `shared_math.py`: reusable math helpers and constrained ridge class
- `scenario_compare.py`: scenario upload and compare logic
- `eda_service.py`: EDA options/scope/overview helpers

#### `backend/services/steps/`
- `step1_segmentation.py`: Step 1 RFM segmentation flow
- `step2_discount.py`: Step 2 slab/base-depth and summary logic
- `step3_modeling.py`: Step 3 stage-1/stage-2 modeling and ROI
- `step4_cross_size_planner.py`: Step 4 cross-size planner logic
- `step5_baseline_forecast.py`: Step 5 baseline forecast logic

#### `backend/requirements.txt`
```
fastapi==0.109.0
uvicorn[standard]==0.27.0
pandas==2.1.4
numpy==1.26.3
scikit-learn==1.4.0
pyarrow==14.0.2
pydantic==2.5.3
python-multipart==0.0.6
```

### Frontend Files

#### `frontend/src/App.jsx`
- Main application component
- Router setup
- Layout wrapper

#### `frontend/src/pages/Dashboard.jsx`
- Home page
- Feature cards
- Overview statistics
- Navigation to modules

#### `frontend/src/pages/RFMAnalysis.jsx`
- RFM analysis workflow
- State management
- Component orchestration
- API integration

#### `frontend/src/components/Layout.jsx`
- Header with navigation
- Footer
- Page wrapper
- Responsive layout

#### `frontend/src/components/rfm/FilterPanel.jsx`
- Multi-select filters
- Configuration inputs
- Calculate button
- Collapsible panel

#### `frontend/src/components/rfm/RFMSummary.jsx`
- Key metrics display
- Summary statistics
- Data info

#### `frontend/src/components/rfm/SegmentGrid.jsx`
- 8-segment visualization
- Color-coded cards
- State breakdown
- Market share display

#### `frontend/src/components/rfm/ClusterSummary.jsx`
- Frequency cluster table
- Monetary cluster table
- Collapsible view

#### `frontend/src/components/rfm/OutletTable.jsx`
- Searchable data table
- Sortable columns
- Pagination
- CSV export

#### `frontend/src/services/api.js`
- Axios HTTP client
- API endpoint functions
- Base URL configuration

#### `frontend/package.json`
```json
{
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "axios": "^1.6.5",
    "react-router-dom": "^6.21.3",
    "recharts": "^2.10.4",
    "lucide-react": "^0.309.0",
    "@tanstack/react-query": "^5.17.19"
  }
}
```

### Documentation Files

#### `README.md`
- Complete project documentation
- Setup instructions
- API documentation
- Feature list
- Technology stack

#### `QUICKSTART.md`
- Quick setup guide
- Usage instructions
- Troubleshooting basics
- Tips and tricks

#### `ARCHITECTURE.md`
- System architecture diagrams
- Data flow
- Component hierarchy
- Technology stack details
- Scalability path

#### `PROJECT_SUMMARY.md`
- Project overview
- What was built
- Feature comparison
- Success criteria
- Next steps

#### `TROUBLESHOOTING.md`
- Common issues
- Solutions
- Debugging tips
- Quick fixes

#### `FILE_STRUCTURE.md`
- This file
- Complete file tree
- File descriptions

### Setup Files

#### `setup.bat`
- Creates Python virtual environment
- Installs Python dependencies
- Installs Node.js dependencies
- One-time setup script

#### `start.bat`
- Starts backend server
- Starts frontend server
- Opens in separate terminals
- Convenient development startup

#### `.gitignore`
- Python cache files
- Node modules
- Virtual environment
- Data files
- Environment variables
- Build artifacts

### Reference Files

#### `outlet_analysis_tool.py`
- Original Streamlit application
- Reference for business logic
- Not used in new app
- Keep for comparison

## Directory Purposes

### `backend/models/`
**Purpose**: Data validation and serialization
- Pydantic models for type safety
- Request/response schemas
- API contract definitions

### `backend/services/`
**Purpose**: Business logic layer
- Thin facade + modular core/step composition
- Step-isolated logic for faster changes and lower merge conflicts
- Reusable shared utilities in `core/`

### `frontend/src/components/`
**Purpose**: Reusable UI components
- Presentational components
- Shared across pages
- Modular and testable

### `frontend/src/pages/`
**Purpose**: Route-level components
- Full page views
- Route handlers
- Page-specific logic

### `frontend/src/services/`
**Purpose**: External service integration
- API clients
- Data fetching
- Service abstractions

## File Naming Conventions

### Backend
- **Snake case**: `rfm_service.py`, `rfm_models.py`, `step2_discount.py`
- **Descriptive**: Names indicate purpose
- **Grouped**: Related files in `services/core` and `services/steps`

### Frontend
- **PascalCase**: `RFMAnalysis.jsx`, `FilterPanel.jsx`
- **Component names**: Match component function
- **Folder structure**: Organized by feature

## Adding New Files

### Adding a Backend Endpoint
1. Create model in `backend/models/`
2. Implement logic in `backend/services/steps/` or `backend/services/core/`
3. Add endpoint in `backend/main.py`

### Adding a Frontend Page
1. Create page in `frontend/src/pages/`
2. Add route in `frontend/src/App.jsx`
3. Add navigation in `frontend/src/components/Layout.jsx`

### Adding a Component
1. Create component in `frontend/src/components/`
2. Import and use in page
3. Add to appropriate subfolder if feature-specific

## File Size Guidelines

- **Keep files under 500 lines** for maintainability
- **Split large components** into smaller ones
- **Extract reusable logic** into services
- **Use composition** over large monolithic files

## Import Patterns

### Backend
```python
# Absolute imports from project root
from models.rfm_models import RFMRequest
from services.rfm_service import RFMService
```

### Frontend
```javascript
// Relative imports
import Layout from './components/Layout'
import { calculateRFM } from './services/api'
```

## Configuration Files

### Backend Configuration
- `requirements.txt` - Python dependencies
- `main.py` - App configuration

### Frontend Configuration
- `package.json` - Node dependencies
- `vite.config.js` - Build configuration
- `tailwind.config.js` - Style configuration
- `postcss.config.js` - CSS processing

## Data Files Location

Place your parquet files in:
```
backend/step3_filtered_engineered/
```

This folder is:
- ✅ Gitignored (won't be committed)
- ✅ Automatically searched by backend
- ✅ Can contain multiple parquet files
- ✅ Files are loaded and cached on startup

## Build Artifacts

### Backend
- `__pycache__/` - Python bytecode cache
- `venv/` - Virtual environment

### Frontend
- `node_modules/` - Node dependencies
- `dist/` - Production build output

All build artifacts are gitignored.

---

**Note**: This structure is designed for scalability. As you add more features (Discount Analysis, ROI, Planner), follow the same pattern for consistency.
