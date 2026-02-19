# Architecture Overview

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Frontend                             │
│                    (React + Vite)                            │
│                   Port: 3000                                 │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Dashboard  │  │ RFM Analysis │  │ Future Pages │      │
│  │     Page     │  │     Page     │  │  (Discount,  │      │
│  │              │  │              │  │   ROI, etc)  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              React Components                        │    │
│  │  - FilterPanel  - SegmentGrid  - OutletTable       │    │
│  │  - RFMSummary   - ClusterSummary                   │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │           Services (API Client)                      │    │
│  │  - Axios HTTP Client                                │    │
│  │  - React Query for caching                          │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                               │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        │ HTTP/REST API
                        │ (JSON)
                        │
┌───────────────────────▼─────────────────────────────────────┐
│                         Backend                              │
│                    (FastAPI + Python)                        │
│                      Port: 8000                              │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              API Endpoints                           │    │
│  │  POST /api/rfm/calculate                            │    │
│  │  GET  /api/rfm/filters                              │    │
│  │  GET  /api/rfm/segments                             │    │
│  │  GET  /health                                        │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │           Pydantic Models                            │    │
│  │  - RFMRequest   - RFMResponse                       │    │
│  │  - OutletRFM    - SegmentSummary                    │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Services Layer                          │    │
│  │  - RFMService (Business Logic)                      │    │
│  │    • Data loading & caching                         │    │
│  │    • RFM calculation                                │    │
│  │    • Clustering (K-means)                           │    │
│  │    • Filtering & aggregation                        │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │          Data Processing Libraries                   │    │
│  │  - Pandas (DataFrames)                              │    │
│  │  - NumPy (Numerical operations)                     │    │
│  │  - Scikit-learn (K-means clustering)                │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                               │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        │ File I/O
                        │
┌───────────────────────▼─────────────────────────────────────┐
│                      Data Storage                            │
│                  (Parquet Files)                             │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  step3_filtered_engineered/                                  │
│    ├── file1.parquet                                         │
│    ├── file2.parquet                                         │
│    └── ...                                                   │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow

### RFM Calculation Flow

```
1. User Input (Frontend)
   ↓
2. Filter Selection
   - States, Categories, Brands, etc.
   - Recency & Frequency thresholds
   ↓
3. API Request (POST /api/rfm/calculate)
   ↓
4. Backend Processing
   a. Load parquet files (cached)
   b. Apply filters
   c. Aggregate order-level data
   d. Calculate RFM metrics:
      - Recency: Days since last order
      - Frequency: Unique order days
      - Monetary: Average Order Value (AOV)
   e. K-means clustering for Monetary
   f. Create 8 RFM segments (2×2×2)
   g. Calculate segment summaries
   ↓
5. API Response (JSON)
   - RFM data for each outlet
   - Segment summaries
   - Cluster summaries
   - Metadata
   ↓
6. Frontend Rendering
   - Summary metrics
   - 8-segment grid
   - Cluster tables
   - Outlet details table
```

## Component Hierarchy

```
App
├── Layout
│   ├── Header (Navigation)
│   ├── Main Content
│   │   └── Routes
│   │       ├── Dashboard
│   │       │   └── Feature Cards
│   │       └── RFMAnalysis
│   │           ├── FilterPanel
│   │           │   ├── MultiSelect (States)
│   │           │   ├── MultiSelect (Categories)
│   │           │   ├── MultiSelect (Subcategories)
│   │           │   ├── MultiSelect (Brands)
│   │           │   ├── MultiSelect (Sizes)
│   │           │   └── Configuration Inputs
│   │           ├── RFMSummary
│   │           │   ├── Metric Cards
│   │           │   └── Stats Grid
│   │           ├── ClusterSummary
│   │           │   ├── Frequency Table
│   │           │   └── Monetary Table
│   │           ├── SegmentGrid
│   │           │   └── Segment Cards (×8)
│   │           └── OutletTable
│   │               ├── Search Bar
│   │               ├── Data Table
│   │               └── Pagination
│   └── Footer
```

## Technology Stack Details

### Frontend Stack
```
React 18.2
├── Vite 5.0 (Build tool)
├── React Router 6.21 (Routing)
├── TailwindCSS 3.4 (Styling)
├── React Query 5.17 (Data fetching & caching)
├── Axios 1.6 (HTTP client)
└── Lucide React 0.309 (Icons)
```

### Backend Stack
```
FastAPI 0.109
├── Uvicorn 0.27 (ASGI server)
├── Pydantic 2.5 (Data validation)
├── Pandas 2.1 (Data manipulation)
├── NumPy 1.26 (Numerical computing)
├── Scikit-learn 1.4 (Machine learning)
└── PyArrow 14.0 (Parquet support)
```

## API Design Principles

1. **RESTful**: Standard HTTP methods (GET, POST)
2. **JSON**: All data exchanged in JSON format
3. **Validation**: Pydantic models for request/response validation
4. **Documentation**: Auto-generated OpenAPI/Swagger docs
5. **CORS**: Enabled for frontend communication
6. **Error Handling**: Consistent error responses

## Performance Optimizations

### Backend
- **Data Caching**: Parquet files loaded once and cached in memory
- **Efficient Filtering**: Pandas operations for fast data filtering
- **Vectorized Operations**: NumPy for numerical computations
- **Async Support**: FastAPI async endpoints (ready for future use)

### Frontend
- **React Query**: Automatic caching and background refetching
- **Code Splitting**: Vite's automatic code splitting
- **Lazy Loading**: Components loaded on demand
- **Memoization**: useMemo for expensive computations
- **Virtual Scrolling**: Ready for large datasets

## Security Considerations

1. **CORS**: Configured for specific origins
2. **Input Validation**: Pydantic models validate all inputs
3. **Error Messages**: No sensitive information in error responses
4. **File Access**: Restricted to specific data directories
5. **Rate Limiting**: Ready to implement if needed

## Scalability Path

### Current (Phase 1)
- Single-server deployment
- In-memory data caching
- Synchronous processing

### Future Enhancements
- **Database**: PostgreSQL for persistent storage
- **Caching**: Redis for distributed caching
- **Queue**: Celery for background tasks
- **Load Balancing**: Multiple backend instances
- **CDN**: Static asset delivery
- **Containerization**: Docker deployment
- **Orchestration**: Kubernetes for scaling

## Deployment Architecture

```
Production Environment:

┌─────────────────────────────────────────┐
│           Load Balancer / CDN           │
└────────────┬────────────────────────────┘
             │
    ┌────────┴────────┐
    │                 │
┌───▼────┐      ┌────▼───┐
│Frontend│      │Backend │
│(Nginx) │      │(Uvicorn│
│        │      │Workers)│
└────────┘      └────┬───┘
                     │
              ┌──────▼──────┐
              │  Database   │
              │(PostgreSQL) │
              └─────────────┘
```

## Future Modules Integration

The architecture is designed to easily add new modules:

1. **Discount Analysis Module**
   - New endpoints: `/api/discount/*`
   - New service: `DiscountService`
   - New frontend pages & components

2. **ROI Calculator Module**
   - New endpoints: `/api/roi/*`
   - New service: `ROIService`
   - Reuse RFM data for calculations

3. **Promo Planner Module**
   - New endpoints: `/api/planner/*`
   - New service: `PlannerService`
   - Interactive calendar component

Each module follows the same pattern:
- API endpoints in `main.py`
- Pydantic models in `models/`
- Business logic in `services/`
- Frontend pages in `pages/`
- Reusable components in `components/`
