@echo off
echo ========================================
echo Outlet Analysis Tool - Setup Script
echo ========================================
echo.

echo [1/4] Setting up Python virtual environment...
cd backend
python -m venv venv
call venv\Scripts\activate
echo.

echo [2/4] Installing Python dependencies...
pip install -r requirements.txt
echo.

cd ..

echo [3/4] Installing Node.js dependencies...
cd frontend
call npm install
echo.

cd ..

echo [4/4] Setup complete!
echo.
echo ========================================
echo Next Steps:
echo ========================================
echo.
echo 1. Place your data files in: backend/step3_filtered_engineered/
echo 2. Run: start.bat
echo.
echo Or manually:
echo   Backend: cd backend && venv\Scripts\activate && python main.py
echo   Frontend: cd frontend && npm run dev
echo.
pause
