@echo off
setlocal
cd /d "%~dp0.."
set "PROJECT_PY=python"
if exist ".venv\Scripts\python.exe" set "PROJECT_PY=.venv\Scripts\python.exe"
if not exist "databases\student_management.db" (
    "%PROJECT_PY%" databases\create_student_db.py
    if errorlevel 1 exit /b 1
)
"%PROJECT_PY%" -m streamlit run app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
