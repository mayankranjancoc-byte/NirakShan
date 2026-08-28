@echo off
echo Starting NirakShan Border Screening Dashboard...
cd /d "%~dp0backend"
.\.venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
pause
