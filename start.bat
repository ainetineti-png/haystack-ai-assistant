@echo off
REM Start backend (FastAPI) using virtual environment
start "Backend" cmd /k "cd backend && C:/HayStack/venv/Scripts/python.exe -m uvicorn main:app --reload --host 127.0.0.1 --port 8001"
REM Start frontend (React)
start "Frontend" cmd /k "cd frontend && npm start"
