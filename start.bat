@echo off
echo Starting Ollama RAG Application...

REM Activate virtual environment
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo Warning: Virtual environment not found at venv\Scripts\activate.bat
)

REM Start Ollama (if not running)
tasklist /FI "IMAGENAME eq ollama.exe" | find /I "ollama.exe" >nul
if errorlevel 1 (
    echo Starting Ollama...
    start "Ollama" ollama serve
    echo Waiting for Ollama to start...
    timeout /t 5 >nul
) else (
    echo Ollama is already running.
)

REM Test Ollama connection
echo Testing Ollama connection...
timeout /t 2 >nul

REM Start FastAPI backend with venv
echo Starting backend...
cd backend
start "Backend" cmd /k "cd /d %CD% && ..\venv\Scripts\activate.bat && python main.py"
cd ..

REM Start React frontend
echo Starting frontend...
cd frontend
start "Frontend" cmd /k "npm start"
cd ..

echo All services started. Check terminal windows for any errors.
