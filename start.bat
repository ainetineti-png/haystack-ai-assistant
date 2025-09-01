@echo off
echo ========================================
echo HayStack RAG - Development Startup
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo Error: Virtual environment not found!
    echo Please run setup_env.bat first to create the environment.
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if Ollama is running
echo Checking Ollama status...
tasklist /FI "IMAGENAME eq ollama.exe" | find /I "ollama.exe" >nul
if errorlevel 1 (
    echo Starting Ollama...
    start "Ollama" ollama serve
    echo Waiting for Ollama to start...
    timeout /t 5 >nul
) else (
    echo Ollama is already running.
)

REM Check if llama3 model is available
echo Checking Llama3 model...
ollama list | findstr /I "llama3" >nul
if errorlevel 1 (
    echo Warning: Llama3 model not found!
    echo Please run: ollama pull llama3
    echo.
)

REM Start FastAPI backend
echo Starting backend server...
cd backend
start "Backend" cmd /k "cd /d %CD% && ..\venv\Scripts\activate.bat && python main.py"
cd ..

REM Start React frontend
echo Starting frontend...
cd frontend
start "Frontend" cmd /k "npm start"
cd ..

echo.
echo ========================================
echo All services started!
echo ========================================
echo.
echo Backend: http://localhost:8000
echo Frontend: http://localhost:3000
echo.
echo Check terminal windows for any errors.
echo Press any key to close this window...
pause >nul
