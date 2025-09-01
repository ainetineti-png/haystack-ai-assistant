@echo off
echo ========================================
echo HayStack RAG - Environment Setup
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found! Please install Python 3.11+
    pause
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist "venv\Scripts\activate.bat" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo Error: Failed to create virtual environment
        pause
        exit /b 1
    )
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip and install requirements
echo Installing Python dependencies...
python -m pip install --upgrade pip
pip install -r backend\requirements.txt

if errorlevel 1 (
    echo Error: Failed to install requirements
    pause
    exit /b 1
)

echo.
echo ========================================
echo Python environment setup complete!
echo ========================================
echo.
echo Next steps:
echo 1. Install Ollama from: https://ollama.ai/download
echo 2. Run: ollama pull llama3
echo 3. Run: start.bat to launch the application
echo.
echo Press any key to continue...
pause >nul
