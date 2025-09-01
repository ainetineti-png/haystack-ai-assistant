@echo off
REM Activate venv and install Python requirements
call venv\Scripts\activate.bat
pip install -r backend\requirements.txt

REM Ollama install instructions
echo.
echo Please download and install Ollama from: https://ollama.com/download
echo After installation, continue...

REM Start Ollama if not running
tasklist /FI "IMAGENAME eq ollama.exe" | find /I "ollama.exe" >nul
if errorlevel 1 (
    echo Starting Ollama...
    start "Ollama" ollama serve
    timeout /t 5 >nul
) else (
    echo Ollama is already running.
)

REM Pull Llama3 model only if not present
ollama list | findstr /I "llama3" >nul
if errorlevel 1 (
    echo Pulling Llama3 model... (this may take a while)
    ollama pull llama3
) else (
    echo Llama3 model already available.
)

echo.
echo Setup complete! You can now run start.bat to launch the app.
pause
