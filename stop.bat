@echo off
echo Stopping RAG Application...

REM Stop backend (FastAPI) - Kill python processes running uvicorn
echo Stopping backend...
taskkill /F /IM python.exe /FI "COMMANDLINE eq *uvicorn*" >nul 2>&1

REM Stop frontend (React) - Kill node processes running react-scripts
echo Stopping frontend...
taskkill /F /IM node.exe /FI "COMMANDLINE eq *react-scripts*" >nul 2>&1

REM Also try to kill by window title as backup
taskkill /F /FI "WINDOWTITLE eq Backend*" >nul 2>&1
taskkill /F /FI "WINDOWTITLE eq Frontend*" >nul 2>&1

echo RAG Application stopped successfully!
pause
