@echo off
REM Stop backend (FastAPI)
taskkill /FI "WINDOWTITLE eq Backend"
REM Stop frontend (React)
taskkill /FI "WINDOWTITLE eq Frontend"
