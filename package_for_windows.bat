@echo off
echo Creating Windows deployment package for HayStack RAG...

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found. Please install Python 3.11+
    pause
    exit /b 1
)

REM Install PyInstaller if not present
python -c "import PyInstaller" >nul 2>&1
if errorlevel 1 (
    echo Installing PyInstaller...
    pip install pyinstaller
)

REM Build frontend if not already built
if not exist "frontend\build" (
    echo Building React frontend...
    cd frontend
    if not exist "node_modules" (
        echo Installing frontend dependencies...
        npm install
    )
    npm run build
    cd ..
) else (
    echo Frontend build found, skipping...
)

REM Create deployment directories
mkdir dist\deployment 2>nul
mkdir dist\deployment\portable 2>nul

echo Building EXE with PyInstaller...
pyinstaller haystack.spec

if errorlevel 1 (
    echo Error: PyInstaller build failed
    pause
    exit /b 1
)

echo Creating portable package...

REM Copy EXE to portable folder
copy dist\HayStackRAG.exe dist\deployment\portable\

REM Copy models if they exist
if exist "models" (
    echo Copying models...
    xcopy models dist\deployment\portable\models\ /E /I /Q
)

REM Copy sample data (optional)
if exist "backend\data" (
    echo Copying sample data...
    xcopy backend\data dist\deployment\portable\data\ /E /I /Q
)

REM Copy Qdrant database if it exists
if exist "backend\qdrant_storage" (
    echo Copying pre-computed embeddings...
    xcopy backend\qdrant_storage dist\deployment\portable\qdrant_storage\ /E /I /Q
)

REM Create config file
echo Creating configuration...
(
echo # HayStack RAG Configuration
echo EMBED_MODEL_PATH=./models/all-MiniLM-L6-v2
echo DATA_DIR=./data
echo QDRANT_PATH=./qdrant_storage
echo PORT=8000
echo HOST=0.0.0.0
) > dist\deployment\portable\config.env

REM Create startup script
(
echo @echo off
echo echo Starting HayStack RAG...
echo echo.
echo echo Checking Ollama installation...
echo ollama --version ^>nul 2^>^&1
echo if errorlevel 1 ^(
echo     echo.
echo     echo ========================================
echo     echo OLLAMA NOT FOUND!
echo     echo ========================================
echo     echo.
echo     echo HayStack RAG requires Ollama to generate AI answers.
echo     echo.
echo     echo To install Ollama:
echo     echo 1. Download from: https://ollama.ai/
echo     echo 2. Run the installer
echo     echo 3. Open new terminal and run: ollama pull llama3
echo     echo 4. Restart this application
echo     echo.
echo     echo Press any key to open Ollama download page...
echo     pause ^>nul
echo     start https://ollama.ai/
echo     exit /b 1
echo ^)
echo.
echo echo Checking Llama3 model...
echo ollama list ^| find "llama3" ^>nul 2^>^&1
echo if errorlevel 1 ^(
echo     echo Llama3 model not found. Downloading... ^(4.7GB^)
echo     ollama pull llama3
echo ^)
echo.
echo echo Starting HayStack RAG server...
echo echo Server will be available at: http://localhost:8000
echo echo.
echo HayStackRAG.exe
echo pause
) > dist\deployment\portable\start.bat

REM Create README
(
echo HayStack RAG - Portable Windows Distribution
echo ==========================================
echo.
echo Quick Start:
echo 1. Install Ollama from https://ollama.ai/
echo 2. Run: ollama pull llama3
echo 3. Double-click start.bat to run the application
echo 4. Open browser to http://localhost:8000
echo.
echo What Ollama Does:
echo - Provides the AI language model ^(Llama3^) for generating answers
echo - Runs completely on your local machine ^(no internet needed after setup^)
echo - Required for the chatbot to work
echo.
echo Ollama Setup:
echo 1. Download Ollama from https://ollama.ai/
echo 2. Install and run: ollama serve
echo 3. Download model: ollama pull llama3 ^(4.7GB download^)
echo 4. Model will be cached locally for future use
echo.
echo File Structure:
echo HayStackRAG.exe       - Main application ^(includes embeddings^)
echo start.bat            - Smart startup script
echo config.env           - Configuration file
echo models/              - Sentence transformer model
echo data/                - Document storage ^(optional^)
echo qdrant_storage/      - Pre-computed vector database
echo.
echo Features:
echo - Instant search using pre-computed embeddings
echo - No need to re-parse PDFs on new machines
echo - Local AI processing with Ollama + Llama3
echo - Web interface for document queries
echo.
echo Troubleshooting:
echo - "Ollama not found": Install from https://ollama.ai/
echo - "Model not found": Run: ollama pull llama3
echo - Permission errors: Run as Administrator
echo - Port conflicts: Check if port 8000 is free
echo.
echo For support: Check README_DEPLOYMENT.md
echo For Ollama help: https://ollama.ai/docs
) > dist\deployment\portable\README.txt

REM Create ZIP package
echo Creating ZIP archive...
powershell -command "Compress-Archive -Path 'dist\deployment\portable\*' -DestinationPath 'dist\deployment\HayStackRAG_Portable.zip' -Force"

echo.
echo ===================================
echo Packaging complete!
echo ===================================
echo.
echo Standalone EXE: dist\HayStackRAG.exe
echo Portable Package: dist\deployment\HayStackRAG_Portable.zip
echo.
echo The portable package includes:
echo - Pre-built executable
echo - AI models and embeddings
echo - Sample documents
echo - Configuration files
echo - Startup scripts
echo.
echo File size estimate:
dir dist\HayStackRAG.exe | find "HayStackRAG.exe"
dir dist\deployment\HayStackRAG_Portable.zip | find "HayStackRAG_Portable.zip"
echo.
echo To distribute: Share the ZIP file
echo To run locally: Execute dist\deployment\portable\start.bat
echo.
pause
