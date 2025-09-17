@echo off
REM setup_models.bat - Windows script to download required models and setup large files

echo 🔧 Setting up HayStack AI Assistant models and data...

REM Create directories
if not exist "models" mkdir models
if not exist "data" mkdir data
if not exist "backend\qdrant_storage" mkdir backend\qdrant_storage

echo 📥 Downloading embedding model (all-MiniLM-L6-v2)...
python -c "from sentence_transformers import SentenceTransformer; import os; model_path = os.path.join('models', 'all-MiniLM-L6-v2'); model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); model.save(model_path); print(f'✅ Model saved to {model_path}')"

echo 🦙 Checking Ollama installation...
ollama --version >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ Ollama found
    echo 📥 Pulling Llama3 model...
    ollama pull llama3:latest
) else (
    echo ❌ Ollama not found. Please install from https://ollama.ai
    echo    Then run: ollama pull llama3:latest
)

echo 📁 Setting up data directory...
if not exist "data\README.md" (
    echo # Data Directory > data\README.md
    echo. >> data\README.md
    echo Place your documents here for processing: >> data\README.md
    echo. >> data\README.md
    echo ## Supported Formats >> data\README.md
    echo - PDF files (*.pdf) >> data\README.md
    echo - Word documents (*.docx, *.doc) >> data\README.md
    echo - Text files (*.txt) >> data\README.md
    echo - Markdown files (*.md) >> data\README.md
    echo. >> data\README.md
    echo ## Privacy Note >> data\README.md
    echo This directory is excluded from Git to protect your documents. >> data\README.md
)

echo ✅ Setup complete!
echo.
echo Next steps:
echo 1. Add your documents to the data\ directory
echo 2. Start the backend: cd backend ^&^& python main.py
echo 3. Start the frontend: cd frontend ^&^& npm start

pause
