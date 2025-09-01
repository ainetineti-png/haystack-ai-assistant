# HayStack RAG Application

A Retrieval-Augmented Generation (RAG) application built with FastAPI, React, and Ollama.

## 🚀 Quick Start

1. **Install Python 3.11+**
2. **Run `setup_env.bat`** to create virtual environment
3. **Install Ollama** from https://ollama.ai/
4. **Download model**: `ollama pull llama3`
5. **Run `start.bat`** to launch the application

## 📁 Project Structure

```
HayStack/
├── backend/                 # FastAPI backend
│   ├── main.py            # Main application
│   ├── requirements.txt   # Python dependencies
│   ├── data/              # Document storage
│   ├── qdrant_storage/    # Vector database
│   └── scripts/           # Utility scripts
├── frontend/               # React frontend
├── models/                 # AI models
├── venv/                   # Python virtual environment
├── start.bat               # Development startup script
├── setup_env.bat           # Environment setup script
└── package_for_windows.bat # Windows EXE packaging
```

## 🛠️ Development

- **Backend**: FastAPI with Qdrant vector search
- **Frontend**: React with chat interface
- **AI**: Ollama + Llama3 for answer generation
- **Embeddings**: Sentence transformers for document vectors

## 📦 Deployment

- **Development**: Use `start.bat` for local development
- **Production**: Use `package_for_windows.bat` to create portable EXE
- **Portable**: Export/import embeddings without re-parsing PDFs

## 🔧 Scripts

- `setup_env.bat` - Creates Python environment and installs dependencies
- `start.bat` - Starts backend and frontend development servers
- `package_for_windows.bat` - Creates Windows EXE distribution package

## 📚 Documentation

- `backend/README_DEPLOYMENT.md` - Detailed deployment guide
- `backend/scripts/` - Export/import scripts for embeddings
