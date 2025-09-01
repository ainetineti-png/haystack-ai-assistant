# HayStack RAG - Portable Deployment Guide

## 1. Using Pre-computed Embeddings (No Re-parsing)

### What gets stored where
- **Qdrant Database**: `backend/qdrant_storage/` - Contains all vector embeddings and metadata
- **Original Files**: `backend/data/` - Your PDF/DOCX source documents
- **Embedding Model**: `models/all-MiniLM-L6-v2/` - Sentence transformer model files

### Method 1: Copy Entire qdrant_storage Directory
This is the **simplest approach** for moving to another machine:

```bash
# On source machine - after indexing is complete
zip -r qdrant_database.zip backend/qdrant_storage/

# On target machine - before starting the app
# Extract to backend/qdrant_storage/
unzip qdrant_database.zip -d backend/
```

**Requirements:**
- Target machine must have the **same embedding model** at the same path
- Or set `EMBED_MODEL_PATH` environment variable to the correct location
- SQLite database is cross-platform compatible

### Method 2: Export/Import via JSON (More Portable)

```bash
# Export from source machine
cd backend
python scripts/qdrant_export.py --output my_embeddings.ndjson

# Import on target machine  
cd backend
python scripts/qdrant_import.py --input my_embeddings.ndjson
```

**Benefits:**
- Human-readable format
- Can inspect/modify embeddings
- Works across different Qdrant versions
- Smaller file size (no SQLite overhead)

### Method 3: Hybrid Approach (Recommended for Production)

Create a deployment package:
```bash
# Package everything needed
deployment_package/
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   └── qdrant_storage/          # Pre-computed database
├── models/
│   └── all-MiniLM-L6-v2/        # Embedding model
├── frontend/
│   └── build/                   # Pre-built React app
└── data/                        # Source documents (optional)
```

## 2. Windows EXE Distribution

### Option A: PyInstaller (Recommended)

1. **Create spec file** (`haystack.spec`):
```python
# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Collect all sentence-transformers model files
model_path = 'models/all-MiniLM-L6-v2'
model_data = []
if os.path.exists(model_path):
    for root, dirs, files in os.walk(model_path):
        for file in files:
            src = os.path.join(root, file)
            dst = os.path.relpath(src)
            model_data.append((src, dst))

# Collect frontend build files
frontend_data = []
frontend_build = 'frontend/build'
if os.path.exists(frontend_build):
    for root, dirs, files in os.walk(frontend_build):
        for file in files:
            src = os.path.join(root, file)
            dst = os.path.relpath(src)
            frontend_data.append((src, dst))

a = Analysis(
    ['backend/main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('backend/requirements.txt', 'backend/'),
        ('backend/qdrant_storage', 'backend/qdrant_storage'),
        *model_data,
        *frontend_data,
    ],
    hiddenimports=[
        'uvicorn.loops.auto',
        'uvicorn.protocols.http.auto',
        'uvicorn.protocols.websockets.auto',
        'uvicorn.lifespan.on',
        'sentence_transformers',
        'qdrant_client',
        'fastapi',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

pyt = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyt,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='HayStackRAG',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
```

2. **Build the EXE**:
```bash
# Install PyInstaller
pip install pyinstaller

# Build with spec file
pyinstaller haystack.spec

# Output will be in dist/HayStackRAG.exe
```

### Option B: Docker + Wine (Cross-platform building)

Create `Dockerfile.windows`:
```dockerfile
FROM python:3.11-windowsservercore

WORKDIR /app
COPY . .

RUN pip install -r backend/requirements.txt pyinstaller
RUN pyinstaller haystack.spec

# Export the exe
VOLUME ["/output"]
CMD ["copy", "dist/HayStackRAG.exe", "/output/"]
```

### Option C: NSIS Installer (Professional Distribution)

Create a complete Windows installer that:
- Installs the EXE to Program Files
- Creates Start Menu shortcuts
- Includes Ollama installation (optional)
- Sets up Windows service (optional)

## 3. Deployment Strategies

### Strategy 1: Standalone EXE with Embedded Database
**Best for**: Small to medium document collections (< 10GB)

```
HayStackRAG.exe
├── Embedded: Python runtime
├── Embedded: All dependencies  
├── Embedded: Pre-computed embeddings
├── Embedded: ML model
└── Embedded: Frontend (static files)
```

**Pros**: Single file, no installation
**Cons**: Large file size (500MB+), slower startup

### Strategy 2: Installer Package  
**Best for**: Professional deployment, larger datasets

```
HayStackRAG_Setup.exe
└── Installs:
    ├── HayStackRAG.exe (smaller)
    ├── models/ (downloaded separately)
    ├── data/ (user documents)
    └── config/ (settings)
```

**Pros**: Smaller download, faster updates, user data separation
**Cons**: Requires installation

### Strategy 3: Portable ZIP Package
**Best for**: USB/network distribution, no admin rights

```
HayStackRAG_Portable.zip
├── HayStackRAG.exe
├── models/
├── data/
├── config.json
└── README.txt
```

## 4. Environment Variables for Portability

Make paths configurable:
```python
# In backend/main.py
MODEL_PATH = os.environ.get("EMBED_MODEL_PATH", 
    os.path.join(os.path.dirname(__file__), "..", "models", "all-MiniLM-L6-v2"))
DATA_DIR = os.environ.get("DATA_DIR",
    os.path.join(os.path.dirname(__file__), "data"))
QDRANT_PATH = os.environ.get("QDRANT_PATH",
    os.path.join(os.path.dirname(__file__), "qdrant_storage"))
```

## 5. Quick Start for New Machine

1. **Copy the package**:
   ```bash
   # Option 1: Pre-built EXE
   HayStackRAG.exe
   
   # Option 2: Source + database
   unzip haystack_with_embeddings.zip
   ```

2. **Install Ollama** (if needed):
   ```bash
   # Download from https://ollama.ai/
   ollama pull llama3
   ```

3. **Run**:
   ```bash
   # EXE version
   ./HayStackRAG.exe
   
   # Source version
   python backend/main.py
   ```

4. **Verify**: Open browser to `http://localhost:8000`

The app will use existing embeddings without re-parsing PDFs!

## 6. File Size Estimates

| Component | Size | Required |
|-----------|------|----------|
| Python Runtime (embedded) | ~50MB | Yes |
| Dependencies | ~200MB | Yes |
| Sentence Transformer Model | ~90MB | Yes |
| Frontend Build | ~5MB | Yes |
| Qdrant Database | Variable | Optional* |
| PDF Source Files | Variable | No |

*Required for immediate search without re-indexing

**Total EXE size**: ~350MB minimum + your embeddings database size

## 7. Troubleshooting

### "Model not found" error:
- Check `EMBED_MODEL_PATH` environment variable
- Ensure model files copied correctly

### "Collection not found" error:
- Qdrant database not copied
- Run import script: `python scripts/qdrant_import.py -i embeddings.ndjson`

### Slow startup:
- Large embedding model loading
- Consider model caching or smaller model

### Permission errors:
- Run as administrator (Windows)
- Check file/folder permissions
