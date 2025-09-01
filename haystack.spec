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

# Collect frontend build files (if they exist)
frontend_data = []
frontend_build = 'frontend/build'
if os.path.exists(frontend_build):
    for root, dirs, files in os.walk(frontend_build):
        for file in files:
            src = os.path.join(root, file)
            dst = os.path.relpath(src)
            frontend_data.append((src, dst))

# Collect Qdrant database (if it exists)
qdrant_data = []
qdrant_path = 'backend/qdrant_storage'
if os.path.exists(qdrant_path):
    for root, dirs, files in os.walk(qdrant_path):
        for file in files:
            src = os.path.join(root, file)
            dst = os.path.relpath(src)
            qdrant_data.append((src, dst))

a = Analysis(
    ['backend/main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('backend/requirements.txt', 'backend/'),
        *model_data,
        *frontend_data,
        *qdrant_data,
    ],
    hiddenimports=[
        'uvicorn.loops.auto',
        'uvicorn.protocols.http.auto', 
        'uvicorn.protocols.websockets.auto',
        'uvicorn.lifespan.on',
        'sentence_transformers',
        'sentence_transformers.models',
        'sentence_transformers.models.Transformer',
        'sentence_transformers.models.Pooling',
        'qdrant_client',
        'qdrant_client.local',
        'fastapi',
        'pydantic',
        'PyPDF2',
        'pdfplumber',
        'docx',
        'numpy',
        'torch',
        'transformers',
        'tokenizers',
        'sqlite3',
        'portalocker',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'tkinter',
        'jupyter',
        'IPython',
        'pytest',
    ],
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
    icon=None,  # Add path to .ico file if you have one
)
