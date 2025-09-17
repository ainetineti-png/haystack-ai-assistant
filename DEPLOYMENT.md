# Deployment Guide: Handling Large Files

## 🚨 Important: Why Not Commit Large Files

**DON'T commit these to GitHub:**
- PDFs, DOCX, and other documents (`data/` directory)
- Vector databases (`qdrant_storage/`)
- AI models (`models/` directory - ~500MB+)
- Database files (`*.db`, `*.sqlite`)
- Embeddings and indexes (`*.pkl`, `*.npz`)

## ✅ Recommended Approaches

### Option 1: Separate File Storage (Recommended)
```bash
# 1. Push only code to GitHub
git add .
git commit -m "Initial commit: HayStack AI Assistant"
git push origin main

# 2. Share large files separately:
# - Google Drive/Dropbox for documents
# - Model Hub (HuggingFace) for AI models
# - Cloud storage for databases
```

### Option 2: Git LFS (For Essential Large Files Only)
```bash
# Install Git LFS
git lfs install

# Track specific large files (be selective!)
git lfs track "data/sample.pdf"
git lfs track "models/*.bin"

# Commit LFS configuration
git add .gitattributes
git commit -m "Add Git LFS for essential files"
```

### Option 3: Docker with Volume Mounts
```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY backend/requirements.txt .
RUN pip install -r requirements.txt

COPY backend/ .
COPY frontend/build/ ./static/

# External volumes for large files
VOLUME ["/app/data", "/app/models", "/app/qdrant_storage"]

CMD ["python", "main.py"]
```

```bash
# Docker run with external volumes
docker run -v ./data:/app/data -v ./models:/app/models myapp
```

## 🌐 Production Deployment Strategies

### 1. Cloud-Native Approach
```bash
# Use cloud services for large files:
# - AWS S3 for documents
# - HuggingFace Hub for models  
# - Qdrant Cloud for vector DB
# - RDS/PostgreSQL for metadata

# Environment variables in production:
QDRANT_URL=https://your-qdrant-cloud.com
QDRANT_API_KEY=your-api-key
DOCUMENTS_BUCKET=s3://your-docs-bucket
```

### 2. Self-Hosted with External Storage
```bash
# Mount network storage
mount -t nfs server:/storage/models /app/models
mount -t nfs server:/storage/data /app/data

# Or use object storage client
aws s3 sync s3://your-bucket/models ./models
```

### 3. Hybrid Approach
```bash
# Code in Git, data from external sources
git clone https://github.com/you/haystack-ai-assistant
cd haystack-ai-assistant

# Download models automatically
./setup_models.sh

# Sync data from cloud
aws s3 sync s3://your-docs ./data
```

## 📋 Setup Instructions for New Environments

### 1. Clone Repository
```bash
git clone https://github.com/your-username/haystack-ai-assistant.git
cd haystack-ai-assistant
```

### 2. Set Up Python Environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

pip install -r backend/requirements.txt
```

### 3. Download Models & Setup
```bash
# Automated setup
./setup_models.sh        # Linux/Mac
# or
setup_models.bat         # Windows

# Manual setup
mkdir models data
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2').save('./models/all-MiniLM-L6-v2')"
ollama pull llama3:latest
```

### 4. Add Your Documents
```bash
# Copy your documents to data directory
cp /path/to/your/docs/* ./data/
```

### 5. Run Application
```bash
# Backend
cd backend && python main.py

# Frontend (separate terminal)
cd frontend && npm install && npm start
```

## 🔄 Continuous Integration / Deployment

### GitHub Actions Example
```yaml
name: Deploy HayStack

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: pip install -r backend/requirements.txt
    
    - name: Download models
      run: |
        python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2').save('./models/all-MiniLM-L6-v2')"
    
    - name: Deploy to server
      run: |
        # Your deployment commands
        rsync -av --exclude='data/' . user@server:/app/
```

## 📊 File Size Management

### Current Project Structure (with .gitignore)
```
Repository Size: ~50MB (code only)
├── backend/ (~10MB - Python code)
├── frontend/ (~30MB - React app)
├── docs/ (~5MB - documentation)
└── config files (~5MB)

External Files (not in repo):
├── data/ (varies - your documents)
├── models/ (~500MB - AI models)
├── qdrant_storage/ (varies - vector DB)
└── *.db files (varies - SQLite databases)
```

### Benefits of This Approach
- ✅ Fast cloning (seconds vs hours)
- ✅ No GitHub storage costs
- ✅ Easy collaboration on code
- ✅ Flexible data management
- ✅ Environment-specific configurations

### Team Collaboration
```bash
# Developer workflow:
1. Clone repo (small, fast)
2. Run setup script (downloads models)
3. Add their own documents to data/
4. Start developing

# No need to download everyone's documents
# No conflicts with large binary files
# Easy to test with different datasets
```

This approach separates **code** (version controlled) from **data** (environment-specific), which is the industry best practice for ML/AI applications.
