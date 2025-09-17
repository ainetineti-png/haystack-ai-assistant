#!/bin/bash
# setup_models.sh - Script to download required models and setup large files

echo "🔧 Setting up HayStack AI Assistant models and data..."

# Create directories
mkdir -p models
mkdir -p data
mkdir -p backend/qdrant_storage

echo "📥 Downloading embedding model (all-MiniLM-L6-v2)..."
python -c "
from sentence_transformers import SentenceTransformer
import os
model_path = os.path.join('models', 'all-MiniLM-L6-v2')
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
model.save(model_path)
print(f'✅ Model saved to {model_path}')
"

echo "🦙 Checking Ollama installation..."
if command -v ollama &> /dev/null; then
    echo "✅ Ollama found"
    echo "📥 Pulling Llama3 model..."
    ollama pull llama3:latest
else
    echo "❌ Ollama not found. Please install from https://ollama.ai"
    echo "   Then run: ollama pull llama3:latest"
fi

echo "📁 Setting up data directory..."
if [ ! -f "data/README.md" ]; then
    cat > data/README.md << 'EOF'
# Data Directory

Place your documents here for processing:

## Supported Formats
- PDF files (*.pdf)
- Word documents (*.docx, *.doc)
- Text files (*.txt)
- Markdown files (*.md)

## Example Structure
```
data/
├── research_papers/
│   ├── paper1.pdf
│   └── paper2.pdf
├── textbooks/
│   └── chapter1.pdf
└── notes/
    ├── lecture1.md
    └── summary.txt
```

## Processing
Documents are automatically processed when the backend starts (if AUTO_INGEST=1).
The system will:
1. Extract text content
2. Split into chunks
3. Generate embeddings
4. Store in vector database

## Privacy Note
This directory is excluded from Git to protect your documents.
EOF
fi

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Add your documents to the data/ directory"
echo "2. Start the backend: cd backend && python main.py"
echo "3. Start the frontend: cd frontend && npm start"
