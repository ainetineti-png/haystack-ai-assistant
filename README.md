# Haystack RAG Chat App

## Backend (FastAPI + Simple RAG)
1. Create and activate virtual environment:
   ```pwsh
   python -m venv venv
   venv\Scripts\activate
   ```
2. Install dependencies:
   ```pwsh
   cd backend
   pip install fastapi uvicorn
   ```
3. Add your text files to `./backend/data/`.
4. Run the backend:
   ```pwsh
   cd backend
   C:/HayStack/venv/Scripts/python.exe -m uvicorn main:app --reload --host 127.0.0.1 --port 8001
   ```

## Frontend (React)
1. Install dependencies:
   ```pwsh
   cd frontend
   npm install
   ```
2. Start the frontend:
   ```pwsh
   npm start
   ```

## Quick Start
- Double-click `start.bat` to run both backend and frontend automatically
- Double-click `stop.bat` to stop both services

## Usage
- Backend runs on: http://127.0.0.1:8001
- Frontend runs on: http://localhost:3000 (usually)
- Ask questions in the chatbox. The AI answer and retrieved context will be shown for accuracy verification.
- To reload documents, call `GET http://127.0.0.1:8001/ingest`

## Notes
- No document upload: only parses files from `./backend/data/` at startup or via `/ingest`.
- Uses simple keyword-based search (can be upgraded to BM25/embeddings later).
- Context is always displayed below the AI answer for verification.
