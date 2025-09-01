from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import os
from typing import List
import json
import requests
import glob
import PyPDF2
import pdfplumber
from docx import Document
import uuid  # Added for valid UUID point ids
import hashlib  # For chunk content hashing
from contextlib import asynccontextmanager  # Added for lifespan
# Removed LangExtract import - no longer needed
from qdrant_client import QdrantClient, models as qdrant_models
from sentence_transformers import SentenceTransformer

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load documents at startup
    load_documents()
    try:
        yield
    finally:
        # Ensure Qdrant client closes before interpreter teardown to avoid portalocker ImportError
        try:
            qdrant_client.close()
            print("[INFO] Qdrant client closed cleanly")
        except Exception as e:
            print(f"[WARNING] Qdrant client close error: {e}")

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
documents = []
last_loaded_files = {}

# Progress tracking globals
ingest_total_files = 0
ingest_processed_files = 0

# Qdrant setup (embedded/local mode)
QDRANT_PATH = os.path.join(os.path.dirname(__file__), "qdrant_storage")
QDRANT_COLLECTION = "documents"
qdrant_client = QdrantClient(path=QDRANT_PATH)

# Embedding model setup (local path to avoid SSL). Set EMBED_MODEL_PATH env var to a local folder.
model_path = os.environ.get("EMBED_MODEL_PATH", "C:\\HayStack\\models\\all-MiniLM-L6-v2")
print(f"Attempting to load model from: {model_path}")
print(f"Path exists: {os.path.exists(model_path)}")
print(f"Path is absolute: {os.path.isabs(model_path)}")

try:
    # Try to load the model from local path
    embedder = SentenceTransformer(model_path)
    print(f"Successfully loaded local embedding model from: {model_path}")
except Exception as e:
    print(f"Warning: Could not load sentence transformer model from '{model_path}': {e}")
    print("Using simple keyword-based search only")
    embedder = None

# ---- New chunking helpers ----
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def extract_pdf_pages(fpath: str) -> List[str]:
    """Extract text per page from a PDF using PyPDF2 with pdfplumber fallback. Returns list of page texts."""
    pages_text: List[str] = []
    try:
        with open(fpath, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page_num, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text() or ""
                    page_text = page_text.strip()
                    if page_text:
                        pages_text.append(page_text)
                    else:
                        pages_text.append("")
                except Exception as page_error:
                    print(f"[WARNING] PyPDF2 failed on page {page_num + 1} of {fpath}: {page_error}")
                    pages_text.append("")
        # Assess quality: if aggregated text length is too small try pdfplumber
        if sum(len(p) for p in pages_text) < 50:
            print(f"[INFO] PyPDF2 extracted little content from {fpath}, trying pdfplumber fallback")
            pages_text = []
            with pdfplumber.open(fpath) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    try:
                        page_text = page.extract_text() or ""
                        pages_text.append(page_text.strip())
                    except Exception as page_error:
                        print(f"[WARNING] pdfplumber failed on page {page_num + 1} of {fpath}: {page_error}")
                        pages_text.append("")
    except Exception as pdf_error:
        print(f"[WARNING] PyPDF2 completely failed on {fpath}: {pdf_error}, trying pdfplumber")
        try:
            with pdfplumber.open(fpath) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    try:
                        page_text = page.extract_text() or ""
                        page_text = page_text.strip()
                        pages_text.append(page_text)
                    except Exception as page_error:
                        print(f"[WARNING] pdfplumber failed on page {page_num + 1} of {fpath}: {page_error}")
                        pages_text.append("")
        except Exception as fallback_error:
            print(f"[ERROR] Both PyPDF2 and pdfplumber failed on {fpath}: {fallback_error}")
    # Filter leading/trailing empty pages only if all empty
    if not any(p.strip() for p in pages_text):
        print(f"[WARNING] No content extracted from {fpath}")
        return []
    return pages_text


def chunk_text(pages: List[str]) -> List[dict]:
    """Chunk list of page texts into overlapping segments. Returns list of chunk dicts."""
    chunks: List[dict] = []
    for page_idx, page_text in enumerate(pages):
        text = page_text.strip()
        if not text:
            continue
        start = 0
        length = len(text)
        chunk_number = 0
        while start < length:
            end = min(start + CHUNK_SIZE, length)
            chunk_str = text[start:end]
            # Extend to sentence boundary minimal (look ahead for next period within 40 chars)
            if end < length:
                next_period = text.find('.', end, min(end + 40, length))
                if next_period != -1:
                    chunk_str = text[start:next_period + 1]
                    end = next_period + 1
            md5 = hashlib.md5(chunk_str.encode('utf-8')).hexdigest()
            chunks.append({
                "page": page_idx + 1,
                "chunk_index": chunk_number,
                "text": chunk_str,
                "md5": md5
            })
            chunk_number += 1
            if end >= length:
                break
            start = max(end - CHUNK_OVERLAP, 0)
    return chunks

# ---- End chunking helpers ----


def load_documents():
    global documents, ingest_total_files, ingest_processed_files
    documents = []
    ingest_processed_files = 0
    file_patterns = ["**/*.txt", "**/*.pdf", "**/*.docx", "**/*.xls", "**/*.xlsx"]
    all_files = []
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        ingest_total_files = 0
        return
    for pattern in file_patterns:
        all_files.extend(glob.glob(os.path.join(DATA_DIR, pattern), recursive=True))
    ingest_total_files = len(all_files)
    # Ensure Qdrant collection exists only if embedder is available
    if embedder is not None and QDRANT_COLLECTION not in [c.name for c in qdrant_client.get_collections().collections]:
        qdrant_client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=qdrant_models.VectorParams(size=384, distance=qdrant_models.Distance.COSINE)
        )
    # Ingest pipeline
    for fpath in all_files:
        ext = os.path.splitext(fpath)[1].lower()
        try:
            pages: List[str] = []
            if ext == ".txt":
                with open(fpath, "r", encoding="utf-8") as f:
                    pages = [f.read()]
            elif ext == ".pdf":
                pages = extract_pdf_pages(fpath)
                if not pages:
                    ingest_processed_files += 1
                    continue
            elif ext == ".docx":
                doc = Document(fpath)
                pages = ["\n".join([paragraph.text for paragraph in doc.paragraphs])]
            else:
                # Unsupported type in initial ingest; skip
                ingest_processed_files += 1
                continue

            content = "\n".join(pages)
            lines = content.split('\n')
            first_line = lines[0] if lines else ""
            structured = {
                "filename": os.path.basename(fpath),
                "first_line": first_line[:100],
                "content_length": len(content),
                "file_type": ext
            }

            # Store whole document (original behaviour) for fallback simple search
            documents.append({"content": content, "filename": os.path.relpath(fpath, DATA_DIR), "structured": str(structured)})

            # Chunking
            chunk_list = chunk_text(pages)
            if embedder is not None and chunk_list:
                points = []
                for ch in chunk_list:
                    try:
                        embedding = embedder.encode(ch["text"]).tolist()
                        chunk_uuid_seed = f"{fpath}#page={ch['page']}#chunk={ch['chunk_index']}#md5={ch['md5']}"
                        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_uuid_seed))
                        payload = {
                            "filename": os.path.relpath(fpath, DATA_DIR),
                            "page": ch["page"],
                            "chunk_index": ch["chunk_index"],
                            "text": ch["text"],
                            "md5": ch["md5"],
                            "structured": str(structured)
                        }
                        points.append(
                            qdrant_models.PointStruct(
                                id=point_id,
                                vector=embedding,
                                payload=payload
                            )
                        )
                    except Exception as embed_error:
                        print(f"[WARNING] Could not embed chunk page {ch['page']} chunk {ch['chunk_index']} of {fpath}: {embed_error}")
                if points:
                    try:
                        qdrant_client.upsert(collection_name=QDRANT_COLLECTION, points=points)
                        print(f"[INFO] Stored {len(points)} chunks for {fpath}")
                    except Exception as up_err:
                        print(f"[ERROR] Upsert failed for {fpath}: {up_err}")
            elif embedder is None:
                print(f"[INFO] Skipping vector storage for {fpath} (no embedder available)")
            else:
                print(f"[WARNING] No chunks produced for {fpath}")
        except Exception as e:
            print(f"[ERROR] Failed to process {fpath}: {e}")
        ingest_processed_files += 1


def simple_search(query: str, docs: List[dict], top_k: int = 3) -> List[dict]:
    """Simple keyword-based search"""
    query_words = query.lower().split()
    scored_docs = []
    
    for doc in docs:
        content_lower = doc["content"].lower()
        score = sum(1 for word in query_words if word in content_lower)
        if score > 0:
            scored_docs.append({"doc": doc, "score": score})
    
    # Sort by score and return top_k
    scored_docs.sort(key=lambda x: x["score"], reverse=True)
    return [item["doc"] for item in scored_docs[:top_k]]


def vector_search(query: str, top_k: int = 3) -> List[dict]:
    if embedder is None:
        return []
    try:
        query_vec = embedder.encode(query).tolist()
        results = []
        try:
            # New preferred API
            qp = qdrant_client.query_points(
                collection_name=QDRANT_COLLECTION,
                query=query_vec,
                limit=top_k,
                with_payload=True
            )
            hits = qp.points
        except Exception as new_api_err:
            print(f"[WARNING] query_points failed, falling back to search: {new_api_err}")
            hits = qdrant_client.search(
                collection_name=QDRANT_COLLECTION,
                query_vector=query_vec,
                limit=top_k,
                with_payload=True
            )
        for hit in hits:
            payload = hit.payload or {}
            results.append({
                "filename": payload.get("filename", "unknown"),
                "content": payload.get("text", payload.get("content", "")),
                "page": payload.get("page"),
                "chunk_index": payload.get("chunk_index")
            })
        return results
    except Exception as e:
        print(f"[ERROR] Vector search failed: {e}")
        return []


def generate_answer(question: str, context_docs: List[dict]) -> str:
    """Answer generation using Llama3 via Ollama API"""
    print(f"[DEBUG] Received question: {question}")
    print(f"[DEBUG] Context docs: {[doc['filename'] for doc in context_docs]}")
    if not context_docs:
        print("[DEBUG] No relevant documents found.")
        return "I couldn't find relevant information to answer your question."

    context_text = "\n\n".join([f"From {doc.get('filename','unknown')} (page {doc.get('page','?')}):\n{doc.get('content','')[:500]}..." for doc in context_docs])
    prompt = f"Answer the following question using the provided context.\n\nQuestion: {question}\n\nContext:\n{context_text}"
    print(f"[DEBUG] Sending prompt to Ollama: {prompt[:200]}...")
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3:latest",
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )
        print(f"[DEBUG] Ollama response status: {response.status_code}")
        response.raise_for_status()
        data = response.json()
        print(f"[DEBUG] Ollama raw response: {data}")
        answer = data.get("response", "Sorry, I couldn't generate an answer.")
        print(f"[DEBUG] Final answer: {answer[:200]}...")
        return answer
    except requests.exceptions.ConnectionError as e:
        print(f"[ERROR] Connection error: {e}")
        return "Error: Ollama is not running. Please start Ollama first."
    except requests.exceptions.Timeout as e:
        print(f"[ERROR] Timeout error: {e}")
        return "Error: Request to Ollama timed out. The model might be loading."
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return f"Error communicating with Ollama: {str(e)}"


@app.get("/ingest")
def ingest():
    load_documents()
    return {"status": "reloaded", "documents_loaded": len(documents)}

@app.get("/health")
def health_check():
    """Check if Ollama and Llama3 are available"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        response.raise_for_status()
        models = response.json().get("models", [])
        llama3_available = any("llama3" in model.get("name", "") for model in models)
        
        return {
            "ollama_running": True,
            "llama3_available": llama3_available,
            "available_models": [model.get("name", "") for model in models]
        }
    except Exception as e:
        return {
            "ollama_running": False,
            "llama3_available": False,
            "error": str(e)
        }

@app.post("/ask")
async def ask(request: Request):
    print("[DEBUG] /ask endpoint called")
    data = await request.json()
    question = data.get("question", "")
    print(f"[DEBUG] Question received: '{question}'")
    if not question:
        print("[DEBUG] No question provided")
        return {"error": "Question is required"}
    # Prefer vector search if embeddings available
    if embedder is not None and QDRANT_COLLECTION in [c.name for c in qdrant_client.get_collections().collections]:
        context_docs = vector_search(question, top_k=3)
    else:
        context_docs = simple_search(question, documents)
    answer = generate_answer(question, context_docs)
    context = [f"{doc['filename']}: {doc['content'][:200]}..." for doc in context_docs]
    return {
        "answer": answer,
        "context": context,
        "documents_found": len(context_docs)
    }

@app.get("/doc_stats")
def doc_stats():
    """Return filename and content length for all loaded documents."""
    return {
        "count": len(documents),
        "docs": [
            {"filename": d["filename"], "length": len(d["content"]) } for d in documents
        ]
    }

@app.get("/doc")
def get_doc(filename: str):
    """Return a snippet of a specific document by its relative filename."""
    for doc in documents:
        if doc["filename"] == filename:
            return {"filename": doc["filename"], "content": doc["content"][:500]}
    return {"error": "Document not found"}


@app.get("/ingest_status")
def ingest_status():
    return {
        "total": ingest_total_files,
        "processed": ingest_processed_files,
        "percent": int(100 * ingest_processed_files / ingest_total_files) if ingest_total_files else 100
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)