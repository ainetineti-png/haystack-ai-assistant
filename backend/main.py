from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import os
from typing import List
import json
import requests

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = "./data/"
documents = []

def load_documents():
    global documents
    documents = []
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        return
    
    for fname in os.listdir(DATA_DIR):
        fpath = os.path.join(DATA_DIR, fname)
        if os.path.isfile(fpath):
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
                documents.append({"content": content, "filename": fname})

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

def generate_answer(question: str, context_docs: List[dict]) -> str:
    """Answer generation using Llama3 via Ollama API"""
    print(f"[DEBUG] Received question: {question}")
    print(f"[DEBUG] Context docs: {[doc['filename'] for doc in context_docs]}")
    if not context_docs:
        print("[DEBUG] No relevant documents found.")
        return "I couldn't find relevant information to answer your question."

    context_text = "\n\n".join([f"From {doc['filename']}:\n{doc['content'][:500]}..." for doc in context_docs])
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

@app.on_event("startup")
def startup_event():
    load_documents()

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
    
    # Find relevant documents
    relevant_docs = simple_search(question, documents)
    
    # Generate answer
    answer = generate_answer(question, relevant_docs)
    
    # Return context for verification
    context = [f"{doc['filename']}: {doc['content'][:200]}..." for doc in relevant_docs]
    
    return {
        "answer": answer,
        "context": context,
        "documents_found": len(relevant_docs)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
