from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import os
from typing import List
import json

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
    """Simple answer generation using context"""
    if not context_docs:
        return "I couldn't find relevant information to answer your question."
    
    context_text = "\n\n".join([f"From {doc['filename']}:\n{doc['content'][:500]}..." 
                                for doc in context_docs])
    
    # For now, return a simple response - you can integrate with Ollama later
    return f"Based on the available documents, here's what I found relevant to your question '{question}':\n\n{context_text[:1000]}..."

@app.on_event("startup")
def startup_event():
    load_documents()

@app.get("/ingest")
def ingest():
    load_documents()
    return {"status": "reloaded", "documents_loaded": len(documents)}

@app.post("/ask")
async def ask(request: Request):
    data = await request.json()
    question = data.get("question", "")
    
    if not question:
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
