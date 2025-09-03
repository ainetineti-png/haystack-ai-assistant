from typing import List

def batch_embed_texts(embedder, texts: List[str], batch_size: int = 16) -> List[list]:
    """
    Embed texts in batches for efficiency.
    """
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_embeds = embedder.encode(batch)
        embeddings.extend(batch_embeds)
    return embeddings

# Example dense search using batch embedding
def dense_search(embedder, qdrant_client, query: str, top_k: int = 5) -> List[dict]:
    query_vec = embedder.encode([query])[0].tolist()
    results = []
    hits = qdrant_client.search(
        collection_name="documents",
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
