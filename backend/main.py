from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import os
from typing import List, Dict, Any, Set
import json
import requests
import glob
import PyPDF2
import pdfplumber
from docx import Document
import uuid  # Added for valid UUID point ids
import hashlib  # For chunk content hashing
from contextlib import asynccontextmanager  # Added for lifespan
import atexit  # Safe shutdown handler
import unicodedata  # For block normalization
# Removed LangExtract import - no longer needed
from qdrant_client import QdrantClient, models as qdrant_models
from sentence_transformers import SentenceTransformer

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load documents at startup only if AUTO_INGEST
    if AUTO_INGEST:
        load_documents()
    else:
        print("[INFO] AUTO_INGEST disabled; skipping ingestion at startup")
    try:
        yield
    finally:
            _safe_close_qdrant()

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

# Disable noisy destructor that triggers portalocker ImportError at shutdown
try:
    qdrant_client.__del__ = lambda self: None  # type: ignore
except Exception:
    pass


def _safe_close_qdrant():
    """Close Qdrant client defensively before interpreter teardown."""
    global qdrant_client
    if qdrant_client is None:
        return
    try:
        qdrant_client.close()
        print("[INFO] Qdrant client closed cleanly (atexit)")
    except Exception as e:
        print(f"[DEBUG] Ignored Qdrant close issue during shutdown: {e}")

atexit.register(_safe_close_qdrant)

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
# Phase 1 additions
SCHEMA_VERSION = int(os.environ.get("SCHEMA_VERSION", "3"))  # Bumped schema version for per-page + patterns + normalization
ENABLE_PYMUPDF = os.environ.get("ENABLE_PYMUPDF", "1") == "1"
AUTO_INGEST = os.environ.get("AUTO_INGEST", "1") == "1"  # If 0, skip automatic ingest on startup
FORCE_REINDEX = os.environ.get("FORCE_REINDEX", "0") == "1"  # Force full reindex
TARGET_TOKENS = 200
MAX_TOKENS = 280
HEADER_FOOTER_FREQ_THRESHOLD = 0.6
ENABLE_TABLES = os.environ.get("ENABLE_TABLES", "1") == "1"
# New: allow skipping expensive PDF re-parse on unchanged corpus
FAST_SKIP_PDF_REPARSE = os.environ.get("FAST_SKIP_PDF_REPARSE", "1") == "1"
MAX_PYPDF2_WARNINGS_PER_FILE = int(os.environ.get("MAX_PYPDF2_WARNINGS_PER_FILE", "5"))

try:
    if ENABLE_PYMUPDF:
        import fitz  # PyMuPDF
        _PYMUPDF_AVAILABLE = True
    else:
        _PYMUPDF_AVAILABLE = False
except Exception:
    _PYMUPDF_AVAILABLE = False
    fitz = None  # type: ignore
    print("[INFO] PyMuPDF not available; install with 'pip install pymupdf' to enable structured parsing")

def extract_pdf_pages(fpath: str) -> List[str]:
    """Extract text per page from a PDF using PyPDF2 with per-page pdfplumber fallback.
    Returns list of page texts. Throttles noisy warnings and retries empty pages individually.
    """
    pages_text: List[str] = []
    warn_count = 0
    try:
        with open(fpath, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page_num, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text() or ""
                    page_text = page_text.strip()
                    pages_text.append(page_text)
                except Exception as page_error:
                    warn_count += 1
                    if warn_count <= MAX_PYPDF2_WARNINGS_PER_FILE:
                        print(f"[WARNING] PyPDF2 failed on page {page_num + 1} of {fpath}: {page_error}")
                    elif warn_count == MAX_PYPDF2_WARNINGS_PER_FILE + 1:
                        print(f"[WARNING] Further PyPDF2 page parse warnings for {fpath} suppressed...")
                    pages_text.append("")
        # Per-page fallback for empty pages (rather than whole-file threshold only)
        if any(not p.strip() for p in pages_text):
            try:
                with pdfplumber.open(fpath) as pdf:
                    page_count = min(len(pdf.pages), len(pages_text))
                    for idx in range(page_count):
                        if pages_text[idx].strip():
                            continue  # already have text
                        try:
                            txt = pdf.pages[idx].extract_text() or ""
                            pages_text[idx] = txt.strip()
                        except Exception as fallback_page_err:
                            # Only log first few
                            if warn_count < MAX_PYPDF2_WARNINGS_PER_FILE:
                                print(f"[WARNING] pdfplumber fallback failed page {idx+1} {fpath}: {fallback_page_err}")
            except Exception as per_page_fb_err:
                # Silent unless debug desired
                print(f"[INFO] Per-page pdfplumber fallback unavailable for {fpath}: {per_page_fb_err}")
        # Legacy heuristic: if overall still minimal content, attempt full pdfplumber extraction
        if sum(len(p) for p in pages_text) < 50:
            try:
                print(f"[INFO] Low aggregate text from PyPDF2 for {fpath}, trying full pdfplumber fallback")
                pages_text_full: List[str] = []
                with pdfplumber.open(fpath) as pdf:
                    for page_num, page in enumerate(pdf.pages):
                        try:
                            page_text = page.extract_text() or ""
                            pages_text_full.append(page_text.strip())
                        except Exception as page_error:
                            if page_num < MAX_PYPDF2_WARNINGS_PER_FILE:
                                print(f"[WARNING] pdfplumber failed on page {page_num + 1} of {fpath}: {page_error}")
                            pages_text_full.append("")
                pages_text = pages_text_full
            except Exception as whole_fb_err:
                print(f"[INFO] Full pdfplumber fallback failed for {fpath}: {whole_fb_err}")
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
                        if page_num < MAX_PYPDF2_WARNINGS_PER_FILE:
                            print(f"[WARNING] pdfplumber failed on page {page_num + 1} of {fpath}: {page_error}")
                        pages_text.append("")
        except Exception as fallback_error:
            print(f"[ERROR] Both PyPDF2 and pdfplumber failed on {fpath}: {fallback_error}")
    if not any(p.strip() for p in pages_text):
        print(f"[WARNING] No content extracted from {fpath}")
        return []
    return pages_text


def normalize_block_text(text: str) -> str:
    """Basic normalization: NFKC, collapse whitespace, trim."""
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text)
    t = ' '.join(t.split())
    return t.strip()

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
                "pages": [page_idx + 1],
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


# Manifest for incremental ingest
INGEST_MANIFEST_PATH = os.path.join(os.path.dirname(__file__), "ingest_manifest.json")
MANIFEST_BACKUPS = 3

# Track skipped files
ingest_skipped_files = 0
# Track removed (deleted from disk) files
ingest_removed_files = 0
# Additional ingest metric
ingest_pages_reindexed = 0

# Helper: delete all vector points for a given relative filename

def delete_file_points(rel_filename: str):
    try:
        from qdrant_client.http import models as rest_models
        qdrant_client.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="filename",
                        match=qdrant_models.MatchValue(value=rel_filename)
                    )
                ]
            )
        )
        print(f"[INFO] Deleted existing vectors for {rel_filename}")
    except Exception as e:
        print(f"[WARNING] Failed to delete vectors for {rel_filename}: {e}")

def delete_file_pages_points(rel_filename: str, pages: List[int]):
    """Delete only vectors for specific pages of a file."""
    if not pages:
        return
    try:
        for p in pages:
            qdrant_client.delete(
                collection_name=QDRANT_COLLECTION,
                points_selector=qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(key="filename", match=qdrant_models.MatchValue(value=rel_filename)),
                        qdrant_models.FieldCondition(key="pages", match=qdrant_models.MatchValue(value=p))
                    ]
                )
            )
        print(f"[INFO] Deleted vectors for {rel_filename} pages {pages}")
    except Exception as e:
        print(f"[WARNING] Failed selective page delete for {rel_filename}: {e}")


def _load_manifest():
    if os.path.exists(INGEST_MANIFEST_PATH):
        try:
            with open(INGEST_MANIFEST_PATH, 'r', encoding='utf-8') as mf:
                return json.load(mf)
        except Exception as e:
            print(f"[WARNING] Could not read manifest: {e}")
    return {"files": {}}


def _save_manifest(manifest: dict):
    try:
        # Rotate backups
        if os.path.exists(INGEST_MANIFEST_PATH):
            for i in range(MANIFEST_BACKUPS-1, 0, -1):
                older = f"{INGEST_MANIFEST_PATH}.bak{i}"
                newer = f"{INGEST_MANIFEST_PATH}.bak{i-1}" if i>1 else f"{INGEST_MANIFEST_PATH}.bak1"
                if os.path.exists(newer):
                    os.replace(newer, older)
            # Copy current to .bak1
            import shutil
            shutil.copy2(INGEST_MANIFEST_PATH, f"{INGEST_MANIFEST_PATH}.bak1")
        with open(INGEST_MANIFEST_PATH, 'w', encoding='utf-8') as mf:
            json.dump(manifest, mf, indent=2)
    except Exception as e:
        print(f"[WARNING] Could not write manifest: {e}")


def load_documents():
    global documents, ingest_total_files, ingest_processed_files, ingest_skipped_files, ingest_removed_files, ingest_pages_reindexed
    documents = []
    ingest_processed_files = 0
    ingest_skipped_files = 0
    ingest_removed_files = 0
    ingest_pages_reindexed = 0
    manifest = _load_manifest()
    files_section = manifest.get('files', {})
    file_patterns = ["**/*.txt", "**/*.pdf", "**/*.docx", "**/*.xls", "**/*.xlsx"]
    all_files = []
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        ingest_total_files = 0
        return
    for pattern in file_patterns:
        all_files.extend(glob.glob(os.path.join(DATA_DIR, pattern), recursive=True))
    ingest_total_files = len(all_files)
    if embedder is not None and qdrant_client is not None and QDRANT_COLLECTION not in [c.name for c in qdrant_client.get_collections().collections]:
        qdrant_client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=qdrant_models.VectorParams(size=384, distance=qdrant_models.Distance.COSINE)
        )
    updated_manifest_files = files_section

    # Early exit if force reindex not requested and all hashes + schema match (schema optional for legacy entries)
    if not FORCE_REINDEX and updated_manifest_files and all_files:
        all_unchanged = True
        for fpath in all_files:
            rel = os.path.relpath(fpath, DATA_DIR)
            prev_entry = updated_manifest_files.get(rel)
            if not prev_entry:
                all_unchanged = False
                break
            prev_schema = prev_entry.get('schema_version')
            # If schema_version present and different -> must reindex. If missing, treat as legacy acceptable.
            if prev_schema is not None and prev_schema != SCHEMA_VERSION:
                all_unchanged = False
                break
            # compute file_md5
            try:
                file_md5_hash = hashlib.md5()
                with open(fpath, 'rb') as fb:
                    for chunk in iter(lambda: fb.read(1024 * 1024), b''):
                        file_md5_hash.update(chunk)
                file_md5 = file_md5_hash.hexdigest()
                if prev_entry.get('file_md5') != file_md5:
                    all_unchanged = False
                    break
            except Exception:
                all_unchanged = False
                break
        if all_unchanged:
            print("[INFO] All files unchanged – skipping embedding phase (legacy schema tolerated)")
            for fpath in all_files:
                rel = os.path.relpath(fpath, DATA_DIR)
                ext = os.path.splitext(fpath)[1].lower()
                try:
                    if ext == '.txt':
                        with open(fpath, 'r', encoding='utf-8') as f:
                            content = f.read()
                    elif ext == '.pdf':
                        if FAST_SKIP_PDF_REPARSE:
                            # Avoid triggering PyPDF2 warnings again for unchanged PDFs
                            content = f"(unchanged pdf: {rel} – content not re-parsed; set FAST_SKIP_PDF_REPARSE=0 to load text)"
                        else:
                            pages = extract_pdf_pages(fpath)
                            content = "\n".join(pages)
                    elif ext == '.docx':
                        doc = Document(fpath)
                        content = "\n".join(p.text for p in doc.paragraphs)
                    else:
                        continue
                    documents.append({"content": content, "filename": rel, "structured": "{}"})
                except Exception as e:
                    print(f"[WARNING] Failed to load content for {rel} during fast path: {e}")
            ingest_processed_files = ingest_total_files
            ingest_skipped_files = ingest_total_files
            return
    current_rel_set = set()
    # Normal processing path
    for fpath in all_files:
        rel = os.path.relpath(fpath, DATA_DIR)
        current_rel_set.add(rel)
        ext = os.path.splitext(fpath)[1].lower()
        try:
            file_md5_hash = hashlib.md5()
            with open(fpath, 'rb') as fb:
                for chunk in iter(lambda: fb.read(1024 * 1024), b''):
                    file_md5_hash.update(chunk)
            file_md5 = file_md5_hash.hexdigest()
            prev_entry = updated_manifest_files.get(rel)
            prev_pages_meta = prev_entry.get('pages') if prev_entry else []
            prev_page_map = {p['page']: p for p in prev_pages_meta} if prev_pages_meta else {}
            prev_schema = prev_entry.get('schema_version') if prev_entry else None
            unchanged_whole = (
                not FORCE_REINDEX and
                prev_entry and
                prev_entry.get('file_md5') == file_md5 and
                (prev_schema is None or prev_schema == SCHEMA_VERSION)
            )
            parse_needed = True
            pages: List[str] = []
            structured_pages_blocks = None
            page_md5s: List[str] = []
            per_page_num_blocks: List[int] = []
            header_patterns: List[str] = []
            footer_patterns: List[str] = []
            block_md5s: List[str] = []
            # Extraction
            if ext == '.txt':
                with open(fpath, 'r', encoding='utf-8') as f:
                    txt = f.read()
                pages = [txt]
                norm = normalize_block_text(txt)
                page_md5s = [hashlib.md5(norm.encode('utf-8')).hexdigest()]
                per_page_num_blocks = [1 if norm else 0]
                block_md5s = [hashlib.md5(norm.encode('utf-8')).hexdigest()]
            elif ext == '.pdf':
                structured_result = _structured_parse_pdf(fpath) if _PYMUPDF_AVAILABLE else None
                if structured_result:
                    structured_pages_blocks = structured_result['pages_blocks']
                    pages = structured_result['page_texts']
                    header_patterns = structured_result['header_lines']
                    footer_patterns = structured_result['footer_lines']
                    per_page_num_blocks = structured_result['per_page_num_blocks']
                    block_md5s = structured_result['block_md5s']
                    page_md5s = [hashlib.md5(normalize_block_text(p).encode('utf-8')).hexdigest() for p in pages]
                else:
                    pages = extract_pdf_pages(fpath)
                    page_md5s = [hashlib.md5(normalize_block_text(p).encode('utf-8')).hexdigest() for p in pages]
                    per_page_num_blocks = [1 if p.strip() else 0 for p in pages]
                    block_md5s = [hashlib.md5(normalize_block_text(p).encode('utf-8')).hexdigest() for p in pages]
                if not pages:
                    ingest_processed_files += 1
                    continue
            elif ext == '.docx':
                doc = Document(fpath)
                doc_text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
                pages = [doc_text]
                norm = normalize_block_text(doc_text)
                page_md5s = [hashlib.md5(norm.encode('utf-8')).hexdigest()]
                per_page_num_blocks = [1 if norm else 0]
                block_md5s = [hashlib.md5(norm.encode('utf-8')).hexdigest()]
            else:
                ingest_processed_files += 1
                continue
            # Determine selective page changes
            changed_pages: List[int] = []
            if prev_page_map and prev_entry and prev_entry.get('schema_version') == SCHEMA_VERSION:
                for idx, pg_md5 in enumerate(page_md5s, start=1):
                    prev_pg = prev_page_map.get(idx)
                    if not prev_pg or prev_pg.get('page_md5') != pg_md5:
                        changed_pages.append(idx)
            else:
                # No prior per-page info -> treat all as changed if file changed
                changed_pages = list(range(1, len(page_md5s)+1)) if not unchanged_whole else []
            # If whole file unchanged, skip
            if unchanged_whole:
                ingest_skipped_files += 1
                ingest_processed_files += 1
                updated_manifest_files[rel] = prev_entry  # retain existing rich manifest
                print(f"[INFO] Skipped unchanged file (schema+hash match): {rel}")
                content = "\n".join(pages)
                structured_doc_meta = {"filename": os.path.basename(fpath), "content_length": len(content), "file_type": ext}
                documents.append({"content": content, "filename": rel, "structured": str(structured_doc_meta)})
                continue
            # Partial reindex if some pages changed but not all
            partial_reindex = prev_entry is not None and changed_pages and len(changed_pages) < len(page_md5s)
            if partial_reindex:
                delete_file_pages_points(rel, changed_pages)
                ingest_pages_reindexed += len(changed_pages)
            elif prev_entry:  # full reindex
                delete_file_points(rel)
            content = "\n".join(pages)
            first_line = content.split('\n')[0] if content else ""
            structured_doc_meta = {"filename": os.path.basename(fpath), "first_line": first_line[:100], "content_length": len(content), "file_type": ext}
            documents.append({"content": content, "filename": rel, "structured": str(structured_doc_meta)})
            # Chunking
            if structured_pages_blocks:
                chunk_list = _structure_aware_chunk(structured_pages_blocks)
            else:
                chunk_list = chunk_text(pages)
            # Filter chunks for partial reindex
            if partial_reindex:
                changed_set = set(changed_pages)
                chunk_list_to_embed = [c for c in chunk_list if any(p in changed_set for p in (c.get('pages') or [c.get('page')]))]
            else:
                chunk_list_to_embed = chunk_list
            chunk_md5s_all_previous = set(prev_entry.get('chunk_md5s', [])) if prev_entry else set()
            new_chunk_md5s: List[str] = []
            if embedder is not None and chunk_list_to_embed:
                points = []
                for ch in chunk_list_to_embed:
                    try:
                        embedding = embedder.encode(ch["text"]).tolist()
                        base_pages = ch.get('pages') or [ch.get('page')]
                        pages_key = ','.join(str(p) for p in base_pages)
                        chunk_uuid_seed = f"{fpath}#pages={pages_key}#chunk={ch['chunk_index']}#md5={ch['md5']}#schema={SCHEMA_VERSION}"
                        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_uuid_seed))
                        payload = {
                            "filename": rel,
                            "page": ch.get("page"),
                            "pages": base_pages,
                            "page_range": ch.get("page_range"),
                            "chunk_index": ch["chunk_index"],
                            "text": ch["text"],
                            "md5": ch["md5"],
                            "structured": str(structured_doc_meta),
                            "schema_version": SCHEMA_VERSION,
                        }
                        if 'heading_path' in ch:
                            payload['heading_path'] = ch['heading_path']
                        if 'block_types' in ch:
                            payload['block_types'] = ch['block_types']
                        # Include table metadata if this chunk is a single table block
                        if 'block_types' in ch and len(ch['block_types']) == 1 and ch['block_types'][0] == 'table':
                            payload['table_markdown'] = ch['text'][:5000]
                        points.append(qdrant_models.PointStruct(id=point_id, vector=embedding, payload=payload))
                        new_chunk_md5s.append(ch['md5'])
                    except Exception as embed_error:
                        print(f"[WARNING] Could not embed chunk {ch.get('chunk_index')} of {rel}: {embed_error}")
                if points:
                    try:
                        qdrant_client.upsert(collection_name=QDRANT_COLLECTION, points=points)
                        action = "partially reindexed" if partial_reindex else "indexed"
                        print(f"[INFO] {action} {len(points)} chunks for {rel} (schema {SCHEMA_VERSION})")
                    except Exception as up_err:
                        print(f"[ERROR] Upsert failed for {fpath}: {up_err}")
            elif embedder is None:
                print(f"[INFO] Skipping vector storage for {rel} (no embedder available)")
            else:
                print(f"[WARNING] No chunks produced for {rel}")
            # Merge chunk md5s
            final_chunk_md5s = list(chunk_md5s_all_previous.union(set(new_chunk_md5s))) if partial_reindex else new_chunk_md5s
            pages_meta = [{"page": i+1, "page_md5": page_md5s[i], "num_blocks": per_page_num_blocks[i], "ocr": False} for i in range(len(page_md5s))]
            block_index_hash = hashlib.md5(''.join(block_md5s).encode('utf-8')).hexdigest() if block_md5s else ''
            tables_count = 0
            if structured_pages_blocks and 'table' in [b.get('type') for page_b in structured_pages_blocks for b in page_b]:
                tables_count = sum(1 for page_b in structured_pages_blocks for b in page_b if b.get('type') == 'table')
            updated_manifest_files[rel] = {
                "file_md5": file_md5,
                "num_chunks": len(final_chunk_md5s),
                "chunk_md5s": final_chunk_md5s,
                "schema_version": SCHEMA_VERSION,
                "pages": pages_meta,
                "header_patterns": header_patterns,
                "footer_patterns": footer_patterns,
                "block_index_hash": block_index_hash,
                "counts": {"tables": tables_count, "equations": 0, "figures": 0, "ocr_pages": 0, "chars": sum(len(p) for p in pages)}
            }
        except Exception as e:
            print(f"[ERROR] Failed to process {fpath}: {e}")
        ingest_processed_files += 1

    # Detect removed files
    previous_rel_set = set(updated_manifest_files.keys())
    removed = previous_rel_set - current_rel_set
    if removed:
        for rel in removed:
            delete_file_points(rel)
            updated_manifest_files.pop(rel, None)
            ingest_removed_files += 1
            print(f"[INFO] Removed stale file entry: {rel}")

    manifest['files'] = updated_manifest_files
    _save_manifest(manifest)


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
    if embedder is not None and qdrant_client is not None and QDRANT_COLLECTION in [c.name for c in qdrant_client.get_collections().collections]:
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
        "skipped": ingest_skipped_files,
        "removed": ingest_removed_files,
        "pages_reindexed": ingest_pages_reindexed,
        "percent": int(100 * ingest_processed_files / ingest_total_files) if ingest_total_files else 100
    }

# Endpoint to reindex a single file (force re-embed)
@app.post("/reindex_file")
def reindex_file(filename: str):
    """Force reindex a single file (relative to data dir)."""
    target_path = os.path.join(DATA_DIR, filename)
    if not os.path.isfile(target_path):
        return {"error": "File not found"}
    # Load manifest and wipe existing entry so load_documents reprocesses only that file
    manifest = _load_manifest()
    files = manifest.get('files', {})
    if filename in files:
        delete_file_points(filename)
        files.pop(filename, None)
        manifest['files'] = files
        _save_manifest(manifest)
    # Process only this file by temporarily limiting patterns
    # Simplest: call load_documents (will process all). For large corpora implement targeted path logic.
    load_documents()
    return {"status": "reindexed", "file": filename}


# Phase 1: structured PDF extraction with PyMuPDF

def _structured_parse_pdf(fpath: str):
    """Return structured representation if PyMuPDF available.
    Returns dict with pages_blocks, page_texts, header/footer lines, per-page block counts, block md5s.
    """
    if not _PYMUPDF_AVAILABLE:
        return None
    try:
        if fitz is None:
            return None
        doc = fitz.open(fpath)
    except Exception as e:
        print(f"[WARNING] PyMuPDF failed to open {fpath}: {e}")
        return None
    raw_pages_blocks = []
    font_sizes = []
    # Explicit index iteration to satisfy static analyzers (PyMuPDF Document is iterable but some type checkers flag it)
    for page_index in range(len(doc)):
        page = doc[page_index]
        try:
            page_dict = page.get_text("dict")  # type: ignore[attr-defined]
            page_blocks = []
            for blk in page_dict.get("blocks", []):
                if 'lines' not in blk:
                    continue
                spans = []
                for line in blk.get('lines', []):
                    for span in line.get('spans', []):
                        text = span.get('text', '').strip('\n')
                        if text:
                            spans.append((text, span.get('size', 0)))
                if not spans:
                    continue
                text_join = ' '.join(s[0] for s in spans).strip()
                text_join = normalize_block_text(text_join)
                if not text_join:
                    continue
                avg_size = sum(s[1] for s in spans) / len(spans)
                font_sizes.append(avg_size)
                page_blocks.append({
                    'page': page_index + 1,
                    'text': text_join,
                    'font_size': avg_size,
                    'type': 'paragraph',
                    'heading_level': None
                })
            raw_pages_blocks.append(page_blocks)
        except Exception as pe:
            print(f"[WARNING] PyMuPDF parse error page {page_index+1} {fpath}: {pe}")
            raw_pages_blocks.append([])
    if not raw_pages_blocks:
        return None
    if not font_sizes:
        return None
    font_sizes_sorted = sorted(font_sizes)
    median_size = font_sizes_sorted[len(font_sizes_sorted)//2]
    heading_threshold = median_size * 1.15
    for page_blocks in raw_pages_blocks:
        for blk in page_blocks:
            if blk['font_size'] >= heading_threshold and len(blk['text']) < 120:
                blk['type'] = 'heading'
                ratio = blk['font_size'] / median_size if median_size else 1.0
                if ratio > 1.6:
                    blk['heading_level'] = 1
                elif ratio > 1.4:
                    blk['heading_level'] = 2
                elif ratio > 1.25:
                    blk['heading_level'] = 3
                else:
                    blk['heading_level'] = 4
    # --- Basic table extraction (Phase 2) ---
    tables_per_page = [[] for _ in range(len(raw_pages_blocks))]
    tables_count = 0
    if ENABLE_TABLES:
        try:
            with pdfplumber.open(fpath) as pdf_tbl:
                for p_idx in range(min(len(pdf_tbl.pages), len(raw_pages_blocks))):
                    try:
                        page_obj = pdf_tbl.pages[p_idx]
                        extracted_tables = page_obj.extract_tables() or []
                        for t in extracted_tables:
                            if not t or not any(any(cell and cell.strip() for cell in row) for row in t):
                                continue
                            # Convert to markdown
                            rows = [[(cell or '').strip() for cell in row] for row in t]
                            n_cols = max(len(r) for r in rows)
                            # Pad rows
                            for r in rows:
                                while len(r) < n_cols:
                                    r.append('')
                            header = rows[0]
                            md_lines = ['|' + '|'.join(c or ' ' for c in header) + '|']
                            md_lines.append('|' + '|'.join(['---']*n_cols) + '|')
                            for data_row in rows[1:]:
                                md_lines.append('|' + '|'.join(c or ' ' for c in data_row) + '|')
                            table_markdown = '\n'.join(md_lines)
                            table_md5 = hashlib.md5(table_markdown.encode('utf-8')).hexdigest()
                            tables_per_page[p_idx].append({
                                'page': p_idx + 1,
                                'text': table_markdown,
                                'font_size': 0,
                                'type': 'table',
                                'heading_level': None,
                                'table_markdown': table_markdown,
                                'n_rows': len(rows),
                                'n_cols': n_cols,
                                'table_md5': table_md5
                            })
                            tables_count += 1
                    except Exception as tpe:
                        print(f"[WARNING] Table extraction failed on page {p_idx+1} {fpath}: {tpe}")
        except Exception as te:
            print(f"[INFO] pdfplumber table extraction unavailable for {fpath}: {te}")
    # Merge table blocks at end of each page (simple strategy; future: preserve order via bbox)
    for idx, page_blocks in enumerate(raw_pages_blocks):
        if tables_per_page[idx]:
            page_blocks.extend(tables_per_page[idx])
    # --- End table extraction ---
    num_pages = len(raw_pages_blocks)
    first_counts: Dict[str,int] = {}
    last_counts: Dict[str,int] = {}
    for page_blocks in raw_pages_blocks:
        if not page_blocks:
            continue
        first_line = page_blocks[0]['text'][:120].strip()
        last_line = page_blocks[-1]['text'][:120].strip()
        if first_line:
            first_counts[first_line] = first_counts.get(first_line, 0) + 1
        if last_line:
            last_counts[last_line] = last_counts.get(last_line, 0) + 1
    freq_cut = int(num_pages * HEADER_FOOTER_FREQ_THRESHOLD)
    header_lines = {t for t,c in first_counts.items() if c >= freq_cut}
    footer_lines = {t for t,c in last_counts.items() if c >= freq_cut}
    cleaned_pages_blocks = []
    for page_blocks in raw_pages_blocks:
        new_blocks = []
        for i, blk in enumerate(page_blocks):
            txt120 = blk['text'][:120].strip()
            if i == 0 and txt120 in header_lines:
                continue
            if i == len(page_blocks)-1 and txt120 in footer_lines:
                continue
            new_blocks.append(blk)
        cleaned_pages_blocks.append(new_blocks)
    page_texts = ['\n'.join(b['text'] for b in page_blocks) for page_blocks in cleaned_pages_blocks]
    # Block md5s flattened (normalized already)
    block_md5s: List[str] = []
    per_page_counts: List[int] = []
    for page_blocks in cleaned_pages_blocks:
        per_page_counts.append(len(page_blocks))
        for blk in page_blocks:
            block_md5s.append(hashlib.md5(blk['text'].encode('utf-8')).hexdigest())
    return {
        'pages_blocks': cleaned_pages_blocks,
        'page_texts': page_texts,
        'header_lines': list(header_lines),
        'footer_lines': list(footer_lines),
        'per_page_num_blocks': per_page_counts,
        'block_md5s': block_md5s,
        'tables_count': tables_count
    }


def _structure_aware_chunk(pages_blocks):
    """Chunk blocks respecting headings and token limits."""
    chunks = []
    heading_path = []
    chunk_blocks_accum = []
    chunk_tokens = 0
    chunk_index_counter = 0
    def flush():
        nonlocal chunk_blocks_accum, chunk_tokens, chunk_index_counter
        if not chunk_blocks_accum:
            return
        text = '\n'.join(b['text'] for b in chunk_blocks_accum)
        md5 = hashlib.md5(text.encode('utf-8')).hexdigest()
        pages = [b['page'] for b in chunk_blocks_accum]
        block_types = [b['type'] for b in chunk_blocks_accum]
        pages_sorted = sorted(set(pages))
        chunks.append({
            'page': min(pages_sorted),
            'pages': pages_sorted,
            'chunk_index': chunk_index_counter,
            'text': text,
            'md5': md5,
            'heading_path': ' > '.join(heading_path),
            'block_types': block_types,
            'page_range': f"{min(pages_sorted)}-{max(pages_sorted)}" if len(pages_sorted)>1 else str(pages_sorted[0])
        })
        chunk_index_counter += 1
        chunk_blocks_accum = []
        chunk_tokens = 0
    for page_blocks in pages_blocks:
        for blk in page_blocks:
            # Update heading path
            if blk['type'] == 'heading':
                # Flush current chunk before starting new section
                flush()
                # Trim heading path to level-1 + (level-1) items
                level = blk.get('heading_level', 4) or 4
                # Ensure heading_path list length >= level-1 then set
                heading_path = heading_path[:level-1]
                heading_path.append(blk['text'])
                continue  # headings not embedded directly; or embed? choose skip to avoid duplication
            # Merge short paragraphs logic: if last block short, combine naturally by not flushing
            blk_tokens = len(blk['text'].split())
            if chunk_tokens + blk_tokens > MAX_TOKENS or (chunk_tokens > 0 and chunk_tokens >= TARGET_TOKENS):
                flush()
            chunk_blocks_accum.append(blk)
            chunk_tokens += blk_tokens
    flush()
    return chunks

if __name__ == "__main__":
    import uvicorn
    # Run the FastAPI app if executed directly
    uvicorn.run(app, host="0.0.0.0", port=8000)