from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import os
from typing import List, Dict, Any, Set, Optional  # Added Optional
import json
import requests
import glob
import PyPDF2
import pdfplumber
from docx import Document
import uuid  # Added for valid UUID point ids
import hashlib  # For chunk content hashing
from contextlib import asynccontextmanager  # re-add lifespan decorator after earlier edit
import atexit  # Safe shutdown handler
import unicodedata  # For block normalization
import statistics  # For table summarization stats
from collections import OrderedDict  # Added for rerank LRU cache
# Removed LangExtract import - no longer needed
from qdrant_client import QdrantClient, models as qdrant_models
from sentence_transformers import SentenceTransformer
import sys
from pathlib import Path
import time  # For timing operations

# Load environment variables from config.env if it exists
def load_env_from_file(env_file):
    if os.path.exists(env_file):
        print(f"Loading environment from {env_file}")
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                key, value = line.split('=', 1)
                os.environ[key] = value

# Try to load from different possible locations
config_paths = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config.env'),  # Project root
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.env'),  # Backend folder
    os.path.join(os.getcwd(), 'config.env'),  # Current working directory
    os.path.join(Path(sys.executable).parent, 'config.env')  # Executable directory (for PyInstaller)
]

for config_path in config_paths:
    if os.path.exists(config_path):
        load_env_from_file(config_path)
        break
from threading import Lock
import numpy as np  # ensure np for similarity array handling
import pickle
from scipy import sparse as _scipy_sparse  # for persistence of sparse matrix

# ---------------- New: Reload / Supervisor Detection ---------------- #
IS_RELOAD_SUPERVISOR = os.environ.get('WATCHGOD_RELOADER') == 'true'
# If running under uvicorn --reload and using embedded Qdrant, concurrent access will fail.
# We will skip heavy init in supervisor and optionally auto-disable reload for worker if embedded path in use.
FORCE_DISABLE_RELOAD_FOR_EMBEDDED = os.environ.get('FORCE_DISABLE_RELOAD_FOR_EMBEDDED', '1') == '1'
# --------------------------------------------------------------- #

# ---------------- New: Deferred heavy init flag ---------------- #
DEFER_STARTUP_INIT = os.environ.get('DEFER_STARTUP_INIT', '1') == '1'
_STARTUP_INITIALIZED = False
# --------------------------------------------------------------- #

# ---------------- New: BM25 Globals ---------------- #
SPARSE_METHOD = os.environ.get('SPARSE_METHOD', 'tfidf').lower()  # 'tfidf' or 'bm25'
_bm25_corpus_tokens: List[List[str]] = []
_bm25_doc_freq: Dict[str, int] = {}
_bm25_inverted_index: Dict[str, List[tuple]] = {}
_bm25_doc_len: List[int] = []
_bm25_avgdl: float = 0.0
_bm25_k1 = float(os.environ.get('BM25_K1', '1.5'))
_bm25_b = float(os.environ.get('BM25_B', '0.75'))
# --------------------------------------------------- #

# Sparse / rerank imports
try:
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
    _SPARSE_OK = True
except Exception:
    TfidfVectorizer = None  # type: ignore
    cosine_similarity = None  # type: ignore
    _SPARSE_OK = False
try:
    from sentence_transformers import CrossEncoder
    _CROSS_ENCODER_MODEL_NAME = os.environ.get('CROSS_ENCODER_MODEL', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
    _ENABLE_RERANK = os.environ.get('ENABLE_RERANK', '0') == '1'
    cross_encoder = CrossEncoder(_CROSS_ENCODER_MODEL_NAME) if _ENABLE_RERANK else None
except Exception:
    cross_encoder = None
    _ENABLE_RERANK = False

ENABLE_SPARSE = os.environ.get('ENABLE_SPARSE', '1') == '1'
SPARSE_MAX_DOCS = int(os.environ.get('SPARSE_MAX_DOCS', '50000'))
RRF_K = int(os.environ.get('RRF_K', '60'))
TABLE_SUMMARY_MAX_ROWS = int(os.environ.get('TABLE_SUMMARY_MAX_ROWS', '8'))
TABLE_SUMMARY_MAX_COLS = int(os.environ.get('TABLE_SUMMARY_MAX_COLS', '8'))
TABLE_CSV_MAX_CHARS = int(os.environ.get('TABLE_CSV_MAX_CHARS', '20000'))  # New: cap stored CSV size
ENABLE_SPARSE_PERSIST = os.environ.get('ENABLE_SPARSE_PERSIST', '1') == '1'
SPARSE_PERSIST_DIR = os.environ.get('SPARSE_PERSIST_DIR', os.path.join(os.path.dirname(__file__), 'sparse_index'))
# Rerank caching controls
RERANK_CACHE_SIZE = int(os.environ.get('RERANK_CACHE_SIZE', '5000'))
RERANK_TOP_N = int(os.environ.get('RERANK_TOP_N', '50'))  # max candidates to send to cross-encoder
# --------------------------------------------------- #

# New: context assembly / diversity flags
MAX_CONTEXT_TOKENS = int(os.environ.get('MAX_CONTEXT_TOKENS', '1800'))
ENABLE_DIVERSITY = os.environ.get('ENABLE_DIVERSITY', '1') == '1'
TABLE_KEYWORDS = [
    'table', 'tabulate', 'dataset', 'results', 'values', 'fig.', 'figure', 'data', 'statistics'
]

_sparse_lock = Lock()
_sparse_vectorizer = None
_sparse_matrix = None
_sparse_payload_refs: List[dict] = []  # store minimal chunk refs
# Rerank LRU cache: key=(query_md5 + chunk_md5) -> score
_rerank_cache: 'OrderedDict[str,float]' = OrderedDict()

# Cross-encoder rerank helper (LRU cache)

def _apply_cross_encoder_rerank(query: str, fused: List[dict]):
    if not fused or cross_encoder is None:
        return
    candidates = fused[:RERANK_TOP_N]
    q_md5 = hashlib.md5(query.encode('utf-8')).hexdigest()
    pairs = []
    idxs = []
    for i, item in enumerate(candidates):
        chunk_hash = item.get('md5') or item.get('id') or str(i)
        cache_key = f"{q_md5}|{chunk_hash}"
        item['_rerank_cache_key'] = cache_key
        cached = _rerank_cache.get(cache_key)
        if cached is not None:
            item['rerank_score'] = cached
            _rerank_cache.move_to_end(cache_key)
            continue
        text = item.get('content') or item.get('text') or ''
        if not text:
            item['rerank_score'] = 0.0
            continue
        pairs.append((query, text))
        idxs.append(i)
    if pairs:
        try:
            scores = cross_encoder.predict(pairs)
            for local_i, score in zip(idxs, scores):
                cand = candidates[local_i]
                cand['rerank_score'] = float(score)
                ck = cand.get('_rerank_cache_key')
                if ck:
                    _rerank_cache[ck] = float(score)
        except Exception as e:
            print(f"[WARNING] Cross-encoder prediction failed: {e}")
    while len(_rerank_cache) > RERANK_CACHE_SIZE:
        try:
            _rerank_cache.popitem(last=False)
        except Exception:
            break
    for item in candidates:
        if 'rerank_score' not in item:
            item['rerank_score'] = 0.0
    fused.sort(key=lambda x: (-(x.get('rerank_score', 0.0)), -(x.get('rrf_score', 0.0))))

# ---------------- Sparse Persistence Helpers ---------------- #

def _persist_sparse_index():
    if not ENABLE_SPARSE_PERSIST:
        return
    # TF-IDF persistence
    try:
        if _sparse_vectorizer is not None and _sparse_matrix is not None:
            os.makedirs(SPARSE_PERSIST_DIR, exist_ok=True)
            with open(os.path.join(SPARSE_PERSIST_DIR, 'vectorizer.pkl'), 'wb') as vf:
                pickle.dump(_sparse_vectorizer, vf)
            sparse_path = os.path.join(SPARSE_PERSIST_DIR, 'matrix.npz')
            _scipy_sparse.save_npz(sparse_path, _sparse_matrix)
            with open(os.path.join(SPARSE_PERSIST_DIR, 'payload_refs.json'), 'w', encoding='utf-8') as pf:
                json.dump(_sparse_payload_refs, pf)
    except Exception as e:
        print(f"[WARNING] Failed to persist TF-IDF sparse index: {e}")
    # BM25 persistence
    try:
        if SPARSE_METHOD == 'bm25' and _bm25_corpus_tokens:
            os.makedirs(SPARSE_PERSIST_DIR, exist_ok=True)
            bm25_state = {
                'corpus_tokens': _bm25_corpus_tokens,
                'doc_freq': _bm25_doc_freq,
                'inverted_index': _bm25_inverted_index,
                'doc_len': _bm25_doc_len,
                'avgdl': _bm25_avgdl,
                'payload_refs': _sparse_payload_refs,
                'k1': _bm25_k1,
                'b': _bm25_b
            }
            with open(os.path.join(SPARSE_PERSIST_DIR, 'bm25_index.pkl'), 'wb') as bf:
                pickle.dump(bm25_state, bf)
    except Exception as e:
        print(f"[WARNING] Failed to persist BM25 index: {e}")
    try:
        print(f"[INFO] Persisted sparse index to {SPARSE_PERSIST_DIR}")
    except Exception:
        pass

def _load_sparse_index_from_disk():
    global _sparse_vectorizer, _sparse_matrix, _sparse_payload_refs
    global _bm25_corpus_tokens, _bm25_doc_freq, _bm25_inverted_index, _bm25_doc_len, _bm25_avgdl
    if not ENABLE_SPARSE_PERSIST:
        return
    # Load TF-IDF if present (even if SPARSE_METHOD is bm25, for debugging)
    try:
        vec_path = os.path.join(SPARSE_PERSIST_DIR, 'vectorizer.pkl')
        mat_path = os.path.join(SPARSE_PERSIST_DIR, 'matrix.npz')
        refs_path = os.path.join(SPARSE_PERSIST_DIR, 'payload_refs.json')
        if os.path.isfile(vec_path) and os.path.isfile(mat_path) and os.path.isfile(refs_path):
            with open(vec_path, 'rb') as vf:
                _sparse_vectorizer = pickle.load(vf)
            _sparse_matrix = _scipy_sparse.load_npz(mat_path)
            with open(refs_path, 'r', encoding='utf-8') as pf:
                _sparse_payload_refs = json.load(pf)
            print(f"[INFO] Loaded persisted TF-IDF index: docs={len(_sparse_payload_refs)} features={_sparse_matrix.shape[1] if _sparse_matrix is not None else 0}")
    except Exception as e:
        print(f"[WARNING] Could not load persisted TF-IDF index: {e}")
    # Load BM25 if requested
    if SPARSE_METHOD == 'bm25':
        try:
            bm25_path = os.path.join(SPARSE_PERSIST_DIR, 'bm25_index.pkl')
            if os.path.isfile(bm25_path):
                with open(bm25_path, 'rb') as bf:
                    state = pickle.load(bf)
                _bm25_corpus_tokens = state.get('corpus_tokens', [])
                _bm25_doc_freq = state.get('doc_freq', {})
                _bm25_inverted_index = state.get('inverted_index', {})
                _bm25_doc_len = state.get('doc_len', [])
                _bm25_avgdl = state.get('avgdl', 0.0)
                _sparse_payload_refs = state.get('payload_refs', [])  # reuse refs
                print(f"[INFO] Loaded persisted BM25 index: docs={len(_bm25_corpus_tokens)} avgdl={_bm25_avgdl:.2f}")
        except Exception as e:
            print(f"[WARNING] Could not load persisted BM25 index: {e}")
# ---------------- End Sparse Persistence Helpers ---------------- #

# Define lifespan before app instantiation (ensure AVAILABLE global symbols referenced inside) 
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _STARTUP_INITIALIZED, embedder, qdrant_client
    # Perform heavy initialization only once at real server startup (not in reloader supervisor)
    if DEFER_STARTUP_INIT and not _STARTUP_INITIALIZED:
        try:
            # Initialize Qdrant client
            if qdrant_client is None:
                _initialize_qdrant_client()
            # Load embedding model lazily
            if embedder is None:
                default_model_path_local = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "all-MiniLM-L6-v2")
                model_path_local = os.environ.get("EMBED_MODEL_PATH", default_model_path_local)
                print(f"[INIT] Deferred model load from: {model_path_local}")
                try:
                    embedder = SentenceTransformer(model_path_local)
                    print(f"[INIT] Successfully loaded embedding model (deferred) from: {model_path_local}")
                except Exception as e:
                    print(f"[WARNING] Deferred model load failed: {e}")
                    embedder = None
            _STARTUP_INITIALIZED = True
        except Exception as e:
            print(f"[WARNING] Deferred heavy init encountered error: {e}")
    # Only ingest after heavy init to ensure embedder/qdrant available
    if DEFER_STARTUP_INIT:
        if 'AUTO_INGEST' in globals() and globals().get('AUTO_INGEST'):
            try:
                load_documents()
            except Exception as e:
                print(f"[WARNING] Initial load_documents failed (deferred): {e}")
        else:
            print("[INFO] AUTO_INGEST disabled; skipping ingestion at startup (deferred)")
    else:
        # Original path (no defer) retains previous behavior
        if 'AUTO_INGEST' in globals() and globals().get('AUTO_INGEST'):
            try:
                load_documents()
            except Exception as e:
                print(f"[WARNING] Initial load_documents failed: {e}")
        else:
            print("[INFO] AUTO_INGEST disabled; skipping ingestion at startup")
    # Attempt to load persisted sparse index after initial load
    _load_sparse_index_from_disk()
    try:
        yield
    finally:
        try:
            _safe_close_qdrant()
        except Exception:
            pass

# ---------------- Sparse / Hybrid Retrieval Helpers (Phase 3) ---------------- #

def _tokenize(text: str) -> List[str]:
    return [t for t in text.lower().split() if t]

def _build_sparse_index():
    # If vectorizer class unavailable and method tfidf, skip
    if SPARSE_METHOD == 'tfidf' and TfidfVectorizer is None:
        return
    """Build a sparse index over all chunk texts currently in Qdrant.
    Supports TF-IDF (default) or BM25 based on SPARSE_METHOD env variable.
    Thread-safe under _sparse_lock."""
    global _sparse_vectorizer, _sparse_matrix, _sparse_payload_refs
    global _bm25_corpus_tokens, _bm25_doc_freq, _bm25_inverted_index, _bm25_doc_len, _bm25_avgdl
    if not (ENABLE_SPARSE and _SPARSE_OK and embedder is not None):
        # Allow BM25 even if scikit is missing (we don't depend on sklearn for BM25)
        if SPARSE_METHOD != 'bm25':
            return
    if qdrant_client is None:
        print("[INFO] Qdrant client not available - skipping sparse index build")
        return
    try:
        with _sparse_lock:
            texts: List[str] = []
            refs: List[dict] = []
            if QDRANT_COLLECTION not in [c.name for c in qdrant_client.get_collections().collections]:
                print("[INFO] Qdrant collection not found - skipping sparse index build")
                return
            offset = None
            batch = 512
            count = 0
            while True:
                points, offset = qdrant_client.scroll(collection_name=QDRANT_COLLECTION, scroll_filter=None, limit=batch, with_payload=True, offset=offset)
                if not points:
                    break
                for pt in points:
                    if count >= SPARSE_MAX_DOCS:
                        break
                    payload = pt.payload or {}
                    text = payload.get('text')
                    if not text:
                        continue
                    texts.append(text)
                    refs.append({
                        'id': str(pt.id),
                        'filename': payload.get('filename'),
                        'page': payload.get('page'),
                        'chunk_index': payload.get('chunk_index'),
                        'heading_path': payload.get('heading_path'),
                        'block_types': payload.get('block_types'),
                        'md5': payload.get('md5'),  # added for rerank caching key
                        'text': text
                    })
                    count += 1
                if count >= SPARSE_MAX_DOCS or offset is None:
                    break
            if not texts:
                print("[INFO] No texts found for sparse index build")
                return
            _sparse_payload_refs = refs
            if SPARSE_METHOD == 'tfidf':
                if TfidfVectorizer is None:  # Defensive guard for type checker
                    return
                try:
                    _sparse_vectorizer = TfidfVectorizer(max_df=0.9, min_df=1, ngram_range=(1,2))
                    _sparse_matrix = _sparse_vectorizer.fit_transform(texts)
                    print(f"[INFO] Built TF-IDF index: docs={len(refs)} features={_sparse_matrix.shape[1]}")
                except Exception as ve:
                    print(f"[WARNING] TF-IDF vectorizer build failed: {ve}")
                    return
            else:  # BM25
                _bm25_corpus_tokens = []
                _bm25_doc_freq = {}
                _bm25_inverted_index = {}
                _bm25_doc_len = []
                for doc_idx, txt in enumerate(texts):
                    tokens = _tokenize(txt)
                    _bm25_corpus_tokens.append(tokens)
                    _bm25_doc_len.append(len(tokens))
                    tf_local: Dict[str, int] = {}
                    for t in tokens:
                        tf_local[t] = tf_local.get(t, 0) + 1
                    for term, tf in tf_local.items():
                        _bm25_doc_freq[term] = _bm25_doc_freq.get(term, 0) + 1
                        _bm25_inverted_index.setdefault(term, []).append((doc_idx, tf))
                N = len(_bm25_corpus_tokens)
                _bm25_avgdl = sum(_bm25_doc_len)/N if N else 0.0
                print(f"[INFO] Built BM25 index: docs={N} avgdl={_bm25_avgdl:.2f} vocab={len(_bm25_doc_freq)}")
            _persist_sparse_index()
    except Exception as e:
        print(f"[WARNING] Failed to build sparse index: {e}")


def _sparse_search(query: str, top_k: int) -> List[dict]:
    if SPARSE_METHOD == 'bm25':
        return _bm25_search(query, top_k)
    # TF-IDF path (existing)
    if TfidfVectorizer is None or cosine_similarity is None:
        return []
    try:
        q_vec = _sparse_vectorizer.transform([query]) if _sparse_vectorizer is not None else None
        if q_vec is None or _sparse_matrix is None:
            return []
        sims_arr = cosine_similarity(q_vec, _sparse_matrix).ravel() if cosine_similarity is not None else None
        if sims_arr is None or getattr(sims_arr, 'size', 0) == 0:
            return []
        sims = np.asarray(sims_arr)
        top_idx = sims.argsort()[::-1][:top_k]
        results = []
        for rank, idx in enumerate(top_idx):
            ref = _sparse_payload_refs[idx]
            results.append({
                'id': ref['id'],
                'filename': ref['filename'],
                'page': ref['page'],
                'chunk_index': ref['chunk_index'],
                'heading_path': ref.get('heading_path'),
                'block_types': ref.get('block_types'),
                'content': ref['text'],
                'md5': ref.get('md5'),  # propagate md5
                'score': float(sims[idx]),
                'source': 'sparse_tfidf',
                'rank': rank+1
            })
        return results
    except Exception as e:
        print(f"[WARNING] Sparse search failed: {e}")
        return []


def _bm25_search(query: str, top_k: int) -> List[dict]:
    if SPARSE_METHOD != 'bm25' or not _bm25_corpus_tokens:
        return []
    try:
        q_terms = _tokenize(query)
        if not q_terms:
            return []
        scores: Dict[int, float] = {}
        N = len(_bm25_corpus_tokens)
        avgdl = _bm25_avgdl if _bm25_avgdl else 1.0
        k1 = _bm25_k1
        b = _bm25_b
        seen_docs: Set[int] = set()
        for qt in q_terms:
            postings = _bm25_inverted_index.get(qt)
            if not postings:
                continue
            df = _bm25_doc_freq.get(qt, 0)
            # BM25 IDF (Robertson/Sparck Jones)
            idf = np.log(1 + (N - df + 0.5)/(df + 0.5)) if df else 0.0
            for doc_idx, tf in postings:
                dl = _bm25_doc_len[doc_idx] or 1
                denom = tf + k1 * (1 - b + b * dl / avgdl)
                score_add = idf * (tf * (k1 + 1)) / denom if denom else 0.0
                scores[doc_idx] = scores.get(doc_idx, 0.0) + score_add
                seen_docs.add(doc_idx)
        if not scores:
            return []
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results: List[dict] = []
        for rank, (doc_idx, sc) in enumerate(ranked, start=1):
            ref = _sparse_payload_refs[doc_idx]
            results.append({
                'id': ref['id'],
                'filename': ref['filename'],
                'page': ref['page'],
                'chunk_index': ref['chunk_index'],
                'heading_path': ref.get('heading_path'),
                'block_types': ref.get('block_types'),
                'content': ref['text'],
                'md5': ref.get('md5'),  # propagate md5
                'score': float(sc),
                'source': 'sparse_bm25',
                'rank': rank
            })
        return results
    except Exception as e:
        print(f"[WARNING] BM25 search failed: {e}")
        return []


def _dense_search_internal(query: str, top_k: int) -> List[dict]:
    if embedder is None or qdrant_client is None:
        return []
    try:
        query_vec = embedder.encode(query).tolist()
        try:
            qp = qdrant_client.query_points(collection_name=QDRANT_COLLECTION, query=query_vec, limit=top_k, with_payload=True)
            hits = qp.points
        except Exception:
            hits = qdrant_client.search(collection_name=QDRANT_COLLECTION, query_vector=query_vec, limit=top_k, with_payload=True)
        out = []
        for rank, hit in enumerate(hits):
            payload = hit.payload or {}
            out.append({
                'id': str(hit.id),
                'filename': payload.get('filename'),
                'page': payload.get('page'),
                'chunk_index': payload.get('chunk_index'),
                'heading_path': payload.get('heading_path'),
                'block_types': payload.get('block_types'),
                'content': payload.get('text', ''),
                'md5': payload.get('md5'),  # for rerank cache
                'score': float(hit.score) if hasattr(hit, 'score') else 0.0,
                'source': 'dense',
                'rank': rank+1
            })
        return out
    except Exception as e:
        print(f"[WARNING] Dense search failed: {e}")
        return []


def _rrf_fuse(dense: List[dict], sparse: List[dict], final_k: int) -> List[dict]:
    """Reciprocal Rank Fusion over two ranked lists."""
    fused: Dict[str, dict] = {}
    for lst in (dense, sparse):
        for item in lst:
            key = item['id']
            rank = item['rank']
            contrib = 1.0 / (RRF_K + rank)
            if key not in fused:
                fused[key] = {**item, 'rrf_score': contrib}
            else:
                fused[key]['rrf_score'] += contrib
    fused_list = list(fused.values())
    fused_list.sort(key=lambda x: (-x['rrf_score'], x['filename'] or '', x['page'] or 0))
    return fused_list[:final_k]


def hybrid_search(query: str, top_k_dense: int = 20, top_k_sparse: int = 20, final_k: int = 5) -> List[dict]:
    dense_hits = _dense_search_internal(query, top_k_dense) if embedder is not None else []
    sparse_hits = _sparse_search(query, top_k_sparse)
    if not dense_hits and not sparse_hits:
        return []
    fused = _rrf_fuse(dense_hits, sparse_hits, final_k * 3)
    # Optional cross-encoder rerank with caching
    if cross_encoder is not None and _ENABLE_RERANK and fused:
        try:
            _apply_cross_encoder_rerank(query, fused)
        except Exception as e:
            print(f"[WARNING] Cross-encoder rerank failed: {e}")
    final = fused[:final_k]
    return [
        {
            'filename': c['filename'],
            'content': c['content'],
            'page': c['page'],
            'chunk_index': c['chunk_index'],
            'heading_path': c.get('heading_path'),
            'block_types': c.get('block_types'),
            'md5': c.get('md5')
        } for c in final
    ]
# ---------------- End Sparse / Hybrid Retrieval Helpers ---------------- #

# Instantiate app here after helper definitions (lifespan previously defined above)
app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Use environment variables with fallbacks for paths
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
documents = []
last_loaded_files = {}

# Progress tracking globals
ingest_total_files = 0
ingest_processed_files = 0

# Qdrant setup (embedded/local mode)
# Use environment variable with fallback for Qdrant path
QDRANT_PATH = os.environ.get("QDRANT_PATH", os.path.join(os.path.dirname(__file__), "qdrant_storage"))
QDRANT_COLLECTION = "documents"
print(f"Using Qdrant storage path: {QDRANT_PATH}" if not IS_RELOAD_SUPERVISOR else "[SUPERVISOR] Detected reloader supervisor; deferring embedded Qdrant init")

# Create Qdrant storage directory if it doesn't exist
os.makedirs(QDRANT_PATH, exist_ok=True)

# Initialize Qdrant client with the configured path
qdrant_client = None

def _initialize_qdrant_client():
    global qdrant_client
    if qdrant_client is not None:
        return qdrant_client
    
    try:
        qdrant_client = QdrantClient(path=QDRANT_PATH)
        print(f"Successfully initialized Qdrant client with path: {QDRANT_PATH}")
        
        # Disable noisy destructor that triggers portalocker ImportError at shutdown
        try:
            qdrant_client.__del__ = lambda self: None  # type: ignore
        except Exception:
            pass
            
        return qdrant_client
    except Exception as e:
        print(f"Error initializing Qdrant client: {e}")
        # If it's a lock error, try to wait and retry once
        if "already accessed by another instance" in str(e):
            import time
            print("Waiting 2 seconds and retrying Qdrant initialization...")
            time.sleep(2)
            try:
                qdrant_client = QdrantClient(path=QDRANT_PATH)
                print(f"Successfully initialized Qdrant client on retry with path: {QDRANT_PATH}")
                try:
                    qdrant_client.__del__ = lambda self: None  # type: ignore
                except Exception:
                    pass
                return qdrant_client
            except Exception as retry_e:
                print(f"Retry failed: {retry_e}")
                qdrant_client = None
        else:
            qdrant_client = None
        return None

# Try initial initialization
if not DEFER_STARTUP_INIT and not IS_RELOAD_SUPERVISOR:
    _initialize_qdrant_client()


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
# Use a relative path as default fallback instead of hardcoded absolute path
default_model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "all-MiniLM-L6-v2")
model_path = os.environ.get("EMBED_MODEL_PATH", default_model_path)
print(f"Attempting to load model from: {model_path}" if not IS_RELOAD_SUPERVISOR else "[SUPERVISOR] Skipping model load (deferred)")
print(f"Path exists: {os.path.exists(model_path)}" if not IS_RELOAD_SUPERVISOR else "")
print(f"Path is absolute: {os.path.isabs(model_path)}" if not IS_RELOAD_SUPERVISOR else "")

try:
    if not DEFER_STARTUP_INIT:  # Only load at import time if not deferring
        # Try to load the model from local path
        embedder = SentenceTransformer(model_path)
        print(f"Successfully loaded local embedding model from: {model_path}")
    else:
        embedder = None  # Will be loaded in lifespan
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
    if qdrant_client is None:
        print(f"[WARNING] Cannot delete vectors for {rel_filename}: Qdrant client not available")
        return
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
    if not pages or qdrant_client is None:
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
    elif qdrant_client is None:
        print("[WARNING] Qdrant client not available - vector operations will be skipped")
    elif embedder is None:
        print("[WARNING] Embedder not available - vector operations will be skipped")
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
            if embedder is not None and qdrant_client is not None and chunk_list_to_embed:
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
                            # Attach summary if available
                            # Attempt to parse summary from first line heuristic
                            if '\n' in ch['text']:
                                first_line = ch['text'].split('\n',1)[0]
                                if 'rows=' in first_line and 'cols=' in first_line:
                                    payload['table_summary'] = first_line[:300]
                            # Propagate table metadata if available from block accumulation
                            if 'table_md5' in ch:
                                payload['table_md5'] = ch['table_md5']
                            if 'table_csv' in ch:
                                payload['table_csv'] = ch['table_csv'][:TABLE_CSV_MAX_CHARS]
                            if 'n_rows' in ch:
                                payload['n_rows'] = ch['n_rows']
                            if 'n_cols' in ch:
                                payload['n_cols'] = ch['n_cols']
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
            elif qdrant_client is None:
                print(f"[INFO] Skipping vector storage for {rel} (Qdrant client not available)")
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
    _build_sparse_index()


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
    if embedder is None or qdrant_client is None:
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
    """Answer generation using Llama3 via Ollama API with adaptive timeout and retry"""
    print(f"[DEBUG] Received question: {question}")
    print(f"[DEBUG] Context docs: {[doc['filename'] for doc in context_docs]}")
    if not context_docs:
        print("[DEBUG] No relevant documents found.")
        return "I couldn't find relevant information to answer your question."

    # Optimize context by limiting the number of documents and snippet length
    max_docs = min(len(context_docs), 8)  # Limit to 8 most relevant documents
    snippet_length = 300  # Initial snippet length
    
    # Adaptive timeout and retry mechanism
    max_retries = 3  # Increased from 2 to 3
    initial_timeout = 90  # Initial timeout in seconds
    min_timeout = 30  # Minimum timeout for subsequent retries
    timeout_reduction_factor = 0.7  # Reduce timeout by 30% on each retry
    
    # Track timeouts to reduce context size on multiple timeouts
    timeout_count = 0
    
    for attempt in range(max_retries + 1):
        # Adjust context size if we've had multiple timeouts
        current_max_docs = max(4, max_docs - timeout_count)  # Reduce docs but keep at least 4
        current_snippet_length = max(150, snippet_length - (timeout_count * 50))  # Reduce snippet length but keep at least 150 chars
        
        # Build context with current parameters
        context_text = "\n\n".join([f"From {doc.get('filename','unknown')} (page {doc.get('page','?')}):\n{doc.get('content','')[:current_snippet_length]}..." 
                                for doc in context_docs[:current_max_docs]])
        
        # Streamlined prompt to reduce size
        prompt = f"Answer the following question using ONLY the provided context. Be concise.\n\nQuestion: {question}\n\nContext:\n{context_text}"
        
        # Calculate current timeout (reduce on each retry)
        current_timeout = max(min_timeout, initial_timeout * (timeout_reduction_factor ** attempt))
        
        print(f"[INFO] Attempt {attempt+1}/{max_retries+1}: Using {current_max_docs} docs, {current_snippet_length} chars per snippet, {current_timeout:.1f}s timeout")
        print(f"[DEBUG] Sending prompt to Ollama: {prompt[:200]}...")
        
        try:
            print(f"[INFO] Sending request to Ollama (attempt {attempt+1}/{max_retries+1})")
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3:latest",
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.7  # Add temperature parameter to potentially speed up generation
                },
                timeout=current_timeout
            )
            print(f"[DEBUG] Ollama response status: {response.status_code}")
            response.raise_for_status()
            data = response.json()
            answer = data.get("response", "")
            
            # Check for empty response
            if not answer.strip():
                print(f"[WARNING] Empty response received on attempt {attempt+1}")
                if attempt < max_retries:
                    timeout_count += 1  # Treat empty response like a timeout
                    continue
                else:
                    return "Error: Received empty response from Ollama. Please try again with a simpler question."
            
            print(f"[DEBUG] Final answer: {answer[:200]}...")
            print(f"[INFO] Ollama response received successfully on attempt {attempt+1}")
            return answer
            
        except requests.exceptions.ConnectionError as e:
            print(f"[ERROR] Connection error on attempt {attempt+1}: {e}")
            # For connection errors, we might need to wait longer for Ollama to recover
            if attempt < max_retries:
                wait_time = (attempt + 1) * 3  # Longer wait for connection issues
                print(f"[INFO] Connection issue - waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                return "Error: Cannot connect to Ollama. Please check if Ollama is running and restart if necessary."
                
        except requests.exceptions.Timeout as e:
            print(f"[WARNING] Timeout on attempt {attempt+1}/{max_retries+1}: {e}")
            timeout_count += 1
            
            if attempt < max_retries:
                # Wait before retrying, with increasing backoff
                wait_time = (attempt + 1) * 2
                print(f"[INFO] Timeout occurred - waiting {wait_time} seconds before retry with reduced context...")
                time.sleep(wait_time)
            else:
                return f"Error: Request to Ollama timed out after {max_retries+1} attempts. Try asking a simpler question or check if Ollama is running properly."
                
        except Exception as e:
            print(f"[ERROR] Unexpected error on attempt {attempt+1}: {e}")
            if attempt < max_retries:
                wait_time = (attempt + 1) * 2
                print(f"[INFO] Error occurred - waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                return f"Error: Unexpected issue when generating answer: {str(e)}. Please try again later."
            import traceback
            traceback.print_exc()
            return f"Error communicating with Ollama: {str(e)}"


@app.get("/ingest")
def ingest():
    load_documents()
    return {"status": "reloaded", "documents_loaded": len(documents)}

@app.get("/health")
def health_check():
    """Check if Ollama and Llama3 are available with detailed diagnostics"""
    try:
        # Use a shorter timeout for health checks
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        response.raise_for_status()
        models = response.json().get("models", [])
        llama3_available = any("llama3" in model.get("name", "") for model in models)
        
        # Check if the model is ready by sending a minimal request
        if llama3_available:
            try:
                # Send a minimal prompt to check if model is responsive
                test_response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": "llama3:latest",
                        "prompt": "Hello",
                        "stream": False,
                        "temperature": 0.7
                    },
                    timeout=3
                )
                test_response.raise_for_status()
                
                # Check for empty response
                response_data = test_response.json()
                if not response_data.get("response"):
                    print("[WARNING] Model returned empty response")
                    model_ready = False
                    model_error = "Model returned empty response"
                else:
                    model_ready = True
                    model_error = None
            except requests.exceptions.Timeout as model_timeout:
                print(f"[WARNING] Model health check timed out: {model_timeout}")
                model_ready = False
                model_error = "Model response timed out"
            except requests.exceptions.ConnectionError as model_conn_err:
                print(f"[WARNING] Model health check connection error: {model_conn_err}")
                model_ready = False
                model_error = "Connection error during model check"
            except Exception as model_err:
                print(f"[WARNING] Model health check failed: {model_err}")
                model_ready = False
                model_error = str(model_err)
        else:
            model_ready = False
            model_error = "Llama3 model not found in available models"
            
        return {
            "ollama_running": True,
            "llama3_available": llama3_available,
            "model_ready": model_ready,
            "available_models": [model.get("name", "") for model in models],
            "model_error": model_error if not model_ready else None
        }
    except requests.exceptions.Timeout:
        return {
            "ollama_running": False,
            "llama3_available": False,
            "model_ready": False,
            "error": "Connection to Ollama timed out",
            "troubleshooting": "Check if Ollama is running but overloaded or unresponsive"
        }
    except requests.exceptions.ConnectionError:
        return {
            "ollama_running": False,
            "llama3_available": False,
            "model_ready": False,
            "error": "Ollama is not running",
            "troubleshooting": "Start Ollama service or check if it's running on a different port"
        }
    except Exception as e:
        return {
            "ollama_running": False,
            "llama3_available": False,
            "model_ready": False,
            "error": str(e),
            "troubleshooting": "Unexpected error, check Ollama installation and configuration"
        }

# Simple LRU cache for recent questions to avoid redundant processing
from functools import lru_cache

@lru_cache(maxsize=20)
def get_context_for_question(question: str, use_hybrid: bool, top_k_dense=6, top_k_sparse=6, final_k=6):
    """Cache context retrieval for similar questions with optimized parameters"""
    if use_hybrid:
        raw_docs = hybrid_search(question, top_k_dense=top_k_dense, top_k_sparse=top_k_sparse, final_k=final_k)
        return assemble_context(question, raw_docs, max_tokens=MAX_CONTEXT_TOKENS)
    elif embedder is not None and qdrant_client is not None and QDRANT_COLLECTION in [c.name for c in qdrant_client.get_collections().collections]:
        raw_docs = vector_search(question, top_k=final_k)
        return assemble_context(question, raw_docs, max_tokens=MAX_CONTEXT_TOKENS)
    else:
        print("[INFO] Using simple search fallback")
        return simple_search(question, documents, top_k=final_k)

@app.post("/ask")
async def ask(request: Request):
    print("[DEBUG] /ask endpoint called")
    start_time = time.time()
    data = await request.json()
    question = data.get("question", "")
    print(f"[DEBUG] Question received: '{question}'")
    if not question:
        print("[DEBUG] No question provided")
        return {"error": "Question is required"}
    
    # Try to initialize Qdrant client if not available
    if qdrant_client is None:
        _initialize_qdrant_client()
    
    # Determine search method
    use_hybrid = ENABLE_SPARSE and _SPARSE_OK and embedder is not None and qdrant_client is not None
    
    try:
        # Get context with caching - optimized parameters for better performance
        context_time_start = time.time()
        try:
            # Use optimized parameters for context retrieval
            context_docs = get_context_for_question(question, use_hybrid)
        except Exception as context_error:
            print(f"[ERROR] Context retrieval failed: {context_error}")
            # Fallback to simpler search method
            context_docs = simple_search(question, documents, top_k=5)
            
        context_time = time.time() - context_time_start
        print(f"[TIMING] Context retrieval took {context_time:.2f}s")
        
        # Generate answer with improved error handling
        answer_time_start = time.time()
        try:
            answer, citations = generate_answer_with_citations(question, context_docs)
        except Exception as answer_error:
            print(f"[ERROR] Answer generation failed: {answer_error}")
            # Force fallback path
            answer = f"Error: Failed to generate answer: {str(answer_error)}"
            citations = []
            
        answer_time = time.time() - answer_time_start
        print(f"[TIMING] Answer generation took {answer_time:.2f}s")
        
        # Fallback if LLM call failed or timed out
        if answer.lower().startswith("error"):  # Ollama error fallback
            print("[WARNING] LLM generation failed, using extractive fallback response")
            # Build extractive summary from top context
            summary_lines = []
            for d in context_docs[:4]:  # Use 4 docs for faster response
                snippet = (d.get('content','') or '')[:120].strip().replace("\n", " ")  # 120 chars per snippet
                summary_lines.append(f"- {d.get('filename')} (page {d.get('page')}): {snippet}...")
            
            # Extract error message for better user feedback
            error_msg = answer.split("Error:", 1)[1].strip() if "Error:" in answer else "Unknown error"
            error_msg = error_msg.split(".", 1)[0] + "." if "." in error_msg else error_msg
            answer = f"I'm sorry, I couldn't generate an answer at this time ({error_msg}). Here are the relevant excerpts:\n" + "\n".join(summary_lines)
        
        # Prepare context snippets for frontend display - optimized to 6 docs
        context_for_display = []
        for doc in context_docs[:6]:  # 6 docs is a good balance
            snippet = (doc.get('content','') or '')[:120].strip()  # 120 chars per snippet
            context_for_display.append({
                "id": doc.get('citation_id', 0),
                "filename": doc.get('filename', 'Unknown'),
                "page": doc.get('page', 0),
                "content": snippet
            })
        
        total_time = time.time() - start_time
        print(f"[TIMING] Total processing took {total_time:.2f}s")
        
        return {
            "answer": answer,
            "context": context_for_display,
            "documents_found": len(context_docs),
            "citations": citations,
            "processing_time": round(total_time, 2)
        }
    except requests.exceptions.Timeout as e:
        print(f"[ERROR] Timeout in /ask endpoint: {e}")
        # Specific handling for timeout errors
        try:
            # Get fewer documents for extractive summary
            context_docs = simple_search(question, documents, top_k=4)  # Use simple search as fallback
            
            # Create a simple extractive summary
            summary = "I'm having trouble generating a complete answer due to a timeout. Here's relevant information:\n\n"
            context_for_display = []
            
            for i, doc in enumerate(context_docs[:4], 1):
                snippet = (doc.get('content', '') or '')[:120].strip()  # Shorter snippets
                summary += f"[{i}] From {doc.get('filename', 'Unknown')} (page {doc.get('page', 0)}): {snippet}\n\n"
                context_for_display.append({
                    "id": i,
                    "filename": doc.get('filename', 'Unknown'),
                    "page": doc.get('page', 0),
                    "content": snippet
                })
            
            return {
                "answer": summary,
                "context": context_for_display,
                "documents_found": len(context_docs),
                "citations": [],
                "error": "The request to Ollama timed out. Try asking a simpler question or check if Ollama is running properly.",
                "processing_time": round(time.time() - start_time, 2)
            }
        except Exception as fallback_error:
            return {
                "answer": "Error: Request to Ollama timed out. I couldn't generate a fallback response either. Please try again with a simpler question.",
                "context": [],
                "documents_found": 0,
                "citations": [],
                "error": f"Timeout: {str(e)}. Fallback error: {str(fallback_error)}",
                "processing_time": round(time.time() - start_time, 2)
            }
    except requests.exceptions.ConnectionError as e:
        print(f"[ERROR] Connection error in /ask endpoint: {e}")
        return {
            "answer": "Error: Could not connect to Ollama. Please check if Ollama is running and restart if necessary.",
            "context": [],
            "documents_found": 0,
            "citations": [],
            "error": "Connection error: Cannot connect to Ollama. Please check if Ollama is running and restart if necessary.",
            "processing_time": round(time.time() - start_time, 2)
        }
    except Exception as e:
        print(f"[ERROR] Error processing question: {str(e)}")
        import traceback
        traceback.print_exc()
        try:
            # Get fewer documents for extractive summary using simple search
            context_docs = simple_search(question, documents, top_k=4)  # Use simple search as fallback
            
            # Create a simple extractive summary
            summary = "I encountered an error while processing your question, but here's relevant information:\n\n"
            context_for_display = []
            
            for i, doc in enumerate(context_docs[:4], 1):
                snippet = (doc.get('content', '') or '')[:120].strip()  # Shorter snippets
                summary += f"[{i}] From {doc.get('filename', 'Unknown')} (page {doc.get('page', 0)}): {snippet}\n\n"
                context_for_display.append({
                    "id": i,
                    "filename": doc.get('filename', 'Unknown'),
                    "page": doc.get('page', 0),
                    "content": snippet
                })
            
            return {
                "answer": summary,
                "context": context_for_display,
                "documents_found": len(context_docs),
                "citations": [],
                "error": f"Error processing your question: {str(e)}",
                "processing_time": round(time.time() - start_time, 2)
            }
        except Exception as fallback_error:
            return {
                "error": f"Error processing your question: {str(e)}. Fallback error: {str(fallback_error)}",
                "answer": "I encountered an error while processing your question. Please try again with a simpler question or check if Ollama is running properly.",
                "context": [],
                "documents_found": 0,
                "citations": [],
                "processing_time": round(time.time() - start_time, 2)
            }

@app.get('/search')
def search(q: str, k_dense: int = 10, k_sparse: int = 10, k_final: int = 10, sparse_method: Optional[str] = None):
    global SPARSE_METHOD
    if sparse_method:
        sm = sparse_method.lower()
        if sm in ('bm25','tfidf') and sm != SPARSE_METHOD:
            print(f"[INFO] Requested sparse_method={sm} differs from active={SPARSE_METHOD}; rebuilding on-the-fly")
            SPARSE_METHOD = sm
            _build_sparse_index()
    if not q:
        return {"error": "q required"}
    dense_hits = _dense_search_internal(q, k_dense)
    sparse_hits = _sparse_search(q, k_sparse)
    fused = _rrf_fuse(dense_hits, sparse_hits, k_final*3) if (dense_hits or sparse_hits) else []
    return {
        'query': q,
        'sparse_method': SPARSE_METHOD,
        'dense': dense_hits[:k_final],
        'sparse': sparse_hits[:k_final],
        'fused': fused[:k_final]
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

@app.get("/ingest_metrics")
def ingest_metrics():
    """Aggregate ingestion metrics from manifest."""
    manifest = _load_manifest()
    files = manifest.get('files', {})
    total_files = len(files)
    total_chunks = sum(f.get('num_chunks', 0) for f in files.values())
    total_tables = sum(f.get('counts', {}).get('tables', 0) for f in files.values())
    total_equations = sum(f.get('counts', {}).get('equations', 0) for f in files.values())
    total_figures = sum(f.get('counts', {}).get('figures', 0) for f in files.values())
    total_chars = sum(f.get('counts', {}).get('chars', 0) for f in files.values())
    ocr_pages = sum(f.get('counts', {}).get('ocr_pages', 0) for f in files.values())
    return {
        "files": total_files,
        "chunks": total_chunks,
        "tables": total_tables,
        "equations": total_equations,
        "figures": total_figures,
        "chars": total_chars,
        "ocr_pages": ocr_pages,
        "avg_chunks_per_file": (total_chunks/total_files) if total_files else 0,
        "avg_chars_per_file": (total_chars/total_files) if total_files else 0
    }

@app.get("/doc_structure")
def doc_structure(filename: str):
    """Return headings (heading_path values) and table summaries for a file from vector store payloads."""
    if embedder is None or qdrant_client is None:
        return {"error": "Vector store unavailable"}
    try:
        # Query all points for filename (limit large to safeguard)
        # Use scroll API via filter
        filter_condition = qdrant_models.Filter(must=[qdrant_models.FieldCondition(key="filename", match=qdrant_models.MatchValue(value=filename))])
        headings: Set[str] = set()
        tables: List[Dict[str, Any]] = []
        offset = None
        batch_limit = 256
        while True:
            scroll_res = qdrant_client.scroll(collection_name=QDRANT_COLLECTION, scroll_filter=filter_condition, limit=batch_limit, with_payload=True, offset=offset)
            points, offset = scroll_res
            if not points:
                break
            for pt in points:
                payload = pt.payload or {}
                hp = payload.get('heading_path')
                if hp:
                    headings.add(hp)
                if 'table_markdown' in payload:
                    tables.append({
                        'page': payload.get('page'),
                        'chunk_index': payload.get('chunk_index'),
                        'preview': (payload.get('table_markdown') or '')[:400]
                    })
            if offset is None:
                break
        return {
            'filename': filename,
            'headings': sorted(headings),
            'tables': tables,
            'table_count': len(tables)
        }
    except Exception as e:
        return {"error": str(e)}


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
                bbox = blk.get('bbox') if isinstance(blk, dict) else None
                y_top = bbox[1] if bbox else None
                x_left = bbox[0] if bbox else None  # new for refined ordering
                page_blocks.append({
                    'page': page_index + 1,
                    'text': text_join,
                    'font_size': avg_size,
                    'type': 'paragraph',
                    'heading_level': None,
                    'y_top': y_top,
                    'x_left': x_left
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
                        # Use find_tables for bbox ordering
                        found_tables = page_obj.find_tables() or []
                        page_table_objs = []
                        for tbl in found_tables:
                            try:
                                rows = [[(cell or '').strip() for cell in row] for row in tbl.extract()]  # type: ignore[attr-defined]
                            except Exception:
                                continue
                            if not rows or not any(any(cell for cell in r) for r in rows):
                                continue
                            n_cols = max(len(r) for r in rows)
                            for r in rows:
                                while len(r) < n_cols:
                                    r.append('')
                            # Summarize/truncate
                            disp_rows = rows[:TABLE_SUMMARY_MAX_ROWS]
                            truncated = len(rows) > TABLE_SUMMARY_MAX_ROWS
                            if truncated:
                                disp_rows.append(['…'] * n_cols)
                            if n_cols > TABLE_SUMMARY_MAX_COLS:
                                for r in disp_rows:
                                    del r[TABLE_SUMMARY_MAX_COLS:]
                                if disp_rows and disp_rows[0]:
                                    disp_rows[0][-1] = (disp_rows[0][-1] or '') + '…'
                            header = disp_rows[0]
                            md_lines = ['|' + '|'.join(c or ' ' for c in header) + '|']
                            md_lines.append('|' + '|'.join(['---']*len(header)) + '|')
                            for data_row in disp_rows[1:]:
                                md_lines.append('|' + '|'.join(c or ' ' for c in data_row) + '|')
                            table_markdown = '\n'.join(md_lines)
                            table_md5 = hashlib.md5(table_markdown.encode('utf-8')).hexdigest()
                            # Basic numeric column stats for summary
                            col_stats = []
                            for col_idx in range(n_cols):
                                col_vals_raw = [r[col_idx] for r in rows[1:80] if len(r) > col_idx]
                                numeric_vals = []
                                for v in col_vals_raw:
                                    try:
                                        numeric_vals.append(float(v.replace('%','').replace(',','')))
                                    except Exception:
                                        continue
                                if numeric_vals:
                                    col_stats.append(f"c{col_idx+1}:n={len(numeric_vals)} min={min(numeric_vals):.2f} max={max(numeric_vals):.2f}")
                            table_summary_stats = '; '.join(col_stats[:4]) if col_stats else ''
                            table_summary = f"rows={len(rows)} cols={n_cols} {'truncated ' if truncated else ''}{table_summary_stats}".strip()
                            y_top = tbl.bbox[1] if getattr(tbl, 'bbox', None) else 0  # type: ignore
                            x_left = tbl.bbox[0] if getattr(tbl, 'bbox', None) else 0  # type: ignore
                            # Build CSV (full rows) and cap later when storing in payload
                            csv_lines = []
                            for r in rows:
                                # simple CSV quoting
                                csv_cells = []
                                for cell in r:
                                    cell = cell or ''
                                    if any(ch in cell for ch in [',','"','\n']):
                                        cell = '"' + cell.replace('"','""') + '"'
                                    csv_cells.append(cell)
                                csv_lines.append(','.join(csv_cells))
                            table_csv_full = '\n'.join(csv_lines)
                            page_table_objs.append((y_top, x_left, {
                                'page': p_idx + 1,
                                'text': table_markdown,
                                'font_size': 0,
                                'type': 'table',
                                'heading_level': None,
                                'table_markdown': table_markdown,
                                'n_rows': len(rows),
                                'n_cols': n_cols,
                                'table_md5': table_md5,
                                'table_summary': table_summary,
                                'table_csv': table_csv_full[:TABLE_CSV_MAX_CHARS],
                                'y_top': y_top,
                                'x_left': x_left
                            }))
                        if page_table_objs:
                            text_blocks = raw_pages_blocks[p_idx]
                            if any(b.get('y_top') is not None for b in text_blocks):
                                combined = [
                                    (
                                        (b.get('y_top') if b.get('y_top') is not None else 9e9),
                                        (b.get('x_left') if b.get('x_left') is not None else 9e9),
                                        b
                                    ) for b in text_blocks
                                ]
                                combined.extend(page_table_objs)  # already tuples (y_top,x_left,blk)
                                combined.sort(key=lambda x: (x[0], x[1]))
                                raw_pages_blocks[p_idx] = [b for _,__, b in combined]
                            else:
                                # Fallback: append at end preserving existing order
                                for _,__, tbl_blk in page_table_objs:
                                    raw_pages_blocks[p_idx].append(tbl_blk)
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
        # Table metadata propagation (single table block case)
        tbl_blocks = [b for b in chunk_blocks_accum if b['type'] == 'table']
        if len(tbl_blocks) == 1 and len(chunk_blocks_accum) == 1:
            tbl = tbl_blocks[0]
       

        else:
            tbl = None
        pages = [b['page'] for b in chunk_blocks_accum]
        block_types = [b['type'] for b in chunk_blocks_accum]
        pages_sorted = sorted(set(pages))
        chunk_entry = {
            'page': min(pages_sorted),
            'pages': pages_sorted,
            'chunk_index': chunk_index_counter,
            'text': text,
            'md5': md5,
            'heading_path': ' > '.join(heading_path),
            'block_types': block_types,
            'page_range': f"{min(pages_sorted)}-{max(pages_sorted)}" if len(pages_sorted)>1 else str(pages_sorted[0])
        }
        if tbl is not None:
            # Carry table metadata
            for k in ['table_md5','table_summary','table_csv','n_rows','n_cols']:
                if k in tbl:
                    chunk_entry[k] = tbl[k]
        chunks.append(chunk_entry)
        chunk_blocks_accum = []
        chunk_tokens = 0
        chunk_index_counter += 1
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

# ---------------- Diversity-aware Context Assembly ---------------- #

def _estimate_tokens(text: str) -> int:
    return max(1, len(text.split()))

def assemble_context(question: str, hits: List[dict], max_tokens: int = MAX_CONTEXT_TOKENS) -> List[dict]:
    if not ENABLE_DIVERSITY:
        # simple return
        return hits
    # Identify if question wants tables
    q_lower = question.lower()
    wants_tables = any(k in q_lower for k in TABLE_KEYWORDS)
    by_heading: Dict[str, List[dict]] = {}
    for h in hits:
        hp = h.get('heading_path') or ''
        by_heading.setdefault(hp, []).append(h)
    ordered_headings = sorted(by_heading.keys())
    assembled: List[dict] = []
    used_tokens = 0
    # Round-robin pick per heading
    while ordered_headings and used_tokens < max_tokens:
        new_ordered = []
        for hp in ordered_headings:
            bucket = by_heading[hp]
            if not bucket:
                continue
            cand = bucket.pop(0)
            # Skip table-heavy chunks unless requested
            block_types = cand.get('block_types') or []
            if not wants_tables and block_types == ['table'] and len(assembled) >= 2:
                # push to end for later
                bucket.append(cand)
            else:
                tks = _estimate_tokens(cand.get('content',''))
                if used_tokens + tks <= max_tokens:
                    assembled.append(cand)
                    used_tokens += tks
            if bucket:
                new_ordered.append(hp)
        if new_ordered == ordered_headings:  # no progress
            break
        ordered_headings = new_ordered
    return assembled
# ---------------- End Diversity-aware Context Assembly ---------------- #

# ---------------- New: Answer generation with citation markers ---------------- #

def generate_answer_with_citations(question: str, context_docs: List[dict]) -> tuple[str, List[dict]]:
    if not context_docs:
        return ("I couldn't find relevant information to answer your question.", [])
    
    # Assign citation ids if not already
    for idx, d in enumerate(context_docs, start=1):
        d['citation_id'] = idx
    
    # Optimize context by limiting the number of documents and snippet length
    # This reduces the prompt size and processing time
    max_docs = min(len(context_docs), 6)  # Reduced from 8 to 6 most relevant documents
    snippet_length = 250  # Initial snippet length, reduced from 300
    
    # Adaptive timeout and retry mechanism
    max_retries = 3
    initial_timeout = 90  # Initial timeout in seconds
    min_timeout = 30  # Minimum timeout for subsequent retries
    timeout_reduction_factor = 0.7  # Reduce timeout by 30% on each retry
    
    # Track timeouts to reduce context size on multiple timeouts
    timeout_count = 0
    
    for attempt in range(max_retries + 1):
        # Adjust context size if we've had multiple timeouts
        current_max_docs = max(3, max_docs - timeout_count)  # Reduce docs but keep at least 3
        current_snippet_length = max(150, snippet_length - (timeout_count * 50))  # Reduce snippet length but keep at least 150 chars
        
        # Build context with current parameters
        context_lines = []
        for d in context_docs[:current_max_docs]:
            snippet = (d.get('content','') or '')[:current_snippet_length].strip()
            context_lines.append(f"[C{d['citation_id']}] Source: {d.get('filename')} page {d.get('page')} chunk {d.get('chunk_index')}\n{snippet}")
        
        context_text = '\n\n'.join(context_lines)
        
        # Streamlined instructions to reduce prompt size
        instructions = (
            "You are an academic assistant. Use ONLY the provided context. "
            "Cite facts with [C1], [C2], etc. Include References section at the end. Be concise."
        )
        
        prompt = f"{instructions}\n\nQuestion: {question}\n\nContext:\n{context_text}\n\nAnswer:"
        
        # Calculate current timeout (reduce on each retry)
        current_timeout = max(min_timeout, initial_timeout * (timeout_reduction_factor ** attempt))
        
        print(f"[INFO] Attempt {attempt+1}/{max_retries+1}: Using {current_max_docs} docs, {current_snippet_length} chars per snippet, {current_timeout:.1f}s timeout")
        
        try:
            print(f"[INFO] Sending request to Ollama (attempt {attempt+1}/{max_retries+1})")
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3:latest", 
                    "prompt": prompt, 
                    "stream": False,
                    "temperature": 0.7
                },
                timeout=current_timeout
            )
            response.raise_for_status()
            data = response.json()
            answer = data.get('response', '')
            
            # Check for empty response
            if not answer.strip():
                print(f"[WARNING] Empty response received on attempt {attempt+1}")
                if attempt < max_retries:
                    timeout_count += 1  # Treat empty response like a timeout
                    wait_time = (attempt + 1) * 2
                    print(f"[INFO] Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                    continue
                else:
                    return "Error: Received empty response from Ollama. Please try again with a simpler question.", []
            
            print(f"[INFO] Ollama response received successfully on attempt {attempt+1}")
            break  # Success, exit retry loop
            
        except requests.exceptions.Timeout as e:
            print(f"[WARNING] Timeout on attempt {attempt+1}/{max_retries+1}: {e}")
            timeout_count += 1
            
            if attempt < max_retries:
                # Wait before retrying, with increasing backoff
                wait_time = (attempt + 1) * 2
                print(f"[INFO] Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                answer = f"Error: Request to Ollama timed out after {max_retries+1} attempts. Try asking a simpler question or check if Ollama is running properly."
                
        except requests.exceptions.ConnectionError as e:
            print(f"[ERROR] Connection error on attempt {attempt+1}: {e}")
            if attempt < max_retries:
                wait_time = (attempt + 1) * 3  # Longer wait for connection issues
                print(f"[INFO] Connection issue - waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                answer = f"Error: Cannot connect to Ollama. Please check if Ollama is running and restart if necessary."
                
        except Exception as e:
            print(f"[ERROR] Failed on attempt {attempt+1}/{max_retries+1}: {e}")
            if attempt < max_retries:
                wait_time = (attempt + 1) * 2
                print(f"[INFO] Waiting {wait_time} seconds before retry due to error: {e}")
                time.sleep(wait_time)
                timeout_count += 1  # Reduce context on next attempt
            else:
                answer = f"Error communicating with Ollama: {str(e)}. Please try again or check if Ollama is running properly."
                break
    
    # Extract error message if present for better user feedback
    error_msg = None
    if answer.lower().startswith("error:"):
        parts = answer.split(".", 1)
        if len(parts) > 1:
            error_msg = parts[0].strip() + "."
            answer = parts[1].strip()
    
    citations = [
        {
            'id': d['citation_id'],
            'filename': d.get('filename'),
            'page': d.get('page'),
            'chunk_index': d.get('chunk_index')
        } for d in context_docs[:current_max_docs]  # Only include citations for docs actually used
    ]
    
    return answer, citations
# ---------------- End: Answer generation with citation markers ---------------- #

@app.get('/tables')
def list_tables(filename: Optional[str] = None, limit: int = 200):
    """List table chunks (single-table chunks with table_markdown). Optional filter by filename."""
    if embedder is None or qdrant_client is None:
        return {"error": "Vector store unavailable"}
    try:
        must = []
        if filename:
            must.append(qdrant_models.FieldCondition(key="filename", match=qdrant_models.MatchValue(value=filename)))
        filt = qdrant_models.Filter(must=must) if must else None
        tables = []
        offset = None
        batch_limit = 256
        while len(tables) < limit:
            points, offset = qdrant_client.scroll(collection_name=QDRANT_COLLECTION, scroll_filter=filt, limit=batch_limit, with_payload=True, offset=offset)
            if not points:
                break
            for pt in points:
                payload = pt.payload or {}
                if 'table_markdown' in payload:
                    tables.append({
                        'id': str(pt.id),
                        'filename': payload.get('filename'),
                        'page': payload.get('page'),
                        'chunk_index': payload.get('chunk_index'),
                        'table_md5': payload.get('table_md5'),
                        'n_rows': payload.get('n_rows'),
                        'n_cols': payload.get('n_cols'),
                        'table_summary': payload.get('table_summary'),
                        'preview': (payload.get('table_markdown') or '')[:250]
                    })
                    if len(tables) >= limit:
                        break
            if offset is None or len(tables) >= limit:
                break
        return {'count': len(tables), 'tables': tables, 'filename': filename}
    except Exception as e:
        return {'error': str(e)}

@app.get('/table_csv')
def get_table_csv(table_md5: Optional[str] = None, filename: Optional[str] = None, page: Optional[int] = None, chunk_index: Optional[int] = None):
    """Return CSV + markdown for a specific table. Identify via table_md5 OR (filename,page,chunk_index)."""
    if embedder is None or qdrant_client is None:
        return {"error": "Vector store unavailable"}
    if not table_md5 and not (filename and page is not None and chunk_index is not None):
        return {"error": "Provide table_md5 or (filename,page,chunk_index)"}
    try:
        # Strategy: scroll (filtered by filename if provided) and match conditions client-side.
        must = []
        if filename:
            must.append(qdrant_models.FieldCondition(key="filename", match=qdrant_models.MatchValue(value=filename)))
        filt = qdrant_models.Filter(must=must) if must else None
        offset = None
        batch_limit = 256
        while True:
            points, offset = qdrant_client.scroll(collection_name=QDRANT_COLLECTION, scroll_filter=filt, limit=batch_limit, with_payload=True, offset=offset)
            if not points:
                break
            for pt in points:
                payload = pt.payload or {}
                if 'table_markdown' not in payload:
                    continue
                if table_md5 and payload.get('table_md5') != table_md5:
                    continue
                if not table_md5 and filename and (payload.get('page') != page or payload.get('chunk_index') != chunk_index):
                    continue
                if not table_md5 and not filename:
                    continue
                return {
                    'id': str(pt.id),
                    'filename': payload.get('filename'),
                    'page': payload.get('page'),
                    'chunk_index': payload.get('chunk_index'),
                    'table_md5': payload.get('table_md5'),
                    'n_rows': payload.get('n_rows'),
                    'n_cols': payload.get('n_cols'),
                    'table_summary': payload.get('table_summary'),
                    'table_markdown': payload.get('table_markdown'),
                    'table_csv': payload.get('table_csv')
                }
            if offset is None:
                break
        return {'error': 'Table not found'}
    except Exception as e:
        return {'error': str(e)}


if __name__ == '__main__':
    import uvicorn
    import socket
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', '8000'))
    reload_flag = os.environ.get('RELOAD','1')=='1'
    # Auto-disable reload if using embedded Qdrant path to avoid lock contention
    if reload_flag and FORCE_DISABLE_RELOAD_FOR_EMBEDDED and not os.environ.get('QDRANT_URL'):
        print('[INFO] Auto-disabling uvicorn reload because embedded Qdrant does not support multi-process access. Set FORCE_DISABLE_RELOAD_FOR_EMBEDDED=0 to override.')
        reload_flag = False
    # Port availability check (Windows)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((host, port))
        sock.close()
    except OSError as e:
        print(f'[ERROR] Port {port} is already in use. Please stop any other server using this port and try again.')
        import sys
        sys.exit(1)
    uvicorn.run('main:app', host=host, port=port, reload=reload_flag)