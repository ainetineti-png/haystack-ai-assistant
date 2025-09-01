# RAG Ingestion & Retrieval Improvement Roadmap

Purpose: High-fidelity academic PDF / DOCX ingestion for AI tutor with incremental, structured, reliable retrieval.

## 0. Current State (Baseline)
- Plain per-page extraction (PyPDF2 -> pdfplumber fallback)
- Naive chunking (char window + overlap, sentence boundary heuristic)
- Whole-document fallback list for simple keyword search
- Vector store (Qdrant) with deterministic chunk UUIDs
- Incremental ingest (file-level MD5) + skip unchanged; delete stale
- Phase 1 partial implementation: PyMuPDF-based block extraction, heading detection, header/footer removal, structure-aware chunking, schema_version and page_md5s in manifest

## 1. Target Architecture Summary
Layers:
1. File Dispatcher + Manifest
2. Multi-stage Structured Extraction (layout + tables + OCR fallback)
3. Block Normalization & Classification
4. Structure-aware Chunking
5. Embedding & Hybrid Indexing (dense + sparse)
6. Reranking + Context Assembly
7. Answer Generation & Post-processing
8. Monitoring & Quality Metrics

## 2. Manifest & Incremental Strategy
Add fields:
- schema_version (int) – bump when extraction logic changes
- file_md5 (binary hash)
- page_level: [{page, page_md5, num_blocks, ocr(bool)}]
- counts: tables, equations, figures, ocr_pages, total_chars
- block_index_hash (MD5 of normalized blocks text)
Selective reindex logic:
- If file_md5 unchanged AND schema_version same -> skip
- If only subset of page_md5 changed -> re-extract embeddings for those pages (delete by filename+page)
- If schema_version bumped -> reprocess all
(Current: implemented file_md5 + schema_version + page_md5s at file level; selective page reindex still pending.)

## 3. Multi-stage Extraction Pipeline
Order (short-circuit if coverage >= 90% non-empty pages):
A. Structured Parser (Option 1: Docling; Option 2: Unstructured partition)
B. PyMuPDF (fitz) fallback (extract blocks + headings via font size) [Implemented basic version]
C. pdfplumber specific table recovery
D. OCR (PaddleOCR preferred; Tesseract fallback) for low-text pages
E. (Optional) Math extraction (Nougat / Mathpix API) -> LaTeX text blocks

Each block: {page, type, text, bbox, source_stage, font_size?, heading_level?}
Block types: heading, paragraph, list_item, table, figure_caption, equation, footer/header_candidate

## 4. Header/Footer & Boilerplate Removal
Detection:
- Collect top N lines (first ~120 chars) of each page’s first/last 3 text blocks
- Frequency > 60% pages → classify as header/footer; remove from blocks
Store removed_patterns in manifest for audit (Pending: currently removed but not stored).

## 5. Table Handling
Preferred extraction sources:
- Docling/Unstructured structured tables
- pdfplumber (extract table -> 2D list)
Normalize -> Markdown grid table and CSV string
Payload for table blocks: {table_markdown, table_csv, n_rows, n_cols, table_md5}
(Status: Basic implemented – pdfplumber tables extracted via find_tables(), ordered by bbox top (approximate), normalized to markdown with row/col truncation + added n_rows, n_cols, table_md5, table_summary. CSV export still pending; advanced ordering using text block bboxes pending.)

## 6. Equations & Figures (Optional Phase 4/5)
- Equation blocks: LaTeX text or placeholder [EQUATION]
- Figure captions retained; OCR figure regions if small (<500k pixels)
- Tag blocks for potential specialized embedding model later

## 7. Block Normalization
Steps:
1. Unicode NFKC
2. Collapse multiple spaces
3. Trim
4. Language detect (fasttext-lite) – mark non-target segments
5. Tag prefix: [H1], [H2], [TABLE], [EQ], [CAPTION], [LIST]
6. Generate block_md5 (post-normalization)
(Status: Not yet implemented; current extraction stores raw text with heading inference.)

## 8. Structure-aware Chunking Algorithm
Parameters:
- target_tokens ≈ 180–220 (for MiniLM) or dynamic based on downstream LLM context
- max_tokens_per_chunk ≈ 256–300
Rules:
- Do not split tables/equations
- Start new chunk at heading if current chunk not empty
- Merge small adjacent paragraphs (< 40 chars) before token counting
- Overlap (15% tokens) only across narrative blocks; no overlap across table boundary
Data kept:
- chunk.blocks = [block_ids]
- chunk.heading_path (constructed from last seen headings hierarchy)
(Current: Basic version implemented—headings trigger chunk flush, heading_path tracked, no overlap logic yet, no block IDs stored.)

## 9. Embedding & Hybrid Index
Dense:
- Current: all-MiniLM-L6-v2 (384-d). Future: allow config (e5, instructor, jina)
Sparse (Phase 3):
- BM25 / elasticlike OR SPLADE (if resource available) OR simple TF-IDF vector using scikit / rapidfuzz
Store in Qdrant:
- Primary collection: dense vectors
- Optionally secondary collection or payload field for sparse tokens; or external Whoosh/Elasticsearch
Payload add:
```
{
  filename,
  page_range,
  heading_path,
  block_types,          # list
  ocr: bool,
  num_tables,
  schema_version,
  source_stages,
  chunk_md5,
  blocks_md5_concat,
  coverage_ratio
}
```
(Current: Added heading_path, block_types, schema_version, page_range, table_markdown (single-table chunks), table_summary (heuristic). TF-IDF sparse index built in-memory; pending: persistence & BM25 variant.)

## 10. Reranking & Context Assembly
Retrieval Steps:
1. Dense top K (e.g. 50)
2. Sparse top K (50)
3. Merge + Reciprocal Rank Fusion (RRF)
4. Cross-encoder rerank (e.g. cross-encoder/ms-marco-MiniLM-L-6-v2) -> top N (e.g. 8)
5. Assemble context respecting token budget (e.g. 2000 tokens)
   - Prioritize diversity (distinct heading_path)
   - Collapse sequential chunks from same section
   - Convert tables to concise markdown with row/col limits + ellipsis
(Status: Partial – Dense + TF-IDF sparse + RRF fusion implemented; optional cross-encoder rerank (env ENABLE_RERANK) integrated. Diversity-aware context assembly & advanced table condensation still pending.)

## 11. Answer Prompt Enrichment
Prompt sections:
- System instructions (tutor persona, cite filenames & pages)
- Condensed context (include heading, page, snippet)
- Safety fallback if context empty
Add citations marker format: [filename pX]
(Status: Basic prompt only.)

## 12. Monitoring & Metrics
Store (per file) in manifest or separate JSONL:
- extraction_duration_ms
- ocr_pages, ocr_time_ms
- char_coverage (extracted_chars / estimated_text_bytes)
- tables_detected, tables_extracted
- avg_tokens_per_chunk
- embedding_time_ms_per_chunk
Expose /ingest_metrics endpoint.
(Status: Not implemented.)

## 13. Phase Plan & Tasks
### Phase 1 (Structure Basics)
- [x] Add SCHEMA_VERSION constant (now 3 for per-page + patterns + normalization)
- [x] Integrate PyMuPDF extraction (basic blocks)
- [x] Header/footer detection & removal (without manifest logging yet)
- [x] Block classification (headings via font size heuristic)
- [x] Structure-aware chunking (basic; no overlap or block IDs)
- [x] Extend manifest schema (store per-page num_blocks, header/footer patterns, block_index_hash)
- [x] Update ingestion to page-level hashes with selective page reindex (partial page deletion + re-embed)
- [x] Payload includes block_types, heading_path (partial: implemented)
- [ ] Tests with sample multi-column academic PDF

### Phase 2 (Tables + Manifest Upgrade)
- [x] Table extraction (basic pdfplumber integration; ordering simplistic)
- [x] Add table_markdown to payload (for pure table chunks)
- [x] Manifest: schema_version, page_md5 list (completed for page objects) add per-page num_blocks
- [x] Selective page reindex (implemented Phase 1)
- [x] Manifest counts.tables populated (equations/figures pending)
- [x] Throttled PyPDF2 warnings + per-page pdfplumber fallback
- [x] FAST_SKIP_PDF_REPARSE flag to avoid re-parsing unchanged PDFs
- [x] Table ordering by bbox (basic via pdfplumber find_tables; refine with text block bboxes pending)
- [x] Table summarization (row/col truncation + basic numeric column stats; advanced semantic summary pending)

### Phase 3 (Hybrid Retrieval + Rerank)
- [x] Add sparse index (TF-IDF) in-memory (persistence & BM25 variant pending)
- [x] Implement RRF fusion
- [x] Integrate cross-encoder reranking (optional env flag)
- [x] Add diversity-aware context assembly prototype (basic round-robin implemented; advanced citation & dedupe pending)
- [x] Persist sparse artifacts (vectorizer, matrix, refs) across restarts (ENV ENABLE_SPARSE_PERSIST)
- [x] BM25 / alternative sparse scorer (custom BM25 implementation with persistence)
- [x] Rerank caching (LRU for (query_md5, chunk_md5))
- [x] Citation IDs and inline markers in /ask responses
- [x] /tables and /table_csv endpoints for table chunk export and inspection
- [x] Qdrant client initialization robustness (retry logic, null checks for uvicorn reload)

### Phase 4 (OCR & Low Coverage Recovery)
- [ ] PaddleOCR for low-text pages
- [ ] OCR confidence threshold & logging
- [ ] Re-extract changed pages with OCR blocks inserted

### Phase 5 (Equations & Figures)
- [ ] Integrate Nougat or Mathpix API (configurable)
- [ ] Equation block tagging and optional specialized embedding

### Phase 6 (Advanced Context Assembly + Citations)
- [ ] Diversity-aware context selection
- [ ] Table summarization (truncate large tables) (basic stats done; advanced semantic summarization pending)
- [ ] Inline citation markers, structured citations list

### Phase 7 (Quality & Ops)
- [x] /ingest_metrics endpoint
- [ ] Export/import script include manifest (currently only vectors)
- [x] /doc_structure endpoint (headings + tables summary)
- [ ] Logging enrichment (structured JSON option)
- [ ] API key / rate limiting
- [ ] Sparse index persistence

## 14. Data Structures (Draft)
```
SCHEMA_VERSION = 2

Manifest:
{
  "files": {
    "rel/path.pdf": {
      "schema_version": 2,
      "file_md5": "...",
      "pages": [ {"page":1,"page_md5":"...","num_blocks":42,"ocr":false}, ...],
      "counts": {"tables":3,"equations":5,"ocr_pages":1,"chars":54321},
      "block_index_hash": "...",
      "last_indexed": "2025-09-01T10:00:00Z"
    }
  }
}
```
(Current: storing file_md5, schema_version, page_md5s array; not storing per-page objects or counts.)

## 15. New/Updated Endpoints
- POST /reindex_file (exists) – extend to accept force=true to ignore checksum (Pending)
- GET /ingest_metrics – summary JSON (Implemented)
- GET /doc_structure?filename= – returns headings & table summary (Implemented)
- GET /search – expose dense/sparse/fused scores for debugging (Implemented)
- (Planned) GET /table_csv – return CSV for a table chunk
- (Planned) GET /tables?filename= – list all tables with stats

## 16. Configuration Flags (ENV)
- ENABLE_PYMUPDF=1
- ENABLE_OCR=0/1
- ENABLE_HYBRID=1
- ENABLE_RERANK=1
- SCHEMA_VERSION=2 (override build number)
- OCR_ENGINE=paddle|tesseract

## 17. Dependencies to Add (Progressively)
Phase 1: pymupdf (Added)
Phase 2: (already) pdfplumber (tables) + tabulate
Phase 3: scikit-learn (TF-IDF) or whoosh, rapidfuzz
Phase 4: paddleocr (includes paddlepaddle) OR pytesseract + system tesseract
Phase 5: nougat-ocr (pip) OR external Mathpix API client
Phase 3/4 optional: cross-encoder (sentence-transformers already supports) model download at startup

## 18. Risk Mitigation
- Large dependencies (paddlepaddle) gated by ENABLE_OCR
- Manifest corruption: keep rotate backups ingest_manifest.json.bakN
- Memory: Cap max chunks per file; large tables summarized not fully embedded twice

## 19. Minimal Phase 1 Implementation Checklist (Updated)
[x] Add SCHEMA_VERSION constant
[x] Basic PyMuPDF block extraction
[x] Heading detection heuristic
[x] Header/footer removal
[x] Structure-aware chunking (basic)
[x] Manifest extended with per-page block counts (+ header/footer patterns, block_index_hash)
[x] Selective page reindex logic (page-diff + partial vector deletion)
[x] Store removed header/footer patterns (header_patterns/footer_patterns in manifest)
[x] Block normalization pipeline (basic: NFKC + whitespace collapse + md5 hashes)
[ ] Tests with complex PDFs

## 20. Future Enhancements
- Section-aware answer grounding (return section_id with answer spans)
- Feedback loop: log user question + retrieved chunks + rating to improve reranking
- Knowledge graph extraction (entities / relations) augment retrieval for concept queries
- Table CSV export & advanced semantic summarization
- BM25 scorer + hybrid fusion ablation metrics
- Inline citation markers with numbered references in answers
- Rerank score caching & configurable top-N rerank
- Structured logging (JSON) & /metrics Prometheus endpoint
- API key auth + rate limiting
- Export/import full corpus (vectors + manifest + sparse index + checksums)

---
Additional Notes (Phase 1 + Phase 2/3 Progress):
- SCHEMA_VERSION bumped to 3.
- Manifest stores per-page metadata, header/footer patterns, block_index_hash, counts.
- Table pipeline: bbox-ordered extraction, markdown normalization, truncation, numeric column stats -> table_summary.
- Hybrid retrieval: Dense + TF-IDF sparse + RRF; optional cross-encoder rerank (ENABLE_RERANK=1). Sparse persistence implemented.
- Diversity-aware context assembly implemented (round-robin by heading, table suppression heuristic) – advanced citation/dedupe pending.
- BM25 scorer implemented with persistence and /search endpoint parameter switching.
- Citation IDs and inline markers implemented in /ask responses with reference listing.
- Rerank caching (LRU) and configurable top-N rerank implemented.
- /tables and /table_csv endpoints implemented for table chunk export and inspection.
- Qdrant client initialization improved with retry logic and comprehensive null checks to handle uvicorn reload scenarios.
- Next priority: (a) table ordering refinement with x/y coordinates, (b) logging enrichment & /metrics, (c) API key auth, (d) export/import tooling, (e) OCR fallback for low-text pages.
