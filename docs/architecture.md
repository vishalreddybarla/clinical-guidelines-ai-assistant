# System Architecture

## Overview

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────┐
│              Streamlit Frontend (port 8501)      │
│  - Chat interface   - Source citations panel     │
│  - Guideline filter - Feedback buttons           │
└───────────────────────┬─────────────────────────┘
                        │ HTTP POST /api/ask
                        ▼
┌─────────────────────────────────────────────────┐
│              FastAPI Backend (port 8000)         │
│  - Request validation (Pydantic)                 │
│  - Monitoring (SQLite logging)                   │
│  - Endpoints: /ask, /feedback, /health           │
└───────────────────────┬─────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────┐
│                  Agent Layer                     │
│  1. Intent Detection                             │
│     - Needs drug check?  ─── RxNorm/NIH API      │
│     - Needs PubMed?      ─── NCBI E-utilities    │
│  2. Hybrid Retrieval                             │
│     - Semantic (ChromaDB cosine similarity)      │
│     - Keyword (BM25 rank_bm25)                   │
│     - Fusion + Cohere Rerank                     │
│  3. Answer Generation (OpenAI / Anthropic API)   │
└─────────────────────────────────────────────────┘
                        │
            ┌───────────┴──────────┐
            ▼                      ▼
┌─────────────────┐     ┌────────────────────────┐
│    ChromaDB     │     │   In-memory BM25        │
│  (persistent)   │     │  (loaded from JSON)     │
│  text-embedding │     │  rank_bm25              │
│  -3-small vecs  │     └────────────────────────┘
└─────────────────┘
```

## Data Flow

1. **Ingestion** (offline, run once): PDFs → `document_loader` → `chunking` → `embeddings` → ChromaDB
2. **Retrieval** (per query): Query → embed → cosine search + BM25 → score fusion → Cohere rerank
3. **Generation** (per query): Top-k chunks → prompt template → LLM → structured response

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Vector DB | ChromaDB | Local, zero ops cost, cosine space |
| Embeddings | `text-embedding-3-small` | Strong quality/cost ratio |
| Retrieval | Hybrid (0.7 semantic + 0.3 BM25) | Better recall on medical terms |
| Reranker | Cohere `rerank-english-v3.0` | Free tier, significant precision boost |
| LLM Default | `gpt-4o-mini` | Sufficient quality, very low cost |
| Temperature | 0.1 | Factual clinical answers require low variation |
| Chunk size | 512 chars, 100 overlap | Good balance for multi-sentence medical statements |
