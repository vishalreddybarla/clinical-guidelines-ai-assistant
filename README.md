# 🏥 Clinical Guidelines AI Assistant

A **RAG-powered AI assistant** that answers clinical questions by searching through WHO, NIH, AHA, CDC, and USPSTF clinical guidelines — returning accurate, sourced answers with exact citations.

---

## Problem Statement

Doctors spend 15–20 minutes searching through lengthy PDF guidelines to find specific recommendations. This system answers clinical questions instantly with page-level source references, allowing clinicians to focus on patient care rather than document hunting.

**Covered guidelines:**
- 🩺 WHO Hypertension Treatment Guidelines (2021)
- 💉 ADA Standards of Care in Diabetes 2024 (Pharmacologic Treatment)
- ❤️ AHA/ACC Heart Failure Guidelines 2022
- 🦠 CDC STI Treatment Guidelines 2021
- 🔬 USPSTF Cancer Screening Recommendations

---

## Architecture

```
User Query
    │
    ▼
Streamlit Frontend (port 8501)
    │ HTTP POST /api/ask
    ▼
FastAPI Backend (port 8000)
    │
    ▼
Agent Layer
  ├── Intent Detection
  │     ├── Drug interaction? → RxNorm/NIH API
  │     └── Recent research? → NCBI PubMed API
  ├── Hybrid Retrieval
  │     ├── Semantic search (ChromaDB + OpenAI embeddings)
  │     ├── Keyword search (BM25 via rank_bm25)
  │     ├── Score fusion (0.7 semantic + 0.3 keyword)
  │     └── Cohere Reranking
  └── Answer Generation (GPT-4o-mini / GPT-4o / Claude Haiku)
```

See [`docs/architecture.md`](docs/architecture.md) for a detailed diagram.

---

## Tech Stack

| Component | Tool |
|-----------|------|
| LLM APIs | OpenAI (GPT-4o-mini, GPT-4o) + Anthropic (Claude Haiku) |
| Embeddings | `text-embedding-3-small` |
| Vector DB | ChromaDB (persistent, local) |
| Retrieval | Semantic + BM25 hybrid + Cohere Rerank |
| Backend API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| PDF parsing | pdfplumber |
| Drug interactions | RxNorm / NLM (free) |
| PubMed search | NCBI E-utilities (free) |
| Monitoring | SQLite + Streamlit dashboard |
| Data validation | Pydantic v2 |

---

## Setup

### 1. Prerequisites

- Python 3.11+
- Git
- API keys: OpenAI (required), Anthropic (optional), Cohere (optional)

### 2. Clone and create virtual environment

```bash
git clone https://github.com/vishalreddybarla/clinical-guidelines-ai-assistant.git
cd clinical-guidelines-ai-assistant

python -m venv venv
source venv/bin/activate        # Linux/Mac
# OR
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
# Edit .env and add your API keys
```

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | **Yes** | Used for embeddings and GPT models |
| `ANTHROPIC_API_KEY` | No | Required only to use Claude models |
| `COHERE_API_KEY` | No | Enables Cohere reranking (free tier) |
| `NCBI_API_KEY` | No | Increases PubMed rate limits |

### 5. Add PDF guidelines

Place PDFs in `data/raw/` with these exact names:

```
data/raw/
├── who_hypertension_2021.pdf
├── ada_diabetes_2024.pdf
├── aha_heart_failure_2022.pdf
├── cdc_sti_2021.pdf
└── uspstf_cancer_screening/
    ├── breast_cancer.pdf
    ├── colorectal_cancer.pdf
    ├── lung_cancer.pdf
    └── cervical_cancer.pdf
```

### 6. Build the index

```bash
python index_pipeline.py
```

This extracts PDF text, creates chunks, generates embeddings, and stores them in ChromaDB (~5–10 minutes depending on the number of chunks and API speed).

---

## Running the Application

### Start the FastAPI backend

```bash
uvicorn api.main:app --reload
# → http://localhost:8000
# → Docs: http://localhost:8000/docs
```

### Start the Streamlit chat interface

```bash
streamlit run frontend/app.py
# → http://localhost:8501
```

### Start the monitoring dashboard

```bash
streamlit run monitoring/dashboard.py --server.port 8502
# → http://localhost:8502
```

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/ask` | POST | Ask a clinical question |
| `/api/feedback` | POST | Submit a rating (1–5) for an answer |
| `/api/health` | GET | Check API health and data stats |

### Example request

```bash
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the first-line treatment for Stage 1 hypertension?",
    "model": "gpt-4o-mini"
  }'
```

---

## Running Evaluation

After indexing, evaluate the full 50-question test set:

```bash
python -m src.evaluation
```

Results are saved to `data/processed/evaluation_report.json`. See [`docs/model_comparison.md`](docs/model_comparison.md) for the comparison table.

---

## Project Structure

```
.
├── data/
│   ├── raw/                  # PDFs (not committed)
│   ├── metadata/             # Guideline metadata JSON
│   ├── processed/            # Extracted pages, chunks, eval reports
│   └── test_set/             # 50-question evaluation set
├── prompts/                  # Versioned prompt templates
├── src/                      # Core library
│   ├── config.py
│   ├── document_loader.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retrieval.py
│   ├── generation.py
│   ├── agents.py
│   ├── evaluation.py
│   └── monitoring.py
├── api/                      # FastAPI application
│   ├── main.py
│   ├── routes.py
│   └── models.py
├── frontend/
│   └── app.py                # Streamlit chat UI
├── monitoring/
│   └── dashboard.py          # Streamlit monitoring dashboard
├── docs/                     # Architecture, comparison tables
├── index_pipeline.py         # One-time indexing script
├── requirements.txt
├── .env.example
└── ASSUMPTIONS.md
```

---

## Key Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Chunking default | Recursive | Respects paragraph/sentence boundaries; best for medical prose |
| Chunk size | 512 chars + 100 overlap | Fits meaningful clinical statements within context |
| Retrieval | Hybrid (0.7 semantic + 0.3 BM25) | Medical abbreviations and drug names benefit from exact keyword matching |
| Reranker | Cohere `rerank-english-v3.0` | Free tier; cross-encoder gives significant precision boost |
| LLM temperature | 0.1 | Clinical answers require determinism, not creativity |
| Default model | `gpt-4o-mini` | Best cost/quality for factual extraction |

---

## Limitations and Next Steps

- **No HIPAA compliance** — not suitable for real patient data without proper hardening
- **Static guidelines** — index must be rebuilt when guidelines are updated
- **English only** — no multilingual support
- **No authentication** — API endpoints are open; add OAuth/API keys for production
- **Evaluation is automated** — source accuracy measured by substring match; human review improves metrics

---

## Disclaimer

This tool is for **informational and educational purposes only**. It is NOT a substitute for professional clinical judgment. Always consult current guidelines and apply clinical expertise when making patient care decisions.

---

## License

MIT — see [LICENSE](LICENSE).
