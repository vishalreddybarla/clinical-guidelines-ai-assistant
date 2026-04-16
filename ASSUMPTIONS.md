# Assumptions and Design Decisions

This file documents the assumptions and choices made during implementation where the project spec left room for interpretation.

## Data & PDF Naming

| Assumption | Decision |
|-----------|----------|
| PDF filenames may differ from spec | Renamed all PDFs to snake_case standard names (`who_hypertension_2021.pdf`, etc.) at ingestion |
| USPSTF PDFs come as 4 separate files | Stored them in `data/raw/uspstf_cancer_screening/` sub-directory; `load_all_guidelines` uses `rglob` to find them |
| Metadata matching | Metadata JSON matched first by stem (`who_hypertension_2021.json`), then by `filename` field in JSON |

## Model IDs

| Assumption | Decision |
|-----------|----------|
| "Claude Haiku" in spec is vague | Used `claude-3-5-haiku-20241022` as the most current Haiku model as of build date |
| Supported models | `gpt-4o-mini`, `gpt-4o`, `claude-3-5-haiku-20241022`, `claude-3-haiku-20240307` |

## Cohere Reranking

| Assumption | Decision |
|-----------|----------|
| Cohere is optional | If `COHERE_API_KEY` is absent, `rerank()` silently returns the top-k results in their original order |

## Drug Interaction Detection

| Assumption | Decision |
|-----------|----------|
| Drug name list scope | A curated list of ~35 common drugs is used for keyword detection; the RxNorm API is called for lookups |
| Drug interactions only returned when another drug in the query matches | Avoids surfacing unrelated interactions for single-drug queries |

## Chunking Default

| Assumption | Decision |
|-----------|----------|
| Best default strategy | `recursive` chosen as it respects paragraph/sentence boundaries — most meaningful for clinical text |
| Default chunk file name | Always saved as `chunks_recursive.json` (the name the API loads) regardless of which strategy built it |

## Test Set

| Assumption | Decision |
|-----------|----------|
| 50 questions with difficulty distribution | 20 easy / 20 medium / 10 hard, 10 per guideline, as specified |
| Evaluation metric | `correct_source_retrieved` = expected source name found in citation strings (case-insensitive substring match) |

## API Defaults

| Assumption | Decision |
|-----------|----------|
| Default prompt version | `v3_few_shot` (highest quality, few-shot with abstention example) |
| Default LLM | `gpt-4o-mini` for balance of cost/quality |
| Chunk corpus for BM25 | Loaded once at module import from `data/processed/chunks_recursive.json`; hybrid search degrades to semantic-only if file absent |

## Frontend

| Assumption | Decision |
|-----------|----------|
| API base URL | Defaults to `http://localhost:8000/api`; overridable via `API_BASE_URL` env var |
| Feedback star rating | Streamlit `st.feedback("stars")` returns 0-indexed value; converted to 1–5 before posting |

## Security

| Assumption | Decision |
|-----------|----------|
| API keys | Never committed; `.env` in `.gitignore`; `.env.example` committed with placeholder values |
| PDF files | In `.gitignore` to avoid large binary commits |
