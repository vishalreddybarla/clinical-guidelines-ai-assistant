"""FastAPI route definitions."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException

from api.models import (
    AskRequest,
    AskResponse,
    Citation,
    DrugInteraction,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    PubMedArticle,
)
from src.agents import run_agent
from src.config import DEFAULT_CHUNKS_PATH, estimate_cost
from src.monitoring import log_feedback, log_request
from src.vector_store import get_or_create_collection

router = APIRouter()

# ---------------------------------------------------------------------------
# Load chunk corpus once at import time for BM25 / hybrid search
# ---------------------------------------------------------------------------
_ALL_CHUNKS: list[dict] = []

def _load_chunks() -> list[dict]:
    global _ALL_CHUNKS
    if _ALL_CHUNKS:
        return _ALL_CHUNKS
    chunks_path = Path(DEFAULT_CHUNKS_PATH)
    if chunks_path.exists():
        with open(chunks_path, "r", encoding="utf-8") as f:
            _ALL_CHUNKS = json.load(f)
        print(f"Loaded {len(_ALL_CHUNKS)} chunks from {chunks_path}")
    else:
        print(
            f"Warning: chunk file not found at {chunks_path}. "
            "Run `python index_pipeline.py` first."
        )
    return _ALL_CHUNKS


# Eagerly load at module import (non-fatal if file is absent)
try:
    _load_chunks()
except Exception as _e:  # noqa: BLE001
    print(f"Warning: could not pre-load chunks: {_e}")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/ask", response_model=AskResponse, tags=["core"])
async def ask_question(request: AskRequest):
    """Ask a clinical question and receive a sourced answer with citations."""
    start_time = time.time()
    query_id = str(uuid.uuid4())

    all_chunks = _load_chunks()

    filter_meta: dict | None = None
    if request.filter_source:
        filter_meta = {"source_organization": request.filter_source}

    try:
        result = run_agent(
            query=request.query,
            all_chunks=all_chunks or None,
            prompt_version=request.prompt_version or "v3_few_shot",
            model=request.model or "gpt-4o-mini",
            filter_metadata=filter_meta,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    latency_ms = int((time.time() - start_time) * 1000)
    cost = estimate_cost(result["model"], result["tokens_in"], result["tokens_out"])

    try:
        log_request(
            query_id=query_id,
            query=request.query,
            model=result["model"],
            tokens_in=result["tokens_in"],
            tokens_out=result["tokens_out"],
            latency_ms=latency_ms,
            cost_usd=cost,
            tools_used=",".join(result.get("tools_used", ["rag"])),
        )
    except Exception as log_exc:  # noqa: BLE001
        print(f"Warning: logging failed: {log_exc}")

    # Build response citation objects (cap at 5)
    raw_citations = result.get("citations", [])[:5]
    citations = []
    for c in raw_citations:
        citations.append(
            Citation(
                source_organization=str(c.get("source_organization", "Unknown")),
                guideline_title=str(c.get("guideline_title", "Unknown")),
                page_number=str(c.get("page_number", "?")),
                relevance_score=float(c.get("relevance_score", 0.0)),
            )
        )

    # Drug interaction objects
    drug_interactions = None
    raw_drug = result.get("drug_interactions")
    if raw_drug and raw_drug.get("interactions"):
        drug_interactions = [
            DrugInteraction(
                drugs=i.get("drugs", []),
                description=i.get("description", ""),
                severity=i.get("severity", "Unknown"),
            )
            for i in raw_drug["interactions"]
        ]

    # PubMed article objects
    pubmed_articles = None
    raw_pubmed = result.get("pubmed_articles")
    if raw_pubmed:
        pubmed_articles = [
            PubMedArticle(
                pmid=a.get("pmid", ""),
                title=a.get("title", ""),
                authors=a.get("authors", ""),
                journal=a.get("journal", ""),
                pub_date=a.get("pub_date", ""),
                url=a.get("url", ""),
            )
            for a in raw_pubmed
        ]

    return AskResponse(
        answer=result["answer"],
        citations=citations,
        model_used=result["model"],
        tokens_used=result["total_tokens"],
        latency_ms=latency_ms,
        tools_used=result.get("tools_used", ["rag"]),
        cost_estimate_usd=round(cost, 6),
        drug_interactions=drug_interactions,
        pubmed_articles=pubmed_articles,
        query_id=query_id,
    )


@router.post("/feedback", response_model=FeedbackResponse, tags=["core"])
async def submit_feedback(request: FeedbackRequest):
    """Attach a user rating and optional comment to a previous query."""
    try:
        log_feedback(request.query_id, request.rating, request.comment)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return FeedbackResponse(status="feedback_recorded")


@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check():
    """Return API health status and data loaded statistics."""
    try:
        collection = get_or_create_collection()
        total_chunks = collection.count()
    except Exception:  # noqa: BLE001
        total_chunks = 0

    all_chunks = _load_chunks()
    guidelines_loaded = len({c.get("guideline_id") for c in all_chunks}) if all_chunks else 0

    return HealthResponse(
        status="healthy",
        guidelines_loaded=guidelines_loaded,
        total_chunks=total_chunks,
        vector_db="ChromaDB (persistent)",
    )
