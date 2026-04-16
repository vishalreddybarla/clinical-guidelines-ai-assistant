"""Pydantic request/response models for the FastAPI endpoints."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    query: str = Field(..., min_length=5, description="Clinical question to answer")
    filter_source: Optional[str] = Field(None, description="Filter by source organization name")
    filter_topic: Optional[str] = Field(None, description="Filter by medical topic")
    model: Optional[str] = Field("gpt-4o-mini", description="LLM model to use")
    prompt_version: Optional[str] = Field("v3_few_shot", description="Prompt version to use")


class Citation(BaseModel):
    source_organization: str
    guideline_title: str
    page_number: str
    relevance_score: float


class DrugInteraction(BaseModel):
    drugs: list[str]
    description: str
    severity: str


class PubMedArticle(BaseModel):
    pmid: str
    title: str
    authors: str
    journal: str
    pub_date: str
    url: str


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    model_used: str
    tokens_used: int
    latency_ms: int
    tools_used: list[str]
    cost_estimate_usd: float
    drug_interactions: Optional[list[DrugInteraction]] = None
    pubmed_articles: Optional[list[PubMedArticle]] = None
    query_id: str


class FeedbackRequest(BaseModel):
    query_id: str
    rating: int = Field(..., ge=1, le=5, description="Rating from 1 (poor) to 5 (excellent)")
    comment: Optional[str] = None


class FeedbackResponse(BaseModel):
    status: str


class HealthResponse(BaseModel):
    status: str
    guidelines_loaded: int
    total_chunks: int
    vector_db: str
