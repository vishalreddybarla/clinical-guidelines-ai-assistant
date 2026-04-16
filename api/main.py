"""FastAPI application entry point."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router

app = FastAPI(
    title="Clinical Guidelines AI Assistant",
    description=(
        "RAG-powered API for answering clinical questions from WHO, ADA, AHA, CDC, "
        "and USPSTF guidelines. Supports hybrid search, reranking, drug interaction "
        "checks, and PubMed lookups."
    ),
    version="1.0.0",
    contact={"name": "Vishal Reddy Barla", "email": "vishalreddybarla@gmail.com"},
    license_info={"name": "MIT"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/", tags=["root"])
def root():
    return {
        "message": "Clinical Guidelines AI Assistant API",
        "docs": "/docs",
        "health": "/api/health",
    }
