"""Retrieval: semantic search, BM25 keyword search, hybrid fusion, and reranking."""

from __future__ import annotations

import numpy as np
from rank_bm25 import BM25Okapi

from src.config import COHERE_API_KEY
from src.embeddings import embed_text
from src.vector_store import query_collection


def semantic_search(
    query: str,
    top_k: int = 10,
    filter_metadata: dict | None = None,
) -> list[dict]:
    """Basic semantic search using embedding similarity."""
    query_embedding = embed_text(query)
    results = query_collection(query_embedding, top_k=top_k, filter_metadata=filter_metadata)

    formatted: list[dict] = []
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for doc, meta, dist in zip(documents, metadatas, distances):
        formatted.append(
            {
                "text": doc,
                "metadata": meta,
                "distance": dist,
                "relevance_score": 1 - dist,
            }
        )
    return formatted


def keyword_search(query: str, all_chunks: list[dict], top_k: int = 10) -> list[dict]:
    """BM25 keyword search over the provided chunk corpus."""
    if not all_chunks:
        return []

    tokenized_corpus = [chunk["text"].lower().split() for chunk in all_chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        chunk = all_chunks[int(idx)]
        results.append(
            {
                "text": chunk["text"],
                "metadata": {
                    "source_file": chunk["source_file"],
                    "page_number": chunk["page_number"],
                    "guideline_id": chunk["guideline_id"],
                    "guideline_title": chunk["guideline_title"],
                    "source_organization": chunk["source_organization"],
                    "topic": chunk["topic"],
                },
                "bm25_score": float(scores[idx]),
            }
        )
    return results


def hybrid_search(
    query: str,
    all_chunks: list[dict],
    top_k: int = 10,
    semantic_weight: float = 0.7,
    keyword_weight: float = 0.3,
    filter_metadata: dict | None = None,
) -> list[dict]:
    """Combine normalized semantic and BM25 scores."""
    semantic_results = semantic_search(query, top_k=top_k * 2, filter_metadata=filter_metadata)
    keyword_results = keyword_search(query, all_chunks, top_k=top_k * 2)

    scores: dict[str, dict] = {}

    for result in semantic_results:
        key = result["text"][:100]
        scores[key] = {
            "text": result["text"],
            "metadata": result["metadata"],
            "semantic_score": result["relevance_score"],
            "keyword_score": 0.0,
        }

    for result in keyword_results:
        key = result["text"][:100]
        if key in scores:
            scores[key]["keyword_score"] = result["bm25_score"]
        else:
            scores[key] = {
                "text": result["text"],
                "metadata": result["metadata"],
                "semantic_score": 0.0,
                "keyword_score": result["bm25_score"],
            }

    max_semantic = max((s["semantic_score"] for s in scores.values()), default=1.0) or 1.0
    max_keyword = max((s["keyword_score"] for s in scores.values()), default=1.0) or 1.0

    for entry in scores.values():
        norm_semantic = entry["semantic_score"] / max_semantic
        norm_keyword = entry["keyword_score"] / max_keyword
        entry["combined_score"] = semantic_weight * norm_semantic + keyword_weight * norm_keyword

    ranked = sorted(scores.values(), key=lambda x: x["combined_score"], reverse=True)
    return ranked[:top_k]


def rerank(query: str, results: list[dict], top_k: int = 5) -> list[dict]:
    """Rerank results via Cohere. Falls back to the original order if unavailable."""
    if not results:
        return []
    if not COHERE_API_KEY:
        return results[:top_k]

    try:
        import cohere  # local import to keep optional

        co = cohere.Client(COHERE_API_KEY)
        documents = [r["text"] for r in results]
        rerank_response = co.rerank(
            query=query,
            documents=documents,
            top_n=top_k,
            model="rerank-english-v3.0",
        )
        reranked: list[dict] = []
        for item in rerank_response.results:
            enriched = results[item.index].copy()
            enriched["rerank_score"] = item.relevance_score
            reranked.append(enriched)
        return reranked
    except Exception as exc:  # noqa: BLE001
        print(f"Reranking failed: {exc}. Using original order.")
        return results[:top_k]
