"""LLM answer generation with retrieved context and citation formatting."""

from __future__ import annotations

import os

from anthropic import Anthropic
from openai import OpenAI

from src.config import (
    ANTHROPIC_API_KEY,
    DEFAULT_MODEL,
    MAX_TOKENS,
    OPENAI_API_KEY,
    PROMPTS_DIR,
    TEMPERATURE,
)


def _openai_client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return OpenAI(api_key=OPENAI_API_KEY)


def _anthropic_client() -> Anthropic:
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY is not set.")
    return Anthropic(api_key=ANTHROPIC_API_KEY)


def format_context(retrieved_chunks: list[dict]) -> str:
    """Format retrieved chunks into a single context string with inline source markers."""
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        meta = chunk.get("metadata", {}) or {}
        org = meta.get("source_organization", "Unknown")
        title = meta.get("guideline_title", "Unknown")
        page = meta.get("page_number", "?")
        source = f"[{org} - {title}, Page {page}]"
        context_parts.append(f"Source {i} {source}:\n{chunk['text']}")
    return "\n\n---\n\n".join(context_parts)


def load_prompt(prompt_version: str) -> str:
    """Load a prompt template from the prompts directory."""
    path = os.path.join(PROMPTS_DIR, f"{prompt_version}.txt")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def generate_answer(
    query: str,
    retrieved_chunks: list[dict],
    prompt_version: str = "v3_few_shot",
    model: str = DEFAULT_MODEL,
) -> dict:
    """Generate an answer using the given LLM with retrieved context."""
    context = format_context(retrieved_chunks)
    system_prompt = load_prompt(prompt_version)
    user_message = (
        f"Context from clinical guidelines:\n\n{context}\n\n"
        f"---\n\nClinical question: {query}"
    )

    if model.startswith("claude"):
        client = _anthropic_client()
        response = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        answer_text = response.content[0].text
        tokens_in = response.usage.input_tokens
        tokens_out = response.usage.output_tokens
    else:
        client = _openai_client()
        response = client.chat.completions.create(
            model=model,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        answer_text = response.choices[0].message.content
        tokens_in = response.usage.prompt_tokens
        tokens_out = response.usage.completion_tokens

    citations = []
    for chunk in retrieved_chunks:
        meta = chunk.get("metadata", {}) or {}
        citations.append(
            {
                "source_organization": meta.get("source_organization", "Unknown"),
                "guideline_title": meta.get("guideline_title", "Unknown"),
                "page_number": str(meta.get("page_number", "Unknown")),
                "relevance_score": float(
                    chunk.get(
                        "relevance_score",
                        chunk.get("combined_score", chunk.get("rerank_score", 0.0)),
                    )
                ),
            }
        )

    return {
        "answer": answer_text,
        "citations": citations,
        "model": model,
        "prompt_version": prompt_version,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "total_tokens": tokens_in + tokens_out,
    }
