"""Embedding helpers - single and batched calls with retry/backoff."""

from __future__ import annotations

import time

from openai import OpenAI

from src.config import EMBEDDING_MODEL, OPENAI_API_KEY


def _get_client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your .env file to generate embeddings."
        )
    return OpenAI(api_key=OPENAI_API_KEY)


def embed_text(text: str) -> list[float]:
    """Embed a single text string and return the vector."""
    client = _get_client()
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


def embed_batch(texts: list[str], batch_size: int = 100, max_retries: int = 3) -> list[list[float]]:
    """
    Embed a batch of texts with basic rate-limit handling.
    OpenAI accepts up to ~2048 inputs per request but 100 is a safe conservative batch.
    """
    client = _get_client()
    all_embeddings: list[list[float]] = []

    total_batches = max((len(texts) - 1) // batch_size + 1, 1)
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_num = i // batch_size + 1
        print(f"  Embedding batch {batch_num}/{total_batches}")

        attempt = 0
        while True:
            try:
                response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
                all_embeddings.extend(item.embedding for item in response.data)
                break
            except Exception as exc:  # noqa: BLE001
                attempt += 1
                if attempt > max_retries:
                    raise
                wait = 5 * attempt
                print(f"  Error: {exc}. Retrying in {wait}s (attempt {attempt}/{max_retries})...")
                time.sleep(wait)

        time.sleep(0.5)  # Gentle pacing between batches

    return all_embeddings
