"""Three text-chunking strategies: fixed-size, recursive, and section-based."""

from __future__ import annotations

import json
import os
import re
from typing import Literal

from src.config import CHUNK_OVERLAP, CHUNK_SIZE, EXTRACTED_PAGES_PATH, PROCESSED_DATA_DIR


def chunk_fixed_size(text: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
    """Strategy 1: fixed-size chunking with character overlap."""
    chunks: list[str] = []
    start = 0
    step = max(chunk_size - overlap, 1)
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        if end >= len(text):
            break
        start += step
    return chunks


def chunk_recursive(text: str, chunk_size: int = 512, overlap: int = 100) -> list[str]:
    """Strategy 2: recursive splitting - paragraphs first, then sentences, then chars.

    `overlap` is unused in the current implementation but kept for API symmetry.
    """
    paragraphs = text.split("\n\n")

    chunks: list[str] = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current_chunk) + len(para) <= chunk_size:
            current_chunk = f"{current_chunk}\n\n{para}" if current_chunk else para
            continue

        if current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = ""

        if len(para) > chunk_size:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sentence in sentences:
                if len(current_chunk) + len(sentence) <= chunk_size:
                    current_chunk = f"{current_chunk} {sentence}" if current_chunk else sentence
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    # Sentence longer than chunk_size: hard-split
                    if len(sentence) > chunk_size:
                        for i in range(0, len(sentence), chunk_size):
                            chunks.append(sentence[i : i + chunk_size].strip())
                        current_chunk = ""
                    else:
                        current_chunk = sentence
        else:
            current_chunk = para

    if current_chunk:
        chunks.append(current_chunk.strip())

    # Enforce overlap between consecutive chunks for retrieval continuity
    if overlap and len(chunks) > 1:
        overlapped = [chunks[0]]
        for prev, curr in zip(chunks, chunks[1:]):
            tail = prev[-overlap:] if len(prev) > overlap else prev
            overlapped.append(f"{tail} {curr}".strip())
        chunks = overlapped

    return chunks


def chunk_section_based(text: str) -> list[str]:
    """Strategy 3: split on detected section headers common to clinical guidelines."""
    section_patterns = [
        r"\n(?=\d+\.\s+[A-Z])",  # "1. Introduction"
        r"\n(?=\d+\.\d+\s+[A-Z])",  # "1.1 Background"
        r"\n(?=[A-Z][A-Z\s]{5,})\n",  # "RECOMMENDATIONS"
        r"\n(?=Chapter\s+\d+)",
        r"\n(?=Section\s+\d+)",
        r"\n(?=Table\s+\d+)",
        r"\n(?=Recommendation\s+\d+)",
    ]
    combined_pattern = "|".join(section_patterns)
    sections = re.split(combined_pattern, text)
    return [s.strip() for s in sections if s and len(s.strip()) > 50]


def create_chunks_with_metadata(
    pages: list[dict],
    strategy: Literal["fixed", "recursive", "section"] = "recursive",
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """Turn page dicts into chunk dicts with full metadata preserved."""
    all_chunks: list[dict] = []
    chunk_index = 0

    for page in pages:
        text = page["text"]

        if strategy == "fixed":
            text_chunks = chunk_fixed_size(text, chunk_size, overlap)
        elif strategy == "recursive":
            text_chunks = chunk_recursive(text, chunk_size, overlap)
        elif strategy == "section":
            text_chunks = chunk_section_based(text)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        for chunk_text in text_chunks:
            all_chunks.append(
                {
                    "chunk_id": f"{page['guideline_id']}_chunk_{chunk_index}",
                    "text": chunk_text,
                    "source_file": page["source_file"],
                    "page_number": page["page_number"],
                    "guideline_id": page["guideline_id"],
                    "guideline_title": page["guideline_title"],
                    "source_organization": page["source_organization"],
                    "year": page["year"],
                    "topic": page["topic"],
                    "specialty": page["specialty"],
                    "chunking_strategy": strategy,
                    "chunk_length": len(chunk_text),
                }
            )
            chunk_index += 1

    print(f"Created {len(all_chunks)} chunks using '{strategy}' strategy")
    return all_chunks


if __name__ == "__main__":
    with open(EXTRACTED_PAGES_PATH, "r", encoding="utf-8") as f:
        pages = json.load(f)

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    for strategy in ("fixed", "recursive", "section"):
        chunks = create_chunks_with_metadata(pages, strategy=strategy)  # type: ignore[arg-type]
        output_path = os.path.join(PROCESSED_DATA_DIR, f"chunks_{strategy}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2)

        lengths = [c["chunk_length"] for c in chunks] or [0]
        print(f"  Strategy: {strategy}")
        print(f"  Total chunks: {len(chunks)}")
        print(f"  Avg chunk length: {sum(lengths) // max(len(lengths), 1)} chars")
        print(f"  Min: {min(lengths)}, Max: {max(lengths)}")
        print()
