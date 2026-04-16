"""
Run this script once to extract PDFs, chunk them, and index into ChromaDB.

Usage:
    python index_pipeline.py                  # default recursive chunking
    python index_pipeline.py --strategy fixed
    python index_pipeline.py --strategy section
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Make sure project root is on sys.path when run directly
ROOT = str(Path(__file__).resolve().parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.chunking import create_chunks_with_metadata
from src.config import (
    EXTRACTED_PAGES_PATH,
    METADATA_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)
from src.document_loader import load_all_guidelines, save_extracted_pages
from src.vector_store import index_chunks


def main():
    parser = argparse.ArgumentParser(description="Build the clinical guidelines index.")
    parser.add_argument(
        "--strategy",
        choices=["fixed", "recursive", "section"],
        default="recursive",
        help="Chunking strategy to use (default: recursive)",
    )
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Re-use existing extracted_pages.json instead of re-parsing PDFs",
    )
    args = parser.parse_args()

    # 1) Extract text from PDFs
    if args.skip_extraction and Path(EXTRACTED_PAGES_PATH).exists():
        print(f"Loading existing pages from {EXTRACTED_PAGES_PATH}")
        with open(EXTRACTED_PAGES_PATH, "r", encoding="utf-8") as f:
            pages = json.load(f)
    else:
        print("=== Step 1: Extract text from PDFs ===")
        pages = load_all_guidelines(RAW_DATA_DIR, METADATA_DIR)
        save_extracted_pages(pages, EXTRACTED_PAGES_PATH)

    if not pages:
        print("Error: no pages extracted. Check that PDF files are present in data/raw/.")
        sys.exit(1)

    # 2) Chunk
    print(f"\n=== Step 2: Chunk text (strategy={args.strategy}) ===")
    chunks = create_chunks_with_metadata(pages, strategy=args.strategy)  # type: ignore[arg-type]

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    chunks_path = os.path.join(PROCESSED_DATA_DIR, f"chunks_{args.strategy}.json")
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)
    print(f"Saved {len(chunks)} chunks to {chunks_path}")

    # Also save as the default filename so the API picks it up
    default_path = os.path.join(PROCESSED_DATA_DIR, "chunks_recursive.json")
    if args.strategy == "recursive" or not Path(default_path).exists():
        with open(default_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2)
        print(f"Saved as default chunks file: {default_path}")

    # 3) Index into ChromaDB
    print("\n=== Step 3: Index into ChromaDB ===")
    index_chunks(chunks)

    print("\n✅ Indexing complete.")
    print(f"   PDFs processed : {len({p['guideline_id'] for p in pages})}")
    print(f"   Pages extracted: {len(pages)}")
    print(f"   Chunks created : {len(chunks)}")
    print(f"   Chunking strat : {args.strategy}")


if __name__ == "__main__":
    main()
