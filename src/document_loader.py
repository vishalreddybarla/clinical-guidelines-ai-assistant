"""PDF ingestion: extract text per page and attach guideline-level metadata."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pdfplumber

from src.config import EXTRACTED_PAGES_PATH, METADATA_DIR, RAW_DATA_DIR


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """Extract text from a PDF page by page. Returns list of dicts per non-empty page."""
    pages: list[dict] = []
    filename = Path(pdf_path).name

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        for page_num, page in enumerate(pdf.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception as exc:  # noqa: BLE001 - pdfplumber can raise varied errors
                print(f"  Warning: failed to extract page {page_num} of {filename}: {exc}")
                continue
            if text.strip():
                pages.append(
                    {
                        "text": text.strip(),
                        "source_file": filename,
                        "page_number": page_num,
                        "total_pages": total_pages,
                    }
                )
    return pages


def load_metadata(metadata_path: str) -> dict:
    """Load a single metadata JSON file."""
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_metadata_for_pdf(pdf_path: Path, metadata_dir: str) -> dict:
    """Match a PDF file to its metadata JSON by stem or by filename field."""
    metadata_dir_path = Path(metadata_dir)

    # 1) Direct stem match (e.g., who_hypertension_2021.pdf -> who_hypertension_2021.json)
    candidate = metadata_dir_path / f"{pdf_path.stem}.json"
    if candidate.exists():
        return load_metadata(str(candidate))

    # 2) Scan all metadata files and match on the "filename" field
    for meta_file in metadata_dir_path.glob("*.json"):
        try:
            meta = load_metadata(str(meta_file))
        except Exception:  # noqa: BLE001
            continue
        if meta.get("filename") == pdf_path.name:
            return meta

    return {}


def load_all_guidelines(raw_dir: str, metadata_dir: str) -> list[dict]:
    """
    Load all PDFs from `raw_dir` (recursively), attach metadata, and return pages.
    """
    all_pages: list[dict] = []
    raw_path = Path(raw_dir)

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    pdf_files = sorted(raw_path.rglob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found under {raw_dir}")
        return []

    for pdf_file in pdf_files:
        print(f"Loading: {pdf_file.relative_to(raw_path)}")
        pages = extract_text_from_pdf(str(pdf_file))

        metadata = _find_metadata_for_pdf(pdf_file, metadata_dir)
        if not metadata:
            print(f"  Warning: no metadata file found for {pdf_file.name}; using defaults")

        for page in pages:
            page["guideline_id"] = metadata.get("id", pdf_file.stem)
            page["guideline_title"] = metadata.get("title", pdf_file.stem)
            page["source_organization"] = metadata.get("source_organization", "Unknown")
            page["year"] = metadata.get("year", "Unknown")
            page["topic"] = metadata.get("topic", "Unknown")
            page["specialty"] = metadata.get("specialty", "Unknown")
            page["url"] = metadata.get("url", "")

        all_pages.extend(pages)
        print(f"  Extracted {len(pages)} pages")

    print(f"\nTotal pages extracted: {len(all_pages)}")
    return all_pages


def save_extracted_pages(pages: list[dict], output_path: str) -> None:
    """Save extracted pages to JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pages, f, indent=2)
    print(f"Saved {len(pages)} pages to {output_path}")


if __name__ == "__main__":
    pages = load_all_guidelines(RAW_DATA_DIR, METADATA_DIR)
    save_extracted_pages(pages, EXTRACTED_PAGES_PATH)
