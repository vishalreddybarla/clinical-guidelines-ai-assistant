"""ChromaDB wrapper - index chunks and query by embedding."""

from __future__ import annotations

import chromadb

from src.config import CHROMA_DB_DIR, COLLECTION_NAME
from src.embeddings import embed_batch


def get_chroma_client() -> chromadb.api.ClientAPI:
    """Return a persistent ChromaDB client."""
    return chromadb.PersistentClient(path=CHROMA_DB_DIR)


def get_or_create_collection(client: chromadb.api.ClientAPI | None = None, name: str = COLLECTION_NAME):
    """Get or create a ChromaDB collection with cosine similarity."""
    if client is None:
        client = get_chroma_client()
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def index_chunks(chunks: list[dict], collection_name: str = COLLECTION_NAME):
    """Embed and store all chunks in ChromaDB. Replaces any existing collection."""
    client = get_chroma_client()

    try:
        client.delete_collection(collection_name)
    except Exception:  # noqa: BLE001
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    texts = [chunk["text"] for chunk in chunks]
    ids = [chunk["chunk_id"] for chunk in chunks]
    metadatas = [
        {
            "source_file": chunk["source_file"],
            "page_number": chunk["page_number"],
            "guideline_id": chunk["guideline_id"],
            "guideline_title": chunk["guideline_title"],
            "source_organization": chunk["source_organization"],
            "year": str(chunk["year"]),
            "topic": chunk["topic"],
            "specialty": chunk["specialty"],
        }
        for chunk in chunks
    ]

    print(f"Generating embeddings for {len(texts)} chunks...")
    embeddings = embed_batch(texts)

    batch_size = 500
    for i in range(0, len(texts), batch_size):
        end = min(i + batch_size, len(texts))
        collection.add(
            ids=ids[i:end],
            embeddings=embeddings[i:end],
            documents=texts[i:end],
            metadatas=metadatas[i:end],
        )
        print(f"  Indexed {end}/{len(texts)} chunks")

    print(f"Indexing complete. Total chunks: {collection.count()}")
    return collection


def query_collection(
    query_embedding: list[float],
    top_k: int = 10,
    filter_metadata: dict | None = None,
    collection_name: str = COLLECTION_NAME,
) -> dict:
    """Query the vector store for the top-k most similar chunks."""
    collection = get_or_create_collection(name=collection_name)

    query_params: dict = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if filter_metadata:
        query_params["where"] = filter_metadata

    return collection.query(**query_params)


def delete_collection(collection_name: str = COLLECTION_NAME) -> None:
    """Delete the named collection."""
    client = get_chroma_client()
    client.delete_collection(collection_name)
    print(f"Deleted collection: {collection_name}")
