"""
Embedding + idempotent upsert into Qdrant (hybrid: dense + sparse/BM25).

- Dense embeddings go through a provider factory driven by config
  (``EMBEDDING_PROVIDER`` / ``EMBEDDING_MODEL`` / ``EMBEDDING_DIM``); the sparse
  branch is fastembed BM25 (``SPARSE_MODEL``). Nothing is locked before the
  Phase 5 benchmark.
- Every point carries a named ``dense`` vector and a named ``sparse`` vector;
  the collection also declares the ``IDF`` modifier so Qdrant computes BM25 IDF
  server-side.
- Point IDs are deterministic (``uuid5`` of ``course_id`` + ``chunk_index``) and
  every existing point for the course is deleted before upsert, so re-running
  ingestion never duplicates or leaves stale vectors (CLAUDE.md §6.1 step 6).

Query-time hybrid search + metadata filters live in ``query/retrieve.py``.
"""

from __future__ import annotations

import uuid
from functools import lru_cache
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    Modifier,
    PayloadSchemaType,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from config import settings

# Fixed namespace so point IDs are stable across runs and machines.
_POINT_NAMESPACE = uuid.UUID("6b1f4e9a-1c2d-4e3f-8a5b-9c0d1e2f3a4b")

# Named vectors on every point / in the collection schema.
DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"


def get_qdrant_client() -> QdrantClient:
    if not settings.qdrant_url or not settings.qdrant_api_key:
        raise RuntimeError("QDRANT_URL / QDRANT_API_KEY are not set — fill them in .env.")
    return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=60)


@lru_cache(maxsize=1)
def _build_embedder():
    """Return a LlamaIndex embedding model for the configured provider."""
    provider = settings.embedding_provider
    if provider == "openai":
        from llama_index.embeddings.openai import OpenAIEmbedding

        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set — fill it in .env.")
        return OpenAIEmbedding(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
            dimensions=settings.embedding_dim,
        )
    # Phase 5 benchmark candidates — wire cohere / voyage here when needed.
    raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {provider!r}")


def embed_texts(texts: list[str]) -> list[list[float]]:
    return _build_embedder().get_text_embedding_batch(texts, show_progress=False)


@lru_cache(maxsize=1)
def _build_sparse_embedder():
    """fastembed BM25 sparse model (lazy import — heavy-ish dependency)."""
    from fastembed import SparseTextEmbedding

    return SparseTextEmbedding(model_name=settings.sparse_model)


def sparse_embed_texts(texts: list[str]) -> list[tuple[list[int], list[float]]]:
    """BM25 sparse vectors for DOCUMENTS (term-frequency weighted)."""
    return [
        (emb.indices.tolist(), emb.values.tolist())
        for emb in _build_sparse_embedder().embed(texts)
    ]


def sparse_embed_query(text: str) -> tuple[list[int], list[float]]:
    """BM25 sparse vector for a QUERY (presence indicators; IDF applied server-side)."""
    emb = next(iter(_build_sparse_embedder().query_embed(text)))
    return emb.indices.tolist(), emb.values.tolist()


# Payload fields we filter on. course_id is needed now for idempotent
# delete-by-course; the rest are for Phase 3 metadata filters (price < X, etc.).
_PAYLOAD_INDEXES: dict[str, PayloadSchemaType] = {
    "course_id": PayloadSchemaType.KEYWORD,
    "category": PayloadSchemaType.KEYWORD,
    "language": PayloadSchemaType.KEYWORD,
    "price": PayloadSchemaType.FLOAT,
}


def _create_hybrid_collection(client: QdrantClient, name: str) -> None:
    client.create_collection(
        collection_name=name,
        vectors_config={
            DENSE_VECTOR: VectorParams(
                size=settings.embedding_dim, distance=Distance.COSINE
            )
        },
        sparse_vectors_config={
            SPARSE_VECTOR: SparseVectorParams(modifier=Modifier.IDF)
        },
    )


def ensure_collection(client: QdrantClient) -> None:
    name = settings.qdrant_collection

    if not client.collection_exists(name):
        _create_hybrid_collection(client, name)
    else:
        params = client.get_collection(name).config.params
        vectors = params.vectors
        sparse = params.sparse_vectors or {}
        is_hybrid = (
            isinstance(vectors, dict)
            and DENSE_VECTOR in vectors
            and SPARSE_VECTOR in sparse
        )
        if not is_hybrid:
            print(
                f"WARNING: collection '{name}' predates hybrid search — dropping and "
                "recreating it. Re-ingest every course."
            )
            client.delete_collection(name)
            _create_hybrid_collection(client, name)
        elif vectors[DENSE_VECTOR].size != settings.embedding_dim:
            raise RuntimeError(
                f"Collection '{name}' dense vector size is {vectors[DENSE_VECTOR].size}, "
                f"but config EMBEDDING_DIM is {settings.embedding_dim}. Point "
                "QDRANT_COLLECTION at a fresh name or recreate the collection."
            )

    existing_indexes = client.get_collection(name).payload_schema or {}
    for field, schema in _PAYLOAD_INDEXES.items():
        if field not in existing_indexes:
            client.create_payload_index(
                collection_name=name, field_name=field, field_schema=schema, wait=True
            )


def _point_id(course_id: str, chunk_index: int) -> str:
    return str(uuid.uuid5(_POINT_NAMESPACE, f"{course_id}:{chunk_index}"))


def delete_course(client: QdrantClient, course_id: str) -> None:
    """Remove all points for a course so re-ingestion is fully idempotent."""
    client.delete(
        collection_name=settings.qdrant_collection,
        points_selector=Filter(
            must=[FieldCondition(key="course_id", match=MatchValue(value=course_id))]
        ),
        wait=True,
    )


def list_course_ids(client: QdrantClient) -> set[str]:
    """Distinct course_id values currently stored in the collection."""
    ids: set[str] = set()
    offset = None
    while True:
        records, offset = client.scroll(
            collection_name=settings.qdrant_collection,
            with_payload=["course_id"],
            with_vectors=False,
            limit=256,
            offset=offset,
        )
        ids.update(
            cid for r in records if (cid := (r.payload or {}).get("course_id"))
        )
        if offset is None:
            break
    return ids


def prune_courses(client: QdrantClient, keep_ids: set[str]) -> list[str]:
    """Delete every course in the collection whose id is not in keep_ids."""
    removed = sorted(list_course_ids(client) - set(keep_ids))
    for course_id in removed:
        delete_course(client, course_id)
    return removed


def upsert_chunks(client: QdrantClient, chunks: list[dict[str, Any]]) -> int:
    if not chunks:
        return 0
    texts = [c["text"] for c in chunks]
    dense_vectors = embed_texts(texts)
    sparse_vectors = sparse_embed_texts(texts)
    points = [
        PointStruct(
            id=_point_id(c["metadata"]["course_id"], c["metadata"]["chunk_index"]),
            vector={
                DENSE_VECTOR: dense,
                SPARSE_VECTOR: SparseVector(indices=indices, values=values),
            },
            payload={**c["metadata"], "text": c["text"]},
        )
        for c, dense, (indices, values) in zip(chunks, dense_vectors, sparse_vectors)
    ]
    client.upsert(collection_name=settings.qdrant_collection, points=points, wait=True)
    return len(points)
