"""
Retrieval (CLAUDE.md §6.2 steps 1-3): detect the question language, embed the
question, and run a hybrid dense + sparse/BM25 search in Qdrant with optional
metadata filters (``price`` range, ``category``, ``language``, ``course_id``).

The two branches are fused server-side with Reciprocal Rank Fusion in one
``query_points`` call. Assembling context and calling the LLM (steps 4-6) is
Phase 4.
"""

from __future__ import annotations

from dataclasses import dataclass

from qdrant_client import models

from config import settings
from ingest.embed_upsert import (
    DENSE_VECTOR,
    SPARSE_VECTOR,
    embed_texts,
    get_qdrant_client,
    sparse_embed_query,
)
from ingest.parse import detect_language as _detect_language


@dataclass
class RetrievalFilters:
    course_id: str | None = None
    category: str | None = None
    language: str | None = None
    min_price: float | None = None
    max_price: float | None = None

    def is_empty(self) -> bool:
        return all(v is None for v in vars(self).values())


@dataclass
class RetrievedChunk:
    course_id: str | None
    chunk_index: int | None
    text: str
    score: float
    title: str | None
    category: str | None
    price: float | None
    currency: str | None
    language: str | None
    source_file: str | None
    duration: str | None = None
    target_audience: str | None = None


def detect_language(text: str) -> str:
    """Language of the user's question: 'ar' | 'en' | 'mixed' (§6.2 step 1)."""
    return _detect_language(text)


def _build_filter(filters: RetrievalFilters | None) -> models.Filter | None:
    if filters is None or filters.is_empty():
        return None
    must: list[models.FieldCondition] = []
    for key, value in (
        ("course_id", filters.course_id),
        ("category", filters.category),
        ("language", filters.language),
    ):
        if value is not None:
            must.append(models.FieldCondition(key=key, match=models.MatchValue(value=value)))
    if filters.min_price is not None or filters.max_price is not None:
        must.append(
            models.FieldCondition(
                key="price",
                range=models.Range(gte=filters.min_price, lte=filters.max_price),
            )
        )
    return models.Filter(must=must) if must else None


def _to_chunk(point) -> RetrievedChunk:
    payload = point.payload or {}
    return RetrievedChunk(
        course_id=payload.get("course_id"),
        chunk_index=payload.get("chunk_index"),
        text=payload.get("text") or "",
        score=point.score,
        title=payload.get("title"),
        category=payload.get("category"),
        price=payload.get("price"),
        currency=payload.get("currency"),
        language=payload.get("language"),
        source_file=payload.get("source_file"),
        duration=payload.get("duration"),
        target_audience=payload.get("target_audience"),
    )


def retrieve(
    query: str,
    *,
    top_k: int | None = None,
    filters: RetrievalFilters | None = None,
) -> list[RetrievedChunk]:
    """Hybrid (dense + BM25) top-k retrieval with optional metadata filters."""
    top_k = top_k or settings.retrieval_top_k
    query_filter = _build_filter(filters)
    prefetch_limit = settings.hybrid_prefetch_limit

    dense_vector = embed_texts([query])[0]
    sparse_indices, sparse_values = sparse_embed_query(query)

    result = get_qdrant_client().query_points(
        collection_name=settings.qdrant_collection,
        prefetch=[
            models.Prefetch(
                query=dense_vector,
                using=DENSE_VECTOR,
                filter=query_filter,
                limit=prefetch_limit,
            ),
            models.Prefetch(
                query=models.SparseVector(indices=sparse_indices, values=sparse_values),
                using=SPARSE_VECTOR,
                filter=query_filter,
                limit=prefetch_limit,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_k,
        with_payload=True,
    )
    return [_to_chunk(p) for p in result.points]
