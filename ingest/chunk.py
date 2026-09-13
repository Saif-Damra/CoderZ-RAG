"""
Chunking of the descriptive course text (Phase 1).

Token-based splitting via LlamaIndex's ``SentenceSplitter``, sized from config
(``chunk_size`` / ``chunk_overlap``; start ~500-800 tokens with light overlap per
CLAUDE.md §6.1, tuned in Phase 5). Every chunk carries the full course metadata
(CLAUDE.md §5) plus its ``chunk_index``.
"""

from __future__ import annotations

from typing import Any

from llama_index.core.node_parser import SentenceSplitter

from config import settings


def chunk_text(descriptive_text: str, metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """Split text into ``[{"text": str, "metadata": {..., "chunk_index": int}}]``."""
    splitter = SentenceSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    pieces = [p.strip() for p in splitter.split_text(descriptive_text) if p.strip()]
    return [
        {"text": text, "metadata": {**metadata, "chunk_index": idx}}
        for idx, text in enumerate(pieces)
    ]
