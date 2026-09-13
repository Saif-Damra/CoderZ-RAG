"""
Central configuration for the RAG courses system.

All settings are loaded from environment variables (or a local .env file) via
pydantic-settings. NEVER hardcode secrets here — put them in .env, which is
git-ignored. See .env.example for the full list of keys.

The LLM provider and the embedding provider are both swappable at runtime by
changing a single env var (LLM_PROVIDER / EMBEDDING_PROVIDER). No embedding
model is locked in yet — the winner is chosen after the Phase 5 benchmark
(CLAUDE.md §4).
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------
    # Qdrant Cloud (vector DB) — required for the Phase 0 connection check
    # ------------------------------------------------------------------
    qdrant_url: str = Field(default="", description="Qdrant Cloud cluster URL")
    qdrant_api_key: str = Field(default="", description="Qdrant Cloud API key")
    qdrant_collection: str = Field(default="courses", description="Qdrant collection name")

    # ------------------------------------------------------------------
    # LLM / generation — swap provider via LLM_PROVIDER (needed from Phase 4)
    # ------------------------------------------------------------------
    llm_provider: Literal["anthropic", "openai"] = "anthropic"

    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    anthropic_model: str = Field(
        default="claude-sonnet-5",
        description="Claude model id used for generation (Phase 4).",
    )

    openai_api_key: str = Field(default="", description="OpenAI API key")
    openai_llm_model: str = Field(
        default="gpt-4o",
        description="OpenAI chat model used when LLM_PROVIDER=openai.",
    )

    # Generation tuning (Phase 4). Low temperature keeps answers grounded in the
    # retrieved context; the token cap keeps replies concise (CLAUDE.md §7).
    llm_temperature: float = Field(
        default=0.0, description="Sampling temperature for the generation LLM."
    )
    llm_max_tokens: int = Field(
        default=600, description="Max tokens in a generated answer."
    )
    llm_request_timeout: float = Field(
        default=60.0, description="Per-request timeout (seconds) for the generation LLM."
    )

    # ------------------------------------------------------------------
    # API (Phase 6) — FastAPI /chat
    # ------------------------------------------------------------------
    cors_allow_origins: str = Field(
        default="*",
        description=(
            "Comma-separated browser origins allowed to call the API. '*' for "
            "local dev; set the real frontend origin(s) before deploying."
        ),
    )

    # ------------------------------------------------------------------
    # Embeddings — swap provider via EMBEDDING_PROVIDER.
    # NOT finalised: the model is chosen after the Phase 5 benchmark.
    # ------------------------------------------------------------------
    embedding_provider: Literal["openai", "cohere", "voyage"] = "openai"
    embedding_model: str = Field(
        default="text-embedding-3-large",
        description="Embedding model id for the active provider.",
    )
    embedding_dim: int = Field(
        default=3072,
        description=(
            "Vector size for the active embedding model — must match the model. "
            "openai text-embedding-3-large=3072, text-embedding-3-small=1536, "
            "cohere embed-multilingual-v3.0=1024, voyage-3=1024."
        ),
    )
    cohere_api_key: str = Field(default="", description="Cohere API key (Phase 5 benchmark)")
    voyage_api_key: str = Field(default="", description="Voyage API key (Phase 5 benchmark)")

    # ------------------------------------------------------------------
    # Chunking + retrieval — safe defaults now, tuned in Phase 5 (CLAUDE.md §6.1)
    # ------------------------------------------------------------------
    chunk_size: int = Field(default=700, description="Target chunk size in tokens")
    chunk_overlap: int = Field(default=100, description="Overlap between chunks in tokens")
    retrieval_top_k: int = Field(default=5, description="Chunks returned per query (after fusion)")

    # Hybrid search (dense + sparse/BM25), CLAUDE.md §3 / §6.2
    sparse_model: str = Field(
        default="Qdrant/bm25",
        description="fastembed sparse model id for the BM25 branch.",
    )
    hybrid_prefetch_limit: int = Field(
        default=20,
        description="Candidates fetched from each branch (dense, sparse) before RRF fusion.",
    )


settings = Settings()


def _mask(value: str) -> str:
    """Mask a secret for display: keep first/last 4 chars only."""
    if not value:
        return "<empty>"
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}…{value[-4:]}"


if __name__ == "__main__":
    # Sanity check that .env is being parsed. Secrets are masked.
    print("RAG courses — resolved configuration")
    print("-" * 44)
    print(f"qdrant_url          : {settings.qdrant_url or '<empty>'}")
    print(f"qdrant_api_key      : {_mask(settings.qdrant_api_key)}")
    print(f"qdrant_collection   : {settings.qdrant_collection}")
    print(f"llm_provider        : {settings.llm_provider}")
    print(f"anthropic_model     : {settings.anthropic_model}")
    print(f"anthropic_api_key   : {_mask(settings.anthropic_api_key)}")
    print(f"openai_llm_model    : {settings.openai_llm_model}")
    print(f"openai_api_key      : {_mask(settings.openai_api_key)}")
    print(f"llm_temperature     : {settings.llm_temperature}")
    print(f"llm_max_tokens      : {settings.llm_max_tokens}")
    print(f"llm_request_timeout : {settings.llm_request_timeout}")
    print(f"cors_allow_origins  : {settings.cors_allow_origins}")
    print(f"embedding_provider  : {settings.embedding_provider}")
    print(f"embedding_model     : {settings.embedding_model}")
    print(f"embedding_dim       : {settings.embedding_dim}")
    print(f"chunk_size          : {settings.chunk_size}")
    print(f"chunk_overlap       : {settings.chunk_overlap}")
    print(f"retrieval_top_k     : {settings.retrieval_top_k}")
    print(f"sparse_model        : {settings.sparse_model}")
    print(f"hybrid_prefetch_limit: {settings.hybrid_prefetch_limit}")
