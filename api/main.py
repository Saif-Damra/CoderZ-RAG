"""
FastAPI application (CLAUDE.md §9).

Exposes one customer-facing endpoint, ``POST /chat``, that wraps the existing
pipeline: hybrid retrieval (``query.retrieve``) + guarded bilingual generation
(``query.generate``). Plus ``GET /health`` for deploy checks.

The audience is end customers, so responses never carry stack traces or
provider error text — failures map to a short, safe message and the real error
is logged server-side.

Run:
    uvicorn api.main:app --reload
    python -m api.main            # convenience wrapper around uvicorn
"""

from __future__ import annotations

import logging
import time
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator, model_validator

from api import storage
from config import settings
from query.generate import answer_question
from query.retrieve import RetrievalFilters

logger = logging.getLogger("api")

app = FastAPI(
    title="RAG Courses API",
    version="0.6.0",
    description="Bilingual (AR/EN) course Q&A, grounded only in the course PDFs.",
)


def _cors_origins() -> list[str]:
    origins = [o.strip() for o in settings.cors_allow_origins.split(",") if o.strip()]
    return origins or ["*"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Request / response models
# --------------------------------------------------------------------------- #
class ChatFilters(BaseModel):
    """Optional metadata filters applied during retrieval (CLAUDE.md §6.2)."""

    course_id: str | None = None
    category: str | None = None
    language: str | None = Field(
        default=None, description="Filter courses by content language: 'ar' or 'en'."
    )
    min_price: float | None = Field(default=None, ge=0)
    max_price: float | None = Field(default=None, ge=0)

    @field_validator("language")
    @classmethod
    def _lang_enum(cls, v: str | None) -> str | None:
        if v is not None and v not in ("ar", "en"):
            raise ValueError("language filter must be 'ar' or 'en'")
        return v

    @model_validator(mode="after")
    def _price_range(self) -> "ChatFilters":
        if (
            self.min_price is not None
            and self.max_price is not None
            and self.min_price > self.max_price
        ):
            raise ValueError("min_price must not exceed max_price")
        return self

    def to_retrieval_filters(self) -> RetrievalFilters:
        return RetrievalFilters(
            course_id=self.course_id,
            category=self.category,
            language=self.language,
            min_price=self.min_price,
            max_price=self.max_price,
        )


class ChatRequest(BaseModel):
    question: str = Field(max_length=2000)
    language: str | None = Field(
        default=None, description="Force the reply language: 'ar' or 'en'. Auto-detected if omitted."
    )
    filters: ChatFilters | None = None

    @field_validator("question")
    @classmethod
    def _question_not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question must not be empty")
        return v

    @field_validator("language")
    @classmethod
    def _lang_enum(cls, v: str | None) -> str | None:
        if v is not None and v not in ("ar", "en"):
            raise ValueError("language must be 'ar' or 'en'")
        return v


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    used_context: bool
    language: str
    interaction_id: int | None = None


class FeedbackRequest(BaseModel):
    interaction_id: int
    rating: Literal["up", "down"]
    comment: str | None = Field(default=None, max_length=2000)


class FeedbackResponse(BaseModel):
    ok: bool


# --------------------------------------------------------------------------- #
# Error handling — keep internals off the wire
# --------------------------------------------------------------------------- #
_UNAVAILABLE = "The assistant is temporarily unavailable. Please try again later."
_FAILED = "The assistant could not process the request right now. Please try again."


@app.on_event("startup")
def _init_storage() -> None:
    try:
        storage.init_db()
    except Exception:
        logger.exception("feedback DB init failed — interaction logging will be skipped")


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": _FAILED})


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/")
def root() -> dict:
    return {"service": "RAG Courses API", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health() -> dict:
    model = (
        settings.openai_llm_model
        if settings.llm_provider == "openai"
        else settings.anthropic_model
    )
    return {
        "status": "ok",
        "llm_provider": settings.llm_provider,
        "llm_model": model,
        "embedding_model": settings.embedding_model,
        "collection": settings.qdrant_collection,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """Answer a customer question, grounded only in retrieved course context.

    Defined as a sync ``def`` on purpose: retrieval and generation do blocking
    network I/O, so Starlette runs this in its threadpool instead of stalling
    the event loop.
    """
    rf = req.filters.to_retrieval_filters() if req.filters else None
    started = time.monotonic()
    try:
        result = answer_question(req.question, filters=rf, language=req.language)
    except RuntimeError as exc:
        # Misconfiguration: missing API key / Qdrant creds.
        logger.error("Service dependency not ready: %s", exc)
        return JSONResponse(status_code=503, content={"detail": _UNAVAILABLE})
    except Exception:
        logger.exception("chat pipeline failed")
        return JSONResponse(status_code=502, content={"detail": _FAILED})

    elapsed_ms = int((time.monotonic() - started) * 1000)
    interaction_id: int | None = None
    try:
        interaction_id = storage.log_interaction(
            question=req.question,
            detected_language=result.language,
            answer=result.answer,
            sources=result.sources,
            used_context=result.used_context,
            response_time_ms=elapsed_ms,
        )
    except Exception:
        # Never let a logging failure turn a good answer into an error response.
        logger.exception("failed to log interaction")

    return ChatResponse(
        answer=result.answer,
        sources=result.sources,
        used_context=result.used_context,
        language=result.language,
        interaction_id=interaction_id,
    )


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(req: FeedbackRequest) -> FeedbackResponse:
    """Attach a thumbs up/down (+ optional comment) to a prior /chat reply."""
    try:
        ok = storage.record_feedback(req.interaction_id, req.rating, req.comment)
    except Exception:
        logger.exception("failed to record feedback")
        return JSONResponse(status_code=502, content={"detail": _FAILED})
    if not ok:
        return JSONResponse(status_code=404, content={"detail": "interaction_id not found"})
    return FeedbackResponse(ok=True)


# Serve the chat widget at /widget (kept off "/" so it doesn't shadow the
# JSON service banner above). Mounted last so it can't shadow API routes.
app.mount("/widget", StaticFiles(directory="web", html=True), name="widget")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
