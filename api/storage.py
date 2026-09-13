"""
SQLite persistence for interaction logging + team feedback (thumbs up/down).

Single table, one row per /chat call. Feedback (rating/comment) is added later
via UPDATE from POST /feedback. Uses stdlib sqlite3 only — no new dependency.

Never let a storage failure break the user-facing /chat response: callers in
api/main.py must catch and log-and-continue.
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

from config import settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS interactions (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at        TEXT    NOT NULL,
    question          TEXT    NOT NULL,
    detected_language TEXT,
    answer            TEXT    NOT NULL,
    sources           TEXT    NOT NULL,
    used_context      INTEGER NOT NULL,
    response_time_ms  INTEGER,
    rating            TEXT,
    comment           TEXT,
    feedback_at       TEXT
);
"""


def _db_path() -> Path:
    p = Path(settings.feedback_db_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@contextmanager
def _connect():
    conn = sqlite3.connect(_db_path(), timeout=5.0)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _connect() as conn:
        conn.execute(_SCHEMA)


def log_interaction(
    *,
    question: str,
    detected_language: str | None,
    answer: str,
    sources: list[str],
    used_context: bool,
    response_time_ms: int | None,
) -> int:
    """Insert one interaction row and return its id."""
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO interactions "
            "(created_at, question, detected_language, answer, sources, used_context, response_time_ms) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                _now(),
                question,
                detected_language,
                answer,
                json.dumps(sources, ensure_ascii=False),
                int(used_context),
                response_time_ms,
            ),
        )
        return cur.lastrowid


def record_feedback(interaction_id: int, rating: str, comment: str | None) -> bool:
    """Attach a rating/comment to an existing interaction. False if id unknown."""
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE interactions SET rating = ?, comment = ?, feedback_at = ? WHERE id = ?",
            (rating, comment, _now(), interaction_id),
        )
        return cur.rowcount > 0
