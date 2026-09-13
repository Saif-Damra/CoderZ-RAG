"""
Review collected /chat interactions + team feedback (thumbs up/down/comments).

    python scripts/export_feedback.py                    # console table, newest first
    python scripts/export_feedback.py --only-feedback     # just rated rows
    python scripts/export_feedback.py --csv out.csv       # full untruncated dump

Reads directly from the SQLite DB at settings.feedback_db_path — no server
needs to be running. Works the same inside the Docker container:

    docker compose exec api python scripts/export_feedback.py --only-feedback
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from pathlib import Path

# Windows consoles default to cp1252 — force UTF-8 so Arabic prints.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings  # noqa: E402

_COLUMNS = (
    "id",
    "created_at",
    "question",
    "detected_language",
    "answer",
    "sources",
    "used_context",
    "response_time_ms",
    "rating",
    "comment",
    "feedback_at",
)


def _truncate(value: object, width: int) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\n", " ")
    return text if len(text) <= width else text[: width - 1] + "…"


def fetch_rows(*, only_feedback: bool) -> list[sqlite3.Row]:
    db_path = Path(settings.feedback_db_path)
    if not db_path.exists():
        print(f"No feedback DB found at {db_path} — has the API served any /chat requests yet?")
        return []
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        query = f"SELECT {', '.join(_COLUMNS)} FROM interactions"
        if only_feedback:
            query += " WHERE rating IS NOT NULL"
        query += " ORDER BY created_at DESC"
        return conn.execute(query).fetchall()
    finally:
        conn.close()


def print_table(rows: list[sqlite3.Row]) -> None:
    if not rows:
        print("No interactions found.")
        return
    widths = {"id": 5, "created_at": 21, "question": 41, "rating": 7, "comment": 31, "used_context": 6, "response_time_ms": 8}
    header = (
        f"{'id':<{widths['id']}}{'created_at':<{widths['created_at']}}"
        f"{'question':<{widths['question']}}{'rating':<{widths['rating']}}"
        f"{'comment':<{widths['comment']}}{'ctx':<{widths['used_context']}}{'ms':<{widths['response_time_ms']}}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['id']:<{widths['id']}}"
            f"{_truncate(r['created_at'], widths['created_at'] - 1):<{widths['created_at']}}"
            f"{_truncate(r['question'], widths['question'] - 1):<{widths['question']}}"
            f"{_truncate(r['rating'] or '-', widths['rating'] - 1):<{widths['rating']}}"
            f"{_truncate(r['comment'] or '-', widths['comment'] - 1):<{widths['comment']}}"
            f"{str(bool(r['used_context'])):<{widths['used_context']}}"
            f"{r['response_time_ms'] if r['response_time_ms'] is not None else '-':<{widths['response_time_ms']}}"
        )
    print(f"\n{len(rows)} row(s).")


def write_csv(rows: list[sqlite3.Row], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(_COLUMNS)
        for r in rows:
            writer.writerow([r[c] for c in _COLUMNS])
    print(f"Wrote {len(rows)} row(s) to {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Review /chat interactions + feedback")
    ap.add_argument("--only-feedback", action="store_true", help="Only rows with a rating")
    ap.add_argument("--csv", metavar="PATH", help="Write the full dump to this CSV file")
    args = ap.parse_args()

    rows = fetch_rows(only_feedback=args.only_feedback)
    if args.csv:
        write_csv(rows, Path(args.csv))
    else:
        print_table(rows)


if __name__ == "__main__":
    main()
