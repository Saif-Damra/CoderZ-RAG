"""
Retrieval sanity check — hybrid (dense + BM25) search with optional metadata
filters. Run from the repo root:

    python scripts/check_retrieval.py "ما هي مواضيع الدورة؟"
    python scripts/check_retrieval.py "Power BI" --k 5
    python scripts/check_retrieval.py "دورة قصيرة" --max-price 400 --category AI
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings  # noqa: E402
from query.retrieve import RetrievalFilters, detect_language, retrieve  # noqa: E402


def run_query(
    query: str,
    k: int | None = None,
    *,
    category: str | None = None,
    language: str | None = None,
    course_id: str | None = None,
    min_price: float | None = None,
    max_price: float | None = None,
) -> None:
    filters = RetrievalFilters(
        course_id=course_id,
        category=category,
        language=language,
        min_price=min_price,
        max_price=max_price,
    )
    print(f"detected query language: {detect_language(query)}")
    if not filters.is_empty():
        print(f"filters: {filters}")

    hits = retrieve(query, top_k=k, filters=filters)
    if not hits:
        print("No results.")
        return
    for rank, hit in enumerate(hits, 1):
        price = f"{hit.price} {hit.currency or ''}".strip() if hit.price is not None else "-"
        print(
            f"\n#{rank}  score={hit.score:.4f}  course_id={hit.course_id}  "
            f"chunk_index={hit.chunk_index}  lang={hit.language}  "
            f"category={hit.category}  price={price}"
        )
        print(textwrap.indent(textwrap.fill(" ".join(hit.text.split()), width=100)[:600], "    "))


def main() -> None:
    ap = argparse.ArgumentParser(description="Hybrid retrieval check")
    ap.add_argument("query", help="Question to search for")
    ap.add_argument("--k", type=int, default=None, help=f"top-k (default {settings.retrieval_top_k})")
    ap.add_argument("--category")
    ap.add_argument("--language")
    ap.add_argument("--course", dest="course_id")
    ap.add_argument("--min-price", type=float, dest="min_price")
    ap.add_argument("--max-price", type=float, dest="max_price")
    args = ap.parse_args()
    run_query(
        args.query,
        args.k,
        category=args.category,
        language=args.language,
        course_id=args.course_id,
        min_price=args.min_price,
        max_price=args.max_price,
    )


if __name__ == "__main__":
    main()
