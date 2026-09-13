"""
Ingestion entrypoint (CLAUDE.md §6.1): batch over every course file (.pdf or
.docx) in data/pdfs/, or a single file with --pdf.

    parse -> chunk -> embed -> ensure collection -> replace course points -> upsert

Per-file failures are isolated (one bad file never aborts the run). A JSON report
is written for review (CLAUDE.md §12.1).

    python -m ingest.run_ingest                     # all .pdf/.docx in data/pdfs/
    python -m ingest.run_ingest --pdf data/pdfs/DataHub.pdf
    python -m ingest.run_ingest --pdf data/pdfs/DataHub.docx
    python -m ingest.run_ingest --prune            # also drop courses whose PDF is gone
    python -m ingest.run_ingest --query "كم عدد الساعات التدريبية؟"

Hybrid retrieval is Phase 3; generation is Phase 4.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Windows consoles default to cp1252, which can't encode characters that show
# up in real course text (Arabic, narrow no-break spaces in price/duration
# scans, etc.) — without this, printing the manifest crashes mid-run, before
# the JSON report is written.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from config import settings  # noqa: E402
from ingest.chunk import chunk_text  # noqa: E402
from ingest.embed_upsert import (  # noqa: E402
    delete_course,
    ensure_collection,
    get_qdrant_client,
    prune_courses,
    upsert_chunks,
)
from ingest.parse import parse_course_docx, parse_course_pdf  # noqa: E402

_DEFAULT_DIR = Path("data/pdfs")
_DEFAULT_REPORT = Path("data/ingest_report.json")
_PARSERS = {".pdf": parse_course_pdf, ".docx": parse_course_docx}


@dataclasses.dataclass
class CourseResult:
    course_id: str
    source_file: str
    title: str | None = None
    category: str | None = None
    price: float | None = None
    currency: str | None = None
    language: str | None = None
    chunks: int = 0
    status: str = "ok"  # "ok" | "error"
    warnings: list[str] = dataclasses.field(default_factory=list)
    error: str | None = None
    metadata: dict | None = None


def run_one(client, pdf_path: Path) -> CourseResult:
    res = CourseResult(course_id=pdf_path.stem.lower(), source_file=pdf_path.name)
    try:
        parser = _PARSERS.get(pdf_path.suffix.lower())
        if parser is None:
            raise ValueError(f"unsupported file type: {pdf_path.suffix}")
        descriptive_text, metadata, warnings = parser(pdf_path)
        chunks = chunk_text(descriptive_text, metadata)
        delete_course(client, metadata["course_id"])
        n = upsert_chunks(client, chunks)
        res.course_id = metadata["course_id"]
        res.title = metadata.get("title")
        res.category = metadata.get("category")
        res.price = metadata.get("price")
        res.currency = metadata.get("currency")
        res.language = metadata.get("language")
        res.chunks = n
        res.warnings = warnings
        res.metadata = metadata
    except Exception as exc:  # noqa: BLE001 - isolate per-file failures
        res.status = "error"
        res.error = f"{type(exc).__name__}: {exc}"
    return res


def _flag_duplicate_ids(results: list[CourseResult]) -> None:
    seen: dict[str, str] = {}
    for r in results:
        if r.status != "ok":
            continue
        if r.course_id in seen:
            r.status = "error"
            r.error = f"duplicate course_id '{r.course_id}' (also from {seen[r.course_id]})"
        else:
            seen[r.course_id] = r.source_file


def _print_manifest(results: list[CourseResult]) -> None:
    header = f"{'course_id':<18}{'lang':<7}{'category':<14}{'price':<14}{'chunks':>7}  status"
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        price = f"{r.price:g} {r.currency}" if r.price is not None else "-"
        print(
            f"{r.course_id:<18}{(r.language or '-'):<7}{(r.category or '-'):<14}"
            f"{price:<14}{r.chunks:>7}  {r.status}"
        )
        for w in r.warnings:
            print(f"    ! {w}")
        if r.error:
            print(f"    x {r.error}")


def _write_report(path: Path, results: list[CourseResult]) -> None:
    payload = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "collection": settings.qdrant_collection,
        "courses": [dataclasses.asdict(r) for r in results],
        "needs_attention": [
            {"course_id": r.course_id, "issues": r.warnings}
            for r in results
            if r.status == "ok" and r.warnings
        ],
        "errors": [
            {"source_file": r.source_file, "error": r.error}
            for r in results
            if r.status == "error"
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nReport written to {path}")


def _finish(client, results: list[CourseResult], report_path: Path, query: str | None) -> None:
    _print_manifest(results)
    total = client.count(settings.qdrant_collection).count
    ok = sum(r.status == "ok" for r in results)
    print(
        f"\n{ok}/{len(results)} course(s) ingested "
        f"(chunk_size={settings.chunk_size}, overlap={settings.chunk_overlap}). "
        f"Collection '{settings.qdrant_collection}' holds {total} points."
    )
    _write_report(report_path, results)
    if query:
        from scripts.check_retrieval import run_query

        print(f"\n--- retrieval check: {query!r} ---")
        run_query(query)


def run_all(
    pdf_dir: Path = _DEFAULT_DIR,
    *,
    query: str | None = None,
    prune: bool = False,
    report_path: Path = _DEFAULT_REPORT,
) -> None:
    pdf_dir = Path(pdf_dir)
    pdfs = sorted(p for ext in _PARSERS for p in pdf_dir.glob(f"*{ext}"))
    if not pdfs:
        print(f"No course files (.pdf/.docx) found in {pdf_dir}/")
        return

    client = get_qdrant_client()
    ensure_collection(client)
    results = [run_one(client, p) for p in pdfs]
    _flag_duplicate_ids(results)

    if prune:
        keep = {r.course_id for r in results if r.status == "ok"}
        removed = prune_courses(client, keep)
        print(f"Pruned {len(removed)} orphan course(s): {removed}" if removed else "Prune: nothing to remove.")

    _finish(client, results, report_path, query)


def run_single(
    pdf_path: str, *, query: str | None = None, report_path: Path = _DEFAULT_REPORT
) -> None:
    p = Path(pdf_path)
    if not p.exists():
        print(f"File not found: {p}")
        sys.exit(1)

    client = get_qdrant_client()
    ensure_collection(client)
    result = run_one(client, p)
    if result.metadata is not None:
        print("\nExtracted metadata (CLAUDE.md §5):")
        print(json.dumps(result.metadata, ensure_ascii=False, indent=2))
    _finish(client, [result], report_path, query)


def main() -> None:
    ap = argparse.ArgumentParser(description="Course PDF ingestion (batch or single)")
    ap.add_argument("--pdf", help="Ingest just this one PDF (default: all in --dir)")
    ap.add_argument("--dir", default=str(_DEFAULT_DIR), help="Folder of course PDFs")
    ap.add_argument("--query", help="Run a retrieval check after ingesting")
    ap.add_argument(
        "--prune",
        action="store_true",
        help="Also delete courses in Qdrant whose PDF is no longer on disk",
    )
    ap.add_argument("--report", default=str(_DEFAULT_REPORT), help="Where to write the JSON report")
    args = ap.parse_args()

    if args.pdf:
        run_single(args.pdf, query=args.query, report_path=Path(args.report))
    else:
        run_all(
            Path(args.dir),
            query=args.query,
            prune=args.prune,
            report_path=Path(args.report),
        )


if __name__ == "__main__":
    main()
