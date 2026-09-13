"""
PDF parsing + structured-metadata extraction (CLAUDE.md §6.1 step 2).

PyMuPDF extracts Arabic in correct logical order (pdfplumber returns reversed
presentation-form glyphs), so it is the text extractor; pdfplumber is used only
for `extract_tables()` in the best-effort price scan.

Structured fields are resolved in this order (later wins):
  1. best-effort scrape from the PDF text + tables (`title`, `language`,
     `price`/`currency`, `duration`);
  2. an optional per-course sidecar ``<stem>.meta.json`` — the source of record
     for `price` / `category` / hand-verified Arabic fields. The client will
     supply a separate per-course price file later; until then the sidecar is it.

`parse_course_pdf` returns ``(descriptive_text, metadata, warnings)`` where
`metadata` matches CLAUDE.md §5 with every key present (``None`` when unknown),
and `warnings` lists fields that stayed unresolved or fell back to a guess
(surfaced in the ingest report for review, CLAUDE.md §12.1). ``chunk_index`` is
added later, per chunk, in ``chunk.py``.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import pdfplumber
import pymupdf
from docx import Document
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table as DocxTable
from docx.text.paragraph import Paragraph as DocxParagraph

TATWEEL = "ـ"

# Arabic-Indic and Eastern-Arabic digits -> ASCII, for numeric scans.
_AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

# Currency token -> ISO 4217. Order matters: more specific tokens first.
_CURRENCY_PATTERNS: tuple[tuple[str, str], ...] = (
    ("JOD", r"دينار(?:\s*أردني)?|د\.?\s*[أا]|JOD|JD\b"),
    ("USD", r"دولار|USD|US\$|\$"),
    ("SAR", r"ريال(?:\s*سعودي)?|SAR|ر\.?\s*س"),
    ("AED", r"درهم(?:\s*إماراتي)?|AED|د\.?\s*إ"),
    ("EGP", r"جنيه(?:\s*مصري)?|EGP|ج\.?\s*م"),
    ("EUR", r"يورو|EUR|€"),
)
_AMOUNT = r"\d[\d,]*(?:\.\d+)?"

# Repeated header/footer noise on every page of this deck.
_BOILERPLATE = {
    "@CoderzTech",
    "00962778111014",
    "www.CoderZ-Tech.com",
}

# CLAUDE.md §5 — canonical payload keys (chunk_index is added per-chunk later).
META_KEYS: tuple[str, ...] = (
    "course_id",
    "title",
    "category",
    "language",
    "target_audience",
    "price",
    "currency",
    "duration",
    "source_file",
)


def _clean_line(line: str) -> str:
    # NFKC folds Arabic presentation forms (U+FExx) back to standard letters
    # (U+06xx) and expands ligatures — essential for consistent Arabic embeddings.
    line = unicodedata.normalize("NFKC", line)
    line = line.replace(TATWEEL, "")
    # Strip C0/C1 control codes — some decks render a stylised logo/title with a
    # custom-encoded font whose glyphs map to control code points (U+0011..U+001A
    # seen on PowerPlatform.pdf page 1), which otherwise leak into the title and
    # the first chunk.
    line = re.sub(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]", "", line)  # keep \t and \n
    # Collapse runs of 4+ identical chars — glyph noise left where inline Latin
    # words were dropped (e.g. repeated Arabic letters, dot leaders).
    line = re.sub(r"(.)\1{3,}", r"\1", line)
    line = re.sub(r"[ \t ]+", " ", line).strip()
    return line


def _is_noise(line: str) -> bool:
    if len(line) < 2:
        return True
    if line in _BOILERPLATE:
        return True
    # Lines that are only punctuation / separators.
    return re.fullmatch(r"[.،؛\-_=•·]+", line) is not None


def _extract_pages(pdf_path: Path) -> list[str]:
    doc = pymupdf.open(pdf_path)
    try:
        pages: list[str] = []
        for page in doc:
            lines = [
                cleaned
                for raw_line in page.get_text("text").splitlines()
                if not _is_noise(cleaned := _clean_line(raw_line))
            ]
            pages.append("\n".join(lines))
        return pages
    finally:
        doc.close()


def _detect_language(text: str) -> str:
    arabic = len(re.findall(r"[؀-ۿ]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    total = arabic + latin
    if total == 0:
        return "en"
    ar_ratio = arabic / total
    if 0.2 <= ar_ratio <= 0.8:
        return "mixed"
    return "ar" if ar_ratio > 0.8 else "en"


# Public alias — reused by query/retrieve.py for §6.2 step 1 (query language).
detect_language = _detect_language


def _load_sidecar(pdf_path: Path) -> dict[str, Any]:
    sidecar = pdf_path.with_suffix(".meta.json")
    if not sidecar.exists():
        return {}
    with sidecar.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{sidecar} must contain a JSON object")
    return {k: v for k, v in data.items() if k in META_KEYS}


def _extract_table_rows(pdf_path: Path) -> list[str]:
    """Best-effort: flatten every table row to a cleaned ' | '-joined string.

    pdfplumber can choke on some PDFs; tables are a bonus signal for the price
    scan, so failures here are swallowed and the text scan still runs.
    """
    rows: list[str] = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                for table in page.extract_tables() or []:
                    for row in table:
                        cells = [_clean_line(c) for c in row if c and c.strip()]
                        if cells:
                            rows.append(" | ".join(cells))
    except Exception:  # noqa: BLE001 - optional signal, never fatal
        pass
    return rows


def _scan_price(
    text: str, table_rows: list[str]
) -> tuple[float | None, str | None, str | None]:
    """Find a price + currency in PDF text/tables. Returns (price, ISO, evidence).

    Returns (None, None, None) when absent, and (None, None, 'ambiguous: ...')
    when several different amounts/currencies appear — we never guess.
    """
    haystack = "\n".join([text, *table_rows]).translate(_AR_DIGITS)
    found: list[tuple[float, str, str]] = []
    for iso, currency_rx in _CURRENCY_PATTERNS:
        pattern = rf"({_AMOUNT})\s*(?:{currency_rx})|(?:{currency_rx})\s*({_AMOUNT})"
        for m in re.finditer(pattern, haystack, re.IGNORECASE):
            raw = m.group(1) or m.group(2)
            try:
                value = float(raw.replace(",", ""))
            except (TypeError, ValueError):
                continue
            if not 0 < value <= 1_000_000:
                continue
            snippet = haystack[max(0, m.start() - 40) : m.end() + 40]
            found.append((value, iso, " ".join(snippet.split())))
    if not found:
        return None, None, None
    distinct = {(v, c) for v, c, _ in found}
    if len(distinct) == 1:
        return found[0]
    evidence = "; ".join(sorted({f"{v:g} {c}" for v, c, _ in found}))
    return None, None, f"ambiguous: {evidence}"


def _scan_duration(text: str) -> str | None:
    t = text.translate(_AR_DIGITS)
    ar = re.search(r"ساعة\s*تدريبية\s*(\d{2,4})|(\d{2,4})\s*ساعة\s*تدريبية", t) or re.search(
        r"(\d{2,4})\s*ساعة", t
    )
    if ar:
        n = next(g for g in ar.groups() if g)
        return f"{n} ساعة تدريبية"
    en = re.search(r"(\d{2,4})\s*\+?\s*(?:training\s+)?hours", t, re.IGNORECASE)
    return f"{en.group(1)} training hours" if en else None


def _build_metadata(
    source_path: Path,
    first_lines: list[str],
    descriptive_text: str,
    table_rows: list[str],
    *,
    source_label: str,
) -> tuple[dict[str, Any], list[str]]:
    """Shared metadata-resolution + warnings tail, used by every format parser."""
    metadata: dict[str, Any] = {key: None for key in META_KEYS}
    metadata["source_file"] = source_path.name
    metadata["course_id"] = source_path.stem.lower()
    if first_lines:
        metadata["title"] = " ".join(first_lines[:2])
    metadata["language"] = _detect_language(descriptive_text)

    price, currency, price_evidence = _scan_price(descriptive_text, table_rows)
    if price is not None:
        metadata["price"], metadata["currency"] = price, currency
    duration = _scan_duration(descriptive_text)
    if duration:
        metadata["duration"] = duration

    # Sidecar overrides win.
    sidecar = _load_sidecar(source_path)
    metadata.update(sidecar)
    for key in META_KEYS:
        metadata.setdefault(key, None)

    warnings: list[str] = []
    for field in ("price", "currency", "category", "target_audience"):
        if metadata.get(field) is None:
            warnings.append(f"{field}: not found in {source_label} and no sidecar value")
    if "title" not in sidecar and metadata.get("title"):
        warnings.append(f"title: best-effort guess from {source_label} start (no sidecar)")
    if "duration" not in sidecar and metadata.get("duration"):
        warnings.append(f"duration: best-effort scan of {source_label} text (no sidecar)")
    if price_evidence and price_evidence.startswith("ambiguous"):
        warnings.append(f"price: {price_evidence} — left null, set via sidecar")

    return metadata, warnings


def parse_course_pdf(pdf_path: str | Path) -> tuple[str, dict[str, Any], list[str]]:
    """Parse one course PDF into (descriptive_text, metadata, warnings)."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    pages = _extract_pages(pdf_path)
    descriptive_text = "\n\n".join(p for p in pages if p).strip()
    if not descriptive_text:
        raise ValueError(f"no extractable text in {pdf_path.name}")
    table_rows = _extract_table_rows(pdf_path)

    first_lines = [ln for ln in pages[0].splitlines() if ln] if pages and pages[0] else []
    metadata, warnings = _build_metadata(
        pdf_path, first_lines, descriptive_text, table_rows, source_label="PDF"
    )
    return descriptive_text, metadata, warnings


def _iter_docx_block_items(document: Document):
    """Yield paragraphs and tables in document order (python-docx has no built-in for this)."""
    for child in document.element.body.iterchildren():
        if isinstance(child, CT_P):
            yield DocxParagraph(child, document)
        elif isinstance(child, CT_Tbl):
            yield DocxTable(child, document)


def _extract_docx_blocks(docx_path: Path) -> tuple[list[str], list[str]]:
    """Return (text_lines, table_rows) in document order; tables feed both."""
    document = Document(docx_path)
    text_lines: list[str] = []
    table_rows: list[str] = []
    for block in _iter_docx_block_items(document):
        if isinstance(block, DocxParagraph):
            cleaned = _clean_line(block.text)
            if not _is_noise(cleaned):
                text_lines.append(cleaned)
        else:  # DocxTable
            for row in block.rows:
                cells = [_clean_line(c.text) for c in row.cells if c.text and c.text.strip()]
                if cells:
                    row_str = " | ".join(cells)
                    table_rows.append(row_str)
                    text_lines.append(row_str)
    return text_lines, table_rows


def parse_course_docx(docx_path: str | Path) -> tuple[str, dict[str, Any], list[str]]:
    """Parse one course Word (.docx) file into (descriptive_text, metadata, warnings)."""
    docx_path = Path(docx_path)
    if not docx_path.exists():
        raise FileNotFoundError(docx_path)

    text_lines, table_rows = _extract_docx_blocks(docx_path)
    descriptive_text = "\n".join(text_lines).strip()
    if not descriptive_text:
        raise ValueError(f"no extractable text in {docx_path.name}")

    metadata, warnings = _build_metadata(
        docx_path, text_lines, descriptive_text, table_rows, source_label="DOCX"
    )
    return descriptive_text, metadata, warnings
