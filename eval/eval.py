"""
Evaluation runner (CLAUDE.md §8).

Loads eval/questions.json — a gold set of real Arabic + English questions
(factual, semantic, and out-of-scope/refusal cases) with expected answers — and
measures, per question and in aggregate:

  - retrieval  : hit@1, hit@k, MRR  (was a relevant chunk retrieved, and how high)
  - answer     : keyword check (deterministic) and, with --judge, an LLM grade
                 against the expected answer
  - guardrail  : for refusal cases, did the assistant decline / defer to sales
                 instead of inventing an answer

The run header echoes the active embedding model, LLM, chunking and top_k, so a
saved --json result is self-describing and comparable across runs (the §4
embedding benchmark: change the model in .env, re-ingest, re-run, diff).

Run from the repo root:

    python -m eval.eval                    # retrieval + keyword correctness
    python -m eval.eval --judge            # + LLM grading vs expected_answer
    python -m eval.eval --retrieval-only   # skip generation (fast, free)
    python -m eval.eval --k 8              # override retrieval depth
    python -m eval.eval --only ar,factual  # subset by lang and/or type
    python -m eval.eval --json eval/runs/baseline.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import unicodedata
from pathlib import Path

# Windows consoles default to cp1252 — force UTF-8 so Arabic prints.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings  # noqa: E402
from query.generate import generate_answer  # noqa: E402
from query.retrieve import detect_language, retrieve  # noqa: E402

QUESTIONS_PATH = Path(__file__).resolve().parent / "questions.json"

# Arabic-Indic / Eastern-Arabic digits -> ASCII, for robust keyword matching.
_AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

# A correct refusal defers to sales / says the info isn't in the material /
# declines an out-of-scope request. Matched case-insensitively on the answer.
_REFERRAL_MARKERS = (
    "مبيعات",
    "لا تتوفر",
    "لا تتوفّر",
    "غير متوفرة",
    "غير متوفّرة",
    "غير مذكور",
    "لا يذكر",
    "لا تحتوي",
    "خارج نطاق",
    "خارج النطاق",
    "لا أستطيع",
    "لا يمكنني",
    "لا أملك",
    "لا يسعني",
    "أساعدك في",
    "المتعلقة بالدورات",
    "الدورات التدريبية",
    "sales team",
    "contact sales",
    "don't have",
    "do not have",
    "not available",
    "not mentioned",
    "does not mention",
    "no mention",
    "outside",
    "out of scope",
    "can only help",
    "can only assist",
    "i can help with",
    "i'm here to help with",
    "questions about",
    "can't help",
    "cannot help",
)

_JUDGE_SYSTEM = (
    "You are a strict grader for a bilingual (Arabic/English) customer assistant "
    "that answers only from retrieved course material. Compare the ASSISTANT "
    "ANSWER to the EXPECTED ANSWER for the given QUESTION. Judge only factual "
    "agreement, not wording, length, or language. Do NOT penalise the assistant "
    "for including extra information beyond the expected answer, as long as it is "
    "consistent with it; only penalise facts that are missing AND needed to "
    "answer the question, or facts that are wrong or invented. For a question "
    "whose expected answer is a refusal/deferral, the assistant is 'correct' if "
    "it also declines or defers instead of inventing specifics. Reply with "
    "exactly one word on the first line: correct | partial | incorrect. Then one "
    "short line explaining why."
)


def _norm(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "").translate(_AR_DIGITS).casefold()


def _hit_rank(hits, question: dict) -> int | None:
    """1-based rank of the first retrieved chunk that is a gold-relevant chunk."""
    relevant = set(question.get("relevant_chunks") or [])
    if not relevant:
        return None
    course_id = question.get("course_id")
    for rank, hit in enumerate(hits, 1):
        if hit.chunk_index in relevant and (course_id is None or hit.course_id == course_id):
            return rank
    return None


def _keyword_pass(answer: str, needles: list[str]) -> bool | None:
    if not needles:
        return None
    a = _norm(answer)
    return all(_norm(n) in a for n in needles)


def _forbidden_hit(answer: str, needles: list[str]) -> bool:
    a = _norm(answer)
    return any(_norm(n) in a for n in needles or [])


def _refusal_pass(answer: str, question: dict) -> bool:
    a = _norm(answer)
    marker = any(m in a for m in _REFERRAL_MARKERS)
    return marker and not _forbidden_hit(answer, question.get("must_not_contain") or [])


def _judge(llm, question: dict, answer: str) -> tuple[str, str]:
    from llama_index.core.llms import ChatMessage, MessageRole

    user = (
        f"QUESTION:\n{question['question']}\n\n"
        f"EXPECTED ANSWER:\n{question.get('expected_answer', '')}\n\n"
        f"ASSISTANT ANSWER:\n{answer}\n"
    )
    resp = llm.chat(
        [
            ChatMessage(role=MessageRole.SYSTEM, content=_JUDGE_SYSTEM),
            ChatMessage(role=MessageRole.USER, content=user),
        ]
    )
    raw = (resp.message.content or "").strip()
    first, _, rest = raw.partition("\n")
    grade = first.strip().lower().strip(".:")
    if grade not in ("correct", "partial", "incorrect"):
        low = raw.lower()
        grade = next((g for g in ("incorrect", "partial", "correct") if g in low), "incorrect")
    return grade, " ".join(rest.split())


# --------------------------------------------------------------------------- #
# Aggregation helpers
# --------------------------------------------------------------------------- #
def _rate(values: list[bool]) -> float | None:
    vals = [v for v in values if v is not None]
    return (sum(1 for v in vals if v) / len(vals)) if vals else None


def _pct(x: float | None) -> str:
    return "  n/a" if x is None else f"{x * 100:5.1f}%"


def _aggregate(rows: list[dict]) -> dict:
    retr = [r for r in rows if not r["expect_refusal"]]
    ref = [r for r in rows if r["expect_refusal"]]
    mrr_vals = [(1.0 / r["rank"]) if r["rank"] else 0.0 for r in retr if r["relevant_chunks"]]
    return {
        "n": len(rows),
        "hit@1": _rate([r["rank"] == 1 for r in retr if r["relevant_chunks"]]),
        "hit@k": _rate([r["rank"] is not None for r in retr if r["relevant_chunks"]]),
        "mrr": (sum(mrr_vals) / len(mrr_vals)) if mrr_vals else None,
        "keyword_pass": _rate([r["keyword_pass"] for r in rows]),
        "judge_correct": _rate(
            [r["judge"] == "correct" for r in rows if r.get("judge")]
        ),
        "judge_correct_or_partial": _rate(
            [r["judge"] in ("correct", "partial") for r in rows if r.get("judge")]
        ),
        "refusal_pass": _rate([r["refusal_pass"] for r in ref]),
    }


def _print_group(label: str, agg: dict) -> None:
    mrr = "n/a" if agg["mrr"] is None else f"{agg['mrr']:.3f}"
    print(
        f"  {label:<14} n={agg['n']:<3} "
        f"hit@1={_pct(agg['hit@1'])}  hit@k={_pct(agg['hit@k'])}  "
        f"MRR={mrr:<6}  "
        f"kw={_pct(agg['keyword_pass'])}  "
        f"judge={_pct(agg['judge_correct'])}  "
        f"refuse={_pct(agg['refusal_pass'])}"
    )


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def run(
    *,
    k: int,
    retrieval_only: bool,
    judge: bool,
    only: set[str] | None,
    json_path: Path | None,
    min_hit: float,
    min_kw: float,
) -> int:
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    if only:
        questions = [
            q for q in questions if q["lang"] in only or q["type"] in only
        ]
    if not questions:
        print("No questions match the --only filter.")
        return 1

    model = (
        settings.openai_llm_model
        if settings.llm_provider == "openai"
        else settings.anthropic_model
    )
    print("=" * 96)
    print("RAG evaluation (CLAUDE.md §8)")
    print(
        f"  embedding : {settings.embedding_provider} / {settings.embedding_model} "
        f"(dim {settings.embedding_dim})"
    )
    print(f"  llm       : {settings.llm_provider} / {model}"
          + ("" if not retrieval_only else "   [retrieval-only: not called]"))
    print(
        f"  chunking  : size {settings.chunk_size} / overlap {settings.chunk_overlap}"
        f"   |   top_k = {k}   |   questions = {len(questions)}"
    )
    print("=" * 96)
    header = f"{'id':<32} {'lang':<4} {'type':<9} {'hit@k':<9} {'kw':<5}"
    if judge:
        header += f" {'judge':<9}"
    print(header)
    print("-" * 96)

    llm = None
    if judge and not retrieval_only:
        from query.generate import _build_llm

        llm = _build_llm()

    rows: list[dict] = []
    for q in questions:
        question = q["question"]
        hits = retrieve(question, top_k=k)
        rank = _hit_rank(hits, q)

        row = {
            "id": q["id"],
            "lang": q["lang"],
            "type": q["type"],
            "expect_refusal": bool(q.get("expect_refusal")),
            "relevant_chunks": q.get("relevant_chunks") or [],
            "rank": rank,
            "retrieved": [
                {"course_id": h.course_id, "chunk_index": h.chunk_index,
                 "score": round(h.score, 4)}
                for h in hits
            ],
            "answer": None,
            "keyword_pass": None,
            "refusal_pass": None,
            "judge": None,
            "judge_reason": None,
        }

        if not retrieval_only:
            result = generate_answer(question, hits, language=detect_language(question))
            row["answer"] = result.answer
            row["keyword_pass"] = _keyword_pass(result.answer, q.get("answer_contains") or [])
            if row["expect_refusal"]:
                row["refusal_pass"] = _refusal_pass(result.answer, q)
            if llm is not None:
                row["judge"], row["judge_reason"] = _judge(llm, q, result.answer)

        rows.append(row)

        if q.get("relevant_chunks"):
            hit_cell = f"{'HIT' if rank else 'miss':<4}"
            hit_cell += f"@{rank}" if rank else "  "
        else:
            hit_cell = " -   "
        kw = {True: "ok", False: "FAIL", None: "-"}[row["keyword_pass"]]
        if row["expect_refusal"]:
            kw = {True: "ok", False: "FAIL", None: "-"}[row["refusal_pass"]]
        line = f"{q['id']:<32} {q['lang']:<4} {q['type']:<9} {hit_cell:<9} {kw:<5}"
        if judge:
            line += f" {(row['judge'] or '-'):<9}"
        print(line)

    print("-" * 96)
    overall = _aggregate(rows)
    _print_group("OVERALL", overall)
    print()
    for lang in sorted({r["lang"] for r in rows}):
        _print_group(f"lang={lang}", _aggregate([r for r in rows if r["lang"] == lang]))
    print()
    for typ in sorted({r["type"] for r in rows}):
        _print_group(f"type={typ}", _aggregate([r for r in rows if r["type"] == typ]))
    print("=" * 96)

    if judge and not retrieval_only:
        misses = [r for r in rows if r.get("judge") and r["judge"] != "correct"]
        if misses:
            print("\nJudge flagged (not 'correct'):")
            for r in misses:
                print(f"  [{r['judge']:<9}] {r['id']}: {r['judge_reason']}")

    if json_path is not None:
        payload = {
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "config": {
                "embedding_provider": settings.embedding_provider,
                "embedding_model": settings.embedding_model,
                "embedding_dim": settings.embedding_dim,
                "llm_provider": settings.llm_provider,
                "llm_model": model,
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap,
                "top_k": k,
                "retrieval_only": retrieval_only,
                "judged": bool(llm),
            },
            "overall": overall,
            "by_lang": {
                lang: _aggregate([r for r in rows if r["lang"] == lang])
                for lang in sorted({r["lang"] for r in rows})
            },
            "by_type": {
                typ: _aggregate([r for r in rows if r["type"] == typ])
                for typ in sorted({r["type"] for r in rows})
            },
            "questions": rows,
        }
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nResults written to {json_path}")

    # Optional CI gate.
    exit_code = 0
    if overall["hit@k"] is not None and overall["hit@k"] < min_hit:
        print(f"\nFAIL: hit@k {overall['hit@k']:.3f} < --min-hit {min_hit}")
        exit_code = 1
    if (
        not retrieval_only
        and overall["keyword_pass"] is not None
        and overall["keyword_pass"] < min_kw
    ):
        print(f"FAIL: keyword-pass {overall['keyword_pass']:.3f} < --min-kw {min_kw}")
        exit_code = 1
    return exit_code


def main() -> None:
    ap = argparse.ArgumentParser(description="RAG evaluation runner (CLAUDE.md §8)")
    ap.add_argument("--k", type=int, default=settings.retrieval_top_k,
                    help=f"retrieval depth for hit@k (default {settings.retrieval_top_k})")
    ap.add_argument("--retrieval-only", action="store_true",
                    help="only score retrieval; do not call the LLM")
    ap.add_argument("--judge", action="store_true",
                    help="also grade answers against expected_answer with the configured LLM")
    ap.add_argument("--only", default=None,
                    help="comma list of lang/type values to include (e.g. 'ar,factual')")
    ap.add_argument("--json", dest="json_path", default=None,
                    help="write full results to this JSON path")
    ap.add_argument("--min-hit", type=float, default=0.0,
                    help="exit non-zero if overall hit@k is below this")
    ap.add_argument("--min-kw", type=float, default=0.0,
                    help="exit non-zero if overall keyword-pass is below this")
    args = ap.parse_args()

    only = {s.strip() for s in args.only.split(",")} if args.only else None
    sys.exit(
        run(
            k=args.k,
            retrieval_only=args.retrieval_only,
            judge=args.judge,
            only=only,
            json_path=Path(args.json_path) if args.json_path else None,
            min_hit=args.min_hit,
            min_kw=args.min_kw,
        )
    )


if __name__ == "__main__":
    main()
