"""
API smoke test (Phase 6) — exercises POST /chat and GET /health in-process via
Starlette's TestClient (no live server needed). Run from the repo root:

    python scripts/check_api.py

Needs a populated Qdrant collection and a working LLM key for the two cases
that actually generate an answer; the validation cases run offline.
Prints one line per case; exits non-zero if any case fails.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Windows consoles default to cp1252 — force UTF-8 so Arabic prints.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from starlette.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402

client = TestClient(app, raise_server_exceptions=False)

_passed = 0
_failed = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global _passed, _failed
    mark = "PASS" if ok else "FAIL"
    if ok:
        _passed += 1
    else:
        _failed += 1
    print(f"[{mark}] {name}" + (f"  — {detail}" if detail else ""))


# --- health -------------------------------------------------------------------
r = client.get("/health")
check("GET /health -> 200", r.status_code == 200, str(r.status_code))
check(
    "health reports provider/collection",
    r.status_code == 200
    and {"llm_provider", "llm_model", "embedding_model", "collection"} <= r.json().keys(),
    r.text[:160],
)

# --- validation (offline) ---------------------------------------------------
r = client.post("/chat", json={"question": "   "})
check("empty question -> 422", r.status_code == 422, str(r.status_code))

r = client.post("/chat", json={})
check("missing question -> 422", r.status_code == 422, str(r.status_code))

r = client.post(
    "/chat", json={"question": "x", "filters": {"min_price": 900, "max_price": 100}}
)
check("min_price > max_price -> 422", r.status_code == 422, str(r.status_code))

r = client.post("/chat", json={"question": "x", "language": "fr"})
check("bad language enum -> 422", r.status_code == 422, str(r.status_code))

# --- CORS preflight --------------------------------------------------------
r = client.options(
    "/chat",
    headers={
        "Origin": "http://localhost:5173",
        "Access-Control-Request-Method": "POST",
    },
)
check(
    "CORS preflight exposes allow-origin",
    "access-control-allow-origin" in {k.lower() for k in r.headers},
    r.headers.get("access-control-allow-origin", "<none>"),
)

# --- real answers (need Qdrant + LLM) ------------------------------------
r = client.post("/chat", json={"question": "ما هي مواضيع دورة DataHub؟"})
body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
check(
    "AR question -> 200 grounded answer",
    r.status_code == 200 and body.get("used_context") is True and bool(body.get("answer")),
    f"{r.status_code} used_context={body.get('used_context')} sources={body.get('sources')}",
)

r = client.post(
    "/chat", json={"question": "How much does the bootcamp cost?", "language": "en"}
)
body = r.json() if r.status_code == 200 else {}
check(
    "EN question -> 200, language honoured",
    r.status_code == 200 and body.get("language") == "en" and bool(body.get("answer")),
    f"{r.status_code} language={body.get('language')} answer={str(body.get('answer'))[:80]}",
)

r = client.post(
    "/chat", json={"question": "test", "filters": {"category": "NO_SUCH_CATEGORY"}}
)
body = r.json() if r.status_code == 200 else {}
check(
    "no-hit filter -> used_context false",
    r.status_code == 200 and body.get("used_context") is False,
    f"{r.status_code} used_context={body.get('used_context')}",
)

print(f"\n{_passed} passed, {_failed} failed")
sys.exit(1 if _failed else 0)
