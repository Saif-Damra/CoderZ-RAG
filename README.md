# RAG Courses — bilingual (AR/EN) course Q&A

A retrieval-augmented system that answers a training company's customers about
its courses, grounded **only** in the course PDFs. It replies in the language of
the question (Arabic ↔ Arabic, English ↔ English) and never invents a price or
detail. Full specification: [`CLAUDE.md`](CLAUDE.md).

**Stack:** Python 3.11+ · LlamaIndex · Qdrant Cloud (hybrid search) · OpenAI
embeddings (not finalised — chosen after the Phase 5 benchmark) · Claude / OpenAI
for generation (swappable) · FastAPI.

## Build status

| Phase | Scope | Status |
|---|---|---|
| 0 | Repo scaffold, `config.py`, `.env.example`, deps, Qdrant Cloud connection | done |
| 1 | One course end to end (parse → chunk → embed → upsert → retrieve) | done |
| 2 | Ingest all PDFs in `data/pdfs/`, structured-field extraction, per-run report | done |
| 3 | Retrieval: hybrid dense + BM25 search with metadata filters | done |
| 4 | Generation: LLM + guardrail prompt + reply in question language | done |
| 5 | `eval.py` + question set; tune chunking / embedding model | done |
| 6 | FastAPI `/chat` endpoint | done |
| 7 | Simple embeddable chat widget | **in review** |

One phase at a time, with a review after each (see `CLAUDE.md` §10).

## Setup

### 1. Environment

```bash
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Secrets

```bash
cp .env.example .env      # Windows: copy .env.example .env
```

Then edit `.env`. For Phase 0 you only need `QDRANT_URL` and `QDRANT_API_KEY`
(plus `OPENAI_API_KEY` when you reach embeddings). `.env` is git-ignored — never
commit it.

Check the config parses:

```bash
python config.py
```

### 3. Qdrant Cloud

1. Sign up at <https://cloud.qdrant.io> and create a **Free** cluster (1 GB is
   plenty for 30-50 courses).
2. Copy the **cluster URL** and generate an **API key**; put both in `.env`.
3. Verify connectivity:

   ```bash
   python scripts/check_qdrant.py
   ```

   Expected output ends with `QDRANT OK`.

## Ingestion

```bash
python -m ingest.run_ingest                      # ingest every PDF in data/pdfs/
python -m ingest.run_ingest --pdf data/pdfs/DataHub.pdf
python -m ingest.run_ingest --prune              # + drop courses whose PDF is gone
python -m ingest.run_ingest --query "سؤال تجريبي"  # + dense retrieval check
python scripts/check_retrieval.py "your question" --k 5
```

Each run writes `data/ingest_report.json` (git-ignored) — one row per course with
resolved metadata plus a `needs_attention` list of unresolved fields (CLAUDE.md
§12.1). Structured fields come from a best-effort scan of the PDF text/tables,
overridden by a per-course sidecar `data/pdfs/<stem>.meta.json` (the price /
category source of record until the client supplies a dedicated price file).
Re-running is idempotent — points are keyed by `course_id` + `chunk_index`.

Points carry a named `dense` (OpenAI) vector **and** a `sparse` BM25 vector;
`check_retrieval.py` / `query.retrieve.retrieve()` fuse the two with RRF and
accept metadata filters (`--category`, `--language`, `--course`, `--min-price`,
`--max-price`). Changing the embedding model or vector schema recreates the
collection — re-run ingestion afterwards.

## API (Phase 6)

```bash
uvicorn api.main:app --reload            # dev server on http://localhost:8000
python -m api.main                       # same, no reload
python scripts/check_api.py              # in-process smoke test (no server)
```

`POST /chat` — body: `question` (required), optional `language` (`"ar"`/`"en"`,
auto-detected otherwise), optional `filters`
(`course_id`, `category`, `language`, `min_price`, `max_price`). Returns
`answer`, `sources`, `used_context`, `language`, `interaction_id` (id of the
logged interaction — see Feedback below; `null` if logging failed, which never
blocks the answer). Bad input → `422`; a missing key or unreachable dependency
→ `503`; a downstream failure → `502`, always with a short safe message (no
internals). `GET /health` reports the active provider / model / collection.
Interactive docs at `/docs`.

Browser origins allowed to call the API come from `CORS_ALLOW_ORIGINS` (comma
separated; `*` for local dev — set real origins before deploying). Send request
bodies as UTF-8 JSON.

## Feedback logging (for team testing)

Every `POST /chat` call is logged to a SQLite DB at `FEEDBACK_DB_PATH`
(default `data/feedback.db`, git-ignored) — question, detected language,
answer, sources, `used_context`, and response time. `POST /feedback` — body:
`interaction_id` (from the `/chat` response), `rating` (`"up"`/`"down"`),
optional `comment` — attaches a rating to that row. A logging or feedback
failure never breaks the chat response itself.

Review what's been collected:

```bash
python scripts/export_feedback.py                  # console table, newest first
python scripts/export_feedback.py --only-feedback   # just rated rows
python scripts/export_feedback.py --csv out.csv     # full untruncated dump
```

The chat widget (below) shows 👍/👎 under each answer and posts to
`/feedback` automatically; 👎 reveals an optional one-line comment box.

## Chat widget (Phase 7)

`web/widget.html` is one self-contained file (inline CSS + vanilla JS, no build,
no external assets): a floating launcher button that opens a chat panel over any
page. It auto-switches RTL/LTR per message and shows the source course under each
answer.

Local test — with the API running:

```bash
python -m http.server -d web 5500
# open http://localhost:5500/widget.html?api=http://localhost:8000
```

Embed on a site — host the file and add an iframe (the page behind stays
clickable):

```html
<iframe src="https://YOUR-HOST/widget.html"
        style="position:fixed;inset:0;border:0;width:100%;height:100%;
               pointer-events:none;z-index:2147483000"
        title="Courses assistant"></iframe>
```

Point it at the API via (first match wins): `window.CODERZ_CHAT_API`, a `?api=`
query param, or the `API_BASE` default in the file. Then add the host site's
origin to `CORS_ALLOW_ORIGINS`. Rebrand by editing the four `--cz-*` custom
properties at the top of the `<style>` block.

When the API is running (locally or via Docker, below), it also serves the
widget itself at `/widget/widget.html` — nothing extra to host for internal
team testing.

## Deployment (Docker, for team testing on a VPS)

One container runs the API and serves the widget — no separate web server
needed. `data/` is mounted as a volume so `data/feedback.db` survives rebuilds.

```bash
git clone <your-repo-url> && cd rag-courses   # or: git pull, if already cloned
cp .env.example .env                          # then fill in real secrets
docker compose up -d --build
```

Qdrant Cloud already holds the ingested courses, so the VPS doesn't need
`data/pdfs/`. Leave `CORS_ALLOW_ORIGINS=*` for internal testing (the widget is
served same-origin from the same container anyway). Set `ADMIN_TOKEN` in
`.env` to a real secret before using `/review` — an empty value keeps it
locked.

The compose file carries Traefik labels routing `rag.saifdamra.com` to this
container (HTTPS via the existing Traefik + Let's Encrypt setup on the VPS —
no port is published directly). If you're deploying this to a different host
or domain, edit the `labels:` block in `docker-compose.yml` to match, or drop
it and publish `8000:8000` directly instead.

Your team then uses:

- Widget: `https://rag.saifdamra.com/widget/widget.html`
- Feedback review: `https://rag.saifdamra.com/review`
- API docs: `https://rag.saifdamra.com/docs`

Useful commands on the VPS:

```bash
docker compose logs -f api                              # tail logs
docker compose restart                                  # restart after an env change
docker compose exec api python scripts/export_feedback.py --only-feedback
```

## Repository layout

```
config.py              pydantic-settings: keys, model names, collection name
requirements.txt
.env.example            template for .env (no secrets)
scripts/check_qdrant.py Qdrant connectivity check
scripts/check_retrieval.py dense retrieval check
data/pdfs/              course PDFs (git-ignored) + <stem>.meta.json sidecars (tracked)
data/feedback.db        SQLite interaction log + team feedback (git-ignored, runtime)
ingest/                 parse.py · chunk.py · embed_upsert.py · run_ingest.py
query/                  retrieve.py · generate.py
api/main.py             FastAPI /chat, /feedback (Phase 6)
api/storage.py          SQLite persistence for /chat and /feedback
web/widget.html         embeddable chat widget incl. 👍/👎 feedback (Phase 7)
scripts/export_feedback.py review logged Q&A + feedback
eval/                   questions.json · eval.py (Phase 5)
Dockerfile, docker-compose.yml   VPS deployment
```

`scripts/` and the package `__init__.py` files are small additions to the layout
in `CLAUDE.md` §9 (a place for the connection check; clean `python -m` imports).
