# DocuLoom — Campus Helpdesk RAG

**Platform name:** DocuLoom  
Hack4Haldia'26 · Team A starter evolved into a production-oriented multi-tenant RAG helpdesk.

Students stop hunting through PDFs. Ask a question → get a grounded answer with **inline `[filename, page N]` citations**.

## Core features (must-have + bonus)

| Feature | Status |
|---------|--------|
| PDF / text upload (drag-drop + picker) | ✅ real-time ingest |
| Deterministic recursive chunking | ✅ reproducible boundaries |
| Hybrid dense + sparse retrieval | ✅ Qdrant + RRF fusion |
| Passage-constrained generation | ✅ Groq LLM |
| Inline + structured citations | ✅ |
| Pre-ingested corpus (`./corpus/`) | ✅ startup auto-ingest |
| Multi-turn memory (10 turns / 4k tokens) | ✅ |
| “I don’t know” when evidence missing | ✅ |
| Targeted OCR (zero-text pages only) | ✅ OCR.space / tesseract |
| Dark mode + theme persistence | ✅ |
| Multi-tenant (Neon Postgres + Qdrant filters) | ✅ |
| Auth (JWT register / login) | ✅ |
| Evaluation endpoint (p50 / p95 latency) | ✅ |
| Payload indexes for low latency | ✅ |

## Stack

- **Backend:** Flask + SQLAlchemy  
- **DB:** Neon Postgres (users, documents, conversations, chunk meta)  
- **Vector DB:** Qdrant (hybrid dense MiniLM + BM25 sparse, RRF)  
- **LLM:** Groq (`llama-3.3-70b-versatile` or any Groq chat model)  
- **Embeddings:** FastEmbed / sentence-transformers via Qdrant Document API  
- **OCR:** OCR.space (free tier) + local Tesseract fallback  
- **Frontend:** Single-page HTML/CSS/JS — neobrutalism  

## Quick start

```bash
cp .env.example .env
# fill GROQ_API_KEY, DATABASE_URL (Neon), QDRANT_URL (or leave for :memory: demo)

pip install -r requirements.txt
# optional system deps for OCR: poppler-utils, tesseract-ocr

# Place campus PDFs in ./corpus/ for auto-ingest
python app.py
# → http://localhost:5000
```

### API

| Method | Path | Body / notes |
|--------|------|----------------|
| GET | `/health` | liveness |
| POST | `/api/auth/register` | `{email, password, display_name?}` |
| POST | `/api/auth/login` | `{email, password}` |
| POST | `/api/ingest` | `multipart/form-data` field `file` |
| GET | `/api/docs` | list documents for tenant |
| DELETE | `/api/docs/<id>` | remove doc + vectors |
| POST | `/api/ask` | `{question, conversation_id?}` → `{answer, sources[], latency_ms}` |
| GET | `/api/conversations` | (auth) |
| POST | `/api/eval` | `{queries:[{question, expected_keywords?}]}` |

Headers: `Authorization: Bearer <jwt>`, optional `X-Tenant-Id`.

## Deploy

**Render** (recommended):

- Web Service, connect repo  
- Start: `gunicorn -b 0.0.0.0:$PORT app:app`  
- Env: copy from `.env.example`  
- Add a Qdrant Cloud free cluster + Neon free project  

**Railway / Fly.io / Vercel (Python)** also work. Put the live URL at the top of your PR.

## Design notes

- **Latency:** Qdrant payload indexes on `tenant_id` / `document_id`; hybrid prefetch limited; optional rerank truncated by `RERANK_TOP_K`; Groq is extremely fast on LPU.  
- **Multi-tenant:** every document, chunk, and conversation is scoped by `tenant_id`; vector queries always filter.  
- **Grounding:** system prompt forbids invention; empty retrieval → fixed “I don’t know…” string.  
- **OCR strategy:** pypdf first; only blank pages go through OCR → saves quota and time.  

## Team / submit

1. `git checkout -b team-a-solution`  
2. Commit often, push, open PR against `main`  
3. PR body: **live URL**, what you built, tech, members  
4. Also submit on MLH / OrganizerHQ  

Built for Hack4Haldia'26. AI tools allowed — understand every line you ship.
