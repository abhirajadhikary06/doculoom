"""
DocuLoom — Campus Helpdesk RAG
Flask backend + neobrutalism frontend.
"""
from __future__ import annotations
import logging
import os
import time
import uuid
from pathlib import Path

from flask import (
    Flask,
    request,
    jsonify,
    render_template,
    send_from_directory,
    g,
)
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.utils import secure_filename

from config import Config
from models import (
    db,
    Tenant,
    User,
    Document,
    Conversation,
    Message,
    EvalRun,
    CreditLog,
    MemoryGraph,
)
from services.auth import (
    hash_password,
    verify_password,
    create_token,
    require_auth,
    ensure_default_tenant,
)
from services.vectorstore import VectorStore
from services.ingest import ingest_file, ingest_corpus
from services.llm import LLMService
from services.reranker import rerank

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("doculoom")

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config.from_object(Config)
app.config["MAX_CONTENT_LENGTH"] = Config.MAX_CONTENT_LENGTH
CORS(app, supports_credentials=True)

db.init_app(app)

# Rate limiter — fall back to in-memory if Redis is configured but unavailable
_limiter_storage = Config.REDIS_URL or "memory://"
try:
    limiter = Limiter(
        key_func=get_remote_address,
        app=app,
        default_limits=[Config.RATELIMIT_DEFAULT],
        storage_uri=_limiter_storage,
    )
except Exception as e:
    logger.warning(
        "Rate limiter storage '%s' unavailable, falling back to memory: %s",
        _limiter_storage,
        e,
    )
    limiter = Limiter(
        key_func=get_remote_address,
        app=app,
        default_limits=[Config.RATELIMIT_DEFAULT],
        storage_uri="memory://",
    )

# Singletons (lazy)
_vs: VectorStore | None = None
_llm: LLMService | None = None


def get_vs() -> VectorStore:
    global _vs
    if _vs is None:
        _vs = VectorStore()
    return _vs


def get_llm() -> LLMService:
    global _llm
    if _llm is None:
        _llm = LLMService()
    return _llm


ALLOWED_EXT = {".pdf", ".txt", ".md"}


# ─── Credit helpers ──────────────────────────────────────────
def _check_credits(user: User, cost: int) -> tuple[bool, str]:
    if user.credits < cost:
        return False, f"Insufficient credits. You have {user.credits}, this action costs {cost}."
    return True, ""


def _deduct_credits(user: User, cost: int, reason: str, meta: dict | None = None) -> int:
    user.credits = max(0, user.credits - cost)
    db.session.add(CreditLog(
        user_id=user.id,
        tenant_id=user.tenant_id,
        amount=-cost,
        balance_after=user.credits,
        reason=reason,
        meta=meta or {},
    ))
    db.session.commit()
    return user.credits


def _credit_info(user: User) -> dict:
    return {"credits": user.credits, "max_credits": user.max_credits}


# ─── Bootstrap ───────────────────────────────────────────────
@app.before_request
def _ensure_tenant():
    # Soft default; real auth overrides in require_auth
    g.tenant_id = (
        request.headers.get("X-Tenant-Id")
        or request.args.get("tenant")
        or Config.DEFAULT_TENANT
    )


def _migrate_sqlite_columns():
    """Ensure SQLite documents table has all v2 columns if created in older migrations."""
    from sqlalchemy import inspect, text
    try:
        inspector = inspect(db.engine)
        if "documents" in inspector.get_table_names():
            existing_cols = {col["name"] for col in inspector.get_columns("documents")}
            cols_to_add = [
                ("pub_date", "DATE"),
                ("valid_from", "DATE"),
                ("valid_until", "DATE"),
                ("version", "INTEGER DEFAULT 1"),
                ("superseded_by", "VARCHAR(36)"),
                ("trust_score", "FLOAT DEFAULT 0.5"),
                ("trust_factors", "TEXT DEFAULT '{}'"),
                ("ingestion_quality_id", "VARCHAR(36)"),
                ("impact_analysis", "TEXT DEFAULT '{}'"),
            ]
            with db.engine.begin() as conn:
                for col_name, col_type in cols_to_add:
                    if col_name not in existing_cols:
                        try:
                            conn.execute(text(f"ALTER TABLE documents ADD COLUMN {col_name} {col_type}"))
                        except Exception as e:
                            logger.debug("Column %s add skipped: %s", col_name, e)
    except Exception as e:
        logger.debug("Column migration check: %s", e)


def init_db():
    Config.ensure_dirs()
    with app.app_context():
        db.create_all()
        _migrate_sqlite_columns()
        ensure_default_tenant()
        # Optional auto-ingest corpus
        try:
            vs = get_vs()
            docs = ingest_corpus(Config.DEFAULT_TENANT, vector_store=vs)
            logger.info("Corpus ingest: %d docs ready", len(docs))
        except Exception as e:
            logger.warning("Corpus auto-ingest skipped: %s", e)


# ─── Health ──────────────────────────────────────────────────
@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "service": "doculoom",
            "version": "1.0.0",
        }
    )


# ─── Auth ────────────────────────────────────────────────────
@app.post("/api/auth/register")
def register():
    data = request.get_json(force=True, silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    name = data.get("display_name") or email.split("@")[0]
    tenant_id = data.get("tenant_id") or Config.DEFAULT_TENANT

    if not email or len(password) < 6:
        return jsonify({"error": "email and password (min 6) required"}), 400
    if User.query.filter_by(email=email).first():
        return jsonify({"error": "email already registered"}), 409

    ensure_default_tenant()
    if not Tenant.query.get(tenant_id):
        db.session.add(Tenant(id=tenant_id, name=tenant_id.title()))
        db.session.commit()

    user = User(
        id=str(uuid.uuid4()),
        tenant_id=tenant_id,
        email=email,
        password_hash=hash_password(password),
        display_name=name,
        role="admin" if data.get("admin") else "user",
        credits=Config.MAX_CREDITS,
        max_credits=Config.MAX_CREDITS,
    )
    db.session.add(user)
    db.session.commit()
    # Log initial credit grant
    db.session.add(CreditLog(
        user_id=user.id,
        tenant_id=tenant_id,
        amount=Config.MAX_CREDITS,
        balance_after=Config.MAX_CREDITS,
        reason="signup",
    ))
    db.session.commit()
    token = create_token(user)
    return jsonify(
        {
            "token": token,
            "user": {
                "id": user.id,
                "email": user.email,
                "display_name": user.display_name,
                "tenant_id": user.tenant_id,
                "role": user.role,
                "credits": user.credits,
                "max_credits": user.max_credits,
            },
        }
    ), 201


@app.post("/api/auth/login")
def login():
    data = request.get_json(force=True, silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    user = User.query.filter_by(email=email).first()
    if not user or not verify_password(password, user.password_hash):
        return jsonify({"error": "invalid credentials"}), 401
    from models import utcnow

    user.last_login = utcnow()
    db.session.commit()
    token = create_token(user)
    return jsonify(
        {
            "token": token,
            "user": {
                "id": user.id,
                "email": user.email,
                "display_name": user.display_name,
                "tenant_id": user.tenant_id,
                "role": user.role,
                "credits": user.credits,
                "max_credits": user.max_credits,
            },
        }
    )


@app.get("/api/auth/me")
@require_auth()
def me():
    u = g.user
    return jsonify(
        {
            "id": u.id,
            "email": u.email,
            "display_name": u.display_name,
            "tenant_id": u.tenant_id,
            "role": u.role,
            "credits": u.credits,
            "max_credits": u.max_credits,
        }
    )


# ─── Ingest ──────────────────────────────────────────────────
@app.post("/api/ingest")
@require_auth(optional=True)
@limiter.limit("10 per minute")
def api_ingest():
    tenant_id = g.tenant_id
    user = g.user
    # Credit check if authenticated
    if user:
        ok, msg = _check_credits(user, Config.CREDITS_PER_INGEST)
        if not ok:
            return jsonify({"error": msg, "credits": user.credits}), 402

    if "file" not in request.files:
        return jsonify({"error": "file required"}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "empty filename"}), 400
    ext = Path(f.filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        return jsonify({"error": f"unsupported type {ext}"}), 400

    Config.ensure_dirs()
    safe = secure_filename(f.filename)
    dest = Config.UPLOAD_DIR / f"{uuid.uuid4().hex}_{safe}"
    f.save(dest)

    doc = ingest_file(
        dest,
        tenant_id,
        original_name=f.filename,
        source="upload",
        vector_store=get_vs(),
    )
    # Deduct credits for successful ingest
    if user and doc.status == "ready":
        balance = _deduct_credits(user, Config.CREDITS_PER_INGEST, "ingest", {"doc_id": doc.id})
    else:
        balance = user.credits if user else 100

    return jsonify(
        {
            "id": doc.id,
            "filename": doc.original_name or doc.filename,
            "status": doc.status,
            "page_count": doc.page_count,
            "chunk_count": doc.chunk_count,
            "trust_score": doc.trust_score,
            "error": doc.error_message,
            "source": doc.source,
            "credits": balance,
            "max_credits": user.max_credits if user else 100,
        }
    ), (201 if doc.status == "ready" else 422)


@app.get("/api/docs")
@require_auth(optional=True)
def list_docs():
    tenant_id = g.tenant_id
    q = Document.query.filter_by(tenant_id=tenant_id).order_by(Document.created_at.desc())
    docs = [
        {
            "id": d.id,
            "filename": d.original_name or d.filename,
            "status": d.status,
            "page_count": d.page_count,
            "chunk_count": d.chunk_count,
            "trust_score": d.trust_score,
            "source": d.source,
            "created_at": d.created_at.isoformat() if hasattr(d.created_at, "isoformat") else (str(d.created_at) if d.created_at else None),
            "size_bytes": d.size_bytes,
        }
        for d in q.limit(200).all()
    ]
    return jsonify({"documents": docs, "count": len(docs)})


@app.delete("/api/docs/<doc_id>")
@require_auth(optional=True)
def delete_doc(doc_id):
    tenant_id = g.tenant_id
    doc = Document.query.filter_by(id=doc_id, tenant_id=tenant_id).first()
    if not doc:
        return jsonify({"error": "not found"}), 404
    try:
        get_vs().delete_document(doc_id, tenant_id)
    except Exception as e:
        logger.warning("Qdrant delete: %s", e)
    try:
        from models import (
            ChunkMeta, DocumentVersion, DocumentSupersession,
            DocumentConflict, IngestionQuality, KnowledgeExpiration
        )
        ChunkMeta.query.filter_by(document_id=doc_id).delete()
        DocumentVersion.query.filter_by(document_id=doc_id).delete()
        DocumentSupersession.query.filter(
            (DocumentSupersession.old_document_id == doc_id) |
            (DocumentSupersession.new_document_id == doc_id)
        ).delete()
        DocumentConflict.query.filter(
            (DocumentConflict.doc_a_id == doc_id) |
            (DocumentConflict.doc_b_id == doc_id)
        ).delete()
        KnowledgeExpiration.query.filter_by(document_id=doc_id).delete()
        IngestionQuality.query.filter_by(document_id=doc_id).delete()
    except Exception as e:
        logger.warning("Error cleaning child records on doc delete: %s", e)
    db.session.delete(doc)
    db.session.commit()
    return jsonify({"ok": True})


# ─── Ask (RAG) ───────────────────────────────────────────────
@app.post("/api/ask")
@require_auth(optional=True)
def api_ask():
    t0 = time.perf_counter()
    data = request.get_json(force=True, silent=True) or {}
    question = (data.get("question") or data.get("query") or "").strip()
    if not question:
        return jsonify({"error": "question required"}), 400

    tenant_id = g.tenant_id
    conversation_id = data.get("conversation_id")
    history = data.get("history") or []  # optional client-side history

    # Persist conversation if user is logged in
    conv = None
    user = getattr(g, "user", None)
    if conversation_id:
        conv = Conversation.query.filter_by(
            id=conversation_id, tenant_id=tenant_id
        ).first()
    if not conv and user:
        conv = Conversation(
            id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            user_id=user.id,
            title=question[:80],
        )
        db.session.add(conv)
        db.session.commit()

    # Load server-side history if conversation exists
    if conv and not history:
        msgs = (
            Message.query.filter_by(conversation_id=conv.id)
            .order_by(Message.created_at.desc())
            .limit(Config.HISTORY_TURNS * 2)
            .all()
        )
        history = [
            {"role": m.role, "content": m.content}
            for m in reversed(msgs)
            if m.role in ("user", "assistant")
        ]

    # Credit check
    ok, msg = _check_credits(user, Config.CREDITS_PER_ASK)
    if not ok:
        return jsonify({"error": msg, "credits": user.credits}), 402

    # Query rewriting — expand pronouns/references into standalone query
    llm = get_llm()
    search_query = llm.rewrite_query(question, history)

    # Hybrid retrieval
    vs = get_vs()
    passages = vs.hybrid_search(
        search_query, tenant_id=tenant_id, top_k=Config.TOP_K
    )

    # Cross-encoder rerank with timeout fallback
    passages = rerank(
        question, passages,
        top_k=Config.RERANK_TOP_K,
        timeout_s=Config.RERANK_TIMEOUT_S,
    )

    answer, sources, usage = llm.generate_answer(question, passages, history=history)

    latency_ms = (time.perf_counter() - t0) * 1000

    # Persist messages
    if conv:
        db.session.add(
            Message(
                conversation_id=conv.id,
                role="user",
                content=question,
            )
        )
        db.session.add(
            Message(
                conversation_id=conv.id,
                role="assistant",
                content=answer,
                sources=sources,
                latency_ms=latency_ms,
                token_usage=usage,
            )
        )
        db.session.commit()

    # Memory graph extraction — entities/relationships from Q&A
    try:
        graph_data = llm.extract_graph(f"Q: {question}\nA: {answer}")
        if graph_data.get("nodes"):
            db.session.add(MemoryGraph(
                tenant_id=tenant_id,
                user_id=user.id if user else None,
                conversation_id=conv.id if conv else None,
                nodes=graph_data["nodes"],
                edges=graph_data["edges"],
            ))
            db.session.commit()
    except Exception as e:
        logger.debug("Memory graph extraction skipped: %s", e)

    # Deduct credits for successful ask
    if user:
        balance = _deduct_credits(user, Config.CREDITS_PER_ASK, "ask", {
            "conversation_id": conv.id if conv else None,
        })
    else:
        balance = 50

    return jsonify(
        {
            "answer": answer,
            "sources": sources,
            "conversation_id": conv.id if conv else None,
            "latency_ms": round(latency_ms, 1),
            "usage": usage,
            "retrieved": len(passages),
            "credits": balance,
            "max_credits": user.max_credits if user else 50,
            "search_query": search_query if search_query != question else None,
        }
    )


# ─── Conversations ───────────────────────────────────────────
@app.get("/api/conversations")
@require_auth()
def list_conversations():
    rows = (
        Conversation.query.filter_by(user_id=g.user.id, tenant_id=g.tenant_id)
        .order_by(Conversation.updated_at.desc())
        .limit(50)
        .all()
    )
    return jsonify(
        {
            "conversations": [
                {
                    "id": c.id,
                    "title": c.title,
                    "updated_at": c.updated_at.isoformat() if c.updated_at else None,
                }
                for c in rows
            ]
        }
    )


@app.get("/api/conversations/<cid>")
@require_auth()
def get_conversation(cid):
    conv = Conversation.query.filter_by(
        id=cid, user_id=g.user.id, tenant_id=g.tenant_id
    ).first()
    if not conv:
        return jsonify({"error": "not found"}), 404
    msgs = [
        {
            "id": m.id,
            "role": m.role,
            "content": m.content,
            "sources": m.sources or [],
            "latency_ms": m.latency_ms,
            "created_at": m.created_at.isoformat() if m.created_at else None,
        }
        for m in conv.messages
    ]
    return jsonify({"id": conv.id, "title": conv.title, "messages": msgs})


# ─── Evaluation ──────────────────────────────────────────────
@app.post("/api/eval")
@require_auth(optional=True)
def run_eval():
    """
    Body: { "queries": [ {"question": "...", "expected_keywords": ["..."] }, ... ] }
    Returns p95 latency + optional keyword hit rate.
    """
    data = request.get_json(force=True, silent=True) or {}
    queries = data.get("queries") or [
        {"question": "What is the attendance policy?"},
        {"question": "How do I apply for leave?"},
        {"question": "Exam form submission deadline"},
    ]
    tenant_id = g.tenant_id
    latencies = []
    hits = 0
    total = 0

    vs = get_vs()
    llm = get_llm()

    for q in queries[:20]:
        question = q.get("question") or ""
        if not question:
            continue
        t0 = time.perf_counter()
        passages = vs.hybrid_search(question, tenant_id, top_k=Config.TOP_K)
        answer, sources, _ = llm.generate_answer(question, passages[: Config.RERANK_TOP_K])
        lat = (time.perf_counter() - t0) * 1000
        latencies.append(lat)
        total += 1
        kws = q.get("expected_keywords") or []
        if kws:
            low = answer.lower()
            if any(k.lower() in low for k in kws):
                hits += 1

    latencies.sort()
    p95 = latencies[int(len(latencies) * 0.95)] if latencies else 0
    metrics = {
        "n": total,
        "p50_ms": latencies[len(latencies) // 2] if latencies else 0,
        "p95_ms": round(p95, 1),
        "mean_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0,
        "keyword_hit_rate": round(hits / total, 3) if total and any(
            q.get("expected_keywords") for q in queries
        ) else None,
    }
    run = EvalRun(tenant_id=tenant_id, metrics=metrics)
    db.session.add(run)
    db.session.commit()
    return jsonify({"metrics": metrics, "run_id": run.id})


# ─── Frontend ────────────────────────────────────────────────
@app.get("/")
def landing():
    """Landing page — entry point for new visitors."""
    return render_template("landing.html")


@app.get("/app")
def index():
    """Chat application — the main DocuLoom interface."""
    return render_template("index.html")


@app.get("/static/<path:path>")
def static_files(path):
    return send_from_directory("static", path)


# ─── Main ────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE INTELLIGENCE ENGINE — Advanced Feature Endpoints
# ═══════════════════════════════════════════════════════════════════════════════

# ── A: Temporal Knowledge Engine ─────────────────────────────────────────────

@app.get("/api/temporal/timeline")
@require_auth()
def temporal_timeline():
    """Get version timeline for documents matching a name pattern."""
    pattern = request.args.get("name", "").strip()
    if not pattern:
        return jsonify({"error": "name parameter required"}), 400
    from services.temporal import get_document_timeline
    timeline = get_document_timeline(pattern, g.tenant_id)
    return jsonify({"timeline": timeline, "count": len(timeline)})


@app.get("/api/temporal/supersessions")
@require_auth()
def temporal_supersessions():
    """Get supersession chain for a document."""
    doc_id = request.args.get("doc_id", "").strip()
    if not doc_id:
        return jsonify({"error": "doc_id required"}), 400
    from services.temporal import get_supersession_chain
    chain = get_supersession_chain(doc_id, g.tenant_id)
    return jsonify({"chain": chain, "count": len(chain)})


@app.get("/api/temporal/versions/<doc_id>")
@require_auth()
def temporal_versions(doc_id):
    """Get version info for a specific document."""
    from sqlalchemy import text as sql_text
    rows = db.session.execute(
        sql_text(
            "SELECT id, version_number, content_hash, pub_date, valid_from, "
            "valid_until, created_at FROM document_versions "
            "WHERE document_id = :doc AND tenant_id = :tenant "
            "ORDER BY version_number DESC"
        ),
        {"doc": doc_id, "tenant": g.tenant_id},
    ).fetchall()
    versions = [
        {
            "id": r[0], "version": r[1], "content_hash": r[2][:12] + "...",
            "pub_date": str(r[3]) if r[3] else None,
            "valid_from": str(r[4]) if r[4] else None,
            "valid_until": str(r[5]) if r[5] else None,
            "created_at": r[6].isoformat() if r[6] else None,
        }
        for r in rows
    ]
    return jsonify({"document_id": doc_id, "versions": versions})


# ── B: Evidence Verification ──────────────────────────────────────────────────

@app.post("/api/verify")
@require_auth()
def verify_answer():
    """Verify claims in an answer against supporting passages."""
    data = request.get_json(force=True, silent=True) or {}
    answer = (data.get("answer") or "").strip()
    passages = data.get("passages") or []
    if not answer:
        return jsonify({"error": "answer required"}), 400
    from services.evidence import verify_answer as ev_verify
    result = ev_verify(answer, passages, llm_client=get_llm())
    return jsonify(result)


# ── C: Adaptive Retrieval Planner ─────────────────────────────────────────────

@app.post("/api/retrieval/classify")
@require_auth()
def classify_query_endpoint():
    """Classify a query intent and recommend retrieval strategy."""
    data = request.get_json(force=True, silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "question required"}), 400
    from services.retrieval_planner import classify_query
    from services.difficulty import estimate_difficulty
    clf = classify_query(question)
    diff = estimate_difficulty(question, clf)
    return jsonify({"classification": clf, "difficulty": diff})


@app.post("/api/retrieval/evaluate")
@require_auth()
def evaluate_retrieval():
    """Evaluate retrieval strategies on a labeled question set."""
    data = request.get_json(force=True, silent=True) or {}
    questions = data.get("questions") or []
    k = int(data.get("k", 5))
    if not questions:
        return jsonify({"error": "questions array required"}), 400
    from services.retrieval_planner import evaluate_strategy
    result = evaluate_strategy(questions, g.tenant_id, get_vs(), k=k)
    return jsonify(result)


# ── D: Document Impact Analysis ───────────────────────────────────────────────

@app.get("/api/impact/<doc_id>")
@require_auth()
def get_impact(doc_id):
    """Get the stored impact analysis report for a document."""
    from sqlalchemy import text as sql_text
    row = db.session.execute(
        sql_text(
            "SELECT impact_analysis FROM documents WHERE id = :id AND tenant_id = :tenant"
        ),
        {"id": doc_id, "tenant": g.tenant_id},
    ).fetchone()
    if not row:
        return jsonify({"error": "not found"}), 404
    return jsonify({"document_id": doc_id, "impact": row[0] or {}})


@app.post("/api/impact/<doc_id>/reanalyse")
@require_auth()
def reanalyse_impact(doc_id):
    """Re-run impact analysis for a document (admin action)."""
    doc = Document.query.filter_by(id=doc_id, tenant_id=g.tenant_id).first()
    if not doc:
        return jsonify({"error": "not found"}), 404
    # Reconstruct text from chunks
    from services.vectorstore import VectorStore
    from services.impact import run_impact_analysis
    chunks = get_vs().get_document_chunks(doc_id, g.tenant_id)
    full_text = "\n\n".join(c["text"] for c in chunks)
    report = run_impact_analysis(
        doc_id, g.tenant_id, full_text, doc.original_name or doc.filename
    )
    return jsonify(report)


@app.get("/api/impact/graph")
@require_auth()
def dependency_graph():
    """Get the document dependency graph for a tenant."""
    from services.impact import get_document_dependency_graph
    graph = get_document_dependency_graph(g.tenant_id)
    return jsonify(graph)


# ── E: Provenance / Reproducibility ──────────────────────────────────────────

@app.get("/api/provenance")
@require_auth()
def list_provenance():
    """List recent execution logs for the current user."""
    limit = min(int(request.args.get("limit", 20)), 100)
    offset = int(request.args.get("offset", 0))
    from services.provenance import list_executions
    logs = list_executions(g.tenant_id, user_id=g.user.id, limit=limit, offset=offset)
    return jsonify({"logs": logs, "count": len(logs)})


@app.get("/api/provenance/<log_id>")
@require_auth()
def get_provenance(log_id):
    """Get the full provenance trace for a query execution."""
    from services.provenance import get_execution_trace
    trace = get_execution_trace(log_id, g.tenant_id)
    if not trace:
        return jsonify({"error": "not found"}), 404
    return jsonify(trace)


@app.post("/api/provenance/<log_id>/replay")
@require_auth()
def replay_provenance(log_id):
    """Replay a historical query using its original evidence snapshot."""
    from services.provenance import replay_execution
    result = replay_execution(log_id, g.tenant_id, get_llm())
    if "error" in result:
        return jsonify(result), 404
    return jsonify(result)


@app.get("/api/provenance/regressions")
@require_auth()
def provenance_regressions():
    """Detect answer drift / regressions across repeated queries."""
    from services.provenance import detect_retrieval_regression
    result = detect_retrieval_regression(g.tenant_id)
    return jsonify(result)


# ── F: Semantic Cache ─────────────────────────────────────────────────────────

@app.get("/api/cache/metrics")
@require_auth()
def cache_metrics():
    """Get semantic cache performance metrics."""
    from services.semantic_cache import get_metrics
    return jsonify(get_metrics(g.tenant_id))


@app.post("/api/cache/invalidate")
@require_auth()
def cache_invalidate():
    """Invalidate all cache entries for the current tenant."""
    from services.semantic_cache import invalidate_for_document
    count = invalidate_for_document("manual_invalidation", g.tenant_id)
    return jsonify({"invalidated": count, "tenant_id": g.tenant_id})


# ── G: Ingestion Quality ──────────────────────────────────────────────────────

@app.get("/api/quality")
@require_auth()
def quality_report():
    """Get ingestion quality reports for the current tenant."""
    include_q = request.args.get("quarantined", "true").lower() == "true"
    from services.ingestion_quality import get_quality_report
    reports = get_quality_report(g.tenant_id, include_quarantined=include_q)
    return jsonify({"reports": reports, "count": len(reports)})


@app.post("/api/quality/<doc_id>/reprocess")
@require_auth()
def reprocess_doc(doc_id):
    """Trigger reprocessing of a quarantined/failed document."""
    from services.ingestion_quality import reprocess_document
    result = reprocess_document(doc_id, g.tenant_id, get_vs())
    return jsonify(result)


# ── H: Counterfactual Testing ─────────────────────────────────────────────────

@app.post("/api/counterfactual")
@require_auth()
def create_counterfactual():
    """Create a counterfactual test."""
    data = request.get_json(force=True, silent=True) or {}
    required = ["base_document_id", "modification_type", "modification_spec", "test_questions"]
    for f in required:
        if not data.get(f):
            return jsonify({"error": f"{f} required"}), 400
    from services.counterfactual import create_test
    test_id = create_test(
        tenant_id=g.tenant_id,
        base_document_id=data["base_document_id"],
        name=data.get("name", "Unnamed test"),
        modification_type=data["modification_type"],
        modification_spec=data["modification_spec"],
        test_questions=data["test_questions"],
        created_by=g.user.id if g.user else None,
    )
    return jsonify({"test_id": test_id}), 201


@app.post("/api/counterfactual/<test_id>/run")
@require_auth()
def run_counterfactual(test_id):
    """Execute a counterfactual test."""
    from services.counterfactual import run_test
    result = run_test(test_id, g.tenant_id, get_vs(), get_llm())
    if "error" in result and "test_id" not in result:
        return jsonify(result), 404
    return jsonify(result)


@app.get("/api/counterfactual")
@require_auth()
def list_counterfactuals():
    from services.counterfactual import list_tests
    return jsonify({"tests": list_tests(g.tenant_id)})


@app.get("/api/counterfactual/<test_id>")
@require_auth()
def get_counterfactual(test_id):
    from services.counterfactual import get_test_result
    result = get_test_result(test_id, g.tenant_id)
    if not result:
        return jsonify({"error": "not found"}), 404
    return jsonify(result)


# ── K: Contradiction Discovery ────────────────────────────────────────────────

@app.get("/api/conflicts")
@require_auth()
def get_conflicts():
    """Get detected document conflicts for the current tenant."""
    resolved = request.args.get("resolved", "false").lower() == "true"
    severity = request.args.get("severity")
    limit = min(int(request.args.get("limit", 50)), 200)
    from services.contradiction import get_conflicts as get_c
    conflicts = get_c(g.tenant_id, resolved=resolved, severity=severity, limit=limit)
    return jsonify({"conflicts": conflicts, "count": len(conflicts)})


@app.post("/api/conflicts/<conflict_id>/resolve")
@require_auth()
def resolve_conflict(conflict_id):
    """Mark a conflict as resolved."""
    from services.contradiction import resolve_conflict as res_c
    ok = res_c(conflict_id, g.tenant_id)
    if not ok:
        return jsonify({"error": "not found"}), 404
    return jsonify({"ok": True})


# ── L: Knowledge Expiration ───────────────────────────────────────────────────

@app.get("/api/expiry")
@require_auth()
def expiry_report():
    """Get knowledge expiration report."""
    statuses = request.args.getlist("status") or None
    from services.expiration import get_expiry_report, get_expiry_summary
    report = get_expiry_report(g.tenant_id, statuses=statuses)
    summary = get_expiry_summary(g.tenant_id)
    return jsonify({"report": report, "summary": summary})


@app.post("/api/expiry/refresh")
@require_auth()
def refresh_expiry():
    """Recompute expiry statuses for all tenant documents."""
    from services.expiration import refresh_expiry_statuses
    counts = refresh_expiry_statuses(g.tenant_id)
    return jsonify({"status_counts": counts})


@app.post("/api/expiry/<doc_id>")
@require_auth()
def set_expiry(doc_id):
    """Set or update expiry metadata for a document."""
    data = request.get_json(force=True, silent=True) or {}
    from services.expiration import upsert_expiration
    eid = upsert_expiration(
        doc_id, g.tenant_id,
        expiry_date=data.get("expiry_date"),
        review_date=data.get("review_date"),
        effective_period_end=data.get("effective_period_end"),
        expiry_source="admin",
        notes=data.get("notes", ""),
    )
    return jsonify({"expiry_id": eid})


# ── M: Drift Detection ────────────────────────────────────────────────────────

@app.post("/api/drift/evaluate")
@require_auth()
def run_drift():
    """Run retrieval drift evaluation."""
    from services.drift import run_drift_evaluation
    result = run_drift_evaluation(g.tenant_id, get_vs(), trigger_event="manual")
    return jsonify(result)


@app.get("/api/drift/history")
@require_auth()
def drift_history():
    """Get drift run history for the current tenant."""
    limit = min(int(request.args.get("limit", 20)), 100)
    from services.drift import get_drift_history
    history = get_drift_history(g.tenant_id, limit=limit)
    return jsonify({"history": history, "count": len(history)})


@app.post("/api/drift/baseline")
@require_auth()
def add_baseline():
    """Add a question to the retrieval evaluation baseline."""
    data = request.get_json(force=True, silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "question required"}), 400
    from services.drift import create_baseline_question
    bid = create_baseline_question(
        tenant_id=g.tenant_id,
        question=question,
        relevant_doc_ids=data.get("relevant_doc_ids", []),
        relevant_chunk_ids=data.get("relevant_chunk_ids", []),
        expected_keywords=data.get("expected_keywords", []),
        difficulty_class=data.get("difficulty_class"),
        created_by=g.user.id if g.user else None,
    )
    return jsonify({"baseline_id": bid}), 201


@app.get("/api/drift/baseline")
@require_auth()
def list_baseline():
    """List baseline evaluation questions."""
    from services.drift import get_baseline_questions
    questions = get_baseline_questions(g.tenant_id)
    return jsonify({"questions": questions, "count": len(questions)})


# ── N: Document Trust Scoring ─────────────────────────────────────────────────

@app.get("/api/trust")
@require_auth()
def get_trust_scores():
    """Get trust scores for all tenant documents."""
    limit = min(int(request.args.get("limit", 50)), 200)
    from services.trust import get_trust_scores as get_ts
    scores = get_ts(g.tenant_id, limit=limit)
    return jsonify({"trust_scores": scores, "count": len(scores)})


@app.get("/api/trust/<doc_id>")
@require_auth()
def get_doc_trust(doc_id):
    """Get detailed trust score for a specific document."""
    doc = Document.query.filter_by(id=doc_id, tenant_id=g.tenant_id).first()
    if not doc:
        return jsonify({"error": "not found"}), 404
    from services.trust import compute_trust_score
    return jsonify(compute_trust_score(doc_id))


@app.post("/api/trust/recompute")
@require_auth()
def recompute_trust():
    """Recompute trust scores for all documents in tenant."""
    from services.trust import compute_trust_for_tenant
    result = compute_trust_for_tenant(g.tenant_id)
    return jsonify(result)


# ── P: Information Gap Discovery ──────────────────────────────────────────────

@app.get("/api/gaps")
@require_auth()
def get_gaps():
    """Get information gap report for the current tenant."""
    min_count = int(request.args.get("min_count", 1))
    limit = min(int(request.args.get("limit", 50)), 200)
    from services.gaps import get_gap_report, get_gap_summary
    report = get_gap_report(g.tenant_id, min_count=min_count, limit=limit)
    summary = get_gap_summary(g.tenant_id)
    return jsonify({"gaps": report, "summary": summary})


@app.post("/api/gaps/aggregate")
@require_auth()
def aggregate_gaps():
    """Re-cluster unanswered questions into gap groups."""
    from services.gaps import aggregate_gaps as agg
    result = agg(g.tenant_id)
    return jsonify(result)


@app.post("/api/gaps/<gap_id>/resolve")
@require_auth()
def resolve_gap(gap_id):
    """Mark a knowledge gap as resolved."""
    from services.gaps import resolve_gap as res_gap
    ok = res_gap(gap_id, g.tenant_id)
    if not ok:
        return jsonify({"error": "not found"}), 404
    return jsonify({"ok": True})


# ── Admin: Tenant-wide intelligence dashboard ─────────────────────────────────

@app.get("/api/admin/dashboard")
@require_auth()
def admin_dashboard():
    """Aggregate intelligence dashboard for the current tenant."""
    tenant_id = g.tenant_id

    # Document stats
    from sqlalchemy import text as sql_text
    doc_stats = db.session.execute(
        sql_text(
            "SELECT status, COUNT(*) FROM documents "
            "WHERE tenant_id = :t GROUP BY status"
        ),
        {"t": tenant_id},
    ).fetchall()

    from services.contradiction import get_conflicts as get_c
    from services.expiration import get_expiry_summary
    from services.gaps import get_gap_summary
    from services.semantic_cache import get_metrics as cache_metrics_fn
    from services.drift import get_drift_history

    conflicts = get_c(tenant_id, resolved=False, limit=5)
    expiry_summary = get_expiry_summary(tenant_id)
    gap_summary = get_gap_summary(tenant_id)
    cache_stats = cache_metrics_fn(tenant_id)
    drift_runs = get_drift_history(tenant_id, limit=1)

    return jsonify({
        "tenant_id": tenant_id,
        "documents": {s: c for s, c in doc_stats},
        "conflicts": {
            "unresolved": len(conflicts),
            "recent": conflicts[:3],
        },
        "expiry": expiry_summary,
        "knowledge_gaps": gap_summary,
        "cache": {
            "exact_hit_rate": cache_stats.get("exact_hit_rate"),
            "semantic_hit_rate": cache_stats.get("semantic_hit_rate"),
            "active_entries": cache_stats.get("active_entries"),
        },
        "drift": drift_runs[0] if drift_runs else None,
    })


# ── Updated /api/ask — orchestrates all new features ──────────────────────────
# The existing /api/ask is preserved as-is for backward compatibility.
# A new /api/ask/v2 endpoint uses the full intelligence pipeline.

@app.post("/api/ask/v2")
@require_auth(optional=True)
def api_ask_v2():
    """
    Advanced ask endpoint using all intelligence features:
    - Difficulty estimation (O)
    - Adaptive retrieval planning (C)
    - Semantic cache (F)
    - Query decomposition (J)
    - Temporal awareness (A)
    - Trust-weighted reranking (N)
    - Evidence verification (B)
    - Provenance recording (E)
    - Information gap tracking (P)
    """
    t0 = time.perf_counter()
    data = request.get_json(force=True, silent=True) or {}
    question = (data.get("question") or data.get("query") or "").strip()
    if not question:
        return jsonify({"error": "question required"}), 400

    tenant_id = g.tenant_id
    user = g.user
    conversation_id = data.get("conversation_id")
    history = data.get("history") or []
    as_of = data.get("as_of")  # Feature A: historical query

    # Credit check
    if user:
        ok, msg = _check_credits(user, Config.CREDITS_PER_ASK)
        if not ok:
            return jsonify({"error": msg, "credits": user.credits}), 402

    llm = get_llm()
    vs = get_vs()

    # ── O: Estimate difficulty ─────────────────────────────────────────────
    from services.difficulty import estimate_difficulty, should_skip_rerank, suggested_top_k
    from services.retrieval_planner import classify_query, execute_retrieval
    from services.decomposition import detect_decomposition_need, execute_decomposed_retrieval
    from services.semantic_cache import lookup as cache_lookup, store as cache_store
    from services.evidence import verify_answer as ev_verify
    from services.trust import apply_trust_reranking
    from services.gaps import record_if_unanswered

    clf = classify_query(question)
    difficulty = estimate_difficulty(question, clf)
    top_k = suggested_top_k(difficulty, Config.TOP_K)

    # ── F: Semantic cache lookup ───────────────────────────────────────────
    cache_entry, hit_type = cache_lookup(question, tenant_id, as_of=as_of)
    if cache_entry and Config.SEMANTIC_CACHE_ENABLED:
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        return jsonify({
            "answer": cache_entry["answer"],
            "sources": cache_entry["sources"],
            "conversation_id": conversation_id,
            "latency_ms": latency_ms,
            "cache_hit": hit_type,
            "difficulty": difficulty["difficulty_class"],
            "retrieval_strategy": clf["strategy"],
            "credits": user.credits if user else 50,
            "max_credits": user.max_credits if user else 50,
        })

    # ── Conversation history ───────────────────────────────────────────────
    conv = None
    if conversation_id:
        conv = Conversation.query.filter_by(
            id=conversation_id, tenant_id=tenant_id
        ).first()
    if not conv and user:
        conv = Conversation(
            id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            user_id=user.id,
            title=question[:80],
        )
        db.session.add(conv)
        db.session.commit()
    if conv and not history:
        msgs = (
            Message.query.filter_by(conversation_id=conv.id)
            .order_by(Message.created_at.desc())
            .limit(Config.HISTORY_TURNS * 2)
            .all()
        )
        history = [
            {"role": m.role, "content": m.content}
            for m in reversed(msgs)
            if m.role in ("user", "assistant")
        ]

    # Query rewrite for follow-ups
    search_query = llm.rewrite_query(question, history)

    # ── J: Query decomposition for complex questions ───────────────────────
    was_decomposed = False
    sub_results_data = None
    if (Config.DECOMPOSITION_ENABLED
            and detect_decomposition_need(search_query, clf)):
        decomp = execute_decomposed_retrieval(
            search_query, tenant_id, vs, llm, clf,
            top_k=Config.RERANK_TOP_K, as_of=as_of,
        )
        answer = decomp["final_answer"]
        sources = decomp["all_sources"]
        passages = []
        for sr in decomp.get("sub_results", []):
            passages.extend(sr.get("passages", []) if isinstance(sr.get("passages"), list) else [])
        was_decomposed = decomp.get("was_decomposed", False)
        sub_results_data = decomp.get("sub_results")
    else:
        # ── C: Adaptive retrieval ──────────────────────────────────────────
        passages, retrieval_cost = execute_retrieval(
            search_query, tenant_id, vs, clf,
            top_k=top_k, as_of=as_of,
        )

        # ── A: Temporal conflict resolution ───────────────────────────────
        from services.temporal import resolve_temporal_conflict
        passages = resolve_temporal_conflict(passages, as_of=as_of)

        # ── N: Trust-weighted reranking ────────────────────────────────────
        passages = apply_trust_reranking(passages, trust_weight=Config.TRUST_RERANK_WEIGHT)

        # ── Cross-encoder reranking (skip for simple questions) ────────────
        if not should_skip_rerank(difficulty):
            from services.reranker import rerank
            passages = rerank(
                question, passages,
                top_k=Config.RERANK_TOP_K,
                timeout_s=Config.RERANK_TIMEOUT_S,
            )
        else:
            passages = passages[:Config.RERANK_TOP_K]

        # ── Generate answer ────────────────────────────────────────────────
        answer, sources, usage = llm.generate_answer_with_context(
            question, passages, history=history,
            as_of=as_of,
            difficulty_class=difficulty["difficulty_class"],
        )

    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    # ── B: Evidence verification ───────────────────────────────────────────
    verification = {}
    if Config.EVIDENCE_VERIFY_CLAIMS and passages:
        verification = ev_verify(answer, passages, llm_client=llm,
                                  max_claims=Config.EVIDENCE_MAX_CLAIMS)

    # ── P: Track unanswered questions ─────────────────────────────────────
    log_id = None
    if user:
        record_if_unanswered(question, answer, tenant_id, user_id=user.id)

    # ── Persist conversation ───────────────────────────────────────────────
    if conv:
        db.session.add(Message(
            conversation_id=conv.id, role="user", content=question,
        ))
        db.session.add(Message(
            conversation_id=conv.id, role="assistant",
            content=answer, sources=sources,
            latency_ms=latency_ms, token_usage={},
        ))
        db.session.commit()

    # ── F: Store in cache ──────────────────────────────────────────────────
    if Config.SEMANTIC_CACHE_ENABLED and answer and "don't know" not in answer.lower():
        cache_store(question, tenant_id, answer, sources, as_of=as_of)

    # ── E: Record provenance ───────────────────────────────────────────────
    if Config.PROVENANCE_ENABLED:
        from services.provenance import record_execution
        try:
            log_id = record_execution(
                tenant_id=tenant_id,
                user_id=user.id if user else None,
                conversation_id=conv.id if conv else None,
                question=question,
                search_query=search_query,
                retrieval_strategy=clf.get("strategy", "hybrid"),
                query_intent=clf.get("intent"),
                difficulty_class=difficulty.get("difficulty_class"),
                passages=passages if not was_decomposed else [],
                answer=answer,
                sources=sources,
                latency_ms=latency_ms,
                token_usage={},
                reranked=not should_skip_rerank(difficulty),
                was_decomposed=was_decomposed,
                cache_hit=False,
                as_of_date=as_of,
            )
        except Exception as e:
            logger.debug("Provenance record failed: %s", e)

    # Deduct credits
    if user:
        balance = _deduct_credits(user, Config.CREDITS_PER_ASK, "ask", {
            "conversation_id": conv.id if conv else None,
        })
    else:
        balance = 50

    return jsonify({
        "answer": answer,
        "sources": sources,
        "conversation_id": conv.id if conv else None,
        "latency_ms": latency_ms,
        "retrieved": len(passages) if not was_decomposed else None,
        "credits": balance,
        "max_credits": user.max_credits if user else 50,
        "search_query": search_query if search_query != question else None,
        "as_of": as_of,
        # Intelligence metadata
        "difficulty": difficulty["difficulty_class"],
        "complexity_score": difficulty.get("complexity_score"),
        "retrieval_strategy": clf.get("strategy"),
        "query_intent": clf.get("intent"),
        "was_decomposed": was_decomposed,
        "sub_questions": [r["question"] for r in (sub_results_data or [])],
        "cache_hit": False,
        "log_id": log_id,
        # Verification
        "verification": {
            "total_claims": verification.get("total_claims", 0),
            "supported_claims": verification.get("supported_claims", 0),
            "citation_coverage": verification.get("citation_coverage", 1.0),
        } if verification else None,
    })


if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=Config.PORT, debug=Config.DEBUG)
else:
    # Gunicorn / production
    with app.app_context():
        try:
            init_db()
        except Exception as e:
            logger.warning("init_db on import: %s", e)
