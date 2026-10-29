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

# Rate limiter
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=[Config.RATELIMIT_DEFAULT],
    storage_uri=Config.REDIS_URL or "memory://",
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


def init_db():
    Config.ensure_dirs()
    with app.app_context():
        db.create_all()
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
@require_auth()
@limiter.limit("10 per minute")
def api_ingest():
    tenant_id = g.tenant_id
    user = g.user
    # Credit check
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
    if doc.status == "ready":
        balance = _deduct_credits(user, Config.CREDITS_PER_INGEST, "ingest", {"doc_id": doc.id})
    else:
        balance = user.credits
    return jsonify(
        {
            "id": doc.id,
            "filename": doc.original_name or doc.filename,
            "status": doc.status,
            "page_count": doc.page_count,
            "chunk_count": doc.chunk_count,
            "error": doc.error_message,
            "source": doc.source,
            "credits": balance,
            "max_credits": user.max_credits,
        }
    ), (201 if doc.status == "ready" else 422)


@app.get("/api/docs")
@require_auth()
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
            "source": d.source,
            "created_at": d.created_at.isoformat() if d.created_at else None,
            "size_bytes": d.size_bytes,
        }
        for d in q.limit(200).all()
    ]
    return jsonify({"documents": docs, "count": len(docs)})


@app.delete("/api/docs/<doc_id>")
@require_auth()
def delete_doc(doc_id):
    tenant_id = g.tenant_id
    doc = Document.query.filter_by(id=doc_id, tenant_id=tenant_id).first()
    if not doc:
        return jsonify({"error": "not found"}), 404
    try:
        get_vs().delete_document(doc_id, tenant_id)
    except Exception as e:
        logger.warning("Qdrant delete: %s", e)
    db.session.delete(doc)
    db.session.commit()
    return jsonify({"ok": True})


# ─── Ask (RAG) ───────────────────────────────────────────────
@app.post("/api/ask")
@require_auth()
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
    balance = _deduct_credits(user, Config.CREDITS_PER_ASK, "ask", {
        "conversation_id": conv.id if conv else None,
    })

    return jsonify(
        {
            "answer": answer,
            "sources": sources,
            "conversation_id": conv.id if conv else None,
            "latency_ms": round(latency_ms, 1),
            "usage": usage,
            "retrieved": len(passages),
            "credits": balance,
            "max_credits": user.max_credits,
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
def index():
    return render_template("index.html")


@app.get("/static/<path:path>")
def static_files(path):
    return send_from_directory("static", path)


# ─── Main ────────────────────────────────────────────────────
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
