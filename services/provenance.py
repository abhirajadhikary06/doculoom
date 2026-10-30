"""
Features E + I: Retrieval Reproducibility & Query-to-Answer Provenance Explorer

Stores a complete execution record for every query:
- original question, tenant, conversation
- document IDs/versions at retrieval time
- retrieved chunk IDs, ranks, scores (dense, sparse, RRF, rerank)
- retrieval config, embedding model, LLM model, prompt version
- answer hash + latency

Supports:
- Historical replay (rerun a query against the original evidence snapshot)
- Answer diff detection
- Retrieval regression identification
- Full provenance trace for the UI explorer (Feature I)
"""
from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import text

from config import Config
from models import db

logger = logging.getLogger(__name__)

PROMPT_VERSION = "v2"  # bump when system prompt changes


# ── Answer hashing ────────────────────────────────────────────────────────────

def hash_answer(answer: str) -> str:
    """Stable hash of an answer text for diff detection."""
    normalized = " ".join(answer.lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:32]


# ── Record a query execution ──────────────────────────────────────────────────

def record_execution(
    *,
    tenant_id: str,
    user_id: Optional[str],
    conversation_id: Optional[str],
    question: str,
    search_query: str,
    retrieval_strategy: str,
    query_intent: Optional[str],
    difficulty_class: Optional[str],
    passages: List[Dict[str, Any]],
    answer: str,
    sources: List[Dict[str, Any]],
    latency_ms: float,
    token_usage: Dict[str, Any],
    reranked: bool = False,
    was_decomposed: bool = False,
    cache_hit: bool = False,
    as_of_date: Optional[str] = None,
) -> str:
    """
    Persist a full execution record (query_execution_logs + execution_evidence).
    Returns the log_id.
    """
    log_id = str(uuid.uuid4())
    answer_hash = hash_answer(answer)

    retrieval_config = {
        "top_k": Config.TOP_K,
        "rerank_top_k": Config.RERANK_TOP_K,
        "chunk_size": Config.CHUNK_SIZE,
        "chunk_overlap": Config.CHUNK_OVERLAP,
    }

    try:
        from models import QueryExecutionLog, ExecutionEvidence
        qlog = QueryExecutionLog(
            id=log_id,
            tenant_id=tenant_id,
            user_id=user_id,
            conversation_id=conversation_id,
            question=question,
            search_query=search_query,
            retrieval_strategy=retrieval_strategy,
            query_intent=query_intent,
            difficulty_class=difficulty_class,
            answer_hash=answer_hash,
            answer_text=answer,
            sources=sources,
            retrieval_config=retrieval_config,
            embedding_model=Config.EMBEDDING_MODEL,
            llm_model=Config.GROQ_MODEL,
            prompt_version=PROMPT_VERSION,
            latency_ms=latency_ms,
            token_usage=token_usage,
            reranked=reranked,
            was_decomposed=was_decomposed,
            cache_hit=cache_hit,
            as_of_date=as_of_date,
        )
        db.session.add(qlog)

        # Store per-chunk evidence
        for rank, p in enumerate(passages):
            ev_id = str(uuid.uuid4())
            # Try to get document version hash
            doc_version_row = db.session.execute(
                text(
                    "SELECT content_hash FROM document_versions "
                    "WHERE document_id = :doc_id "
                    "ORDER BY version_number DESC LIMIT 1"
                ),
                {"doc_id": p.get("document_id", "")},
            ).fetchone()
            doc_version = doc_version_row[0] if doc_version_row else None

            ev = ExecutionEvidence(
                id=ev_id,
                log_id=log_id,
                tenant_id=tenant_id,
                chunk_id=p.get("id", ""),
                document_id=p.get("document_id", ""),
                chunk_index=p.get("chunk_index", 0),
                page_start=p.get("page_start", 1),
                retrieval_rank=rank,
                dense_score=p.get("dense_score"),
                sparse_score=p.get("sparse_score"),
                rrf_score=p.get("score"),
                rerank_score=p.get("rerank_score"),
                text_snapshot=(p.get("text") or "")[:2000],
                doc_version=doc_version,
            )
            db.session.add(ev)

        db.session.commit()
        logger.debug("Execution recorded: log_id=%s", log_id)
        return log_id

    except Exception as e:
        logger.error("Failed to record execution: %s", e)
        db.session.rollback()
        return log_id


# ── Retrieve a full provenance trace ─────────────────────────────────────────

def get_execution_trace(log_id: str, tenant_id: str) -> Optional[Dict[str, Any]]:
    """
    Return the full provenance trace for a log_id.
    Tenant-isolated: returns None if the log doesn't belong to tenant.
    """
    row = db.session.execute(
        text(
            """
            SELECT id, tenant_id, user_id, conversation_id,
                   question, search_query, retrieval_strategy,
                   query_intent, difficulty_class,
                   answer_hash, answer_text, sources,
                   retrieval_config, embedding_model, llm_model, prompt_version,
                   latency_ms, token_usage, reranked, was_decomposed,
                   cache_hit, as_of_date, created_at
            FROM query_execution_logs
            WHERE id = :id AND tenant_id = :tenant
            """
        ),
        {"id": log_id, "tenant": tenant_id},
    ).fetchone()

    if not row:
        return None

    (
        lid, tid, uid, cid, question, sq, strategy,
        intent, diff_class, ahash, answer, sources,
        rc, emb, llm, pv, lat, usage, reranked, decomposed,
        cache_hit, as_of, created_at,
    ) = row

    # Get evidence passages
    evidence_rows = db.session.execute(
        text(
            """
            SELECT ee.chunk_id, ee.document_id, ee.chunk_index, ee.page_start,
                   ee.retrieval_rank, ee.dense_score, ee.sparse_score,
                   ee.rrf_score, ee.rerank_score, ee.text_snapshot, ee.doc_version,
                   d.original_name
            FROM execution_evidence ee
            LEFT JOIN documents d ON d.id = ee.document_id
            WHERE ee.log_id = :log_id
            ORDER BY ee.retrieval_rank ASC
            """
        ),
        {"log_id": log_id},
    ).fetchall()

    evidence = []
    for er in evidence_rows:
        evidence.append(
            {
                "chunk_id": er[0],
                "document_id": er[1],
                "chunk_index": er[2],
                "page": er[3],
                "rank": er[4],
                "dense_score": er[5],
                "sparse_score": er[6],
                "rrf_score": er[7],
                "rerank_score": er[8],
                "text_snapshot": er[9],
                "doc_version": er[10],
                "filename": er[11],
            }
        )

    return {
        "log_id": lid,
        "tenant_id": tid,
        "question": question,
        "search_query": sq,
        "retrieval_strategy": strategy,
        "query_intent": intent,
        "difficulty_class": diff_class,
        "answer_hash": ahash,
        "answer": answer,
        "sources": sources,
        "retrieval_config": rc,
        "embedding_model": emb,
        "llm_model": llm,
        "prompt_version": pv,
        "latency_ms": lat,
        "token_usage": usage,
        "reranked": reranked,
        "was_decomposed": decomposed,
        "cache_hit": cache_hit,
        "as_of_date": str(as_of) if as_of else None,
        "created_at": created_at.isoformat() if hasattr(created_at, "isoformat") else (str(created_at) if created_at else None),
        "evidence": evidence,
        "evidence_count": len(evidence),
    }


# ── Historical replay ─────────────────────────────────────────────────────────

def replay_execution(
    log_id: str,
    tenant_id: str,
    llm_service,
) -> Dict[str, Any]:
    """
    Replay a historical query using its original evidence snapshot.
    Does NOT re-retrieve from Qdrant — uses stored text_snapshots.

    Returns {question, original_answer_hash, new_answer_hash, is_same, diff_summary}.
    """
    trace = get_execution_trace(log_id, tenant_id)
    if not trace:
        return {"error": "execution log not found"}

    # Rebuild passages from stored snapshots
    passages = []
    for ev in trace["evidence"]:
        passages.append(
            {
                "text": ev.get("text_snapshot", ""),
                "filename": ev.get("filename", ""),
                "document_id": ev.get("document_id", ""),
                "page_start": ev.get("page", 1),
                "page_end": ev.get("page", 1),
                "chunk_index": ev.get("chunk_index", 0),
                "score": ev.get("rrf_score", 0),
            }
        )

    if not passages:
        return {
            "log_id": log_id,
            "question": trace["question"],
            "error": "No evidence snapshots stored for this execution",
        }

    # Generate answer from the original evidence
    try:
        new_answer, new_sources, new_usage = llm_service.generate_answer(
            trace["question"], passages, history=None
        )
    except Exception as e:
        return {
            "log_id": log_id,
            "question": trace["question"],
            "error": f"LLM call failed: {e}",
        }

    original_hash = trace["answer_hash"]
    new_hash = hash_answer(new_answer)
    is_same = original_hash == new_hash

    # Simple diff: word count delta
    original_words = set((trace.get("answer") or "").lower().split())
    new_words = set(new_answer.lower().split())
    added = new_words - original_words
    removed = original_words - new_words

    return {
        "log_id": log_id,
        "question": trace["question"],
        "original_answer": trace.get("answer"),
        "replayed_answer": new_answer,
        "original_answer_hash": original_hash,
        "new_answer_hash": new_hash,
        "is_same": is_same,
        "diff_summary": {
            "words_added": len(added),
            "words_removed": len(removed),
            "sample_added": list(added)[:10],
            "sample_removed": list(removed)[:10],
        },
        "evidence_count": len(passages),
        "original_latency_ms": trace.get("latency_ms"),
    }


# ── Regression detection ──────────────────────────────────────────────────────

def detect_retrieval_regression(
    tenant_id: str,
    recent_n: int = 50,
) -> Dict[str, Any]:
    """
    Compare recent query logs to detect answer drift.
    For each conversation that has multiple Q&A turns for the same question,
    check if the answer changed.
    Returns a regression summary.
    """
    rows = db.session.execute(
        text(
            """
            SELECT question, answer_hash, created_at, id
            FROM query_execution_logs
            WHERE tenant_id = :tenant
            ORDER BY created_at DESC
            LIMIT :n
            """
        ),
        {"tenant": tenant_id, "n": recent_n},
    ).fetchall()

    # Group by question (exact match)
    by_question: Dict[str, List] = {}
    for q, ah, ts, lid in rows:
        by_question.setdefault(q, []).append(
            {"hash": ah, "ts": ts, "log_id": lid}
        )

    regressions = []
    for question, entries in by_question.items():
        if len(entries) < 2:
            continue
        # Sort by timestamp
        entries_sorted = sorted(entries, key=lambda x: x["ts"])
        hashes = [e["hash"] for e in entries_sorted]
        if len(set(hashes)) > 1:
            regressions.append(
                {
                    "question": question[:120],
                    "occurrence_count": len(entries),
                    "distinct_answers": len(set(hashes)),
                    "first_log_id": entries_sorted[0]["log_id"],
                    "latest_log_id": entries_sorted[-1]["log_id"],
                    "first_seen": (
                        entries_sorted[0]["ts"].isoformat()
                        if hasattr(entries_sorted[0]["ts"], "isoformat")
                        else str(entries_sorted[0]["ts"])
                    )
                    if entries_sorted[0]["ts"]
                    else None,
                }
            )

    return {
        "tenant_id": tenant_id,
        "logs_analysed": len(rows),
        "regressions_found": len(regressions),
        "regressions": regressions[:20],
    }


# ── Recent executions listing ─────────────────────────────────────────────────

def list_executions(
    tenant_id: str,
    user_id: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """List recent execution logs for a tenant (or specific user)."""
    filters = "WHERE tenant_id = :tenant"
    params: Dict[str, Any] = {"tenant": tenant_id}
    if user_id:
        filters += " AND user_id = :user"
        params["user"] = user_id

    rows = db.session.execute(
        text(
            f"""
            SELECT id, question, retrieval_strategy, query_intent,
                   difficulty_class, latency_ms, cache_hit,
                   was_decomposed, created_at
            FROM query_execution_logs
            {filters}
            ORDER BY created_at DESC
            LIMIT :limit OFFSET :offset
            """
        ),
        {**params, "limit": limit, "offset": offset},
    ).fetchall()

    return [
        {
            "log_id": r[0],
            "question": r[1][:120],
            "retrieval_strategy": r[2],
            "query_intent": r[3],
            "difficulty_class": r[4],
            "latency_ms": r[5],
            "cache_hit": r[6],
            "was_decomposed": r[7],
            "created_at": r[8].isoformat() if hasattr(r[8], "isoformat") else (str(r[8]) if r[8] else None),
        }
        for r in rows
    ]
