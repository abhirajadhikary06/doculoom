"""
Feature M: Retrieval Drift Detection

Detects deterioration in retrieval quality after changes to documents,
chunking, embeddings, or indexing by comparing against a stored baseline.

Metrics: Recall@K, MRR, nDCG@10
Trigger: manual, post-ingest, post-delete, scheduled
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import text

from config import Config
from models import db

logger = logging.getLogger(__name__)

# Drift alert threshold: if recall drops by > this much, alert
DRIFT_ALERT_THRESHOLD = float(getattr(Config, "DRIFT_ALERT_THRESHOLD", 0.15))


# ── Baseline management ───────────────────────────────────────────────────────

def create_baseline_question(
    tenant_id: str,
    question: str,
    relevant_doc_ids: List[str],
    relevant_chunk_ids: List[str],
    expected_keywords: List[str],
    difficulty_class: Optional[str] = None,
    created_by: Optional[str] = None,
) -> str:
    """Add a question to the evaluation baseline dataset."""
    bid = str(uuid.uuid4())
    from models import EvalBaseline
    eb = EvalBaseline(
        id=bid,
        tenant_id=tenant_id,
        question=question,
        relevant_doc_ids=relevant_doc_ids,
        relevant_chunk_ids=relevant_chunk_ids,
        expected_keywords=expected_keywords,
        difficulty_class=difficulty_class,
        created_by=created_by,
    )
    db.session.add(eb)
    db.session.commit()
    return bid


def get_baseline_questions(
    tenant_id: str,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """Retrieve the evaluation baseline for a tenant."""
    rows = db.session.execute(
        text(
            """
            SELECT id, question, relevant_doc_ids, relevant_chunk_ids,
                   expected_keywords, difficulty_class
            FROM eval_baseline
            WHERE tenant_id = :tenant
            LIMIT :lim
            """
        ),
        {"tenant": tenant_id, "lim": limit},
    ).fetchall()

    return [
        {
            "id": r[0],
            "question": r[1],
            "relevant_doc_ids": r[2] or [],
            "relevant_chunk_ids": r[3] or [],
            "expected_keywords": r[4] or [],
            "difficulty_class": r[5],
        }
        for r in rows
    ]


# ── Run drift evaluation ──────────────────────────────────────────────────────

def run_drift_evaluation(
    tenant_id: str,
    vector_store,
    trigger_event: str = "manual",
    k: int = 5,
) -> Dict[str, Any]:
    """
    Run retrieval evaluation against the baseline and detect drift.

    Returns a drift report with:
    - recall@k, MRR, nDCG@k
    - comparison to previous run
    - alert_triggered bool
    """
    baseline = get_baseline_questions(tenant_id)
    if not baseline:
        return {
            "error": "No baseline questions found. Add baseline questions first.",
            "tenant_id": tenant_id,
        }

    from services.retrieval_planner import (
        compute_recall_at_k,
        compute_mrr,
        compute_ndcg_at_k,
    )

    recalls, mrrs, ndcgs = [], [], []
    details = []

    for item in baseline:
        question = item["question"]
        relevant_chunks = item["relevant_chunk_ids"]
        relevant_docs = item["relevant_doc_ids"]
        expected_kws = item["expected_keywords"]

        try:
            passages = vector_store.hybrid_search(
                question, tenant_id=tenant_id, top_k=k * 2
            )
            retrieved_ids = [p["id"] for p in passages[:k]]
            retrieved_docs = [p.get("document_id", "") for p in passages[:k]]

            # Prefer chunk-level, fall back to doc-level
            if relevant_chunks:
                r = compute_recall_at_k(retrieved_ids, relevant_chunks, k)
                mrr = compute_mrr(retrieved_ids, relevant_chunks)
                ndcg = compute_ndcg_at_k(retrieved_ids, relevant_chunks, k)
            elif relevant_docs:
                r = compute_recall_at_k(retrieved_docs, relevant_docs, k)
                mrr = compute_mrr(retrieved_docs, relevant_docs)
                ndcg = compute_ndcg_at_k(retrieved_docs, relevant_docs, k)
            else:
                # Keyword-based evaluation only
                first_answer_text = " ".join(p.get("text", "") for p in passages[:k])
                kw_hit = any(
                    kw.lower() in first_answer_text.lower() for kw in expected_kws
                ) if expected_kws else None
                r, mrr, ndcg = (1.0 if kw_hit else 0.0), 0.0, 0.0

            recalls.append(r)
            mrrs.append(mrr)
            ndcgs.append(ndcg)
            details.append(
                {
                    "question": question[:80],
                    "recall": r,
                    "mrr": mrr,
                    "ndcg": ndcg,
                    "retrieved_count": len(passages),
                }
            )
        except Exception as e:
            logger.warning("Drift eval question failed: %s — %s", question[:40], e)
            details.append({"question": question[:80], "error": str(e)})

    n = len(recalls)
    if n == 0:
        return {"error": "All evaluation questions failed", "tenant_id": tenant_id}

    recall_at_k = round(sum(recalls) / n, 4)
    mrr_score = round(sum(mrrs) / n, 4)
    ndcg_score = round(sum(ndcgs) / n, 4)

    # Compare with previous run
    prev_run = _get_latest_drift_run(tenant_id)
    baseline_recall = prev_run["recall_at_5"] if prev_run else recall_at_k
    drift_pct = round((baseline_recall - recall_at_k) / max(baseline_recall, 0.01), 4)
    alert_triggered = drift_pct > DRIFT_ALERT_THRESHOLD

    if alert_triggered:
        logger.warning(
            "Retrieval drift alert for tenant %s: %.1f%% drop (was %.4f, now %.4f)",
            tenant_id,
            drift_pct * 100,
            baseline_recall,
            recall_at_k,
        )

    # Persist drift run
    run_id = _save_drift_run(
        tenant_id=tenant_id,
        trigger_event=trigger_event,
        recall_at_5=recall_at_k,
        recall_at_10=ndcg_score,  # reuse ndcg slot
        mrr=mrr_score,
        ndcg_at_10=ndcg_score,
        baseline_recall=baseline_recall,
        drift_pct=drift_pct,
        alert_triggered=alert_triggered,
        details={"questions": details[:50], "k": k},
    )

    return {
        "run_id": run_id,
        "tenant_id": tenant_id,
        "trigger_event": trigger_event,
        "k": k,
        "questions_evaluated": n,
        f"recall_at_{k}": recall_at_k,
        "mrr": mrr_score,
        f"ndcg_at_{k}": ndcg_score,
        "baseline_recall": baseline_recall,
        "drift_pct": drift_pct,
        "drift_pct_display": f"{drift_pct * 100:+.1f}%",
        "alert_triggered": alert_triggered,
        "embedding_model": Config.EMBEDDING_MODEL,
        "details": details[:20],
    }


# ── Persistence helpers ───────────────────────────────────────────────────────

def _save_drift_run(
    tenant_id: str,
    trigger_event: str,
    recall_at_5: float,
    recall_at_10: float,
    mrr: float,
    ndcg_at_10: float,
    baseline_recall: float,
    drift_pct: float,
    alert_triggered: bool,
    details: Dict,
) -> str:
    run_id = str(uuid.uuid4())
    try:
        from models import DriftRun
        dr = DriftRun(
            id=run_id,
            tenant_id=tenant_id,
            trigger_event=trigger_event,
            embedding_model=Config.EMBEDDING_MODEL,
            recall_at_5=recall_at_5,
            recall_at_10=recall_at_10,
            mrr=mrr,
            ndcg_at_10=ndcg_at_10,
            baseline_recall=baseline_recall,
            drift_pct=drift_pct,
            alert_triggered=alert_triggered,
            details=details,
        )
        db.session.add(dr)
        db.session.commit()
    except Exception as e:
        logger.error("Drift run save failed: %s", e)
        db.session.rollback()
    return run_id


def _get_latest_drift_run(tenant_id: str) -> Optional[Dict]:
    row = db.session.execute(
        text(
            """
            SELECT recall_at_5, mrr, ndcg_at_10, created_at
            FROM drift_runs
            WHERE tenant_id = :tenant
            ORDER BY created_at DESC
            LIMIT 1
            """
        ),
        {"tenant": tenant_id},
    ).fetchone()
    if not row:
        return None
    return {
        "recall_at_5": row[0],
        "mrr": row[1],
        "ndcg_at_10": row[2],
        "created_at": row[3].isoformat() if row[3] else None,
    }


def get_drift_history(
    tenant_id: str,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """Return recent drift run history for a tenant."""
    rows = db.session.execute(
        text(
            """
            SELECT id, trigger_event, recall_at_5, mrr, ndcg_at_10,
                   baseline_recall, drift_pct, alert_triggered, created_at
            FROM drift_runs
            WHERE tenant_id = :tenant
            ORDER BY created_at DESC
            LIMIT :lim
            """
        ),
        {"tenant": tenant_id, "lim": limit},
    ).fetchall()

    return [
        {
            "run_id": r[0],
            "trigger_event": r[1],
            "recall_at_5": r[2],
            "mrr": r[3],
            "ndcg_at_10": r[4],
            "baseline_recall": r[5],
            "drift_pct": r[6],
            "drift_pct_display": f"{(r[6] or 0) * 100:+.1f}%",
            "alert_triggered": r[7],
            "created_at": r[8].isoformat() if r[8] else None,
        }
        for r in rows
    ]
