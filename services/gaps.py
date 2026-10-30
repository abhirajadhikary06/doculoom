"""
Feature P: Information Gap Discovery

Discovers what the existing document corpus cannot answer.

Tracks unanswered questions, groups them by semantic similarity,
identifies recurring information gaps, detects missing document
categories, and generates administrator-facing gap reports.

An "unanswered" question is one where the LLM returned the abstention
phrase ("I don't know based on the available campus documents").
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import text

from models import db

logger = logging.getLogger(__name__)

ABSTENTION_PATTERN = re.compile(r"i don't know", re.IGNORECASE)
GAP_CLUSTER_THRESHOLD = 0.80   # cosine similarity to consider same gap
MIN_CLUSTER_SIZE = 2            # minimum questions to form a named gap


# ── Track unanswered questions ────────────────────────────────────────────────

def record_if_unanswered(
    question: str,
    answer: str,
    tenant_id: str,
    user_id: Optional[str] = None,
    log_id: Optional[str] = None,
) -> bool:
    """
    If the answer is an abstention, record the question as an information gap.
    Returns True if recorded, False if the question was answered.
    """
    if not ABSTENTION_PATTERN.search(answer):
        return False

    # Embed question for clustering
    embedding = _embed_question(question)

    # Find existing gap to link to
    gap_id = _find_matching_gap(question, tenant_id, embedding)

    q_id = str(uuid.uuid4())
    try:
        from models import GapQuestion
        gq = GapQuestion(
            id=q_id,
            tenant_id=tenant_id,
            gap_id=gap_id,
            question=question[:500],
            user_id=user_id,
            log_id=log_id,
            embedding=embedding or [],
        )
        db.session.add(gq)
        db.session.commit()

        # Update gap record if we matched one
        if gap_id:
            db.session.execute(
                text(
                    """
                    UPDATE information_gaps
                    SET question_count = question_count + 1,
                        last_seen_at = CURRENT_TIMESTAMP
                    WHERE id = :gap_id
                    """
                ),
                {"gap_id": gap_id},
            )
            db.session.commit()
        else:
            # Create a new gap cluster for this single question
            _create_gap(tenant_id, question, q_id, embedding)

        return True

    except Exception as e:
        logger.debug("Gap record failed: %s", e)
        db.session.rollback()
        return False


def _embed_question(question: str) -> Optional[List[float]]:
    try:
        from config import Config
        from fastembed import TextEmbedding
        model = TextEmbedding(model_name=Config.EMBEDDING_MODEL)
        vecs = list(model.embed([question]))
        return vecs[0].tolist()
    except Exception:
        return None


def _find_matching_gap(
    question: str,
    tenant_id: str,
    embedding: Optional[List[float]],
) -> Optional[str]:
    """Find an existing gap cluster that this question belongs to."""
    if not embedding:
        # Fallback: simple keyword match
        return _keyword_match_gap(question, tenant_id)

    # Load recent gaps with embeddings
    rows = db.session.execute(
        text(
            """
            SELECT g.id, q.embedding
            FROM information_gaps g
            JOIN gap_questions q ON q.gap_id = g.id
            WHERE g.tenant_id = :tenant AND g.resolved = FALSE
            LIMIT 200
            """
        ),
        {"tenant": tenant_id},
    ).fetchall()

    best_gap_id = None
    best_score = 0.0

    import numpy as np

    q_vec = np.array(embedding, dtype=np.float32)
    q_norm = np.linalg.norm(q_vec)

    for gap_id, gap_emb in rows:
        if not gap_emb:
            continue
        try:
            g_vec = np.array(gap_emb, dtype=np.float32)
            g_norm = np.linalg.norm(g_vec)
            if q_norm == 0 or g_norm == 0:
                continue
            score = float(np.dot(q_vec, g_vec) / (q_norm * g_norm))
            if score > best_score:
                best_score = score
                best_gap_id = gap_id
        except Exception:
            continue

    if best_score >= GAP_CLUSTER_THRESHOLD:
        return best_gap_id
    return None


def _keyword_match_gap(question: str, tenant_id: str) -> Optional[str]:
    """Simple keyword overlap matching for gap clustering."""
    q_words = set(re.findall(r"\b\w{4,}\b", question.lower()))
    rows = db.session.execute(
        text(
            """
            SELECT id, cluster_label FROM information_gaps
            WHERE tenant_id = :tenant AND resolved = FALSE
            LIMIT 100
            """
        ),
        {"tenant": tenant_id},
    ).fetchall()

    best_id, best_overlap = None, 0
    for gid, label in rows:
        if not label:
            continue
        label_words = set(re.findall(r"\b\w{4,}\b", label.lower()))
        overlap = len(q_words & label_words)
        if overlap > best_overlap and overlap >= 2:
            best_overlap = overlap
            best_id = gid

    return best_id


def _create_gap(
    tenant_id: str,
    question: str,
    question_id: str,
    embedding: Optional[List[float]],
) -> str:
    """Create a new gap cluster."""
    gap_id = str(uuid.uuid4())
    category = _classify_gap_category(question)

    from models import InformationGap
    ig = InformationGap(
        id=gap_id,
        tenant_id=tenant_id,
        cluster_label=question[:200],
        question_count=1,
        sample_questions=[question[:300]],
        category=category,
    )
    db.session.add(ig)
    db.session.commit()

    # Link the question to the new gap
    db.session.execute(
        text("UPDATE gap_questions SET gap_id = :gap WHERE id = :q"),
        {"gap": gap_id, "q": question_id},
    )
    db.session.commit()
    return gap_id


def _classify_gap_category(question: str) -> str:
    """Classify the information gap into a document category."""
    q = question.lower()
    categories = {
        "hostel_facilities": ["hostel", "accommodation", "room", "warden"],
        "fee_structure": ["fee", "tuition", "payment", "refund", "challan"],
        "placement": ["placement", "recruitment", "job", "company", "campus drive"],
        "scholarship": ["scholarship", "fellowship", "financial aid", "stipend"],
        "library": ["library", "book", "journal", "e-resource"],
        "transport": ["bus", "transport", "route", "shuttle"],
        "medical": ["medical", "health", "doctor", "hospital", "sick"],
        "sports": ["sports", "gym", "playground", "tournament"],
        "clubs": ["club", "society", "association", "cultural", "technical"],
        "research": ["research", "project", "publication", "patent", "grant"],
        "examination": ["exam", "examination", "marks", "result", "grade"],
        "attendance": ["attendance", "leave", "absent", "presence"],
        "admission": ["admission", "enrollment", "registration", "apply"],
    }
    for cat, keywords in categories.items():
        if any(kw in q for kw in keywords):
            return cat
    return "general"


# ── Aggregate and refresh gaps ────────────────────────────────────────────────

def aggregate_gaps(tenant_id: str) -> Dict[str, Any]:
    """
    Re-cluster unanswered questions and update gap summaries.
    Returns aggregation report.
    """
    # Get all ungrouped questions
    rows = db.session.execute(
        text(
            """
            SELECT id, question, embedding, created_at
            FROM gap_questions
            WHERE tenant_id = :tenant AND gap_id IS NULL
            ORDER BY created_at ASC
            LIMIT 500
            """
        ),
        {"tenant": tenant_id},
    ).fetchall()

    for q_id, question, embedding, _ in rows:
        # Try to find existing gap
        gap_id = _find_matching_gap(question, tenant_id, embedding)
        if not gap_id:
            gap_id = _create_gap(tenant_id, question, q_id, embedding)
        else:
            db.session.execute(
                text("UPDATE gap_questions SET gap_id = :g WHERE id = :q"),
                {"g": gap_id, "q": q_id},
            )
            db.session.execute(
                text(
                    "UPDATE information_gaps SET question_count = question_count + 1,"
                    " last_seen_at = CURRENT_TIMESTAMP WHERE id = :g"
                ),
                {"g": gap_id},
            )

    db.session.commit()

    # Update sample_questions for each gap
    gap_rows = db.session.execute(
        text(
            "SELECT id FROM information_gaps WHERE tenant_id = :t AND resolved = FALSE",
        ),
        {"t": tenant_id},
    ).fetchall()

    for (gid,) in gap_rows:
        sample_rows = db.session.execute(
            text(
                "SELECT question FROM gap_questions WHERE gap_id = :g "
                "ORDER BY created_at DESC LIMIT 5"
            ),
            {"g": gid},
        ).fetchall()
        samples = [r[0] for r in sample_rows]
        from models import InformationGap
        ig = InformationGap.query.get(gid)
        if ig:
            ig.sample_questions = samples

    db.session.commit()
    return {"tenant_id": tenant_id, "gaps_processed": len(gap_rows)}


# ── Gap report ────────────────────────────────────────────────────────────────

def get_gap_report(
    tenant_id: str,
    min_count: int = 1,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """Return information gap report for a tenant."""
    rows = db.session.execute(
        text(
            """
            SELECT id, cluster_label, question_count, sample_questions,
                   category, severity, first_seen_at, last_seen_at, resolved
            FROM information_gaps
            WHERE tenant_id = :tenant
              AND question_count >= :min_count
              AND resolved = FALSE
            ORDER BY question_count DESC, last_seen_at DESC
            LIMIT :lim
            """
        ),
        {"tenant": tenant_id, "min_count": min_count, "lim": limit},
    ).fetchall()

    return [
        {
            "gap_id": r[0],
            "cluster_label": r[1],
            "question_count": r[2],
            "sample_questions": r[3] or [],
            "category": r[4],
            "severity": r[5],
            "first_seen": r[6].isoformat() if r[6] else None,
            "last_seen": r[7].isoformat() if r[7] else None,
            "resolved": r[8],
            "recommendation": _gap_recommendation(r[4], r[2]),
        }
        for r in rows
    ]


def get_gap_summary(tenant_id: str) -> Dict[str, Any]:
    """Summary statistics for information gaps."""
    total_gaps = db.session.execute(
        text(
            "SELECT COUNT(*), SUM(question_count) FROM information_gaps "
            "WHERE tenant_id = :t AND resolved = FALSE"
        ),
        {"t": tenant_id},
    ).fetchone()

    by_category = db.session.execute(
        text(
            """
            SELECT category, COUNT(*), SUM(question_count)
            FROM information_gaps
            WHERE tenant_id = :t AND resolved = FALSE
            GROUP BY category
            ORDER BY SUM(question_count) DESC
            """
        ),
        {"t": tenant_id},
    ).fetchall()

    return {
        "total_gaps": total_gaps[0] or 0,
        "total_unanswered_questions": int(total_gaps[1] or 0),
        "by_category": [
            {
                "category": r[0],
                "gap_clusters": r[1],
                "question_count": r[2],
            }
            for r in by_category
        ],
    }


def _gap_recommendation(category: str, count: int) -> str:
    """Generate a human-readable recommendation for the gap."""
    recs = {
        "hostel_facilities": "Upload hostel policy, accommodation guidelines, or warden contact documents.",
        "fee_structure": "Upload fee structure, payment schedules, or refund policy documents.",
        "placement": "Upload placement statistics, company list, or recruitment policy.",
        "scholarship": "Upload scholarship criteria, application procedures, or award notifications.",
        "library": "Upload library rules, catalogue, or e-resource access guides.",
        "transport": "Upload bus schedules, transport routes, or vehicle allocation policy.",
        "medical": "Upload health centre guidelines, doctor availability, or emergency procedures.",
        "sports": "Upload sports policy, facility booking procedures, or tournament schedules.",
        "clubs": "Upload student club registry, activity calendar, or club formation guidelines.",
        "research": "Upload research policy, funding guidelines, or publication norms.",
        "examination": "Upload examination schedule, grading policy, or result procedures.",
        "attendance": "Upload attendance policy, leave application procedures, or related circulars.",
        "admission": "Upload admission criteria, eligibility, or enrollment procedures.",
    }
    base = recs.get(category, "Upload relevant institutional documents for this topic.")
    if count >= 10:
        return f"HIGH PRIORITY ({count} students asked). " + base
    elif count >= 5:
        return f"MEDIUM PRIORITY ({count} students asked). " + base
    return base


def resolve_gap(gap_id: str, tenant_id: str) -> bool:
    """Mark a gap as resolved (e.g., after uploading the missing document)."""
    result = db.session.execute(
        text(
            """
            UPDATE information_gaps
            SET resolved = TRUE, resolved_at = CURRENT_TIMESTAMP
            WHERE id = :id AND tenant_id = :tenant
            """
        ),
        {"id": gap_id, "tenant": tenant_id},
    )
    db.session.commit()
    return result.rowcount > 0
