"""
Feature N: Document Trust Scoring

Estimates the reliability and authority of retrieved documents using a
transparent, multi-factor model. Each factor is individually exposed
so the score is explainable — not an opaque AI confidence number.

Trust factors:
  source_authority       — corpus (admin-uploaded) > upload (user)
  verification_status    — admin-approved documents score higher
  pub_date_recency       — newer = higher trust (with decay)
  effective_validity     — currently valid > expired/superseded
  supersession_status    — superseded = lower trust
  extraction_quality     — quality score from ingestion_quality table
  metadata_completeness  — has pub_date, valid_from, title, etc.
  admin_approval         — explicit admin approval flag in doc.meta

Trust is used as a secondary ranking signal in retrieval, not as a
replacement for relevance.
"""
from __future__ import annotations

import json
import logging
import math
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import text

from models import db

logger = logging.getLogger(__name__)

# ── Factor weights (must sum to 1.0) ─────────────────────────────────────────
WEIGHTS = {
    "source_authority": 0.15,
    "verification_status": 0.20,
    "pub_date_recency": 0.15,
    "effective_validity": 0.20,
    "supersession_status": 0.10,
    "extraction_quality": 0.10,
    "metadata_completeness": 0.05,
    "admin_approval": 0.05,
}

assert abs(sum(WEIGHTS.values()) - 1.0) < 0.001, "Trust factor weights must sum to 1"


# ── Individual factor scorers ─────────────────────────────────────────────────

def _score_source_authority(source: str) -> float:
    """corpus > upload > unknown."""
    return {"corpus": 1.0, "upload": 0.75, "api": 0.65}.get(source, 0.5)


def _score_verification_status(meta: Dict) -> float:
    """Admin-verified documents score higher."""
    if meta.get("verified") or meta.get("admin_approved"):
        return 1.0
    if meta.get("reviewed"):
        return 0.75
    return 0.5


def _score_pub_date_recency(pub_date) -> float:
    """
    Recency decay: score = exp(-age_years / 5).
    A document 5 years old scores ~0.37; 10 years old ~0.14.
    """
    if not pub_date:
        return 0.5  # Unknown date → neutral
    try:
        if isinstance(pub_date, str):
            pub_date = date.fromisoformat(pub_date)
        age_days = (date.today() - pub_date).days
        age_years = max(age_days / 365, 0)
        return round(math.exp(-age_years / 5), 4)
    except Exception:
        return 0.5


def _score_effective_validity(valid_from, valid_until, superseded_by) -> float:
    """Currently valid documents score highest."""
    if superseded_by:
        return 0.2  # Superseded

    today = date.today()
    if valid_from:
        try:
            vf = date.fromisoformat(str(valid_from))
            if vf > today:
                return 0.3  # Not yet effective
        except ValueError:
            pass

    if valid_until:
        try:
            vu = date.fromisoformat(str(valid_until))
            if vu < today:
                return 0.2  # Expired
        except ValueError:
            pass

    return 1.0  # Active


def _score_supersession_status(superseded_by) -> float:
    return 0.1 if superseded_by else 1.0


def _score_extraction_quality(document_id: str) -> float:
    """Retrieve from ingestion_quality table."""
    try:
        row = db.session.execute(
            text(
                "SELECT overall_quality_score FROM ingestion_quality "
                "WHERE document_id = :doc ORDER BY created_at DESC LIMIT 1"
            ),
            {"doc": document_id},
        ).fetchone()
        if row:
            return float(row[0])
    except Exception:
        pass
    return 0.7  # Default neutral


def _score_metadata_completeness(doc_row: Any) -> float:
    """Check how many important metadata fields are populated."""
    fields = [
        doc_row.pub_date if hasattr(doc_row, "pub_date") else None,
        doc_row.valid_from if hasattr(doc_row, "valid_from") else None,
        doc_row.original_name if hasattr(doc_row, "original_name") else None,
    ]
    populated = sum(1 for f in fields if f is not None)
    return populated / len(fields)


def _score_metadata_completeness_dict(d: Dict) -> float:
    fields = [d.get("pub_date"), d.get("valid_from"), d.get("original_name")]
    populated = sum(1 for f in fields if f)
    return populated / len(fields)


def _score_admin_approval(meta: Dict) -> float:
    return 1.0 if meta.get("admin_approved") or meta.get("approved") else 0.5


# ── Main trust computation ────────────────────────────────────────────────────

def compute_trust_score(document_id: str) -> Dict[str, Any]:
    """
    Compute the full trust score for a document.
    Returns {score: float, factors: {factor_name: {score, weight, contribution}}}
    """
    # Load document
    row = db.session.execute(
        text(
            """
            SELECT id, source, original_name, pub_date, valid_from,
                   valid_until, superseded_by, meta, trust_score, trust_factors
            FROM documents
            WHERE id = :id
            """
        ),
        {"id": document_id},
    ).fetchone()

    if not row:
        return {"score": 0.5, "factors": {}, "error": "document not found"}

    (
        doc_id, source, name, pub_date, valid_from,
        valid_until, superseded_by, meta, old_score, old_factors,
    ) = row
    meta = meta or {}

    # Compute each factor
    raw_factors = {
        "source_authority": _score_source_authority(source or "upload"),
        "verification_status": _score_verification_status(meta),
        "pub_date_recency": _score_pub_date_recency(pub_date),
        "effective_validity": _score_effective_validity(
            valid_from, valid_until, superseded_by
        ),
        "supersession_status": _score_supersession_status(superseded_by),
        "extraction_quality": _score_extraction_quality(document_id),
        "metadata_completeness": _score_metadata_completeness_dict(
            {
                "pub_date": pub_date,
                "valid_from": valid_from,
                "original_name": name,
            }
        ),
        "admin_approval": _score_admin_approval(meta),
    }

    # Weighted sum
    total_score = sum(
        raw_factors[f] * WEIGHTS[f] for f in WEIGHTS
    )
    total_score = round(max(0.0, min(1.0, total_score)), 4)

    # Build factor detail
    factors = {
        f: {
            "score": round(raw_factors[f], 4),
            "weight": WEIGHTS[f],
            "contribution": round(raw_factors[f] * WEIGHTS[f], 4),
            "label": _factor_label(f, raw_factors[f]),
        }
        for f in WEIGHTS
    }

    # Persist updated trust score
    try:
        db.session.execute(
            text(
                "UPDATE documents SET trust_score = :score, "
                "trust_factors = :factors::jsonb WHERE id = :id"
            ),
            {
                "score": total_score,
                "factors": json.dumps(factors),
                "id": document_id,
            },
        )
        db.session.commit()
    except Exception as e:
        logger.debug("Trust score persist failed: %s", e)
        db.session.rollback()

    return {
        "document_id": document_id,
        "filename": name,
        "score": total_score,
        "factors": factors,
        "interpretation": _interpret_score(total_score),
    }


def _factor_label(factor_name: str, score: float) -> str:
    """Human-readable label for a factor score."""
    labels = {
        "source_authority": {0.0: "Unknown source", 0.5: "User upload", 1.0: "Official corpus"},
        "verification_status": {0.0: "Unverified", 0.5: "Unverified", 1.0: "Admin-verified"},
        "effective_validity": {0.1: "Superseded", 0.2: "Expired", 1.0: "Currently valid"},
        "supersession_status": {0.1: "Superseded by newer doc", 1.0: "Not superseded"},
    }
    if factor_name in labels:
        thresholds = sorted(labels[factor_name].items())
        for thresh, label in thresholds:
            if score <= thresh + 0.05:
                return label
        return thresholds[-1][1]

    # Generic
    if score >= 0.8:
        return "Good"
    elif score >= 0.5:
        return "Fair"
    else:
        return "Poor"


def _interpret_score(score: float) -> str:
    if score >= 0.8:
        return "High trust — authoritative, current, verified source"
    elif score >= 0.6:
        return "Moderate trust — mostly reliable, some gaps"
    elif score >= 0.4:
        return "Low trust — may be outdated or unverified"
    else:
        return "Very low trust — superseded, expired, or poor quality"


# ── Batch / retrieval helpers ─────────────────────────────────────────────────

def compute_trust_for_tenant(tenant_id: str) -> Dict[str, Any]:
    """Recompute trust scores for all documents in a tenant."""
    rows = db.session.execute(
        text("SELECT id FROM documents WHERE tenant_id = :t AND status = 'ready'"),
        {"t": tenant_id},
    ).fetchall()

    updated = 0
    for (doc_id,) in rows:
        try:
            compute_trust_score(doc_id)
            updated += 1
        except Exception as e:
            logger.debug("Trust compute failed for %s: %s", doc_id[:8], e)

    return {"updated": updated, "tenant_id": tenant_id}


def apply_trust_reranking(
    passages: List[Dict[str, Any]],
    trust_weight: float = 0.15,
) -> List[Dict[str, Any]]:
    """
    Re-score passages by blending RRF score with document trust score.
    trust_weight: how much weight to give trust vs retrieval score (0–1).
    """
    if not passages or trust_weight <= 0:
        return passages

    # Batch load trust scores
    doc_ids = list({p.get("document_id", "") for p in passages if p.get("document_id")})
    if not doc_ids:
        return passages

    trust_rows = db.session.execute(
        text(
            "SELECT id, trust_score FROM documents WHERE id = ANY(:ids)"
        ),
        {"ids": doc_ids},
    ).fetchall()
    trust_map = {r[0]: float(r[1] or 0.5) for r in trust_rows}

    for p in passages:
        rrf_score = float(p.get("score", 0))
        trust = trust_map.get(p.get("document_id", ""), 0.5)
        blended = (1 - trust_weight) * rrf_score + trust_weight * trust
        p["trust_score"] = round(trust, 4)
        p["blended_score"] = round(blended, 4)

    return sorted(passages, key=lambda x: x.get("blended_score", 0), reverse=True)


def get_trust_scores(tenant_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Get all trust scores for a tenant's documents."""
    rows = db.session.execute(
        text(
            """
            SELECT id, original_name, trust_score, trust_factors,
                   superseded_by, pub_date, status
            FROM documents
            WHERE tenant_id = :tenant AND status = 'ready'
            ORDER BY trust_score DESC NULLS LAST
            LIMIT :lim
            """
        ),
        {"tenant": tenant_id, "lim": limit},
    ).fetchall()

    return [
        {
            "document_id": r[0],
            "filename": r[1],
            "trust_score": r[2],
            "trust_factors": r[3],
            "superseded": r[4] is not None,
            "pub_date": str(r[5]) if r[5] else None,
            "interpretation": _interpret_score(float(r[2] or 0.5)),
        }
        for r in rows
    ]
