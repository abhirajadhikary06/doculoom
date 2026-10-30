"""
Feature L: Knowledge Expiration Engine

Identifies information that may no longer be reliable or applicable.

Tracks:
  - Explicit expiry dates from document metadata
  - Effective period end dates
  - Supersession status
  - Missing review dates
  - Documents requiring manual review

Expiry status categories:
  active       — valid and within effective period
  expired      — past explicit expiry date
  superseded   — replaced by a newer document
  pending_review — no review date set or overdue
  unknown      — no temporal metadata available
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import text

from models import db

logger = logging.getLogger(__name__)

# Documents without review dates older than this are flagged
REVIEW_OVERDUE_DAYS = int(getattr(__import__("config").Config,
                                   "EXPIRY_REVIEW_DAYS", 365))


# ── Create / update expiration record ────────────────────────────────────────

def upsert_expiration(
    document_id: str,
    tenant_id: str,
    expiry_date: Optional[str] = None,
    review_date: Optional[str] = None,
    effective_period_end: Optional[str] = None,
    expiry_source: str = "metadata",
    notes: str = "",
) -> str:
    """Create or update an expiration record for a document."""
    # Check if record exists
    existing = db.session.execute(
        text(
            "SELECT id FROM knowledge_expiration "
            "WHERE document_id = :doc AND tenant_id = :tenant"
        ),
        {"doc": document_id, "tenant": tenant_id},
    ).fetchone()

    if existing:
        db.session.execute(
            text(
                """
                UPDATE knowledge_expiration
                SET expiry_date = :exp, review_date = :rev,
                    effective_period_end = :eff, expiry_source = :src,
                    notes = :notes, updated_at = CURRENT_TIMESTAMP
                WHERE document_id = :doc AND tenant_id = :tenant
                """
            ),
            {
                "exp": expiry_date,
                "rev": review_date,
                "eff": effective_period_end,
                "src": expiry_source,
                "notes": notes,
                "doc": document_id,
                "tenant": tenant_id,
            },
        )
        db.session.commit()
        return existing[0]

    eid = str(uuid.uuid4())
    status = compute_expiry_status(document_id, tenant_id, expiry_date, effective_period_end)
    db.session.execute(
        text(
            """
            INSERT INTO knowledge_expiration (
                id, document_id, tenant_id, expiry_date, review_date,
                effective_period_end, expiry_status, expiry_source, notes
            ) VALUES (
                :id, :doc, :tenant, :exp, :rev,
                :eff, :status, :src, :notes
            )
            """
        ),
        {
            "id": eid,
            "doc": document_id,
            "tenant": tenant_id,
            "exp": expiry_date,
            "rev": review_date,
            "eff": effective_period_end,
            "status": status,
            "src": expiry_source,
            "notes": notes,
        },
    )
    db.session.commit()
    return eid


def compute_expiry_status(
    document_id: str,
    tenant_id: str,
    expiry_date: Optional[str],
    effective_period_end: Optional[str],
) -> str:
    """Determine the expiry status for a document."""
    today = date.today()

    # Check if superseded
    sup_row = db.session.execute(
        text("SELECT superseded_by FROM documents WHERE id = :id"),
        {"id": document_id},
    ).fetchone()
    if sup_row and sup_row[0]:
        return "superseded"

    # Explicit expiry date
    if expiry_date:
        try:
            exp = date.fromisoformat(str(expiry_date))
            if exp < today:
                return "expired"
        except ValueError:
            pass

    # Effective period end
    if effective_period_end:
        try:
            eff = date.fromisoformat(str(effective_period_end))
            if eff < today:
                return "expired"
        except ValueError:
            pass

    return "active"


def refresh_expiry_statuses(tenant_id: str) -> Dict[str, int]:
    """
    Recompute expiry statuses for all documents in a tenant.
    Returns counts per status.
    """
    rows = db.session.execute(
        text(
            """
            SELECT ke.id, ke.document_id, ke.expiry_date, ke.effective_period_end,
                   d.superseded_by, d.created_at
            FROM knowledge_expiration ke
            JOIN documents d ON d.id = ke.document_id
            WHERE ke.tenant_id = :tenant
            """
        ),
        {"tenant": tenant_id},
    ).fetchall()

    today = date.today()
    counts: Dict[str, int] = {}

    for rec_id, doc_id, exp_date, eff_end, superseded_by, created_at in rows:
        # Superseded check
        if superseded_by:
            new_status = "superseded"
        elif exp_date and date.fromisoformat(str(exp_date)) < today:
            new_status = "expired"
        elif eff_end and date.fromisoformat(str(eff_end)) < today:
            new_status = "expired"
        elif created_at:
            # Overdue for review?
            age = (datetime.now(timezone.utc) - created_at).days
            if age > REVIEW_OVERDUE_DAYS:
                new_status = "pending_review"
            else:
                new_status = "active"
        else:
            new_status = "unknown"

        db.session.execute(
            text(
                "UPDATE knowledge_expiration SET expiry_status = :s, "
                "updated_at = CURRENT_TIMESTAMP WHERE id = :id"
            ),
            {"s": new_status, "id": rec_id},
        )
        counts[new_status] = counts.get(new_status, 0) + 1

    db.session.commit()
    return counts


def create_expiration_records_from_temporal(
    document_id: str,
    tenant_id: str,
    date_info: Dict[str, Any],
) -> Optional[str]:
    """
    Auto-create expiration record from temporal extraction results.
    Called during ingestion.
    """
    valid_until = date_info.get("valid_until")
    pub_date = date_info.get("pub_date")

    # Estimate review date: 1 year after publication
    review_date = None
    if pub_date:
        try:
            pub = date.fromisoformat(str(pub_date))
            review_date = str(pub.replace(year=pub.year + 1))
        except ValueError:
            pass

    return upsert_expiration(
        document_id,
        tenant_id,
        expiry_date=valid_until,
        review_date=review_date,
        effective_period_end=valid_until,
        expiry_source="temporal_extraction",
    )


# ── Query expiration status ───────────────────────────────────────────────────

def get_expiry_report(
    tenant_id: str,
    statuses: Optional[List[str]] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """Get expiration report for a tenant, optionally filtered by status."""
    status_filter = ""
    params: Dict[str, Any] = {"tenant": tenant_id, "lim": limit}

    if statuses:
        placeholders = ", ".join(f":s{i}" for i in range(len(statuses)))
        status_filter = f"AND ke.expiry_status IN ({placeholders})"
        for i, s in enumerate(statuses):
            params[f"s{i}"] = s

    rows = db.session.execute(
        text(
            f"""
            SELECT ke.id, ke.document_id, d.original_name,
                   ke.expiry_date, ke.review_date, ke.effective_period_end,
                   ke.expiry_status, ke.expiry_source, ke.admin_notified,
                   ke.notes, ke.updated_at, d.superseded_by
            FROM knowledge_expiration ke
            JOIN documents d ON d.id = ke.document_id
            WHERE ke.tenant_id = :tenant {status_filter}
            ORDER BY
                CASE ke.expiry_status
                    WHEN 'expired' THEN 1
                    WHEN 'superseded' THEN 2
                    WHEN 'pending_review' THEN 3
                    WHEN 'unknown' THEN 4
                    ELSE 5 END,
                ke.expiry_date ASC NULLS LAST
            LIMIT :lim
            """
        ),
        params,
    ).fetchall()

    return [
        {
            "expiry_id": r[0],
            "document_id": r[1],
            "filename": r[2],
            "expiry_date": str(r[3]) if r[3] else None,
            "review_date": str(r[4]) if r[4] else None,
            "effective_period_end": str(r[5]) if r[5] else None,
            "expiry_status": r[6],
            "expiry_source": r[7],
            "admin_notified": r[8],
            "notes": r[9],
            "updated_at": r[10].isoformat() if r[10] else None,
            "superseded_by": r[11],
        }
        for r in rows
    ]


def get_expiry_summary(tenant_id: str) -> Dict[str, Any]:
    """Get summary counts of expiry statuses."""
    rows = db.session.execute(
        text(
            """
            SELECT expiry_status, COUNT(*)
            FROM knowledge_expiration
            WHERE tenant_id = :tenant
            GROUP BY expiry_status
            """
        ),
        {"tenant": tenant_id},
    ).fetchall()

    summary = {r[0]: r[1] for r in rows}

    # Also count documents with no expiration record
    doc_count = db.session.execute(
        text(
            "SELECT COUNT(*) FROM documents WHERE tenant_id = :t AND status = 'ready'"
        ),
        {"t": tenant_id},
    ).scalar() or 0

    tracked = sum(summary.values())
    summary["no_record"] = max(0, doc_count - tracked)
    summary["total_documents"] = doc_count

    return summary


def mark_admin_notified(expiry_id: str, tenant_id: str) -> bool:
    result = db.session.execute(
        text(
            """
            UPDATE knowledge_expiration SET admin_notified = TRUE
            WHERE id = :id AND tenant_id = :tenant
            """
        ),
        {"id": expiry_id, "tenant": tenant_id},
    )
    db.session.commit()
    return result.rowcount > 0


def get_documents_needing_review(tenant_id: str) -> List[Dict[str, Any]]:
    """Return documents requiring admin review (expired, superseded, or overdue)."""
    return get_expiry_report(
        tenant_id,
        statuses=["expired", "superseded", "pending_review", "unknown"],
        limit=100,
    )
