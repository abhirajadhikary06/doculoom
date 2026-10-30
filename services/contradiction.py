"""
Feature K: Contradiction Discovery

Proactively detects conflicting information across documents.
Extracts factual claims (percentages, dates, policy statements) and compares
them across documents in the same tenant.

Distinguishes:
- Genuine contradictions (same period, conflicting facts)
- Temporal changes (different periods, different values)
- Unresolved conflicts (same topic, unclear which is authoritative)
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import text

from models import db

logger = logging.getLogger(__name__)


# ── Claim extraction ──────────────────────────────────────────────────────────

_NUMERIC_PATTERNS = [
    # Percentages: "75%", "75 percent"
    (r"(\d+(?:\.\d+)?)\s*(?:%|percent(?:age)?)", "percentage"),
    # Dates with context
    (r"(?:by|before|after|on|from)\s+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", "deadline"),
    (
        r"(?:by|before|after|on|from)\s+((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
        r"\S*\s+\d{1,2},?\s+\d{4})",
        "deadline",
    ),
    # Counts / limits: "3 attempts", "2 years"
    (r"(\d+)\s+(attempt|year|semester|credit|hour|day|week|month)s?", "count"),
    # Fees: "Rs. 500", "₹1000"
    (r"(?:Rs\.?|INR|₹)\s*(\d[\d,]*)", "amount"),
]

_POLICY_SENTENCES = [
    # attendance
    r"attendance\s+(?:requirement|criteria|policy|rule)[^.]*\.",
    r"(?:minimum|mandatory|compulsory)\s+attendance[^.]*\.",
    r"attendance\s+of\s+\d+[^.]*\.",
    # fees
    r"fee[s]?\s+(?:must|shall|should|to be)[^.]*\.",
    r"(?:tuition|exam|registration)\s+fee[^.]*\.",
    # eligibility
    r"(?:eligible|eligibility)[^.]*\.",
    r"(?:minimum|qualifying)\s+marks[^.]*\.",
    # deadlines
    r"(?:last date|deadline|due date|submission)[^.]*\.",
]


def extract_factual_claims(
    text: str,
    document_id: str,
    filename: str,
    page: int = 1,
) -> List[Dict[str, Any]]:
    """
    Extract factual/policy claims from document text.
    Returns list of {claim_text, claim_type, value, document_id, filename, page}.
    """
    claims = []

    # Numeric fact claims
    for pattern, claim_type in _NUMERIC_PATTERNS:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            start = max(0, m.start() - 80)
            end = min(len(text), m.end() + 80)
            context = text[start:end].strip()
            # Normalize context to one sentence
            sentences = re.split(r"[.!?\n]", context)
            for s in sentences:
                if m.group(0).lower() in s.lower():
                    claims.append(
                        {
                            "claim_text": s.strip(),
                            "claim_type": claim_type,
                            "value": m.group(1) if len(m.groups()) >= 1 else m.group(0),
                            "raw_match": m.group(0),
                            "document_id": document_id,
                            "filename": filename,
                            "page": page,
                        }
                    )
                    break

    # Policy sentence claims
    for pattern in _POLICY_SENTENCES:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            claim_text = m.group(0).strip()
            if len(claim_text) < 20:
                continue
            claims.append(
                {
                    "claim_text": claim_text,
                    "claim_type": "policy",
                    "value": None,
                    "raw_match": claim_text,
                    "document_id": document_id,
                    "filename": filename,
                    "page": page,
                }
            )

    # Deduplicate on claim_text
    seen = set()
    unique = []
    for c in claims:
        key = c["claim_text"][:80]
        if key not in seen:
            seen.add(key)
            unique.append(c)

    return unique


# ── Conflict detection ────────────────────────────────────────────────────────

def _normalize_claim(claim: str) -> str:
    return re.sub(r"[^a-z0-9%]", " ", claim.lower())


def _claims_conflict(claim_a: Dict, claim_b: Dict) -> Tuple[bool, str, str]:
    """
    Compare two claims of the same type.
    Returns (is_conflict, conflict_type, description).
    """
    # Same claim type?
    if claim_a["claim_type"] != claim_b["claim_type"]:
        return False, "", ""

    claim_type = claim_a["claim_type"]

    # Percentage comparison
    if claim_type == "percentage":
        try:
            val_a = float(str(claim_a.get("value", "0")).replace(",", ""))
            val_b = float(str(claim_b.get("value", "0")).replace(",", ""))
            if abs(val_a - val_b) > 0.01:
                desc = (
                    f"Attendance/percentage conflict: "
                    f"{val_a}% vs {val_b}%"
                )
                return True, "percentage_conflict", desc
        except (ValueError, TypeError):
            pass

    # Amount comparison
    elif claim_type == "amount":
        try:
            val_a = float(str(claim_a.get("value", "0")).replace(",", ""))
            val_b = float(str(claim_b.get("value", "0")).replace(",", ""))
            if abs(val_a - val_b) > 0.01:
                desc = f"Amount conflict: {val_a} vs {val_b}"
                return True, "amount_conflict", desc
        except (ValueError, TypeError):
            pass

    # Policy semantic conflict: keyword-based
    elif claim_type == "policy":
        # Check if both mention the same topic but with different values/negations
        text_a = _normalize_claim(claim_a["claim_text"])
        text_b = _normalize_claim(claim_b["claim_text"])
        # Topic overlap
        tokens_a = set(text_a.split())
        tokens_b = set(text_b.split())
        overlap = len(tokens_a & tokens_b) / max(len(tokens_a | tokens_b), 1)
        if overlap > 0.4:
            # High overlap — likely same topic
            # Check for negation patterns
            neg_a = any(w in text_a for w in ["not", "no", "without", "exempt"])
            neg_b = any(w in text_b for w in ["not", "no", "without", "exempt"])
            if neg_a != neg_b:
                return True, "policy_negation_conflict", (
                    "One policy affirms while the other negates the same rule"
                )
            # Check for percentage mismatch within policy text
            pcts_a = re.findall(r"\d+", text_a)
            pcts_b = re.findall(r"\d+", text_b)
            if pcts_a and pcts_b and pcts_a != pcts_b:
                return True, "policy_value_conflict", (
                    f"Policy values differ: {pcts_a} vs {pcts_b}"
                )

    return False, "", ""


def _is_temporal_conflict(
    doc_id_a: str,
    doc_id_b: str,
) -> bool:
    """
    Check if the conflict is temporal (one doc supersedes the other).
    """
    row = db.session.execute(
        text(
            """
            SELECT id FROM document_supersessions
            WHERE (old_document_id = :a AND new_document_id = :b)
               OR (old_document_id = :b AND new_document_id = :a)
            LIMIT 1
            """
        ),
        {"a": doc_id_a, "b": doc_id_b},
    ).fetchone()
    return row is not None


def _classify_severity(conflict_type: str, doc_temporal: bool) -> str:
    if doc_temporal:
        return "low"  # Expected temporal change
    if conflict_type in ("percentage_conflict", "amount_conflict"):
        return "high"
    if conflict_type == "policy_negation_conflict":
        return "critical"
    return "medium"


def detect_conflicts_for_document(
    document_id: str,
    tenant_id: str,
    document_text: str,
    document_filename: str,
) -> List[Dict[str, Any]]:
    """
    After ingesting a new document, find conflicts with existing documents.
    Returns list of conflict records (does NOT save them — caller persists).
    """
    new_claims = extract_factual_claims(
        document_text, document_id, document_filename
    )
    if not new_claims:
        return []

    # Get all other ready documents for this tenant
    existing_docs = db.session.execute(
        text(
            """
            SELECT id, original_name FROM documents
            WHERE tenant_id = :tenant AND status = 'ready' AND id != :doc_id
            """
        ),
        {"tenant": tenant_id, "doc_id": document_id},
    ).fetchall()

    # For each existing document, get stored entities/claims (from document_entities)
    conflicts_found = []

    for (old_doc_id, old_filename) in existing_docs:
        # We store claim_text in document_entities with entity_type = 'claim'
        old_claims_rows = db.session.execute(
            text(
                """
                SELECT entity_type, entity_value, context, page_ref
                FROM document_entities
                WHERE document_id = :doc_id
                  AND entity_type IN ('percentage', 'policy', 'amount', 'count', 'deadline')
                """
            ),
            {"doc_id": old_doc_id},
        ).fetchall()

        if not old_claims_rows:
            continue

        old_claims = [
            {
                "claim_type": r[0],
                "claim_text": r[1],
                "value": None,  # stored in claim_text
                "document_id": old_doc_id,
                "filename": old_filename,
                "page": r[3],
            }
            for r in old_claims_rows
        ]

        # Compare new claims vs old claims
        for nc in new_claims:
            for oc in old_claims:
                is_conflict, conflict_type, description = _claims_conflict(nc, oc)
                if not is_conflict:
                    continue

                is_temporal = _is_temporal_conflict(document_id, old_doc_id)
                severity = _classify_severity(conflict_type, is_temporal)

                conflicts_found.append(
                    {
                        "doc_id_a": document_id,
                        "doc_id_b": old_doc_id,
                        "conflict_type": conflict_type,
                        "severity": severity,
                        "description": description,
                        "claim_a": nc["claim_text"][:500],
                        "claim_b": oc["claim_text"][:500],
                        "page_a": nc.get("page", 1),
                        "page_b": oc.get("page", 1),
                        "is_temporal": is_temporal,
                        "filename_a": document_filename,
                        "filename_b": old_filename,
                    }
                )

    return conflicts_found


def save_conflicts(tenant_id: str, conflicts: List[Dict[str, Any]]) -> int:
    """Persist conflict records. Returns count saved."""
    saved = 0
    for c in conflicts:
        try:
            conflict_id = str(uuid.uuid4())
            db.session.execute(
                text(
                    """
                    INSERT INTO document_conflicts (
                        id, tenant_id, doc_id_a, doc_id_b, conflict_type,
                        severity, description, claim_a, claim_b,
                        page_a, page_b, is_temporal
                    ) VALUES (
                        :id, :tenant, :a, :b, :ctype,
                        :sev, :desc, :ca, :cb, :pa, :pb, :temp
                    )
                    """
                ),
                {
                    "id": conflict_id,
                    "tenant": tenant_id,
                    "a": c["doc_id_a"],
                    "b": c["doc_id_b"],
                    "ctype": c["conflict_type"],
                    "sev": c["severity"],
                    "desc": c.get("description", ""),
                    "ca": c.get("claim_a", ""),
                    "cb": c.get("claim_b", ""),
                    "pa": c.get("page_a", 1),
                    "pb": c.get("page_b", 1),
                    "temp": c.get("is_temporal", False),
                },
            )
            saved += 1
        except Exception as e:
            logger.debug("Conflict save skipped: %s", e)

    try:
        db.session.commit()
    except Exception as e:
        logger.error("Conflict batch commit failed: %s", e)
        db.session.rollback()

    return saved


def get_conflicts(
    tenant_id: str,
    resolved: bool = False,
    severity: Optional[str] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """Get conflict records for a tenant."""
    conditions = "WHERE c.tenant_id = :tenant AND c.resolved = :resolved"
    params: Dict[str, Any] = {"tenant": tenant_id, "resolved": resolved}

    if severity:
        conditions += " AND c.severity = :sev"
        params["sev"] = severity

    rows = db.session.execute(
        text(
            f"""
            SELECT c.id, c.conflict_type, c.severity, c.description,
                   c.claim_a, c.claim_b, c.page_a, c.page_b,
                   c.is_temporal, c.detected_at, c.resolved,
                   da.original_name AS fn_a, db.original_name AS fn_b
            FROM document_conflicts c
            LEFT JOIN documents da ON da.id = c.doc_id_a
            LEFT JOIN documents db ON db.id = c.doc_id_b
            {conditions}
            ORDER BY
                CASE c.severity WHEN 'critical' THEN 1 WHEN 'high' THEN 2
                                WHEN 'medium' THEN 3 ELSE 4 END,
                c.detected_at DESC
            LIMIT :lim
            """
        ),
        {**params, "lim": limit},
    ).fetchall()

    return [
        {
            "id": r[0],
            "conflict_type": r[1],
            "severity": r[2],
            "description": r[3],
            "claim_a": r[4],
            "claim_b": r[5],
            "page_a": r[6],
            "page_b": r[7],
            "is_temporal": r[8],
            "detected_at": r[9].isoformat() if r[9] else None,
            "resolved": r[10],
            "filename_a": r[11],
            "filename_b": r[12],
        }
        for r in rows
    ]


def resolve_conflict(conflict_id: str, tenant_id: str) -> bool:
    """Mark a conflict as resolved."""
    from datetime import datetime, timezone

    result = db.session.execute(
        text(
            """
            UPDATE document_conflicts
            SET resolved = TRUE, reviewed_at = CURRENT_TIMESTAMP
            WHERE id = :id AND tenant_id = :tenant
            """
        ),
        {"id": conflict_id, "tenant": tenant_id},
    )
    db.session.commit()
    return result.rowcount > 0


def save_document_claims(
    document_id: str,
    tenant_id: str,
    claims: List[Dict[str, Any]],
) -> int:
    """Persist extracted claims as document_entities for future conflict detection."""
    saved = 0
    for claim in claims:
        try:
            eid = str(uuid.uuid4())
            db.session.execute(
                text(
                    """
                    INSERT INTO document_entities
                        (id, document_id, tenant_id, entity_type, entity_value,
                         confidence, page_ref, context)
                    VALUES (:id, :doc, :tenant, :type, :value, :conf, :page, :ctx)
                    """
                ),
                {
                    "id": eid,
                    "doc": document_id,
                    "tenant": tenant_id,
                    "type": claim.get("claim_type", "policy"),
                    "value": claim.get("claim_text", "")[:500],
                    "conf": 0.8,
                    "page": claim.get("page", 1),
                    "ctx": claim.get("raw_match", "")[:300],
                },
            )
            saved += 1
        except Exception as e:
            logger.debug("Entity save error: %s", e)

    try:
        db.session.commit()
    except Exception:
        db.session.rollback()

    return saved
