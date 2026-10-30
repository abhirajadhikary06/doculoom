"""
Feature A: Temporal Knowledge Engine

Makes documents time-aware by extracting publication/effective dates,
tracking version relationships, resolving conflicts between old and new
versions, and supporting historical "as_of" queries.
"""
from __future__ import annotations

import hashlib
import logging
import re
import uuid
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import text

from models import db, Document

logger = logging.getLogger(__name__)

# ── Regex patterns for date extraction ───────────────────────────────────────

_DATE_PATTERNS = [
    # ISO: 2024-08-15
    (r"\b(\d{4})-(\d{2})-(\d{2})\b", "%Y-%m-%d"),
    # Indian/international: 15/08/2024 or 15-08-2024
    (r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b", None),
    # Long form: August 15, 2024 / 15 August 2024
    (
        r"\b(\d{1,2})\s+(January|February|March|April|May|June|July|August"
        r"|September|October|November|December),?\s+(\d{4})\b",
        None,
    ),
    (
        r"\b(January|February|March|April|May|June|July|August"
        r"|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})\b",
        None,
    ),
    # Year only: "Academic Year 2024-25" → extract 2024
    (r"\b(?:academic year|ay|batch|session)\s+(\d{4})", None),
]

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}

_EFFECTIVE_KEYWORDS = [
    "effective from", "effective date", "w.e.f", "with effect from",
    "applicable from", "valid from", "comes into effect",
]
_EXPIRY_KEYWORDS = [
    "valid until", "valid till", "expires on", "expiry date",
    "effective until", "effective till", "upto", "up to",
]
_SUPERSESSION_KEYWORDS = [
    "supersedes", "replaces", "in lieu of", "cancels", "revokes",
    "amends", "overrides", "annuls",
]


# ── Date parsing helpers ──────────────────────────────────────────────────────

def _parse_date_str(s: str) -> Optional[date]:
    """Try to parse a raw date string into a date object."""
    s = s.strip()
    # ISO
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    # Long form
    for fmt in ("%d %B %Y", "%B %d %Y", "%B %d, %Y", "%d %B, %Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    return None


def extract_dates_from_text(text: str) -> Dict[str, Any]:
    """
    Extract dates from document text.
    Returns dict with:
      - all_dates: sorted list of all detected date strings
      - pub_date: best guess at publication date
      - valid_from: effective-from date if found
      - valid_until: expiry date if found
      - supersession_hints: text snippets suggesting supersession
    """
    results: Dict[str, Any] = {
        "all_dates": [],
        "pub_date": None,
        "valid_from": None,
        "valid_until": None,
        "supersession_hints": [],
    }

    text_lower = text.lower()

    # Collect all date matches with context
    date_contexts: List[Tuple[date, str, int]] = []  # (date, context, pos)

    # ISO and numeric patterns
    for pattern, fmt in _DATE_PATTERNS[:3]:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            pos = m.start()
            context = text[max(0, pos - 80):pos + 80]
            raw = m.group(0)
            parsed = _parse_date_str(raw)
            if parsed and 1990 <= parsed.year <= 2035:
                date_contexts.append((parsed, context, pos))

    # Long form patterns
    for m in re.finditer(
        r"\b(\d{1,2})\s+(January|February|March|April|May|June|July|August"
        r"|September|October|November|December),?\s+(\d{4})\b",
        text, re.IGNORECASE,
    ):
        day, month_name, year = int(m.group(1)), m.group(2), int(m.group(3))
        month = _MONTHS.get(month_name.lower(), 0)
        if month and 1990 <= year <= 2035:
            try:
                d = date(year, month, day)
                context = text[max(0, m.start() - 80):m.start() + 80]
                date_contexts.append((d, context, m.start()))
            except ValueError:
                pass

    for m in re.finditer(
        r"\b(January|February|March|April|May|June|July|August"
        r"|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})\b",
        text, re.IGNORECASE,
    ):
        month_name, day, year = m.group(1), int(m.group(2)), int(m.group(3))
        month = _MONTHS.get(month_name.lower(), 0)
        if month and 1990 <= year <= 2035:
            try:
                d = date(year, month, day)
                context = text[max(0, m.start() - 80):m.start() + 80]
                date_contexts.append((d, context, m.start()))
            except ValueError:
                pass

    if not date_contexts:
        return results

    results["all_dates"] = sorted(set(str(d) for d, _, _ in date_contexts))

    # Classify dates using surrounding context
    for d, ctx, pos in date_contexts:
        ctx_lower = ctx.lower()

        # Effective / valid from
        if any(kw in ctx_lower for kw in _EFFECTIVE_KEYWORDS):
            existing_vf = results["valid_from"]
            if existing_vf is None:
                results["valid_from"] = str(d)
            else:
                try:
                    existing_date = date.fromisoformat(str(existing_vf))
                    if d > existing_date:
                        results["valid_from"] = str(d)
                except ValueError:
                    results["valid_from"] = str(d)

        # Expiry / valid until
        if any(kw in ctx_lower for kw in _EXPIRY_KEYWORDS):
            existing_vu = results["valid_until"]
            if existing_vu is None:
                results["valid_until"] = str(d)
            else:
                try:
                    existing_date = date.fromisoformat(str(existing_vu))
                    if d < existing_date:
                        results["valid_until"] = str(d)
                except ValueError:
                    results["valid_until"] = str(d)

    # Supersession hints
    for kw in _SUPERSESSION_KEYWORDS:
        for m in re.finditer(re.escape(kw), text_lower):
            snippet = text[max(0, m.start() - 20):m.end() + 120].strip()
            results["supersession_hints"].append(snippet)

    # Best pub_date: if no valid_from found, use the most recent date
    if not results["pub_date"]:
        if date_contexts:
            # Prefer the earliest meaningful date (publication vs future effective)
            sorted_dates = sorted(date_contexts, key=lambda x: x[0])
            # Pick the latest date that's not obviously a future expiry
            candidates = [d for d, ctx, _ in sorted_dates
                          if d.year >= 2000]
            if candidates:
                results["pub_date"] = str(candidates[-1])

    return results


# ── Version management ────────────────────────────────────────────────────────

def compute_content_hash(text: str) -> str:
    """SHA-256 of normalised text (for version detection)."""
    normalized = re.sub(r"\s+", " ", text).strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def register_document_version(
    document_id: str,
    tenant_id: str,
    content_text: str,
    date_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Register or update the version record for a document.
    Returns the version record as a dict.
    """
    content_hash = compute_content_hash(content_text)
    date_info = date_info or {}

    # Check if this exact content already has a version record
    existing = db.session.execute(
        text(
            "SELECT id, version_number FROM document_versions "
            "WHERE document_id = :doc_id AND content_hash = :hash"
        ),
        {"doc_id": document_id, "hash": content_hash},
    ).fetchone()

    if existing:
        logger.debug("Version already registered for doc %s", document_id)
        return {
            "id": existing[0],
            "version_number": existing[1],
            "content_hash": content_hash,
            "is_new": False,
        }

    # Get current max version for this document
    row = db.session.execute(
        text(
            "SELECT MAX(version_number) FROM document_versions "
            "WHERE document_id = :doc_id"
        ),
        {"doc_id": document_id},
    ).fetchone()
    next_version = (row[0] or 0) + 1

    version_id = str(uuid.uuid4())
    pub_date = date_info.get("pub_date")
    valid_from = date_info.get("valid_from")
    valid_until = date_info.get("valid_until")

    db.session.execute(
        text(
            """
            INSERT INTO document_versions
                (id, document_id, tenant_id, version_number, content_hash,
                 pub_date, valid_from, valid_until, extracted_dates, raw_meta)
            VALUES
                (:id, :doc_id, :tenant_id, :version, :hash,
                 :pub_date, :valid_from, :valid_until, :dates::jsonb, :meta::jsonb)
            """
        ),
        {
            "id": version_id,
            "doc_id": document_id,
            "tenant_id": tenant_id,
            "version": next_version,
            "hash": content_hash,
            "pub_date": pub_date,
            "valid_from": valid_from,
            "valid_until": valid_until,
            "dates": str(date_info.get("all_dates", [])).replace("'", '"'),
            "meta": "{}",
        },
    )

    # Also update the documents table with temporal metadata
    update_fields = {"version": next_version}
    if pub_date:
        update_fields["pub_date"] = pub_date
    if valid_from:
        update_fields["valid_from"] = valid_from
    if valid_until:
        update_fields["valid_until"] = valid_until

    set_clause = ", ".join(f"{k} = :{k}" for k in update_fields)
    update_fields["doc_id"] = document_id
    db.session.execute(
        text(f"UPDATE documents SET {set_clause} WHERE id = :doc_id"),
        update_fields,
    )

    db.session.commit()
    logger.info(
        "Registered version %d for document %s (hash=%s)",
        next_version,
        document_id,
        content_hash[:16],
    )
    return {
        "id": version_id,
        "version_number": next_version,
        "content_hash": content_hash,
        "is_new": True,
        "pub_date": pub_date,
        "valid_from": valid_from,
        "valid_until": valid_until,
    }


# ── Supersession detection ────────────────────────────────────────────────────

def detect_supersession(
    new_document_id: str,
    tenant_id: str,
    new_text: str,
    new_date_info: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Compare new document against existing documents to find supersessions.
    Uses entity overlap + temporal signals + supersession keywords.
    Returns list of supersession candidates.
    """
    candidates = []

    # Get supersession hints from text
    hints = new_date_info.get("supersession_hints", [])
    if hints:
        logger.info(
            "Supersession hints found in doc %s: %d hints",
            new_document_id,
            len(hints),
        )

    # Look for documents with similar entity overlap (same topic)
    # We use a lightweight keyword approach: extract key policy terms
    policy_terms = _extract_policy_terms(new_text)
    if not policy_terms:
        return []

    # Query existing ready documents in the same tenant
    existing_docs = db.session.execute(
        text(
            """
            SELECT id, original_name, pub_date, valid_from, valid_until, meta
            FROM documents
            WHERE tenant_id = :tenant_id
              AND status = 'ready'
              AND id != :new_doc_id
            ORDER BY created_at DESC
            LIMIT 50
            """
        ),
        {"tenant_id": tenant_id, "new_doc_id": new_document_id},
    ).fetchall()

    new_valid_from = new_date_info.get("valid_from")
    new_pub_date = new_date_info.get("pub_date")
    new_year = None
    if new_valid_from:
        try:
            new_year = int(str(new_valid_from)[:4])
        except Exception:
            pass
    elif new_pub_date:
        try:
            new_year = int(str(new_pub_date)[:4])
        except Exception:
            pass

    for row in existing_docs:
        old_id, old_name, old_pub, old_from, old_until, old_meta = row
        if old_id == new_document_id:
            continue

        # Check name similarity (same base document different year)
        name_sim = _name_similarity(new_document_id, str(old_name or ""))

        # Check temporal: new doc's valid_from > old doc's valid_from → likely supersedes
        temporal_supersession = False
        if new_year and old_from:
            old_year = int(str(old_from)[:4])
            if new_year > old_year:
                temporal_supersession = True
        elif new_year and old_pub:
            old_year = int(str(old_pub)[:4])
            if new_year > old_year:
                temporal_supersession = True

        # Check explicit supersession keyword mentioning old doc name
        explicit = any(
            _normalize(str(old_name or ""))[:20] in _normalize(h)
            for h in hints
        )

        confidence = 0.0
        if explicit:
            confidence = 0.9
        elif temporal_supersession and name_sim > 0.5:
            confidence = 0.7
        elif temporal_supersession:
            confidence = 0.4

        if confidence > 0.35:
            candidates.append(
                {
                    "old_document_id": old_id,
                    "new_document_id": new_document_id,
                    "relationship": "supersedes",
                    "confidence": confidence,
                    "reason": "explicit" if explicit else "temporal",
                }
            )

    return sorted(candidates, key=lambda x: x["confidence"], reverse=True)


def record_supersession(
    tenant_id: str,
    old_document_id: str,
    new_document_id: str,
    relationship: str = "supersedes",
    evidence: Optional[Dict] = None,
) -> str:
    """Persist a supersession relationship. Returns supersession record id."""
    sup_id = str(uuid.uuid4())
    import json

    db.session.execute(
        text(
            """
            INSERT INTO document_supersessions
                (id, tenant_id, old_document_id, new_document_id, relationship, evidence)
            VALUES (:id, :tenant, :old, :new, :rel, :ev::jsonb)
            ON CONFLICT DO NOTHING
            """
        ),
        {
            "id": sup_id,
            "tenant": tenant_id,
            "old": old_document_id,
            "new": new_document_id,
            "rel": relationship,
            "ev": json.dumps(evidence or {}),
        },
    )
    # Mark old document as superseded
    db.session.execute(
        text("UPDATE documents SET superseded_by = :new WHERE id = :old"),
        {"new": new_document_id, "old": old_document_id},
    )
    db.session.commit()
    logger.info(
        "Recorded supersession: %s → %s", old_document_id[:8], new_document_id[:8]
    )
    return sup_id


# ── Historical / as_of retrieval filter ──────────────────────────────────────

def get_valid_document_ids(
    tenant_id: str,
    as_of: Optional[str] = None,
) -> Optional[List[str]]:
    """
    Returns document IDs valid as_of a given date.
    If as_of is None, returns documents valid today (not superseded, not expired).
    Returns None to mean "all documents" (no filter).
    """
    if as_of:
        try:
            as_of_date = date.fromisoformat(as_of)
        except ValueError:
            logger.warning("Invalid as_of date: %s", as_of)
            as_of_date = date.today()
    else:
        as_of_date = date.today()

    as_of_str = str(as_of_date)

    rows = db.session.execute(
        text(
            """
            SELECT id FROM documents
            WHERE tenant_id = :tenant_id
              AND status = 'ready'
              AND (valid_from IS NULL OR valid_from <= :as_of)
              AND (valid_until IS NULL OR valid_until >= :as_of)
              AND (superseded_by IS NULL OR :current = FALSE)
            """
        ),
        {
            "tenant_id": tenant_id,
            "as_of": as_of_str,
            "current": as_of == None,
        },
    ).fetchall()

    if not rows:
        # Fall back to all ready documents (no date metadata available)
        return None

    return [r[0] for r in rows]


def get_document_timeline(
    document_name_pattern: str,
    tenant_id: str,
) -> List[Dict[str, Any]]:
    """
    Get the full version timeline for documents matching a name pattern.
    Returns list of {document_id, version, pub_date, valid_from, valid_until,
    superseded_by, filename, created_at}.
    """
    rows = db.session.execute(
        text(
            """
            SELECT d.id, d.original_name, d.pub_date, d.valid_from,
                   d.valid_until, d.superseded_by, d.version, d.created_at,
                   dv.content_hash
            FROM documents d
            LEFT JOIN document_versions dv ON dv.document_id = d.id
                AND dv.version_number = d.version
            WHERE d.tenant_id = :tenant_id
              AND d.status = 'ready'
              AND LOWER(d.original_name) LIKE LOWER(:pattern)
            ORDER BY COALESCE(d.valid_from, d.pub_date, d.created_at) ASC
            """
        ),
        {"tenant_id": tenant_id, "pattern": f"%{document_name_pattern}%"},
    ).fetchall()

    return [
        {
            "document_id": r[0],
            "filename": r[1],
            "pub_date": str(r[2]) if r[2] else None,
            "valid_from": str(r[3]) if r[3] else None,
            "valid_until": str(r[4]) if r[4] else None,
            "superseded_by": r[5],
            "version": r[6],
            "created_at": r[7].isoformat() if r[7] else None,
            "content_hash": r[8],
            "is_current": r[5] is None,
        }
        for r in rows
    ]


def get_supersession_chain(document_id: str, tenant_id: str) -> List[Dict]:
    """Walk the supersession chain from oldest to newest for a document."""
    chain = []
    seen = set()
    current_id = document_id

    # Walk backwards to find root (oldest)
    while current_id and current_id not in seen:
        seen.add(current_id)
        row = db.session.execute(
            text(
                "SELECT id, original_name, pub_date, valid_from, created_at "
                "FROM documents WHERE id = :id AND tenant_id = :tenant"
            ),
            {"id": current_id, "tenant": tenant_id},
        ).fetchone()
        if row:
            chain.append({
                "document_id": row[0],
                "filename": row[1],
                "pub_date": str(row[2]) if row[2] else None,
                "valid_from": str(row[3]) if row[3] else None,
                "created_at": row[4].isoformat() if row[4] else None,
            })
        # Find what superseded this
        sup = db.session.execute(
            text(
                "SELECT new_document_id FROM document_supersessions "
                "WHERE old_document_id = :id AND tenant_id = :tenant "
                "ORDER BY detected_at DESC LIMIT 1"
            ),
            {"id": current_id, "tenant": tenant_id},
        ).fetchone()
        current_id = sup[0] if sup else None

    return chain


# ── Conflict resolution ───────────────────────────────────────────────────────

def resolve_temporal_conflict(
    passages: List[Dict[str, Any]],
    as_of: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Given retrieved passages, resolve conflicts when multiple versions of the
    same document exist. Prefer the version valid for the as_of date.
    When as_of is None (current), prefer the most recently effective document.
    """
    if not passages:
        return passages

    # Group passages by document_id
    by_doc: Dict[str, List[Dict]] = {}
    for p in passages:
        did = p.get("document_id", "")
        by_doc.setdefault(did, []).append(p)

    as_of_date = None
    if as_of:
        try:
            as_of_date = date.fromisoformat(as_of)
        except ValueError:
            pass

    # For each document, check if it has been superseded
    filtered = []
    for doc_id, doc_passages in by_doc.items():
        row = db.session.execute(
            text(
                "SELECT superseded_by, valid_from, valid_until "
                "FROM documents WHERE id = :id"
            ),
            {"id": doc_id},
        ).fetchone()

        if not row:
            filtered.extend(doc_passages)
            continue

        superseded_by, valid_from, valid_until = row

        # If asking historically (as_of set), include even if now superseded
        if as_of_date and valid_from:
            if valid_from <= as_of_date:
                filtered.extend(doc_passages)
            # else skip — wasn't valid at that point
        elif not as_of_date:
            # Current query: skip superseded docs unless specifically requested
            if not superseded_by:
                filtered.extend(doc_passages)
            else:
                # Superseded — still include but flag it
                for p in doc_passages:
                    p = dict(p)
                    p["superseded_by"] = superseded_by
                    p["temporal_warning"] = "This document has been superseded"
                filtered.extend(doc_passages)
        else:
            filtered.extend(doc_passages)

    return filtered if filtered else passages


# ── Internal helpers ──────────────────────────────────────────────────────────

def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower())


def _name_similarity(doc_id: str, other_name: str) -> float:
    """Very simple token overlap ratio between two document names."""
    row = db.session.execute(
        text("SELECT original_name FROM documents WHERE id = :id"),
        {"id": doc_id},
    ).fetchone()
    if not row:
        return 0.0
    name_a = set(_normalize(row[0] or "").split())
    name_b = set(_normalize(other_name).split())
    if not name_a or not name_b:
        return 0.0
    intersection = name_a & name_b
    union = name_a | name_b
    return len(intersection) / len(union)


def _extract_policy_terms(text: str) -> List[str]:
    """Extract key policy terms from text for similarity matching."""
    # Simple keyword extraction: numbers, percentages, policy words
    terms = []
    # Percentages
    for m in re.finditer(r"\d+\s*%", text):
        terms.append(m.group(0).replace(" ", ""))
    # Policy keywords
    keywords = [
        "attendance", "fee", "exam", "result", "admission", "syllabus",
        "regulation", "circular", "guideline", "policy", "rule", "act",
        "semester", "year", "batch",
    ]
    text_lower = text.lower()
    terms.extend(kw for kw in keywords if kw in text_lower)
    return list(set(terms))
