"""
Feature D: Document Impact Analysis

After a new document is ingested, analyses how it affects the existing
knowledge base:
- Extracts entities (departments, courses, regulations, dates)
- Extracts relationships (modifies, supersedes, references, applies_to)
- Builds a document dependency graph in PostgreSQL
- Identifies potentially affected existing documents
- Detects possible policy conflicts (calls into contradiction.py)

Uses PostgreSQL relationships — no graph DB required.
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

# ── Entity extraction ─────────────────────────────────────────────────────────

_DEPT_PATTERNS = [
    (r"\b(CSE|Computer Science(?: and Engineering)?)\b", "department"),
    (r"\b(IT|Information Technology)\b", "department"),
    (r"\b(ECE|Electronics(?: and Communication)?)\b", "department"),
    (r"\b(EEE|Electrical(?: and Electronics)?)\b", "department"),
    (r"\b(Mechanical|MECH)\b", "department"),
    (r"\b(Civil)\b", "department"),
    (r"\b(MBA|Master of Business)\b", "program"),
    (r"\b(MCA|Master of Computer)\b", "program"),
    (r"\bB\.?Tech\b", "program"),
    (r"\bM\.?Tech\b", "program"),
    (r"\bPh\.?D\b", "program"),
]

_COURSE_PATTERNS = [
    (r"\b([A-Z]{2,4}\d{3,4})\b", "course_code"),
    (r"\bsemester\s+(\d{1,2})\b", "semester"),
    (r"\bbatch\s+(20\d{2}(?:-\d{2,4})?)\b", "batch"),
    (r"\bacademic year\s+(20\d{2}(?:-\d{2,4})?)\b", "academic_year"),
]

_REGULATION_PATTERNS = [
    (r"\bRegulation\s+(20\d{2})\b", "regulation"),
    (r"\bCircular\s+No\.?\s*[\w/-]+", "circular"),
    (r"\bOrder\s+No\.?\s*[\w/-]+", "order"),
    (r"\bNotification\s+No\.?\s*[\w/-]+", "notification"),
    (r"\bPolicy\s+(?:on|for)\s+\w[\w\s]{2,40}", "policy"),
    (r"\bGuidelines?\s+(?:for|on)\s+\w[\w\s]{2,40}", "guideline"),
    (r"\bRule\s+\d+\b", "rule"),
]

_DATE_PATTERNS = [
    (r"\b(\d{4})\b", "year"),
    (r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b", "date"),
]

_RELATIONSHIP_PATTERNS = [
    (r"supersede[sd]?\s+(?:the\s+)?(.{5,60})", "supersedes"),
    (r"replace[sd]?\s+(?:the\s+)?(.{5,60})", "supersedes"),
    (r"amend[sd]?\s+(?:the\s+)?(.{5,60})", "modifies"),
    (r"modif(?:ies|y|ied)\s+(?:the\s+)?(.{5,60})", "modifies"),
    (r"reference[sd]?\s+(?:to\s+)?(.{5,60})", "references"),
    (r"refers?\s+to\s+(.{5,60})", "references"),
    (r"appl(?:ies|y|ied)\s+to\s+(.{5,60})", "applies_to"),
    (r"applicable\s+to\s+(.{5,60})", "applies_to"),
    (r"pursuant\s+to\s+(.{5,60})", "references"),
    (r"in\s+accordance\s+with\s+(.{5,60})", "references"),
]


def extract_entities(text: str, document_id: str, filename: str) -> List[Dict]:
    """
    Extract named entities from document text.
    Returns list of entity dicts.
    """
    entities = []
    seen = set()

    def add(entity_type: str, value: str, page: int = 1, ctx: str = ""):
        key = (entity_type, value[:50].lower())
        if key in seen:
            return
        seen.add(key)
        entities.append(
            {
                "document_id": document_id,
                "entity_type": entity_type,
                "entity_value": value[:255],
                "confidence": 0.85,
                "page_ref": page,
                "context": ctx[:300],
            }
        )

    # Department / program
    for pattern, etype in _DEPT_PATTERNS:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            ctx = text[max(0, m.start() - 40):m.end() + 40]
            add(etype, m.group(0), ctx=ctx)

    # Course codes / semesters / batches
    for pattern, etype in _COURSE_PATTERNS:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            ctx = text[max(0, m.start() - 40):m.end() + 40]
            add(etype, m.group(1) if m.lastindex else m.group(0), ctx=ctx)

    # Regulations / circulars
    for pattern, etype in _REGULATION_PATTERNS:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            ctx = text[max(0, m.start() - 40):m.end() + 40]
            add(etype, m.group(0)[:120], ctx=ctx)

    # Key years (only years in range 2000-2035)
    for m in re.finditer(r"\b(20[0-3]\d)\b", text):
        ctx = text[max(0, m.start() - 40):m.end() + 40]
        add("year", m.group(1), ctx=ctx)

    return entities


def extract_relationships(
    text: str,
    source_doc_id: str,
    tenant_id: str,
) -> List[Dict]:
    """
    Extract cross-document relationship hints from text.
    Returns list of relationship candidates with target filenames/descriptions.
    """
    relationships = []

    for pattern, rel_type in _RELATIONSHIP_PATTERNS:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            target_hint = m.group(1).strip()
            # Clean up: remove trailing punctuation
            target_hint = re.sub(r"[,;.:].*$", "", target_hint).strip()
            if len(target_hint) < 4:
                continue
            relationships.append(
                {
                    "source_doc_id": source_doc_id,
                    "relationship": rel_type,
                    "target_hint": target_hint[:200],
                    "tenant_id": tenant_id,
                    "confidence": 0.7,
                }
            )

    return relationships


def resolve_relationship_targets(
    relationships: List[Dict],
    tenant_id: str,
) -> List[Dict]:
    """
    Try to resolve target_hints to actual document IDs in the tenant.
    """
    resolved = []
    all_docs = db.session.execute(
        text(
            "SELECT id, original_name FROM documents "
            "WHERE tenant_id = :tenant AND status = 'ready'"
        ),
        {"tenant": tenant_id},
    ).fetchall()

    doc_map = {row[0]: (row[1] or "").lower() for row in all_docs}

    for rel in relationships:
        hint = rel["target_hint"].lower()
        # Find best matching document
        best_match = None
        best_score = 0.0
        for doc_id, doc_name in doc_map.items():
            if doc_id == rel["source_doc_id"]:
                continue
            # Simple: check if hint words appear in doc name
            hint_words = set(re.findall(r"\w+", hint))
            doc_words = set(re.findall(r"\w+", doc_name))
            overlap = len(hint_words & doc_words)
            if overlap > best_score:
                best_score = overlap
                best_match = doc_id

        if best_match and best_score > 0:
            resolved.append(
                {
                    **rel,
                    "target_doc_id": best_match,
                    "target_doc_name": doc_map.get(best_match, ""),
                    "resolution_score": best_score,
                }
            )
        else:
            resolved.append(
                {
                    **rel,
                    "target_doc_id": None,
                    "target_doc_name": None,
                    "resolution_score": 0.0,
                }
            )

    return resolved


def save_entities(
    entities: List[Dict],
    tenant_id: str,
) -> int:
    """Persist extracted entities to document_entities table."""
    saved = 0
    for e in entities:
        try:
            eid = str(uuid.uuid4())
            db.session.execute(
                text(
                    """
                    INSERT INTO document_entities
                        (id, document_id, tenant_id, entity_type, entity_value,
                         confidence, page_ref, context)
                    VALUES (:id, :doc, :tenant, :type, :value, :conf, :page, :ctx)
                    ON CONFLICT DO NOTHING
                    """
                ),
                {
                    "id": eid,
                    "doc": e["document_id"],
                    "tenant": tenant_id,
                    "type": e["entity_type"],
                    "value": e["entity_value"],
                    "conf": e.get("confidence", 0.8),
                    "page": e.get("page_ref", 1),
                    "ctx": e.get("context", ""),
                },
            )
            saved += 1
        except Exception as ex:
            logger.debug("Entity insert error: %s", ex)
    try:
        db.session.commit()
    except Exception:
        db.session.rollback()
    return saved


def save_relationships(relationships: List[Dict]) -> int:
    """Persist resolved relationships to document_relationships table."""
    saved = 0
    for r in relationships:
        if not r.get("target_doc_id"):
            continue
        try:
            rid = str(uuid.uuid4())
            db.session.execute(
                text(
                    """
                    INSERT INTO document_relationships
                        (id, tenant_id, source_doc_id, target_doc_id,
                         relationship, confidence, evidence)
                    VALUES (:id, :tenant, :src, :tgt, :rel, :conf, :ev)
                    ON CONFLICT DO NOTHING
                    """
                ),
                {
                    "id": rid,
                    "tenant": r["tenant_id"],
                    "src": r["source_doc_id"],
                    "tgt": r["target_doc_id"],
                    "rel": r["relationship"],
                    "conf": r.get("confidence", 0.7),
                    "ev": r.get("target_hint", "")[:300],
                },
            )
            saved += 1
        except Exception as ex:
            logger.debug("Relationship insert error: %s", ex)
    try:
        db.session.commit()
    except Exception:
        db.session.rollback()
    return saved


def find_affected_documents(
    new_document_id: str,
    tenant_id: str,
    new_entities: List[Dict],
) -> List[Dict]:
    """
    Given a new document's entities, find existing documents likely affected.
    Returns list of {document_id, filename, overlap_score, shared_entities}.
    """
    if not new_entities:
        return []

    # Group new entities by type
    new_by_type: Dict[str, List[str]] = {}
    for e in new_entities:
        new_by_type.setdefault(e["entity_type"], []).append(
            e["entity_value"].lower()
        )

    # Query existing documents' entities
    rows = db.session.execute(
        text(
            """
            SELECT de.document_id, de.entity_type, de.entity_value,
                   d.original_name
            FROM document_entities de
            JOIN documents d ON d.id = de.document_id
            WHERE de.tenant_id = :tenant
              AND de.document_id != :new_id
              AND d.status = 'ready'
            """
        ),
        {"tenant": tenant_id, "new_id": new_document_id},
    ).fetchall()

    # Aggregate by document
    doc_entities: Dict[str, Dict] = {}
    for doc_id, etype, evalue, fname in rows:
        if doc_id not in doc_entities:
            doc_entities[doc_id] = {"filename": fname, "entities": {}}
        doc_entities[doc_id]["entities"].setdefault(etype, []).append(
            evalue.lower()
        )

    affected = []
    for doc_id, info in doc_entities.items():
        shared = []
        score = 0
        for etype, new_vals in new_by_type.items():
            old_vals = info["entities"].get(etype, [])
            overlap = set(new_vals) & set(old_vals)
            if overlap:
                shared.extend([f"{etype}:{v}" for v in overlap])
                # Weight by entity importance
                weight = {"regulation": 3, "policy": 3, "department": 2,
                          "program": 2, "circular": 3, "percentage": 2}.get(etype, 1)
                score += len(overlap) * weight

        if score > 0:
            affected.append(
                {
                    "document_id": doc_id,
                    "filename": info["filename"],
                    "overlap_score": score,
                    "shared_entities": shared[:10],
                }
            )

    return sorted(affected, key=lambda x: x["overlap_score"], reverse=True)[:20]


def run_impact_analysis(
    document_id: str,
    tenant_id: str,
    document_text: str,
    document_filename: str,
    document_date_info: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Full impact analysis pipeline for a newly ingested document.
    Returns an impact report dict.
    """
    logger.info("Running impact analysis for doc %s", document_id[:8])

    # 1. Extract entities
    entities = extract_entities(document_text, document_id, document_filename)
    entity_count = save_entities(entities, tenant_id)

    # 2. Extract relationships
    raw_relationships = extract_relationships(document_text, document_id, tenant_id)
    resolved_rels = resolve_relationship_targets(raw_relationships, tenant_id)
    rel_count = save_relationships(resolved_rels)

    # 3. Find affected documents
    affected = find_affected_documents(document_id, tenant_id, entities)

    # 4. Detect conflicts via contradiction module
    from services.contradiction import (
        detect_conflicts_for_document,
        save_conflicts,
        save_document_claims,
        extract_factual_claims,
    )
    claims = extract_factual_claims(document_text, document_id, document_filename)
    save_document_claims(document_id, tenant_id, claims)
    conflicts = detect_conflicts_for_document(
        document_id, tenant_id, document_text, document_filename
    )
    conflict_count = save_conflicts(tenant_id, conflicts)

    # 5. Supersession detection
    supersessions = []
    if document_date_info:
        from services.temporal import detect_supersession, record_supersession
        candidates = detect_supersession(
            document_id, tenant_id, document_text, document_date_info
        )
        for c in candidates:
            if c["confidence"] >= 0.65:
                record_supersession(
                    tenant_id,
                    c["old_document_id"],
                    document_id,
                    evidence={"confidence": c["confidence"], "reason": c["reason"]},
                )
                supersessions.append(c)

    # 6. Build summary
    report = {
        "document_id": document_id,
        "tenant_id": tenant_id,
        "entities_extracted": entity_count,
        "entity_types": list({e["entity_type"] for e in entities}),
        "relationships_found": rel_count,
        "affected_documents": affected,
        "affected_count": len(affected),
        "conflicts_detected": conflict_count,
        "high_severity_conflicts": sum(
            1 for c in conflicts if c.get("severity") in ("high", "critical")
        ),
        "supersessions": supersessions,
        "supersession_count": len(supersessions),
        "claims_indexed": len(claims),
    }

    # Persist impact summary to documents.impact_analysis
    try:
        from models import Document
        doc_obj = Document.query.get(document_id)
        if doc_obj:
            doc_obj.impact_analysis = report
            db.session.commit()
    except Exception as e:
        logger.warning("Impact analysis persist failed: %s", e)
        db.session.rollback()

    logger.info(
        "Impact analysis: %d entities, %d affected docs, %d conflicts",
        entity_count,
        len(affected),
        conflict_count,
    )
    return report


def get_document_dependency_graph(
    tenant_id: str,
    limit: int = 100,
) -> Dict[str, Any]:
    """
    Build and return the document dependency graph for a tenant.
    Nodes = documents, edges = relationships.
    """
    # Nodes
    doc_rows = db.session.execute(
        text(
            """
            SELECT id, original_name, status, pub_date, superseded_by,
                   trust_score
            FROM documents
            WHERE tenant_id = :tenant AND status = 'ready'
            LIMIT :lim
            """
        ),
        {"tenant": tenant_id, "lim": limit},
    ).fetchall()

    nodes = [
        {
            "id": r[0],
            "label": r[1] or r[0][:16],
            "status": r[2],
            "pub_date": str(r[3]) if r[3] else None,
            "superseded": r[4] is not None,
            "trust_score": r[5],
        }
        for r in doc_rows
    ]

    # Edges from document_relationships
    rel_rows = db.session.execute(
        text(
            """
            SELECT source_doc_id, target_doc_id, relationship, confidence
            FROM document_relationships
            WHERE tenant_id = :tenant
            LIMIT :lim
            """
        ),
        {"tenant": tenant_id, "lim": limit * 3},
    ).fetchall()

    # Edges from supersessions
    sup_rows = db.session.execute(
        text(
            """
            SELECT old_document_id, new_document_id, relationship
            FROM document_supersessions
            WHERE tenant_id = :tenant
            LIMIT :lim
            """
        ),
        {"tenant": tenant_id, "lim": limit},
    ).fetchall()

    edges = [
        {
            "source": r[0],
            "target": r[1],
            "type": r[2],
            "confidence": r[3],
        }
        for r in rel_rows
    ]
    for r in sup_rows:
        edges.append({"source": r[0], "target": r[1], "type": r[2], "confidence": 1.0})

    return {
        "nodes": nodes,
        "edges": edges,
        "node_count": len(nodes),
        "edge_count": len(edges),
    }
