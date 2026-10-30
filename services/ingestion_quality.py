"""
Feature G: Self-Healing Document Ingestion

Turns document ingestion into a quality-controlled and recoverable pipeline.

Pipeline:
  Upload → Quality Inspection → Extraction → Validation → Chunking
         → Embedding → Indexing → Consistency Verification

Measures:
  - Text extraction coverage (chars extracted / estimated total)
  - OCR confidence per page
  - Duplicate content detection
  - Empty / malformed chunk detection
  - Page structural integrity
  - Index consistency validation
  - Failed-document quarantine
  - Per-page targeted reprocessing
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import text

from config import Config
from models import db

logger = logging.getLogger(__name__)

# ── Quality thresholds ────────────────────────────────────────────────────────
MIN_COVERAGE = float(getattr(Config, "IQ_MIN_COVERAGE", 0.5))      # 50%
MIN_AVG_CHARS_PER_PAGE = int(getattr(Config, "IQ_MIN_CHARS_PAGE", 50))
MAX_EMPTY_CHUNK_RATIO = float(getattr(Config, "IQ_MAX_EMPTY_RATIO", 0.3))
MIN_QUALITY_SCORE = float(getattr(Config, "IQ_MIN_QUALITY", 0.4))


# ── Page-level checks ─────────────────────────────────────────────────────────

def inspect_pages(pages: List[str], ocr_page_indices: List[int] = None) -> Dict[str, Any]:
    """
    Analyse extracted pages for quality issues.
    Returns per-page and aggregate quality metrics.
    """
    ocr_pages = set(ocr_page_indices or [])
    total = len(pages)
    zero_text = 0
    short_text = 0
    page_issues: Dict[int, str] = {}
    char_counts = []

    for i, page_text in enumerate(pages):
        page_num = i + 1
        char_count = len(page_text.strip())
        char_counts.append(char_count)

        if char_count == 0:
            zero_text += 1
            page_issues[page_num] = "no_text"
        elif char_count < MIN_AVG_CHARS_PER_PAGE:
            short_text += 1
            page_issues[page_num] = "short_text"
        elif _looks_garbled(page_text):
            page_issues[page_num] = "garbled"

    avg_chars = sum(char_counts) / total if total else 0
    coverage = (total - zero_text) / total if total else 0.0

    return {
        "total_pages": total,
        "zero_text_pages": zero_text,
        "short_text_pages": short_text,
        "avg_chars_per_page": round(avg_chars, 1),
        "text_coverage": round(coverage, 4),
        "page_issues": page_issues,
        "ocr_page_count": len(ocr_pages),
    }


def _looks_garbled(text: str) -> bool:
    """Heuristic: high ratio of non-printable or non-ASCII characters."""
    if not text:
        return False
    non_ascii = sum(1 for c in text if ord(c) > 127 and ord(c) < 160)
    ratio = non_ascii / max(len(text), 1)
    return ratio > 0.2


# ── Chunk-level checks ────────────────────────────────────────────────────────

def inspect_chunks(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Inspect chunks for quality issues.
    chunks: list of chunk dicts with 'text', 'token_count', etc.
    """
    total = len(chunks)
    empty = 0
    malformed = 0
    duplicate_hashes: Dict[str, int] = {}

    for c in chunks:
        text = c.get("text", "").strip()
        if not text or len(text) < 20:
            empty += 1
            continue

        # Check for malformed (too short, repeated chars, only whitespace/numbers)
        if len(set(text.split())) < 3:
            malformed += 1
            continue

        # Check for duplicate content
        h = hashlib.sha256(text.lower().encode()).hexdigest()[:16]
        duplicate_hashes[h] = duplicate_hashes.get(h, 0) + 1

    duplicate_count = sum(v - 1 for v in duplicate_hashes.values() if v > 1)
    empty_ratio = empty / total if total else 0.0

    return {
        "total_chunks": total,
        "empty_chunks": empty,
        "malformed_chunks": malformed,
        "duplicate_chunks": duplicate_count,
        "empty_ratio": round(empty_ratio, 4),
        "unique_chunk_ratio": round(
            (total - duplicate_count) / total if total else 0, 4
        ),
    }


# ── Overall quality score ─────────────────────────────────────────────────────

def compute_quality_score(
    page_metrics: Dict[str, Any],
    chunk_metrics: Dict[str, Any],
    extract_meta: Dict[str, Any],
) -> float:
    """
    Compute an overall quality score [0, 1].

    Factors:
    - Text coverage (weight 0.4)
    - Empty chunk ratio (weight 0.3) — inverted
    - OCR confidence proxy (weight 0.2) — based on OCR page ratio
    - Malformed chunk ratio (weight 0.1) — inverted
    """
    coverage = page_metrics.get("text_coverage", 0)
    empty_ratio = chunk_metrics.get("empty_ratio", 0)
    total_pages = max(page_metrics.get("total_pages", 1), 1)
    ocr_pages = page_metrics.get("ocr_page_count", 0)
    malformed = chunk_metrics.get("malformed_chunks", 0)
    total_chunks = max(chunk_metrics.get("total_chunks", 1), 1)
    malformed_ratio = malformed / total_chunks

    # OCR confidence proxy: 1.0 if no OCR needed, 0.7 if all OCR
    ocr_ratio = ocr_pages / total_pages
    ocr_confidence = 1.0 - (ocr_ratio * 0.3)

    score = (
        coverage * 0.40
        + (1.0 - empty_ratio) * 0.30
        + ocr_confidence * 0.20
        + (1.0 - malformed_ratio) * 0.10
    )
    return round(max(0.0, min(1.0, score)), 4)


# ── Index consistency check ───────────────────────────────────────────────────

def verify_index_consistency(
    document_id: str,
    tenant_id: str,
    expected_chunk_count: int,
    vector_store,
) -> Dict[str, Any]:
    """
    Verify that the number of Qdrant points matches the expected chunk count.
    Returns consistency report.
    """
    try:
        # Count chunks in Qdrant for this document
        actual_count = vector_store.count_document_chunks(document_id, tenant_id)
    except Exception as e:
        logger.warning("Index consistency check failed: %s", e)
        actual_count = -1

    is_consistent = actual_count == expected_chunk_count
    delta = actual_count - expected_chunk_count if actual_count >= 0 else None

    return {
        "document_id": document_id,
        "expected_chunks": expected_chunk_count,
        "actual_chunks": actual_count,
        "is_consistent": is_consistent,
        "delta": delta,
        "issue": None if is_consistent else (
            "missing_chunks" if actual_count < expected_chunk_count
            else "extra_chunks"
        ),
    }


# ── Quarantine ────────────────────────────────────────────────────────────────

def quarantine_document(
    document_id: str,
    tenant_id: str,
    reason: str,
) -> bool:
    """Mark a document as quarantined due to quality issues."""
    try:
        from models import IngestionQuality, Document
        iq = IngestionQuality.query.filter_by(document_id=document_id, tenant_id=tenant_id).first()
        if iq:
            iq.quarantined = True
            iq.quarantine_reason = reason
        doc = Document.query.get(document_id)
        if doc:
            doc.status = "quarantined"
            doc.error_message = f"Quarantined: {reason}"
        db.session.commit()
        logger.warning("Document %s quarantined: %s", document_id[:8], reason)
        return True
    except Exception as e:
        logger.error("Quarantine failed: %s", e)
        db.session.rollback()
        return False


# ── Save quality record ───────────────────────────────────────────────────────

def save_quality_record(
    document_id: str,
    tenant_id: str,
    page_metrics: Dict,
    chunk_metrics: Dict,
    extract_meta: Dict,
    quality_score: float,
    quarantined: bool = False,
    quarantine_reason: str = "",
) -> str:
    """Persist ingestion quality metrics. Returns quality record id."""
    qid = str(uuid.uuid4())
    try:
        from models import IngestionQuality, Document
        avg_ocr = 1.0 - (
            page_metrics.get("ocr_page_count", 0)
            / max(page_metrics.get("total_pages", 1), 1) * 0.3
        )
        iq = IngestionQuality(
            id=qid,
            document_id=document_id,
            tenant_id=tenant_id,
            text_coverage=page_metrics.get("text_coverage", 0),
            avg_ocr_confidence=avg_ocr,
            zero_text_pages=page_metrics.get("zero_text_pages", 0),
            total_pages=page_metrics.get("total_pages", 0),
            duplicate_chunk_count=chunk_metrics.get("duplicate_chunks", 0),
            empty_chunk_count=chunk_metrics.get("empty_chunks", 0),
            malformed_chunk_count=chunk_metrics.get("malformed_chunks", 0),
            ocr_pages=extract_meta.get("ocr_pages", []),
            page_issues=page_metrics.get("page_issues", {}),
            overall_quality_score=quality_score,
            quarantined=quarantined,
            quarantine_reason=quarantine_reason,
        )
        db.session.merge(iq)
        doc = Document.query.get(document_id)
        if doc:
            doc.ingestion_quality_id = qid
        db.session.commit()
        logger.info(
            "Quality record saved for doc %s: score=%.2f", document_id[:8], quality_score
        )
    except Exception as e:
        logger.error("Quality save failed: %s", e)
        db.session.rollback()
    return qid


# ── Full quality inspection ───────────────────────────────────────────────────

def inspect_document(
    document_id: str,
    tenant_id: str,
    pages: List[str],
    chunks: List[Dict],
    extract_meta: Dict,
    vector_store=None,
) -> Dict[str, Any]:
    """
    Run the complete quality inspection pipeline.
    Returns a quality report and optionally quarantines the document.
    """
    page_metrics = inspect_pages(
        pages,
        ocr_page_indices=[p - 1 for p in extract_meta.get("ocr_pages", [])],
    )
    chunk_metrics = inspect_chunks(chunks)
    quality_score = compute_quality_score(page_metrics, chunk_metrics, extract_meta)

    # Index consistency
    consistency = {}
    if vector_store:
        consistency = verify_index_consistency(
            document_id,
            tenant_id,
            len(chunks),
            vector_store,
        )

    # Quarantine decision
    quarantined = False
    quarantine_reason = ""
    if quality_score < MIN_QUALITY_SCORE:
        quarantine_reason = f"Quality score {quality_score:.2f} below threshold {MIN_QUALITY_SCORE}"
        quarantined = True
    elif page_metrics["text_coverage"] < MIN_COVERAGE and page_metrics["total_pages"] > 3:
        quarantine_reason = f"Text coverage {page_metrics['text_coverage']:.0%} below minimum"
        quarantined = True

    qid = save_quality_record(
        document_id,
        tenant_id,
        page_metrics,
        chunk_metrics,
        extract_meta,
        quality_score,
        quarantined,
        quarantine_reason,
    )

    if quarantined:
        quarantine_document(document_id, tenant_id, quarantine_reason)

    return {
        "quality_id": qid,
        "document_id": document_id,
        "quality_score": quality_score,
        "page_metrics": page_metrics,
        "chunk_metrics": chunk_metrics,
        "consistency": consistency,
        "quarantined": quarantined,
        "quarantine_reason": quarantine_reason,
        "passes_threshold": quality_score >= MIN_QUALITY_SCORE,
    }


# ── Get quality reports ───────────────────────────────────────────────────────

def get_quality_report(
    tenant_id: str,
    include_quarantined: bool = True,
    limit: int = 50,
) -> List[Dict]:
    """Get ingestion quality reports for a tenant."""
    quarantine_filter = "" if include_quarantined else "AND iq.quarantined = FALSE"
    rows = db.session.execute(
        text(
            f"""
            SELECT iq.id, iq.document_id, d.original_name,
                   iq.text_coverage, iq.avg_ocr_confidence,
                   iq.zero_text_pages, iq.total_pages,
                   iq.duplicate_chunk_count, iq.empty_chunk_count,
                   iq.overall_quality_score, iq.quarantined,
                   iq.quarantine_reason, iq.reprocess_attempts,
                   iq.created_at
            FROM ingestion_quality iq
            JOIN documents d ON d.id = iq.document_id
            WHERE iq.tenant_id = :tenant {quarantine_filter}
            ORDER BY iq.created_at DESC
            LIMIT :lim
            """
        ),
        {"tenant": tenant_id, "lim": limit},
    ).fetchall()

    return [
        {
            "quality_id": r[0],
            "document_id": r[1],
            "filename": r[2],
            "text_coverage": r[3],
            "avg_ocr_confidence": r[4],
            "zero_text_pages": r[5],
            "total_pages": r[6],
            "duplicate_chunks": r[7],
            "empty_chunks": r[8],
            "quality_score": r[9],
            "quarantined": r[10],
            "quarantine_reason": r[11],
            "reprocess_attempts": r[12],
            "created_at": r[13].isoformat() if hasattr(r[13], "isoformat") else (str(r[13]) if r[13] else None),
        }
        for r in rows
    ]


def get_quarantined_documents(tenant_id: str) -> List[Dict]:
    """Return documents in quarantine for a tenant."""
    return get_quality_report(tenant_id, include_quarantined=True, limit=100)


def reprocess_document(
    document_id: str,
    tenant_id: str,
    vector_store,
) -> Dict[str, Any]:
    """
    Trigger reprocessing of a quarantined/failed document.
    Increments reprocess_attempts counter and re-runs ingestion.
    """
    try:
        # Get document record
        doc_row = db.session.execute(
            text("SELECT filename FROM documents WHERE id = :id AND tenant_id = :tenant"),
            {"id": document_id, "tenant": tenant_id},
        ).fetchone()

        if not doc_row:
            return {"error": "document not found"}

        # Increment attempt counter
        db.session.execute(
            text(
                """
                UPDATE ingestion_quality
                SET reprocess_attempts = reprocess_attempts + 1,
                    last_attempt_at = CURRENT_TIMESTAMP
                WHERE document_id = :doc AND tenant_id = :tenant
                """
            ),
            {"doc": document_id, "tenant": tenant_id},
        )
        # Reset document status
        db.session.execute(
            text(
                "UPDATE documents SET status = 'pending', error_message = NULL "
                "WHERE id = :id"
            ),
            {"id": document_id},
        )
        db.session.commit()

        # Re-run ingest
        from pathlib import Path as _P
        from services.ingest import ingest_file as _ingest

        doc_path = Config.UPLOAD_DIR / doc_row[0]
        if not doc_path.exists():
            doc_path = next(Config.UPLOAD_DIR.glob(f"*_{doc_row[0]}"), None)

        if doc_path and doc_path.exists():
            from models import Document
            doc_obj = Document.query.get(document_id)
            if doc_obj:
                result = _ingest(
                    doc_path, tenant_id,
                    original_name=doc_obj.original_name,
                    source=doc_obj.source,
                    vector_store=vector_store,
                )
                return {"status": result.status, "document_id": document_id}

        return {"error": "document file not found for reprocessing"}

    except Exception as e:
        logger.exception("Reprocess failed: %s", e)
        db.session.rollback()
        return {"error": str(e)}
