"""
Document ingestion: extract → quality-inspect → chunk → embed → store
(Postgres + Qdrant), with integrated quality inspection, temporal analysis,
impact analysis, contradiction detection, and cache invalidation.
"""
from __future__ import annotations
import logging
import threading
import uuid
from pathlib import Path
from typing import Optional

from config import Config
from models import db, Document, ChunkMeta
from services.chunking import recursive_chunk, build_page_map_from_pages
from services.pdf_extract import extract_document, sha256_file
from services.vectorstore import VectorStore

logger = logging.getLogger(__name__)


def ingest_file(
    path: Path,
    tenant_id: str,
    *,
    original_name: Optional[str] = None,
    source: str = "upload",
    vector_store: Optional[VectorStore] = None,
) -> Document:
    """
    Full ingestion pipeline:
    Upload → Quality Inspection → Extraction → Validation →
    Chunking → Embedding → Indexing → Consistency Verification →
    Temporal Analysis → Impact Analysis
    """
    path = Path(path)
    original_name = original_name or path.name
    checksum = sha256_file(path)

    # ── Dedup per tenant ──────────────────────────────────────────────────────
    existing = Document.query.filter_by(
        tenant_id=tenant_id, checksum=checksum, status="ready"
    ).first()
    if existing:
        logger.info("Document already ingested (checksum match): %s", existing.id)
        return existing

    doc = Document(
        id=str(uuid.uuid4()),
        tenant_id=tenant_id,
        filename=path.name,
        original_name=original_name,
        size_bytes=path.stat().st_size,
        status="processing",
        source=source,
        checksum=checksum,
        meta={},
    )
    db.session.add(doc)
    db.session.commit()

    try:
        # ── Extract ───────────────────────────────────────────────────────────
        full_text, pages, extract_meta = extract_document(path)
        if not full_text.strip():
            doc.status = "failed"
            doc.error_message = "No extractable text (even after OCR)"
            db.session.commit()
            return doc

        # ── Chunk ─────────────────────────────────────────────────────────────
        page_map = build_page_map_from_pages(pages)
        chunks = recursive_chunk(
            full_text,
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            page_map=page_map,
        )

        vs = vector_store or VectorStore()
        chunk_payloads = []
        for c in chunks:
            cid = str(uuid.uuid4())
            chunk_payloads.append(
                {
                    "id": cid,
                    "text": c.text,
                    "chunk_index": c.chunk_index,
                    "page_start": c.page_start,
                    "page_end": c.page_end,
                    "char_start": c.char_start,
                    "char_end": c.char_end,
                    "token_count": c.token_count,
                }
            )
            db.session.add(ChunkMeta(
                id=cid,
                document_id=doc.id,
                tenant_id=tenant_id,
                chunk_index=c.chunk_index,
                page_start=c.page_start,
                page_end=c.page_end,
                char_start=c.char_start,
                char_end=c.char_end,
                token_count=c.token_count,
                text_preview=c.text[:200],
            ))

        # ── Index ─────────────────────────────────────────────────────────────
        n = vs.upsert_chunks(
            chunk_payloads,
            tenant_id=tenant_id,
            document_id=doc.id,
            filename=original_name,
        )

        doc.page_count = len(pages)
        doc.chunk_count = n
        doc.status = "ready"
        doc.meta = extract_meta
        db.session.commit()

        logger.info(
            "Ingested %s → %d pages, %d chunks (tenant=%s)",
            original_name, doc.page_count, n, tenant_id,
        )

        # ── Post-ingest pipelines (best-effort, non-blocking) ─────────────────
        _run_post_ingest_pipelines(
            doc_id=doc.id,
            tenant_id=tenant_id,
            full_text=full_text,
            pages=pages,
            chunks=chunk_payloads,
            extract_meta=extract_meta,
            original_name=original_name,
            vector_store=vs,
            background=Config.IMPACT_ANALYSIS_BACKGROUND,
        )

        return doc

    except Exception as e:
        logger.exception("Ingest failed for %s", path)
        doc.status = "failed"
        doc.error_message = str(e)[:500]
        db.session.commit()
        return doc


def _run_post_ingest_pipelines(
    *,
    doc_id: str,
    tenant_id: str,
    full_text: str,
    pages: list,
    chunks: list,
    extract_meta: dict,
    original_name: str,
    vector_store,
    background: bool = True,
):
    """
    Run all post-ingest analysis pipelines.
    These are best-effort — failures don't affect document status.
    """
    app_obj = None
    try:
        from flask import current_app
        app_obj = current_app._get_current_object()
    except Exception:
        try:
            from app import app as flask_app
            app_obj = flask_app
        except Exception:
            app_obj = None

    def _run():
        try:
            if app_obj:
                with app_obj.app_context():
                    _execute_post_ingest(
                        doc_id, tenant_id, full_text, pages, chunks,
                        extract_meta, original_name, vector_store,
                    )
            else:
                _execute_post_ingest(
                    doc_id, tenant_id, full_text, pages, chunks,
                    extract_meta, original_name, vector_store,
                )
        except Exception as ex:
            logger.warning("Post-ingest execution error: %s", ex)

    if background:
        t = threading.Thread(target=_run, daemon=True)
        t.start()
    else:
        _run()


def _execute_post_ingest(
    doc_id, tenant_id, full_text, pages, chunks,
    extract_meta, original_name, vector_store,
):
    """Actual post-ingest logic (runs in thread or inline)."""

    # ── G: Ingestion Quality Inspection ──────────────────────────────────────
    try:
        from services.ingestion_quality import inspect_document
        inspect_document(
            document_id=doc_id,
            tenant_id=tenant_id,
            pages=pages,
            chunks=chunks,
            extract_meta=extract_meta,
            vector_store=vector_store,
        )
    except Exception as e:
        logger.warning("Quality inspection failed: %s", e)

    # ── A: Temporal Analysis ──────────────────────────────────────────────────
    date_info = {}
    try:
        from services.temporal import (
            extract_dates_from_text,
            register_document_version,
        )
        date_info = extract_dates_from_text(full_text)
        register_document_version(doc_id, tenant_id, full_text, date_info)
    except Exception as e:
        logger.warning("Temporal analysis failed: %s", e)

    # ── L: Knowledge Expiration ───────────────────────────────────────────────
    try:
        from services.expiration import create_expiration_records_from_temporal
        create_expiration_records_from_temporal(doc_id, tenant_id, date_info)
    except Exception as e:
        logger.warning("Expiration record creation failed: %s", e)

    # ── D: Document Impact Analysis (includes K: contradiction detection) ─────
    if Config.IMPACT_ANALYSIS_ENABLED:
        try:
            from services.impact import run_impact_analysis
            run_impact_analysis(
                document_id=doc_id,
                tenant_id=tenant_id,
                document_text=full_text,
                document_filename=original_name,
                document_date_info=date_info,
            )
        except Exception as e:
            logger.warning("Impact analysis failed: %s", e)

    # ── N: Trust Score Computation ────────────────────────────────────────────
    try:
        from services.trust import compute_trust_score
        compute_trust_score(doc_id)
    except Exception as e:
        logger.warning("Trust score computation failed: %s", e)

    # ── F: Invalidate semantic cache ──────────────────────────────────────────
    try:
        from services.semantic_cache import invalidate_for_document
        invalidate_for_document(doc_id, tenant_id)
    except Exception as e:
        logger.warning("Cache invalidation failed: %s", e)

    # ── M: Trigger drift detection if enabled ─────────────────────────────────
    if Config.DRIFT_ON_INGEST:
        try:
            from services.drift import run_drift_evaluation
            run_drift_evaluation(tenant_id, vector_store, trigger_event="post_ingest")
        except Exception as e:
            logger.warning("Drift evaluation failed: %s", e)

    logger.info("Post-ingest pipelines complete for doc %s", doc_id[:8])


def ingest_corpus(tenant_id: str, vector_store: Optional[VectorStore] = None) -> list:
    """Auto-ingest every PDF/txt under CORPUS_DIR at startup."""
    Config.ensure_dirs()
    results = []
    for p in sorted(Config.CORPUS_DIR.rglob("*")):
        if p.suffix.lower() not in {".pdf", ".txt", ".md"}:
            continue
        if p.is_file():
            try:
                if p.suffix.lower() == ".pdf":
                    doc = ingest_file(p, tenant_id, source="corpus", vector_store=vector_store)
                else:
                    # Plain text
                    text = p.read_text(encoding="utf-8", errors="ignore")
                    checksum = sha256_file(p) if p.exists() else ""
                    existing = Document.query.filter_by(
                        tenant_id=tenant_id, checksum=checksum, status="ready"
                    ).first()
                    if existing:
                        results.append(existing)
                        continue
                    doc = Document(
                        id=str(uuid.uuid4()),
                        tenant_id=tenant_id,
                        filename=p.name,
                        original_name=p.name,
                        size_bytes=p.stat().st_size,
                        status="processing",
                        source="corpus",
                        checksum=checksum,
                    )
                    db.session.add(doc)
                    db.session.commit()
                    chunks_objs = recursive_chunk(
                        text,
                        chunk_size=Config.CHUNK_SIZE,
                        chunk_overlap=Config.CHUNK_OVERLAP,
                    )
                    vs = vector_store or VectorStore()
                    payloads = []
                    for c in chunks_objs:
                        cid = str(uuid.uuid4())
                        payloads.append({
                            "id": cid, "text": c.text, "chunk_index": c.chunk_index,
                            "page_start": 1, "page_end": 1,
                            "char_start": c.char_start, "char_end": c.char_end,
                            "token_count": c.token_count,
                        })
                        db.session.add(ChunkMeta(
                            id=cid, document_id=doc.id, tenant_id=tenant_id,
                            chunk_index=c.chunk_index, page_start=1, page_end=1,
                            char_start=c.char_start, char_end=c.char_end,
                            token_count=c.token_count, text_preview=c.text[:200],
                        ))
                    n = vs.upsert_chunks(payloads, tenant_id, doc.id, p.name)
                    doc.page_count = 1
                    doc.chunk_count = n
                    doc.status = "ready"
                    doc.meta = {"extraction": "text"}
                    db.session.commit()
                    # Post-ingest pipelines for corpus text files
                    _run_post_ingest_pipelines(
                        doc_id=doc.id, tenant_id=tenant_id,
                        full_text=text, pages=[text], chunks=payloads,
                        extract_meta={"extraction": "text", "page_count": 1,
                                      "ocr_pages": [], "image_ocr_pages": []},
                        original_name=p.name, vector_store=vs, background=False,
                    )
                results.append(doc)
            except Exception as e:
                logger.warning("Corpus ingest skip %s: %s", p, e)
    return results
