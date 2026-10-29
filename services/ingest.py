"""Document ingestion: extract → chunk → embed → store (Postgres + Qdrant)."""
from __future__ import annotations
import logging
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
    path = Path(path)
    original_name = original_name or path.name
    checksum = sha256_file(path)

    # Dedup per tenant
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
        full_text, pages, extract_meta = extract_document(path)
        if not full_text.strip():
            doc.status = "failed"
            doc.error_message = "No extractable text (even after OCR)"
            db.session.commit()
            return doc

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
            meta = ChunkMeta(
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
            )
            db.session.add(meta)

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
            original_name,
            doc.page_count,
            n,
            tenant_id,
        )
        return doc
    except Exception as e:
        logger.exception("Ingest failed for %s", path)
        doc.status = "failed"
        doc.error_message = str(e)[:500]
        db.session.commit()
        return doc


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
                    # Minimal path: write temp and reuse or inline
                    from services.chunking import recursive_chunk
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
                    chunks = recursive_chunk(text, chunk_size=Config.CHUNK_SIZE, chunk_overlap=Config.CHUNK_OVERLAP)
                    vs = vector_store or VectorStore()
                    payloads = []
                    for c in chunks:
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
                    db.session.commit()
                results.append(doc)
            except Exception as e:
                logger.warning("Corpus ingest skip %s: %s", p, e)
    return results
