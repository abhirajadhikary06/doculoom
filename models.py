"""SQLAlchemy models for multi-tenant DocuLoom (Postgres & SQLite)."""
from __future__ import annotations
import os
import uuid
from datetime import datetime, timezone

import sqlalchemy
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import (
    Index,
    JSON,
    String,
    Text,
    Integer,
    Float,
    Boolean,
    DateTime,
    Date,
    ForeignKey,
)
from sqlalchemy.dialects.postgresql import JSONB as _PG_JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.types import TypeDecorator

# Global patch for SQLAlchemy text() so raw queries work seamlessly on SQLite & PostgreSQL
_orig_text = sqlalchemy.text


def patched_text(s, *args, **kwargs):
    if isinstance(s, str):
        is_pg = "postgresql" in os.environ.get("DATABASE_URL", "").lower()
        if not is_pg:
            # SQLite compatibility: strip Postgres-specific casts and replace NOW() with CURRENT_TIMESTAMP
            s = s.replace("::jsonb", "").replace("NOW()", "CURRENT_TIMESTAMP")
        else:
            s = s.replace("NOW()", "CURRENT_TIMESTAMP")
    return _orig_text(s, *args, **kwargs)


sqlalchemy.text = patched_text
sql_text = patched_text

db = SQLAlchemy()


def utcnow():
    return datetime.now(timezone.utc)


class JSONB(TypeDecorator):
    """
    Dialect-aware JSON column.
    - PostgreSQL  → native JSONB  (indexed, fast operators)
    - SQLite/other → JSON         (development / test fallback)
    """
    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(_PG_JSONB())
        return dialect.type_descriptor(JSON())


class Tenant(db.Model):
    __tablename__ = "tenants"
    id = db.Column(String(64), primary_key=True)  # slug e.g. "campus", "dept-cse"
    name = db.Column(String(128), nullable=False)
    created_at = db.Column(DateTime(timezone=True), default=utcnow)
    settings = db.Column(JSONB, default=dict)

    users = relationship("User", back_populates="tenant")
    documents = relationship("Document", back_populates="tenant")


class User(db.Model):
    __tablename__ = "users"
    id = db.Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id = db.Column(String(64), ForeignKey("tenants.id"), nullable=False, index=True)
    email = db.Column(String(255), unique=True, nullable=False, index=True)
    password_hash = db.Column(String(255), nullable=False)
    display_name = db.Column(String(128))
    role = db.Column(String(32), default="user")  # user | admin
    credits = db.Column(Integer, default=50, nullable=False)
    max_credits = db.Column(Integer, default=50, nullable=False)
    created_at = db.Column(DateTime(timezone=True), default=utcnow)
    last_login = db.Column(DateTime(timezone=True))

    tenant = relationship("Tenant", back_populates="users")
    conversations = relationship("Conversation", back_populates="user")


class Document(db.Model):
    __tablename__ = "documents"
    id = db.Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id = db.Column(String(64), ForeignKey("tenants.id"), nullable=False, index=True)
    filename = db.Column(String(512), nullable=False)
    original_name = db.Column(String(512))
    mime_type = db.Column(String(128), default="application/pdf")
    size_bytes = db.Column(Integer, default=0)
    page_count = db.Column(Integer, default=0)
    chunk_count = db.Column(Integer, default=0)
    status = db.Column(String(32), default="pending")  # pending|processing|ready|failed|quarantined
    error_message = db.Column(Text)
    source = db.Column(String(32), default="upload")  # upload | corpus
    checksum = db.Column(String(64), index=True)  # sha256 for dedup
    meta = db.Column(JSONB, default=dict)
    created_at = db.Column(DateTime(timezone=True), default=utcnow)
    updated_at = db.Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    # V2 Intelligence engine extensions
    pub_date = db.Column(Date)
    valid_from = db.Column(Date)
    valid_until = db.Column(Date)
    version = db.Column(Integer, default=1)
    superseded_by = db.Column(String(36))
    trust_score = db.Column(Float, default=0.5)
    trust_factors = db.Column(JSONB, default=dict)
    ingestion_quality_id = db.Column(String(36))
    impact_analysis = db.Column(JSONB, default=dict)

    tenant = relationship("Tenant", back_populates="documents")
    chunks = relationship("ChunkMeta", back_populates="document", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_docs_tenant_status", "tenant_id", "status"),
        Index("ix_docs_tenant_checksum", "tenant_id", "checksum"),
    )


class ChunkMeta(db.Model):
    """Lightweight metadata; vectors live in Qdrant."""
    __tablename__ = "chunk_meta"
    id = db.Column(String(36), primary_key=True)  # same as Qdrant point id
    document_id = db.Column(String(36), ForeignKey("documents.id"), nullable=False, index=True)
    tenant_id = db.Column(String(64), nullable=False, index=True)
    chunk_index = db.Column(Integer, nullable=False)
    page_start = db.Column(Integer, default=1)
    page_end = db.Column(Integer, default=1)
    char_start = db.Column(Integer, default=0)
    char_end = db.Column(Integer, default=0)
    token_count = db.Column(Integer, default=0)
    text_preview = db.Column(String(256))  # first ~200 chars for UI
    created_at = db.Column(DateTime(timezone=True), default=utcnow)

    document = relationship("Document", back_populates="chunks")

    __table_args__ = (
        Index("ix_chunk_doc_idx", "document_id", "chunk_index"),
        Index("ix_chunk_tenant", "tenant_id"),
    )


class Conversation(db.Model):
    __tablename__ = "conversations"
    id = db.Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id = db.Column(String(64), nullable=False, index=True)
    user_id = db.Column(String(36), ForeignKey("users.id"), nullable=True, index=True)
    title = db.Column(String(256), default="New chat")
    created_at = db.Column(DateTime(timezone=True), default=utcnow)
    updated_at = db.Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    user = relationship("User", back_populates="conversations")
    messages = relationship(
        "Message", back_populates="conversation", cascade="all, delete-orphan",
        order_by="Message.created_at"
    )


class Message(db.Model):
    __tablename__ = "messages"
    id = db.Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id = db.Column(String(36), ForeignKey("conversations.id"), nullable=False, index=True)
    role = db.Column(String(16), nullable=False)  # user | assistant | system
    content = db.Column(Text, nullable=False)
    sources = db.Column(JSONB, default=list)  # structured citations
    latency_ms = db.Column(Float)
    token_usage = db.Column(JSONB, default=dict)
    created_at = db.Column(DateTime(timezone=True), default=utcnow)

    conversation = relationship("Conversation", back_populates="messages")


class EvalRun(db.Model):
    __tablename__ = "eval_runs"
    id = db.Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id = db.Column(String(64), index=True)
    metrics = db.Column(JSONB, default=dict)
    created_at = db.Column(DateTime(timezone=True), default=utcnow)


class CreditLog(db.Model):
    """Audit trail of every credit award / deduction."""
    __tablename__ = "credit_logs"
    id = db.Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    tenant_id = db.Column(String(64), nullable=False, index=True)
    amount = db.Column(Integer, nullable=False)  # +credit / -deduct
    balance_after = db.Column(Integer, nullable=False)
    reason = db.Column(String(64), nullable=False)  # signup | ask | ingest | refund | admin_grant
    meta = db.Column(JSONB, default=dict)
    created_at = db.Column(DateTime(timezone=True), default=utcnow)


class MemoryGraph(db.Model):
    """Knowledge graph extracted from conversation memory."""
    __tablename__ = "memory_graphs"
    id = db.Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id = db.Column(String(64), nullable=False, index=True)
    user_id = db.Column(String(36), ForeignKey("users.id"), nullable=True, index=True)
    conversation_id = db.Column(String(36), ForeignKey("conversations.id"), nullable=True, index=True)
    nodes = db.Column(JSONB, default=list)   # [{id, label, type}]
    edges = db.Column(JSONB, default=list)   # [{source, target, label}]
    created_at = db.Column(DateTime(timezone=True), default=utcnow)
    updated_at = db.Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


# ── V2 Intelligence Engine Models ───────────────────────────────────────────────

class DocumentVersion(db.Model):
    """Feature A: Temporal version history of documents."""
    __tablename__ = "document_versions"
    id = db.Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = db.Column(String(36), nullable=False, index=True)
    tenant_id = db.Column(String(64), nullable=False, index=True)
    version_number = db.Column(Integer, nullable=False, default=1)
    content_hash = db.Column(String(64), nullable=False, index=True)
    pub_date = db.Column(Date)
    valid_from = db.Column(Date)
    valid_until = db.Column(Date)
    extracted_dates = db.Column(JSONB, default=list)
    raw_meta = db.Column(JSONB, default=dict)
    created_at = db.Column(DateTime(timezone=True), default=utcnow)


class DocumentSupersession(db.Model):
    """Feature A: Supersession tracking between older and newer circulars."""
    __tablename__ = "document_supersessions"
    id = db.Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id = db.Column(String(64), nullable=False, index=True)
    old_document_id = db.Column(String(36), nullable=False, index=True)
    new_document_id = db.Column(String(36), nullable=False, index=True)
    relationship = db.Column(String(32), default="supersedes")
    detected_at = db.Column(DateTime(timezone=True), default=utcnow)
    evidence = db.Column(JSONB, default=dict)


class DocumentEntity(db.Model):
    """Feature D: Entities discovered across campus documents."""
    __tablename__ = "document_entities"
    id = db.Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = db.Column(String(36), nullable=False, index=True)
    tenant_id = db.Column(String(64), nullable=False, index=True)
    entity_type = db.Column(String(64), nullable=False, index=True)
    entity_value = db.Column(Text, nullable=False)
    confidence = db.Column(Float, default=1.0)
    page_ref = db.Column(Integer, default=1)
    context = db.Column(Text)
    created_at = db.Column(DateTime(timezone=True), default=utcnow)


class DocumentRelationship(db.Model):
    """Feature D: Cross-document relationships & knowledge graph links."""
    __tablename__ = "document_relationships"
    id = db.Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id = db.Column(String(64), nullable=False, index=True)
    source_doc_id = db.Column(String(36), nullable=False, index=True)
    target_doc_id = db.Column(String(36), nullable=False)
    relationship = db.Column(String(64), nullable=False)
    confidence = db.Column(Float, default=1.0)
    evidence = db.Column(Text)
    detected_at = db.Column(DateTime(timezone=True), default=utcnow)


class DocumentConflict(db.Model):
    """Feature K: Contradictions and conflicting statements between documents."""
    __tablename__ = "document_conflicts"
    id = db.Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id = db.Column(String(64), nullable=False, index=True)
    doc_id_a = db.Column(String(36), nullable=False)
    doc_id_b = db.Column(String(36), nullable=False)
    conflict_type = db.Column(String(64), nullable=False)
    severity = db.Column(String(16), default="medium")
    description = db.Column(Text)
    claim_a = db.Column(Text)
    claim_b = db.Column(Text)
    page_a = db.Column(Integer)
    page_b = db.Column(Integer)
    is_temporal = db.Column(Boolean, default=False)
    resolved = db.Column(Boolean, default=False)
    detected_at = db.Column(DateTime(timezone=True), default=utcnow)
    reviewed_at = db.Column(DateTime(timezone=True))


class QueryExecutionLog(db.Model):
    """Feature E: Full reproducibility trace for every query."""
    __tablename__ = "query_execution_logs"
    id = db.Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id = db.Column(String(64), nullable=False, index=True)
    user_id = db.Column(String(36), index=True)
    conversation_id = db.Column(String(36), index=True)
    question = db.Column(Text, nullable=False)
    search_query = db.Column(Text)
    retrieval_strategy = db.Column(String(64), default="hybrid")
    query_intent = db.Column(String(64))
    difficulty_class = db.Column(String(64))
    answer_hash = db.Column(String(64))
    answer_text = db.Column(Text)
    sources = db.Column(JSONB, default=list)
    retrieval_config = db.Column(JSONB, default=dict)
    embedding_model = db.Column(String(128))
    llm_model = db.Column(String(128))
    prompt_version = db.Column(String(32), default="v1")
    latency_ms = db.Column(Float)
    token_usage = db.Column(JSONB, default=dict)
    reranked = db.Column(Boolean, default=False)
    was_decomposed = db.Column(Boolean, default=False)
    cache_hit = db.Column(Boolean, default=False)
    as_of_date = db.Column(Date)
    created_at = db.Column(DateTime(timezone=True), default=utcnow)


class ExecutionEvidence(db.Model):
    """Feature E: Per-chunk evidence snapshot linking answer claims to chunks."""
    __tablename__ = "execution_evidence"
    id = db.Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    log_id = db.Column(String(36), nullable=False, index=True)
    tenant_id = db.Column(String(64), nullable=False, index=True)
    chunk_id = db.Column(String(36), nullable=False)
    document_id = db.Column(String(36), nullable=False)
    chunk_index = db.Column(Integer, default=0)
    page_start = db.Column(Integer, default=1)
    retrieval_rank = db.Column(Integer, nullable=False)
    dense_score = db.Column(Float)
    sparse_score = db.Column(Float)
    rrf_score = db.Column(Float)
    rerank_score = db.Column(Float)
    text_snapshot = db.Column(Text)
    doc_version = db.Column(String(64))
    created_at = db.Column(DateTime(timezone=True), default=utcnow)


class IngestionQuality(db.Model):
    """Feature G: Document quality metrics & quarantine tracking."""
    __tablename__ = "ingestion_quality"
    id = db.Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = db.Column(String(36), nullable=False, index=True)
    tenant_id = db.Column(String(64), nullable=False, index=True)
    text_coverage = db.Column(Float, default=0)
    avg_ocr_confidence = db.Column(Float, default=0)
    zero_text_pages = db.Column(Integer, default=0)
    total_pages = db.Column(Integer, default=0)
    duplicate_chunk_count = db.Column(Integer, default=0)
    empty_chunk_count = db.Column(Integer, default=0)
    malformed_chunk_count = db.Column(Integer, default=0)
    ocr_pages = db.Column(JSONB, default=list)
    page_issues = db.Column(JSONB, default=dict)
    overall_quality_score = db.Column(Float, default=1.0)
    quarantined = db.Column(Boolean, default=False)
    quarantine_reason = db.Column(Text)
    reprocess_attempts = db.Column(Integer, default=0)
    last_attempt_at = db.Column(DateTime(timezone=True))
    created_at = db.Column(DateTime(timezone=True), default=utcnow)
    updated_at = db.Column(DateTime(timezone=True), default=utcnow)


class SemanticCache(db.Model):
    """Feature F: Semantic cache for instant repeatable answers."""
    __tablename__ = "semantic_cache"
    id = db.Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id = db.Column(String(64), nullable=False, index=True)
    question_hash = db.Column(String(64), nullable=False, index=True)
    question_text = db.Column(Text, nullable=False)
    question_embedding = db.Column(JSONB, default=list)
    answer_text = db.Column(Text, nullable=False)
    sources = db.Column(JSONB, default=list)
    evidence_fingerprint = db.Column(String(64))
    doc_version_hashes = db.Column(JSONB, default=list)
    metadata_filters = db.Column(JSONB, default=dict)
    as_of_date = db.Column(Date)
    hit_count = db.Column(Integer, default=0)
    created_at = db.Column(DateTime(timezone=True), default=utcnow)
    expires_at = db.Column(DateTime(timezone=True))
    invalidated = db.Column(Boolean, default=False)
    invalidated_at = db.Column(DateTime(timezone=True))
    invalidation_reason = db.Column(Text)


class EvalBaseline(db.Model):
    """Feature M: Labeled test questions for drift & regression detection."""
    __tablename__ = "eval_baseline"
    id = db.Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id = db.Column(String(64), nullable=False, index=True)
    question = db.Column(Text, nullable=False)
    relevant_doc_ids = db.Column(JSONB, default=list)
    relevant_chunk_ids = db.Column(JSONB, default=list)
    expected_keywords = db.Column(JSONB, default=list)
    difficulty_class = db.Column(String(64))
    created_at = db.Column(DateTime(timezone=True), default=utcnow)
    created_by = db.Column(String(36))


class DriftRun(db.Model):
    """Feature M: Historical index benchmark runs."""
    __tablename__ = "drift_runs"
    id = db.Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id = db.Column(String(64), nullable=False, index=True)
    trigger_event = db.Column(String(64), default="manual")
    embedding_model = db.Column(String(128))
    index_version = db.Column(String(64))
    recall_at_5 = db.Column(Float)
    recall_at_10 = db.Column(Float)
    mrr = db.Column(Float)
    ndcg_at_10 = db.Column(Float)
    baseline_recall = db.Column(Float)
    drift_pct = db.Column(Float)
    alert_triggered = db.Column(Boolean, default=False)
    details = db.Column(JSONB, default=dict)
    created_at = db.Column(DateTime(timezone=True), default=utcnow)


class InformationGap(db.Model):
    """Feature P: Unanswered question clusters indicating missing documents."""
    __tablename__ = "information_gaps"
    id = db.Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id = db.Column(String(64), nullable=False, index=True)
    cluster_label = db.Column(Text)
    question_count = db.Column(Integer, default=1)
    sample_questions = db.Column(JSONB, default=list)
    category = db.Column(String(128))
    severity = db.Column(String(16), default="medium")
    first_seen_at = db.Column(DateTime(timezone=True), default=utcnow)
    last_seen_at = db.Column(DateTime(timezone=True), default=utcnow)
    resolved = db.Column(Boolean, default=False)
    resolved_at = db.Column(DateTime(timezone=True))


class GapQuestion(db.Model):
    """Feature P: Raw unanswered questions attached to gap clusters."""
    __tablename__ = "gap_questions"
    id = db.Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id = db.Column(String(64), nullable=False, index=True)
    gap_id = db.Column(String(36), index=True)
    question = db.Column(Text, nullable=False)
    user_id = db.Column(String(36))
    log_id = db.Column(String(36))
    embedding = db.Column(JSONB, default=list)
    created_at = db.Column(DateTime(timezone=True), default=utcnow)


class CounterfactualTest(db.Model):
    """Feature H: Regression and counterfactual tests."""
    __tablename__ = "counterfactual_tests"
    id = db.Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id = db.Column(String(64), nullable=False, index=True)
    name = db.Column(String(256))
    base_document_id = db.Column(String(36), nullable=False)
    modification_type = db.Column(String(64), nullable=False)
    modification_spec = db.Column(JSONB, default=dict, nullable=False)
    test_questions = db.Column(JSONB, default=list, nullable=False)
    status = db.Column(String(32), default="pending")
    results = db.Column(JSONB, default=dict)
    regression_detected = db.Column(Boolean, default=False)
    created_by = db.Column(String(36))
    created_at = db.Column(DateTime(timezone=True), default=utcnow)
    completed_at = db.Column(DateTime(timezone=True))


class KnowledgeExpiration(db.Model):
    """Feature L: Document expiration, review dates, and validity."""
    __tablename__ = "knowledge_expiration"
    id = db.Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = db.Column(String(36), nullable=False, index=True)
    tenant_id = db.Column(String(64), nullable=False, index=True)
    expiry_date = db.Column(Date)
    review_date = db.Column(Date)
    effective_period_end = db.Column(Date)
    expiry_status = db.Column(String(32), default="active")
    expiry_source = db.Column(String(64), default="metadata")
    superseded_by = db.Column(String(36))
    admin_notified = db.Column(Boolean, default=False)
    notes = db.Column(Text)
    created_at = db.Column(DateTime(timezone=True), default=utcnow)
    updated_at = db.Column(DateTime(timezone=True), default=utcnow)
