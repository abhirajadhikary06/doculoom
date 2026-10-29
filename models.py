"""SQLAlchemy models for multi-tenant DocuLoom (Neon Postgres)."""
from datetime import datetime, timezone
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Index, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy import String, Text, Integer, Float, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import relationship
import uuid

db = SQLAlchemy()


def utcnow():
    return datetime.now(timezone.utc)


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
    status = db.Column(String(32), default="pending")  # pending|processing|ready|failed
    error_message = db.Column(Text)
    source = db.Column(String(32), default="upload")  # upload | corpus
    checksum = db.Column(String(64), index=True)  # sha256 for dedup
    meta = db.Column(JSONB, default=dict)
    created_at = db.Column(DateTime(timezone=True), default=utcnow)
    updated_at = db.Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

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
