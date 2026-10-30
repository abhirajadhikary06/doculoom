"""
Migration v2: Add all new tables for the Campus Knowledge Intelligence Engine.

New tables:
  - document_versions       (A: temporal)
  - document_supersessions  (A: temporal)
  - document_entities       (D: impact)
  - document_relationships  (D: impact)
  - document_conflicts      (K/D: contradictions + impact)
  - query_execution_logs    (E: reproducibility)
  - execution_evidence      (E: reproducibility)
  - ingestion_quality       (G: self-healing)
  - semantic_cache          (F: semantic cache)
  - eval_baseline           (M: drift)
  - drift_runs              (M: drift)
  - information_gaps        (P: gap discovery)
  - counterfactual_tests    (H: counterfactual)
  - knowledge_expiration    (L: expiration)

Alters:
  - documents: add valid_from, valid_until, pub_date, version, superseded_by,
               trust_score, trust_factors, ingestion_quality_id
"""
import os
from dotenv import load_dotenv

load_dotenv()

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get(
    "DATABASE_URL", "sqlite:///doculoom.db"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {"pool_pre_ping": True}

db = SQLAlchemy(app)

DDL_STATEMENTS = [
    # ── document_versions ─────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS document_versions (
        id              VARCHAR(36) PRIMARY KEY,
        document_id     VARCHAR(36) NOT NULL,
        tenant_id       VARCHAR(64) NOT NULL,
        version_number  INTEGER     NOT NULL DEFAULT 1,
        content_hash    VARCHAR(64) NOT NULL,
        pub_date        DATE,
        valid_from      DATE,
        valid_until     DATE,
        extracted_dates JSONB       DEFAULT '[]',
        raw_meta        JSONB       DEFAULT '{}',
        created_at      TIMESTAMPTZ DEFAULT NOW()
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_docver_doc ON document_versions(document_id)",
    "CREATE INDEX IF NOT EXISTS ix_docver_tenant ON document_versions(tenant_id)",
    "CREATE INDEX IF NOT EXISTS ix_docver_hash ON document_versions(content_hash)",

    # ── document_supersessions ────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS document_supersessions (
        id              VARCHAR(36) PRIMARY KEY,
        tenant_id       VARCHAR(64) NOT NULL,
        old_document_id VARCHAR(36) NOT NULL,
        new_document_id VARCHAR(36) NOT NULL,
        relationship    VARCHAR(32) NOT NULL DEFAULT 'supersedes',
        detected_at     TIMESTAMPTZ DEFAULT NOW(),
        evidence        JSONB       DEFAULT '{}'
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_supersession_tenant ON document_supersessions(tenant_id)",
    "CREATE INDEX IF NOT EXISTS ix_supersession_old ON document_supersessions(old_document_id)",
    "CREATE INDEX IF NOT EXISTS ix_supersession_new ON document_supersessions(new_document_id)",

    # ── document_entities ─────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS document_entities (
        id           VARCHAR(36) PRIMARY KEY,
        document_id  VARCHAR(36) NOT NULL,
        tenant_id    VARCHAR(64) NOT NULL,
        entity_type  VARCHAR(64) NOT NULL,
        entity_value TEXT        NOT NULL,
        confidence   FLOAT       DEFAULT 1.0,
        page_ref     INTEGER     DEFAULT 1,
        context      TEXT,
        created_at   TIMESTAMPTZ DEFAULT NOW()
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_entities_doc ON document_entities(document_id)",
    "CREATE INDEX IF NOT EXISTS ix_entities_tenant ON document_entities(tenant_id)",
    "CREATE INDEX IF NOT EXISTS ix_entities_type ON document_entities(entity_type)",

    # ── document_relationships ────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS document_relationships (
        id              VARCHAR(36) PRIMARY KEY,
        tenant_id       VARCHAR(64) NOT NULL,
        source_doc_id   VARCHAR(36) NOT NULL,
        target_doc_id   VARCHAR(36) NOT NULL,
        relationship    VARCHAR(64) NOT NULL,
        confidence      FLOAT       DEFAULT 1.0,
        evidence        TEXT,
        detected_at     TIMESTAMPTZ DEFAULT NOW()
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_docrel_tenant ON document_relationships(tenant_id)",
    "CREATE INDEX IF NOT EXISTS ix_docrel_source ON document_relationships(source_doc_id)",

    # ── document_conflicts ────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS document_conflicts (
        id              VARCHAR(36) PRIMARY KEY,
        tenant_id       VARCHAR(64) NOT NULL,
        doc_id_a        VARCHAR(36) NOT NULL,
        doc_id_b        VARCHAR(36) NOT NULL,
        conflict_type   VARCHAR(64) NOT NULL,
        severity        VARCHAR(16) NOT NULL DEFAULT 'medium',
        description     TEXT,
        claim_a         TEXT,
        claim_b         TEXT,
        page_a          INTEGER,
        page_b          INTEGER,
        is_temporal     BOOLEAN     DEFAULT FALSE,
        resolved        BOOLEAN     DEFAULT FALSE,
        detected_at     TIMESTAMPTZ DEFAULT NOW(),
        reviewed_at     TIMESTAMPTZ
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_conflicts_tenant ON document_conflicts(tenant_id)",
    "CREATE INDEX IF NOT EXISTS ix_conflicts_resolved ON document_conflicts(tenant_id, resolved)",

    # ── query_execution_logs ──────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS query_execution_logs (
        id                  VARCHAR(36) PRIMARY KEY,
        tenant_id           VARCHAR(64) NOT NULL,
        user_id             VARCHAR(36),
        conversation_id     VARCHAR(36),
        question            TEXT        NOT NULL,
        search_query        TEXT,
        retrieval_strategy  VARCHAR(64) DEFAULT 'hybrid',
        query_intent        VARCHAR(64),
        difficulty_class    VARCHAR(64),
        answer_hash         VARCHAR(64),
        answer_text         TEXT,
        sources             JSONB       DEFAULT '[]',
        retrieval_config    JSONB       DEFAULT '{}',
        embedding_model     VARCHAR(128),
        llm_model           VARCHAR(128),
        prompt_version      VARCHAR(32) DEFAULT 'v1',
        latency_ms          FLOAT,
        token_usage         JSONB       DEFAULT '{}',
        reranked            BOOLEAN     DEFAULT FALSE,
        was_decomposed      BOOLEAN     DEFAULT FALSE,
        cache_hit           BOOLEAN     DEFAULT FALSE,
        as_of_date          DATE,
        created_at          TIMESTAMPTZ DEFAULT NOW()
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_qlog_tenant ON query_execution_logs(tenant_id)",
    "CREATE INDEX IF NOT EXISTS ix_qlog_conv ON query_execution_logs(conversation_id)",
    "CREATE INDEX IF NOT EXISTS ix_qlog_user ON query_execution_logs(user_id)",
    "CREATE INDEX IF NOT EXISTS ix_qlog_created ON query_execution_logs(created_at DESC)",

    # ── execution_evidence ────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS execution_evidence (
        id              VARCHAR(36) PRIMARY KEY,
        log_id          VARCHAR(36) NOT NULL,
        tenant_id       VARCHAR(64) NOT NULL,
        chunk_id        VARCHAR(36) NOT NULL,
        document_id     VARCHAR(36) NOT NULL,
        chunk_index     INTEGER     DEFAULT 0,
        page_start      INTEGER     DEFAULT 1,
        retrieval_rank  INTEGER     NOT NULL,
        dense_score     FLOAT,
        sparse_score    FLOAT,
        rrf_score       FLOAT,
        rerank_score    FLOAT,
        text_snapshot   TEXT,
        doc_version     VARCHAR(64),
        created_at      TIMESTAMPTZ DEFAULT NOW()
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_evidence_log ON execution_evidence(log_id)",
    "CREATE INDEX IF NOT EXISTS ix_evidence_tenant ON execution_evidence(tenant_id)",

    # ── ingestion_quality ─────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS ingestion_quality (
        id                      VARCHAR(36) PRIMARY KEY,
        document_id             VARCHAR(36) NOT NULL,
        tenant_id               VARCHAR(64) NOT NULL,
        text_coverage           FLOAT       DEFAULT 0,
        avg_ocr_confidence      FLOAT       DEFAULT 0,
        zero_text_pages         INTEGER     DEFAULT 0,
        total_pages             INTEGER     DEFAULT 0,
        duplicate_chunk_count   INTEGER     DEFAULT 0,
        empty_chunk_count       INTEGER     DEFAULT 0,
        malformed_chunk_count   INTEGER     DEFAULT 0,
        ocr_pages               JSONB       DEFAULT '[]',
        page_issues             JSONB       DEFAULT '{}',
        overall_quality_score   FLOAT       DEFAULT 1.0,
        quarantined             BOOLEAN     DEFAULT FALSE,
        quarantine_reason       TEXT,
        reprocess_attempts      INTEGER     DEFAULT 0,
        last_attempt_at         TIMESTAMPTZ,
        created_at              TIMESTAMPTZ DEFAULT NOW(),
        updated_at              TIMESTAMPTZ DEFAULT NOW()
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_iq_doc ON ingestion_quality(document_id)",
    "CREATE INDEX IF NOT EXISTS ix_iq_tenant ON ingestion_quality(tenant_id)",
    "CREATE INDEX IF NOT EXISTS ix_iq_quarantine ON ingestion_quality(tenant_id, quarantined)",

    # ── semantic_cache ────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS semantic_cache (
        id                  VARCHAR(36) PRIMARY KEY,
        tenant_id           VARCHAR(64) NOT NULL,
        question_hash       VARCHAR(64) NOT NULL,
        question_text       TEXT        NOT NULL,
        question_embedding  JSONB       DEFAULT '[]',
        answer_text         TEXT        NOT NULL,
        sources             JSONB       DEFAULT '[]',
        evidence_fingerprint VARCHAR(64),
        doc_version_hashes  JSONB       DEFAULT '[]',
        metadata_filters    JSONB       DEFAULT '{}',
        as_of_date          DATE,
        hit_count           INTEGER     DEFAULT 0,
        created_at          TIMESTAMPTZ DEFAULT NOW(),
        expires_at          TIMESTAMPTZ,
        invalidated         BOOLEAN     DEFAULT FALSE,
        invalidated_at      TIMESTAMPTZ,
        invalidation_reason TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_scache_tenant ON semantic_cache(tenant_id)",
    "CREATE INDEX IF NOT EXISTS ix_scache_hash ON semantic_cache(tenant_id, question_hash)",
    "CREATE INDEX IF NOT EXISTS ix_scache_valid ON semantic_cache(tenant_id, invalidated, expires_at)",

    # ── eval_baseline ─────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS eval_baseline (
        id              VARCHAR(36) PRIMARY KEY,
        tenant_id       VARCHAR(64) NOT NULL,
        question        TEXT        NOT NULL,
        relevant_doc_ids JSONB      DEFAULT '[]',
        relevant_chunk_ids JSONB    DEFAULT '[]',
        expected_keywords JSONB     DEFAULT '[]',
        difficulty_class VARCHAR(64),
        created_at      TIMESTAMPTZ DEFAULT NOW(),
        created_by      VARCHAR(36)
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_baseline_tenant ON eval_baseline(tenant_id)",

    # ── drift_runs ────────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS drift_runs (
        id              VARCHAR(36) PRIMARY KEY,
        tenant_id       VARCHAR(64) NOT NULL,
        trigger_event   VARCHAR(64) DEFAULT 'manual',
        embedding_model VARCHAR(128),
        index_version   VARCHAR(64),
        recall_at_5     FLOAT,
        recall_at_10    FLOAT,
        mrr             FLOAT,
        ndcg_at_10      FLOAT,
        baseline_recall FLOAT,
        drift_pct       FLOAT,
        alert_triggered BOOLEAN     DEFAULT FALSE,
        details         JSONB       DEFAULT '{}',
        created_at      TIMESTAMPTZ DEFAULT NOW()
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_drift_tenant ON drift_runs(tenant_id)",
    "CREATE INDEX IF NOT EXISTS ix_drift_created ON drift_runs(tenant_id, created_at DESC)",

    # ── information_gaps ──────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS information_gaps (
        id              VARCHAR(36) PRIMARY KEY,
        tenant_id       VARCHAR(64) NOT NULL,
        cluster_label   TEXT,
        question_count  INTEGER     DEFAULT 1,
        sample_questions JSONB      DEFAULT '[]',
        category        VARCHAR(128),
        severity        VARCHAR(16) DEFAULT 'medium',
        first_seen_at   TIMESTAMPTZ DEFAULT NOW(),
        last_seen_at    TIMESTAMPTZ DEFAULT NOW(),
        resolved        BOOLEAN     DEFAULT FALSE,
        resolved_at     TIMESTAMPTZ
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_gaps_tenant ON information_gaps(tenant_id)",
    "CREATE INDEX IF NOT EXISTS ix_gaps_resolved ON information_gaps(tenant_id, resolved)",

    # ── gap_questions (raw unanswered questions linked to gaps) ───────────
    """
    CREATE TABLE IF NOT EXISTS gap_questions (
        id          VARCHAR(36) PRIMARY KEY,
        tenant_id   VARCHAR(64) NOT NULL,
        gap_id      VARCHAR(36),
        question    TEXT        NOT NULL,
        user_id     VARCHAR(36),
        log_id      VARCHAR(36),
        embedding   JSONB       DEFAULT '[]',
        created_at  TIMESTAMPTZ DEFAULT NOW()
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_gapq_tenant ON gap_questions(tenant_id)",
    "CREATE INDEX IF NOT EXISTS ix_gapq_gap ON gap_questions(gap_id)",

    # ── counterfactual_tests ──────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS counterfactual_tests (
        id                  VARCHAR(36) PRIMARY KEY,
        tenant_id           VARCHAR(64) NOT NULL,
        name                VARCHAR(256),
        base_document_id    VARCHAR(36) NOT NULL,
        modification_type   VARCHAR(64) NOT NULL,
        modification_spec   JSONB       NOT NULL DEFAULT '{}',
        test_questions      JSONB       NOT NULL DEFAULT '[]',
        status              VARCHAR(32) DEFAULT 'pending',
        results             JSONB       DEFAULT '{}',
        regression_detected BOOLEAN     DEFAULT FALSE,
        created_by          VARCHAR(36),
        created_at          TIMESTAMPTZ DEFAULT NOW(),
        completed_at        TIMESTAMPTZ
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_cf_tenant ON counterfactual_tests(tenant_id)",

    # ── knowledge_expiration ──────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS knowledge_expiration (
        id                  VARCHAR(36) PRIMARY KEY,
        document_id         VARCHAR(36) NOT NULL,
        tenant_id           VARCHAR(64) NOT NULL,
        expiry_date         DATE,
        review_date         DATE,
        effective_period_end DATE,
        expiry_status       VARCHAR(32) DEFAULT 'active',
        expiry_source       VARCHAR(64) DEFAULT 'metadata',
        superseded_by       VARCHAR(36),
        admin_notified      BOOLEAN     DEFAULT FALSE,
        notes               TEXT,
        created_at          TIMESTAMPTZ DEFAULT NOW(),
        updated_at          TIMESTAMPTZ DEFAULT NOW()
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_expiry_doc ON knowledge_expiration(document_id)",
    "CREATE INDEX IF NOT EXISTS ix_expiry_tenant ON knowledge_expiration(tenant_id)",
    "CREATE INDEX IF NOT EXISTS ix_expiry_status ON knowledge_expiration(tenant_id, expiry_status)",
]

# ALTER TABLE statements for existing tables (idempotent)
ALTER_STATEMENTS = [
    # documents table extensions
    ("documents", "pub_date",
     "ALTER TABLE documents ADD COLUMN IF NOT EXISTS pub_date DATE"),
    ("documents", "valid_from",
     "ALTER TABLE documents ADD COLUMN IF NOT EXISTS valid_from DATE"),
    ("documents", "valid_until",
     "ALTER TABLE documents ADD COLUMN IF NOT EXISTS valid_until DATE"),
    ("documents", "version",
     "ALTER TABLE documents ADD COLUMN IF NOT EXISTS version INTEGER DEFAULT 1"),
    ("documents", "superseded_by",
     "ALTER TABLE documents ADD COLUMN IF NOT EXISTS superseded_by VARCHAR(36)"),
    ("documents", "trust_score",
     "ALTER TABLE documents ADD COLUMN IF NOT EXISTS trust_score FLOAT DEFAULT 0.5"),
    ("documents", "trust_factors",
     "ALTER TABLE documents ADD COLUMN IF NOT EXISTS trust_factors JSONB DEFAULT '{}'"),
    ("documents", "ingestion_quality_id",
     "ALTER TABLE documents ADD COLUMN IF NOT EXISTS ingestion_quality_id VARCHAR(36)"),
    ("documents", "impact_analysis",
     "ALTER TABLE documents ADD COLUMN IF NOT EXISTS impact_analysis JSONB DEFAULT '{}'"),
]


def run_migration():
    with app.app_context():
        with db.engine.connect() as conn:
            print("=== DocuLoom v2 Migration ===\n")

            # Run CREATE TABLE / INDEX statements
            for stmt in DDL_STATEMENTS:
                stmt_clean = stmt.strip()
                if not stmt_clean:
                    continue
                label = stmt_clean[:60].replace("\n", " ").strip()
                try:
                    conn.execute(text(stmt_clean))
                    print(f"✓ {label}...")
                except Exception as e:
                    print(f"  ✗ {label}... ERROR: {e}")

            conn.commit()

            # Run ALTER TABLE for existing tables
            for table, col, alter_sql in ALTER_STATEMENTS:
                try:
                    conn.execute(text(alter_sql))
                    print(f"✓ ALTER {table}.{col}")
                except Exception as e:
                    if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
                        print(f"  ~ {table}.{col} already exists")
                    else:
                        print(f"  ✗ ALTER {table}.{col}: {e}")

            conn.commit()

            # Add indexes on altered columns
            extra_indexes = [
                "CREATE INDEX IF NOT EXISTS ix_docs_valid_from ON documents(tenant_id, valid_from)",
                "CREATE INDEX IF NOT EXISTS ix_docs_valid_until ON documents(tenant_id, valid_until)",
                "CREATE INDEX IF NOT EXISTS ix_docs_superseded ON documents(superseded_by)",
            ]
            for idx_sql in extra_indexes:
                try:
                    conn.execute(text(idx_sql))
                    print(f"✓ {idx_sql[:60]}...")
                except Exception as e:
                    print(f"  ~ index: {e}")
            conn.commit()

        print("\n=== Migration v2 complete ===")


if __name__ == "__main__":
    run_migration()
