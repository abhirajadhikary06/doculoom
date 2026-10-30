import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-me")
    FLASK_ENV = os.getenv("FLASK_ENV", "development")
    DEBUG = FLASK_ENV == "development"
    PORT = int(os.getenv("PORT", 5000))

    # Neon Postgres
    SQLALCHEMY_DATABASE_URI = os.getenv(
        "DATABASE_URL",
        "sqlite:///" + str(BASE_DIR / "doculoom.db"),
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        "pool_pre_ping": True,
        "pool_recycle": 300,
        "pool_size": 5,
        "max_overflow": 10,
    }

    # Qdrant
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "doculoom_chunks")

    # Groq
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    # Embeddings
    EMBEDDING_MODEL = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", 384))

    # Auth
    JWT_SECRET = os.getenv("JWT_SECRET", SECRET_KEY)
    JWT_EXPIRE_HOURS = int(os.getenv("JWT_EXPIRE_HOURS", 72))
    JWT_ALGORITHM = "HS256"

    # Chunking / retrieval
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))
    TOP_K = int(os.getenv("TOP_K", 8))
    RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", 4))
    RERANK_TIMEOUT_S = float(os.getenv("RERANK_TIMEOUT_S", 1.5))

    # Upload / OCR
    MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", 25))
    MAX_CONTENT_LENGTH = MAX_UPLOAD_MB * 1024 * 1024
    OCR_SPACE_API_KEY = os.getenv("OCR_SPACE_API_KEY", "")
    OCR_SPACE_URL = os.getenv("OCR_SPACE_URL", "https://api.ocr.space/parse/image")

    # Paths
    CORPUS_DIR = Path(os.getenv("CORPUS_DIR", BASE_DIR / "corpus"))
    UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", BASE_DIR / "uploads"))

    # Conversation
    DEFAULT_TENANT = os.getenv("DEFAULT_TENANT", "campus")
    HISTORY_TURNS = int(os.getenv("HISTORY_TURNS", 10))
    HISTORY_TOKEN_BUDGET = int(os.getenv("HISTORY_TOKEN_BUDGET", 4000))

    # Credits
    MAX_CREDITS = int(os.getenv("MAX_CREDITS", 50))
    CREDITS_PER_ASK = int(os.getenv("CREDITS_PER_ASK", 1))
    CREDITS_PER_INGEST = int(os.getenv("CREDITS_PER_INGEST", 5))

    # Rate limiting
    REDIS_URL = os.getenv("REDIS_URL", "")
    RATELIMIT_DEFAULT = "60 per minute"

    # ── Feature A: Temporal Knowledge Engine ────────────────────────────────
    TEMPORAL_SUPERSESSION_CONFIDENCE = float(os.getenv("TEMPORAL_SUPERSESSION_CONFIDENCE", 0.65))

    # ── Feature B: Evidence Verification ────────────────────────────────────
    EVIDENCE_VERIFY_CLAIMS = os.getenv("EVIDENCE_VERIFY_CLAIMS", "true").lower() == "true"
    EVIDENCE_MAX_CLAIMS = int(os.getenv("EVIDENCE_MAX_CLAIMS", 12))

    # ── Feature C: Adaptive Retrieval Planner ───────────────────────────────
    ADAPTIVE_RETRIEVAL = os.getenv("ADAPTIVE_RETRIEVAL", "true").lower() == "true"
    TRUST_RERANK_WEIGHT = float(os.getenv("TRUST_RERANK_WEIGHT", 0.15))

    # ── Feature F: Semantic Cache ────────────────────────────────────────────
    CACHE_SEMANTIC_THRESHOLD = float(os.getenv("CACHE_SEMANTIC_THRESHOLD", 0.92))
    CACHE_TTL_HOURS = int(os.getenv("CACHE_TTL_HOURS", 24))
    MAX_CACHE_ENTRIES = int(os.getenv("MAX_CACHE_ENTRIES", 1000))
    SEMANTIC_CACHE_ENABLED = os.getenv("SEMANTIC_CACHE_ENABLED", "true").lower() == "true"

    # ── Feature G: Ingestion Quality ─────────────────────────────────────────
    IQ_MIN_COVERAGE = float(os.getenv("IQ_MIN_COVERAGE", 0.5))
    IQ_MIN_CHARS_PAGE = int(os.getenv("IQ_MIN_CHARS_PAGE", 50))
    IQ_MAX_EMPTY_RATIO = float(os.getenv("IQ_MAX_EMPTY_RATIO", 0.3))
    IQ_MIN_QUALITY = float(os.getenv("IQ_MIN_QUALITY", 0.4))

    # ── Feature J: Query Decomposition ───────────────────────────────────────
    DECOMPOSITION_ENABLED = os.getenv("DECOMPOSITION_ENABLED", "true").lower() == "true"
    DECOMPOSITION_MAX_SUBQUESTIONS = int(os.getenv("DECOMPOSITION_MAX_SUBQUESTIONS", 4))

    # ── Feature L: Knowledge Expiration ──────────────────────────────────────
    EXPIRY_REVIEW_DAYS = int(os.getenv("EXPIRY_REVIEW_DAYS", 365))

    # ── Feature M: Drift Detection ───────────────────────────────────────────
    DRIFT_ALERT_THRESHOLD = float(os.getenv("DRIFT_ALERT_THRESHOLD", 0.15))
    DRIFT_ON_INGEST = os.getenv("DRIFT_ON_INGEST", "false").lower() == "true"

    # ── Feature P: Information Gaps ───────────────────────────────────────────
    GAP_CLUSTER_THRESHOLD = float(os.getenv("GAP_CLUSTER_THRESHOLD", 0.80))

    # ── Provenance / reproducibility ─────────────────────────────────────────
    PROVENANCE_ENABLED = os.getenv("PROVENANCE_ENABLED", "true").lower() == "true"

    # ── Impact analysis ───────────────────────────────────────────────────────
    IMPACT_ANALYSIS_ENABLED = os.getenv("IMPACT_ANALYSIS_ENABLED", "true").lower() == "true"
    IMPACT_ANALYSIS_BACKGROUND = os.getenv("IMPACT_ANALYSIS_BACKGROUND", "true").lower() == "true"

    @classmethod
    def ensure_dirs(cls):
        cls.CORPUS_DIR.mkdir(parents=True, exist_ok=True)
        cls.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)