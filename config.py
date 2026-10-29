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

    @classmethod
    def ensure_dirs(cls):
        cls.CORPUS_DIR.mkdir(parents=True, exist_ok=True)
        cls.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)