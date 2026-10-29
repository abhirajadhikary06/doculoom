"""
One-off migration: add credits/max_credits to users table,
and create any missing tables (credit_logs, memory_graphs).
Skips Qdrant/corpus init to run fast.
"""
import os
from dotenv import load_dotenv
load_dotenv()

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text

# Minimal Flask app — no Qdrant, no corpus, no limiter
app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ["DATABASE_URL"]
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {"pool_pre_ping": True}

db = SQLAlchemy(app)

with app.app_context():
    with db.engine.connect() as conn:
        # Check existing columns
        result = conn.execute(text(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'users'"
        ))
        cols = [r[0] for r in result.fetchall()]
        print("Existing users columns:", cols)

        if "credits" not in cols:
            conn.execute(text(
                "ALTER TABLE users ADD COLUMN credits INTEGER NOT NULL DEFAULT 50"
            ))
            print("✓ Added 'credits' column")
        else:
            print("  'credits' already exists")

        if "max_credits" not in cols:
            conn.execute(text(
                "ALTER TABLE users ADD COLUMN max_credits INTEGER NOT NULL DEFAULT 50"
            ))
            print("✓ Added 'max_credits' column")
        else:
            print("  'max_credits' already exists")

        conn.commit()
        print("✓ Columns committed")

    # Now create any missing tables using the real models
    import importlib, sys
    # Remove cached modules so models bind to our db instance
    for mod in list(sys.modules.keys()):
        if mod in ("models", "app"):
            del sys.modules[mod]

    # Re-create tables via raw SQL for the new ones we know about
    conn2 = db.engine.connect()
    conn2.execute(text("""
        CREATE TABLE IF NOT EXISTS credit_logs (
            id VARCHAR(36) PRIMARY KEY,
            user_id VARCHAR(36) NOT NULL,
            tenant_id VARCHAR(64) NOT NULL,
            amount INTEGER NOT NULL,
            balance_after INTEGER NOT NULL,
            reason VARCHAR(64) NOT NULL,
            meta JSONB DEFAULT '{}',
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
    """))
    conn2.execute(text("""
        CREATE TABLE IF NOT EXISTS memory_graphs (
            id VARCHAR(36) PRIMARY KEY,
            tenant_id VARCHAR(64) NOT NULL,
            user_id VARCHAR(36),
            conversation_id VARCHAR(36),
            nodes JSONB DEFAULT '[]',
            edges JSONB DEFAULT '[]',
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        )
    """))
    conn2.commit()
    conn2.close()
    print("✓ credit_logs and memory_graphs tables created/verified")

print("Migration complete.")
