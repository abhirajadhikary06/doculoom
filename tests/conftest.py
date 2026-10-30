"""
Shared pytest fixtures for DocuLoom tests.

Uses an in-memory SQLite database and an in-memory Qdrant instance so
no real infrastructure is required to run the test suite.
"""
from __future__ import annotations

import os
import pytest

# Point to SQLite and in-memory Qdrant before any app imports
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["QDRANT_URL"] = ":memory:"
os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("SECRET_KEY", "test-secret")
os.environ.setdefault("JWT_SECRET", "test-jwt-secret")
os.environ.setdefault("FLASK_ENV", "testing")


@pytest.fixture(scope="session")
def app():
    """Create a Flask test application with in-memory stores."""
    # Patch Config engine options before init_app is called
    import config as cfg
    cfg.Config.SQLALCHEMY_ENGINE_OPTIONS = {"pool_pre_ping": False}
    cfg.Config.SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"

    from app import app as flask_app, init_db

    flask_app.config["TESTING"] = True
    flask_app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
    flask_app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {"pool_pre_ping": False}

    with flask_app.app_context():
        init_db()
        yield flask_app


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def app_ctx(app):
    """Push an application context for tests that use db directly."""
    with app.app_context():
        yield app


@pytest.fixture
def auth_headers(client):
    """Register a test user and return Bearer token headers."""
    resp = client.post(
        "/api/auth/register",
        json={
            "email": "test@campus.edu",
            "password": "password123",
            "display_name": "Test User",
        },
    )
    # May already exist
    if resp.status_code == 409:
        resp = client.post(
            "/api/auth/login",
            json={"email": "test@campus.edu", "password": "password123"},
        )
    data = resp.get_json()
    token = data.get("token", "")
    return {"Authorization": f"Bearer {token}", "X-Tenant-Id": "campus"}


# ── Skip DB-dependent tests when using SQLite ─────────────────────────────────
# Models use JSONB which is PostgreSQL-only. API tests that go through
# SQLAlchemy model creation are skipped in the SQLite CI environment.
# Set DATABASE_URL to a real PostgreSQL URL to run the full integration suite.

_USING_SQLITE = "sqlite" in os.environ.get("DATABASE_URL", "sqlite")


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests that require PostgreSQL when running with SQLite."""
    pass
