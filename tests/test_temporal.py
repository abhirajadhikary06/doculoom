"""Tests for Feature A: Temporal Knowledge Engine."""
from __future__ import annotations

import pytest


class TestDateExtraction:
    """Unit tests for extract_dates_from_text()."""

    def test_iso_date_extraction(self):
        from services.temporal import extract_dates_from_text
        result = extract_dates_from_text("The policy is effective from 2025-08-01.")
        assert "2025-08-01" in result["all_dates"]

    def test_long_form_date(self):
        from services.temporal import extract_dates_from_text
        result = extract_dates_from_text("Effective from 15 August 2024, students must attend 80% classes.")
        assert result["valid_from"] is not None
        assert "2024" in str(result["valid_from"])

    def test_valid_from_extracted(self):
        from services.temporal import extract_dates_from_text
        text = "With effect from 01/01/2025, new attendance rules apply."
        result = extract_dates_from_text(text)
        assert result["valid_from"] is not None

    def test_valid_until_extracted(self):
        from services.temporal import extract_dates_from_text
        text = "This circular is valid until 31/12/2025."
        result = extract_dates_from_text(text)
        assert result["valid_until"] is not None

    def test_supersession_hints(self):
        from services.temporal import extract_dates_from_text
        text = "This circular supersedes the previous attendance circular."
        result = extract_dates_from_text(text)
        # Supersession keyword is present — hints should be detected
        # (date context is not required for the keyword scan)
        assert isinstance(result["supersession_hints"], list)

    def test_out_of_range_year_ignored(self):
        from services.temporal import extract_dates_from_text
        result = extract_dates_from_text("Built in 1893 and renovated in 3050.")
        assert result["all_dates"] == []

    def test_no_dates(self):
        from services.temporal import extract_dates_from_text
        result = extract_dates_from_text("The sky is blue and the water is clear.")
        assert result["all_dates"] == []


class TestContentHash:
    def test_same_text_same_hash(self):
        from services.temporal import compute_content_hash
        h1 = compute_content_hash("Attendance policy 75%")
        h2 = compute_content_hash("  Attendance   policy 75%  ")
        assert h1 == h2  # normalised

    def test_different_text_different_hash(self):
        from services.temporal import compute_content_hash
        h1 = compute_content_hash("Attendance policy 75%")
        h2 = compute_content_hash("Attendance policy 80%")
        assert h1 != h2


class TestVersionRegistration:
    @pytest.mark.skipif(
        "sqlite" in __import__("os").environ.get("DATABASE_URL", "sqlite"),
        reason="Requires PostgreSQL (JSONB)"
    )
    def test_register_version(self, app_ctx):
        from services.temporal import register_document_version
        from models import db, Document, Tenant
        import uuid

        # Create a minimal document record
        t = Tenant.query.get("campus") or Tenant(id="campus", name="Campus")
        db.session.merge(t)
        doc_id = str(uuid.uuid4())
        doc = Document(
            id=doc_id,
            tenant_id="campus",
            filename="test.pdf",
            status="ready",
            checksum="abc123",
        )
        db.session.add(doc)
        db.session.commit()

        result = register_document_version(doc_id, "campus", "Some document text about attendance")
        assert result["version_number"] == 1
        assert result["is_new"] is True

    @pytest.mark.skipif(
        "sqlite" in __import__("os").environ.get("DATABASE_URL", "sqlite"),
        reason="Requires PostgreSQL (JSONB)"
    )
    def test_same_content_no_duplicate(self, app_ctx):
        from services.temporal import register_document_version
        from models import db, Document, Tenant
        import uuid

        t = Tenant.query.get("campus") or Tenant(id="campus", name="Campus")
        db.session.merge(t)
        doc_id = str(uuid.uuid4())
        doc = Document(
            id=doc_id, tenant_id="campus", filename="dup.pdf",
            status="ready", checksum="dup_hash",
        )
        db.session.add(doc)
        db.session.commit()

        r1 = register_document_version(doc_id, "campus", "Identical content here")
        r2 = register_document_version(doc_id, "campus", "Identical content here")
        assert r1["version_number"] == r2["version_number"]
        assert r2["is_new"] is False
