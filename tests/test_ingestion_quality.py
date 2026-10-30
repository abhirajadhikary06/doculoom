"""Tests for Feature G: Self-Healing Document Ingestion."""
from __future__ import annotations

import pytest


class TestPageInspection:
    def test_full_coverage(self):
        from services.ingestion_quality import inspect_pages
        pages = ["Page one text content here.", "Page two content here.", "Page three here."]
        result = inspect_pages(pages)
        assert result["text_coverage"] == pytest.approx(1.0)
        assert result["zero_text_pages"] == 0

    def test_detects_zero_text_pages(self):
        from services.ingestion_quality import inspect_pages
        pages = ["Good content here.", "", "Another good page."]
        result = inspect_pages(pages)
        assert result["zero_text_pages"] == 1
        assert result["text_coverage"] == pytest.approx(2 / 3, abs=1e-3)

    def test_page_issues_recorded(self):
        from services.ingestion_quality import inspect_pages
        pages = ["", "Only three words"]
        result = inspect_pages(pages)
        assert 1 in result["page_issues"]  # page 1 is empty

    def test_short_text_flagged(self):
        from services.ingestion_quality import inspect_pages
        pages = ["X"]  # 1 char — below MIN_AVG_CHARS_PER_PAGE
        result = inspect_pages(pages)
        assert result["short_text_pages"] >= 1


class TestChunkInspection:
    def test_normal_chunks(self):
        from services.ingestion_quality import inspect_chunks
        chunks = [
            {"text": "This is a normal chunk with several distinct words about policy."},
            {"text": "Another chunk discussing attendance requirements for students."},
        ]
        result = inspect_chunks(chunks)
        assert result["empty_chunks"] == 0
        assert result["malformed_chunks"] == 0

    def test_empty_chunks_detected(self):
        from services.ingestion_quality import inspect_chunks
        chunks = [{"text": ""}, {"text": "  "}, {"text": "Good content here for testing."}]
        result = inspect_chunks(chunks)
        assert result["empty_chunks"] == 2

    def test_duplicate_chunks_detected(self):
        from services.ingestion_quality import inspect_chunks
        same_text = "Attendance requirement is 75% minimum for all students."
        chunks = [{"text": same_text}, {"text": same_text}, {"text": "Different content here now."}]
        result = inspect_chunks(chunks)
        assert result["duplicate_chunks"] >= 1


class TestQualityScore:
    def test_perfect_quality(self):
        from services.ingestion_quality import compute_quality_score
        page_metrics = {
            "text_coverage": 1.0, "zero_text_pages": 0,
            "total_pages": 10, "ocr_page_count": 0,
        }
        chunk_metrics = {
            "total_chunks": 20, "empty_chunks": 0,
            "malformed_chunks": 0, "empty_ratio": 0.0,
        }
        score = compute_quality_score(page_metrics, chunk_metrics, {})
        assert score > 0.9

    def test_poor_quality(self):
        from services.ingestion_quality import compute_quality_score
        page_metrics = {
            "text_coverage": 0.1, "zero_text_pages": 9,
            "total_pages": 10, "ocr_page_count": 9,
        }
        chunk_metrics = {
            "total_chunks": 10, "empty_chunks": 7,
            "malformed_chunks": 2, "empty_ratio": 0.7,
        }
        score = compute_quality_score(page_metrics, chunk_metrics, {})
        assert score < 0.5

    def test_score_in_range(self):
        from services.ingestion_quality import compute_quality_score
        page_metrics = {
            "text_coverage": 0.6, "zero_text_pages": 4,
            "total_pages": 10, "ocr_page_count": 3,
        }
        chunk_metrics = {
            "total_chunks": 15, "empty_chunks": 2,
            "malformed_chunks": 1, "empty_ratio": 0.13,
        }
        score = compute_quality_score(page_metrics, chunk_metrics, {})
        assert 0.0 <= score <= 1.0


class TestQualityAPI:
    def test_quality_report_endpoint(self, client, auth_headers):
        resp = client.get("/api/quality", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.get_json()
        assert "reports" in data
        assert isinstance(data["reports"], list)
