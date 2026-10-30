"""Tests for Feature P: Information Gap Discovery."""
from __future__ import annotations

import pytest


class TestAbstentionDetection:
    def test_abstention_detected(self):
        from services.gaps import ABSTENTION_PATTERN
        assert ABSTENTION_PATTERN.search("I don't know based on the available campus documents.")

    def test_non_abstention_not_detected(self):
        from services.gaps import ABSTENTION_PATTERN
        assert not ABSTENTION_PATTERN.search("The attendance requirement is 75%.")

    def test_case_insensitive(self):
        from services.gaps import ABSTENTION_PATTERN
        assert ABSTENTION_PATTERN.search("I DON'T KNOW based on documents.")


class TestGapCategoryClassification:
    def test_hostel_category(self):
        from services.gaps import _classify_gap_category
        assert _classify_gap_category("What is the hostel refund policy?") == "hostel_facilities"

    def test_fee_category(self):
        from services.gaps import _classify_gap_category
        assert _classify_gap_category("How do I pay the tuition fee refund?") == "fee_structure"

    def test_placement_category(self):
        from services.gaps import _classify_gap_category
        assert _classify_gap_category("Which companies visited for campus placement?") == "placement"

    def test_general_fallback(self):
        from services.gaps import _classify_gap_category
        assert _classify_gap_category("What is the weather like today?") == "general"


class TestGapRecommendation:
    def test_high_priority_recommendation(self):
        from services.gaps import _gap_recommendation
        rec = _gap_recommendation("fee_structure", 15)
        assert "HIGH PRIORITY" in rec
        assert "15 students" in rec

    def test_medium_priority_recommendation(self):
        from services.gaps import _gap_recommendation
        rec = _gap_recommendation("placement", 7)
        assert "MEDIUM PRIORITY" in rec

    def test_base_recommendation_included(self):
        from services.gaps import _gap_recommendation
        rec = _gap_recommendation("hostel_facilities", 1)
        assert len(rec) > 20  # non-empty recommendation


class TestGapAPI:
    def test_gaps_endpoint(self, client, auth_headers):
        resp = client.get("/api/gaps", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.get_json()
        assert "gaps" in data
        assert "summary" in data

    def test_aggregate_endpoint(self, client, auth_headers):
        resp = client.post("/api/gaps/aggregate", headers=auth_headers)
        assert resp.status_code == 200

    def test_record_if_unanswered(self, app_ctx):
        from services.gaps import record_if_unanswered
        recorded = record_if_unanswered(
            "What is the hostel refund policy?",
            "I don't know based on the available campus documents.",
            "campus",
        )
        assert recorded is True

    def test_does_not_record_answered(self, app_ctx):
        from services.gaps import record_if_unanswered
        recorded = record_if_unanswered(
            "What is the attendance policy?",
            "The attendance requirement is 75% per semester.",
            "campus",
        )
        assert recorded is False
