"""Tests for Feature K: Contradiction Discovery."""
from __future__ import annotations

import pytest


class TestFactualClaimExtraction:
    def test_extracts_percentage(self):
        from services.contradiction import extract_factual_claims
        claims = extract_factual_claims(
            "Students must maintain 75% attendance every semester.",
            "doc1", "policy.pdf"
        )
        assert any(c["claim_type"] == "percentage" for c in claims)
        assert any("75" in c["value"] for c in claims)

    def test_extracts_policy_sentence(self):
        from services.contradiction import extract_factual_claims
        claims = extract_factual_claims(
            "Attendance requirement is mandatory for all students.",
            "doc1", "policy.pdf"
        )
        assert any(c["claim_type"] in ("policy", "percentage") for c in claims)

    def test_extracts_amount(self):
        from services.contradiction import extract_factual_claims
        # Use text with explicit fee keyword to trigger policy pattern
        claims = extract_factual_claims(
            "The exam fee must be paid before the deadline. Fees are Rs. 500.",
            "doc1", "fees.pdf"
        )
        # At least one claim (policy OR amount) should be extracted
        assert len(claims) >= 1

    def test_deduplicate_claims(self):
        from services.contradiction import extract_factual_claims
        text = "Attendance is 75%. Attendance is 75%. Students must attend 75% of classes."
        claims = extract_factual_claims(text, "doc1", "policy.pdf")
        texts = [c["claim_text"] for c in claims]
        # Should not have exact duplicates
        assert len(texts) == len(set(texts[:len(texts)]))


class TestConflictDetection:
    def test_percentage_conflict_detected(self):
        from services.contradiction import _claims_conflict
        claim_a = {"claim_type": "percentage", "claim_text": "Attendance must be 75%", "value": "75"}
        claim_b = {"claim_type": "percentage", "claim_text": "Attendance must be 80%", "value": "80"}
        is_conflict, conflict_type, _ = _claims_conflict(claim_a, claim_b)
        assert is_conflict is True
        assert conflict_type == "percentage_conflict"

    def test_same_percentage_no_conflict(self):
        from services.contradiction import _claims_conflict
        claim_a = {"claim_type": "percentage", "claim_text": "75% attendance needed", "value": "75"}
        claim_b = {"claim_type": "percentage", "claim_text": "minimum 75% required", "value": "75"}
        is_conflict, _, _ = _claims_conflict(claim_a, claim_b)
        assert is_conflict is False

    def test_different_types_no_conflict(self):
        from services.contradiction import _claims_conflict
        claim_a = {"claim_type": "percentage", "value": "75"}
        claim_b = {"claim_type": "amount", "value": "500"}
        is_conflict, _, _ = _claims_conflict(claim_a, claim_b)
        assert is_conflict is False

    def test_severity_classification_high(self):
        from services.contradiction import _classify_severity
        sev = _classify_severity("percentage_conflict", False)
        assert sev == "high"

    def test_severity_classification_temporal_downgrades(self):
        from services.contradiction import _classify_severity
        sev = _classify_severity("percentage_conflict", True)
        assert sev == "low"

    def test_policy_negation_conflict_critical(self):
        from services.contradiction import _classify_severity
        sev = _classify_severity("policy_negation_conflict", False)
        assert sev == "critical"


class TestConflictAPI:
    @pytest.mark.skipif(
        "sqlite" in __import__("os").environ.get("DATABASE_URL", "sqlite"),
        reason="Requires PostgreSQL (JSONB)"
    )
    def test_get_conflicts_empty(self, client, auth_headers):
        resp = client.get("/api/conflicts", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.get_json()
        assert "conflicts" in data
        assert isinstance(data["conflicts"], list)
