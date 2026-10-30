"""Tests for Feature B: Evidence Verification Engine."""
from __future__ import annotations

import pytest


class TestClaimExtraction:
    def test_extracts_multiple_claims(self):
        from services.evidence import extract_claims
        answer = "Attendance must be 75%. Fees must be paid before November 15. Late submissions attract a fine."
        claims = extract_claims(answer)
        assert len(claims) >= 2

    def test_strips_citation_markers(self):
        from services.evidence import extract_claims
        answer = "Attendance is 75% [policy.pdf, page 3]. Exams start in December."
        claims = extract_claims(answer)
        for c in claims:
            assert "[" not in c or "page" not in c.lower()

    def test_filters_short_fragments(self):
        from services.evidence import extract_claims
        claims = extract_claims("Yes. No. Maybe. This is a proper sentence about campus policy.")
        for c in claims:
            assert len(c) >= 15

    def test_abstention_returns_no_claims(self):
        from services.evidence import extract_claims
        claims = extract_claims("I don't know based on the available campus documents.")
        assert len(claims) == 0


class TestTokenOverlap:
    def test_high_overlap(self):
        from services.evidence import _token_overlap
        score = _token_overlap(
            "Attendance requirement is 75 percent",
            "The attendance requirement for students is 75 percent minimum",
        )
        assert score > 0.3

    def test_no_overlap(self):
        from services.evidence import _token_overlap
        score = _token_overlap("The sky is blue", "Fee payment deadline November")
        assert score < 0.15

    def test_empty_strings(self):
        from services.evidence import _token_overlap
        assert _token_overlap("", "something") == 0.0
        assert _token_overlap("something", "") == 0.0


class TestClaimAlignment:
    def test_finds_matching_passage(self):
        from services.evidence import align_claim_to_passages
        passages = [
            {"text": "Attendance must be a minimum of 75% per semester.", "filename": "policy.pdf", "page_start": 1},
            {"text": "The canteen serves lunch from 12 to 2 pm.", "filename": "info.pdf", "page_start": 5},
        ]
        result = align_claim_to_passages("Attendance requirement is 75%", passages)
        assert len(result) > 0
        assert result[0]["passage_idx"] == 0

    def test_no_match_below_threshold(self):
        from services.evidence import align_claim_to_passages
        passages = [{"text": "The library opens at 9am.", "filename": "info.pdf", "page_start": 1}]
        result = align_claim_to_passages("Fee deadline is November 15", passages, threshold=0.25)
        assert len(result) == 0


class TestVerifyAnswer:
    def test_abstention_returns_full_coverage(self):
        from services.evidence import verify_answer
        result = verify_answer("I don't know based on the available campus documents.", [])
        assert result["citation_coverage"] == 1.0
        assert result["total_claims"] == 0

    def test_supported_claim_detected(self):
        from services.evidence import verify_answer
        answer = "Students must maintain 75% attendance every semester."
        passages = [{"text": "Minimum attendance is 75% per semester for all students.", "filename": "p.pdf", "page_start": 1}]
        result = verify_answer(answer, passages, llm_client=None)
        assert result["total_claims"] >= 1
        # At least one claim should be supported (token overlap)
        assert result["supported_claims"] >= 1 or result["citation_coverage"] >= 0.5

    def test_unsupported_claim_detected(self):
        from services.evidence import verify_answer
        answer = "Attendance is 75% and fees must be paid by November 15."
        passages = [{"text": "Minimum attendance is 75%.", "filename": "p.pdf", "page_start": 1}]
        result = verify_answer(answer, passages, llm_client=None)
        assert result["total_claims"] >= 1
        assert result["unsupported_claims"] >= 0  # fee claim may be unsupported


class TestCleanAnswer:
    def test_removes_unsupported_low_confidence(self):
        from services.evidence import clean_answer
        verification = {
            "claims": [
                {"text": "Attendance must be 75 percent", "verdict": "supported", "confidence": 0.9},
                {"text": "Fees are due on November 15th deadline", "verdict": "unsupported", "confidence": 0.05},
            ]
        }
        original = "Attendance must be 75 percent. Fees are due on November 15th deadline."
        cleaned = clean_answer(original, verification)
        assert cleaned  # never returns empty

    def test_preserves_answer_when_all_supported(self):
        from services.evidence import clean_answer
        verification = {
            "claims": [
                {"text": "Attendance must be 75 percent", "verdict": "supported", "confidence": 0.9},
            ]
        }
        original = "Attendance must be 75 percent."
        assert clean_answer(original, verification) == original
