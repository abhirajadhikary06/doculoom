"""Tests for Feature N: Document Trust Scoring."""
from __future__ import annotations

import pytest
from datetime import date, timedelta


class TestTrustFactors:
    def test_corpus_source_higher_than_upload(self):
        from services.trust import _score_source_authority
        assert _score_source_authority("corpus") > _score_source_authority("upload")

    def test_verified_doc_full_score(self):
        from services.trust import _score_verification_status
        assert _score_verification_status({"verified": True}) == 1.0

    def test_unverified_doc_partial_score(self):
        from services.trust import _score_verification_status
        assert _score_verification_status({}) == 0.5

    def test_recent_doc_higher_recency(self):
        from services.trust import _score_pub_date_recency
        recent = date.today() - timedelta(days=30)
        old = date.today() - timedelta(days=3650)
        assert _score_pub_date_recency(recent) > _score_pub_date_recency(old)

    def test_none_pub_date_neutral(self):
        from services.trust import _score_pub_date_recency
        score = _score_pub_date_recency(None)
        assert score == 0.5

    def test_active_doc_full_validity(self):
        from services.trust import _score_effective_validity
        assert _score_effective_validity(None, None, None) == 1.0

    def test_superseded_doc_low_validity(self):
        from services.trust import _score_effective_validity
        assert _score_effective_validity(None, None, "some-other-doc-id") < 0.5

    def test_expired_doc_low_validity(self):
        from services.trust import _score_effective_validity
        past = str(date.today() - timedelta(days=365))
        assert _score_effective_validity(None, past, None) < 0.5

    def test_not_superseded_full_supersession_score(self):
        from services.trust import _score_supersession_status
        assert _score_supersession_status(None) == 1.0

    def test_superseded_low_supersession_score(self):
        from services.trust import _score_supersession_status
        assert _score_supersession_status("some-doc") < 0.5

    def test_weights_sum_to_one(self):
        from services.trust import WEIGHTS
        assert abs(sum(WEIGHTS.values()) - 1.0) < 0.001


class TestTrustInterpretation:
    def test_high_score_label(self):
        from services.trust import _interpret_score
        assert "High trust" in _interpret_score(0.9)

    def test_low_score_label(self):
        from services.trust import _interpret_score
        assert "low" in _interpret_score(0.2).lower()


class TestTrustReranking:
    def test_reranking_preserves_passage_count(self):
        from services.trust import apply_trust_reranking
        passages = [
            {"id": "1", "score": 0.9, "document_id": ""},  # empty doc_id — no DB hit
            {"id": "2", "score": 0.7, "document_id": ""},
        ]
        result = apply_trust_reranking(passages, trust_weight=0.1)
        assert len(result) == len(passages)

    def test_zero_weight_no_change(self):
        from services.trust import apply_trust_reranking
        passages = [{"id": "1", "score": 0.9, "document_id": "doc1"}]
        result = apply_trust_reranking(passages, trust_weight=0.0)
        assert result == passages


class TestTrustAPI:
    def test_trust_scores_endpoint(self, client, auth_headers):
        resp = client.get("/api/trust", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.get_json()
        assert "trust_scores" in data

    def test_recompute_endpoint(self, client, auth_headers):
        resp = client.post("/api/trust/recompute", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.get_json()
        assert "updated" in data
