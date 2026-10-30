"""Tests for Feature C: Adaptive Retrieval Planner."""
from __future__ import annotations

import pytest


class TestQueryClassification:
    def test_factual_lookup_detected(self):
        from services.retrieval_planner import classify_query
        result = classify_query("What is the attendance requirement?")
        assert result["intent"] in ("factual_lookup", "unknown")
        assert result["strategy"] in ("exact_keyword", "hybrid")

    def test_temporal_detected(self):
        from services.retrieval_planner import classify_query
        result = classify_query("What was the attendance requirement in 2024?")
        assert result["intent"] == "historical_temporal"
        assert result["strategy"] == "temporal"
        assert result["extracted_year"] == "2024"

    def test_conceptual_explanation_detected(self):
        from services.retrieval_planner import classify_query
        result = classify_query("Explain the grading system for B.Tech students")
        assert result["intent"] == "conceptual_explanation"
        assert result["strategy"] == "dense_semantic"

    def test_department_specific_detected(self):
        from services.retrieval_planner import classify_query
        result = classify_query("What is the attendance rule for CSE students?")
        assert result["extracted_dept"] == "CSE"

    def test_multi_document_needs_decomposition(self):
        from services.retrieval_planner import classify_query
        result = classify_query(
            "What are all attendance requirements, how have they changed since 2024, and what is the fee structure?"
        )
        assert result["needs_decomposition"] is True

    def test_short_factual_high_confidence(self):
        from services.retrieval_planner import classify_query
        result = classify_query("Exam fee amount?")
        assert result["confidence"] > 0
        assert result["word_count"] <= 4

    def test_year_extraction(self):
        from services.retrieval_planner import classify_query
        result = classify_query("Was the rule different in 2023?")
        assert result["extracted_year"] == "2023"


class TestRetrievalMetrics:
    def test_recall_at_k_perfect(self):
        from services.retrieval_planner import compute_recall_at_k
        retrieved = ["a", "b", "c", "d"]
        relevant = ["a", "b"]
        assert compute_recall_at_k(retrieved, relevant, k=4) == 1.0

    def test_recall_at_k_zero(self):
        from services.retrieval_planner import compute_recall_at_k
        assert compute_recall_at_k(["x", "y"], ["a", "b"], k=2) == 0.0

    def test_recall_empty_relevant(self):
        from services.retrieval_planner import compute_recall_at_k
        assert compute_recall_at_k(["a"], [], k=5) == 0.0

    def test_mrr_first_hit(self):
        from services.retrieval_planner import compute_mrr
        assert compute_mrr(["a", "b", "c"], ["a"]) == 1.0

    def test_mrr_second_hit(self):
        from services.retrieval_planner import compute_mrr
        assert compute_mrr(["x", "a", "c"], ["a"]) == pytest.approx(0.5)

    def test_mrr_no_hit(self):
        from services.retrieval_planner import compute_mrr
        assert compute_mrr(["x", "y"], ["a"]) == 0.0

    def test_ndcg_perfect(self):
        from services.retrieval_planner import compute_ndcg_at_k
        assert compute_ndcg_at_k(["a", "b"], ["a", "b"], k=2) == pytest.approx(1.0)

    def test_ndcg_no_hit(self):
        from services.retrieval_planner import compute_ndcg_at_k
        assert compute_ndcg_at_k(["x", "y"], ["a", "b"], k=2) == 0.0


class TestKeywordExpansion:
    def test_attendance_expanded(self):
        from services.retrieval_planner import _expand_keywords
        result = _expand_keywords("What is the attendance rule?")
        assert "presence" in result.lower() or "absence" in result.lower()

    def test_no_expansion_for_unknown(self):
        from services.retrieval_planner import _expand_keywords
        result = _expand_keywords("Campus location directions?")
        assert result == "Campus location directions?"
