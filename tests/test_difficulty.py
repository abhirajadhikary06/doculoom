"""Tests for Feature O: Question Difficulty Estimator."""
from __future__ import annotations

import pytest


class TestDifficultyEstimation:
    def test_simple_factual(self):
        from services.difficulty import estimate_difficulty
        result = estimate_difficulty("What is the attendance percentage?")
        assert result["difficulty_class"] == "simple_factual"
        assert result["complexity_score"] < 0.5

    def test_temporal_comparison(self):
        from services.difficulty import estimate_difficulty
        result = estimate_difficulty("Has the attendance requirement changed since 2023?")
        assert result["difficulty_class"] == "temporal_comparison"
        assert result["complexity_score"] > 0.5

    def test_multi_document(self):
        from services.difficulty import estimate_difficulty
        result = estimate_difficulty("List all requirements for all departments across all semesters")
        assert result["difficulty_class"] == "multi_document"
        assert result["needs_decomposition"] is True

    def test_semantic_explanation(self):
        from services.difficulty import estimate_difficulty
        result = estimate_difficulty("Explain the grading system in detail")
        assert result["difficulty_class"] == "semantic_explanation"

    def test_short_factual_low_llm_calls(self):
        from services.difficulty import estimate_difficulty
        result = estimate_difficulty("Exam fee?")
        assert result["estimated_llm_calls"] <= 1

    def test_complexity_ordering(self):
        from services.difficulty import estimate_difficulty
        simple = estimate_difficulty("What is the fee?")
        complex_ = estimate_difficulty("Compare all attendance rules across all years and departments")
        assert simple["complexity_score"] < complex_["complexity_score"]


class TestDifficultyHelpers:
    def test_skip_rerank_for_simple(self):
        from services.difficulty import should_skip_rerank
        diff = {"difficulty_class": "simple_factual", "complexity_score": 0.1}
        assert should_skip_rerank(diff) is True

    def test_no_skip_rerank_for_complex(self):
        from services.difficulty import should_skip_rerank
        diff = {"difficulty_class": "multi_document", "complexity_score": 0.85}
        assert should_skip_rerank(diff) is False

    def test_suggested_top_k_scales(self):
        from services.difficulty import suggested_top_k
        simple_diff = {"estimated_passages": 1}
        multi_diff = {"estimated_passages": 6}
        assert suggested_top_k(simple_diff, 8) <= 8
        assert suggested_top_k(multi_diff, 8) >= 6
