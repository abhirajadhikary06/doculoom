"""Tests for Feature J: Query Decomposition Engine."""
from __future__ import annotations

import pytest


class TestDecompositionNeed:
    def test_simple_factual_no_decomposition(self):
        from services.decomposition import detect_decomposition_need
        clf = {"needs_decomposition": False, "intent": "factual_lookup"}
        assert detect_decomposition_need("What is the attendance requirement?", clf) is False

    def test_complex_conjunction_needs_decomposition(self):
        from services.decomposition import detect_decomposition_need
        clf = {"needs_decomposition": False, "intent": "factual_lookup"}
        q = "What are the attendance requirements for CSE students, and how have they changed since 2024, and what fees apply?"
        assert detect_decomposition_need(q, clf) is True

    def test_classifier_flag_triggers_decomposition(self):
        from services.decomposition import detect_decomposition_need
        clf = {"needs_decomposition": True, "intent": "multi_document_reasoning"}
        assert detect_decomposition_need("Simple question?", clf) is True

    def test_compare_keyword_triggers(self):
        from services.decomposition import detect_decomposition_need
        clf = {"needs_decomposition": False}
        assert detect_decomposition_need("Compare the attendance rules for 2023 and 2025", clf) is True

    def test_long_question_triggers(self):
        from services.decomposition import detect_decomposition_need
        clf = {"needs_decomposition": False}
        long_q = " ".join(["word"] * 35)
        assert detect_decomposition_need(long_q, clf) is True


class TestConsistencyCheck:
    def test_no_issues_for_consistent_answers(self):
        from services.decomposition import _check_consistency
        results = [
            {"question": "Q1", "answer": "Attendance is 75% minimum."},
            {"question": "Q2", "answer": "The fee is Rs 500."},
        ]
        issues = _check_consistency(results)
        assert len(issues) == 0

    def test_percentage_conflict_detected(self):
        from services.decomposition import _check_consistency
        results = [
            {"question": "Q1", "answer": "Attendance is 75% for CSE."},
            {"question": "Q2", "answer": "Attendance is 80% for all students."},
        ]
        issues = _check_consistency(results)
        assert len(issues) > 0
        assert "75%" in issues[0] or "80%" in issues[0]

    def test_empty_results_no_issues(self):
        from services.decomposition import _check_consistency
        assert _check_consistency([]) == []


class TestDecomposeFallback:
    def test_no_llm_returns_original(self):
        from services.decomposition import decompose_query
        result = decompose_query("Complex multi-part question here?", llm_service=None)
        assert result == ["Complex multi-part question here?"]
