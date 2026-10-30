"""Tests for Feature H: Counterfactual Document Testing."""
from __future__ import annotations

import pytest


class TestTextModification:
    def test_replace_value(self):
        from services.counterfactual import apply_modification
        text = "Students must attend 75% of all classes."
        result = apply_modification(text, "replace_value", {"find": "75%", "replace": "80%"})
        assert "80%" in result
        assert "75%" not in result

    def test_remove_paragraph(self):
        from services.counterfactual import apply_modification
        text = "First paragraph about fees.\n\nSecond paragraph about attendance requirements.\n\nThird paragraph."
        result = apply_modification(text, "remove_paragraph", {"phrase": "attendance requirements"})
        assert "attendance requirements" not in result
        assert "fees" in result
        assert "Third paragraph" in result

    def test_add_contradiction(self):
        from services.counterfactual import apply_modification
        text = "The attendance requirement is 75%. Students must comply."
        result = apply_modification(
            text,
            "add_contradiction",
            {
                "after_phrase": "75%",
                "insert": "However, some departments require only 60% attendance.",
            },
        )
        assert "60%" in result
        assert "75%" in result

    def test_replace_entity(self):
        from services.counterfactual import apply_modification
        text = "CSE department students must attend 75% of classes."
        result = apply_modification(text, "replace_entity", {"find": "CSE", "replace": "ECE"})
        assert "ECE" in result
        assert "CSE" not in result

    def test_unknown_modification_returns_original(self):
        from services.counterfactual import apply_modification
        text = "Original text unchanged."
        result = apply_modification(text, "unknown_type", {})
        assert result == text

    def test_replace_value_empty_find_returns_original(self):
        from services.counterfactual import apply_modification
        text = "Original text."
        result = apply_modification(text, "replace_value", {"find": "", "replace": "something"})
        assert result == text


class TestExpectationCheck:
    def test_expectation_met(self):
        from services.counterfactual import _check_expectation
        assert _check_expectation("The attendance is 80%.", "80%") is True

    def test_expectation_not_met(self):
        from services.counterfactual import _check_expectation
        assert _check_expectation("The attendance is 75%.", "80%") is False

    def test_empty_expectation_always_passes(self):
        from services.counterfactual import _check_expectation
        assert _check_expectation("Any answer", "") is True


class TestAnswerSimilarity:
    def test_identical_answers_similar(self):
        from services.counterfactual import _answers_similar
        a = "Attendance must be 75% per semester for all students."
        assert _answers_similar(a, a) is True

    def test_very_different_answers_not_similar(self):
        from services.counterfactual import _answers_similar
        a = "Attendance must be 75% per semester."
        b = "The library opens at 9am on weekdays and holidays."
        assert _answers_similar(a, b) is False

    def test_empty_answers_similar(self):
        from services.counterfactual import _answers_similar
        assert _answers_similar("", "") is True


class TestCounterfactualAPI:
    def test_create_test(self, client, auth_headers):
        resp = client.post(
            "/api/counterfactual",
            headers=auth_headers,
            json={
                "base_document_id": "test-doc-id",
                "name": "Attendance test",
                "modification_type": "replace_value",
                "modification_spec": {"find": "75%", "replace": "80%"},
                "test_questions": [
                    {
                        "question": "What is the attendance requirement?",
                        "expected_original": "75%",
                        "expected_modified": "80%",
                    }
                ],
            },
        )
        assert resp.status_code == 201
        data = resp.get_json()
        assert "test_id" in data

    def test_list_tests(self, client, auth_headers):
        resp = client.get("/api/counterfactual", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.get_json()
        assert "tests" in data
