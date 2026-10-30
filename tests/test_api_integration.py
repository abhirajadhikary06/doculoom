"""
Integration tests for existing and new API endpoints.
These tests verify that all routes respond correctly with auth and
correct input. They do NOT test LLM calls (those require a live API key).
"""
from __future__ import annotations

import pytest


class TestExistingEndpoints:
    """Verify existing endpoints still work (backward compatibility)."""

    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ok"
        assert data["service"] == "doculoom"

    def test_register_and_login(self, client):
        resp = client.post(
            "/api/auth/register",
            json={"email": "integ@test.edu", "password": "pass1234"},
        )
        assert resp.status_code in (201, 409)

        resp = client.post(
            "/api/auth/login",
            json={"email": "integ@test.edu", "password": "pass1234"},
        )
        assert resp.status_code == 200
        assert "token" in resp.get_json()

    def test_me_endpoint(self, client, auth_headers):
        resp = client.get("/api/auth/me", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.get_json()
        assert "email" in data

    def test_list_docs(self, client, auth_headers):
        resp = client.get("/api/docs", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.get_json()
        assert "documents" in data

    def test_ask_requires_question(self, client, auth_headers):
        resp = client.post("/api/ask", json={}, headers=auth_headers)
        assert resp.status_code == 400

    def test_list_conversations(self, client, auth_headers):
        resp = client.get("/api/conversations", headers=auth_headers)
        assert resp.status_code == 200


class TestTemporalEndpoints:
    def test_timeline_requires_name(self, client, auth_headers):
        resp = client.get("/api/temporal/timeline", headers=auth_headers)
        assert resp.status_code == 400

    def test_timeline_with_name(self, client, auth_headers):
        resp = client.get("/api/temporal/timeline?name=attendance", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.get_json()
        assert "timeline" in data

    def test_supersessions_requires_doc_id(self, client, auth_headers):
        resp = client.get("/api/temporal/supersessions", headers=auth_headers)
        assert resp.status_code == 400

    def test_versions_for_unknown_doc(self, client, auth_headers):
        resp = client.get("/api/temporal/versions/nonexistent-doc", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["versions"] == []


class TestVerifyEndpoint:
    def test_verify_requires_answer(self, client, auth_headers):
        resp = client.post("/api/verify", json={}, headers=auth_headers)
        assert resp.status_code == 400

    def test_verify_returns_structure(self, client, auth_headers):
        resp = client.post(
            "/api/verify",
            json={
                "answer": "Attendance must be 75%. Fees are due in November.",
                "passages": [{"text": "Attendance is 75%.", "filename": "p.pdf", "page_start": 1}],
            },
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert "total_claims" in data
        assert "supported_claims" in data
        assert "citation_coverage" in data


class TestRetrievalPlannerEndpoints:
    def test_classify_requires_question(self, client, auth_headers):
        resp = client.post("/api/retrieval/classify", json={}, headers=auth_headers)
        assert resp.status_code == 400

    def test_classify_returns_strategy(self, client, auth_headers):
        resp = client.post(
            "/api/retrieval/classify",
            json={"question": "What is the attendance requirement?"},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert "classification" in data
        assert "strategy" in data["classification"]
        assert "difficulty" in data


class TestConflictsEndpoints:
    def test_get_conflicts(self, client, auth_headers):
        resp = client.get("/api/conflicts", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.get_json()
        assert "conflicts" in data

    def test_resolve_nonexistent_conflict(self, client, auth_headers):
        resp = client.post(
            "/api/conflicts/nonexistent-id/resolve",
            headers=auth_headers,
        )
        assert resp.status_code == 404


class TestExpiryEndpoints:
    def test_expiry_report(self, client, auth_headers):
        resp = client.get("/api/expiry", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.get_json()
        assert "report" in data
        assert "summary" in data

    def test_refresh_expiry(self, client, auth_headers):
        resp = client.post("/api/expiry/refresh", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.get_json()
        assert "status_counts" in data


class TestCacheEndpoints:
    def test_cache_metrics(self, client, auth_headers):
        resp = client.get("/api/cache/metrics", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.get_json()
        assert "exact_hit_rate" in data
        assert "semantic_hit_rate" in data

    def test_cache_invalidate(self, client, auth_headers):
        resp = client.post("/api/cache/invalidate", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.get_json()
        assert "invalidated" in data


class TestDriftEndpoints:
    def test_drift_history_empty(self, client, auth_headers):
        resp = client.get("/api/drift/history", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.get_json()
        assert "history" in data

    def test_add_baseline_requires_question(self, client, auth_headers):
        resp = client.post("/api/drift/baseline", json={}, headers=auth_headers)
        assert resp.status_code == 400

    def test_add_baseline_question(self, client, auth_headers):
        resp = client.post(
            "/api/drift/baseline",
            json={
                "question": "What is the attendance requirement?",
                "expected_keywords": ["75%", "attendance"],
            },
            headers=auth_headers,
        )
        assert resp.status_code == 201
        data = resp.get_json()
        assert "baseline_id" in data

    def test_list_baseline(self, client, auth_headers):
        resp = client.get("/api/drift/baseline", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.get_json()
        assert "questions" in data


class TestProvenanceEndpoints:
    def test_list_provenance(self, client, auth_headers):
        resp = client.get("/api/provenance", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.get_json()
        assert "logs" in data

    def test_get_nonexistent_trace(self, client, auth_headers):
        resp = client.get("/api/provenance/nonexistent-id", headers=auth_headers)
        assert resp.status_code == 404


class TestAdminDashboard:
    def test_dashboard_returns_structure(self, client, auth_headers):
        resp = client.get("/api/admin/dashboard", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.get_json()
        assert "tenant_id" in data
        assert "documents" in data
        assert "knowledge_gaps" in data
        assert "cache" in data


class TestAskV2Endpoint:
    def test_ask_v2_requires_question(self, client, auth_headers):
        resp = client.post("/api/ask/v2", json={}, headers=auth_headers)
        assert resp.status_code == 400

    def test_ask_v2_returns_intelligence_fields(self, client, auth_headers):
        resp = client.post(
            "/api/ask/v2",
            json={"question": "What is the attendance policy?"},
            headers=auth_headers,
        )
        # May return 402 if no credits, 200 if ok
        assert resp.status_code in (200, 402)
        if resp.status_code == 200:
            data = resp.get_json()
            assert "answer" in data
            assert "difficulty" in data
            assert "retrieval_strategy" in data

    def test_ask_v2_credit_response(self, client, auth_headers):
        # Just verify the endpoint exists and handles the request
        resp = client.post(
            "/api/ask/v2",
            json={"question": "Test question for v2 endpoint"},
            headers=auth_headers,
        )
        assert resp.status_code in (200, 402)
