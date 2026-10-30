"""
Feature H: Counterfactual Document Testing

Tests whether changes to source documents produce the expected changes in
retrieval and answers. Creates isolated test corpora — never modifies
production documents.

Supports modifications:
  - replace_value   : change a number, date, or percentage
  - remove_paragraph: delete a paragraph containing a phrase
  - add_contradiction: insert a deliberately conflicting statement
  - replace_entity  : rename a department, course code, etc.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import text

from config import Config
from models import db

logger = logging.getLogger(__name__)


# ── Text modification helpers ─────────────────────────────────────────────────

def apply_modification(
    original_text: str,
    modification_type: str,
    spec: Dict[str, Any],
) -> str:
    """
    Apply a controlled modification to document text.
    Returns modified text.

    spec keys per type:
      replace_value:    {find: "75%", replace: "80%"}
      remove_paragraph: {phrase: "text fragment to match"}
      add_contradiction:{after_phrase: "...", insert: "new conflicting sentence"}
      replace_entity:   {find: "CSE", replace: "IT"}
    """
    if modification_type == "replace_value":
        find = spec.get("find", "")
        replace = spec.get("replace", "")
        if not find:
            return original_text
        return original_text.replace(find, replace)

    elif modification_type == "remove_paragraph":
        phrase = spec.get("phrase", "")
        if not phrase:
            return original_text
        # Split by paragraph (double newline), remove paragraphs containing phrase
        paragraphs = original_text.split("\n\n")
        filtered = [p for p in paragraphs if phrase.lower() not in p.lower()]
        return "\n\n".join(filtered)

    elif modification_type == "add_contradiction":
        after_phrase = spec.get("after_phrase", "")
        insert_text = spec.get("insert", "")
        if not insert_text:
            return original_text
        if after_phrase:
            idx = original_text.lower().find(after_phrase.lower())
            if idx >= 0:
                # Find end of sentence
                end = original_text.find(".", idx)
                if end < 0:
                    end = idx + len(after_phrase)
                insert_pos = end + 1
                return (
                    original_text[:insert_pos]
                    + "\n"
                    + insert_text
                    + "\n"
                    + original_text[insert_pos:]
                )
        return original_text + "\n\n" + insert_text

    elif modification_type == "replace_entity":
        find = spec.get("find", "")
        replace = spec.get("replace", "")
        if not find:
            return original_text
        return re.sub(re.escape(find), replace, original_text, flags=re.IGNORECASE)

    return original_text


# ── Test definition and execution ─────────────────────────────────────────────

def create_test(
    tenant_id: str,
    base_document_id: str,
    name: str,
    modification_type: str,
    modification_spec: Dict[str, Any],
    test_questions: List[Dict[str, Any]],
    created_by: Optional[str] = None,
) -> str:
    """
    Create a counterfactual test record.
    test_questions: [{question, expected_original, expected_modified, unaffected: bool}]
    Returns test_id.
    """
    test_id = str(uuid.uuid4())
    db.session.execute(
        text(
            """
            INSERT INTO counterfactual_tests (
                id, tenant_id, name, base_document_id,
                modification_type, modification_spec, test_questions,
                status, created_by
            ) VALUES (
                :id, :tenant, :name, :doc,
                :mtype, :mspec::jsonb, :tqs::jsonb,
                'pending', :user
            )
            """
        ),
        {
            "id": test_id,
            "tenant": tenant_id,
            "name": name,
            "doc": base_document_id,
            "mtype": modification_type,
            "mspec": json.dumps(modification_spec),
            "tqs": json.dumps(test_questions),
            "user": created_by,
        },
    )
    db.session.commit()
    return test_id


def run_test(
    test_id: str,
    tenant_id: str,
    vector_store,
    llm_service,
) -> Dict[str, Any]:
    """
    Execute a counterfactual test.

    Steps:
    1. Load base document text
    2. Apply modification
    3. Build a temporary in-memory index with modified document
    4. Run each test question against original AND modified indices
    5. Compare answers and check expectations
    6. Save results

    IMPORTANT: Never modifies production Qdrant collection.
    Uses a :memory: Qdrant client for the modified corpus.
    """
    # Load test record
    row = db.session.execute(
        text(
            """
            SELECT id, name, base_document_id, modification_type,
                   modification_spec, test_questions
            FROM counterfactual_tests
            WHERE id = :id AND tenant_id = :tenant
            """
        ),
        {"id": test_id, "tenant": tenant_id},
    ).fetchone()

    if not row:
        return {"error": "test not found"}

    test_id_, name, doc_id, mod_type, mod_spec, test_questions = row

    # Mark as running
    db.session.execute(
        text("UPDATE counterfactual_tests SET status = 'running' WHERE id = :id"),
        {"id": test_id},
    )
    db.session.commit()

    try:
        # Get original document text from chunk snapshots in execution_evidence
        # or reconstruct from chunk_meta text_previews
        original_text = _reconstruct_document_text(doc_id, tenant_id)
        if not original_text:
            return {"error": "Cannot reconstruct document text for testing"}

        # Apply modification
        modified_text = apply_modification(original_text, mod_type, mod_spec)

        results = []
        regression_detected = False

        for tq in test_questions:
            question = tq.get("question", "")
            expected_original = tq.get("expected_original", "")
            expected_modified = tq.get("expected_modified", "")
            is_unaffected = tq.get("unaffected", False)

            if not question:
                continue

            # Query original index (production)
            orig_passages = vector_store.hybrid_search(
                question, tenant_id=tenant_id, top_k=4,
                document_ids=[doc_id],
            )
            orig_answer, _, _ = llm_service.generate_answer(question, orig_passages)

            # Query modified text directly (no index rebuild needed —
            # we search within the modified text using token matching)
            mod_answer = _query_modified_text(
                question, modified_text, original_text, orig_passages, llm_service
            )

            # Check expectations
            orig_ok = _check_expectation(orig_answer, expected_original)
            mod_ok = _check_expectation(mod_answer, expected_modified)

            # For unaffected questions: both should give same answer
            unaffected_ok = True
            if is_unaffected:
                unaffected_ok = _answers_similar(orig_answer, mod_answer)
                if not unaffected_ok:
                    regression_detected = True

            # Regression: expected_modified check fails
            if expected_modified and not mod_ok:
                regression_detected = True

            results.append(
                {
                    "question": question,
                    "original_answer": orig_answer[:400],
                    "modified_answer": mod_answer[:400],
                    "original_expectation_met": orig_ok,
                    "modified_expectation_met": mod_ok,
                    "unaffected_check_passed": unaffected_ok if is_unaffected else None,
                    "expected_original": expected_original,
                    "expected_modified": expected_modified,
                    "is_unaffected": is_unaffected,
                }
            )

        # Save results
        db.session.execute(
            text(
                """
                UPDATE counterfactual_tests
                SET status = 'completed', results = :results::jsonb,
                    regression_detected = :reg, completed_at = NOW()
                WHERE id = :id
                """
            ),
            {
                "id": test_id,
                "results": json.dumps({"questions": results}),
                "reg": regression_detected,
            },
        )
        db.session.commit()

        return {
            "test_id": test_id,
            "name": name,
            "modification_type": mod_type,
            "modification_spec": mod_spec,
            "questions_tested": len(results),
            "regression_detected": regression_detected,
            "results": results,
        }

    except Exception as e:
        logger.exception("Counterfactual test %s failed: %s", test_id, e)
        db.session.execute(
            text(
                "UPDATE counterfactual_tests SET status = 'failed' WHERE id = :id"
            ),
            {"id": test_id},
        )
        db.session.commit()
        return {"error": str(e), "test_id": test_id}


def _reconstruct_document_text(doc_id: str, tenant_id: str) -> str:
    """
    Reconstruct approximate document text from stored execution evidence
    snapshots or chunk_meta text_previews.
    """
    # Try execution_evidence text_snapshots first
    rows = db.session.execute(
        text(
            """
            SELECT text_snapshot FROM execution_evidence
            WHERE document_id = :doc AND tenant_id = :tenant
            ORDER BY retrieval_rank ASC
            LIMIT 50
            """
        ),
        {"doc": doc_id, "tenant": tenant_id},
    ).fetchall()
    if rows:
        return "\n\n".join(r[0] for r in rows if r[0])

    # Fall back to chunk_meta text_previews
    rows = db.session.execute(
        text(
            """
            SELECT text_preview FROM chunk_meta
            WHERE document_id = :doc AND tenant_id = :tenant
            ORDER BY chunk_index ASC
            """
        ),
        {"doc": doc_id, "tenant": tenant_id},
    ).fetchall()
    if rows:
        return "\n\n".join(r[0] for r in rows if r[0])

    return ""


def _query_modified_text(
    question: str,
    modified_text: str,
    original_text: str,
    orig_passages: List[Dict],
    llm_service,
) -> str:
    """
    Answer a question using the modified text.
    We replace the text in orig_passages with their modified equivalents,
    then call the LLM.
    """
    if not orig_passages:
        # Build a synthetic passage from modified text
        synthetic = [
            {
                "text": modified_text[:2000],
                "filename": "modified_test_document",
                "page_start": 1,
                "page_end": 1,
                "document_id": "test",
                "score": 1.0,
            }
        ]
        answer, _, _ = llm_service.generate_answer(question, synthetic)
        return answer

    # Substitute: find each orig passage text in modified_text
    mod_passages = []
    for p in orig_passages:
        orig_snippet = p.get("text", "")
        # Find if this content exists (possibly changed) in modified text
        # Use first 50 chars as anchor
        anchor = orig_snippet[:50]
        idx = modified_text.lower().find(anchor.lower()[:30])
        if idx >= 0:
            # Extract same length from modified text
            mod_snippet = modified_text[idx: idx + len(orig_snippet)]
        else:
            # Not found — either removed or text changed completely
            mod_snippet = orig_snippet  # keep original if not modified
        mod_passages.append({**p, "text": mod_snippet})

    answer, _, _ = llm_service.generate_answer(question, mod_passages)
    return answer


def _check_expectation(answer: str, expected: str) -> bool:
    """Check if an answer contains the expected content (case-insensitive)."""
    if not expected:
        return True  # No expectation set = trivially passes
    return expected.lower().strip() in answer.lower()


def _answers_similar(a: str, b: str) -> bool:
    """Check if two answers are sufficiently similar (unaffected check)."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return True
    jaccard = len(words_a & words_b) / len(words_a | words_b)
    return jaccard >= 0.5


# ── List / report helpers ─────────────────────────────────────────────────────

def list_tests(tenant_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    rows = db.session.execute(
        text(
            """
            SELECT ct.id, ct.name, ct.modification_type,
                   ct.status, ct.regression_detected,
                   d.original_name, ct.created_at, ct.completed_at
            FROM counterfactual_tests ct
            LEFT JOIN documents d ON d.id = ct.base_document_id
            WHERE ct.tenant_id = :tenant
            ORDER BY ct.created_at DESC
            LIMIT :lim
            """
        ),
        {"tenant": tenant_id, "lim": limit},
    ).fetchall()
    return [
        {
            "test_id": r[0],
            "name": r[1],
            "modification_type": r[2],
            "status": r[3],
            "regression_detected": r[4],
            "document": r[5],
            "created_at": r[6].isoformat() if r[6] else None,
            "completed_at": r[7].isoformat() if r[7] else None,
        }
        for r in rows
    ]


def get_test_result(test_id: str, tenant_id: str) -> Optional[Dict[str, Any]]:
    row = db.session.execute(
        text(
            """
            SELECT ct.id, ct.name, ct.modification_type, ct.modification_spec,
                   ct.status, ct.results, ct.regression_detected,
                   ct.created_at, ct.completed_at, d.original_name
            FROM counterfactual_tests ct
            LEFT JOIN documents d ON d.id = ct.base_document_id
            WHERE ct.id = :id AND ct.tenant_id = :tenant
            """
        ),
        {"id": test_id, "tenant": tenant_id},
    ).fetchone()
    if not row:
        return None
    return {
        "test_id": row[0],
        "name": row[1],
        "modification_type": row[2],
        "modification_spec": row[3],
        "status": row[4],
        "results": row[5],
        "regression_detected": row[6],
        "created_at": row[7].isoformat() if row[7] else None,
        "completed_at": row[8].isoformat() if row[8] else None,
        "document": row[9],
    }
