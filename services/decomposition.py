"""
Feature J: Query Decomposition Engine

Handles complex questions by decomposing them into independent sub-questions,
running retrieval for each, aggregating evidence, checking cross-sub-question
consistency, and synthesising a final cited answer.

Only decomposes when the query classifier signals multi_document_reasoning
or the question has clear conjunctive structure — not for simple factual
lookups.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Maximum sub-questions to generate (guards against runaway decomposition)
MAX_SUBQUESTIONS = 4

DECOMPOSE_PROMPT = """You are a query decomposition assistant for a campus knowledge
retrieval system.

Given a complex question, break it into at most {max_sub} independent, self-contained
sub-questions that together fully answer the original question.

Rules:
- Each sub-question must be answerable independently from campus documents
- Use plain English; do NOT include document names or file references
- Return ONLY valid JSON — no prose, no markdown fences:
  {{"sub_questions": ["...", "...", ...]}}
- If the question is already simple and self-contained, return:
  {{"sub_questions": [<original question>]}}"""

SYNTHESIS_PROMPT = """You are a campus helpdesk assistant synthesising answers
from multiple sub-question answers into one coherent final response.

Rules:
- Combine all sub-answers into one well-structured answer
- Include ALL inline citations from the sub-answers ([filename, page N])
- Do not invent new facts — only use what is in the sub-answers
- Check for consistency: if sub-answers contradict each other, say so explicitly
- Be concise and factual"""


def detect_decomposition_need(question: str, classification: Dict[str, Any]) -> bool:
    """
    Return True if this question needs decomposition.
    Uses classifier output + heuristic signals.
    """
    if classification.get("needs_decomposition"):
        return True

    q = question.lower()
    # Explicit conjunctive patterns
    patterns = [
        r"\band\b.{5,}\band\b",        # "X and Y and Z"
        r"\bboth\b.{3,}\band\b",        # "both X and Y"
        r"\bcompare\b",                 # compare questions
        r"\bdifference between\b",
        r"\bhow.{3,}and.{3,}when\b",    # "how X and when Y"
        r"\bwhat.{3,}and.{3,}how\b",
        r"\bhas.{3,}changed\b.{0,30}\b(since|from)\b",
    ]
    for pat in patterns:
        if re.search(pat, q):
            return True

    # Long questions are candidates
    if len(question.split()) > 30:
        return True

    return False


def decompose_query(
    question: str,
    llm_service,
    max_sub: int = MAX_SUBQUESTIONS,
) -> List[str]:
    """
    Use LLM to decompose a complex question into sub-questions.
    Falls back to returning [question] if decomposition fails.
    """
    if not llm_service or not llm_service.client:
        return [question]

    try:
        import json
        messages = [
            {
                "role": "system",
                "content": DECOMPOSE_PROMPT.format(max_sub=max_sub),
            },
            {"role": "user", "content": f"Complex question: {question}"},
        ]
        raw, _ = llm_service._chat(messages, temperature=0.0)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()

        data = json.loads(raw)
        subs = data.get("sub_questions", [])
        # Validate: non-empty strings, reasonable count
        subs = [s.strip() for s in subs if isinstance(s, str) and s.strip()][:max_sub]
        if subs:
            logger.info(
                "Decomposed %r into %d sub-questions", question[:60], len(subs)
            )
            return subs
    except Exception as e:
        logger.debug("Query decomposition failed: %s", e)

    return [question]


def execute_decomposed_retrieval(
    question: str,
    tenant_id: str,
    vector_store,
    llm_service,
    classification: Dict[str, Any],
    top_k: int = 4,
    as_of: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Full decomposition pipeline:
    1. Decompose question into sub-questions
    2. Retrieve passages for each sub-question (adaptive strategy)
    3. Generate per-sub-question answers
    4. Check cross-consistency
    5. Synthesise final answer

    Returns:
      {
        sub_questions: [...],
        sub_results: [{question, passages, answer, sources}],
        consistency_issues: [...],
        final_answer: str,
        all_sources: [...],
        was_decomposed: bool,
      }
    """
    from services.retrieval_planner import classify_query, execute_retrieval

    sub_questions = decompose_query(question, llm_service)
    was_decomposed = len(sub_questions) > 1

    sub_results = []
    all_passages = []
    all_sources = []

    for sub_q in sub_questions:
        # Classify and retrieve for this sub-question
        sub_clf = classify_query(sub_q)
        passages, cost = execute_retrieval(
            sub_q,
            tenant_id,
            vector_store,
            sub_clf,
            top_k=top_k,
            as_of=as_of,
        )

        if not passages:
            sub_results.append(
                {
                    "question": sub_q,
                    "passages": [],
                    "answer": "I don't know based on the available campus documents.",
                    "sources": [],
                    "strategy": sub_clf.get("strategy"),
                }
            )
            continue

        # Generate answer for this sub-question
        sub_answer, sub_sources, _ = llm_service.generate_answer(sub_q, passages)

        sub_results.append(
            {
                "question": sub_q,
                "passages": passages,
                "answer": sub_answer,
                "sources": sub_sources,
                "strategy": sub_clf.get("strategy"),
                "retrieval_cost": cost,
            }
        )
        all_passages.extend(passages)

        # Deduplicate sources
        seen_keys = {
            (s.get("filename"), s.get("page"))
            for s in all_sources
        }
        for s in sub_sources:
            key = (s.get("filename"), s.get("page"))
            if key not in seen_keys:
                all_sources.append(s)
                seen_keys.add(key)

    # Check cross-sub-question consistency
    consistency_issues = _check_consistency(sub_results)

    # Synthesise final answer
    final_answer = _synthesise(question, sub_results, llm_service, consistency_issues)

    return {
        "sub_questions": sub_questions,
        "sub_results": [
            {
                "question": r["question"],
                "answer": r["answer"],
                "sources": r.get("sources", []),
                "strategy": r.get("strategy"),
            }
            for r in sub_results
        ],
        "consistency_issues": consistency_issues,
        "final_answer": final_answer,
        "all_sources": all_sources,
        "was_decomposed": was_decomposed,
        "total_passages": len(all_passages),
    }


def _check_consistency(sub_results: List[Dict[str, Any]]) -> List[str]:
    """
    Simple consistency check: look for numeric contradictions across sub-answers.
    Returns list of issue descriptions.
    """
    issues = []
    answers = [(r["question"], r["answer"]) for r in sub_results]

    # Extract percentages from each answer
    pct_by_sub: Dict[str, List[str]] = {}
    for q, a in answers:
        pcts = re.findall(r"\d+(?:\.\d+)?\s*%", a)
        if pcts:
            pct_by_sub[q[:40]] = pcts

    # Check for different percentages across sub-answers
    all_pcts = set()
    for pcts in pct_by_sub.values():
        all_pcts.update(pcts)

    if len(all_pcts) > 1:
        issues.append(
            f"Sub-answers contain different percentage values: "
            f"{', '.join(sorted(all_pcts))}. "
            f"This may indicate conflicting information or different contexts."
        )

    return issues


def _synthesise(
    original_question: str,
    sub_results: List[Dict[str, Any]],
    llm_service,
    consistency_issues: List[str],
) -> str:
    """Synthesise a final answer from all sub-question answers."""
    if not sub_results:
        return "I don't know based on the available campus documents."

    if len(sub_results) == 1:
        return sub_results[0]["answer"]

    if not llm_service or not llm_service.client:
        # Fallback: concatenate
        parts = []
        for r in sub_results:
            parts.append(f"**{r['question']}**\n{r['answer']}")
        return "\n\n".join(parts)

    # Build synthesis context
    context_parts = []
    for i, r in enumerate(sub_results, 1):
        context_parts.append(
            f"Sub-question {i}: {r['question']}\n"
            f"Answer: {r['answer']}"
        )

    if consistency_issues:
        context_parts.append(
            "\nNOTE — Potential consistency issues detected:\n"
            + "\n".join(f"- {issue}" for issue in consistency_issues)
        )

    context = "\n\n---\n\n".join(context_parts)

    try:
        messages = [
            {"role": "system", "content": SYNTHESIS_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Original question: {original_question}\n\n"
                    f"Sub-question answers:\n\n{context}\n\n"
                    "Please synthesise a final answer."
                ),
            },
        ]
        answer, _ = llm_service._chat(messages, temperature=0.1)
        return answer.strip()
    except Exception as e:
        logger.warning("Synthesis failed: %s", e)
        # Fallback
        parts = [r["answer"] for r in sub_results if r.get("answer")]
        return " ".join(parts)
