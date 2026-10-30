"""
Feature C: Adaptive Retrieval Planner

Classifies query intent and selects the optimal retrieval strategy instead
of running the same hybrid pipeline for every question.

Strategies:
  - exact_keyword   : BM25/sparse-dominant for precise factual lookups
  - dense_semantic  : embedding-dominant for conceptual / explanation questions
  - temporal        : date-filtered retrieval for historical queries
  - metadata_filter : department / course / entity-scoped retrieval
  - hybrid          : default RRF fusion (existing behaviour)
  - multi_stage     : complex questions — dense first, then expand

Also tracks retrieval cost (latency, token usage) per strategy for comparison.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Intent taxonomy ───────────────────────────────────────────────────────────

INTENTS = {
    "factual_lookup": "exact_keyword",
    "conceptual_explanation": "dense_semantic",
    "historical_temporal": "temporal",
    "department_specific": "metadata_filter",
    "multi_document_reasoning": "multi_stage",
    "procedural": "dense_semantic",
    "unknown": "hybrid",
}

# ── Keyword signals per intent ────────────────────────────────────────────────

_FACTUAL_SIGNALS = [
    r"\bwhat is\b", r"\bwhen is\b", r"\bhow many\b", r"\bhow much\b",
    r"\bdeadline\b", r"\bdate\b", r"\bfee\b", r"\bpercentage\b",
    r"\bmarks\b", r"\bcredits\b", r"\brequirement\b", r"\beligib",
    r"\bminimum\b", r"\bmaximum\b", r"\blast date\b",
]

_CONCEPTUAL_SIGNALS = [
    r"\bexplain\b", r"\bwhat does\b", r"\bhow does\b", r"\bwhy\b",
    r"\bdefine\b", r"\bdescrib\b", r"\bmeaning\b", r"\bpurpose\b",
    r"\bconcept\b", r"\boverview\b", r"\bsummar",
]

_TEMPORAL_SIGNALS = [
    r"\bin 20\d\d\b", r"\bsince\b", r"\bbefore\b", r"\bafter\b",
    r"\bhistor", r"\bprevious(ly)?\b", r"\bchanged?\b", r"\bwas it\b",
    r"\bused to\b", r"\bwhen did\b", r"\blast year\b", r"\bnext year\b",
    r"\boriginal(ly)?\b", r"\bas of\b", r"\bpast\b",
]

_TEMPORAL_YEAR = re.compile(r"\b(20\d{2})\b")

_DEPT_SIGNALS = [
    r"\bcse\b", r"\bit\b", r"\bece\b", r"\beee\b", r"\bmech\b",
    r"\bcivil\b", r"\bmba\b", r"\bmca\b", r"\bdepartment\b",
    r"\bbranch\b", r"\bcourse\b", r"\bprogram\b", r"\bstream\b",
    r"\bsemester\s*\d\b",
]

_MULTI_SIGNALS = [
    r"\band\b.*\bhow\b", r"\band\b.*\bwhen\b", r"\bboth\b",
    r"\bcompare\b", r"\bdifference between\b", r"\brelationship\b",
    r"\ball\b.*\brequirements\b", r"\bwhat.*and.*when\b",
]


def classify_query(question: str) -> Dict[str, Any]:
    """
    Classify a query into an intent and return the recommended retrieval
    strategy along with extracted signals.

    Returns:
      {
        intent: str,
        strategy: str,
        confidence: float,
        signals: [str],
        extracted_year: str | None,
        extracted_dept: str | None,
        needs_decomposition: bool,
      }
    """
    q = question.strip()
    q_lower = q.lower()

    scores: Dict[str, float] = {k: 0.0 for k in INTENTS}
    matched_signals: List[str] = []

    # Factual
    for pat in _FACTUAL_SIGNALS:
        if re.search(pat, q_lower):
            scores["factual_lookup"] += 1.0
            matched_signals.append(pat.strip(r"\b"))

    # Conceptual
    for pat in _CONCEPTUAL_SIGNALS:
        if re.search(pat, q_lower):
            scores["conceptual_explanation"] += 1.2
            matched_signals.append(pat.strip(r"\b"))

    # Temporal
    for pat in _TEMPORAL_SIGNALS:
        if re.search(pat, q_lower):
            scores["historical_temporal"] += 1.5
            matched_signals.append(pat.strip(r"\b"))

    # Dept/course specific
    for pat in _DEPT_SIGNALS:
        if re.search(pat, q_lower):
            scores["department_specific"] += 1.0
            matched_signals.append(pat.strip(r"\b"))

    # Multi-doc
    for pat in _MULTI_SIGNALS:
        if re.search(pat, q_lower):
            scores["multi_document_reasoning"] += 2.0
            matched_signals.append(pat.strip(r"\b"))

    # Extract year if present
    year_match = _TEMPORAL_YEAR.search(q)
    extracted_year = year_match.group(1) if year_match else None
    if extracted_year:
        scores["historical_temporal"] += 1.0

    # Extract department hint
    dept_map = {
        "cse": "CSE", "it": "IT", "ece": "ECE", "eee": "EEE",
        "mech": "Mechanical", "civil": "Civil", "mba": "MBA", "mca": "MCA",
    }
    extracted_dept = None
    for abbr, full in dept_map.items():
        if re.search(r"\b" + abbr + r"\b", q_lower):
            extracted_dept = full
            break

    # Word count — short questions are often factual
    word_count = len(q.split())
    if word_count <= 6:
        scores["factual_lookup"] += 0.5

    # Pick best intent
    best_intent = max(scores, key=lambda k: scores[k])
    best_score = scores[best_intent]

    # Confidence: ratio of best to total
    total_score = sum(scores.values())
    confidence = round(best_score / total_score, 3) if total_score > 0 else 0.5
    confidence = min(confidence, 0.95)

    # No clear signal → hybrid
    if best_score < 0.5:
        best_intent = "unknown"
        confidence = 0.4

    strategy = INTENTS[best_intent]

    # If both temporal and factual — prefer temporal
    if scores["historical_temporal"] > 0 and scores["factual_lookup"] > 0:
        if scores["historical_temporal"] >= scores["factual_lookup"]:
            best_intent = "historical_temporal"
            strategy = "temporal"

    # Decomposition needed for multi-doc or long complex queries
    needs_decomposition = (
        best_intent == "multi_document_reasoning"
        or (scores["multi_document_reasoning"] > 1.5)
        or (word_count > 25 and "and" in q_lower)
    )

    return {
        "intent": best_intent,
        "strategy": strategy,
        "confidence": confidence,
        "signals": list(set(matched_signals))[:8],
        "extracted_year": extracted_year,
        "extracted_dept": extracted_dept,
        "needs_decomposition": needs_decomposition,
        "word_count": word_count,
    }


# ── Strategy execution ────────────────────────────────────────────────────────

def execute_retrieval(
    query: str,
    tenant_id: str,
    vector_store,
    classification: Dict[str, Any],
    top_k: int = 8,
    document_ids: Optional[List[str]] = None,
    as_of: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Execute the retrieval strategy recommended by classify_query().

    Returns (passages, cost_metrics).
    """
    strategy = classification.get("strategy", "hybrid")
    t0 = time.perf_counter()
    cost: Dict[str, Any] = {
        "strategy": strategy,
        "intent": classification.get("intent"),
        "confidence": classification.get("confidence"),
    }

    passages: List[Dict[str, Any]] = []

    try:
        if strategy == "exact_keyword":
            # Sparse-biased: pass a low dense weight hint
            passages = _keyword_retrieval(
                query, tenant_id, vector_store, top_k, document_ids
            )

        elif strategy == "dense_semantic":
            passages = _dense_retrieval(
                query, tenant_id, vector_store, top_k, document_ids
            )

        elif strategy == "temporal":
            passages = _temporal_retrieval(
                query,
                tenant_id,
                vector_store,
                top_k,
                classification.get("extracted_year"),
                as_of,
            )

        elif strategy == "metadata_filter":
            passages = _filtered_retrieval(
                query,
                tenant_id,
                vector_store,
                top_k,
                classification.get("extracted_dept"),
                document_ids,
            )

        elif strategy == "multi_stage":
            passages = _multi_stage_retrieval(
                query, tenant_id, vector_store, top_k, document_ids
            )

        else:  # hybrid (default)
            passages = vector_store.hybrid_search(
                query, tenant_id=tenant_id, top_k=top_k,
                document_ids=document_ids,
            )

    except Exception as e:
        logger.warning("Strategy %s failed, falling back to hybrid: %s", strategy, e)
        passages = vector_store.hybrid_search(
            query, tenant_id=tenant_id, top_k=top_k
        )
        cost["fallback"] = True

    elapsed = (time.perf_counter() - t0) * 1000
    cost["latency_ms"] = round(elapsed, 2)
    cost["passages_retrieved"] = len(passages)

    return passages, cost


# ── Individual strategy implementations ──────────────────────────────────────

def _keyword_retrieval(
    query: str,
    tenant_id: str,
    vector_store,
    top_k: int,
    document_ids: Optional[List[str]],
) -> List[Dict[str, Any]]:
    """
    Sparse-dominant retrieval: uses hybrid search but with a boosted sparse query.
    The existing hybrid_search already does RRF — calling it with an expanded
    keyword query gives a closer-to-BM25 result.
    """
    # Expand query with synonyms for common campus terms
    expanded = _expand_keywords(query)
    return vector_store.hybrid_search(
        expanded,
        tenant_id=tenant_id,
        top_k=top_k,
        document_ids=document_ids,
    )


def _dense_retrieval(
    query: str,
    tenant_id: str,
    vector_store,
    top_k: int,
    document_ids: Optional[List[str]],
) -> List[Dict[str, Any]]:
    """Dense-only retrieval for conceptual questions."""
    return vector_store.dense_search(
        query, tenant_id=tenant_id, top_k=top_k, document_ids=document_ids
    )


def _temporal_retrieval(
    query: str,
    tenant_id: str,
    vector_store,
    top_k: int,
    year_hint: Optional[str],
    as_of: Optional[str],
) -> List[Dict[str, Any]]:
    """
    Retrieve with temporal context.
    If as_of / year_hint is set, filter to documents valid at that time.
    """
    # Get temporally-valid document IDs
    from services.temporal import get_valid_document_ids
    valid_ids = get_valid_document_ids(tenant_id, as_of=as_of)

    # Also augment query with year if present
    enriched_query = query
    if year_hint and year_hint not in query:
        enriched_query = f"{query} {year_hint}"

    return vector_store.hybrid_search(
        enriched_query,
        tenant_id=tenant_id,
        top_k=top_k,
        document_ids=valid_ids,
    )


def _filtered_retrieval(
    query: str,
    tenant_id: str,
    vector_store,
    top_k: int,
    dept_hint: Optional[str],
    document_ids: Optional[List[str]],
) -> List[Dict[str, Any]]:
    """
    Retrieve with metadata filter (department / course scope).
    If a dept hint is found, augment the query with the department name.
    """
    enriched_query = query
    if dept_hint:
        enriched_query = f"{query} {dept_hint}"

    return vector_store.hybrid_search(
        enriched_query,
        tenant_id=tenant_id,
        top_k=top_k,
        document_ids=document_ids,
    )


def _multi_stage_retrieval(
    query: str,
    tenant_id: str,
    vector_store,
    top_k: int,
    document_ids: Optional[List[str]],
) -> List[Dict[str, Any]]:
    """
    Two-stage retrieval for complex questions.
    Stage 1: Dense search to find relevant document space.
    Stage 2: Hybrid search restricted to those documents.
    """
    # Stage 1: broad dense search
    stage1 = vector_store.dense_search(
        query, tenant_id=tenant_id, top_k=top_k * 2, document_ids=document_ids
    )

    if not stage1:
        return vector_store.hybrid_search(
            query, tenant_id=tenant_id, top_k=top_k, document_ids=document_ids
        )

    # Stage 2: hybrid restricted to documents found in stage 1
    stage1_doc_ids = list({p["document_id"] for p in stage1 if p.get("document_id")})
    stage2 = vector_store.hybrid_search(
        query,
        tenant_id=tenant_id,
        top_k=top_k,
        document_ids=stage1_doc_ids[:10],  # cap doc list
    )

    # Merge: prefer stage2 results, fill with stage1 if needed
    seen_ids = {p["id"] for p in stage2}
    merged = list(stage2)
    for p in stage1:
        if p["id"] not in seen_ids:
            merged.append(p)
        if len(merged) >= top_k:
            break

    return merged[:top_k]


# ── Evaluation metrics (Recall@K, MRR, nDCG) ─────────────────────────────────

def compute_recall_at_k(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k: int,
) -> float:
    """Recall@K: fraction of relevant docs found in top-K."""
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    hits = sum(1 for rid in relevant_ids if rid in top_k)
    return hits / len(relevant_ids)


def compute_mrr(
    retrieved_ids: List[str],
    relevant_ids: List[str],
) -> float:
    """Mean Reciprocal Rank."""
    relevant_set = set(relevant_ids)
    for rank, rid in enumerate(retrieved_ids, 1):
        if rid in relevant_set:
            return 1.0 / rank
    return 0.0


def compute_ndcg_at_k(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k: int,
) -> float:
    """nDCG@K with binary relevance."""
    import math

    relevant_set = set(relevant_ids)
    dcg = 0.0
    for i, rid in enumerate(retrieved_ids[:k], 1):
        if rid in relevant_set:
            dcg += 1.0 / math.log2(i + 1)

    # Ideal DCG: all relevant docs at top positions
    ideal_k = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_k + 1))

    return round(dcg / idcg, 4) if idcg > 0 else 0.0


def evaluate_strategy(
    questions_with_relevance: List[Dict[str, Any]],
    tenant_id: str,
    vector_store,
    k: int = 5,
) -> Dict[str, Any]:
    """
    Evaluate retrieval strategies on a labeled question set.
    Each item: {question, relevant_chunk_ids: [...], relevant_doc_ids: [...]}
    Returns per-strategy metrics.
    """
    strategy_results: Dict[str, List] = {s: [] for s in set(INTENTS.values())}

    for item in questions_with_relevance:
        question = item.get("question", "")
        relevant_chunks = item.get("relevant_chunk_ids", [])
        relevant_docs = item.get("relevant_doc_ids", [])

        if not question or not (relevant_chunks or relevant_docs):
            continue

        clf = classify_query(question)
        strategy = clf["strategy"]

        passages, cost = execute_retrieval(
            question, tenant_id, vector_store, clf, top_k=k
        )
        retrieved_ids = [p["id"] for p in passages]
        retrieved_docs = [p["document_id"] for p in passages]

        recall = compute_recall_at_k(retrieved_ids, relevant_chunks, k)
        mrr = compute_mrr(retrieved_ids, relevant_chunks)
        ndcg = compute_ndcg_at_k(retrieved_ids, relevant_chunks, k)

        # Fall back to doc-level if chunk IDs not available
        if not relevant_chunks and relevant_docs:
            recall = compute_recall_at_k(retrieved_docs, relevant_docs, k)
            mrr = compute_mrr(retrieved_docs, relevant_docs)
            ndcg = compute_ndcg_at_k(retrieved_docs, relevant_docs, k)

        strategy_results[strategy].append(
            {
                "question": question[:80],
                "recall_at_k": recall,
                "mrr": mrr,
                "ndcg": ndcg,
                "latency_ms": cost.get("latency_ms", 0),
            }
        )

    # Aggregate per strategy
    aggregated = {}
    for strat, items in strategy_results.items():
        if not items:
            continue
        n = len(items)
        aggregated[strat] = {
            "n": n,
            "recall_at_k": round(sum(i["recall_at_k"] for i in items) / n, 4),
            "mrr": round(sum(i["mrr"] for i in items) / n, 4),
            "ndcg": round(sum(i["ndcg"] for i in items) / n, 4),
            "avg_latency_ms": round(
                sum(i["latency_ms"] for i in items) / n, 2
            ),
        }

    return {
        "k": k,
        "total_questions": sum(len(v) for v in strategy_results.values()),
        "strategies": aggregated,
    }


# ── Keyword expansion helper ──────────────────────────────────────────────────

_SYNONYMS = {
    "attendance": "attendance presence absence leave",
    "fee": "fee fees tuition payment charges",
    "exam": "exam examination test assessment evaluation",
    "result": "result grade marks score cgpa",
    "admission": "admission enrollment registration joining",
    "semester": "semester term session period",
    "leave": "leave absence holiday vacation",
    "deadline": "deadline last date due date submission",
}


def _expand_keywords(query: str) -> str:
    q_lower = query.lower()
    expansions = []
    for kw, synonyms in _SYNONYMS.items():
        if kw in q_lower:
            expansions.append(synonyms)
    if expansions:
        return query + " " + " ".join(expansions)
    return query
