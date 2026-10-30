"""
Feature O: Question Difficulty Estimator

Estimates complexity of a question before retrieval to inform the adaptive
retrieval planner and avoid unnecessary LLM calls for simple questions.

Difficulty classes:
  simple_factual        — single-hop, specific fact lookup
  semantic_explanation  — conceptual, requires synthesis
  multi_document        — requires information from multiple docs
  temporal_comparison   — comparing values across time periods
  entity_relationship   — structured relationship queries
  insufficient_info     — likely unanswerable (no relevant docs exist)
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

# ── Difficulty taxonomy ───────────────────────────────────────────────────────

DIFFICULTY_CLASSES = [
    "simple_factual",
    "semantic_explanation",
    "multi_document",
    "temporal_comparison",
    "entity_relationship",
    "insufficient_info",
]

# Estimated retrieval requirements per class
_REQUIREMENTS = {
    "simple_factual":       {"passages": 1, "documents": 1, "decompose": False, "llm_calls": 1},
    "semantic_explanation": {"passages": 3, "documents": 2, "decompose": False, "llm_calls": 1},
    "multi_document":       {"passages": 6, "documents": 3, "decompose": True,  "llm_calls": 3},
    "temporal_comparison":  {"passages": 4, "documents": 2, "decompose": True,  "llm_calls": 2},
    "entity_relationship":  {"passages": 4, "documents": 2, "decompose": False, "llm_calls": 1},
    "insufficient_info":    {"passages": 0, "documents": 0, "decompose": False, "llm_calls": 0},
}

# Signals → difficulty class with weights
_SIGNALS: List[tuple] = [
    # Simple factual
    (r"\bwhat is the\b",                    "simple_factual",       1.5),
    (r"\bwhen is\b",                        "simple_factual",       1.5),
    (r"\bwhat is the deadline\b",           "simple_factual",       2.0),
    (r"\bhow many\b",                       "simple_factual",       1.5),
    (r"\bhow much\b",                       "simple_factual",       1.5),
    (r"\bwhat is the fee\b",                "simple_factual",       2.0),
    (r"\bwhat percentage\b",                "simple_factual",       2.0),
    (r"\bwhat are the marks\b",             "simple_factual",       2.0),

    # Semantic explanation
    (r"\bexplain\b",                        "semantic_explanation", 2.0),
    (r"\bhow does\b",                       "semantic_explanation", 1.5),
    (r"\bwhy\b",                            "semantic_explanation", 1.2),
    (r"\bwhat does\b",                      "semantic_explanation", 1.0),
    (r"\bdescrib\b",                        "semantic_explanation", 1.5),
    (r"\bwhat is the purpose\b",            "semantic_explanation", 2.0),
    (r"\boverview\b",                       "semantic_explanation", 1.5),

    # Multi-document
    (r"\ball\b.{3,}\brequirements\b",       "multi_document",       2.5),
    (r"\blist all\b",                       "multi_document",       2.0),
    (r"\bevery\b",                          "multi_document",       1.5),
    (r"\bcomprehensive\b",                  "multi_document",       2.0),
    (r"\bacross all\b",                     "multi_document",       2.5),
    (r"\bfor all departments\b",            "multi_document",       2.5),

    # Temporal comparison
    (r"\bhas.{2,20}changed\b",              "temporal_comparison",  2.5),
    (r"\bwas.{2,20}in 20\d\d\b",           "temporal_comparison",  2.0),
    (r"\bdifference between.{3,30}20\d\d",  "temporal_comparison",  2.5),
    (r"\bhistorically\b",                   "temporal_comparison",  2.0),
    (r"\bprevious.{2,20}regulation\b",      "temporal_comparison",  2.0),
    (r"\bwhen did.{2,30}change\b",          "temporal_comparison",  2.5),
    (r"\bbefore.{2,20}20\d\d\b",            "temporal_comparison",  1.5),
    (r"\bsince 20\d\d\b",                   "temporal_comparison",  1.5),

    # Entity relationship
    (r"\brelationship between\b",           "entity_relationship",  2.5),
    (r"\bwhat courses\b.{3,30}\bunder\b",   "entity_relationship",  2.0),
    (r"\bwhich departments\b",              "entity_relationship",  2.0),
    (r"\blinked to\b",                      "entity_relationship",  1.5),
    (r"\bbelongs to\b",                     "entity_relationship",  1.5),

    # Insufficient info indicators
    (r"\bhostel refund\b",                  "insufficient_info",    1.0),
    (r"\bcanteen menu\b",                   "insufficient_info",    1.5),
    (r"\bbus schedule\b",                   "insufficient_info",    1.0),
    (r"\bplacement statistics\b",           "insufficient_info",    1.0),
]


def estimate_difficulty(
    question: str,
    classification: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Estimate question difficulty class and retrieval requirements.

    Returns:
      {
        difficulty_class: str,
        confidence: float,
        estimated_passages: int,
        estimated_documents: int,
        needs_decomposition: bool,
        estimated_llm_calls: int,
        complexity_score: float,   # 0 (simple) to 1 (complex)
        signals: [str],
      }
    """
    q = question.strip()
    q_lower = q.lower()

    scores: Dict[str, float] = {dc: 0.0 for dc in DIFFICULTY_CLASSES}
    matched_signals: List[str] = []

    for pattern, dc, weight in _SIGNALS:
        if re.search(pattern, q_lower):
            scores[dc] += weight
            matched_signals.append(pattern.strip(r"\b"))

    # Use planner classification if available
    if classification:
        intent = classification.get("intent", "")
        intent_map = {
            "factual_lookup": "simple_factual",
            "conceptual_explanation": "semantic_explanation",
            "historical_temporal": "temporal_comparison",
            "multi_document_reasoning": "multi_document",
            "department_specific": "entity_relationship",
        }
        if intent in intent_map:
            scores[intent_map[intent]] += 1.5

        if classification.get("needs_decomposition"):
            scores["multi_document"] += 1.0

    word_count = len(q.split())
    # Longer questions tend to be more complex
    if word_count > 25:
        scores["multi_document"] += 1.0
    elif word_count > 15:
        scores["temporal_comparison"] += 0.5

    # Conjunctions → multi-document
    conj_count = len(re.findall(r"\band\b|\bor\b|\bbut\b", q_lower))
    if conj_count >= 2:
        scores["multi_document"] += conj_count * 0.5

    # Pick best
    best_class = max(scores, key=lambda k: scores[k])
    best_score = scores[best_class]
    total = sum(scores.values())
    confidence = round(best_score / total, 3) if total > 0 else 0.5

    if best_score < 0.5:
        # Default to simple if no clear signals
        best_class = "simple_factual"
        confidence = 0.4

    req = _REQUIREMENTS[best_class]

    # Complexity score: 0 for simple_factual, 1 for multi_document/temporal
    complexity_map = {
        "simple_factual": 0.1,
        "semantic_explanation": 0.35,
        "entity_relationship": 0.45,
        "temporal_comparison": 0.65,
        "multi_document": 0.85,
        "insufficient_info": 0.2,
    }
    complexity_score = complexity_map.get(best_class, 0.5)

    return {
        "difficulty_class": best_class,
        "confidence": confidence,
        "estimated_passages": req["passages"],
        "estimated_documents": req["documents"],
        "needs_decomposition": req["decompose"],
        "estimated_llm_calls": req["llm_calls"],
        "complexity_score": complexity_score,
        "signals": list(set(matched_signals))[:6],
        "word_count": word_count,
        "all_scores": {k: round(v, 2) for k, v in scores.items()},
    }


def should_skip_rerank(difficulty: Dict[str, Any]) -> bool:
    """
    Skip expensive cross-encoder reranking for simple factual questions
    that only need 1 passage.
    """
    return (
        difficulty.get("difficulty_class") == "simple_factual"
        and difficulty.get("complexity_score", 1.0) < 0.2
    )


def should_skip_llm(difficulty: Dict[str, Any]) -> bool:
    """
    For insufficient_info queries, skip LLM generation (answer will be
    the abstention phrase anyway).
    """
    return difficulty.get("difficulty_class") == "insufficient_info"


def suggested_top_k(difficulty: Dict[str, Any], base_top_k: int = 8) -> int:
    """Suggest a top_k value based on difficulty."""
    passages = difficulty.get("estimated_passages", base_top_k)
    return max(passages, min(base_top_k, passages * 2))
