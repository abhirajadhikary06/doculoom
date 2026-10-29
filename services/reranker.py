"""Cross-encoder reranker with timeout fallback."""
from __future__ import annotations

import logging
import concurrent.futures
from typing import Any, Dict, List, Optional

from config import Config

logger = logging.getLogger(__name__)

# Lazy-loaded singleton
_model = None
_loaded = False


def _get_model():
    """Lazily load the cross-encoder model (only once)."""
    global _model, _loaded
    if _loaded:
        return _model
    _loaded = True
    try:
        from sentence_transformers import CrossEncoder

        _model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        logger.info("Cross-encoder reranker loaded: cross-encoder/ms-marco-MiniLM-L-6-v2")
    except Exception as e:
        logger.warning("Cross-encoder load failed, reranking disabled: %s", e)
        _model = None
    return _model


def rerank(
    query: str,
    passages: List[Dict[str, Any]],
    top_k: int = 4,
    timeout_s: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Rerank passages with a cross-encoder. Falls back to original order on
    timeout or failure.

    Returns the top_k passages sorted by cross-encoder score.
    """
    if not passages:
        return []

    timeout = timeout_s or Config.RERANK_TIMEOUT_S

    model = _get_model()
    if model is None:
        logger.debug("Reranker unavailable, returning top-%d as-is", top_k)
        return passages[:top_k]

    try:
        # Score in a thread so we can enforce a timeout
        pairs = [(query, p.get("text", "")) for p in passages]
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(model.predict, pairs)
            scores = future.result(timeout=timeout)

        # Attach rerank score and sort
        for p, s in zip(passages, scores):
            p["rerank_score"] = float(s)
        reranked = sorted(passages, key=lambda x: x.get("rerank_score", 0), reverse=True)
        logger.info("Reranked %d passages in <%.1fs", len(passages), timeout)
        return reranked[:top_k]
    except concurrent.futures.TimeoutError:
        logger.warning("Rerank timed out after %.1fs, falling back to RRF order", timeout)
    except Exception as e:
        logger.warning("Rerank failed (%s), falling back to RRF order", e)

    return passages[:top_k]
