"""
Feature B: Evidence Verification Engine

Verifies individual answer claims rather than trusting citation presence.
Extracts atomic claims from generated answers, aligns each claim to a
specific passage, classifies each as supported / unsupported / insufficient,
and returns structured verification metadata.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Claim extraction ──────────────────────────────────────────────────────────

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

def extract_claims(answer: str) -> List[str]:
    """
    Split an answer into atomic claims (sentences / bullet points).
    Filters out meta-phrases like "according to", "based on", etc.
    """
    # Remove citation markers first
    clean = re.sub(r"\[[^\]]+,\s*page\s*\d+[^\]]*\]", "", answer, flags=re.IGNORECASE)
    clean = re.sub(r"\[\d+\]", "", clean)

    # Split on newlines + sentence boundaries
    raw_lines = []
    for line in clean.split("\n"):
        line = line.strip().lstrip("-•·*▶►").strip()
        if not line:
            continue
        # Further split on sentence endings
        sentences = _SENTENCE_END.split(line)
        raw_lines.extend(s.strip() for s in sentences if s.strip())

    # Filter: minimum length, not a meta-phrase
    meta_phrases = {
        "based on the", "according to", "as mentioned", "as stated",
        "the document", "the passage", "i don't know", "unfortunately",
        "please note", "note that",
    }
    claims = []
    for s in raw_lines:
        if len(s) < 15:
            continue
        if any(s.lower().startswith(mp) for mp in meta_phrases) and len(s) < 80:
            continue
        claims.append(s)

    return claims


# ── Claim-passage alignment ───────────────────────────────────────────────────

def _token_overlap(a: str, b: str) -> float:
    """Simple unigram Jaccard overlap between two strings."""
    tokens_a = set(re.findall(r"\b\w+\b", a.lower()))
    tokens_b = set(re.findall(r"\b\w+\b", b.lower()))
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def align_claim_to_passages(
    claim: str,
    passages: List[Dict[str, Any]],
    threshold: float = 0.12,
) -> List[Dict[str, Any]]:
    """
    Find passages that support a given claim by token overlap.
    Returns list of {passage_idx, score, text_preview, filename, page} sorted
    by descending score.
    """
    matches = []
    for i, p in enumerate(passages):
        text = p.get("text", "")
        score = _token_overlap(claim, text)
        if score >= threshold:
            matches.append(
                {
                    "passage_idx": i,
                    "score": round(score, 4),
                    "text_preview": text[:200],
                    "filename": p.get("filename", ""),
                    "page": p.get("page_start", 1),
                    "document_id": p.get("document_id", ""),
                }
            )
    return sorted(matches, key=lambda x: x["score"], reverse=True)


# ── NLI-style verification via LLM ───────────────────────────────────────────

VERIFY_PROMPT = """You are a fact-verification assistant. Your task is to
determine whether a CLAIM is supported by the given PASSAGE.

Classify as:
- "supported"    — the passage clearly and directly supports the claim
- "contradicted" — the passage clearly contradicts the claim
- "insufficient" — the passage is related but doesn't clearly support or deny

Respond with ONLY valid JSON. No prose, no markdown fences:
{"verdict": "<supported|contradicted|insufficient>", "confidence": <0.0-1.0>, "reason": "<one sentence>"}"""


def verify_claim_against_passage(
    claim: str,
    passage_text: str,
    llm_client,  # LLMService instance
) -> Dict[str, Any]:
    """
    Use LLM to verify a claim against a single passage.
    Falls back to token-overlap heuristic if LLM call fails.
    """
    fallback = {
        "verdict": "insufficient",
        "confidence": 0.3,
        "reason": "Verification unavailable",
    }

    if not llm_client or not llm_client.client:
        # Heuristic fallback
        score = _token_overlap(claim, passage_text)
        if score >= 0.25:
            return {"verdict": "supported", "confidence": score, "reason": "token overlap"}
        return fallback

    try:
        messages = [
            {"role": "system", "content": VERIFY_PROMPT},
            {
                "role": "user",
                "content": (
                    f"PASSAGE:\n{passage_text[:1200]}\n\n"
                    f"CLAIM: {claim}"
                ),
            },
        ]
        raw, _ = llm_client._chat(messages, temperature=0.0)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()
        import json
        result = json.loads(raw)
        if result.get("verdict") in ("supported", "contradicted", "insufficient"):
            return result
    except Exception as e:
        logger.debug("Claim verification LLM call failed: %s", e)

    return fallback


# ── Full verification pipeline ────────────────────────────────────────────────

def verify_answer(
    answer: str,
    passages: List[Dict[str, Any]],
    llm_client=None,
    max_claims: int = 12,
) -> Dict[str, Any]:
    """
    Full evidence verification pipeline.

    Steps:
    1. Extract atomic claims from answer
    2. Align each claim to the best matching passage
    3. Verify alignment via LLM (or heuristic)
    4. Classify each claim as supported / unsupported / insufficient
    5. Return structured verification metadata

    Returns:
      {
        total_claims, supported_claims, unsupported_claims, insufficient_claims,
        citation_coverage, claims: [{text, verdict, confidence, supporting_passage}],
        verified_answer  # answer with [UNSUPPORTED] markers if needed
      }
    """
    if not answer or re.search(r"i don't know", answer, re.I):
        return {
            "total_claims": 0,
            "supported_claims": 0,
            "unsupported_claims": 0,
            "insufficient_claims": 0,
            "citation_coverage": 1.0,
            "claims": [],
            "verified_answer": answer,
        }

    claims = extract_claims(answer)[:max_claims]
    if not claims:
        return {
            "total_claims": 0,
            "supported_claims": 0,
            "unsupported_claims": 0,
            "insufficient_claims": 0,
            "citation_coverage": 1.0,
            "claims": [],
            "verified_answer": answer,
        }

    verified_claims = []
    supported = 0
    unsupported = 0
    insufficient = 0
    passages_used: set = set()

    for claim in claims:
        # Find best matching passage
        alignments = align_claim_to_passages(claim, passages)

        if not alignments:
            # No passage even closely related → unsupported
            verified_claims.append(
                {
                    "text": claim,
                    "verdict": "unsupported",
                    "confidence": 0.0,
                    "reason": "No related passage found",
                    "supporting_passage": None,
                }
            )
            unsupported += 1
            continue

        best_alignment = alignments[0]
        best_passage = passages[best_alignment["passage_idx"]]
        passages_used.add(best_alignment["passage_idx"])

        # Verify via LLM if overlap is marginal, use heuristic if clear
        if best_alignment["score"] >= 0.30:
            # High overlap: likely supported, skip expensive LLM call
            result = {
                "verdict": "supported",
                "confidence": min(best_alignment["score"] * 2, 0.95),
                "reason": "High token overlap with source passage",
            }
        else:
            result = verify_claim_against_passage(
                claim,
                best_passage.get("text", ""),
                llm_client,
            )

        verdict = result["verdict"]
        if verdict == "supported":
            supported += 1
        elif verdict == "contradicted":
            unsupported += 1
        else:
            insufficient += 1

        verified_claims.append(
            {
                "text": claim,
                "verdict": verdict,
                "confidence": round(result.get("confidence", 0.5), 3),
                "reason": result.get("reason", ""),
                "supporting_passage": {
                    "filename": best_passage.get("filename"),
                    "page": best_passage.get("page_start"),
                    "document_id": best_passage.get("document_id"),
                    "text_preview": best_passage.get("text", "")[:180],
                    "alignment_score": best_alignment["score"],
                },
            }
        )

    total = len(claims)
    citation_coverage = round(supported / total, 3) if total else 1.0

    # Build verified answer with [UNSUPPORTED] markers on unsupported claims
    verified_answer = answer
    for vc in verified_claims:
        if vc["verdict"] in ("unsupported", "contradicted"):
            original_claim = vc["text"]
            # Escape for regex
            escaped = re.escape(original_claim[:60])
            verified_answer = re.sub(
                escaped + r"[^.!?\n]*",
                lambda m: f"[UNSUPPORTED: {m.group(0)}]",
                verified_answer,
                count=1,
            )

    return {
        "total_claims": total,
        "supported_claims": supported,
        "unsupported_claims": unsupported,
        "insufficient_claims": insufficient,
        "citation_coverage": citation_coverage,
        "claims": verified_claims,
        "verified_answer": verified_answer,
        "passages_used_count": len(passages_used),
    }


# ── Clean answer (remove unsupported claims) ─────────────────────────────────

def clean_answer(answer: str, verification: Dict[str, Any]) -> str:
    """
    Return a cleaned answer with unsupported claims removed.
    Only removes claims that are explicitly 'unsupported' with low confidence.
    """
    claims = verification.get("claims", [])
    clean = answer
    for claim in claims:
        if claim["verdict"] == "unsupported" and claim["confidence"] < 0.2:
            # Remove the claim text from the answer
            escaped = re.escape(claim["text"][:80])
            clean = re.sub(escaped + r"[^.!?\n]*[.!?]?", "", clean).strip()
    # Clean up double spaces / double newlines
    clean = re.sub(r"\n{3,}", "\n\n", clean)
    clean = re.sub(r"  +", " ", clean)
    return clean.strip() or answer  # never return empty
