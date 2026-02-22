"""
Kinesis Reactivation — Draft Evaluator (compliance & confidence scoring).
"""

from __future__ import annotations

import re
from typing import Optional, List

from langchain_core.messages import HumanMessage, SystemMessage

from src.llm.providers import get_llm_evaluator


BANNED_PHRASES = [
    "100%", "guaranteed", "unbreakable", "cure", "cures", "fda approved",
    "clinically proven", "medical advice", "diagnose", "treatment for",
]
NUMERIC_CLAIM_PATTERN = re.compile(r"\d+%\s+(improvement|increase|reduction|success|cure|guarantee)|studies?\s+show|\d+\s+times\s+(better|faster)")


def _deterministic_checks(draft: str) -> tuple[int, list]:
    """Factual safety 0-20. Returns (points, risk_flags_list)."""
    t = (draft or "").lower()
    flags = []
    points = 20
    for phrase in BANNED_PHRASES:
        if phrase in t:
            flags.append("banned_phrase")
            points = max(0, points - 15)
            break
    if NUMERIC_CLAIM_PATTERN.search(t):
        flags.append("unsourced_numeric_claim")
        points = max(0, points - 10)
    efficacy_words = ["cure", "heal", "eliminate", "eradicate", "miracle"]
    if any(w in t for w in efficacy_words):
        flags.append("POTENTIAL_LEGAL_RISK")
        points = max(0, points - 15)
    return points, flags


def _similarity_to_vault_points(similarity_avg: float) -> int:
    """Map avg score (0-1, higher=better) to 0-40. FAISS may return distance (lower=better); we handle both."""
    if similarity_avg is None:
        return 20
    s = float(similarity_avg)
    if s <= 1:
        sim = s
    else:
        sim = max(0, 1 - s / 2)
    return min(40, int(40 * sim))


def evaluate_draft_confidence(
    draft: str,
    citations: str,
    lead_context: str,
    llm,
    citations_list: Optional[List[dict]] = None,
) -> dict:
    """
    Draft Evaluator: similarity_avg from citations, deterministic checks, LLM only for tone + explanation.
    Returns eval_output: CONFIDENCE_SCORE, VAULT_ALIGNMENT, RISK_FLAGS, EXPLANATION, prompt_version, needs_review.
    """
    prompt_version = "kinesis_cursor_master:v1.0"
    if not draft or "Generation failed" in draft or "LLM not loaded" in draft:
        return {
            "score": 0,
            "explanation": "Could not evaluate (invalid draft).",
            "risk_flags": "unknown",
            "vault_alignment": "low",
            "prompt_version": prompt_version,
            "needs_review": True,
        }

    risk_flags_list = []
    similarity_avg = None
    if citations_list and isinstance(citations_list, list):
        raw_scores = [c.get("score") for c in citations_list if c.get("score") is not None]
        if raw_scores:
            sims = [1 / (1 + abs(s)) for s in raw_scores]
            similarity_avg = sum(sims) / len(sims)

    vault_grounding = _similarity_to_vault_points(similarity_avg)
    factual_points, det_flags = _deterministic_checks(draft)
    risk_flags_list.extend(det_flags)

    personalization = 10
    if lead_context:
        ctx_lower = lead_context.lower()
        if "name:" in ctx_lower or "company:" in ctx_lower:
            name_part = re.search(r"name:\s*(\w+)", ctx_lower)
            if name_part and name_part.group(1) != "there" and name_part.group(1) in draft.lower():
                personalization = 20
            elif "company" in ctx_lower or "role" in ctx_lower:
                personalization = 15

    tone_points = 0
    explanation = ""
    llm_eval = get_llm_evaluator() or llm
    if llm_eval:
        sys_prompt = """Judge ONLY tone match to CEO/vault style (professional, helpful, not salesy). Reply:
TONE_SCORE: <0-20>
EXPLANATION: <1-2 sentences>"""
        try:
            resp = llm_eval.invoke([
                SystemMessage(content=sys_prompt),
                HumanMessage(content=f"Draft:\n{(draft or '')[:2000]}\n\nVault excerpts:\n{(citations or '')[:1500]}"),
            ])
            text = (resp.content or "").strip()
            for line in text.split("\n"):
                line = line.strip()
                if line.upper().startswith("TONE_SCORE:"):
                    try:
                        tone_points = max(0, min(20, int(re.search(r"\d+", line).group(0))))
                    except Exception:
                        pass
                elif line.upper().startswith("EXPLANATION:"):
                    explanation = line.split(":", 1)[-1].strip()[:300]
        except Exception as e:
            explanation = str(e)[:150]
            tone_points = 10

    if not explanation:
        explanation = f"Vault grounding: {vault_grounding}/40, Safety: {factual_points}/20, Personalization: {personalization}/20, Tone: {tone_points}/20."
    score = vault_grounding + factual_points + personalization + tone_points
    score = max(1, min(100, score))

    if vault_grounding >= 32:
        vault_alignment = "high"
    elif vault_grounding >= 20:
        vault_alignment = "medium"
    else:
        vault_alignment = "low"

    risk_flags_str = ", ".join(risk_flags_list) if risk_flags_list else "none"
    needs_review = score < 60 or any("legal" in f.lower() or "risk" in f.lower() for f in risk_flags_list)
    if needs_review and risk_flags_str == "none":
        risk_flags_str = "low_confidence" if score < 60 else risk_flags_str

    return {
        "score": score,
        "explanation": explanation,
        "risk_flags": risk_flags_str,
        "vault_alignment": vault_alignment,
        "prompt_version": prompt_version,
        "needs_review": needs_review,
    }
