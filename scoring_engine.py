"""
Real Confidence & Certainty Scoring — Grounded in measurable signals.

Combines: behavioral signals, objection severity, tone, recency, engagement depth.
Produces: intent_score (0–100), certainty_score (0–100).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, List


OBJECTION_SEVERITY = {
    "spam": 90, "no_need": 85, "wrong_contact": 80, "budget": 75,
    "product_mismatch": 70, "legal": 65, "timing": 50, "unresponsive": 40, "other": 30,
}


def _detect_objection(text: str) -> tuple[str, int]:
    t = (text or "").lower().strip()
    if not t:
        return "unresponsive", OBJECTION_SEVERITY["unresponsive"]
    if "unsubscribe" in t or "spam" in t or "remove" in t:
        return "spam", OBJECTION_SEVERITY["spam"]
    if "not interested" in t or "no thanks" in t:
        return "no_need", OBJECTION_SEVERITY["no_need"]
    if "wrong person" in t or "different department" in t:
        return "wrong_contact", OBJECTION_SEVERITY["wrong_contact"]
    if "budget" in t or "cost" in t or "$" in t or "expensive" in t:
        return "budget", OBJECTION_SEVERITY["budget"]
    if "not a fit" in t or "wrong product" in t:
        return "product_mismatch", OBJECTION_SEVERITY["product_mismatch"]
    if "legal" in t or "compliance" in t:
        return "legal", OBJECTION_SEVERITY["legal"]
    if "timing" in t or "later" in t or "next quarter" in t:
        return "timing", OBJECTION_SEVERITY["timing"]
    return "other", OBJECTION_SEVERITY["other"]


def _tone_signal(tone: str) -> int:
    t = (tone or "").lower()
    if "positive" in t or "enthusiastic" in t or "engaged" in t:
        return 15
    if "neutral" in t or "formal" in t:
        return 5
    if "negative" in t or "frustrated" in t or "angry" in t:
        return -15
    if "casual" in t or "friendly" in t:
        return 10
    return 0


def _recency_bonus(last_event_ts: Optional[str]) -> int:
    if not last_event_ts:
        return 0
    try:
        dt = datetime.fromisoformat(last_event_ts.replace("Z", "+00:00"))
    except Exception:
        return 0
    now = datetime.now(timezone.utc)
    delta = now - dt
    if delta.days < 1:
        return 15
    if delta.days < 3:
        return 10
    if delta.days < 7:
        return 5
    return 0


def compute_intent_and_certainty(
    *,
    behavior: str = "",
    objection: str = "",
    reply_text: str = "",
    reply_count: int = 0,
    email_num: int = 1,
    tone_analysis: str = "",
    history: Optional[List[dict]] = None,
    llm_intent: Optional[int] = None,
) -> tuple[int, int]:
    intent_base, certainty_base = 40, 30
    if reply_count > 0:
        intent_base += min(25, reply_count * 10)
        certainty_base += min(25, reply_count * 12)
    if reply_text and len(reply_text.strip()) > 50:
        intent_base += 5
    obj_label, obj_sev = _detect_objection(reply_text or objection)
    intent_base -= min(40, obj_sev // 2)
    certainty_base -= min(30, obj_sev // 3)
    tone_delta = _tone_signal(tone_analysis)
    intent_base += tone_delta
    certainty_base += tone_delta // 2
    last_ts = None
    if history:
        for h in reversed(history):
            if h.get("timestamp"):
                last_ts = h["timestamp"]
                break
    recency = _recency_bonus(last_ts)
    intent_base += recency
    certainty_base += recency
    if history:
        scores = [h.get("intent_score") for h in history if h.get("intent_score") is not None]
        if len(scores) > 1:
            trend = scores[-1] - scores[0]
            if trend > 10:
                certainty_base += 10
            elif trend < -10:
                certainty_base -= 10
    if llm_intent is not None:
        intent_base = int(0.4 * intent_base + 0.6 * llm_intent)
    intent_score = max(0, min(100, intent_base))
    certainty_score = max(0, min(100, certainty_base))
    if reply_count == 0 and not reply_text:
        certainty_score = min(certainty_score, 60)
    if obj_sev >= 80:
        certainty_score = min(certainty_score, 40)
    if reply_count >= 2 and tone_delta > 0:
        certainty_score = min(100, certainty_score + 15)
    return intent_score, certainty_score


def qualification_status(intent_score: int) -> str:
    if intent_score >= 80:
        return "hot"
    if intent_score >= 50:
        return "warm"
    return "cold"
