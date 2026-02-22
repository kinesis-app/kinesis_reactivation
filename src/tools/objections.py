"""
Kinesis Reactivation — Objection extraction (deterministic + LLM classifier).
"""

from __future__ import annotations

import re

from langchain_core.messages import HumanMessage, SystemMessage


OBJECTION_LABELS = ["budget", "timing", "product_mismatch", "wrong_contact", "no_need", "legal", "spam", "unresponsive", "other"]


def extract_objection(reply_text: str) -> tuple[str, int]:
    """Deterministic: normalize objection from reply. Returns (label, confidence 0-100)."""
    t = (reply_text or "").lower().strip()
    if not t:
        return "other", 0
    if "not interested" in t or "no thanks" in t or "not at this time" in t:
        return "no_need", 85
    if "$" in t or "budget" in t or "cost" in t or "expensive" in t:
        return "budget", 80
    if "not the right person" in t or "wrong person" in t or "different department" in t:
        return "wrong_contact", 85
    if "timing" in t or "later" in t or "next quarter" in t:
        return "timing", 75
    if "unsubscribe" in t or "remove" in t or "spam" in t:
        return "spam", 90
    if "legal" in t or "compliance" in t or "policy" in t:
        return "legal", 70
    if "wrong product" in t or "not a fit" in t or "different solution" in t:
        return "product_mismatch", 75
    return "other", 50


def classify_objection_llm(reply_text: str, llm) -> tuple[str, int, str]:
    """LLM classifier when deterministic is uncertain. Returns (label, confidence 0-100, one_line_reason)."""
    if not llm or not (reply_text or "").strip():
        return "other", 0, ""
    prompt = f"""Extract a single normalized objection from the following reply. Choose exactly one from: {", ".join(OBJECTION_LABELS)}.
Reply with exactly: LABEL: <label> CONFIDENCE: <0-100> REASON: <one line>
Reply text:
{(reply_text or "")[:800]}"""
    try:
        resp = llm.invoke([
            SystemMessage(content="You are an objection classifier. Output only LABEL: CONFIDENCE: REASON: on one or two lines."),
            HumanMessage(content=prompt),
        ])
        text = (resp.content or "").strip()
        label = "other"
        confidence = 50
        reason = ""
        for line in text.split("\n"):
            line = line.strip()
            if line.upper().startswith("LABEL:"):
                val = line.split(":", 1)[-1].strip().lower()
                if val in OBJECTION_LABELS:
                    label = val
                else:
                    label = "other"
            elif line.upper().startswith("CONFIDENCE:"):
                try:
                    confidence = max(0, min(100, int(re.search(r"\d+", line).group(0))))
                except Exception:
                    pass
            elif line.upper().startswith("REASON:"):
                reason = line.split(":", 1)[-1].strip()[:200]
        return label, confidence, reason
    except Exception:
        return "other", 0, ""


def extract_objection_with_llm(reply_text: str, llm) -> tuple[str, int]:
    """Deterministic first; if label is 'other' and confidence < 70, use LLM classifier. Returns (label, confidence)."""
    label, conf = extract_objection(reply_text)
    if (label != "other") or conf >= 70:
        return label, conf
    label_llm, conf_llm, _ = classify_objection_llm(reply_text, llm)
    return label_llm, conf_llm
