"""
Deep Funnel Death Analytics — Management intelligence.

Computes: stage deaths, main objection, emotional reason, common pattern, death stories.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional, List, Dict
from collections import Counter


def extract_objection_deterministic(reply_text: str) -> str:
    t = (reply_text or "").lower().strip()
    if not t:
        return "unresponsive"
    if "not interested" in t or "no thanks" in t:
        return "no_need"
    if "budget" in t or "cost" in t or "$" in t:
        return "budget"
    if "wrong person" in t or "different department" in t:
        return "wrong_contact"
    if "timing" in t or "later" in t:
        return "timing"
    if "unsubscribe" in t or "spam" in t:
        return "spam"
    if "legal" in t or "compliance" in t:
        return "legal"
    if "not a fit" in t or "wrong product" in t:
        return "product_mismatch"
    return "other"


OBJECTION_TO_EMOTION = {
    "budget": "low perceived ROI / cost sensitivity",
    "no_need": "lack of urgency / no immediate pain",
    "wrong_contact": "wrong decision maker / poor targeting",
    "timing": "wrong moment / competing priorities",
    "product_mismatch": "feature gap / unmet expectations",
    "legal": "compliance concern",
    "spam": "trust / relevance",
    "unresponsive": "silence / disengagement",
}


def compute_deep_funnel_analytics(sequences_data: List[dict], period: Optional[str] = None, llm=None) -> dict:
    period = period or str(datetime.now().date())
    analytics = {
        "period": period,
        "total_leads": len(sequences_data),
        "by_stage": {},
        "stage_deaths": {},
        "main_objection": None,
        "emotional_reason": None,
        "common_pattern": None,
        "death_stories": [],
        "top_objections": [],
        "avg_confidence_by_stage": {},
    }
    death_candidates = []
    objection_counts = {}
    confidence_by_stage = {}
    for seq in sequences_data:
        sequence_emails = seq.get("sequence_emails", [])
        emails_sent = len(sequence_emails)
        stage = f"email_{min(emails_sent, 5)}" if emails_sent else "email_1"
        status = seq.get("qualification_status", "cold")
        if stage not in analytics["by_stage"]:
            analytics["by_stage"][stage] = {"count": 0, "died": 0, "hot": 0, "warm": 0, "cold": 0}
        analytics["by_stage"][stage]["count"] += 1
        analytics["by_stage"][stage][status] = analytics["by_stage"][stage].get(status, 0) + 1
        if status == "cold" and emails_sent > 0:
            analytics["by_stage"][stage]["died"] = analytics["by_stage"][stage].get("died", 0) + 1
            analytics["stage_deaths"][stage] = analytics["stage_deaths"].get(stage, 0) + 1
            reply = (seq.get("reply_text") or "").strip()
            obj = extract_objection_deterministic(reply) if reply else "opt_out"
            objection_counts[obj] = objection_counts.get(obj, 0) + 1
            raw = (reply[:200] + "…") if len(reply) > 200 else (reply or "do_not_contact")
            death_candidates.append((seq, stage, obj, raw))
        conf = seq.get("draft_confidence_score")
        if conf is not None:
            confidence_by_stage.setdefault(stage, []).append(conf)
    for stage, vals in confidence_by_stage.items():
        analytics["avg_confidence_by_stage"][stage] = int(round(sum(vals) / len(vals), 0)) if vals else 0
    analytics["top_objections"] = sorted([{"label": k, "count": v} for k, v in objection_counts.items()], key=lambda x: -x["count"])[:10]
    main_obj = max(objection_counts, key=objection_counts.get) if objection_counts else "other"
    analytics["main_objection"] = main_obj
    analytics["emotional_reason"] = OBJECTION_TO_EMOTION.get(main_obj, "unknown")
    if death_candidates:
        stages = [c[1] for c in death_candidates]
        stage_cnt = Counter(stages)
        top_stage = stage_cnt.most_common(1)[0][0] if stage_cnt else "email_1"
        hints = {"budget": "pricing or ROI clarity introduced too early or too late", "no_need": "value proposition not compelling enough", "wrong_contact": "targeting mismatch or poor lead source", "timing": "cadence or follow-up timing suboptimal", "product_mismatch": "content or offer not aligned with segment"}
        analytics["common_pattern"] = hints.get(main_obj, f"Drops concentrated at {top_stage}. Review messaging.")
        for seq, stage, obj, raw in death_candidates[:15]:
            narrative = _build_death_story(seq, stage, obj, raw, llm)
            analytics["death_stories"].append({"stage": stage, "main_objection": obj, "raw_excerpt": raw or "do_not_contact", "death_story": narrative})
    return analytics


def _build_death_story(seq: dict, stage: str, main_obj: str, raw_excerpt: str, llm) -> str:
    emo = OBJECTION_TO_EMOTION.get(main_obj, main_obj)
    if not llm:
        return f"Lead showed interest via {seq.get('source', 'unknown')} but dropped at {stage}. Primary objection: {main_obj} ({emo}). Consider adjusting content or cadence."
    try:
        from langchain_core.messages import SystemMessage, HumanMessage
        source = seq.get("source", "unknown")
        behavior = (seq.get("behavior") or "")[:200]
        prompt = f"""Write exactly 3 short sentences for a management "death story" report.
1) What the lead did (source: {source}, behavior: {behavior or 'none'}).
2) Why they dropped (objection: {main_obj}; excerpt: {raw_excerpt[:120]}).
3) One suggested fix (content, cadence, or targeting).
Output only the 3 sentences. No labels. Anonymize."""
        resp = llm.invoke([SystemMessage(content="Brief anonymized funnel narratives for management. Plain English, 3 sentences."), HumanMessage(content=prompt)])
        out = (resp.content or "").strip()
        if out and len(out) > 80:
            return out[:600]
    except Exception:
        pass
    return f"Lead dropped at {stage}; objection: {main_obj} ({emo}). Review content or cadence."
