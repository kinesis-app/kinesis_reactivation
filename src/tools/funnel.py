"""
Kinesis Reactivation — Funnel Analytics (master schema).

Merges root funnel_analytics.py deep analytics with inline monolith funnel code.
"""

from __future__ import annotations

import os
import json
from datetime import datetime, timezone
from typing import Optional, List, Dict
from collections import Counter

from langchain_core.messages import HumanMessage, SystemMessage

from src.config import SEQUENCES_DIR
from src.tools.objections import extract_objection_with_llm
from src.llm.providers import get_llm


# ── From funnel_analytics.py ─────────────────────────────────────────────────

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
            raw = (reply[:200] + "\u2026") if len(reply) > 200 else (reply or "do_not_contact")
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


# ── From monolith inline funnel code ──────────────────────────────────────────

def build_death_story_narrative(seq: dict, stage: str, main_objection: str, raw_excerpt: str, llm) -> str:
    """Generate 3-sentence narrative: behavior => why died => suggested fix."""
    if not llm:
        return f"Lead dropped at {stage}; objection: {main_objection}. Review content or cadence."
    source = seq.get("source", "unknown")
    behavior = (seq.get("behavior") or "")[:300]
    name = seq.get("name", "Lead")
    prompt = f"""Write exactly 3 short sentences for a "death story" report.
Sentence 1: What the lead did (source: {source}, behavior: {behavior or 'none'}).
Sentence 2: Why they dropped (objection: {main_objection}; excerpt: {raw_excerpt[:150]}).
Sentence 3: One suggested fix for management (e.g. change content, cadence, or targeting).
Do not use the lead's name or email. Output only the 3 sentences, no labels."""
    try:
        resp = llm.invoke([
            SystemMessage(content="You write brief, anonymized funnel narratives for internal reports. Plain English, 3 sentences only."),
            HumanMessage(content=prompt),
        ])
        out = (resp.content or "").strip()
        if out and len(out) > 50:
            return out[:600]
    except Exception:
        pass
    return f"Lead showed interest via {source} but dropped at {stage} ({main_objection}). Consider adjusting cadence or content for this segment."


def compute_funnel_analytics(sequences_data: List[dict], period: Optional[str] = None, use_llm_stories: bool = True) -> dict:
    """
    Funnel Analytics: drop-offs by stage, top objections (deterministic + LLM), death_stories (3-sentence narrative).
    Input: sequences_data = array of sequence objects (each has sequence_emails, events if present, reply_text, etc.).
    Output: analytics JSON per master schema.
    """
    period = period or f"{datetime.now().date()}"
    llm = get_llm() if use_llm_stories else None

    analytics = {
        "period": period,
        "total_leads": len(sequences_data),
        "by_stage": {},
        "drop_offs": [],
        "top_objections": [],
        "avg_confidence_by_stage": {},
        "death_stories": [],
    }

    objection_counts: Dict[str, int] = {}
    confidence_by_stage: Dict[str, List[int]] = {}
    death_story_candidates: List[tuple] = []

    for seq in sequences_data:
        events = seq.get("events", [])
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
            analytics["drop_offs"].append({"stage": stage, "lead_id": seq.get("email", "")})

        conf = seq.get("draft_confidence_score")
        if conf is not None:
            confidence_by_stage.setdefault(stage, []).append(conf)

        reply = (seq.get("reply_text") or "").strip()
        if reply:
            label, _ = extract_objection_with_llm(reply, llm)
            objection_counts[label] = objection_counts.get(label, 0) + 1

        if status == "cold" and (reply or seq.get("do_not_contact")):
            label, _ = extract_objection_with_llm(reply, llm) if reply else ("opt_out", 100)
            raw = (reply[:200] + "\u2026") if len(reply) > 200 else (reply or "do_not_contact")
            death_story_candidates.append((seq, stage, label, raw))

    for stage, vals in confidence_by_stage.items():
        analytics["avg_confidence_by_stage"][stage] = int(round(sum(vals) / len(vals), 0)) if vals else 0
    analytics["top_objections"] = sorted(
        [{"label": k, "count": v} for k, v in objection_counts.items()],
        key=lambda x: -x["count"],
    )[:10]

    for seq, stage, main_objection, raw_excerpt in death_story_candidates[:20]:
        short_narrative = build_death_story_narrative(seq, stage, main_objection, raw_excerpt, llm) if use_llm_stories else (
            f"Lead dropped at {stage}; objection: {main_objection}. Review content or cadence."
        )
        analytics["death_stories"].append({
            "lead_id": seq.get("email", ""),
            "main_objection": main_objection,
            "raw_excerpt": raw_excerpt or "do_not_contact",
            "short_narrative": short_narrative,
        })

    return analytics


def compute_funnel_stats(sequences_data: List[dict], period: Optional[str] = None) -> dict:
    """Thin wrapper: computes full funnel analytics (drop-offs, objections with LLM, 3-sentence death stories)."""
    return compute_funnel_analytics(sequences_data, period=period, use_llm_stories=True)


def log_funnel_events(funnel_data: List[dict]) -> None:
    """Persist funnel events for analytics. Appends batch to SEQUENCES_DIR/funnel_events.json."""
    if not funnel_data:
        return
    path = os.path.join(SEQUENCES_DIR, "funnel_events.json")
    batch = {"timestamp": datetime.now(timezone.utc).isoformat(), "events": funnel_data}
    existing = []
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                existing = data.get("batches", [])[-100:]
        except Exception:
            pass
    existing.append(batch)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"batches": existing}, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


def load_logged_funnel_events() -> List[dict]:
    """Load all logged funnel events (flattened) for cross-session analytics."""
    path = os.path.join(SEQUENCES_DIR, "funnel_events.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        out = []
        for b in data.get("batches", []):
            out.extend(b.get("events", []))
        return out
    except Exception:
        return []
