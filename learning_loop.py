"""
Continuous CEO Learning Loop — System gets better over time.

Extracts patterns from conversions, lost deals, objections, tone.
Saves to learning_memory.json. AI incorporates in future generations.
"""

from __future__ import annotations

import os
import json
from datetime import datetime, timezone
from typing import List, Dict
from collections import Counter


APP_DIR = os.path.dirname(os.path.abspath(__file__))
LEARNING_MEMORY_PATH = os.path.join(APP_DIR, "learning_memory.json")
PLAYBOOK_PATH = os.path.join(APP_DIR, "playbook.json")


def load_learning_memory() -> dict:
    default = {"version": 1, "last_updated": None, "winning_messaging": [], "failed_messaging": [], "key_objections": [], "emotional_triggers": [], "tone_patterns": [], "qualification_insights": [], "stage_timing_insights": []}
    if not os.path.isfile(LEARNING_MEMORY_PATH):
        return default
    try:
        with open(LEARNING_MEMORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save_learning_memory(data: dict) -> None:
    data["last_updated"] = datetime.now(timezone.utc).isoformat()
    try:
        with open(LEARNING_MEMORY_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


def extract_patterns_from_outcome(outcome: str, lead_data: dict, sequence_data: List[dict], *, llm=None) -> dict:
    patterns = {"winning_messaging": [], "failed_messaging": [], "key_objections": [], "emotional_triggers": [], "tone_patterns": []}
    objections, tone_signals, messaging_snippets = [], [], []
    for seq in sequence_data:
        if seq.get("reply_text"):
            objections.append(seq.get("detected_objection") or "unknown")
            tone_signals.append(seq.get("reply_tone") or seq.get("tone_analysis") or "")
        draft = (seq.get("final_email") or seq.get("draft") or "")[:500]
        if draft and "DRAFT — DO NOT SEND" not in draft[:50]:
            messaging_snippets.append({"text": draft[:300], "intent": seq.get("intent_score"), "status": seq.get("qualification_status")})
    if outcome == "converted":
        for s in messaging_snippets:
            if (s.get("intent") or 0) >= 80 or s.get("status") == "hot":
                patterns["winning_messaging"].append({"snippet": s.get("text", ""), "source": lead_data.get("source", "")})
    else:
        for s in messaging_snippets:
            if (s.get("intent") or 0) < 50 or s.get("status") == "cold":
                patterns["failed_messaging"].append({"snippet": s.get("text", ""), "objection": objections[-1] if objections else "none"})
    for o in objections:
        if o and o != "unknown":
            patterns["key_objections"].append({"label": o, "outcome": outcome})
    for t in tone_signals:
        if t:
            patterns["tone_patterns"].append({"tone": t, "outcome": outcome})
    if llm:
        try:
            from langchain_core.messages import SystemMessage, HumanMessage
            summary = f"Lead outcome: {outcome}. Source: {lead_data.get('source')}. Objections: {objections}. Last intent: {sequence_data[-1].get('intent_score') if sequence_data else 'N/A'}."
            resp = llm.invoke([SystemMessage(content="Extract ONE emotional trigger or lesson. Reply: EMOTIONAL_TRIGGER: <phrase> LESSON: <one sentence>"), HumanMessage(content=summary)])
            text = (resp.content or "").strip()
            if "EMOTIONAL_TRIGGER:" in text:
                parts = text.split("EMOTIONAL_TRIGGER:")[-1]
                trigger = parts.split("LESSON:")[0].strip()[:80]
                lesson = parts.split("LESSON:")[-1].strip()[:200] if "LESSON:" in parts else ""
                patterns["emotional_triggers"].append({"trigger": trigger, "lesson": lesson, "outcome": outcome})
        except Exception:
            pass
    return patterns


def merge_patterns_into_learning(patterns: dict, max_per_category: int = 50) -> None:
    mem = load_learning_memory()
    for key in ["winning_messaging", "failed_messaging", "key_objections", "emotional_triggers", "tone_patterns"]:
        lst = mem.get(key, [])
        for item in patterns.get(key, []):
            if isinstance(item, dict):
                lst.append({**item, "learned_at": datetime.now(timezone.utc).isoformat()})
        mem[key] = lst[-max_per_category:]
    save_learning_memory(mem)


def run_learning_loop(outcome: str, lead_data: dict, sequence_data: List[dict], *, llm=None) -> None:
    patterns = extract_patterns_from_outcome(outcome, lead_data, sequence_data, llm=llm)
    merge_patterns_into_learning(patterns)


def get_learned_patterns() -> str:
    mem = load_learning_memory()
    lines = []
    wins = mem.get("winning_messaging", [])[-5:]
    if wins:
        lines.append("WINNING MESSAGING (emulate):")
        for w in wins:
            snip = w.get("snippet", "")[:200]
            if snip:
                lines.append(f"  - {snip}…")
    fails = mem.get("failed_messaging", [])[-5:]
    if fails:
        lines.append("FAILED MESSAGING (avoid):")
        for f in fails:
            snip = f.get("snippet", "")[:200]
            if snip:
                lines.append(f"  - {snip}… (objection: {f.get('objection','')})")
    objections = mem.get("key_objections", [])[-10:]
    if objections:
        cnt = Counter(o.get("label", "") for o in objections if o.get("label"))
        lines.append("KEY OBJECTIONS (address proactively): " + ", ".join(f"{k}({v})" for k, v in cnt.most_common(5)))
    triggers = mem.get("emotional_triggers", [])[-3:]
    if triggers:
        lines.append("EMOTIONAL INSIGHTS:")
        for t in triggers:
            lines.append(f"  - {t.get('trigger','')}: {t.get('lesson','')}")
    return "\n".join(lines) if lines else "No patterns learned yet."
