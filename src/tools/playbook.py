"""
Kinesis Reactivation — Playbook (JSON) persistence and learning signals.
"""

from __future__ import annotations

import os
import json
from datetime import datetime, timedelta, timezone

import streamlit as st

from src.config import PLAYBOOK_PATH, LEARNING_SIGNALS_PATH
from src.db import get_all_lead_ids, get_lead, update_lead_status


def load_playbook() -> dict:
    if not os.path.isfile(PLAYBOOK_PATH):
        return {"approved": [], "rejected": [], "non_responses": [], "failures": [], "training_pairs": []}
    try:
        with open(PLAYBOOK_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "approved": data.get("approved", []),
            "rejected": data.get("rejected", []),
            "non_responses": data.get("non_responses", []),
            "failures": data.get("failures", []),
            "training_pairs": data.get("training_pairs", []),
        }
    except Exception:
        return {"approved": [], "rejected": [], "non_responses": [], "failures": [], "training_pairs": []}


def save_playbook(data: dict) -> None:
    data.setdefault("training_pairs", [])
    try:
        with open(PLAYBOOK_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        if "st" in dir():
            st.sidebar.error(f"Could not save playbook: {e}")


def _subject_keywords(subject_line: str, max_words: int = 5) -> list:
    """Extract first max_words significant words (length > 4) from subject line."""
    words = [w.strip().strip(".,!?") for w in (subject_line or "").split() if len(w.strip().strip(".,!?")) > 4]
    return words[:max_words]


def add_approved_to_playbook(
    email: str,
    original_prompt: str,
    company_name: str = "",
    *,
    lead_source: str = "",
    confidence_score: int = 0,
    subject_line: str = "",
) -> None:
    data = load_playbook()
    subject_keywords = _subject_keywords(subject_line)
    data["approved"].append({
        "email": email,
        "email_body": email,
        "original_prompt": original_prompt,
        "company_name": company_name,
        "lead_source": lead_source,
        "confidence_score": confidence_score,
        "subject_keywords": subject_keywords,
        "timestamp": datetime.now().isoformat(),
    })
    save_playbook(data)


def add_rejected_to_playbook(rejection_reason: str, draft_snippet: str = "") -> None:
    data = load_playbook()
    data["rejected"].append({
        "rejection_reason": rejection_reason,
        "draft_snippet": (draft_snippet or "")[:500],
        "timestamp": datetime.now().isoformat(),
    })
    save_playbook(data)


def add_training_pair_to_playbook(original_draft: str, edited_draft: str, lead_id: str) -> None:
    """Save (original, edited) pair for training when reviewer edits then approves."""
    data = load_playbook()
    data.setdefault("training_pairs", []).append({
        "original_draft": original_draft,
        "edited_draft": edited_draft,
        "lead_id": lead_id,
        "timestamp": datetime.now().isoformat(),
    })
    save_playbook(data)


def check_no_response_leads() -> None:
    """Tag leads with no reply 14+ days after send as low_performer; append to learning_signals.json."""
    try:
        cutoff = datetime.now(timezone.utc) - timedelta(days=14)
        signals = []
        if os.path.isfile(LEARNING_SIGNALS_PATH):
            try:
                with open(LEARNING_SIGNALS_PATH, "r", encoding="utf-8") as f:
                    signals = json.load(f)
                if not isinstance(signals, list):
                    signals = []
            except Exception:
                signals = []
        for lead_id in get_all_lead_ids():
            lead = get_lead(lead_id)
            if not lead:
                continue
            history = lead.get("history") or []
            if any(h.get("reply_received") for h in history):
                continue
            last_sent_ts = None
            last_sent_body = None
            for h in history:
                if h.get("email_sent"):
                    ts_str = h.get("timestamp", "")
                    if ts_str:
                        try:
                            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                            if ts.tzinfo:
                                ts = ts.replace(tzinfo=None)
                            if last_sent_ts is None or ts > last_sent_ts:
                                last_sent_ts = ts
                                last_sent_body = h.get("email_sent", "")
                        except Exception:
                            pass
            if last_sent_ts is None or last_sent_body is None or last_sent_ts >= cutoff:
                continue
            existing_ids = {s.get("lead_id") for s in signals if s.get("reason") == "no_response_14_days"}
            if lead_id in existing_ids:
                continue
            update_lead_status(lead_id, "low_performer")
            record = {
                "lead_id": lead_id,
                "email_body": last_sent_body[:2000] if last_sent_body else "",
                "lead_source": lead.get("source", ""),
                "send_date": last_sent_ts.isoformat() if last_sent_ts else "",
                "reason": "no_response_14_days",
            }
            signals.append(record)
        with open(LEARNING_SIGNALS_PATH, "w", encoding="utf-8") as f:
            json.dump(signals, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


def add_non_response_to_playbook(email_num: int, lead_id: str, reason: str = "") -> None:
    data = load_playbook()
    data["non_responses"].append({"email_num": email_num, "lead_id": lead_id, "reason": reason or "No response", "timestamp": datetime.now().isoformat()})
    save_playbook(data)


def add_failure_insight(insight: str, stage: str, lead_id: str = "") -> None:
    data = load_playbook()
    data["failures"].append({"insight": insight, "stage": stage, "lead_id": lead_id, "timestamp": datetime.now().isoformat()})
    save_playbook(data)
