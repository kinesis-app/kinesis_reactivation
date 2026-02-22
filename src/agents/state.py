"""
Kinesis Reactivation — LangGraph State definition.
"""

from __future__ import annotations

from typing import TypedDict, Optional


class InboundLeadState(TypedDict, total=False):
    row: dict
    email: str
    name: str
    source: str
    behavior: str
    initial_objection: str
    enriched_data: dict
    scenario_config: dict
    email_num: int
    sequence_emails: list
    draft: str
    citations: str
    legal_ok: bool
    legal_note: str
    attempts: int
    intent_score: int
    qualification_status: str
    needs_review: bool
    final_email: str
    user_prompt: str
    legal_footer: str
    reply_received: bool
    reply_text: str
    reply_tone: str
    draft_confidence_score: int
    draft_confidence_explanation: str
    draft_risk_flags: str
    draft_vault_alignment: str
    citations_list: list
    draft_summary: str
    draft_keywords: list
    draft_confidence_estimate: dict
    prompt_version: str
    do_not_contact: bool
    certainty_score: int
    lead_id: str
    latest_incoming_message: str
    is_nudge: bool
    reactivation_validation: dict
