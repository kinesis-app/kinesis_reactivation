"""
Kinesis Reactivation — LangGraph node functions.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Optional

import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import (
    _dbg,
    SCENARIO_CONFIG,
    LEGAL_FOOTER,
    GOLDEN_RULES,
)
from src.llm.providers import get_llm, get_llm_evaluator, get_llm_writer
from src.tools.vault import vault_retrieve, format_citations, citations_to_schema
from src.tools.compliance import evaluate_draft_confidence
from src.tools.enrichment import enrich_with_clay
from src.tools.objections import extract_objection
from src.tools.playbook import load_playbook, add_failure_insight
from src.tools.learning_loop import get_learned_patterns
from src.tools.scoring_engine import compute_intent_and_certainty, qualification_status as scoring_qualification_status
from src.db import (
    upsert_lead,
    get_lead_summary_for_prompt,
    get_lead_history,
)
from src.agents.state import InboundLeadState


def validate_reactivation_lead(state: InboundLeadState) -> InboundLeadState:
    """Pre-contact validation for reactivation leads: block irrelevant/outdated before drafting."""
    if state.get("source", "").lower() != "reactivation":
        return state
    enriched = state.get("enriched_data", {})
    row = state.get("row", {})
    context = f"""
  Role: {enriched.get('role', 'unknown')}
  Company: {enriched.get('company', 'unknown')}
  Recent news: {enriched.get('recent_news', 'none')}
  Funding: {enriched.get('funding', 'none')}
  Website: {enriched.get('website', 'none')}
  Last contact: {row.get('last_contact_date', 'unknown')}
  Past project: {row.get('past_project', 'unknown')}
  Previous revenue: {row.get('previous_revenue', 'unknown')}
  Dormant reason: {row.get('dormant_reason', 'unknown')}
  """
    llm = get_llm_evaluator()
    validation = {"relevance": 0, "buying": 0, "overall": 0, "proceed": "no", "explanation": "LLM not available."}
    if llm:
        prompt = f"""Analyze if this old lead is still relevant and in a position to buy/decide.
  {context}

  Score 0-100:
  - RELEVANCE_SCORE: still in industry, same/similar role, no job change signals in news?
  - BUYING_READINESS_SCORE: positive signals (funding, expansion news, decision-making role)?
  - OVERALL_PROCEED_SCORE: 80+ = yes proceed, 60-79 = caution (review), <60 = no (block contact)

  Reply exactly:
  RELEVANCE_SCORE: X
  BUYING_READINESS_SCORE: Y
  OVERALL_PROCEED_SCORE: Z
  PROCEED: yes/caution/no
  EXPLANATION: 1-2 sentences why (e.g. "Person likely left company per news", "Strong funding + decision role = high readiness")"""
        try:
            resp = llm.invoke([HumanMessage(content=prompt)])
            text = (resp.content or "").strip()
            rel_m = re.search(r"RELEVANCE_SCORE:\s*(\d+)", text, re.I)
            buy_m = re.search(r"BUYING_READINESS_SCORE:\s*(\d+)", text, re.I)
            over_m = re.search(r"OVERALL_PROCEED_SCORE:\s*(\d+)", text, re.I)
            proceed_m = re.search(r"PROCEED:\s*(yes|caution|no)", text, re.I)
            expl_m = re.search(r"EXPLANATION:\s*(.+)", text, re.S | re.I)
            validation = {
                "relevance": int(rel_m.group(1)) if rel_m else 0,
                "buying": int(buy_m.group(1)) if buy_m else 0,
                "overall": int(over_m.group(1)) if over_m else 0,
                "proceed": (proceed_m.group(1) or "no").lower() if proceed_m else "no",
                "explanation": expl_m.group(1).strip()[:500] if expl_m else "No explanation parsed.",
            }
        except Exception:
            validation["explanation"] = "Validation LLM call failed."
    out = {**state, "reactivation_validation": validation}
    overall = validation.get("overall", 0)
    proceed = validation.get("proceed", "no")
    if overall < 60 or proceed == "no":
        out["do_not_contact"] = True
        out["qualification_status"] = "cold"
        out["intent_score"] = 0
        out["draft"] = f"VALIDATION BLOCKED CONTACT\n\nExplanation: {validation.get('explanation', '')}"
    return out


def _route_after_validate_reactivation(state: InboundLeadState) -> str:
    """If reactivation validation blocked, go to END; else continue to branch -> writer."""
    if state.get("do_not_contact") and state.get("reactivation_validation"):
        return "end"
    return "branch"


def node_enrich(state: InboundLeadState) -> InboundLeadState:
    """Enrich lead with Clay. Upsert to long-term memory. Load history for context."""
    row = state["row"]
    email = str(row.get("email", "")).strip()
    name = str(row.get("name", "")).strip()
    source = str(row.get("source", "")).strip()
    lead_id = upsert_lead(row)
    enriched = enrich_with_clay(email, name, source, row=row)
    scenario = SCENARIO_CONFIG.get(source.lower(), SCENARIO_CONFIG["default"])
    return {
        **state,
        "lead_id": lead_id,
        "email": email,
        "name": name,
        "source": source,
        "behavior": str(row.get("behavior", "")).strip(),
        "initial_objection": str(row.get("initial_objection", "")).strip(),
        "enriched_data": enriched,
        "scenario_config": scenario,
        "email_num": 0,
        "sequence_emails": [],
    }


def node_branch_scenario(state: InboundLeadState) -> InboundLeadState:
    """Branch based on source scenario."""
    source = state.get("source", "").lower()
    scenario = SCENARIO_CONFIG.get(source, SCENARIO_CONFIG["default"])
    return {**state, "scenario_config": scenario}


def node_writer_nurture(state: InboundLeadState) -> InboundLeadState:
    """Writer node: short CEO-toned email, 1 personalized line from enriched/behavior, <=140 words. Optional latest_incoming_message = REPLY mode."""
    llm_writer = get_llm_writer()
    # #region agent log
    _dbg("node_writer_nurture", {"llm_is_none": llm_writer is None}, "H2")
    # #endregion
    if llm_writer is None:
        return {**state, "draft": "DRAFT \u2014 DO NOT SEND\n\nTEST DRAFT: LLM not loaded \u2014 check key", "citations": "", "needs_review": True}
    latest_incoming = (state.get("latest_incoming_message") or "").strip()
    is_reply_mode = bool(latest_incoming)
    is_reactivation = state.get("source", "").lower() == "reactivation"
    reactivation_context = ""
    if is_reactivation:
        last_contact = state.get("row", {}).get("last_contact_date", "a while")
        past_project = state.get("row", {}).get("past_project", "our previous collaboration")
        dormant_reason = state.get("row", {}).get("dormant_reason", "")
        previous_revenue = state.get("row", {}).get("previous_revenue", "")
        reactivation_context = f"We previously worked together on {past_project} (last contact {last_contact}). Revenue/value was {previous_revenue or 'significant'}. Dormant reason (if known): {dormant_reason or 'unknown'}."
    is_nudge = state.get("is_nudge", False)
    email_num = state.get("email_num", 1)
    total_emails = state.get("scenario_config", {}).get("emails", 3)
    name = state.get("name") or "there"
    source = state.get("source", "")
    behavior = state.get("behavior", "")
    objection = state.get("initial_objection", "")
    enriched = state.get("enriched_data", {})
    previous_emails = state.get("sequence_emails", [])
    legal_note = state.get("legal_note") or ""
    if legal_note:
        legal_block = f"\nPrevious legal rejection (fix this): {legal_note}\n"
    else:
        legal_block = ""
    query = f"CEO tone nurturing email {source} {behavior}" if not is_reply_mode else f"CEO response to reply: {latest_incoming[:200]}"
    top_chunks = vault_retrieve(query, k=6)
    citations_block = format_citations(top_chunks)
    citations_list = citations_to_schema(top_chunks)
    similarity_avg = (sum(c.get("score", 0) for c in top_chunks) / len(top_chunks)) if top_chunks else 0.0
    lead_id = state.get("lead_id", "")
    history_summary = get_lead_summary_for_prompt(lead_id, max_history=10) if lead_id else ""
    learned = get_learned_patterns()
    playbook = load_playbook()
    learning = ""
    if learned and learned != "No patterns learned yet.":
        learning += "\nLEARNED PATTERNS (incorporate):\n" + learned + "\n"
    if playbook["approved"]:
        learning += "\nApproved examples (emulate):\n" + "\n".join([e.get("email", "")[:600] for e in playbook["approved"][-5:]])
    if playbook["rejected"]:
        learning += "\nAvoid these:\n" + "\n".join([e.get("rejection_reason", "") for e in playbook["rejected"][-5:]])
    context = f"Source: {source}. Behavior: {behavior}. Objection: {objection}. Enriched: {enriched}"
    if previous_emails:
        context += f"\nPrevious emails in sequence: {len(previous_emails)} sent."
    if is_reply_mode:
        system = f"""You are the CEO replying to an inbound message. You are NOT starting a conversation \u2014 you are REPLYING.
If latest_incoming_message is present, address the user's specific text using facts from the Vault. Maintain the CEO persona. Keep the reply under 140 words. Do not sound generic.
RULES:
- Directly address what they said. Reference specific points from their message.
- Use vault facts only; never invent. No regulatory/medical claims.
- Maximum 140 words. Professional, helpful, never salesy or robotic.
{learning}
{GOLDEN_RULES}"""
        user = f"""REPLY to this message from the lead.{legal_block}
VAULT CHUNKS (use for tone and facts only):
{citations_block}

LATEST INCOMING MESSAGE (you must address this):
{latest_incoming}

LEAD CONTEXT: {context}
Recipient: {name}. Source: {source}.
{f"\n\nLEAD HISTORY (use for continuity):\n{history_summary}" if history_summary else ""}

Output ONLY the email body (no subject, no DRAFT line). Reply directly to their message using vault-backed facts. Under 140 words. Do not be generic."""
        if is_reactivation:
            system += "\nYou are RE-ACTIVATING an old client/partner. Start warm: reference past positive relationship. Be grateful for past business. No pressure. Express genuine interest in their current situation."
            user += f"\nREACTIVATION CONTEXT (MUST reference past positively):\n{reactivation_context}"
    else:
        system = f"""You are the CEO writing a short nurturing email. Use vault tone and logic only.
RULES:
- First line MUST be personalized: reference one explicit fact from ENRICHED DATA or BEHAVIOR (e.g. role, company, recent news, what they did).
- Maximum 140 words total. Short and helpful.
- Cite vault only when needed. Never invent facts; if not in vault or context, omit or say UNKNOWN. No regulatory/medical claims.
{learning}
{GOLDEN_RULES}"""
        if is_reactivation:
            system += "\nYou are RE-ACTIVATING an old client/partner. Start warm: reference past positive relationship. Be grateful for past business. No pressure. Express genuine interest in their current situation."
        nudge_instruction = "\nThis is a gentle follow-up nudge; they have not replied recently. Keep it brief and value-focused, not pushy." if is_nudge else ""
        user = f"""Write nurturing email {email_num}/{total_emails} for this inbound lead.{legal_block}{nudge_instruction}
VAULT CHUNKS (use for tone and facts only):
{citations_block}

LEAD CONTEXT (use for first-line personalization):
{context}

Recipient: {name}. Source: {source}.
{f"\n\nLEAD HISTORY (use for continuity):\n{history_summary}" if history_summary else ""}

Output ONLY the email body (no subject, no DRAFT line - we add it). Start with one personalized line using a concrete fact from enriched_data or behavior, then keep the rest to <= 140 words."""
        if is_reactivation:
            user += f"\nREACTIVATION CONTEXT (MUST reference past positively):\n{reactivation_context}\nFirst line MUST reference the past relationship (e.g. past project, last contact)."

    prompt_version = "kinesis_cursor_master:v1.0"
    try:
        resp = llm_writer.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        body = (resp.content or "").strip()
        draft = "DRAFT \u2014 DO NOT SEND\n\n" + body
        lead_context_str = f"Name: {name}. Source: {source}. Behavior: {behavior}. Objection: {objection}. Enriched: {enriched}"
        llm = get_llm()
        eval_result = evaluate_draft_confidence(body, citations_block, lead_context_str, llm, citations_list=citations_list)
        needs_review = eval_result.get("needs_review", eval_result["score"] < 60)
        heuristic_flags = []
        if name and name.lower() != "there" and name not in body:
            heuristic_flags.append("no_personalization")
        draft_confidence_estimate = {
            "similarity_avg": round(similarity_avg, 4),
            "vault_alignment": eval_result.get("vault_alignment", "low"),
            "heuristic_flags": heuristic_flags,
        }
        summary, keywords = "", []
        try:
            sum_llm = get_llm()
            if sum_llm:
                sum_resp = sum_llm.invoke([
                    SystemMessage(content="Reply with one line SUMMARY: then KEYWORDS: comma-separated list of 3-5 topics. No other text."),
                    HumanMessage(content=f"Email body:\n{body[:1500]}"),
                ])
                sum_text = (sum_resp.content or "").strip()
                if "SUMMARY:" in sum_text:
                    summary = sum_text.split("SUMMARY:")[-1].split("KEYWORDS:")[0].strip()[:200]
                if "KEYWORDS:" in sum_text:
                    kw = sum_text.split("KEYWORDS:")[-1].strip()
                    keywords = [k.strip() for k in kw.replace(",", " ").split()[:5] if k.strip()]
        except Exception:
            pass
        return {
            **state,
            "draft": draft,
            "citations": citations_block,
            "citations_list": citations_list,
            "user_prompt": user,
            "draft_summary": summary,
            "draft_keywords": keywords,
            "draft_confidence_estimate": draft_confidence_estimate,
            "draft_confidence_score": eval_result["score"],
            "draft_confidence_explanation": eval_result["explanation"],
            "draft_risk_flags": eval_result["risk_flags"],
            "draft_vault_alignment": eval_result["vault_alignment"],
            "needs_review": needs_review,
            "prompt_version": prompt_version,
        }
    except Exception as e:
        err_msg = str(e).replace("\n", " ")[:200]
        return {
            **state,
            "draft": f"DRAFT \u2014 DO NOT SEND\n\nTEST DRAFT: Generation failed - {err_msg}",
            "citations": "",
            "citations_list": [],
            "needs_review": True,
            "draft_confidence_score": 0,
            "draft_confidence_explanation": "Generation failed; no evaluation.",
            "draft_risk_flags": "generation_error",
            "draft_vault_alignment": "low",
            "draft_summary": "",
            "draft_keywords": [],
            "draft_confidence_estimate": {"similarity_avg": 0, "vault_alignment": "low", "heuristic_flags": ["generation_error"]},
            "prompt_version": prompt_version,
        }


def node_legal(state: InboundLeadState) -> InboundLeadState:
    """Legal compliance check. After 3 rejections: mark do_not_contact and add failure insight."""
    row = state.get("row", {})
    if state.get("source") == "reactivation" and (row.get("opt_out") or row.get("do_not_contact")):
        return {**state, "do_not_contact": True, "legal_ok": False, "legal_note": "Old lead with opt-out"}
    llm = get_llm()
    draft = state.get("draft") or ""
    if not draft or llm is None:
        return {**state, "legal_ok": True, "legal_note": ""}
    attempts = state.get("attempts", 0) + 1
    footer = state.get("legal_footer", LEGAL_FOOTER)
    full = draft + footer
    system = """You are a Legal & Pharma Compliance reviewer. Check:
1. NO false medical/therapeutic claims.
2. NO aggressive guarantees (no "100%", "guaranteed", "unbreakable"); rephrase to qualified claims e.g. "highly durable per ISO".
3. GDPR/CAN-SPAM: must have clear unsubscribe and physical address (we append footer).
4. FDA-style: no absolute efficacy claims.
Reply VERDICT: SAFE or VERDICT: UNSAFE and Rejection Note: [what to fix]."""
    # Client-specific compliance rules (from Admin sidebar)
    custom_rules = (st.session_state.get("custom_compliance_rules") or "").strip()
    if custom_rules:
        system += "\n\nCLIENT-SPECIFIC COMPLIANCE RULES (these override and supplement the general rules above \u2014 apply all of them strictly to every draft):\n" + custom_rules
    # Industry regulatory context: inject into vault query and as background for legal reviewer
    industry_context = (st.session_state.get("industry_regulatory_context") or "").strip()
    if industry_context:
        vault_query = (industry_context[:500] + "\u2026") if len(industry_context) > 500 else industry_context
        reg_chunks = vault_retrieve(vault_query, k=5)
        reg_citations = format_citations(reg_chunks)
        system += "\n\nREGULATORY BACKGROUND (client-provided and vault):\n" + industry_context
        if reg_citations:
            system += "\n\nVault excerpts:\n" + reg_citations
    try:
        resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=f"Draft to review:\n{full}")])
        text = (resp.content or "").strip().upper()
        legal_ok = "VERDICT: SAFE" in text
        note = text.split("REJECTION NOTE:")[-1].strip() if "REJECTION NOTE:" in text else ""
        final_email = (draft + footer) if legal_ok else ""
        needs_review = state.get("needs_review", False) or (not legal_ok and attempts >= 3)
        do_not_contact = False
        if not legal_ok and attempts >= 3:
            do_not_contact = True
            add_failure_insight("Legal rejection 3 times for same sequence; lead marked do_not_contact", "legal_3x_reject", state.get("email", ""))
        return {
            **state,
            "legal_ok": legal_ok,
            "legal_note": note,
            "attempts": attempts,
            "final_email": final_email,
            "needs_review": needs_review,
            "do_not_contact": do_not_contact,
        }
    except Exception:
        return {**state, "legal_ok": True, "attempts": attempts}


def node_qualification_scoring(state: InboundLeadState) -> InboundLeadState:
    """Score intent + certainty from behavioral signals + LLM. Returns intent_score, certainty_score."""
    llm = get_llm()
    behavior = state.get("behavior", "")
    objection = state.get("initial_objection", "")
    reply_text = state.get("reply_text", "")
    email_num = state.get("email_num", 0)
    tone_analysis = state.get("reply_tone", "") or state.get("tone_analysis", "")
    lead_id = state.get("lead_id", "")
    history = get_lead_history(lead_id) if lead_id else []
    reply_count = sum(1 for h in history if (h.get("reply_received") or h.get("reply_text")))
    if reply_text:
        reply_count = max(reply_count, 1)
    llm_intent = None
    if llm:
        try:
            context = f"Behavior: {behavior}. Objection: {objection}. Reply: {reply_text[:500] if reply_text else 'None'}. Email {email_num} in sequence."
            resp = llm.invoke([
                SystemMessage(content="Score lead intent 0-100% based on behaviors, replies, objections. Reply: INTENT_SCORE: [0-100] and STATUS: [cold/warm/hot]"),
                HumanMessage(content=f"Context:\n{context}\n\nScore?"),
            ])
            text = (resp.content or "").strip()
            m = re.search(r"INTENT_SCORE:\s*(\d+)", text)
            llm_intent = int(m.group(1)) if m else None
        except Exception:
            pass
    intent_score, certainty_score = compute_intent_and_certainty(
        behavior=behavior,
        objection=objection,
        reply_text=reply_text,
        reply_count=reply_count,
        email_num=email_num,
        tone_analysis=tone_analysis,
        history=history,
        llm_intent=llm_intent,
    )
    qualification = scoring_qualification_status(intent_score)
    return {
        **state,
        "intent_score": intent_score,
        "certainty_score": certainty_score,
        "qualification_status": qualification,
    }


def node_reply_handler(state: InboundLeadState) -> InboundLeadState:
    """Analyze reply and generate response."""
    llm = get_llm()
    reply_text = state.get("reply_text", "")
    if not reply_text or llm is None:
        return state
    msgs = [
        SystemMessage(content="Analyze email reply tone and content. Reply: TONE: [formal/casual/younger/neutral] and KEY_POINTS: [list]"),
        HumanMessage(content=f"Reply:\n{reply_text}"),
    ]
    try:
        resp = llm.invoke(msgs)
        text = (resp.content or "").strip()
        tone_match = re.search(r"TONE:\s*(\w+)", text)
        tone = tone_match.group(1).lower() if tone_match else "neutral"
        # Generate response adapted to tone but anchored to CEO base
        chunks = vault_retrieve("CEO tone response handling", k=5)
        citations = format_citations(chunks)
        system = f"""You are a digital clone of the CEO. Generate a response to this reply.
Adapt slightly to recipient tone ({tone}) but anchor to CEO base/No-Go list. Never sound salesy.
{GOLDEN_RULES}"""
        user = f"""VAULT:\n{citations}\n\nReply received:\n{reply_text}\n\nGenerate CEO-style response (email body only)."""
        resp_gen = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        draft = (resp_gen.content or "").strip()
        return {
            **state,
            "reply_tone": tone,
            "draft": draft,
            "citations": citations,
        }
    except Exception:
        return state


def _route_after_legal(state: InboundLeadState) -> str:
    if state.get("legal_ok"):
        return "qualification"
    if state.get("attempts", 0) >= 3:
        return "end"
    return "writer"


def analyze_reply(incoming_reply: str) -> dict:
    """Analyze reply tone and key points; returns {tone: str, key_points: list} for memory/context."""
    reply = (incoming_reply or "").strip()
    if not reply:
        return {"tone": "neutral", "key_points": []}
    llm = get_llm()
    if llm is None:
        return {"tone": "neutral", "key_points": []}
    try:
        resp = llm.invoke([
            SystemMessage(content="Analyze email reply tone and content. Reply: TONE: [formal/casual/younger/neutral] and KEY_POINTS: [comma-separated list]"),
            HumanMessage(content=f"Reply:\n{reply}"),
        ])
        text = (resp.content or "").strip()
        tone_match = re.search(r"TONE:\s*(\w+)", text)
        tone = tone_match.group(1).lower() if tone_match else "neutral"
        key_points = []
        kp_match = re.search(r"KEY_POINTS?:\s*\[?(.*?)\]?", text, re.DOTALL | re.IGNORECASE)
        if kp_match:
            raw = kp_match.group(1).strip()
            key_points = [x.strip() for x in re.split(r"[,;]", raw) if x.strip()][:8]
        return {"tone": tone, "key_points": key_points}
    except Exception:
        return {"tone": "neutral", "key_points": []}


def _last_event_ts_and_type(history: list) -> tuple:
    """Return (last_timestamp_iso, 'email_sent'|'reply_received'|None)."""
    if not history:
        return None, None
    for h in reversed(history):
        ts = h.get("timestamp")
        if h.get("reply_received"):
            return ts, "reply_received"
        if h.get("email_sent"):
            return ts, "email_sent"
    return None, None


def _days_since(ts_iso: Optional[str]) -> Optional[int]:
    if not ts_iso:
        return None
    try:
        dt = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - dt).days
    except Exception:
        return None
