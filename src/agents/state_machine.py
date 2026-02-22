"""
Kinesis Reactivation — State machine, graph builder, and sequence runner.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, List

from src.config import (
    _dbg,
    SCENARIO_CONFIG,
    LEGAL_FOOTER,
    HOT_SCORE_THRESHOLD,
)
from src.llm.providers import get_llm
from src.tools.scoring_engine import compute_intent_and_certainty, qualification_status as scoring_qualification_status
from src.tools.objections import extract_objection
from src.tools.learning_loop import run_learning_loop
from src.db import (
    get_lead,
    get_lead_by_email,
    get_lead_history,
    upsert_lead,
    append_history,
)
from src.api.sendgrid import send_hot_lead_alert
from src.agents.state import InboundLeadState
from src.agents.nodes import (
    validate_reactivation_lead,
    _route_after_validate_reactivation,
    node_enrich,
    node_branch_scenario,
    node_writer_nurture,
    node_legal,
    node_qualification_scoring,
    node_reply_handler,
    _route_after_legal,
    analyze_reply,
    _last_event_ts_and_type,
    _days_since,
)


def process_lead_state(lead_id: str, incoming_reply: Optional[str] = None) -> dict:
    """
    Reactive state machine: ONE cycle per call.
    - Load lead memory & score.
    - If score > 80 (HOT): return HANDOFF_TO_SALES (no email).
    - If incoming_reply: analyze_reply -> update memory -> generate ONE response draft (reply + Vault).
    - If new lead (no history): generate ONLY first Icebreaker email.
    - If follow-up (no reply for NUDGE_DAYS): generate ONE Nudge email.
    Returns dict with status, draft, intent_score, etc.
    """
    lead = get_lead(lead_id)
    if not lead:
        _dbg("process_lead_state lead not found", {"lead_id": lead_id}, "H4")
        return {"status": "ERROR", "error": "Lead not found", "lead_id": lead_id}
    history = lead.get("history", [])
    # #region agent log
    _dbg("process_lead_state entry", {"lead_id": lead_id, "history_len": len(history), "has_incoming_reply": bool(incoming_reply)}, "H3")
    # #endregion
    raw = lead.get("raw_row") or {}
    row = {
        "email": lead.get("email", ""),
        "name": lead.get("name", ""),
        "source": lead.get("source", ""),
        "behavior": raw.get("behavior", ""),
        "initial_objection": raw.get("initial_objection", ""),
        "recent_news": raw.get("recent_news", ""),
        "role": raw.get("role", ""),
        "company": raw.get("company", ""),
        "funding": raw.get("funding", ""),
        "website": raw.get("website", ""),
    }
    row["opt_out"] = raw.get("opt_out", False)
    row["do_not_contact"] = raw.get("do_not_contact", False)
    reply_count = sum(1 for h in history if h.get("reply_received"))
    last_ts, last_type = _last_event_ts_and_type(history)
    intent_score, certainty_score = compute_intent_and_certainty(
        behavior=row.get("behavior", ""),
        objection=row.get("initial_objection", ""),
        reply_text=incoming_reply or "",
        reply_count=reply_count + (1 if incoming_reply else 0),
        email_num=len([h for h in history if h.get("email_sent")]) + 1,
        tone_analysis="",
        history=history,
    )
    if intent_score >= HOT_SCORE_THRESHOLD and not incoming_reply:
        _dbg("process_lead_state HANDOFF", {"intent_score": intent_score}, "H3")
        lead_data = {
            "lead_id": lead_id,
            "name": lead.get("name", ""),
            "company": row.get("company", ""),
            "source": lead.get("source", ""),
            "intent_score": intent_score,
        }
        send_hot_lead_alert(lead_data)
        return {
            "status": "HANDOFF_TO_SALES",
            "lead_id": lead_id,
            "intent_score": intent_score,
            "certainty_score": certainty_score,
            "qualification_status": scoring_qualification_status(intent_score),
            "message": "Lead is HOT; do not generate email. Alert Admin.",
            "draft": "",
        }
    scenario = SCENARIO_CONFIG.get(str(row.get("source", "")).lower(), SCENARIO_CONFIG["default"])
    legal_footer = LEGAL_FOOTER
    compiled = build_inbound_graph()
    config = {"configurable": {"thread_id": str(lead_id)}}

    if incoming_reply:
        _dbg("process_lead_state branch: incoming_reply", {}, "H3")
        analysis = analyze_reply(incoming_reply)
        obj_label = extract_objection(incoming_reply)[0]
        append_history(
            lead_id,
            stage="reply_received",
            reply_received=incoming_reply,
            tone_analysis=analysis.get("tone", "neutral"),
            detected_objection=obj_label,
            event_data={"key_points": analysis.get("key_points", [])},
        )
        history = get_lead_history(lead_id)
        intent_score, certainty_score = compute_intent_and_certainty(
            behavior=row.get("behavior", ""),
            objection=row.get("initial_objection", ""),
            reply_text=incoming_reply,
            reply_count=len([h for h in history if h.get("reply_received")]),
            email_num=len([h for h in history if h.get("email_sent")]) + 1,
            tone_analysis=analysis.get("tone", ""),
            history=history,
        )
        initial: InboundLeadState = {
            "row": row,
            "email": row.get("email", ""),
            "name": row.get("name", "there"),
            "source": row.get("source", ""),
            "behavior": row.get("behavior", ""),
            "initial_objection": row.get("initial_objection", ""),
            "enriched_data": {},
            "scenario_config": scenario,
            "email_num": len([h for h in history if h.get("email_sent")]) + 1,
            "sequence_emails": [],
            "legal_footer": legal_footer,
            "reply_text": incoming_reply,
            "reply_tone": analysis.get("tone", ""),
            "latest_incoming_message": incoming_reply,
            "is_nudge": False,
        }
        final = compiled.invoke(initial, config=config)
        if final.get("do_not_contact") and final.get("reactivation_validation"):
            return {
                "status": "VALIDATION_BLOCKED",
                "lead_id": lead_id,
                "draft": final.get("draft", ""),
                "intent_score": 0,
                "certainty_score": 0,
                "qualification_status": "cold",
                "needs_review": False,
                "reactivation_validation": final.get("reactivation_validation"),
            }
        draft = final.get("final_email") or final.get("draft", "")
        _dbg("process_lead_state DRAFT_READY (reply)", {"status": "DRAFT_READY", "draft_len": len(draft or "")}, "H3")
        return {
            "status": "DRAFT_READY",
            "lead_id": lead_id,
            "draft": draft,
            "intent_score": final.get("intent_score", intent_score),
            "certainty_score": final.get("certainty_score", certainty_score),
            "qualification_status": final.get("qualification_status", "warm"),
            "needs_review": final.get("needs_review", False),
        }
    if not history:
        _dbg("process_lead_state branch: new_lead (icebreaker)", {"history_len": 0}, "H3")
        initial = {
            "row": row,
            "email": "",
            "name": "",
            "source": "",
            "behavior": "",
            "initial_objection": "",
            "enriched_data": {},
            "scenario_config": scenario,
            "email_num": 1,
            "sequence_emails": [],
            "legal_footer": legal_footer,
            "reply_text": "",
        "latest_incoming_message": "",
        "is_nudge": False,
        }
        try:
            final = compiled.invoke(initial, config=config)
        except Exception as ex:
            _dbg("process_lead_state graph invoke exception (new_lead)", {"err": str(ex)[:300]}, "H4")
            raise
        if final.get("do_not_contact") and final.get("reactivation_validation"):
            _dbg("process_lead_state VALIDATION_BLOCKED (new_lead)", {}, "H3")
            return {
                "status": "VALIDATION_BLOCKED",
                "lead_id": lead_id,
                "draft": final.get("draft", ""),
                "intent_score": 0,
                "certainty_score": 0,
                "qualification_status": "cold",
                "needs_review": False,
                "reactivation_validation": final.get("reactivation_validation"),
            }
        draft = final.get("final_email") or final.get("draft", "")
        lead_id_out = final.get("lead_id", lead_id)
        _dbg("process_lead_state ICEBREAKER_READY", {"status": "ICEBREAKER_READY", "draft_len": len(draft or "")}, "H3")
        append_history(
            lead_id_out,
            stage="icebreaker",
            email_sent=(final.get("final_email") or final.get("draft", ""))[:5000],
            intent_score=final.get("intent_score"),
            certainty_score=final.get("certainty_score"),
            qualification_status=final.get("qualification_status"),
            event_data={"email_num": 1},
        )
        return {
            "status": "ICEBREAKER_READY",
            "lead_id": lead_id_out,
            "draft": draft,
            "intent_score": final.get("intent_score", 0),
            "certainty_score": final.get("certainty_score", 0),
            "qualification_status": final.get("qualification_status", "warm"),
            "needs_review": final.get("needs_review", False),
        }
    days = _days_since(last_ts)
    # Only NO_ACTION when lead replied and we have no incoming_reply (waiting for next reply).
    # When last was email_sent, always generate nudge draft so user can review (cadence is advisory for manual Generate).
    if last_type == "reply_received":
        _dbg("process_lead_state NO_ACTION (reply_received)", {"last_type": last_type}, "H3")
        return {
            "status": "NO_ACTION",
            "lead_id": lead_id,
            "message": "Lead already replied; use handle_incoming_reply with their message to generate a response.",
            "draft": "",
        }
    _dbg("process_lead_state branch: nudge", {"last_type": last_type, "days": days}, "H3")
    sent_bodies = [h.get("email_sent", "") for h in history if h.get("email_sent")]
    initial = {
        "row": row,
        "email": row.get("email", ""),
        "name": row.get("name", "there"),
        "source": row.get("source", ""),
        "behavior": row.get("behavior", ""),
        "initial_objection": row.get("initial_objection", ""),
        "enriched_data": {},
        "scenario_config": scenario,
        "email_num": len(sent_bodies) + 1,
        "sequence_emails": sent_bodies,
        "legal_footer": legal_footer,
        "reply_text": "",
        "latest_incoming_message": "",
        "is_nudge": True,
    }
    final = compiled.invoke(initial, config=config)
    if final.get("do_not_contact") and final.get("reactivation_validation"):
        _dbg("process_lead_state VALIDATION_BLOCKED (nudge)", {}, "H3")
        return {
            "status": "VALIDATION_BLOCKED",
            "lead_id": lead_id,
            "draft": final.get("draft", ""),
            "intent_score": 0,
            "certainty_score": 0,
            "qualification_status": "cold",
            "needs_review": False,
            "reactivation_validation": final.get("reactivation_validation"),
        }
    draft = final.get("final_email") or final.get("draft", "")
    _dbg("process_lead_state NUDGE_READY", {"status": "NUDGE_READY", "draft_len": len(draft or "")}, "H3")
    lead_id_out = final.get("lead_id", lead_id)
    append_history(
        lead_id_out,
        stage=f"nudge_{len(sent_bodies) + 1}",
        email_sent=(final.get("final_email") or final.get("draft", ""))[:5000],
        intent_score=final.get("intent_score"),
        certainty_score=final.get("certainty_score"),
        qualification_status=final.get("qualification_status"),
        event_data={"email_num": len(sent_bodies) + 1},
    )
    return {
        "status": "NUDGE_READY",
        "lead_id": lead_id_out,
        "draft": draft,
        "intent_score": final.get("intent_score", 0),
        "certainty_score": final.get("certainty_score", 0),
        "qualification_status": final.get("qualification_status", "warm"),
        "needs_review": final.get("needs_review", False),
    }


def handle_incoming_reply(lead_email: str, reply_text: str) -> dict:
    """
    Reply trigger: find lead by email, append reply to history, run process_lead_state with the reply.
    Returns the same dict as process_lead_state (status, draft, etc.).
    """
    lead = get_lead_by_email(lead_email)
    if not lead:
        return {"status": "ERROR", "error": f"No lead found for email: {lead_email}", "draft": ""}
    lead_id = lead.get("lead_id", "")
    return process_lead_state(lead_id, incoming_reply=(reply_text or "").strip())


def build_inbound_graph():
    """Build LangGraph for inbound nurturing."""
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    graph = StateGraph(InboundLeadState)
    graph.add_node("enrich", node_enrich)
    graph.add_node("validate_reactivation", validate_reactivation_lead)
    graph.add_node("branch", node_branch_scenario)
    graph.add_node("writer", node_writer_nurture)
    graph.add_node("legal", node_legal)
    graph.add_node("qualification", node_qualification_scoring)
    graph.add_node("reply_handler", node_reply_handler)
    graph.set_entry_point("enrich")
    graph.add_edge("enrich", "validate_reactivation")
    graph.add_conditional_edges("validate_reactivation", _route_after_validate_reactivation, {"end": END, "branch": "branch"})
    graph.add_edge("branch", "writer")
    graph.add_edge("writer", "legal")
    graph.add_conditional_edges("legal", _route_after_legal, {"qualification": "qualification", "writer": "writer", "end": END})
    graph.add_edge("qualification", END)
    return graph.compile(checkpointer=MemorySaver())


def run_inbound_sequence(lead_row: dict, legal_footer: str) -> List[InboundLeadState]:
    """
    Legacy wrapper: one step per lead via process_lead_state(lead_id).
    Ensures lead exists, then runs reactive single-step. Returns list of one result for UI compatibility.
    """
    if lead_row.get("opt_out") or lead_row.get("do_not_contact"):
        return [{
            "row": lead_row,
            "email": lead_row.get("email", ""),
            "do_not_contact": True,
            "qualification_status": "cold",
            "intent_score": 0,
            "final_email": "",
            "draft": "",
            "sequence_emails": [],
        }]
    lead_id = upsert_lead(lead_row)
    result = process_lead_state(lead_id, incoming_reply=None)
    status = result.get("status", "NO_ACTION")
    draft = result.get("draft", "")
    one: InboundLeadState = {
        "row": lead_row,
        "email": lead_row.get("email", ""),
        "name": lead_row.get("name", ""),
        "source": lead_row.get("source", ""),
        "lead_id": lead_id,
        "intent_score": result.get("intent_score", 0),
        "certainty_score": result.get("certainty_score", 0),
        "qualification_status": result.get("qualification_status", "warm"),
        "needs_review": result.get("needs_review", False),
        "final_email": draft if status in ("ICEBREAKER_READY", "NUDGE_READY", "DRAFT_READY") else "",
        "draft": draft,
        "sequence_emails": [],
        "email_num": 1,
    }
    if result.get("reactivation_validation"):
        one["reactivation_validation"] = result["reactivation_validation"]
    if status == "VALIDATION_BLOCKED":
        one["do_not_contact"] = True
        one["qualification_status"] = "cold"
        one["intent_score"] = 0
        one["final_email"] = ""
    if status == "HANDOFF_TO_SALES":
        one["qualification_status"] = "hot"
        one["draft"] = ""
        one["final_email"] = ""
    try:
        run_learning_loop(
            "converted" if status == "HANDOFF_TO_SALES" else "died",
            {**lead_row, "lead_id": lead_id},
            [one],
            llm=get_llm(),
        )
    except Exception:
        pass
    return [one]
