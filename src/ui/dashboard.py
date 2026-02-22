"""
Kinesis Reactivation — Dashboard tab UI component.
"""

from __future__ import annotations

import os

import streamlit as st
import pandas as pd

from src.config import REJECT_REASON_OPTIONS
from src.llm.providers import get_llm
from src.tools.playbook import (
    add_approved_to_playbook,
    add_rejected_to_playbook,
    add_training_pair_to_playbook,
)
from src.tools.funnel import (
    compute_funnel_stats,
    compute_deep_funnel_analytics as compute_deep_funnel,
    log_funnel_events,
    load_logged_funnel_events,
)
from src.api.sendgrid import send_email_via_sendgrid
from src.agents.state_machine import run_inbound_sequence


def render_dashboard(csv_file, legal_footer: str, from_email: str, from_name: str):
    """Render the Dashboard tab contents."""
    if csv_file:
        try:
            df = pd.read_csv(csv_file)
            required_cols = ["name", "email"]
            if not all(c in df.columns for c in required_cols):
                st.error(f"CSV must have columns: {required_cols}")
            else:
                st.subheader("\U0001f4ca Leads Dashboard")
                st.dataframe(df.head(50), width="stretch")

                if st.button("Generate Nurturing Sequences", type="primary"):
                    api_key = os.getenv("MISTRAL_API_KEY")
                    if not api_key or api_key == "your-mistral-key-here":
                        st.error("Missing MISTRAL_API_KEY in .env")
                    else:
                        progress = st.progress(0)
                        sequences_results = []
                        for i, (_, row) in enumerate(df.iterrows()):
                            progress.progress((i + 1) / len(df))
                            row_dict = row.to_dict()
                            sequences = run_inbound_sequence(row_dict, legal_footer)
                            sequences_results.append({"lead": row_dict, "sequences": sequences})
                        progress.empty()
                        st.session_state["sequences_results"] = sequences_results
                        st.session_state["approved_green"] = set()

                if st.button("Test Sequence for First Lead", type="secondary"):
                    api_key = os.getenv("MISTRAL_API_KEY")
                    if not api_key or api_key == "your-mistral-key-here":
                        st.error("Missing MISTRAL_API_KEY in .env")
                    elif len(df) == 0:
                        st.warning("No rows in CSV.")
                    else:
                        with st.spinner("Running sequence for first lead..."):
                            first_lead = df.iloc[0].to_dict()
                            test_sequences = run_inbound_sequence(first_lead, legal_footer)
                            st.session_state["test_sequence_result"] = {"lead": first_lead, "sequences": test_sequences}

                if st.session_state.get("test_sequence_result"):
                    tr = st.session_state["test_sequence_result"]
                    lead_name = tr["lead"].get("name", "Lead")
                    lead_source = tr["lead"].get("source", "unknown")
                    with st.expander("Test Results", expanded=True):
                        st.markdown(f"**Lead:** {lead_name} | **Source:** {lead_source}")
                        for seq_idx, seq in enumerate(tr["sequences"]):
                            intent = seq.get("intent_score", 0)
                            status = seq.get("qualification_status", "warm")
                            conf = seq.get("draft_confidence_score")
                            vault_align = seq.get("draft_vault_alignment", "")
                            conf_val = conf if conf is not None else 0
                            if conf_val >= 80:
                                bar_bg, bar_text = "#28a745", "white"
                            elif conf_val >= 55:
                                bar_bg, bar_text = "#ffc107", "#212529"
                            else:
                                bar_bg, bar_text = "#dc3545", "white"
                            st.markdown(
                                f'<div style="background-color: {bar_bg}; color: {bar_text}; padding: 8px 12px; margin: 8px 0; border-radius: 4px; font-weight: bold;">Score: {conf_val}/100</div>',
                                unsafe_allow_html=True,
                            )
                            with st.expander("Compliance rules active for this client", expanded=False):
                                _rules = st.session_state.get("custom_compliance_rules") or ""
                                if (_rules or "").strip():
                                    st.text(_rules)
                                else:
                                    st.caption("Using general compliance rules only \u2014 no client-specific rules configured.")
                            draft_text = seq.get("final_email") or seq.get("draft") or ""
                            st.markdown(f"---")
                            certainty = seq.get("certainty_score")
                            st.markdown(f"**Email {seq.get('email_num', seq_idx + 1)}** \u2014 Intent: {intent}% | Certainty: {certainty}% | Status: {status}" + (f" | **Confidence:** {conf}/100 ({vault_align})" if conf is not None else ""))
                            if conf is not None and (seq.get("draft_confidence_explanation") or seq.get("draft_risk_flags")):
                                with st.expander("Draft evaluation", expanded=False):
                                    st.caption(seq.get("draft_confidence_explanation", ""))
                                    if seq.get("draft_risk_flags") and str(seq.get("draft_risk_flags", "")).lower() not in ("none", ""):
                                        st.caption(f"Risk flags: {seq.get('draft_risk_flags')}")
                            if draft_text and "Generation failed" in draft_text:
                                st.warning("Generation failed \u2014 see draft below for error details.")
                            if not draft_text:
                                st.error("No draft \u2014 check MISTRAL_API_KEY (or OPENAI fallback) and terminal.")
                            st.text_area("Draft", value=draft_text or "(no draft)", height=180, key=f"test_draft_{seq_idx}", disabled=True)

                if "sequences_results" in st.session_state:
                    results = st.session_state["sequences_results"]
                    all_high_conf = all(s.get("draft_confidence_score", 0) >= 80 and not s.get("needs_review", True) for r in results for s in r["sequences"])
                    if all_high_conf and st.button("\U0001f680 Approve & Send All High Confidence Emails", type="primary"):
                        sent_count = 0
                        for r in results:
                            for s in r["sequences"]:
                                if s.get("draft_confidence_score", 0) >= 80 and not s.get("needs_review", True):
                                    body = s.get("final_email") or s.get("draft") or ""
                                    if body and send_email_via_sendgrid(
                                        r["lead"].get("email", ""),
                                        r["lead"].get("name", ""),
                                        f"Re: {r['lead'].get('source', 'Inquiry')}",
                                        body,
                                        from_email,
                                        from_name,
                                    ):
                                        subj = f"Re: {r['lead'].get('source', 'Inquiry')}"
                                        add_approved_to_playbook(
                                            body, s.get("user_prompt", ""), r["lead"].get("name", ""),
                                            lead_source=r["lead"].get("source", ""),
                                            confidence_score=s.get("draft_confidence_score", 0),
                                            subject_line=subj,
                                        )
                                        sent_count += 1
                        if sent_count:
                            st.success(f"Sent {sent_count} high-confidence email(s) via SendGrid.")
                    funnel_data = []
                    for r in results:
                        for seq in r["sequences"]:
                            funnel_data.append(seq)
                    log_funnel_events(funnel_data)
                    funnel_stats = compute_funnel_stats(funnel_data)
                    deep_funnel = compute_deep_funnel(funnel_data, llm=get_llm())
                    st.subheader("\U0001f4c8 Funnel Analysis (Wald-style)")
                    st.json(funnel_stats)
                    st.subheader("\U0001f3af Deep Funnel Death Analytics (Management)")
                    st.markdown(f"**Main objection:** {deep_funnel.get('main_objection', 'N/A')} | **Emotional reason:** {deep_funnel.get('emotional_reason', 'N/A')}")
                    if deep_funnel.get("common_pattern"):
                        st.caption(f"Common pattern: {deep_funnel['common_pattern']}")
                    with st.expander("Death stories", expanded=False):
                        for ds in deep_funnel.get("death_stories", [])[:5]:
                            st.markdown(f"**{ds.get('stage', '')} \u2014 {ds.get('main_objection', '')}**")
                            st.caption(ds.get("death_story", ""))

                    all_time_events = load_logged_funnel_events()
                    if all_time_events:
                        with st.expander("All-time funnel (logged events)", expanded=False):
                            all_time_funnel = compute_deep_funnel(all_time_events, llm=get_llm())
                            st.caption(f"Total logged events: {len(all_time_events)}")
                            st.markdown(f"**Main objection:** {all_time_funnel.get('main_objection', 'N/A')} | **Emotional reason:** {all_time_funnel.get('emotional_reason', 'N/A')}")

                    # Qualification scoring
                    hot_leads = [r for r in results if any(s.get("intent_score", 0) >= 80 for s in r["sequences"])]
                    st.metric("Hot Leads (\u226580% intent)", len(hot_leads), f"{len(hot_leads) * 1000} potential value")

                    # Batch-level confidence messages (before individual emails)
                    all_emails = [s for r in results for s in r["sequences"]]
                    all_high_conf = all(s.get("draft_confidence_score", 0) >= 80 for s in all_emails)
                    if all_emails and all_high_conf:
                        st.success("All Clear \u2014 all emails are high confidence and ready to approve.")
                    low_conf_count = sum(1 for s in all_emails if s.get("draft_confidence_score", 0) < 55)
                    if low_conf_count > 0:
                        st.warning(f"{low_conf_count} email(s) require review before sending")

                    # Summary panel: green / yellow / red counts and Approve All Green
                    green_count = sum(1 for s in all_emails if s.get("draft_confidence_score", 0) >= 80)
                    yellow_count = sum(1 for s in all_emails if 55 <= s.get("draft_confidence_score", 0) < 80)
                    red_count = sum(1 for s in all_emails if s.get("draft_confidence_score", 0) < 55)
                    st.subheader("\U0001f4cb Email summary")
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("Emails Ready to Send", green_count, delta="Ready to send", delta_color="normal")
                    with c2:
                        st.metric("Emails Need Review", yellow_count, delta="Needs review", delta_color="off")
                    with c3:
                        st.metric("Emails Must Be Edited", red_count, delta="Must edit", delta_color="inverse")
                    approved_green = st.session_state.get("approved_green", set())
                    if st.button("Approve All Green", type="primary", key="approve_all_green"):
                        newly_approved = 0
                        for idx, r in enumerate(results):
                            for seq_idx, s in enumerate(r["sequences"]):
                                if s.get("draft_confidence_score", 0) >= 80 and (idx, seq_idx) not in approved_green:
                                    body = s.get("final_email") or s.get("draft") or ""
                                    if body:
                                        subj = f"Re: {r['lead'].get('source', 'Inquiry')}"
                                        add_approved_to_playbook(
                                            body,
                                            s.get("user_prompt", ""),
                                            r["lead"].get("name", ""),
                                            lead_source=r["lead"].get("source", ""),
                                            confidence_score=s.get("draft_confidence_score", 0),
                                            subject_line=subj,
                                        )
                                        approved_green.add((idx, seq_idx))
                                        newly_approved += 1
                        st.session_state["approved_green"] = approved_green
                        st.success(f"{newly_approved} email(s) approved and queued for sending")

                    # Per-lead display (each lead in expander, each sequence email shown)
                    for idx, r in enumerate(results):
                        lead_label = f"{r['lead'].get('name', 'Lead')} \u2014 {r['lead'].get('source', 'unknown')}"
                        with st.expander(lead_label):
                            # Reactivation leads: show pre-contact validation summary
                            if str(r.get("lead", {}).get("source", "")).lower() == "reactivation":
                                val = None
                                for s in r.get("sequences", []):
                                    val = s.get("reactivation_validation")
                                    if val:
                                        break
                                if val:
                                    overall = val.get("overall", 0)
                                    if overall >= 80:
                                        v_color = "\U0001f7e2"
                                    elif overall >= 60:
                                        v_color = "\U0001f7e1"
                                    else:
                                        v_color = "\U0001f534"
                                    st.markdown(f"**Reactivation validation** \u2014 {v_color} Overall: {overall}/100 (Relevance: {val.get('relevance', 0)}, Buying readiness: {val.get('buying', 0)}) | Proceed: **{val.get('proceed', 'no')}**")
                                    st.caption(val.get("explanation", ""))
                                    st.divider()
                            for seq_idx, seq in enumerate(r["sequences"]):
                                approved_green_ids = st.session_state.get("approved_green", set())
                                if (idx, seq_idx) in approved_green_ids:
                                    st.success("\u2713 Approved and queued for sending")
                                    continue
                                email_num = seq.get("email_num", seq_idx + 1)
                                intent = seq.get("intent_score", 0)
                                qual = seq.get("qualification_status", "warm")
                                conf = seq.get("draft_confidence_score", 0)
                                if conf >= 80:
                                    color = "\U0001f7e2"  # green
                                elif conf >= 60:
                                    color = "\U0001f7e1"  # yellow
                                else:
                                    color = "\U0001f534"  # red
                                vault_align = seq.get("draft_vault_alignment", "")
                                risk_flags = seq.get("draft_risk_flags", "") or "none"
                                draft_text = seq.get("final_email") or seq.get("draft") or ""
                                certainty = seq.get("certainty_score")

                                # Full-width colour-coded header bar (above draft)
                                if conf >= 80:
                                    bar_bg, bar_text = "#28a745", "white"
                                elif conf >= 55:
                                    bar_bg, bar_text = "#ffc107", "#212529"
                                else:
                                    bar_bg, bar_text = "#dc3545", "white"
                                st.markdown(
                                    f'<div style="background-color: {bar_bg}; color: {bar_text}; padding: 8px 12px; margin: 8px 0; border-radius: 4px; font-weight: bold;">Score: {conf}/100</div>',
                                    unsafe_allow_html=True,
                                )
                                with st.expander("Compliance rules active for this client", expanded=False):
                                    _rules = st.session_state.get("custom_compliance_rules") or ""
                                    if (_rules or "").strip():
                                        st.text(_rules)
                                    else:
                                        st.caption("Using general compliance rules only \u2014 no client-specific rules configured.")
                                st.markdown(f"Score {conf}/100 | Vault alignment: {vault_align} | Risk flags: {risk_flags}")
                                st.markdown(f"**Email {email_num}** \u2014 Confidence: {color} {conf}/100 | Intent: {intent}% | Status: {qual}")
                                if conf is not None and (seq.get("draft_confidence_explanation") or seq.get("draft_risk_flags")):
                                    with st.expander("Draft evaluation", expanded=False):
                                        st.caption(seq.get("draft_confidence_explanation", ""))
                                        if seq.get("draft_risk_flags") and str(seq.get("draft_risk_flags", "")).lower() not in ("none", ""):
                                            st.caption(f"Risk flags: {seq.get('draft_risk_flags')}")
                                if draft_text and "Generation failed" in draft_text:
                                    st.warning("Generation failed \u2014 see draft below for error details.")
                                if draft_text:
                                    if f"original_draft_{idx}_{seq_idx}" not in st.session_state:
                                        st.session_state[f"original_draft_{idx}_{seq_idx}"] = draft_text
                                    edited = st.text_area(
                                        "Draft",
                                        value=draft_text,
                                        height=200,
                                        key=f"main_seq_{idx}_{seq_idx}",
                                    )
                                    read_checked = st.checkbox("I have read this email in full", key=f"read_{idx}_{seq_idx}")
                                    c1, c2 = st.columns(2)
                                    with c1:
                                        if read_checked and st.button("\u2705 Approve & Send", key=f"send_{idx}_{seq_idx}"):
                                            subj = f"Re: {r['lead'].get('source', 'Inquiry')}"
                                            if send_email_via_sendgrid(
                                                r["lead"].get("email", ""),
                                                r["lead"].get("name", ""),
                                                subj,
                                                edited,
                                                from_email,
                                                from_name,
                                            ):
                                                add_approved_to_playbook(
                                                    edited, seq.get("user_prompt", ""), r["lead"].get("name", ""),
                                                    lead_source=r["lead"].get("source", ""),
                                                    confidence_score=conf,
                                                    subject_line=subj,
                                                )
                                                original = st.session_state.get(f"original_draft_{idx}_{seq_idx}", "")
                                                if original.strip() and edited.strip() and original.strip() != edited.strip():
                                                    lead_id = seq.get("lead_id") or r.get("lead", {}).get("lead_id") or ""
                                                    if lead_id:
                                                        add_training_pair_to_playbook(original, edited, lead_id)
                                                st.success("Sent via SendGrid.")
                                    with c2:
                                        reject_reason = st.selectbox(
                                            "Why are you rejecting this email?",
                                            options=REJECT_REASON_OPTIONS,
                                            key=f"reject_reason_{idx}_{seq_idx}",
                                        )
                                        other_text = ""
                                        if reject_reason == "Other (describe below)":
                                            other_text = st.text_input("Please describe", key=f"reject_other_{idx}_{seq_idx}", placeholder="Reason for rejection")
                                        has_reason = (reject_reason != "Other (describe below)") or (reject_reason == "Other (describe below)" and (other_text or "").strip())
                                        if has_reason and st.button("Confirm Rejection", key=f"rej_{idx}_{seq_idx}"):
                                            final_reason = (other_text or "").strip() if reject_reason == "Other (describe below)" else reject_reason
                                            add_rejected_to_playbook(final_reason, edited[:500])
                                            st.warning("Rejected saved.")
                                else:
                                    st.error("No draft generated \u2014 check MISTRAL_API_KEY (or OPENAI fallback) and terminal for errors.")

        except Exception as e:
            st.error(str(e))
