"""
Kinesis Reactivation — Statistics tab UI component.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import streamlit as st
import pandas as pd

from src.config import LEARNING_SIGNALS_PATH
from src.db import get_all_lead_ids, get_lead
from src.tools.playbook import load_playbook


def render_statistics():
    """Render the Statistics tab contents."""
    # ---- Section 1: Overview metrics ----
    period = st.radio("Period", ["This month", "All time"], horizontal=True, key="stats_period")
    now = datetime.now(timezone.utc)
    this_month_prefix = now.strftime("%Y-%m")
    lead_ids = get_all_lead_ids()
    if period == "This month":
        lead_ids = [lid for lid in lead_ids if lid and (get_lead(lid) or {}).get("created_at", "").startswith(this_month_prefix)]
    leads = [get_lead(lid) for lid in lead_ids]
    leads = [l for l in leads if l is not None]
    total_emails_sent = sum(len([h for h in (l.get("history") or []) if h.get("email_sent")]) for l in leads)
    total_replies = sum(len([h for h in (l.get("history") or []) if h.get("reply_received")]) for l in leads)
    playbook = load_playbook()
    if period == "This month":
        approved_entries = [e for e in playbook.get("approved", []) if (e.get("timestamp") or "").startswith(this_month_prefix)]
    else:
        approved_entries = playbook.get("approved", [])
    emails_approved = len(approved_entries)
    hot_count = sum(1 for l in leads if any((h.get("intent_score") or 0) >= 80 for h in (l.get("history") or [])))
    reply_rate_pct = (total_replies / total_emails_sent * 100) if total_emails_sent else 0
    avg_conf = sum(e.get("confidence_score", 0) for e in approved_entries) / len(approved_entries) if approved_entries else 0
    st.subheader("SECTION 1 \u2014 OVERVIEW METRICS")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total leads processed", len(leads))
    with col2:
        st.metric("Emails sent and human-approved", emails_approved)
    with col3:
        st.metric("Reply rate (%)", round(reply_rate_pct, 1))
    with col4:
        st.metric("Leads handed to sales", hot_count)
    with col5:
        st.metric("Avg confidence (approved)", round(avg_conf, 1))

    # ---- Section 2: Funnel dropout ----
    st.subheader("SECTION 2 \u2014 FUNNEL DROPOUT ANALYSIS")
    st.markdown("**Where Leads Are Dying \u2014 Abraham Wald Analysis**")
    st.caption("Focus on the stages where leads stop responding. These are the highest-value improvement opportunities.")
    stages = ["Uploaded", "Enriched", "Email 1 Sent", "Email 1 Replied", "Email 2 Sent", "Email 2 Replied", "Email 3 Sent", "HOT (Handed to Sales)"]
    all_leads = [get_lead(lid) for lid in get_all_lead_ids() if get_lead(lid)]
    if period == "This month":
        all_leads = [l for l in all_leads if (l.get("created_at") or "").startswith(this_month_prefix)]
    total_entered = len(all_leads)
    reached = [0] * 9
    reached[1] = reached[2] = total_entered
    for l in all_leads:
        hist = l.get("history") or []
        emails = len([h for h in hist if h.get("email_sent")])
        replies = len([h for h in hist if h.get("reply_received")])
        max_intent = max([h.get("intent_score", 0) or 0 for h in hist], default=0)
        if emails >= 1:
            reached[3] += 1
        if replies >= 1:
            reached[4] += 1
        if emails >= 2:
            reached[5] += 1
        if replies >= 2:
            reached[6] += 1
        if emails >= 3:
            reached[7] += 1
        if max_intent >= 80:
            reached[8] += 1
    funnel_rows = []
    for i, name in enumerate(stages):
        idx = i + 1
        entered = reached[idx] if idx < len(reached) else 0
        next_entered = reached[idx + 1] if idx + 1 < len(reached) else 0
        stopped = entered - next_entered
        pct = (stopped / total_entered * 100) if total_entered else 0
        funnel_rows.append({"Stage": name, "Reached": entered, "Stopped": stopped, "Dropout %": round(pct, 1)})
    funnel_df = pd.DataFrame(funnel_rows)
    st.dataframe(funnel_df, width="stretch")
    max_stopped_idx = max(range(len(funnel_rows)), key=lambda i: funnel_rows[i]["Stopped"]) if funnel_rows else 0
    for i, row in enumerate(funnel_rows):
        dropout_frac = (row["Dropout %"] / 100.0) if row["Dropout %"] else 0
        st.progress(max(0.0, min(dropout_frac, 1.0)))
        if i == max_stopped_idx and row["Stopped"] > 0:
            st.error(f"Critical dropout point \u2014 {row['Dropout %']}% of all leads stop here")
    st.markdown("---")

    # ---- Section 3: What killed the lead ----
    st.subheader("SECTION 3 \u2014 WHAT KILLED THE LEAD (Dropout Theme Analysis)")
    dropout_after_1 = [l for l in all_leads if len([h for h in (l.get("history") or []) if h.get("email_sent")]) >= 1 and len([h for h in (l.get("history") or []) if h.get("reply_received")]) == 0]
    by_source = pd.Series([(l.get("source") or "unknown") for l in dropout_after_1]).value_counts().head(5)
    if not by_source.empty:
        st.bar_chart(by_source.rename("Dropouts"))
    st.caption("Top 5 lead types with highest dropout after first email")
    last_email_silent = []
    for l in all_leads:
        if any((h.get("intent_score") or 0) >= 80 for h in (l.get("history") or [])):
            continue
        n = len([h for h in (l.get("history") or []) if h.get("email_sent")])
        if n >= 1:
            last_email_silent.append(min(n, 3))
    if last_email_silent:
        last_df = pd.Series(last_email_silent).value_counts().sort_index()
        st.bar_chart(last_df.rename("Leads"))
    st.caption("Last email number when leads went silent")
    rej_reasons = [e.get("rejection_reason", "Unknown") for e in playbook.get("rejected", [])]
    if rej_reasons:
        rej_series = pd.Series(rej_reasons).value_counts()
        st.bar_chart(rej_series.rename("Count"))
    st.caption("Top rejection reasons selected by human reviewers")
    st.markdown("---")

    # ---- Section 4: What HOT leads have in common ----
    st.subheader("SECTION 4 \u2014 WHAT HOT LEADS HAVE IN COMMON (Survivor Analysis)")
    hot_leads_list = [l for l in all_leads if any((h.get("intent_score") or 0) >= 80 for h in (l.get("history") or []))]
    if hot_leads_list:
        top_sources = pd.Series([l.get("source") or "unknown" for l in hot_leads_list]).value_counts().head(3)
        st.write("Top 3 lead_source for HOT leads:", top_sources.to_dict())
        emails_to_hot = [len([h for h in (l.get("history") or []) if h.get("email_sent")]) for l in hot_leads_list]
        st.metric("Avg emails to reach HOT", round(sum(emails_to_hot) / len(emails_to_hot), 1) if emails_to_hot else 0)
        confs_first = [e.get("confidence_score", 0) for e in approved_entries if e.get("confidence_score") is not None]
        if confs_first:
            st.metric("Avg confidence of approved emails (proxy for first email quality)", round(sum(confs_first) / len(confs_first), 1))
        kw_list = []
        for e in playbook.get("approved", []):
            kw_list.extend(e.get("subject_keywords") or [])
        if kw_list:
            top_kw = pd.Series(kw_list).value_counts().head(5)
            st.write("Most common subject_keywords (approved):", top_kw.to_dict())
    st.markdown("---")

    # ---- Section 5: Objection intelligence ----
    st.subheader("SECTION 5 \u2014 OBJECTION INTELLIGENCE")
    if os.path.isfile(LEARNING_SIGNALS_PATH):
        try:
            with open(LEARNING_SIGNALS_PATH, "r", encoding="utf-8") as f:
                signals = json.load(f)
            if not isinstance(signals, list):
                signals = []
            low_performers = [s for s in signals if s.get("reason") == "no_response_14_days"]
            st.metric("Total leads tagged low_performer (no response 14 days)", len(low_performers))
            if low_performers:
                by_src = pd.Series([s.get("lead_source", "unknown") for s in low_performers]).value_counts()
                st.bar_chart(by_src.rename("Count"))
                st.caption("Low-performer leads by lead_source (which sources ghost you most?)")
                days_list = []
                for s in low_performers:
                    sd = s.get("send_date", "")
                    if sd:
                        try:
                            dt = datetime.fromisoformat(sd.replace("Z", "+00:00"))
                            if dt.tzinfo:
                                dt = dt.replace(tzinfo=None)
                            days_list.append((now - dt).days)
                        except Exception:
                            pass
                if days_list:
                    st.metric("Avg dormancy before going silent (days)", round(sum(days_list) / len(days_list), 0))
        except Exception:
            st.caption("Could not load learning_signals.json")
    else:
        st.caption("No learning_signals.json yet.")

    st.caption("Data updates each time you load the dashboard. Export this data monthly and compare trends to track system improvement.")
