"""Kinesis Reactivation — Streamlit entrypoint."""

import os

import streamlit as st

from src.config import load_client_config
from src.tools.playbook import check_no_response_leads
from src.ui.sidebar import render_sidebar
from src.ui.dashboard import render_dashboard
from src.ui.statistics import render_statistics

st.set_page_config(page_title="Kinesis Reactivation — Old CRM Leads", layout="wide", initial_sidebar_state="expanded")

# Load client compliance config into session_state on startup (persists across server restarts)
if "custom_compliance_rules" not in st.session_state:
    _cfg = load_client_config()
    st.session_state["custom_compliance_rules"] = _cfg.get("custom_compliance_rules", "")
    st.session_state["industry_regulatory_context"] = _cfg.get("industry_regulatory_context", "")
    st.session_state["client_config_last_updated"] = _cfg.get("last_updated", "")

if not os.getenv("MISTRAL_API_KEY"):
    st.error("Missing MISTRAL_API_KEY in .env")

# Sidebar
legal_footer, from_email, from_name, csv_file = render_sidebar()

# Main
st.title("\u2709\ufe0f Kinesis Reactivation \u2014 Old CRM Leads")
st.caption("Automate CEO-style nurturing of warm inbound leads. Sequences, replies, qualification scoring, funnel analysis.")
st.success("LangGraph ready")

if "no_response_checked" not in st.session_state:
    check_no_response_leads()
    st.session_state["no_response_checked"] = True

tab_dashboard, tab_statistics = st.tabs(["Dashboard", "Statistics"])

with tab_dashboard:
    render_dashboard(csv_file, legal_footer, from_email, from_name)

with tab_statistics:
    render_statistics()

st.caption("Kinesis Reactivation: Old CRM leads, sequences, replies, qualification, funnel analysis. Admin mode: full access. Client mode: dashboard only.")
