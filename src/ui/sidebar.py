"""
Kinesis Reactivation — Sidebar UI component.
"""

from __future__ import annotations

import os

import streamlit as st

from src.config import LEGAL_FOOTER, load_client_config, save_client_config
from src.llm.providers import get_embeddings
from src.tools.vault import file_to_docs, chunk_docs, add_docs_to_vault
from src.api.webhook import start_flask_webhook


def render_sidebar():
    """Render the sidebar and return (legal_footer, from_email, from_name, csv_file)."""
    # Service Mode: Admin vs Client
    service_mode = st.sidebar.selectbox("Service Mode", ["Admin", "Client"], key="service_mode")

    with st.sidebar:
        st.header("\U0001f3e2 Settings")
        biz_name = st.text_input("Your Company Name", "My Agency", key="biz_name")
        biz_address = st.text_area("Your Business Address", "123 Business St", key="biz_address")
        from_email = st.text_input("From Email (SendGrid)", os.getenv("SENDGRID_FROM_EMAIL", ""), key="from_email")
        from_name = st.text_input("From Name", biz_name, key="from_name")
        legal_footer = LEGAL_FOOTER.replace("[Your Company Name]", biz_name).replace("[Your Physical Business Address]", biz_address)

        if service_mode == "Admin":
            st.divider()
            st.subheader("\U0001f4da Knowledge Vault")
            vault_files = st.file_uploader("Upload to Vault", type=["pdf", "txt"], key="vault_upload", accept_multiple_files=True)
            if vault_files and st.button("Add to Vault", key="add_vault"):
                emb = get_embeddings()
                if emb is None:
                    st.error("Missing MISTRAL_API_KEY in .env")
                else:
                    all_docs = []
                    for f in vault_files:
                        all_docs.extend(file_to_docs(f, f.name))
                    if all_docs:
                        chunks = chunk_docs(all_docs)
                        if add_docs_to_vault(chunks):
                            st.success(f"Added {len(chunks)} chunks.")

            st.divider()
            st.subheader("\U0001f3a4 CEO Interview Kit")
            ceo_kit_files = st.file_uploader("Upload CEO Interview Kit", type=["pdf", "txt"], key="ceo_kit", accept_multiple_files=True)
            if ceo_kit_files and st.button("Add CEO Kit", key="add_ceo"):
                all_docs = []
                for f in ceo_kit_files:
                    all_docs.extend(file_to_docs(f, "ceo_kit_" + (f.name or "file")))
                if all_docs:
                    chunks = chunk_docs(all_docs)
                    if add_docs_to_vault(chunks):
                        st.success(f"Added {len(chunks)} CEO kit chunks.")

            st.divider()
            st.subheader("\U0001f517 Clay Enrichment")
            clay_webhook_url = st.text_input("Clay Webhook URL", os.getenv("CLAY_WEBHOOK_URL", ""), key="clay_webhook")
            if clay_webhook_url:
                os.environ["CLAY_WEBHOOK_URL"] = clay_webhook_url

            st.divider()
            st.subheader("\U0001f4e1 Webhook Server")
            if st.button("Start Flask Webhook (port 5000)", key="start_webhook"):
                start_flask_webhook()
                st.success("Webhook running. Use ngrok: ngrok http 5000")

            st.divider()
            st.subheader("\u2696\ufe0f Compliance Rules")
            st.caption("Enter the regulatory and legal rules specific to this client. These are injected into the AI compliance reviewer and applied to every email before it can be approved.")
            st.text_area(
                "Client-specific compliance rules",
                value=st.session_state.get("custom_compliance_rules", ""),
                height=200,
                key="custom_compliance_rules",
                placeholder="Examples: Do not reference clinical outcomes without citing a specific published study. Do not name competitor products by name. Do not contact individuals titled Healthcare Professional in Germany without documented prior consent. Do not make claims about regulatory approval for indications not yet approved by EMA or FDA.",
            )
            st.subheader("\U0001f4dc Industry Regulatory Context")
            st.caption("Upload key excerpts from industry regulations, association codes of conduct, or regulatory guidelines relevant to this client. The AI uses this as background knowledge when reviewing drafts.")
            st.text_area(
                "Industry regulatory context",
                value=st.session_state.get("industry_regulatory_context", ""),
                height=300,
                key="industry_regulatory_context",
            )
            if st.session_state.get("client_config_last_updated"):
                st.caption(f"Last updated: {st.session_state['client_config_last_updated']}")
            if st.button("Save compliance settings", key="save_compliance"):
                if save_client_config():
                    st.session_state["client_config_last_updated"] = load_client_config().get("last_updated", "")
                    st.success("Compliance settings saved.")

        st.divider()
        csv_file = st.file_uploader("Upload Inbound CSV", type=["csv"], key="inbound_csv")
        st.caption("Columns: name, email, source (use 'reactivation'), last_contact_date, past_project, dormant_reason, previous_revenue, behavior, initial_objection")

    return legal_footer, from_email, from_name, csv_file
