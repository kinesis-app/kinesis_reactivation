"""
Kinesis Reactivation — SendGrid email integration and hot lead alerts.
"""

from __future__ import annotations

import os

import requests

from src.db import get_lead


def strip_draft_tag(body: str) -> str:
    """Remove top-line DRAFT \u2014 DO NOT SEND so approved sends are clean."""
    if not body:
        return body
    first = body.split("\n")[0].strip()
    if first.upper().startswith("DRAFT") and "DO NOT SEND" in first.upper():
        return "\n".join(body.split("\n")[1:]).lstrip()
    return body


def send_hot_lead_alert(lead_data: dict) -> None:
    """
    Notify sales when a lead is HOT (HANDOFF_TO_SALES).
    Uses SLACK_WEBHOOK_URL if set, else SALES_ALERT_EMAIL via SendGrid.
    """
    lead_id = lead_data.get("lead_id", "")
    name = lead_data.get("name", "") or "Unknown"
    company = lead_data.get("company", "") or "Unknown"
    source = lead_data.get("source", "") or "Unknown"
    intent_score = lead_data.get("intent_score", 0)
    lead = get_lead(lead_id) if lead_id else None
    history = (lead.get("history", []) or []) if lead else []
    emails_sent = [h.get("email_sent", "") for h in history if h.get("email_sent")]
    replies = [h.get("reply_received", "") for h in history if h.get("reply_received")]
    last_email_snippet = (emails_sent[-1][:200] if emails_sent else "\u2014") or "\u2014"

    slack_url = os.environ.get("SLACK_WEBHOOK_URL", "").strip()
    if slack_url:
        try:
            payload = {
                "text": "HOT LEAD ALERT",
                "blocks": [
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": "*Lead:* {name}\n*Company:* {company}\n*Source:* {source}\n\n*Intent Score:* {score}%\n*Last email sent:* {snippet}".format(
                                name=name,
                                company=company,
                                source=source,
                                score=intent_score,
                                snippet=last_email_snippet,
                            ),
                        },
                    }
                ],
            }
            requests.post(slack_url, json=payload, timeout=10)
        except Exception:
            pass
        return

    alert_email = os.environ.get("SALES_ALERT_EMAIL", "").strip()
    if alert_email:
        try:
            body_lines = [
                f"Lead: {name}",
                f"Company: {company}",
                f"Source: {source}",
                f"Intent Score: {intent_score}%",
                "",
                "--- Emails sent to this lead ---",
            ]
            for i, body in enumerate(emails_sent, 1):
                body_lines.append(f"{i}. {body[:2000]}{'...' if len(body) > 2000 else ''}")
            body_lines.append("")
            body_lines.append("--- Replies received ---")
            for i, reply in enumerate(replies, 1):
                body_lines.append(f"{i}. {reply[:2000]}{'...' if len(reply) > 2000 else ''}")
            body = "\n".join(body_lines)
            subject = "HOT Lead Alert: {} \u2014 {}".format(name, company)
            from_email = os.environ.get("SENDGRID_FROM_EMAIL", "noreply@example.com")
            from_name = os.environ.get("SENDGRID_FROM_NAME", "Kinesis Reactivation")
            send_email_via_sendgrid(alert_email, "Sales", subject, body, from_email, from_name)
        except Exception:
            pass


def send_email_via_sendgrid(to_email: str, to_name: str, subject: str, body: str, from_email: str, from_name: str) -> bool:
    """Send email via SendGrid API. Strips DRAFT tag from body when sending."""
    api_key = os.getenv("SENDGRID_API_KEY", "")
    if not api_key:
        return False
    try:
        url = "https://api.sendgrid.com/v3/mail/send"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        body_clean = strip_draft_tag(body)
        data = {
            "personalizations": [{"to": [{"email": to_email, "name": to_name}]}],
            "from": {"email": from_email, "name": from_name},
            "subject": subject,
            "content": [{"type": "text/html", "value": body_clean.replace("\n", "<br>")}],
        }
        resp = requests.post(url, json=data, headers=headers, timeout=10)
        return resp.status_code in [200, 202]
    except Exception:
        return False
