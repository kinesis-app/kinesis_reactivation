"""
Kinesis Reactivation — Shared configuration, paths, constants, and helpers.
"""

from __future__ import annotations

import os
import pathlib
import json
import time
from datetime import datetime, timezone

import streamlit as st
from dotenv import load_dotenv

# ── Path resolution ──────────────────────────────────────────────────────────
PROJECT_ROOT = str(pathlib.Path(__file__).resolve().parent.parent)

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

LOG_PATH = pathlib.Path(PROJECT_ROOT) / "debug.log"

# Paths
APP_DIR = PROJECT_ROOT
PLAYBOOK_PATH = os.path.join(APP_DIR, "playbook.json")
LEARNING_SIGNALS_PATH = os.path.join(APP_DIR, "learning_signals.json")
CLIENT_CONFIG_PATH = os.path.join(APP_DIR, "client_config.json")
VAULT_DIR = os.path.join(APP_DIR, "vault_data")
SEQUENCES_DIR = os.path.join(APP_DIR, "sequences_data")
os.makedirs(VAULT_DIR, exist_ok=True)
os.makedirs(SEQUENCES_DIR, exist_ok=True)


# ── Debug logger ─────────────────────────────────────────────────────────────
# #region agent log
def _dbg(m: str, d: dict, h: str = "A"):
    try:
        p = {"id": f"log_{id(d)}", "timestamp": int(time.time() * 1000), "location": "kinesis_reactivation", "message": m, "data": d, "hypothesisId": h}
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(p, default=str) + "\n")
    except Exception:
        pass
# #endregion

_dbg("load_dotenv done", {"cwd": os.getcwd(), "mistral_set": os.getenv("MISTRAL_API_KEY") is not None, "mistral_len": len(os.getenv("MISTRAL_API_KEY") or ""), "openai_set": os.getenv("OPENAI_API_KEY") is not None}, "H1")
print(f"MISTRAL_API_KEY loaded: {os.getenv('MISTRAL_API_KEY') is not None}")


# ── Client config persistence ────────────────────────────────────────────────
def load_client_config() -> dict:
    """Load client compliance config from JSON. Returns dict with custom_compliance_rules, industry_regulatory_context, last_updated."""
    default = {"custom_compliance_rules": "", "industry_regulatory_context": "", "last_updated": ""}
    if not os.path.isfile(CLIENT_CONFIG_PATH):
        return default
    try:
        with open(CLIENT_CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {**default, **{k: data.get(k, default[k]) for k in default}}
    except Exception:
        return default


def save_client_config() -> bool:
    """Save session_state compliance fields to client_config.json with last_updated. Returns True on success."""
    try:
        data = {
            "custom_compliance_rules": (st.session_state.get("custom_compliance_rules") or "").strip(),
            "industry_regulatory_context": (st.session_state.get("industry_regulatory_context") or "").strip(),
            "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        with open(CLIENT_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return True
    except Exception:
        return False


# ── Constants ─────────────────────────────────────────────────────────────────
LEGAL_FOOTER = """

---
[Your Company Name]
[Your Physical Business Address \u2013 required for compliance]
[City, State, ZIP, Country]

You received this email because we believe you have an interest in our services. To unsubscribe from future emails, reply with "Unsubscribe" in the subject line or click here: [Unsubscribe Link]. We will remove you within 10 business days."""

GOLDEN_RULES = """
1. NO FLUFF: Start with value, not "I hope you're doing well."
2. CLEAR CTA: One specific next step.
3. LEGAL: No false claims; include disclaimer.
4. BRAND: Professional, helpful, never salesy or robotic."""

# Reactive state machine: HOT handoff threshold and nudge cadence
HOT_SCORE_THRESHOLD = 80
NUDGE_DAYS = 7

# Scenario branching: source -> email count and cadence (days between)
SCENARIO_CONFIG = {
    "brochure download": {"emails": 3, "cadence": [0, 3, 7], "style": "light education"},
    "form fill": {"emails": 4, "cadence": [0, 2, 5, 10], "style": "medium targeted"},
    "webinar sign-up": {"emails": 2, "cadence": [0, 1], "style": "high-intent qualification"},
    "direct contact": {"emails": 2, "cadence": [0, 2], "style": "fast close"},
    "default": {"emails": 3, "cadence": [0, 3, 7], "style": "standard"},
    "reactivation": {
        "emails": 8,  # longer sequence
        "cadence": [0, 14, 35, 60, 90, 120, 180, 240],  # very respectful spacing
        "style": "warm re-engagement referencing past relationship",
    },
}

REJECT_REASON_OPTIONS = [
    "Too generic / not personalised enough",
    "Wrong tone for this lead",
    "Factual error or unverified claim",
    "Too long or too short",
    "Legal or compliance concern",
    "Other (describe below)",
]
