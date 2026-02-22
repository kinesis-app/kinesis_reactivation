"""
Kinesis Reactivation — Flask Webhook (run in background thread).
"""

from __future__ import annotations

import os
import re
from threading import Thread

import pandas as pd

from src.config import SEQUENCES_DIR
from src.db import upsert_lead


def start_flask_webhook():
    """Start Flask webhook server in background thread."""
    from flask import Flask, request, jsonify
    app = Flask(__name__)

    @app.route("/webhook", methods=["POST"])
    def webhook():
        data = request.json
        if not data or not isinstance(data, dict):
            return jsonify({"error": "Invalid JSON payload"}), 400
        email = (data.get("email") or "").strip()
        if not email or not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
            return jsonify({"error": "Valid email is required"}), 400
        name = (data.get("name") or "").strip()[:200]
        source = (data.get("source") or "webhook").strip()[:100]
        row = {"email": email, "name": name, "source": source, "behavior": (data.get("behavior") or "").strip()[:500], "initial_objection": (data.get("objection") or "").strip()[:500]}
        upsert_lead(row)
        webhook_csv = os.path.join(SEQUENCES_DIR, "webhook_leads.csv")
        if os.path.exists(webhook_csv):
            df = pd.read_csv(webhook_csv)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row])
        df.to_csv(webhook_csv, index=False)
        return jsonify({"status": "received"})

    def run():
        app.run(host="0.0.0.0", port=5000, debug=False)

    thread = Thread(target=run, daemon=True)
    thread.start()
    return thread
