"""
Kinesis Reactivation — Clay Enrichment integration.
"""

from __future__ import annotations

import os
from typing import Optional

import requests


def enrich_with_clay(email: str, name: str, source: str = "", row: Optional[dict] = None) -> dict:
    """Enrich lead via Clay webhook, CSV row, or simulation. CSV columns (recent_news, role, company, funding) override mock."""
    row = row or {}
    clay_webhook = os.getenv("CLAY_WEBHOOK_URL", "")
    if clay_webhook:
        try:
            resp = requests.post(clay_webhook, json={"email": email, "name": name, "source": source}, timeout=10)
            if resp.status_code == 200:
                out = resp.json()
                return {**out, **{k: v for k, v in row.items() if k in ("recent_news", "role", "company", "funding", "website") and v}}  # CSV overrides Clay
        except Exception:
            pass
    # Build base: use CSV enrichment if present, else mock
    import hashlib
    key = f"{email}|{name}|{source}".encode()
    idx = int(hashlib.sha256(key).hexdigest(), 16) % 100
    roles = ["VP Marketing", "Head of R&D", "Procurement Director", "Plant Manager", "Quality Director", "Supply Chain Lead"]
    news_items = ["Raised Series B", "New facility opening Q2", "EMA approval for lead product", "Partnership with CDMO", "ISO certification renewal", "Recent acquisition in EU"]
    funding_options = ["$10M", "\u20ac8M", "$15M Series A", "bootstrapped", "pre-revenue"]
    base = {
        "role": str(row.get("role", "")).strip() or roles[idx % len(roles)],
        "company": str(row.get("company", "")).strip() or ((name.split()[0] if name else "Company") + " " + ["Inc", "Ltd", "Pharma", "Labs", "Solutions"][(idx // 20) % 5]),
        "recent_news": str(row.get("recent_news", "")).strip() or news_items[idx % len(news_items)],
        "funding": str(row.get("funding", "")).strip() or funding_options[idx % len(funding_options)],
        "website": str(row.get("website", "")).strip() or (f"https://{(name or 'company').lower().replace(' ', '')}.com" if name else ""),
    }
    return base
