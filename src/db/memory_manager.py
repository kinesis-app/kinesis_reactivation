"""
Long-Term Lead Memory — Persistent storage of each lead's full lifecycle.

Stores: all emails sent, replies, objections, tone, behavior, intent evolution.
Uses SQLite for reliability; falls back to JSON if needed.
Memory NEVER resets between sessions. AI must ALWAYS read past history before generating.
"""

from __future__ import annotations

import os
import json
import sqlite3
import hashlib
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from src.config import PROJECT_ROOT


APP_DIR = PROJECT_ROOT
MEMORY_DIR = os.path.join(APP_DIR, "memory")
os.makedirs(MEMORY_DIR, exist_ok=True)
LEADS_DB = os.path.join(MEMORY_DIR, "leads.db")
LEADS_JSON = os.path.join(MEMORY_DIR, "leads.json")


def _make_lead_id(email: str, source: str = "") -> str:
    key = f"{email.strip().lower()}|{str(source).strip().lower()}"
    return hashlib.sha256(key.encode()).hexdigest()[:24]


def _init_db(conn: sqlite3.Connection) -> None:
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS leads (
        lead_id TEXT PRIMARY KEY,
        name TEXT,
        email TEXT NOT NULL,
        source TEXT,
        created_at TEXT,
        updated_at TEXT,
        raw_row TEXT
    );
    CREATE TABLE IF NOT EXISTS lead_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        lead_id TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        stage TEXT,
        email_sent TEXT,
        reply_received TEXT,
        detected_objection TEXT,
        tone_analysis TEXT,
        intent_score INTEGER,
        certainty_score INTEGER,
        qualification_status TEXT,
        event_data TEXT,
        FOREIGN KEY (lead_id) REFERENCES leads(lead_id)
    );
    CREATE INDEX IF NOT EXISTS idx_history_lead ON lead_history(lead_id);
    """)
    # Migration: add status column if missing
    try:
        info = conn.execute("PRAGMA table_info(leads)").fetchall()
        if not any(c[1] == "status" for c in info):
            conn.execute("ALTER TABLE leads ADD COLUMN status TEXT")
            conn.commit()
    except Exception:
        pass


def _get_conn() -> Optional[sqlite3.Connection]:
    try:
        conn = sqlite3.connect(LEADS_DB)
        _init_db(conn)
        return conn
    except Exception:
        return None


def get_or_create_lead_id(row: dict) -> str:
    email = str(row.get("email", "")).strip()
    source = str(row.get("source", "")).strip()
    return _make_lead_id(email, source)


def upsert_lead(row: dict) -> str:
    lead_id = get_or_create_lead_id(row)
    conn = _get_conn()
    if conn:
        try:
            now = datetime.now(timezone.utc).isoformat()
            conn.execute(
                """INSERT INTO leads (lead_id, name, email, source, created_at, updated_at, raw_row)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(lead_id) DO UPDATE SET
                     name=excluded.name, source=excluded.source, updated_at=excluded.updated_at, raw_row=excluded.raw_row""",
                (
                    lead_id,
                    str(row.get("name", "")).strip(),
                    str(row.get("email", "")).strip(),
                    str(row.get("source", "")).strip(),
                    now,
                    now,
                    json.dumps({k: v for k, v in row.items() if isinstance(v, (str, int, float, bool, type(None)))}, default=str),
                ),
            )
            conn.commit()
        finally:
            conn.close()
    else:
        data = {}
        if os.path.exists(LEADS_JSON):
            try:
                with open(LEADS_JSON, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                pass
        if lead_id not in data:
            data[lead_id] = {"lead_id": lead_id, "name": row.get("name", ""), "email": row.get("email", ""), "source": row.get("source", ""), "created_at": datetime.now(timezone.utc).isoformat(), "history": []}
        data[lead_id].update({"name": row.get("name", ""), "email": row.get("email", ""), "source": row.get("source", ""), "updated_at": datetime.now(timezone.utc).isoformat()})
        try:
            with open(LEADS_JSON, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception:
            pass
    return lead_id


def append_history(
    lead_id: str,
    *,
    stage: Optional[str] = None,
    email_sent: Optional[str] = None,
    reply_received: Optional[str] = None,
    detected_objection: Optional[str] = None,
    tone_analysis: Optional[str] = None,
    intent_score: Optional[int] = None,
    certainty_score: Optional[int] = None,
    qualification_status: Optional[str] = None,
    event_data: Optional[dict] = None,
) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    event_json = json.dumps(event_data, default=str) if event_data else None
    conn = _get_conn()
    if conn:
        try:
            conn.execute(
                """INSERT INTO lead_history (lead_id, timestamp, stage, email_sent, reply_received, detected_objection,
                   tone_analysis, intent_score, certainty_score, qualification_status, event_data)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (lead_id, ts, stage, email_sent[:5000] if email_sent else None, reply_received[:5000] if reply_received else None, detected_objection, tone_analysis, intent_score, certainty_score, qualification_status, event_json),
            )
            conn.execute("UPDATE leads SET updated_at = ? WHERE lead_id = ?", (ts, lead_id))
            conn.commit()
        finally:
            conn.close()
    else:
        data = {}
        if os.path.exists(LEADS_JSON):
            try:
                with open(LEADS_JSON, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                pass
        if lead_id not in data:
            data[lead_id] = {"lead_id": lead_id, "name": "", "email": "", "source": "", "created_at": ts, "history": []}
        data[lead_id].setdefault("history", []).append({"timestamp": ts, "stage": stage, "email_sent": email_sent, "reply_received": reply_received, "detected_objection": detected_objection, "tone_analysis": tone_analysis, "intent_score": intent_score, "certainty_score": certainty_score, "qualification_status": qualification_status, "event_data": event_data})
        data[lead_id]["updated_at"] = ts
        try:
            with open(LEADS_JSON, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception:
            pass


def get_lead_by_email(email: str) -> Optional[Dict[str, Any]]:
    """Return the most recently updated lead with this email (any source)."""
    email = str(email or "").strip().lower()
    if not email:
        return None
    conn = _get_conn()
    if conn:
        try:
            row = conn.execute(
                "SELECT lead_id FROM leads WHERE LOWER(TRIM(email)) = ? ORDER BY updated_at DESC LIMIT 1",
                (email,),
            ).fetchone()
            if row:
                return get_lead(row[0])
        finally:
            conn.close()
    if os.path.exists(LEADS_JSON):
        try:
            with open(LEADS_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)
            for lid, lead in data.items():
                if str(lead.get("email", "")).strip().lower() == email:
                    return lead
        except Exception:
            pass
    return None


def get_lead(lead_id: str) -> Optional[Dict[str, Any]]:
    conn = _get_conn()
    if conn:
        try:
            row = conn.execute("SELECT lead_id, name, email, source, created_at, updated_at, raw_row FROM leads WHERE lead_id = ?", (lead_id,)).fetchone()
            if not row:
                return None
            raw = row[6]
            status = None
            try:
                r2 = conn.execute("SELECT status FROM leads WHERE lead_id = ?", (lead_id,)).fetchone()
                if r2:
                    status = r2[0]
            except Exception:
                pass
            lead = {"lead_id": row[0], "name": row[1], "email": row[2], "source": row[3], "created_at": row[4], "updated_at": row[5], "raw_row": json.loads(raw) if raw else {}, "status": status, "history": []}
            for h in conn.execute("SELECT timestamp, stage, email_sent, reply_received, detected_objection, tone_analysis, intent_score, certainty_score, qualification_status, event_data FROM lead_history WHERE lead_id = ? ORDER BY timestamp ASC", (lead_id,)).fetchall():
                lead["history"].append({"timestamp": h[0], "stage": h[1], "email_sent": h[2], "reply_received": h[3], "detected_objection": h[4], "tone_analysis": h[5], "intent_score": h[6], "certainty_score": h[7], "qualification_status": h[8], "event_data": json.loads(h[9]) if h[9] else {}})
            return lead
        finally:
            conn.close()
    if os.path.exists(LEADS_JSON):
        try:
            with open(LEADS_JSON, "r", encoding="utf-8") as f:
                return json.load(f).get(lead_id)
        except Exception:
            pass
    return None


def update_lead_status(lead_id: str, status: str) -> None:
    """Set status on a lead (e.g. 'low_performer')."""
    conn = _get_conn()
    if conn:
        try:
            conn.execute("UPDATE leads SET status = ?, updated_at = ? WHERE lead_id = ?", (status, datetime.now(timezone.utc).isoformat(), lead_id))
            conn.commit()
        finally:
            conn.close()
    else:
        data = {}
        if os.path.exists(LEADS_JSON):
            try:
                with open(LEADS_JSON, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                pass
        if lead_id in data:
            data[lead_id]["status"] = status
            data[lead_id]["updated_at"] = datetime.now(timezone.utc).isoformat()
            try:
                with open(LEADS_JSON, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            except Exception:
                pass


def get_all_lead_ids() -> List[str]:
    """Return all lead_id values from the database (for no-response scanning)."""
    conn = _get_conn()
    if conn:
        try:
            rows = conn.execute("SELECT lead_id FROM leads").fetchall()
            return [r[0] for r in rows]
        finally:
            conn.close()
    if os.path.exists(LEADS_JSON):
        try:
            with open(LEADS_JSON, "r", encoding="utf-8") as f:
                return list(json.load(f).keys())
        except Exception:
            pass
    return []


def get_lead_history(lead_id: str) -> List[Dict[str, Any]]:
    lead = get_lead(lead_id)
    return (lead.get("history") or []) if lead else []


def get_lead_summary_for_prompt(lead_id: str, max_history: int = 10) -> str:
    lead = get_lead(lead_id)
    if not lead:
        return "No prior history for this lead."
    hist = lead.get("history", [])[-max_history:]
    lines = [f"Lead: {lead.get('name', '')} ({lead.get('email', '')}) | Source: {lead.get('source', '')} | Created: {lead.get('created_at', '')}"]
    for h in hist:
        parts = [h.get("timestamp", "")[:16], f"stage={h.get('stage', '')}", f"intent={h.get('intent_score', '')}", f"certainty={h.get('certainty_score', '')}", f"status={h.get('qualification_status', '')}"]
        if h.get("detected_objection"):
            parts.append(f"objection={h['detected_objection']}")
        if h.get("tone_analysis"):
            parts.append(f"tone={h['tone_analysis']}")
        lines.append(" | ".join(str(p) for p in parts))
    return "\n".join(lines) if lines else "No prior history."
