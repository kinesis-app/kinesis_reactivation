"""
Kinesis Reactivation — RAG + LangGraph (Dormant CRM Lead Re-engagement)
- Knowledge Vault: FAISS + Mistral AI embeddings; PDF/TXT vectorized; cite sources.
- CEO Digital Twin: Mimic tone/rhythm from vault; No-Go list and stylistic rules.
- Legal Bouncer: Pharma/FDA/GDPR compliance filter; block false claims; opt-out P.S.
- Dashboard: Review/approve with confidence scores; Equity Meter; funnel graphs.
- Learning Loop: playbook.json updates from approvals/rejections/non-responses; failure insights.
- Inbound Features: CSV input (name, email, source, behavior, objection); Clay enrichment; scenario branching; nurturing sequences (3-10 emails); reply handling; qualification scoring; funnel graphs; Flask webhook; SendGrid sending/tracking; admin/client modes.
"""

from __future__ import annotations

import os
import pathlib
import re
import json
from io import BytesIO
from datetime import datetime, timedelta
from typing import TypedDict, Optional, Dict, List
from threading import Thread
import time

import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import requests
import langchain_community
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

from memory_manager import (
    get_or_create_lead_id,
    upsert_lead,
    append_history,
    get_lead,
    get_lead_by_email,
    get_lead_history,
    get_lead_summary_for_prompt,
    get_all_lead_ids,
    update_lead_status,
)
from learning_loop import get_learned_patterns, run_learning_loop
from scoring_engine import compute_intent_and_certainty, qualification_status as scoring_qualification_status
from funnel_analytics import compute_deep_funnel_analytics as compute_deep_funnel

# Pipeline integration: memory_manager (lead upsert + history), scoring_engine (intent/certainty),
# learning_loop (post-lifecycle), funnel_analytics (death analytics). CSV + webhook ingestion unchanged.

load_dotenv()

LOG_PATH = pathlib.Path(os.path.dirname(os.path.abspath(__file__))) / "debug.log"

# #region agent log
def _dbg(m: str, d: dict, h: str = "A"):
    try:
        import json
        p = {"id": f"log_{id(d)}", "timestamp": int(time.time() * 1000), "location": "kinesis_inbound", "message": m, "data": d, "hypothesisId": h}
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(p, default=str) + "\n")
    except Exception:
        pass
# #endregion
_dbg("load_dotenv done", {"cwd": os.getcwd(), "mistral_set": os.getenv("MISTRAL_API_KEY") is not None, "mistral_len": len(os.getenv("MISTRAL_API_KEY") or ""), "openai_set": os.getenv("OPENAI_API_KEY") is not None}, "H1")
print(f"MISTRAL_API_KEY loaded: {os.getenv('MISTRAL_API_KEY') is not None}")

# Paths
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PLAYBOOK_PATH = os.path.join(APP_DIR, "playbook.json")
LEARNING_SIGNALS_PATH = os.path.join(APP_DIR, "learning_signals.json")
VAULT_DIR = os.path.join(APP_DIR, "vault_data")
SEQUENCES_DIR = os.path.join(APP_DIR, "sequences_data")
os.makedirs(VAULT_DIR, exist_ok=True)
os.makedirs(SEQUENCES_DIR, exist_ok=True)

# Constants
LEGAL_FOOTER = """

---
[Your Company Name]
[Your Physical Business Address – required for compliance]
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

# Scenario branching: source → email count and cadence (days between)
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

st.set_page_config(page_title="Kinesis Reactivation — Old CRM Leads", layout="wide", initial_sidebar_state="expanded")

if not os.getenv("MISTRAL_API_KEY"):
    st.error("Missing MISTRAL_API_KEY in .env")

# ---------- Knowledge Vault (RAG) — Mistral AI only ----------

def get_embeddings():
    api_key = os.getenv("MISTRAL_API_KEY") or ""
    if not api_key or api_key == "your-mistral-key-here":
        return None
    return MistralAIEmbeddings(model="mistral-embed", mistral_api_key=api_key)


def get_llm():
    print("Trying to load Mistral LLM...")
    mistral_key = os.getenv("MISTRAL_API_KEY") or ""
    if mistral_key and mistral_key != "your-mistral-key-here":
        try:
            llm = ChatMistralAI(model="mistral-large-latest", temperature=0.3, mistral_api_key=mistral_key)
            print("LLM loaded: Mistral")
            return llm
        except Exception as e:
            print(f"Mistral LLM failed: {e}")
    else:
        print("LLM failed — key missing or import error")
    openai_key = os.getenv("OPENAI_API_KEY") or ""
    if openai_key and openai_key != "your-actual-key-here" and ChatOpenAI is not None:
        try:
            llm = ChatOpenAI(model="gpt-4o", temperature=0.3, openai_api_key=openai_key)
            print("LLM loaded: OpenAI fallback")
            return llm
        except Exception as e:
            print(f"OpenAI fallback failed: {e}")
    return None


def get_llm_evaluator():
    """Evaluator node: temperature=0.0 for determinism."""
    mistral_key = os.getenv("MISTRAL_API_KEY") or ""
    if mistral_key and mistral_key != "your-mistral-key-here":
        try:
            return ChatMistralAI(model="mistral-large-latest", temperature=0.0, max_tokens=400, mistral_api_key=mistral_key)
        except Exception:
            pass
    if ChatOpenAI is not None:
        openai_key = os.getenv("OPENAI_API_KEY") or ""
        if openai_key and openai_key != "your-actual-key-here":
            try:
                return ChatOpenAI(model="gpt-4o", temperature=0.0, max_tokens=400, openai_api_key=openai_key)
            except Exception:
                pass
    return None


def get_llm_writer():
    """Writer node: Mistral mistral-large-latest, temperature=0.18, max_tokens=400."""
    mistral_key = os.getenv("MISTRAL_API_KEY") or ""
    # #region agent log
    _dbg("get_llm_writer entry", {"mistral_has_key": bool(mistral_key), "mistral_is_placeholder": mistral_key == "your-mistral-key-here"}, "H2")
    # #endregion
    if mistral_key and mistral_key != "your-mistral-key-here":
        try:
            llm = ChatMistralAI(
                model="mistral-large-latest",
                temperature=0.18,
                max_tokens=400,
                mistral_api_key=mistral_key,
            )
            # #region agent log
            _dbg("get_llm_writer Mistral ok", {"returned": "ChatMistralAI"}, "H2")
            # #endregion
            return llm
        except Exception as e:
            # #region agent log
            _dbg("get_llm_writer Mistral failed", {"err": str(e)[:200]}, "H2")
            # #endregion
            pass
    if ChatOpenAI is not None:
        openai_key = os.getenv("OPENAI_API_KEY") or ""
        if openai_key and openai_key != "your-actual-key-here":
            try:
                llm = ChatOpenAI(model="gpt-4o", temperature=0.18, max_tokens=400, openai_api_key=openai_key)
                # #region agent log
                _dbg("get_llm_writer OpenAI fallback ok", {}, "H2")
                # #endregion
                return llm
            except Exception as e:
                # #region agent log
                _dbg("get_llm_writer OpenAI failed", {"err": str(e)[:200]}, "H2")
                # #endregion
                pass
    # #region agent log
    _dbg("get_llm_writer returning None", {}, "H2")
    # #endregion
    return None


def load_vault():
    index_path = os.path.join(VAULT_DIR, "faiss_index")
    emb = get_embeddings()
    if emb is None:
        return None
    if os.path.exists(index_path) and os.path.isdir(index_path):
        try:
            return FAISS.load_local(index_path, emb, allow_dangerous_deserialization=True)
        except Exception:
            pass
    return None


def add_docs_to_vault(docs: list) -> bool:
    emb = get_embeddings()
    if not docs or emb is None:
        return False
    index_path = os.path.join(VAULT_DIR, "faiss_index")
    try:
        if os.path.exists(index_path) and os.path.isdir(index_path):
            v = FAISS.load_local(index_path, emb, allow_dangerous_deserialization=True)
            v.add_documents(docs)
        else:
            v = FAISS.from_documents(docs, emb)
        v.save_local(index_path)
        return True
    except Exception:
        return False


def file_to_docs(uploaded_file, source_name: str) -> list:
    from pypdf import PdfReader
    if uploaded_file is None:
        return []
    raw = uploaded_file.read()
    name = (uploaded_file.name or "").lower()
    docs = []
    try:
        if name.endswith(".pdf"):
            reader = PdfReader(BytesIO(raw))
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    docs.append(Document(page_content=text.strip(), metadata={"source": source_name, "page": i + 1}))
        elif name.endswith(".txt"):
            text = raw.decode("utf-8", errors="replace")
            docs.append(Document(page_content=text, metadata={"source": source_name}))
        return docs
    except Exception:
        return []


def chunk_docs(docs: list, chunk_size: int = 800, overlap: int = 100) -> list:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap, separators=["\n\n", "\n", " "])
    out = []
    for d in docs:
        out.extend(splitter.split_documents([d]))
    return out


def vault_retrieve(query: str, k: int = 8, filter_source: Optional[str] = None) -> List[dict]:
    """Return list of {"content", "source", "page", "score"} for schema alignment."""
    v = load_vault()
    if v is None:
        return []
    try:
        results = v.similarity_search_with_score(query, k=k)
        out = []
        for doc, score in results:
            if filter_source and doc.metadata.get("source") != filter_source:
                continue
            src = doc.metadata.get("source", "vault")
            page = doc.metadata.get("page", 0)
            if page == "":
                page = 0
            try:
                page = int(page)
            except (TypeError, ValueError):
                page = 0
            out.append({
                "content": doc.page_content,
                "source": src,
                "page": page,
                "score": float(score),
            })
        return out
    except Exception:
        return []


def format_citations(chunks: List[dict]) -> str:
    """Format chunk list for prompt (chunks have content, source, page, score)."""
    lines = []
    for c in chunks:
        content = c.get("content", "")
        src = c.get("source", "vault")
        page = c.get("page", 0)
        cite = f"[{src}" + (f", page {page}" if page else "") + "]"
        quote = (content[:500] + "…") if len(content) > 500 else content
        lines.append(f"As per {cite}: {quote}")
    return "\n\n".join(lines) if lines else ""


def citations_to_schema(chunks: List[dict]) -> List[dict]:
    """Writer output schema: list of {source, page, score}."""
    return [{"source": c.get("source", ""), "page": c.get("page", 0), "score": c.get("score", 0.0)} for c in chunks]


# ---------- Playbook (JSON) — Enhanced for failures ----------

def load_playbook() -> dict:
    if not os.path.isfile(PLAYBOOK_PATH):
        return {"approved": [], "rejected": [], "non_responses": [], "failures": [], "training_pairs": []}
    try:
        with open(PLAYBOOK_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "approved": data.get("approved", []),
            "rejected": data.get("rejected", []),
            "non_responses": data.get("non_responses", []),
            "failures": data.get("failures", []),
            "training_pairs": data.get("training_pairs", []),
        }
    except Exception:
        return {"approved": [], "rejected": [], "non_responses": [], "failures": [], "training_pairs": []}


def save_playbook(data: dict) -> None:
    data.setdefault("training_pairs", [])
    try:
        with open(PLAYBOOK_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        if "st" in dir():
            st.sidebar.error(f"Could not save playbook: {e}")


def _subject_keywords(subject_line: str, max_words: int = 5) -> list:
    """Extract first max_words significant words (length > 4) from subject line."""
    words = [w.strip().strip(".,!?") for w in (subject_line or "").split() if len(w.strip().strip(".,!?")) > 4]
    return words[:max_words]


def add_approved_to_playbook(
    email: str,
    original_prompt: str,
    company_name: str = "",
    *,
    lead_source: str = "",
    confidence_score: int = 0,
    subject_line: str = "",
) -> None:
    data = load_playbook()
    subject_keywords = _subject_keywords(subject_line)
    data["approved"].append({
        "email": email,
        "email_body": email,
        "original_prompt": original_prompt,
        "company_name": company_name,
        "lead_source": lead_source,
        "confidence_score": confidence_score,
        "subject_keywords": subject_keywords,
        "timestamp": datetime.now().isoformat(),
    })
    save_playbook(data)


REJECT_REASON_OPTIONS = [
    "Too generic / not personalised enough",
    "Wrong tone for this lead",
    "Factual error or unverified claim",
    "Too long or too short",
    "Legal or compliance concern",
    "Other (describe below)",
]


def add_rejected_to_playbook(rejection_reason: str, draft_snippet: str = "") -> None:
    data = load_playbook()
    data["rejected"].append({
        "rejection_reason": rejection_reason,
        "draft_snippet": (draft_snippet or "")[:500],
        "timestamp": datetime.now().isoformat(),
    })
    save_playbook(data)


def add_training_pair_to_playbook(original_draft: str, edited_draft: str, lead_id: str) -> None:
    """Save (original, edited) pair for training when reviewer edits then approves."""
    data = load_playbook()
    data.setdefault("training_pairs", []).append({
        "original_draft": original_draft,
        "edited_draft": edited_draft,
        "lead_id": lead_id,
        "timestamp": datetime.now().isoformat(),
    })
    save_playbook(data)


def check_no_response_leads() -> None:
    """Tag leads with no reply 14+ days after send as low_performer; append to learning_signals.json."""
    try:
        cutoff = datetime.utcnow() - timedelta(days=14)
        signals = []
        if os.path.isfile(LEARNING_SIGNALS_PATH):
            try:
                with open(LEARNING_SIGNALS_PATH, "r", encoding="utf-8") as f:
                    signals = json.load(f)
                if not isinstance(signals, list):
                    signals = []
            except Exception:
                signals = []
        for lead_id in get_all_lead_ids():
            lead = get_lead(lead_id)
            if not lead:
                continue
            history = lead.get("history") or []
            if any(h.get("reply_received") for h in history):
                continue
            last_sent_ts = None
            last_sent_body = None
            for h in history:
                if h.get("email_sent"):
                    ts_str = h.get("timestamp", "")
                    if ts_str:
                        try:
                            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                            if ts.tzinfo:
                                ts = ts.replace(tzinfo=None)
                            if last_sent_ts is None or ts > last_sent_ts:
                                last_sent_ts = ts
                                last_sent_body = h.get("email_sent", "")
                        except Exception:
                            pass
            if last_sent_ts is None or last_sent_body is None or last_sent_ts >= cutoff:
                continue
            existing_ids = {s.get("lead_id") for s in signals if s.get("reason") == "no_response_14_days"}
            if lead_id in existing_ids:
                continue
            update_lead_status(lead_id, "low_performer")
            record = {
                "lead_id": lead_id,
                "email_body": last_sent_body[:2000] if last_sent_body else "",
                "lead_source": lead.get("source", ""),
                "send_date": last_sent_ts.isoformat() if last_sent_ts else "",
                "reason": "no_response_14_days",
            }
            signals.append(record)
        with open(LEARNING_SIGNALS_PATH, "w", encoding="utf-8") as f:
            json.dump(signals, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


def add_non_response_to_playbook(email_num: int, lead_id: str, reason: str = "") -> None:
    data = load_playbook()
    data["non_responses"].append({"email_num": email_num, "lead_id": lead_id, "reason": reason or "No response", "timestamp": datetime.now().isoformat()})
    save_playbook(data)


def add_failure_insight(insight: str, stage: str, lead_id: str = "") -> None:
    data = load_playbook()
    data["failures"].append({"insight": insight, "stage": stage, "lead_id": lead_id, "timestamp": datetime.now().isoformat()})
    save_playbook(data)


# ---------- Draft Evaluator (eval_output schema) ----------

BANNED_PHRASES = [
    "100%", "guaranteed", "unbreakable", "cure", "cures", "fda approved",
    "clinically proven", "medical advice", "diagnose", "treatment for",
]
NUMERIC_CLAIM_PATTERN = re.compile(r"\d+%\s+(improvement|increase|reduction|success|cure|guarantee)|studies?\s+show|\d+\s+times\s+(better|faster)")


def _deterministic_checks(draft: str) -> tuple[int, list]:
    """Factual safety 0-20. Returns (points, risk_flags_list)."""
    t = (draft or "").lower()
    flags = []
    points = 20
    for phrase in BANNED_PHRASES:
        if phrase in t:
            flags.append("banned_phrase")
            points = max(0, points - 15)
            break
    if NUMERIC_CLAIM_PATTERN.search(t):
        flags.append("unsourced_numeric_claim")
        points = max(0, points - 10)
    efficacy_words = ["cure", "heal", "eliminate", "eradicate", "miracle"]
    if any(w in t for w in efficacy_words):
        flags.append("POTENTIAL_LEGAL_RISK")
        points = max(0, points - 15)
    return points, flags


def _similarity_to_vault_points(similarity_avg: float) -> int:
    """Map avg score (0-1, higher=better) to 0-40. FAISS may return distance (lower=better); we handle both."""
    if similarity_avg is None:
        return 20
    s = float(similarity_avg)
    if s <= 1:
        sim = s
    else:
        sim = max(0, 1 - s / 2)
    return min(40, int(40 * sim))


def evaluate_draft_confidence(
    draft: str,
    citations: str,
    lead_context: str,
    llm,
    citations_list: Optional[List[dict]] = None,
) -> dict:
    """
    Draft Evaluator: similarity_avg from citations, deterministic checks, LLM only for tone + explanation.
    Returns eval_output: CONFIDENCE_SCORE, VAULT_ALIGNMENT, RISK_FLAGS, EXPLANATION, prompt_version, needs_review.
    """
    prompt_version = "kinesis_cursor_master:v1.0"
    if not draft or "Generation failed" in draft or "LLM not loaded" in draft:
        return {
            "score": 0,
            "explanation": "Could not evaluate (invalid draft).",
            "risk_flags": "unknown",
            "vault_alignment": "low",
            "prompt_version": prompt_version,
            "needs_review": True,
        }

    risk_flags_list = []
    similarity_avg = None
    if citations_list and isinstance(citations_list, list):
        raw_scores = [c.get("score") for c in citations_list if c.get("score") is not None]
        if raw_scores:
            sims = [1 / (1 + abs(s)) for s in raw_scores]
            similarity_avg = sum(sims) / len(sims)

    vault_grounding = _similarity_to_vault_points(similarity_avg)
    factual_points, det_flags = _deterministic_checks(draft)
    risk_flags_list.extend(det_flags)

    personalization = 10
    if lead_context:
        ctx_lower = lead_context.lower()
        if "name:" in ctx_lower or "company:" in ctx_lower:
            name_part = re.search(r"name:\s*(\w+)", ctx_lower)
            if name_part and name_part.group(1) != "there" and name_part.group(1) in draft.lower():
                personalization = 20
            elif "company" in ctx_lower or "role" in ctx_lower:
                personalization = 15

    tone_points = 0
    explanation = ""
    llm_eval = get_llm_evaluator() or llm
    if llm_eval:
        sys_prompt = """Judge ONLY tone match to CEO/vault style (professional, helpful, not salesy). Reply:
TONE_SCORE: <0-20>
EXPLANATION: <1-2 sentences>"""
        try:
            resp = llm_eval.invoke([
                SystemMessage(content=sys_prompt),
                HumanMessage(content=f"Draft:\n{(draft or '')[:2000]}\n\nVault excerpts:\n{(citations or '')[:1500]}"),
            ])
            text = (resp.content or "").strip()
            for line in text.split("\n"):
                line = line.strip()
                if line.upper().startswith("TONE_SCORE:"):
                    try:
                        tone_points = max(0, min(20, int(re.search(r"\d+", line).group(0))))
                    except Exception:
                        pass
                elif line.upper().startswith("EXPLANATION:"):
                    explanation = line.split(":", 1)[-1].strip()[:300]
        except Exception as e:
            explanation = str(e)[:150]
            tone_points = 10

    if not explanation:
        explanation = f"Vault grounding: {vault_grounding}/40, Safety: {factual_points}/20, Personalization: {personalization}/20, Tone: {tone_points}/20."
    score = vault_grounding + factual_points + personalization + tone_points
    score = max(1, min(100, score))

    if vault_grounding >= 32:
        vault_alignment = "high"
    elif vault_grounding >= 20:
        vault_alignment = "medium"
    else:
        vault_alignment = "low"

    risk_flags_str = ", ".join(risk_flags_list) if risk_flags_list else "none"
    needs_review = score < 60 or any("legal" in f.lower() or "risk" in f.lower() for f in risk_flags_list)
    if needs_review and risk_flags_str == "none":
        risk_flags_str = "low_confidence" if score < 60 else risk_flags_str

    return {
        "score": score,
        "explanation": explanation,
        "risk_flags": risk_flags_str,
        "vault_alignment": vault_alignment,
        "prompt_version": prompt_version,
        "needs_review": needs_review,
    }


# ---------- Clay Enrichment ----------

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
    funding_options = ["$10M", "€8M", "$15M Series A", "bootstrapped", "pre-revenue"]
    base = {
        "role": str(row.get("role", "")).strip() or roles[idx % len(roles)],
        "company": str(row.get("company", "")).strip() or ((name.split()[0] if name else "Company") + " " + ["Inc", "Ltd", "Pharma", "Labs", "Solutions"][(idx // 20) % 5]),
        "recent_news": str(row.get("recent_news", "")).strip() or news_items[idx % len(news_items)],
        "funding": str(row.get("funding", "")).strip() or funding_options[idx % len(funding_options)],
        "website": str(row.get("website", "")).strip() or (f"https://{(name or 'company').lower().replace(' ', '')}.com" if name else ""),
    }
    return base


# ---------- LangGraph State & Nodes (Inbound) ----------

class InboundLeadState(TypedDict, total=False):
    row: dict
    email: str
    name: str
    source: str
    behavior: str
    initial_objection: str
    enriched_data: dict
    scenario_config: dict
    email_num: int
    sequence_emails: list
    draft: str
    citations: str
    legal_ok: bool
    legal_note: str
    attempts: int
    intent_score: int
    qualification_status: str
    needs_review: bool
    final_email: str
    user_prompt: str
    legal_footer: str
    reply_received: bool
    reply_text: str
    reply_tone: str
    draft_confidence_score: int
    draft_confidence_explanation: str
    draft_risk_flags: str
    draft_vault_alignment: str
    citations_list: list
    draft_summary: str
    draft_keywords: list
    draft_confidence_estimate: dict
    prompt_version: str
    do_not_contact: bool
    certainty_score: int
    lead_id: str
    latest_incoming_message: str
    is_nudge: bool
    reactivation_validation: dict


def validate_reactivation_lead(state: InboundLeadState) -> InboundLeadState:
    """Pre-contact validation for reactivation leads: block irrelevant/outdated before drafting."""
    if state.get("source", "").lower() != "reactivation":
        return state
    enriched = state.get("enriched_data", {})
    row = state.get("row", {})
    context = f"""
  Role: {enriched.get('role', 'unknown')}
  Company: {enriched.get('company', 'unknown')}
  Recent news: {enriched.get('recent_news', 'none')}
  Funding: {enriched.get('funding', 'none')}
  Website: {enriched.get('website', 'none')}
  Last contact: {row.get('last_contact_date', 'unknown')}
  Past project: {row.get('past_project', 'unknown')}
  Previous revenue: {row.get('previous_revenue', 'unknown')}
  Dormant reason: {row.get('dormant_reason', 'unknown')}
  """
    llm = get_llm_evaluator()
    validation = {"relevance": 0, "buying": 0, "overall": 0, "proceed": "no", "explanation": "LLM not available."}
    if llm:
        prompt = f"""Analyze if this old lead is still relevant and in a position to buy/decide.
  {context}

  Score 0-100:
  - RELEVANCE_SCORE: still in industry, same/similar role, no job change signals in news?
  - BUYING_READINESS_SCORE: positive signals (funding, expansion news, decision-making role)?
  - OVERALL_PROCEED_SCORE: 80+ = yes proceed, 60-79 = caution (review), <60 = no (block contact)

  Reply exactly:
  RELEVANCE_SCORE: X
  BUYING_READINESS_SCORE: Y
  OVERALL_PROCEED_SCORE: Z
  PROCEED: yes/caution/no
  EXPLANATION: 1-2 sentences why (e.g. "Person likely left company per news", "Strong funding + decision role = high readiness")"""
        try:
            resp = llm.invoke([HumanMessage(content=prompt)])
            text = (resp.content or "").strip()
            rel_m = re.search(r"RELEVANCE_SCORE:\s*(\d+)", text, re.I)
            buy_m = re.search(r"BUYING_READINESS_SCORE:\s*(\d+)", text, re.I)
            over_m = re.search(r"OVERALL_PROCEED_SCORE:\s*(\d+)", text, re.I)
            proceed_m = re.search(r"PROCEED:\s*(yes|caution|no)", text, re.I)
            expl_m = re.search(r"EXPLANATION:\s*(.+)", text, re.S | re.I)
            validation = {
                "relevance": int(rel_m.group(1)) if rel_m else 0,
                "buying": int(buy_m.group(1)) if buy_m else 0,
                "overall": int(over_m.group(1)) if over_m else 0,
                "proceed": (proceed_m.group(1) or "no").lower() if proceed_m else "no",
                "explanation": expl_m.group(1).strip()[:500] if expl_m else "No explanation parsed.",
            }
        except Exception:
            validation["explanation"] = "Validation LLM call failed."
    out = {**state, "reactivation_validation": validation}
    overall = validation.get("overall", 0)
    proceed = validation.get("proceed", "no")
    if overall < 60 or proceed == "no":
        out["do_not_contact"] = True
        out["qualification_status"] = "cold"
        out["intent_score"] = 0
        out["draft"] = f"VALIDATION BLOCKED CONTACT\n\nExplanation: {validation.get('explanation', '')}"
    return out


def _route_after_validate_reactivation(state: InboundLeadState) -> str:
    """If reactivation validation blocked, go to END; else continue to branch -> writer."""
    if state.get("do_not_contact") and state.get("reactivation_validation"):
        return "end"
    return "branch"


def node_enrich(state: InboundLeadState) -> InboundLeadState:
    """Enrich lead with Clay. Upsert to long-term memory. Load history for context."""
    row = state["row"]
    email = str(row.get("email", "")).strip()
    name = str(row.get("name", "")).strip()
    source = str(row.get("source", "")).strip()
    lead_id = upsert_lead(row)
    enriched = enrich_with_clay(email, name, source, row=row)
    scenario = SCENARIO_CONFIG.get(source.lower(), SCENARIO_CONFIG["default"])
    return {
        **state,
        "lead_id": lead_id,
        "email": email,
        "name": name,
        "source": source,
        "behavior": str(row.get("behavior", "")).strip(),
        "initial_objection": str(row.get("initial_objection", "")).strip(),
        "enriched_data": enriched,
        "scenario_config": scenario,
        "email_num": 0,
        "sequence_emails": [],
    }


def node_branch_scenario(state: InboundLeadState) -> InboundLeadState:
    """Branch based on source scenario."""
    source = state.get("source", "").lower()
    scenario = SCENARIO_CONFIG.get(source, SCENARIO_CONFIG["default"])
    return {**state, "scenario_config": scenario}


def node_writer_nurture(state: InboundLeadState) -> InboundLeadState:
    """Writer node: short CEO-toned email, 1 personalized line from enriched/behavior, ≤140 words. Optional latest_incoming_message = REPLY mode."""
    llm_writer = get_llm_writer()
    # #region agent log
    _dbg("node_writer_nurture", {"llm_is_none": llm_writer is None}, "H2")
    # #endregion
    if llm_writer is None:
        return {**state, "draft": "DRAFT — DO NOT SEND\n\nTEST DRAFT: LLM not loaded — check key", "citations": "", "needs_review": True}
    latest_incoming = (state.get("latest_incoming_message") or "").strip()
    is_reply_mode = bool(latest_incoming)
    is_reactivation = state.get("source", "").lower() == "reactivation"
    reactivation_context = ""
    if is_reactivation:
        last_contact = state.get("row", {}).get("last_contact_date", "a while")
        past_project = state.get("row", {}).get("past_project", "our previous collaboration")
        dormant_reason = state.get("row", {}).get("dormant_reason", "")
        previous_revenue = state.get("row", {}).get("previous_revenue", "")
        reactivation_context = f"We previously worked together on {past_project} (last contact {last_contact}). Revenue/value was {previous_revenue or 'significant'}. Dormant reason (if known): {dormant_reason or 'unknown'}."
    is_nudge = state.get("is_nudge", False)
    email_num = state.get("email_num", 1)
    total_emails = state.get("scenario_config", {}).get("emails", 3)
    name = state.get("name") or "there"
    source = state.get("source", "")
    behavior = state.get("behavior", "")
    objection = state.get("initial_objection", "")
    enriched = state.get("enriched_data", {})
    previous_emails = state.get("sequence_emails", [])
    legal_note = state.get("legal_note") or ""
    if legal_note:
        legal_block = f"\nPrevious legal rejection (fix this): {legal_note}\n"
    else:
        legal_block = ""
    query = f"CEO tone nurturing email {source} {behavior}" if not is_reply_mode else f"CEO response to reply: {latest_incoming[:200]}"
    top_chunks = vault_retrieve(query, k=6)
    citations_block = format_citations(top_chunks)
    citations_list = citations_to_schema(top_chunks)
    similarity_avg = (sum(c.get("score", 0) for c in top_chunks) / len(top_chunks)) if top_chunks else 0.0
    lead_id = state.get("lead_id", "")
    history_summary = get_lead_summary_for_prompt(lead_id, max_history=10) if lead_id else ""
    learned = get_learned_patterns()
    playbook = load_playbook()
    learning = ""
    if learned and learned != "No patterns learned yet.":
        learning += "\nLEARNED PATTERNS (incorporate):\n" + learned + "\n"
    if playbook["approved"]:
        learning += "\nApproved examples (emulate):\n" + "\n".join([e.get("email", "")[:600] for e in playbook["approved"][-5:]])
    if playbook["rejected"]:
        learning += "\nAvoid these:\n" + "\n".join([e.get("rejection_reason", "") for e in playbook["rejected"][-5:]])
    context = f"Source: {source}. Behavior: {behavior}. Objection: {objection}. Enriched: {enriched}"
    if previous_emails:
        context += f"\nPrevious emails in sequence: {len(previous_emails)} sent."
    if is_reply_mode:
        system = f"""You are the CEO replying to an inbound message. You are NOT starting a conversation — you are REPLYING.
If latest_incoming_message is present, address the user's specific text using facts from the Vault. Maintain the CEO persona. Keep the reply under 140 words. Do not sound generic.
RULES:
- Directly address what they said. Reference specific points from their message.
- Use vault facts only; never invent. No regulatory/medical claims.
- Maximum 140 words. Professional, helpful, never salesy or robotic.
{learning}
{GOLDEN_RULES}"""
        user = f"""REPLY to this message from the lead.{legal_block}
VAULT CHUNKS (use for tone and facts only):
{citations_block}

LATEST INCOMING MESSAGE (you must address this):
{latest_incoming}

LEAD CONTEXT: {context}
Recipient: {name}. Source: {source}.
{f"\n\nLEAD HISTORY (use for continuity):\n{history_summary}" if history_summary else ""}

Output ONLY the email body (no subject, no DRAFT line). Reply directly to their message using vault-backed facts. Under 140 words. Do not be generic."""
        if is_reactivation:
            system += "\nYou are RE-ACTIVATING an old client/partner. Start warm: reference past positive relationship. Be grateful for past business. No pressure. Express genuine interest in their current situation."
            user += f"\nREACTIVATION CONTEXT (MUST reference past positively):\n{reactivation_context}"
    else:
        system = f"""You are the CEO writing a short nurturing email. Use vault tone and logic only.
RULES:
- First line MUST be personalized: reference one explicit fact from ENRICHED DATA or BEHAVIOR (e.g. role, company, recent news, what they did).
- Maximum 140 words total. Short and helpful.
- Cite vault only when needed. Never invent facts; if not in vault or context, omit or say UNKNOWN. No regulatory/medical claims.
{learning}
{GOLDEN_RULES}"""
        if is_reactivation:
            system += "\nYou are RE-ACTIVATING an old client/partner. Start warm: reference past positive relationship. Be grateful for past business. No pressure. Express genuine interest in their current situation."
        nudge_instruction = "\nThis is a gentle follow-up nudge; they have not replied recently. Keep it brief and value-focused, not pushy." if is_nudge else ""
        user = f"""Write nurturing email {email_num}/{total_emails} for this inbound lead.{legal_block}{nudge_instruction}
VAULT CHUNKS (use for tone and facts only):
{citations_block}

LEAD CONTEXT (use for first-line personalization):
{context}

Recipient: {name}. Source: {source}.
{f"\n\nLEAD HISTORY (use for continuity):\n{history_summary}" if history_summary else ""}

Output ONLY the email body (no subject, no DRAFT line - we add it). Start with one personalized line using a concrete fact from enriched_data or behavior, then keep the rest to <= 140 words."""
        if is_reactivation:
            user += f"\nREACTIVATION CONTEXT (MUST reference past positively):\n{reactivation_context}\nFirst line MUST reference the past relationship (e.g. past project, last contact)."

    prompt_version = "kinesis_cursor_master:v1.0"
    try:
        resp = llm_writer.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        body = (resp.content or "").strip()
        draft = "DRAFT — DO NOT SEND\n\n" + body
        lead_context_str = f"Name: {name}. Source: {source}. Behavior: {behavior}. Objection: {objection}. Enriched: {enriched}"
        llm = get_llm()
        eval_result = evaluate_draft_confidence(body, citations_block, lead_context_str, llm, citations_list=citations_list)
        needs_review = eval_result.get("needs_review", eval_result["score"] < 60)
        heuristic_flags = []
        if name and name.lower() != "there" and name not in body:
            heuristic_flags.append("no_personalization")
        draft_confidence_estimate = {
            "similarity_avg": round(similarity_avg, 4),
            "vault_alignment": eval_result.get("vault_alignment", "low"),
            "heuristic_flags": heuristic_flags,
        }
        summary, keywords = "", []
        try:
            sum_llm = get_llm()
            if sum_llm:
                sum_resp = sum_llm.invoke([
                    SystemMessage(content="Reply with one line SUMMARY: then KEYWORDS: comma-separated list of 3-5 topics. No other text."),
                    HumanMessage(content=f"Email body:\n{body[:1500]}"),
                ])
                sum_text = (sum_resp.content or "").strip()
                if "SUMMARY:" in sum_text:
                    summary = sum_text.split("SUMMARY:")[-1].split("KEYWORDS:")[0].strip()[:200]
                if "KEYWORDS:" in sum_text:
                    kw = sum_text.split("KEYWORDS:")[-1].strip()
                    keywords = [k.strip() for k in kw.replace(",", " ").split()[:5] if k.strip()]
        except Exception:
            pass
        return {
            **state,
            "draft": draft,
            "citations": citations_block,
            "citations_list": citations_list,
            "user_prompt": user,
            "draft_summary": summary,
            "draft_keywords": keywords,
            "draft_confidence_estimate": draft_confidence_estimate,
            "draft_confidence_score": eval_result["score"],
            "draft_confidence_explanation": eval_result["explanation"],
            "draft_risk_flags": eval_result["risk_flags"],
            "draft_vault_alignment": eval_result["vault_alignment"],
            "needs_review": needs_review,
            "prompt_version": prompt_version,
        }
    except Exception as e:
        err_msg = str(e).replace("\n", " ")[:200]
        return {
            **state,
            "draft": f"DRAFT — DO NOT SEND\n\nTEST DRAFT: Generation failed - {err_msg}",
            "citations": "",
            "citations_list": [],
            "needs_review": True,
            "draft_confidence_score": 0,
            "draft_confidence_explanation": "Generation failed; no evaluation.",
            "draft_risk_flags": "generation_error",
            "draft_vault_alignment": "low",
            "draft_summary": "",
            "draft_keywords": [],
            "draft_confidence_estimate": {"similarity_avg": 0, "vault_alignment": "low", "heuristic_flags": ["generation_error"]},
            "prompt_version": prompt_version,
        }


def node_legal(state: InboundLeadState) -> InboundLeadState:
    """Legal compliance check. After 3 rejections: mark do_not_contact and add failure insight."""
    row = state.get("row", {})
    if state.get("source") == "reactivation" and (row.get("opt_out") or row.get("do_not_contact")):
        return {**state, "do_not_contact": True, "legal_ok": False, "legal_note": "Old lead with opt-out"}
    llm = get_llm()
    draft = state.get("draft") or ""
    if not draft or llm is None:
        return {**state, "legal_ok": True, "legal_note": ""}
    attempts = state.get("attempts", 0) + 1
    footer = state.get("legal_footer", LEGAL_FOOTER)
    full = draft + footer
    system = """You are a Legal & Pharma Compliance reviewer. Check:
1. NO false medical/therapeutic claims.
2. NO aggressive guarantees (no "100%", "guaranteed", "unbreakable"); rephrase to qualified claims e.g. "highly durable per ISO".
3. GDPR/CAN-SPAM: must have clear unsubscribe and physical address (we append footer).
4. FDA-style: no absolute efficacy claims.
Reply VERDICT: SAFE or VERDICT: UNSAFE and Rejection Note: [what to fix]."""
    try:
        resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=f"Draft to review:\n{full}")])
        text = (resp.content or "").strip().upper()
        legal_ok = "VERDICT: SAFE" in text
        note = text.split("REJECTION NOTE:")[-1].strip() if "REJECTION NOTE:" in text else ""
        final_email = (draft + footer) if legal_ok else ""
        needs_review = state.get("needs_review", False) or (not legal_ok and attempts >= 3)
        do_not_contact = False
        if not legal_ok and attempts >= 3:
            do_not_contact = True
            add_failure_insight("Legal rejection 3 times for same sequence; lead marked do_not_contact", "legal_3x_reject", state.get("email", ""))
        return {
            **state,
            "legal_ok": legal_ok,
            "legal_note": note,
            "attempts": attempts,
            "final_email": final_email,
            "needs_review": needs_review,
            "do_not_contact": do_not_contact,
        }
    except Exception:
        return {**state, "legal_ok": True, "attempts": attempts}


def node_qualification_scoring(state: InboundLeadState) -> InboundLeadState:
    """Score intent + certainty from behavioral signals + LLM. Returns intent_score, certainty_score."""
    llm = get_llm()
    behavior = state.get("behavior", "")
    objection = state.get("initial_objection", "")
    reply_text = state.get("reply_text", "")
    email_num = state.get("email_num", 0)
    tone_analysis = state.get("reply_tone", "") or state.get("tone_analysis", "")
    lead_id = state.get("lead_id", "")
    history = get_lead_history(lead_id) if lead_id else []
    reply_count = sum(1 for h in history if (h.get("reply_received") or h.get("reply_text")))
    if reply_text:
        reply_count = max(reply_count, 1)
    llm_intent = None
    if llm:
        try:
            context = f"Behavior: {behavior}. Objection: {objection}. Reply: {reply_text[:500] if reply_text else 'None'}. Email {email_num} in sequence."
            resp = llm.invoke([
                SystemMessage(content="Score lead intent 0-100% based on behaviors, replies, objections. Reply: INTENT_SCORE: [0-100] and STATUS: [cold/warm/hot]"),
                HumanMessage(content=f"Context:\n{context}\n\nScore?"),
            ])
            text = (resp.content or "").strip()
            m = re.search(r"INTENT_SCORE:\s*(\d+)", text)
            llm_intent = int(m.group(1)) if m else None
        except Exception:
            pass
    intent_score, certainty_score = compute_intent_and_certainty(
        behavior=behavior,
        objection=objection,
        reply_text=reply_text,
        reply_count=reply_count,
        email_num=email_num,
        tone_analysis=tone_analysis,
        history=history,
        llm_intent=llm_intent,
    )
    qualification = scoring_qualification_status(intent_score)
    return {
        **state,
        "intent_score": intent_score,
        "certainty_score": certainty_score,
        "qualification_status": qualification,
    }


def node_reply_handler(state: InboundLeadState) -> InboundLeadState:
    """Analyze reply and generate response."""
    llm = get_llm()
    reply_text = state.get("reply_text", "")
    if not reply_text or llm is None:
        return state
    msgs = [
        SystemMessage(content="Analyze email reply tone and content. Reply: TONE: [formal/casual/younger/neutral] and KEY_POINTS: [list]"),
        HumanMessage(content=f"Reply:\n{reply_text}"),
    ]
    try:
        resp = llm.invoke(msgs)
        text = (resp.content or "").strip()
        tone_match = re.search(r"TONE:\s*(\w+)", text)
        tone = tone_match.group(1).lower() if tone_match else "neutral"
        # Generate response adapted to tone but anchored to CEO base
        chunks = vault_retrieve("CEO tone response handling", k=5)
        citations = format_citations(chunks)
        system = f"""You are a digital clone of the CEO. Generate a response to this reply.
Adapt slightly to recipient tone ({tone}) but anchor to CEO base/No-Go list. Never sound salesy.
{GOLDEN_RULES}"""
        user = f"""VAULT:\n{citations}\n\nReply received:\n{reply_text}\n\nGenerate CEO-style response (email body only)."""
        resp_gen = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        draft = (resp_gen.content or "").strip()
        return {
            **state,
            "reply_tone": tone,
            "draft": draft,
            "citations": citations,
        }
    except Exception:
        return state


def _route_after_legal(state: InboundLeadState) -> str:
    if state.get("legal_ok"):
        return "qualification"
    if state.get("attempts", 0) >= 3:
        return "end"
    return "writer"


def analyze_reply(incoming_reply: str) -> dict:
    """Analyze reply tone and key points; returns {tone: str, key_points: list} for memory/context."""
    reply = (incoming_reply or "").strip()
    if not reply:
        return {"tone": "neutral", "key_points": []}
    llm = get_llm()
    if llm is None:
        return {"tone": "neutral", "key_points": []}
    try:
        resp = llm.invoke([
            SystemMessage(content="Analyze email reply tone and content. Reply: TONE: [formal/casual/younger/neutral] and KEY_POINTS: [comma-separated list]"),
            HumanMessage(content=f"Reply:\n{reply}"),
        ])
        text = (resp.content or "").strip()
        tone_match = re.search(r"TONE:\s*(\w+)", text)
        tone = tone_match.group(1).lower() if tone_match else "neutral"
        key_points = []
        kp_match = re.search(r"KEY_POINTS?:\s*\[?(.*?)\]?", text, re.DOTALL | re.IGNORECASE)
        if kp_match:
            raw = kp_match.group(1).strip()
            key_points = [x.strip() for x in re.split(r"[,;]", raw) if x.strip()][:8]
        return {"tone": tone, "key_points": key_points}
    except Exception:
        return {"tone": "neutral", "key_points": []}


def _last_event_ts_and_type(history: list) -> tuple:
    """Return (last_timestamp_iso, 'email_sent'|'reply_received'|None)."""
    if not history:
        return None, None
    for h in reversed(history):
        ts = h.get("timestamp")
        if h.get("reply_received"):
            return ts, "reply_received"
        if h.get("email_sent"):
            return ts, "email_sent"
    return None, None


def _days_since(ts_iso: Optional[str]) -> Optional[int]:
    if not ts_iso:
        return None
    try:
        dt = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
        return (datetime.utcnow() - dt).days
    except Exception:
        return None


def process_lead_state(lead_id: str, incoming_reply: Optional[str] = None) -> dict:
    """
    Reactive state machine: ONE cycle per call.
    - Load lead memory & score.
    - If score > 80 (HOT): return HANDOFF_TO_SALES (no email).
    - If incoming_reply: analyze_reply → update memory → generate ONE response draft (reply + Vault).
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


# ---------- SendGrid Integration ----------

def strip_draft_tag(body: str) -> str:
    """Remove top-line DRAFT — DO NOT SEND so approved sends are clean."""
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
    last_email_snippet = (emails_sent[-1][:200] if emails_sent else "—") or "—"

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
            subject = "HOT Lead Alert: {} — {}".format(name, company)
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


# ---------- Flask Webhook (run in background) ----------

def start_flask_webhook():
    """Start Flask webhook server in background thread."""
    from flask import Flask, request, jsonify
    app = Flask(__name__)

    @app.route("/webhook", methods=["POST"])
    def webhook():
        data = request.json or {}
        email = data.get("email", "")
        name = data.get("name", "")
        source = data.get("source", "webhook")
        row = {"email": email, "name": name, "source": source, "behavior": data.get("behavior", ""), "initial_objection": data.get("objection", "")}
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


# ---------- Objection extraction (deterministic + LLM classifier) ----------

OBJECTION_LABELS = ["budget", "timing", "product_mismatch", "wrong_contact", "no_need", "legal", "spam", "unresponsive", "other"]


def extract_objection(reply_text: str) -> tuple[str, int]:
    """Deterministic: normalize objection from reply. Returns (label, confidence 0-100)."""
    t = (reply_text or "").lower().strip()
    if not t:
        return "other", 0
    if "not interested" in t or "no thanks" in t or "not at this time" in t:
        return "no_need", 85
    if "$" in t or "budget" in t or "cost" in t or "expensive" in t:
        return "budget", 80
    if "not the right person" in t or "wrong person" in t or "different department" in t:
        return "wrong_contact", 85
    if "timing" in t or "later" in t or "next quarter" in t:
        return "timing", 75
    if "unsubscribe" in t or "remove" in t or "spam" in t:
        return "spam", 90
    if "legal" in t or "compliance" in t or "policy" in t:
        return "legal", 70
    if "wrong product" in t or "not a fit" in t or "different solution" in t:
        return "product_mismatch", 75
    return "other", 50


def classify_objection_llm(reply_text: str, llm) -> tuple[str, int, str]:
    """LLM classifier when deterministic is uncertain. Returns (label, confidence 0-100, one_line_reason)."""
    if not llm or not (reply_text or "").strip():
        return "other", 0, ""
    prompt = f"""Extract a single normalized objection from the following reply. Choose exactly one from: {", ".join(OBJECTION_LABELS)}.
Reply with exactly: LABEL: <label> CONFIDENCE: <0-100> REASON: <one line>
Reply text:
{(reply_text or "")[:800]}"""
    try:
        resp = llm.invoke([
            SystemMessage(content="You are an objection classifier. Output only LABEL: CONFIDENCE: REASON: on one or two lines."),
            HumanMessage(content=prompt),
        ])
        text = (resp.content or "").strip()
        label = "other"
        confidence = 50
        reason = ""
        for line in text.split("\n"):
            line = line.strip()
            if line.upper().startswith("LABEL:"):
                val = line.split(":", 1)[-1].strip().lower()
                if val in OBJECTION_LABELS:
                    label = val
                else:
                    label = "other"
            elif line.upper().startswith("CONFIDENCE:"):
                try:
                    confidence = max(0, min(100, int(re.search(r"\d+", line).group(0))))
                except Exception:
                    pass
            elif line.upper().startswith("REASON:"):
                reason = line.split(":", 1)[-1].strip()[:200]
        return label, confidence, reason
    except Exception:
        return "other", 0, ""


def extract_objection_with_llm(reply_text: str, llm) -> tuple[str, int]:
    """Deterministic first; if label is 'other' and confidence < 70, use LLM classifier. Returns (label, confidence)."""
    label, conf = extract_objection(reply_text)
    if (label != "other") or conf >= 70:
        return label, conf
    label_llm, conf_llm, _ = classify_objection_llm(reply_text, llm)
    return label_llm, conf_llm


# ---------- Funnel Analytics (master schema) ----------

def build_death_story_narrative(seq: dict, stage: str, main_objection: str, raw_excerpt: str, llm) -> str:
    """Generate 3-sentence narrative: behavior => why died => suggested fix."""
    if not llm:
        return f"Lead dropped at {stage}; objection: {main_objection}. Review content or cadence."
    source = seq.get("source", "unknown")
    behavior = (seq.get("behavior") or "")[:300]
    name = seq.get("name", "Lead")
    prompt = f"""Write exactly 3 short sentences for a "death story" report.
Sentence 1: What the lead did (source: {source}, behavior: {behavior or 'none'}).
Sentence 2: Why they dropped (objection: {main_objection}; excerpt: {raw_excerpt[:150]}).
Sentence 3: One suggested fix for management (e.g. change content, cadence, or targeting).
Do not use the lead's name or email. Output only the 3 sentences, no labels."""
    try:
        resp = llm.invoke([
            SystemMessage(content="You write brief, anonymized funnel narratives for internal reports. Plain English, 3 sentences only."),
            HumanMessage(content=prompt),
        ])
        out = (resp.content or "").strip()
        if out and len(out) > 50:
            return out[:600]
    except Exception:
        pass
    return f"Lead showed interest via {source} but dropped at {stage} ({main_objection}). Consider adjusting cadence or content for this segment."


def compute_funnel_analytics(sequences_data: List[dict], period: Optional[str] = None, use_llm_stories: bool = True) -> dict:
    """
    Funnel Analytics: drop-offs by stage, top objections (deterministic + LLM), death_stories (3-sentence narrative).
    Input: sequences_data = array of sequence objects (each has sequence_emails, events if present, reply_text, etc.).
    Output: analytics JSON per master schema.
    """
    period = period or f"{datetime.now().date()}"
    llm = get_llm() if use_llm_stories else None

    analytics = {
        "period": period,
        "total_leads": len(sequences_data),
        "by_stage": {},
        "drop_offs": [],
        "top_objections": [],
        "avg_confidence_by_stage": {},
        "death_stories": [],
    }

    objection_counts: Dict[str, int] = {}
    confidence_by_stage: Dict[str, List[int]] = {}
    death_story_candidates: List[tuple] = []

    for seq in sequences_data:
        events = seq.get("events", [])
        sequence_emails = seq.get("sequence_emails", [])
        emails_sent = len(sequence_emails)
        stage = f"email_{min(emails_sent, 5)}" if emails_sent else "email_1"
        status = seq.get("qualification_status", "cold")
        if stage not in analytics["by_stage"]:
            analytics["by_stage"][stage] = {"count": 0, "died": 0, "hot": 0, "warm": 0, "cold": 0}
        analytics["by_stage"][stage]["count"] += 1
        analytics["by_stage"][stage][status] = analytics["by_stage"][stage].get(status, 0) + 1
        if status == "cold" and emails_sent > 0:
            analytics["by_stage"][stage]["died"] = analytics["by_stage"][stage].get("died", 0) + 1
            analytics["drop_offs"].append({"stage": stage, "lead_id": seq.get("email", "")})

        conf = seq.get("draft_confidence_score")
        if conf is not None:
            confidence_by_stage.setdefault(stage, []).append(conf)

        reply = (seq.get("reply_text") or "").strip()
        if reply:
            label, _ = extract_objection_with_llm(reply, llm)
            objection_counts[label] = objection_counts.get(label, 0) + 1

        if status == "cold" and (reply or seq.get("do_not_contact")):
            label, _ = extract_objection_with_llm(reply, llm) if reply else ("opt_out", 100)
            raw = (reply[:200] + "…") if len(reply) > 200 else (reply or "do_not_contact")
            death_story_candidates.append((seq, stage, label, raw))

    for stage, vals in confidence_by_stage.items():
        analytics["avg_confidence_by_stage"][stage] = int(round(sum(vals) / len(vals), 0)) if vals else 0
    analytics["top_objections"] = sorted(
        [{"label": k, "count": v} for k, v in objection_counts.items()],
        key=lambda x: -x["count"],
    )[:10]

    for seq, stage, main_objection, raw_excerpt in death_story_candidates[:20]:
        short_narrative = build_death_story_narrative(seq, stage, main_objection, raw_excerpt, llm) if use_llm_stories else (
            f"Lead dropped at {stage}; objection: {main_objection}. Review content or cadence."
        )
        analytics["death_stories"].append({
            "lead_id": seq.get("email", ""),
            "main_objection": main_objection,
            "raw_excerpt": raw_excerpt or "do_not_contact",
            "short_narrative": short_narrative,
        })

    return analytics


def compute_funnel_stats(sequences_data: List[dict], period: Optional[str] = None) -> dict:
    """Thin wrapper: computes full funnel analytics (drop-offs, objections with LLM, 3-sentence death stories)."""
    return compute_funnel_analytics(sequences_data, period=period, use_llm_stories=True)


def log_funnel_events(funnel_data: List[dict]) -> None:
    """Persist funnel events for analytics. Appends batch to SEQUENCES_DIR/funnel_events.json."""
    if not funnel_data:
        return
    path = os.path.join(SEQUENCES_DIR, "funnel_events.json")
    batch = {"timestamp": datetime.utcnow().isoformat(), "events": funnel_data}
    existing = []
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                existing = data.get("batches", [])[-100:]
        except Exception:
            pass
    existing.append(batch)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"batches": existing}, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


def load_logged_funnel_events() -> List[dict]:
    """Load all logged funnel events (flattened) for cross-session analytics."""
    path = os.path.join(SEQUENCES_DIR, "funnel_events.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        out = []
        for b in data.get("batches", []):
            out.extend(b.get("events", []))
        return out
    except Exception:
        return []


# ---------- UI ----------

# Service Mode: Admin vs Client
service_mode = st.sidebar.selectbox("Service Mode", ["Admin", "Client"], key="service_mode")

with st.sidebar:
    st.header("🏢 Settings")
    biz_name = st.text_input("Your Company Name", "My Agency", key="biz_name")
    biz_address = st.text_area("Your Business Address", "123 Business St", key="biz_address")
    from_email = st.text_input("From Email (SendGrid)", os.getenv("SENDGRID_FROM_EMAIL", ""), key="from_email")
    from_name = st.text_input("From Name", biz_name, key="from_name")
    legal_footer = LEGAL_FOOTER.replace("[Your Company Name]", biz_name).replace("[Your Physical Business Address]", biz_address)

    if service_mode == "Admin":
        st.divider()
        st.subheader("📚 Knowledge Vault")
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
        st.subheader("🎤 CEO Interview Kit")
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
        st.subheader("🔗 Clay Enrichment")
        clay_webhook_url = st.text_input("Clay Webhook URL", os.getenv("CLAY_WEBHOOK_URL", ""), key="clay_webhook")
        if clay_webhook_url:
            os.environ["CLAY_WEBHOOK_URL"] = clay_webhook_url

        st.divider()
        st.subheader("📡 Webhook Server")
        if st.button("Start Flask Webhook (port 5000)", key="start_webhook"):
            start_flask_webhook()
            st.success("Webhook running. Use ngrok: ngrok http 5000")

    st.divider()
    csv_file = st.file_uploader("Upload Inbound CSV", type=["csv"], key="inbound_csv")
    st.caption("Columns: name, email, source (use 'reactivation'), last_contact_date, past_project, dormant_reason, previous_revenue, behavior, initial_objection")

# Main
st.title("✉️ Kinesis Reactivation — Old CRM Leads")
st.caption("Automate CEO-style nurturing of warm inbound leads. Sequences, replies, qualification scoring, funnel analysis.")
st.success("LangGraph ready")

if "no_response_checked" not in st.session_state:
    check_no_response_leads()
    st.session_state["no_response_checked"] = True

tab_dashboard, tab_statistics = st.tabs(["Dashboard", "Statistics"])

with tab_dashboard:
    if csv_file:
        try:
            df = pd.read_csv(csv_file)
            required_cols = ["name", "email"]
            if not all(c in df.columns for c in required_cols):
                st.error(f"CSV must have columns: {required_cols}")
            else:
                st.subheader("📊 Leads Dashboard")
                st.dataframe(df.head(50), use_container_width=True)

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
                            draft_text = seq.get("final_email") or seq.get("draft") or ""
                            st.markdown(f"---")
                            certainty = seq.get("certainty_score")
                            st.markdown(f"**Email {seq.get('email_num', seq_idx + 1)}** — Intent: {intent}% | Certainty: {certainty}% | Status: {status}" + (f" | **Confidence:** {conf}/100 ({vault_align})" if conf is not None else ""))
                            if conf is not None and (seq.get("draft_confidence_explanation") or seq.get("draft_risk_flags")):
                                with st.expander("Draft evaluation", expanded=False):
                                    st.caption(seq.get("draft_confidence_explanation", ""))
                                    if seq.get("draft_risk_flags") and str(seq.get("draft_risk_flags", "")).lower() not in ("none", ""):
                                        st.caption(f"Risk flags: {seq.get('draft_risk_flags')}")
                            if draft_text and "Generation failed" in draft_text:
                                st.warning("Generation failed — see draft below for error details.")
                            if not draft_text:
                                st.error("No draft — check MISTRAL_API_KEY (or OPENAI fallback) and terminal.")
                            st.text_area("Draft", value=draft_text or "(no draft)", height=180, key=f"test_draft_{seq_idx}", disabled=True)

                if "sequences_results" in st.session_state:
                    results = st.session_state["sequences_results"]
                    all_high_conf = all(s.get("draft_confidence_score", 0) >= 80 and not s.get("needs_review", True) for r in results for s in r["sequences"])
                    if all_high_conf and st.button("🚀 Approve & Send All High Confidence Emails", type="primary"):
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
                    st.subheader("📈 Funnel Analysis (Wald-style)")
                    st.json(funnel_stats)
                    st.subheader("🎯 Deep Funnel Death Analytics (Management)")
                    st.markdown(f"**Main objection:** {deep_funnel.get('main_objection', 'N/A')} | **Emotional reason:** {deep_funnel.get('emotional_reason', 'N/A')}")
                    if deep_funnel.get("common_pattern"):
                        st.caption(f"Common pattern: {deep_funnel['common_pattern']}")
                    with st.expander("Death stories", expanded=False):
                        for ds in deep_funnel.get("death_stories", [])[:5]:
                            st.markdown(f"**{ds.get('stage', '')} — {ds.get('main_objection', '')}**")
                            st.caption(ds.get("death_story", ""))

                    all_time_events = load_logged_funnel_events()
                    if all_time_events:
                        with st.expander("All-time funnel (logged events)", expanded=False):
                            all_time_funnel = compute_deep_funnel(all_time_events, llm=get_llm())
                            st.caption(f"Total logged events: {len(all_time_events)}")
                            st.markdown(f"**Main objection:** {all_time_funnel.get('main_objection', 'N/A')} | **Emotional reason:** {all_time_funnel.get('emotional_reason', 'N/A')}")

                    # Qualification scoring
                    hot_leads = [r for r in results if any(s.get("intent_score", 0) >= 80 for s in r["sequences"])]
                    st.metric("Hot Leads (≥80% intent)", len(hot_leads), f"{len(hot_leads) * 1000} potential value")

                    # Batch-level confidence messages (before individual emails)
                    all_emails = [s for r in results for s in r["sequences"]]
                    all_high_conf = all(s.get("draft_confidence_score", 0) >= 80 for s in all_emails)
                    if all_emails and all_high_conf:
                        st.success("All Clear — all emails are high confidence and ready to approve.")
                    low_conf_count = sum(1 for s in all_emails if s.get("draft_confidence_score", 0) < 55)
                    if low_conf_count > 0:
                        st.warning(f"{low_conf_count} email(s) require review before sending")

                    # Summary panel: green / yellow / red counts and Approve All Green
                    green_count = sum(1 for s in all_emails if s.get("draft_confidence_score", 0) >= 80)
                    yellow_count = sum(1 for s in all_emails if 55 <= s.get("draft_confidence_score", 0) < 80)
                    red_count = sum(1 for s in all_emails if s.get("draft_confidence_score", 0) < 55)
                    st.subheader("📋 Email summary")
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
                        lead_label = f"{r['lead'].get('name', 'Lead')} — {r['lead'].get('source', 'unknown')}"
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
                                        v_color = "🟢"
                                    elif overall >= 60:
                                        v_color = "🟡"
                                    else:
                                        v_color = "🔴"
                                    st.markdown(f"**Reactivation validation** — {v_color} Overall: {overall}/100 (Relevance: {val.get('relevance', 0)}, Buying readiness: {val.get('buying', 0)}) | Proceed: **{val.get('proceed', 'no')}**")
                                    st.caption(val.get("explanation", ""))
                                    st.divider()
                            for seq_idx, seq in enumerate(r["sequences"]):
                                approved_green_ids = st.session_state.get("approved_green", set())
                                if (idx, seq_idx) in approved_green_ids:
                                    st.success("✓ Approved and queued for sending")
                                    continue
                                email_num = seq.get("email_num", seq_idx + 1)
                                intent = seq.get("intent_score", 0)
                                qual = seq.get("qualification_status", "warm")
                                conf = seq.get("draft_confidence_score", 0)
                                if conf >= 80:
                                    color = "🟢"  # green
                                elif conf >= 60:
                                    color = "🟡"  # yellow
                                else:
                                    color = "🔴"  # red
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
                                st.markdown(f"Score {conf}/100 | Vault alignment: {vault_align} | Risk flags: {risk_flags}")
                                st.markdown(f"**Email {email_num}** — Confidence: {color} {conf}/100 | Intent: {intent}% | Status: {qual}")
                                if conf is not None and (seq.get("draft_confidence_explanation") or seq.get("draft_risk_flags")):
                                    with st.expander("Draft evaluation", expanded=False):
                                        st.caption(seq.get("draft_confidence_explanation", ""))
                                        if seq.get("draft_risk_flags") and str(seq.get("draft_risk_flags", "")).lower() not in ("none", ""):
                                            st.caption(f"Risk flags: {seq.get('draft_risk_flags')}")
                                if draft_text and "Generation failed" in draft_text:
                                    st.warning("Generation failed — see draft below for error details.")
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
                                        if read_checked and st.button("✅ Approve & Send", key=f"send_{idx}_{seq_idx}"):
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
                                    st.error("No draft generated — check MISTRAL_API_KEY (or OPENAI fallback) and terminal for errors.")

        except Exception as e:
            st.error(str(e))

with tab_statistics:
    # ---- Section 1: Overview metrics ----
    period = st.radio("Period", ["This month", "All time"], horizontal=True, key="stats_period")
    now = datetime.utcnow()
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
    st.subheader("SECTION 1 — OVERVIEW METRICS")
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
    st.subheader("SECTION 2 — FUNNEL DROPOUT ANALYSIS")
    st.markdown("**Where Leads Are Dying — Abraham Wald Analysis**")
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
    st.dataframe(funnel_df, use_container_width=True)
    max_stopped_idx = max(range(len(funnel_rows)), key=lambda i: funnel_rows[i]["Stopped"]) if funnel_rows else 0
    for i, row in enumerate(funnel_rows):
        dropout_frac = (row["Dropout %"] / 100.0) if row["Dropout %"] else 0
        st.progress(min(dropout_frac, 1.0))
        if i == max_stopped_idx and row["Stopped"] > 0:
            st.error(f"Critical dropout point — {row['Dropout %']}% of all leads stop here")
    st.markdown("---")

    # ---- Section 3: What killed the lead ----
    st.subheader("SECTION 3 — WHAT KILLED THE LEAD (Dropout Theme Analysis)")
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
    st.subheader("SECTION 4 — WHAT HOT LEADS HAVE IN COMMON (Survivor Analysis)")
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
    st.subheader("SECTION 5 — OBJECTION INTELLIGENCE")
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

st.caption("Kinesis Reactivation: Old CRM leads, sequences, replies, qualification, funnel analysis. Admin mode: full access. Client mode: dashboard only.")
