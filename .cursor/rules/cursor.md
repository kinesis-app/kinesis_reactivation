# KINESIS AI — Cursor Rules
# Python + LangGraph + Streamlit | February 2026
# Read this fully before writing any code.

## PROJECT OVERVIEW

Kinesis AI is a Streamlit web application with a LangGraph AI pipeline that writes and sends
CEO-style nurturing emails to leads. There are two separate apps:
- **kinesis_inbound.py** — for new leads (2–4 email sequences)
- **kinesis_reactivation.py** — for cold/dormant leads (up to 8 emails, includes pre-contact validation)

Both apps share four modules: memory_manager.py, scoring_engine.py, learning_loop.py, funnel_analytics.py.
This is NOT a microservices system. It is a single-process Streamlit app (+ a new FastAPI webhook process).

---

## TECH STACK

- **Python 3.10+** — strict requirement
- **Streamlit** — UI and dashboard (do not replace with Flask or FastAPI for UI)
- **LangGraph** — AI pipeline orchestration (state machine via TypedDict + nodes)
- **Mistral AI** — LLM for email generation (via MISTRAL_API_KEY)
- **SQLite** — local persistence for leads, emails, training pairs
- **SendGrid** — outbound email delivery + inbound parse webhooks
- **FastAPI + uvicorn** — webhook server only (webhook_server.py — separate process)
- **Clay** — lead enrichment via webhook (optional; falls back to simulation)
- **Slack** — HOT lead alerts via SLACK_WEBHOOK_URL

---

## FILE STRUCTURE

```
kinesis_inbound.py          # Main Streamlit app — Inbound (~2,221 lines)
kinesis_reactivation.py     # Main Streamlit app — Reactivation (~2,337 lines)
webhook_server.py           # FastAPI webhook server (TO BE BUILT)
memory_manager.py           # Shared — SQLite lead/email persistence
scoring_engine.py           # Shared — confidence + intent scoring logic
learning_loop.py            # Shared — training pair storage + pattern learning
funnel_analytics.py         # Shared — Statistics tab data and analytics
requirements.txt            # Must include fastapi and uvicorn (not yet added)
.env / .env.template        # All secrets and config
Procfile                    # Start both Streamlit + uvicorn together
```

---

## LANGGRAPH PIPELINE — HOW IT WORKS

Both apps use LangGraph. The graph is compiled once and invoked per lead via `process_lead_state()`.
State flows through nodes as a TypedDict (`InboundLeadState`).

Core node flow (Inbound):
```
node_enricher → node_scorer → node_writer_nurture → node_sender → node_logger
```

Reactivation adds before the main flow:
```
node_pre_contact_validator → [blocked? stop] → node_enricher → ...
```

Key rules:
- Max retries = 3. After 3 failures → set `do_not_contact=True`, end graph.
- `reply_handler` node is NOT in the main graph — it is called directly via `handle_incoming_reply()`.
- Do NOT restructure the graph without understanding all downstream effects.
- `process_lead_state()` is the core dispatch function — treat it as sacred. Read it fully before touching.

---



## WHAT NOT TO TOUCH

These are fully built and working. Do not refactor, restructure, or "improve" them
unless explicitly asked:

- `process_lead_state()` — core state machine
- `get_lead_summary_for_prompt()` — prompt context builder
- `handle_incoming_reply()` — reply processing logic
- `memory_manager.py` — any SQLite schema or write functions
- `scoring_engine.py` — scoring weights and logic
- `learning_loop.py` — training pair storage
- The LangGraph graph definition and node wiring
- SendGrid send logic
- Slack alert logic
- The Statistics tab (funnel_analytics.py)

If you think something in the above list needs changing, stop and ask first.

---

## CODE STYLE & CONVENTIONS

### General
- Python 3.10+ — use match/case where appropriate, use `|` union types in hints
- Type hints on all new functions — use TypedDict for state objects
- Docstrings on every new function (one-line minimum, full for complex ones)
- No unused imports — remove them
- Constants in UPPER_SNAKE_CASE at top of file

### Streamlit
- Never use `st.experimental_*` — use stable equivalents
- `delta_color` on `st.metric` must be `'normal'`, `'inverse'`, or `'off'` — never `'green'`/`'red'`/`'yellow'`
- Use `st.session_state` for any state that must persist across reruns
- Sidebar layout must not be changed without approval — users know where things are

### LangGraph
- All nodes must return the full state dict (or a partial update dict)
- Never mutate state in place inside a node — always return updates
- New nodes must be added to the graph with `graph.add_node()` and wired with `graph.add_edge()`
- Conditional edges use `graph.add_conditional_edges()`

### FastAPI (webhook_server.py only)
- Use `async def` for all route handlers
- Parse SendGrid payloads with `await request.form()` — they POST as multipart form data
- Always return `JSONResponse({"status": "ok"}, status_code=200)` to SendGrid — even on errors
  (SendGrid will retry if it gets a non-200, causing duplicate processing)
- Validate incoming data with Pydantic models

### Database (SQLite via memory_manager.py)
- Never write raw SQL outside of memory_manager.py
- All DB access goes through the functions in memory_manager.py
- Never drop or alter existing tables — only ADD columns if needed, with ALTER TABLE

### Environment Variables
- All secrets via `.env` — never hardcode keys
- New env vars must be added to `.env.template` with a comment explaining the value
- Load with `python-dotenv` — already configured

---

## ENVIRONMENT VARIABLES (full reference)

```
MISTRAL_API_KEY          # LLM — required
SENDGRID_API_KEY         # Email sending — required
FROM_EMAIL               # Verified sender address — required
CLAY_WEBHOOK_URL         # Lead enrichment — optional (simulation if missing)
SLACK_WEBHOOK_URL        # HOT lead alerts — optional
SALES_ALERT_EMAIL        # Sales team alert address — optional
WEBHOOK_PORT             # FastAPI port, default 5001 — add this
DATABASE_PATH            # SQLite file path, default ./kinesis.db
```

---

## RUNNING THE SYSTEM

Both processes must run together. Use a Procfile:
```
web: streamlit run kinesis_inbound.py --server.port $PORT
webhook: uvicorn webhook_server:app --host 0.0.0.0 --port ${WEBHOOK_PORT:-5001}
```

Local dev: run both manually in separate terminals, or use `honcho` / `overmind`.

---

## SENDGRID INBOUND PARSE SETUP (for webhook_server.py)

1. SendGrid Dashboard → Settings → Inbound Parse → Add Host & URL
2. Host: your email subdomain (e.g. `leads.yourdomain.com`)
3. URL: `https://your-public-domain.com/webhook/email-reply`
4. DNS: MX record pointing to `mx.sendgrid.net` for that subdomain
5. Do NOT check "Post the raw, full MIME message" — we need parsed fields
6. Test locally with ngrok: `ngrok http 5001` → update SendGrid URL temporarily

---

## CONFIDENCE SCORE — HOW IT IS CALCULATED

| Component      | Max Points | Notes                                      |
|----------------|------------|--------------------------------------------|
| Vault Grounding | 40        | Facts sourced from Knowledge Vault docs    |
| Factual Safety  | 20        | Drops if risky phrases detected            |
| Personalisation | 20        | Based on CSV columns provided              |
| Tone Match      | 20        | Compared against CEO Interview Kit         |

Green = 80–100 | Yellow = 55–79 | Red = 0–54

---

## WHEN IN DOUBT

- Ask before touching the LangGraph graph structure
- Ask before changing any SQLite schema
- Ask before modifying scoring weights in scoring_engine.py
- Changes to both app files (inbound + reactivation) must be mirrored — they share the same logic
- Test webhook endpoints with ngrok before marking complete
- The `past_history_json` feature must be backward-compatible — no CSV with existing columns should break
