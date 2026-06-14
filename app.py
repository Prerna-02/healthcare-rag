"""
Healthcare Knowledge Navigator — Phase 6 (frontend)
A thin Streamlit client. It does NOT talk to the models directly — it POSTs the
user's question to the FastAPI backend (api.py) and renders the JSON it returns.

Run BOTH (in two terminals):
    1) uvicorn api:app --port 8000          # backend
    2) streamlit run app.py                 # this UI
"""
import os

import requests
import streamlit as st

API_URL = os.environ.get("HEALTHRAG_API_URL", "http://127.0.0.1:8000")
DISCLAIMER_BANNER = (
    "⚕️ **Educational use only — not medical advice.** This tool retrieves from "
    "indexed clinical guidelines; it does not diagnose, prescribe, or handle "
    "emergencies. Always consult a qualified healthcare professional."
)
BADGE = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}

st.set_page_config(page_title="Healthcare Knowledge Navigator", page_icon="⚕️")


def recent_history(messages: list, max_msgs: int = 4) -> list:
    """Build a compact {role, content} history from the last few turns, so the
    backend can resolve a follow-up like 'explain this' into a standalone question."""
    hist = []
    for m in messages[-max_msgs:]:
        if m["role"] == "user":
            hist.append({"role": "user", "content": m["content"]})
        else:
            answer = (m.get("data") or {}).get("answer")
            if answer:
                hist.append({"role": "assistant", "content": answer})
    return hist


def call_api(question: str, history: list | None = None) -> dict | None:
    """POST the question (+ recent history) to the backend; return JSON or None."""
    try:
        resp = requests.post(f"{API_URL}/query",
                             json={"question": question, "history": history or []},
                             timeout=300)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException:
        return None


def render_answer(data: dict | None) -> None:
    if data is None:
        st.error(f"Couldn't reach the backend at {API_URL}. "
                 "Is it running?  `uvicorn api:app --port 8000`")
        return

    if data.get("blocked"):
        st.error(data["answer"])          # refusal (emergency / diagnosis / prescribing)
        return

    if data.get("resolved_question"):
        st.caption(f"_Interpreted your follow-up as: “{data['resolved_question']}”_")

    level = data.get("confidence_level")
    if level:
        st.markdown(f"**Confidence:** {BADGE.get(level, '')} {level} "
                    f"&nbsp;·&nbsp; relevance {data['confidence_score']:.2f}")

    st.markdown(data["answer"])

    if data.get("truncated"):
        st.warning("⚠️ This answer may be incomplete — it reached the length limit. "
                   "Try a more specific question.")
    if data.get("conflict"):
        st.info(data["conflict"])
    for warning in data.get("temporal_warnings", []):
        st.warning(warning)

    citations = data.get("citations", [])
    if citations:
        st.markdown("**Sources**")
        for c in citations:
            stale = " ⚠️ >5yr" if not c["is_current"] else ""
            date = f" · {c['date']}" if c["date"] else ""
            with st.expander(f"[{c['index']}] {c['title']} — p.{c['page']}{date}{stale}"):
                st.caption(f"…{c['snippet']}…")
                if c.get("source_url"):
                    st.markdown(f"[Open source]({c['source_url']})")

    if data.get("disclaimer"):
        st.caption(data["disclaimer"])


# ── Page ──────────────────────────────────────────────────────────────────────
st.title("⚕️ Healthcare Knowledge Navigator")
st.warning(DISCLAIMER_BANNER)   # re-rendered every run -> a persistent banner

with st.sidebar:
    # Backend connection status only (no instructions).
    try:
        ok = requests.get(f"{API_URL}/health", timeout=5).json().get("assistant_loaded")
        if ok:
            st.success("Backend: connected ✅")
        else:
            st.warning("Backend: starting…")
    except requests.exceptions.RequestException:
        st.error("Backend: offline ❌")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay conversation history.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            render_answer(msg["data"])

# Handle new input.
if prompt := st.chat_input("Ask a medical question…"):
    history = recent_history(st.session_state.messages)   # before appending current turn
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Searching guidelines and generating a grounded answer…"):
            data = call_api(prompt, history)
        render_answer(data)
    st.session_state.messages.append({"role": "assistant", "data": data})
