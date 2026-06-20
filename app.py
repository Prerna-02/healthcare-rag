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


def call_api(question: str, history: list | None = None,
             session_id: str | None = None) -> dict | None:
    """POST the question (+ recent history, + optional uploaded-doc session) to the
    backend; return JSON or None if unreachable."""
    try:
        resp = requests.post(f"{API_URL}/query",
                             json={"question": question, "history": history or [],
                                   "session_id": session_id},
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
    st.subheader("Your documents")
    st.caption("⚠️ Educational PDFs only — **do not upload patient data / PHI.** "
               "Files are processed in memory for this session only and are never stored.")
    uploads = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)
    if uploads:
        names = sorted(f.name for f in uploads)
        # Re-index only when the set of files changes (added/removed).
        if st.session_state.get("uploaded_names") != names:
            with st.spinner(f"Indexing {len(uploads)} document(s)…"):
                try:
                    files = [("files", (f.name, f.getvalue(), "application/pdf")) for f in uploads]
                    resp = requests.post(f"{API_URL}/upload", files=files, timeout=600)
                    resp.raise_for_status()
                    info = resp.json()
                    st.session_state.session_id = info["session_id"]
                    st.session_state.uploaded_names = names
                    st.success(f"Indexed {info['n_docs']} doc(s) · {info['n_chunks']} chunks")
                except requests.exceptions.RequestException as exc:
                    st.error(f"Upload failed: {exc}")

    # Source toggle — only offer the uploaded docs once at least one exists.
    if st.session_state.get("session_id") and st.session_state.get("uploaded_names"):
        n = len(st.session_state.uploaded_names)
        st.session_state.source = st.radio(
            "Answer from:", ["Guideline corpus", f"Uploaded: {n} document(s)"])
    else:
        st.session_state.source = "Guideline corpus"

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
    # Route to the uploaded doc only when that source is selected.
    use_upload = st.session_state.get("source", "").startswith("Uploaded")
    session_id = st.session_state.get("session_id") if use_upload else None
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Searching and generating a grounded answer…"):
            data = call_api(prompt, history, session_id)
        render_answer(data)
    st.session_state.messages.append({"role": "assistant", "data": data})
