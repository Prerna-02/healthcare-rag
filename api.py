
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

from src.chain import Assistant

# The Assistant holds heavy state (embeddings, vector store, cross-encoder), so we
# build it ONCE at startup and reuse it for every request — not per-request.
_state: dict = {}
# Uploaded documents -> their in-memory retriever, keyed by a session id. These
# live only in this process's RAM and are gone on restart (never persisted).
_sessions: dict = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    print("Loading assistant (models + vector store)... first start takes ~1 min.")
    _state["assistant"] = Assistant()
    print("Assistant ready — API is live.")
    yield
    _state.clear()


app = FastAPI(title="Healthcare Knowledge Navigator API", version="1.0", lifespan=lifespan)


class Turn(BaseModel):
    role: str       # "user" or "assistant"
    content: str


class QueryRequest(BaseModel):
    question: str
    history: list[Turn] = []        # recent turns, so follow-ups can be resolved
    session_id: str | None = None   # if set, answer from the uploaded doc, not the corpus


@app.get("/health")
def health():
    return {"status": "ok", "assistant_loaded": "assistant" in _state}


@app.post("/upload")
def upload(file: UploadFile = File(...)):
    """Ingest an uploaded PDF into an in-memory, session-only store and return a
    session_id the client passes to /query to ask about that document."""
    pdf_bytes = file.file.read()
    retriever, n_chunks = _state["assistant"].build_session_retriever(pdf_bytes, file.filename)
    session_id = uuid.uuid4().hex
    _sessions[session_id] = retriever
    return {"session_id": session_id, "filename": file.filename, "n_chunks": n_chunks}


@app.post("/query")
def query(req: QueryRequest):
    """Run one question through the guarded RAG pipeline and return structured JSON
    (answer, confidence, citations, staleness warnings, conflict flag, disclaimer)."""
    question = req.question.strip()
    if not question:
        return {"blocked": False, "answer": "Please enter a question.",
                "confidence_level": None, "confidence_score": None, "citations": [],
                "temporal_warnings": [], "conflict": None, "truncated": False,
                "disclaimer": None}
    history = [{"role": t.role, "content": t.content} for t in req.history]
    # If a session_id is supplied, answer from that uploaded doc; else the corpus.
    retriever = _sessions.get(req.session_id) if req.session_id else None
    return _state["assistant"].answer_structured(question, history=history, retriever=retriever)
