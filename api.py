
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from src.chain import Assistant

# The Assistant holds heavy state (embeddings, vector store, cross-encoder), so we
# build it ONCE at startup and reuse it for every request — not per-request.
_state: dict = {}


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
    history: list[Turn] = []   # recent turns, so follow-ups can be resolved


@app.get("/health")
def health():
    return {"status": "ok", "assistant_loaded": "assistant" in _state}


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
    return _state["assistant"].answer_structured(question, history=history)
