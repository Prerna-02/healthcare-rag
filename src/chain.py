"""
Healthcare Knowledge Navigator — Phase 4
The guarded RAG pipeline: input guardrails → hybrid+rerank retrieval → grounded
generation (Layer C) → output guardrails.

Flow for every question (see src/guardrails.py for the safety pieces):

    question
      │
      ▼  classify_query()  ── emergency/diagnosis/prescribing? → REFUSE, stop here
      ▼  (general)
      ▼  retriever.invoke()      hybrid (BM25+semantic) + cross-encoder rerank  [Phase 3]
      ▼  cross-encoder scores  → compute_confidence()
      ▼  ollama.chat(think=False) with the medical system prompt (Layer C)
      ▼  assemble: answer + confidence + citations + staleness + conflict + disclaimer

Run the interactive assistant:
    conda activate healthcare-rag
    python -m src.chain
"""
import math
import sys

import ollama
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

import config
from src import guardrails
from src import retriever as R

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── Layer C: the medical system prompt (behavioural backstop during generation) ─
# This is the policy the model must follow WHILE writing. It is the last line of
# defence after the Layer A/B input gates.
SYSTEM_PROMPT = """You are a Healthcare Knowledge Assistant for medical professionals and students.

STRICT RULES:
1. Answer ONLY from the provided context. If the context does not contain the answer, say:
   "I don't have sufficient evidence in my knowledge base to answer this reliably."
   Never use outside knowledge to answer a medical question.
2. NEVER diagnose a specific person, interpret an individual's symptoms, or recommend a
   drug or dose for an individual.
3. Cite the source for every factual claim.
4. If the evidence is limited or sources conflict, say so explicitly. Do not overstate certainty."""


# ── Retrieval relevance → confidence ──────────────────────────────────────────

def _relevance_scores(question: str, docs: list, encoder: HuggingFaceCrossEncoder) -> list[float]:
    """Score each final chunk against the question with the cross-encoder, mapped
    to 0–1 via a sigmoid. These are the 'real' relevance numbers confidence uses."""
    if not docs:
        return []
    raw = encoder.score([(question, doc.page_content) for doc in docs])
    return [1.0 / (1.0 + math.exp(-float(s))) for s in raw]


# ── Generation (Layer C, think=False) ─────────────────────────────────────────

def generate(question: str, context: str) -> tuple[str, bool]:
    """Generate the grounded answer. think=False keeps Qwen3 from wasting its token
    budget on hidden reasoning (verified in Phase 4 probing).

    Returns (answer, truncated). `truncated` is True if generation stopped because
    it hit the length cap (done_reason == 'length') rather than finishing naturally
    — so a cut-off answer is never silently presented as complete."""
    resp = ollama.chat(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",
             "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer (with citations):"},
        ],
        think=False,
        options={"temperature": config.LLM_TEMPERATURE, "num_predict": config.LLM_NUM_PREDICT},
    )
    return resp.message.content.strip(), resp.done_reason == "length"


# When the model can't answer from context it emits this phrase (per SYSTEM_PROMPT).
# We use it to suppress citations/warnings — there is no grounded answer to cite.
_NO_ANSWER_MARKER = "sufficient evidence"


# ── The assistant ─────────────────────────────────────────────────────────────

class Assistant:
    """Holds the loaded models/retriever so a session answers many questions
    without re-loading anything."""

    def __init__(self):
        self.store = R.load_store()
        print("  Loading corpus text for BM25...")
        self.chunks = R.load_and_chunk()
        print("  Loading cross-encoder (shared by reranker + confidence)...")
        self.encoder = HuggingFaceCrossEncoder(model_name=config.RERANK_MODEL)
        self.retriever = R.get_retriever(self.store, self.chunks, encoder=self.encoder)

    def answer(self, question: str) -> str:
        # INPUT guardrails (Layer A + B) — refuse before spending any retrieval/LLM.
        category = guardrails.classify_query(question)
        if guardrails.is_blocked(category):
            return f"🚫 {guardrails.refusal_message(category)}"

        # Retrieve (Phase 3 hybrid + rerank) and score confidence from real relevance.
        docs = self.retriever.invoke(question)
        scores = _relevance_scores(question, docs, self.encoder)
        level, avg = guardrails.compute_confidence(scores)

        # Generate the grounded answer (Layer C).
        context = "\n\n".join(doc.page_content for doc in docs)
        body, truncated = generate(question, context)

        # OUTPUT guardrails — annotate the answer.
        badge = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}[level]
        parts = [f"Confidence: {badge} {level}  (avg relevance {avg:.2f})", "", body]

        if truncated:
            parts += ["", "⚠️  This answer may be incomplete — it reached the length "
                          "limit. Ask a more specific question, or raise LLM_NUM_PREDICT."]

        # Only attach citations/conflict/staleness when there IS a grounded answer.
        # For an "insufficient evidence" non-answer, listing sources would wrongly
        # imply evidence exists.
        if _NO_ANSWER_MARKER not in body.lower():
            conflict = guardrails.detect_conflicts(docs)
            if conflict:
                parts += ["", conflict]
            for warning in guardrails.temporal_warnings(docs):
                parts += ["", warning]
            parts += ["", "Sources:", guardrails.format_citations(docs)]

        parts += ["", "—" * 3, config.DISCLAIMER]
        return "\n".join(parts)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 64)
    print("  Healthcare Knowledge Navigator — Phase 4: Guarded Assistant")
    print(f"  Model: {config.LLM_MODEL}  |  Embeddings: {config.EMBED_MODEL}")
    print("=" * 64)

    assistant = Assistant()
    print("\nReady. Ask a medical question, or type 'quit'.\n")
    while True:
        question = input("Question: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if question:
            print(f"\n{assistant.answer(question)}\n" + "─" * 64)
