from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
DOCS_DIR    = Path("./docs")
CHROMA_DIR  = Path("./chroma_db")
EVAL_DIR    = Path("./eval")

# ── LLM — Ollama (local, no API key needed) ───────────────────────────────────
# Requires Ollama running: https://ollama.com
# Pull model: ollama pull qwen3:8b
# Qwen3 has a "thinking mode" (<think>...</think> reasoning tokens). On this
# stack (langchain-ollama 0.3.2 + Ollama) it is already OFF by default — verified
# empirically. We additionally send Qwen3's '/no_think' switch in the prompt as
# an explicit, portable safeguard. (Note: a ChatOllama(think=...) kwarg is
# silently ignored on this version, so we do NOT rely on it.)
LLM_MODEL        = "qwen3:8b"
LLM_TEMPERATURE  = 0       # 0 = deterministic: same question -> same answer
LLM_NUM_PREDICT  = 512     # max tokens generated per answer; caps CPU latency.
                           # Raise if answers get cut off mid-sentence.

# ── Embeddings — HuggingFace (free, CPU-optimised) ────────────────────────────
# Downloads automatically on first run (~22 MB, ~30 seconds)
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 512
CHUNK_OVERLAP = 50

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K = 5           # final chunks handed to the LLM per query
BM25_WEIGHT   = 0.4  # Phase 3 hybrid weights: how much keyword (BM25) vs
VECTOR_WEIGHT = 0.6  # semantic (vector) search each contributes. Must sum to 1.

# ── Re-ranking (Phase 3) ──────────────────────────────────────────────────────
# Free, local cross-encoder that re-scores candidates against the question.
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# Each base retriever fetches this many candidates; the reranker then trims to TOP_K.
FETCH_K = 10

# ── Confidence thresholds (similarity score 0.0–1.0) ─────────────────────────
CONF_HIGH   = 0.75
CONF_MEDIUM = 0.50

# ── Source freshness ──────────────────────────────────────────────────────────
MAX_SOURCE_AGE_YEARS = 5  # warn if source is older than this

# ── Guardrails (Phase 4) ──────────────────────────────────────────────────────
# The intent classifier only needs to emit a one-word category, so a tiny budget.
CLASSIFIER_NUM_PREDICT = 16

# Refusal categories -> the exact message shown. Edit policy HERE — never bury it
# in code. The classifier returns one of these keys (or "general", which is allowed).
REFUSAL_MESSAGES = {
    "emergency": (
        "⚠️  This sounds like a medical emergency. Please call your local emergency "
        "number (e.g. 911 / 999 / 112) immediately. This system cannot provide "
        "emergency medical guidance."
    ),
    "diagnosis": (
        "🚫 This system offers general, educational information only — it cannot "
        "diagnose a specific person or interpret an individual's symptoms. Please "
        "consult a qualified healthcare professional."
    ),
    "prescribing": (
        "🚫 This system cannot recommend a medication or dose for an individual "
        "patient. Prescribing decisions must be made by a qualified clinician."
    ),
}

# Stamped on every generated answer (refusals are themselves safe, so they skip it).
DISCLAIMER = (
    "Disclaimer: This information is for educational purposes only and does not "
    "constitute medical advice. Always consult a qualified healthcare professional "
    "for clinical decisions."
)
