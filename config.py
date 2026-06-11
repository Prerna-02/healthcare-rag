from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
DOCS_DIR    = Path("./docs")
CHROMA_DIR  = Path("./chroma_db")
EVAL_DIR    = Path("./eval")

# ── LLM — Ollama (local, no API key needed) ───────────────────────────────────
# Requires Ollama running: https://ollama.com
# Pull model: ollama pull qwen3:8b
# think=False disables Qwen3 reasoning tokens (<think>...</think>) which would
# corrupt answer formatting and add 10-30s latency. Always set this for RAG.
LLM_MODEL        = "qwen3:8b"
LLM_TEMPERATURE  = 0

# ── Embeddings — HuggingFace (free, CPU-optimised) ────────────────────────────
# Downloads automatically on first run (~22 MB, ~30 seconds)
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 512
CHUNK_OVERLAP = 50

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K = 5           # chunks returned per query
BM25_WEIGHT   = 0.4  # Phase 3: hybrid search weights
VECTOR_WEIGHT = 0.6

# ── Confidence thresholds (similarity score 0.0–1.0) ─────────────────────────
CONF_HIGH   = 0.75
CONF_MEDIUM = 0.50

# ── Source freshness ──────────────────────────────────────────────────────────
MAX_SOURCE_AGE_YEARS = 5  # warn if source is older than this
