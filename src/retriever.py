"""
Healthcare Knowledge Navigator — Phase 3
Enhanced retrieval: hybrid search (semantic + BM25) + cross-encoder re-ranking.

WHY this exists:
  - Pure SEMANTIC (vector) search understands *meaning* but can miss an *exact*
    term — a drug name, an acronym, a Latin species, a lab value.
  - BM25 is classic *keyword* search — the opposite strength.
  - We blend them (EnsembleRetriever), then re-score the blended candidates with a
    cross-encoder (CrossEncoderReranker) that reads each chunk *together with* the
    question for a sharper final ranking.

Run a side-by-side comparison:
    conda activate healthcare-rag
    python -m src.retriever          # semantic vs hybrid vs hybrid+rerank
"""
import sys

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker

import config
from src.ingest import load_and_chunk

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ── Building blocks ───────────────────────────────────────────────────────────

def _embeddings() -> HuggingFaceEmbeddings:
    """Same embedder used at ingestion time — must match, or vectors won't align."""
    return HuggingFaceEmbeddings(
        model_name=config.EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_store() -> Chroma:
    """Open the ChromaDB built by `python -m src.ingest`. Errors clearly if absent."""
    if not (config.CHROMA_DIR.exists() and any(config.CHROMA_DIR.iterdir())):
        raise FileNotFoundError(
            "No vector store found. Run `python -m src.ingest` first to build it."
        )
    return Chroma(persist_directory=str(config.CHROMA_DIR), embedding_function=_embeddings())


def semantic_retriever(store: Chroma):
    """Plain vector search — finds chunks whose MEANING is closest to the query.
    This is exactly what Phase 1/2 used. It's our baseline to improve on."""
    return store.as_retriever(search_kwargs={"k": config.TOP_K})


def hybrid_retriever(store: Chroma, chunks: list) -> EnsembleRetriever:
    """Blend keyword (BM25) + semantic (vector) search.

    BM25Retriever builds an in-memory keyword index from the chunk *text*, so it
    needs the chunks themselves (not just their vectors). EnsembleRetriever runs
    BOTH retrievers and fuses their ranked lists (Reciprocal Rank Fusion), weighted
    by BM25_WEIGHT / VECTOR_WEIGHT from config."""
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = config.FETCH_K                                  # keyword candidates
    semantic = store.as_retriever(search_kwargs={"k": config.FETCH_K})  # vector candidates
    return EnsembleRetriever(
        retrievers=[bm25, semantic],
        weights=[config.BM25_WEIGHT, config.VECTOR_WEIGHT],
    )


def with_reranker(base_retriever, encoder: HuggingFaceCrossEncoder | None = None):
    """Wrap any retriever so its candidates are re-scored by a cross-encoder.

    A cross-encoder reads (question, chunk) TOGETHER and outputs a precise
    relevance score — far more accurate than the first-pass similarity, but too
    slow to run on the whole corpus. So we only run it on the handful of
    candidates the base retriever already shortlisted, then keep the top TOP_K.

    Pass an existing `encoder` to reuse one model (Phase 4 also uses it to compute
    answer confidence) instead of loading the ~80 MB model twice."""
    encoder = encoder or HuggingFaceCrossEncoder(model_name=config.RERANK_MODEL)
    reranker = CrossEncoderReranker(model=encoder, top_n=config.TOP_K)
    return ContextualCompressionRetriever(
        base_compressor=reranker, base_retriever=base_retriever
    )


def get_retriever(store: Chroma, chunks: list, encoder: HuggingFaceCrossEncoder | None = None):
    """The production retriever for later phases: hybrid candidates, then reranked.
    Fast-and-broad first pass (hybrid) → slow-and-precise final pass (reranker)."""
    return with_reranker(hybrid_retriever(store, chunks), encoder=encoder)


# ── Comparison demo ───────────────────────────────────────────────────────────

def _show(name: str, docs: list) -> None:
    print(f"\n  ── {name} ".ljust(64, "─"))
    for i, doc in enumerate(docs[: config.TOP_K], 1):
        m = doc.metadata
        src = (m.get("source_title") or m.get("source_filename") or "?")[:46]
        page = (m.get("page") or 0) + 1
        snippet = " ".join(doc.page_content.split())[:110]
        print(f"  {i}. {src} (p.{page})")
        print(f"     \"{snippet}...\"")


def compare(query: str, retrievers: dict) -> None:
    """Run the same query through each strategy and print their top chunks, so the
    difference in WHAT gets retrieved (and in what order) is visible."""
    print("\n" + "=" * 64)
    print(f"  QUERY: {query}")
    print("=" * 64)
    for name, retriever in retrievers.items():
        _show(name, retriever.invoke(query))


if __name__ == "__main__":
    print("=" * 64)
    print("  Healthcare Knowledge Navigator — Phase 3: Retrieval comparison")
    print("=" * 64)

    store = load_store()
    print("  Loading corpus text for BM25 (reads the PDFs)...")
    chunks = load_and_chunk()
    print("  Preparing retrievers (cross-encoder downloads ~80 MB on first run)...")

    hybrid = hybrid_retriever(store, chunks)          # built once, reused below
    strategies = {
        "SEMANTIC only (baseline)": semantic_retriever(store),
        "HYBRID (BM25 + semantic)": hybrid,
        "HYBRID + RERANK": with_reranker(hybrid),
    }

    while True:
        q = input("\nQuery to compare (or 'quit'): ").strip()
        if q.lower() in ("quit", "exit", "q"):
            break
        if q:
            compare(q, strategies)
