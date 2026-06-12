"""
Healthcare Knowledge Navigator — Phase 2
Multi-document ingestion with metadata.

Processes EVERY PDF in docs/, attaches rich metadata to each chunk (title, URL,
publication date, evidence level, specialty, currency), embeds them, and persists
the result to ChromaDB so retrieval can later show *where* each answer came from.

Metadata for each file comes from docs/sources.json (keyed by filename). A file
not listed there is still ingested — but with 'unknown' metadata and a warning,
so you know to add it.

Run:
    conda activate healthcare-rag
    python -m src.ingest          # (re)build the vector store from docs/, then search
"""
import json
import shutil
import sys
from datetime import datetime

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import config

# Windows consoles default to cp1252 and crash on the ⚠️/• characters below.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SOURCES_MANIFEST = config.DOCS_DIR / "sources.json"


# ── Metadata ──────────────────────────────────────────────────────────────────

def load_manifest() -> dict:
    """Return {filename: metadata} from docs/sources.json. Helper keys (starting
    with '__') are skipped so we can keep documentation inside the JSON."""
    if not SOURCES_MANIFEST.exists():
        return {}
    data = json.loads(SOURCES_MANIFEST.read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if not k.startswith("__")}


def is_current(publication_date: str) -> bool:
    """True if the source is within config.MAX_SOURCE_AGE_YEARS of today.
    Unknown/unparseable dates are treated as current (not flagged stale)."""
    try:
        year = int(str(publication_date)[:4])
    except (ValueError, TypeError):
        return True
    return (datetime.now().year - year) <= config.MAX_SOURCE_AGE_YEARS


def metadata_for(filename: str, manifest: dict) -> dict:
    """Build per-chunk metadata for one PDF. Every value is Chroma-safe
    (str/bool) — Chroma rejects None, so we use '' / 'unknown' instead."""
    entry = manifest.get(filename, {})
    pub = entry.get("publication_date", "unknown")
    return {
        "source_filename": filename,
        "source_title": entry.get("source_title", filename),
        "source_url": entry.get("source_url", ""),
        "publication_date": pub,
        "evidence_level": entry.get("evidence_level", "unknown"),
        "medical_specialty": entry.get("medical_specialty", "unknown"),
        "is_current": is_current(pub),
    }


# ── Load + chunk + tag ────────────────────────────────────────────────────────

def load_and_chunk() -> list:
    """Load all PDFs in docs/, split into chunks, and stamp metadata on each."""
    pdf_files = sorted(config.DOCS_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs in '{config.DOCS_DIR}'. Add documents and retry.")

    manifest = load_manifest()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks = []
    print(f"\n  Ingesting {len(pdf_files)} document(s) from {config.DOCS_DIR}/\n")
    for path in pdf_files:
        meta = metadata_for(path.name, manifest)
        pages = PyPDFLoader(str(path)).load()
        chunks = splitter.split_documents(pages)
        for chunk in chunks:
            chunk.metadata.update(meta)   # PyPDF's own 'page' key is preserved
        all_chunks.extend(chunks)

        unlisted = "" if path.name in manifest else "  ⚠️ not in sources.json"
        stale = "" if meta["is_current"] else f"  ⚠️ STALE >{config.MAX_SOURCE_AGE_YEARS}yr"
        print(f"  • {path.name}: {len(chunks):>4} chunks | "
              f"{meta['medical_specialty']} | {meta['publication_date']}{stale}{unlisted}")

    print(f"\n  Total: {len(all_chunks)} chunks from {len(pdf_files)} document(s)")
    return all_chunks


# ── Vector store ──────────────────────────────────────────────────────────────

def build_vectorstore(chunks: list, rebuild: bool) -> Chroma:
    """Embed chunks into ChromaDB. With rebuild=True the old store is wiped first
    — necessary whenever the corpus changes, so new docs actually get embedded."""
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    if rebuild and config.CHROMA_DIR.exists():
        print("\n  Rebuilding: clearing previous vector store...")
        try:
            shutil.rmtree(config.CHROMA_DIR)
        except PermissionError:
            sys.exit(
                "\n  ✗ Could not clear chroma_db/ — it is locked by another process.\n"
                "    A query session (phase1_starter_code.py or src.ingest) is still\n"
                "    open somewhere. Close it (type 'quit'), then run this again.\n"
                "    ChromaDB's local store allows only ONE process at a time."
            )

    if config.CHROMA_DIR.exists() and any(config.CHROMA_DIR.iterdir()):
        print("  Loading existing ChromaDB...")
        store = Chroma(persist_directory=str(config.CHROMA_DIR), embedding_function=embeddings)
    else:
        print("  Embedding chunks (rebuild takes ~1 min per few hundred chunks)...")
        store = Chroma.from_documents(chunks, embeddings, persist_directory=str(config.CHROMA_DIR))

    print(f"  {store._collection.count()} chunks indexed in {config.CHROMA_DIR}/\n")
    return store


def ingest(rebuild: bool = True) -> Chroma:
    """Full Phase 2 pipeline: load+chunk+tag all PDFs, then (re)build the store."""
    return build_vectorstore(load_and_chunk(), rebuild=rebuild)


# ── Retrieval demo (no LLM — this phase is about ingestion + metadata) ─────────

def search_demo(store: Chroma) -> None:
    print("=" * 64)
    print("  Retrieval demo — enter a query to see matching chunks + metadata")
    print("  (type 'quit' to exit)")
    print("=" * 64)
    while True:
        query = input("\nSearch: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue
        results = store.similarity_search_with_relevance_scores(query, k=config.TOP_K)
        for i, (doc, score) in enumerate(results, 1):
            m = doc.metadata
            stale = "  ⚠️ STALE" if m.get("is_current") is False else ""
            page = (m.get("page") or 0) + 1
            print(f"\n  [{i}] score={score:.3f} | {m.get('source_title')}")
            print(f"      p.{page} · {m.get('publication_date')} · "
                  f"{m.get('medical_specialty')} · {m.get('evidence_level')}{stale}")
            snippet = " ".join(doc.page_content.split())[:200]
            print(f"      \"{snippet}...\"")


if __name__ == "__main__":
    print("=" * 64)
    print("  Healthcare Knowledge Navigator — Phase 2: Ingestion")
    print("=" * 64)
    vectorstore = ingest(rebuild=True)
    search_demo(vectorstore)
