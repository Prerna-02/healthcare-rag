"""
Healthcare Knowledge Navigator — Phase 1
Stack: LangChain · Ollama (qwen3:8b, local) · HuggingFace Embeddings · ChromaDB

Prerequisites:
  1. conda env create -f environment.yml && conda activate healthcare-rag
  2. ollama pull qwen3:8b   (Ollama must be running: ollama serve)
  3. Drop PDFs into ./docs/
  4. python phase1_starter_code.py
"""

import re
import sys

# Windows consoles default to cp1252, which cannot encode the Unicode box-drawing
# (─) and emoji (🟢🟡🔴) characters printed below — that raises UnicodeEncodeError.
# Force UTF-8 so the script runs on any platform.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

import config


# ── Document loading ─────────────────────────────────────────────────────────

def load_documents() -> list:
    pdf_files = list(config.DOCS_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs in '{config.DOCS_DIR}'. Add documents and retry.")

    docs = []
    for path in pdf_files:
        print(f"  Loading: {path.name}")
        pages = PyPDFLoader(str(path)).load()
        for page in pages:
            page.metadata.update({
                "source_filename": path.name,
                "evidence_level": "unknown",
                "is_current": True,
            })
        docs.extend(pages)

    print(f"  {len(docs)} pages from {len(pdf_files)} file(s)\n")
    return docs


# ── Chunking ─────────────────────────────────────────────────────────────────

def chunk_documents(docs: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"  {len(chunks)} chunks (size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP})\n")
    return chunks


# ── Vector store ─────────────────────────────────────────────────────────────

def build_vectorstore(chunks: list) -> Chroma:
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    if config.CHROMA_DIR.exists() and any(config.CHROMA_DIR.iterdir()):
        print("  Loading existing ChromaDB...")
        store = Chroma(persist_directory=str(config.CHROMA_DIR), embedding_function=embeddings)
    else:
        print("  Embedding chunks — first run takes ~1 min...")
        store = Chroma.from_documents(chunks, embeddings, persist_directory=str(config.CHROMA_DIR))

    print(f"  {store._collection.count()} chunks indexed\n")
    return store


# ── Retrieval with confidence scoring ────────────────────────────────────────
# Real similarity scores from ChromaDB (0.0 = no match → 1.0 = perfect).
# If you see consistent LOW scores, increase CHUNK_SIZE in config.py and
# delete chroma_db/ to force a rebuild.

def retrieve_with_confidence(store: Chroma, query: str):
    results = store.similarity_search_with_relevance_scores(query, k=config.TOP_K)
    docs    = [doc for doc, _ in results]
    scores  = [score for _, score in results]
    avg     = sum(scores) / len(scores) if scores else 0.0

    if avg > config.CONF_HIGH:
        level = "HIGH"
    elif avg > config.CONF_MEDIUM:
        level = "MEDIUM"
    else:
        level = "LOW"

    return docs, scores, level, avg


# ── Guardrails ────────────────────────────────────────────────────────────────

_EMERGENCY_RE = re.compile(
    r"chest pain.+right now|i('m| am) (having|experiencing)|cant? breathe|call.*(911|999)",
    re.IGNORECASE,
)
_REFUSE_RE = re.compile(
    r"diagnose (me|my patient)|what (do i|does my patient) have"
    r"|should i prescribe|what dose should i (give|prescribe)",
    re.IGNORECASE,
)


def classify_query(query: str) -> tuple[str, str | None]:
    if _EMERGENCY_RE.search(query):
        return "block", (
            "⚠️  This sounds like a medical emergency. "
            "Call emergency services (911/999) immediately. "
            "This system cannot provide emergency guidance."
        )
    if _REFUSE_RE.search(query):
        return "block", (
            "This system is for educational reference only. "
            "It cannot assist with patient-specific diagnosis or prescribing."
        )
    if re.search(r"\bmy patient\b", query, re.IGNORECASE):
        return "warn", None
    return "ok", None


# ── Medical system prompt ─────────────────────────────────────────────────────

_PROMPT = PromptTemplate.from_template("""/no_think
You are a Healthcare Knowledge Assistant for medical professionals and students.
Answer ONLY from the context below. If the answer is not in the context, say:
"I don't have sufficient evidence in my knowledge base to answer this reliably."

Rules:
- Never diagnose a patient or prescribe treatment for an individual.
- Cite the source document for every factual claim.
- If evidence is limited or conflicting, say so explicitly.

Context:
{context}

Question: {question}

Answer (with citations):
""")

# The disclaimer is appended in code (see ask()), NOT left to the LLM — that
# guarantees it appears on every generated answer, as the ethics spec requires.
_DISCLAIMER = (
    "Disclaimer: This information is for educational purposes only and does not "
    "constitute medical advice. Always consult a qualified healthcare "
    "professional for clinical decisions."
)


# ── RAG chain ─────────────────────────────────────────────────────────────────

def build_chain(store: Chroma):
    llm       = ChatOllama(
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        num_predict=config.LLM_NUM_PREDICT,
    )
    retriever = store.as_retriever(search_kwargs={"k": config.TOP_K})

    chain = (
        {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
         "question": RunnablePassthrough()}
        | _PROMPT
        | llm
        | StrOutputParser()
    )
    return chain


# ── Citation formatter ────────────────────────────────────────────────────────

def format_citations(docs: list) -> str:
    # Uses the rich metadata added in Phase 2 (src/ingest.py) when present, and
    # falls back gracefully to just the filename if the store was built without it.
    seen, lines = set(), []
    for i, doc in enumerate(docs, 1):
        m = doc.metadata
        key = f"{m.get('source_filename')}:{m.get('page')}"
        if key in seen:
            continue
        seen.add(key)
        title = m.get("source_title") or m.get("source_filename")
        page = (m.get("page") or 0) + 1
        date = m.get("publication_date", "")
        datestr = f", {date}" if date and date != "unknown" else ""
        stale = "  ⚠️ source >5yr old — verify against current guidelines" \
            if m.get("is_current") is False else ""
        lines.append(f"  [{i}] {title} — p.{page}{datestr}{stale}")
    return "\n".join(lines)


# ── Main Q&A ─────────────────────────────────────────────────────────────────

def ask(store: Chroma, chain, question: str):
    print(f"\n{'─' * 60}")
    print(f"Q: {question}")
    print('─' * 60)

    status, msg = classify_query(question)
    if status == "block":
        print(f"\n🚫 {msg}\n")
        return
    if status == "warn":
        print("⚠️  Patient-specific query — providing general educational information only.\n")

    docs, scores, level, avg = retrieve_with_confidence(store, question)

    badge = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}[level]
    print(f"Retrieval confidence: {badge} {level}  (avg={avg:.3f}, scores={[round(s,3) for s in scores]})")
    if level == "LOW":
        print("  ↳ Try increasing CHUNK_SIZE in config.py, then delete chroma_db/ and rerun.")

    answer = chain.invoke(question)
    print(f"\n{answer}")
    print(f"\n---\n{_DISCLAIMER}")
    print(f"\nSources:\n{format_citations(docs)}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Healthcare Knowledge Navigator — Phase 1")
    print(f"  Model: {config.LLM_MODEL}  |  Embeddings: {config.EMBED_MODEL}")
    print("=" * 60 + "\n")

    # If a vector store already exists (e.g. built by `python -m src.ingest` in
    # Phase 2), load it directly and skip re-reading every PDF. Only parse PDFs
    # when there is nothing indexed yet.
    if config.CHROMA_DIR.exists() and any(config.CHROMA_DIR.iterdir()):
        store = build_vectorstore([])
    else:
        store = build_vectorstore(chunk_documents(load_documents()))
    chain  = build_chain(store)
    print("Ready.\n")

    test_questions = [
        "What is the first-line treatment for hypertension?",
        "I'm having chest pain right now, what should I do?",
        "Diagnose my patient who has fatigue and weight gain.",
        "What are the contraindications for metformin?",
        "What is the recommended management of alien flu syndrome?",
    ]
    for q in test_questions:
        ask(store, chain, q)

    print("\n" + "=" * 60)
    print("  Interactive mode — type 'quit' to exit")
    print("=" * 60)
    while True:
        user_q = input("\nQuestion: ").strip()
        if user_q.lower() in ("quit", "exit", "q"):
            break
        if user_q:
            ask(store, chain, user_q)
