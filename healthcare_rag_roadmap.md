# Healthcare Knowledge Navigator — Project Roadmap
> Stack: Python 3.11 · LangChain · Ollama (qwen3:8b) · HuggingFace Embeddings · ChromaDB · RAGAS · Streamlit

---

## Overview

| Phase | Focus | Duration | Key Deliverable |
|-------|-------|----------|-----------------|
| 1 | Foundation — Basic RAG | Week 1 | Single-PDF Q&A working end-to-end |
| 2 | Document Pipeline | Week 2 | Multi-document corpus with metadata |
| 3 | Enhanced Retrieval | Week 3 | Hybrid search + re-ranking |
| 4 | Ethical Guardrails | Week 4 | Safe medical system prompt + confidence scoring |
| 5 | RAGAS Evaluation | Week 5 | Evaluation dashboard with all 5 metrics |
| 6 | UI + Portfolio Polish | Week 6 | Streamlit app, README, demo video |

---

## Phase 1 — Foundation (Week 1)

**Goal:** Understand the core RAG loop. Get a question answered from a single PDF.

### What you'll learn
- How embeddings represent meaning as vectors
- What a vector store is and how similarity search works
- How LangChain chains retrieval + generation together

### Prerequisites
```bash
# Create conda environment
conda env create -f environment.yml
conda activate healthcare-rag

# Start Ollama (keep this running in a separate terminal)
ollama serve

# Verify qwen3:8b is available
ollama list
```

### Tasks
- [ ] Set up conda environment (Python 3.11) using `environment.yml`
- [ ] Download one clinical guideline PDF into `./docs/` (see PDF sources below)
- [ ] Load PDF using `PyPDFLoader`
- [ ] Chunk it with `RecursiveCharacterTextSplitter` (chunk_size=512, overlap=50)
- [ ] Embed chunks with `HuggingFaceEmbeddings` (all-MiniLM-L6-v2, free, CPU)
- [ ] Store in local ChromaDB (persists to `./chroma_db/`)
- [ ] Build RAG chain with `ChatOllama(model="qwen3:8b", think=False)`
- [ ] Ask 5 test questions; check confidence scores and source citations

> **Note on `think=False`:** Qwen3 has a "thinking mode" that generates internal reasoning tokens (`<think>…</think>`) before answering. For RAG this is harmful — those tokens corrupt answer formatting, leak into citations, and add 10–30s latency per query. `think=False` disables it for clean, fast, deterministic answers.

### Key concepts
- **Embedding**: A list of numbers representing meaning. Similar text → similar vectors.
- **Cosine similarity**: Measures how close two vectors are. Closer = more relevant.
- **RAG loop**: Query → embed → find similar chunks → stuff into prompt → LLM answers from chunks only.

### PDF sources for Phase 1 (verified working)
Start with 1–2 of these. All are free, open-access, and directly downloadable:

| Document | Topic | Direct PDF |
|----------|-------|------------|
| JNC7 Hypertension Guidelines (NHLBI) | Hypertension | https://www.nhlbi.nih.gov/files/docs/guidelines/jnc7full.pdf |
| WHO Diabetes Management | Diabetes | https://iris.who.int/bitstream/handle/10665/364999/9789240083080-eng.pdf |
| WHO Global TB Report 2023 | Tuberculosis | https://iris.who.int/bitstream/handle/10665/373828/9789240083851-eng.pdf |
| WHO Cardiovascular Prevention | Cardiology | https://iris.who.int/bitstream/handle/10665/274846/9789241550185-eng.pdf |

**Recommendation for Phase 1:** Download the JNC7 Hypertension PDF — it's well-structured, 253 pages, and ideal for testing Q&A.

### Milestone check
✅ Ask "What is the recommended first-line treatment for hypertension?" → get a cited answer with page number and confidence score.

---

## Phase 2 — Document Pipeline (Week 2)

**Goal:** Build a proper ingestion pipeline that handles multiple documents with rich metadata.

### What you'll learn
- Why metadata is critical for healthcare RAG (trust, traceability, temporal validity)
- How to structure a corpus for medical use
- How chunking strategy affects answer quality

### Tasks
- [ ] Create `src/ingest.py` — processes all PDFs in `./docs/` folder
- [ ] Add metadata to every chunk:
  ```python
  metadata = {
      "source_title": "JNC7 Hypertension Guidelines",
      "source_url": "https://www.nhlbi.nih.gov/files/docs/guidelines/jnc7full.pdf",
      "publication_date": "2003-05",
      "evidence_level": "guideline",  # guideline | RCT | review | case_study
      "medical_specialty": "cardiology",
      "is_current": True
  }
  ```
- [ ] Build a corpus of 5–10 documents (mix of guidelines + papers)
- [ ] Experiment with chunk sizes (256, 512, 1024) — compare confidence scores
- [ ] Add `SemanticChunker` as an alternative to character-based splitting
- [ ] Persist ChromaDB so you don't re-embed every run

### PDF sources for Phase 2 (verified working — 5–10 documents)

| Document | Topic | Direct PDF |
|----------|-------|------------|
| WHO Hypertension Guideline 2021 | Hypertension | https://iris.who.int/bitstream/handle/10665/342005/9789240033986-eng.pdf |
| WHO Diabetes Management | Diabetes | https://iris.who.int/bitstream/handle/10665/364999/9789240083080-eng.pdf |
| WHO TB Report 2023 | Tuberculosis | https://iris.who.int/bitstream/handle/10665/373828/9789240083851-eng.pdf |
| WHO Cardiovascular Prevention | Cardiology | https://iris.who.int/bitstream/handle/10665/274846/9789241550185-eng.pdf |
| JNC7 Hypertension (NHLBI) | Hypertension | https://www.nhlbi.nih.gov/files/docs/guidelines/jnc7full.pdf |
| CDC STI Treatment Guidelines | STI | https://www.cdc.gov/mmwr/pdf/rr/rr6203.pdf |
| WHO Mental Health Action Plan | Mental Health | https://iris.who.int/bitstream/handle/10665/89966/9789241506021_eng.pdf |

### Milestone check
✅ Ingest 5–10 documents in one script run. Query spans multiple documents. Source metadata shows in every retrieved chunk.

---

## Phase 3 — Enhanced Retrieval (Week 3)

**Goal:** Move beyond basic vector search. Hybrid search + re-ranking dramatically improves answer quality.

### What you'll learn
- Why semantic search alone misses exact medical terms (drug names, ICD codes, gene symbols)
- How BM25 keyword search complements semantic search
- How a cross-encoder re-ranker scores relevance more accurately

### Tasks
- [ ] Create `src/retriever.py`
- [ ] Add BM25 retriever:
  ```python
  from langchain.retrievers import BM25Retriever, EnsembleRetriever
  bm25 = BM25Retriever.from_documents(docs)
  semantic = vectorstore.as_retriever(search_kwargs={"k": 10})
  ensemble = EnsembleRetriever(retrievers=[bm25, semantic], weights=[0.4, 0.6])
  ```
- [ ] Add cross-encoder re-ranking (free, local — `ms-marco-MiniLM-L-6-v2`):
  ```python
  from langchain.retrievers.document_compressors import CrossEncoderReranker
  ```
- [ ] Add metadata filtering (search only guidelines from a specific specialty)
- [ ] Add query expansion: generate 3 alternative phrasings before retrieval
- [ ] Compare: semantic-only vs. hybrid vs. hybrid + reranker confidence scores

### Milestone check
✅ Searching "MI troponin threshold" returns chunks containing both "myocardial infarction" (semantic) and exact "troponin" mentions (BM25).

---

## Phase 4 — Ethical Guardrails (Week 4)

**Goal:** Make the system safe for healthcare use. Guardrails are a product feature, not an afterthought.

### What you'll learn
- How system prompts enforce behavioural constraints
- How to detect and refuse out-of-scope queries
- How to implement calibrated confidence scoring

### Tasks
- [ ] Refactor `phase1_starter_code.py` into `src/chain.py` and `src/guardrails.py`
- [ ] Harden the query classifier — refuse:
  - Specific patient diagnoses
  - Drug prescriptions for named individuals
  - Emergency medical advice
- [ ] Calibrate confidence scorer using actual similarity scores (not heuristics)
- [ ] Add citation formatter — display: Title · Date · Source URL
- [ ] Add temporal validity check — warn if source is >5 years old
- [ ] Add conflicting guidelines detector — if top chunks disagree, surface the conflict

### Milestone check
✅ "Diagnose my patient with chest pain" → refused. "Metformin contraindications?" → cited answer with confidence badge and publication date.

---

## Phase 5 — RAGAS Evaluation (Week 5)

**Goal:** Quantify system quality across all 5 metrics. Build a reusable evaluation dataset.

### What you'll learn
- How to build a ground-truth test dataset
- How RAGAS measures faithfulness, relevancy, precision, recall
- How to interpret scores and improve them iteratively

### Tasks
- [ ] Build `eval/dataset.json` (20–30 question-answer pairs from your corpus)
- [ ] Implement `eval/evaluate.py` with RAGAS:
  ```python
  from ragas import evaluate
  from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
  result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
  ```
- [ ] Add citation fidelity check (does the cited chunk actually support the claim?)
- [ ] Log results to `eval/results/` CSV for tracking over time
- [ ] Iterate: faithfulness < 0.8 → tighten system prompt; recall < 0.7 → increase top-k

### Target scores

| Metric | Minimum | Target |
|--------|---------|--------|
| Faithfulness | 0.80 | > 0.90 |
| Answer Relevancy | 0.75 | > 0.85 |
| Context Precision | 0.70 | > 0.80 |
| Context Recall | 0.70 | > 0.80 |
| Hallucination Rate | < 0.15 | < 0.05 |

### Milestone check
✅ RAGAS script runs against test set and prints a score table. You can show before/after scores from retrieval improvements.

---

## Phase 6 — UI + Portfolio Polish (Week 6)

**Goal:** Build a usable interface. Document everything. Record a demo. Ship.

### Tasks
- [ ] Build `app.py` (Streamlit):
  - Persistent disclaimer banner (non-dismissible)
  - Chat input + conversation history
  - Citations displayed as cards (title, date, page)
  - Confidence badge (🔴 LOW / 🟡 MEDIUM / 🟢 HIGH)
  - Specialty filter in sidebar
  - "View source chunk" expander per citation
- [ ] Finalise `README.md` with RAGAS scores, architecture diagram, setup instructions
- [ ] Record a 2-minute demo video
- [ ] Push to GitHub with one commit per phase
- [ ] Add `eval/results/` CSV to repo for transparency

### Portfolio differentiators
- RAGAS scores in README = evaluation maturity (rare in portfolios)
- Ethics section = domain awareness
- Before/after retrieval comparison = iterative thinking

### Milestone check
✅ Working Streamlit app. Clean GitHub repo. Demo video linked in README.

---

## Dependencies

All managed via `environment.yml`:
```bash
conda env create -f environment.yml
conda activate healthcare-rag
```

Key packages: `langchain`, `langchain-ollama`, `langchain-chroma`, `langchain-huggingface`, `chromadb`, `sentence-transformers`, `pypdf`, `pymupdf`, `rank-bm25`, `ragas`, `streamlit`

No paid API keys required. Ollama runs fully locally.

---

## Recommended Learning Resources

- LangChain RAG tutorial: https://python.langchain.com/docs/tutorials/rag/
- RAGAS documentation: https://docs.ragas.io
- ChromaDB getting started: https://docs.trychroma.com
- Ollama model library: https://ollama.com/library
- Qwen3 model card: https://ollama.com/library/qwen3
