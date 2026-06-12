# Healthcare Knowledge Navigator

A medical RAG (Retrieval-Augmented Generation) assistant that retrieves and synthesises evidence-based answers from clinical guidelines, research papers, and treatment protocols — with full citation tracking, confidence scoring, and built-in ethical guardrails.

Built for medical professionals and students. **Not a substitute for clinical judgment.**

---

## What It Does

- Answers medical questions grounded strictly in your indexed document corpus
- Returns cited sources (title, page, evidence level) with every answer
- Scores retrieval confidence so you know when to trust the answer
- Refuses out-of-scope queries (patient diagnosis, emergency guidance, prescribing)
- Evaluated against five RAGAS metrics to quantify and reduce hallucination

---

## Architecture

```
PDFs / Guidelines / Papers
         │
         ▼
  Document Processing          PyMuPDF · RecursiveCharacterTextSplitter
  (load → chunk → metadata)
         │
         ▼
     Embedding                 sentence-transformers/all-MiniLM-L6-v2 (CPU)
         │
         ▼
    Vector Store               ChromaDB (local) → Pinecone (production)
         │
         ▼
     Retrieval                 Phase 1: Semantic search
                               Phase 3: Hybrid (semantic + BM25) + CrossEncoder reranker
         │
         ▼
   RAG Chain + Guardrails      LangChain · Ollama (qwen3:8b, local)
   (system prompt · classifier · confidence scoring · citations)
         │
         ▼
     Response                  Answer · Citations · Confidence · Disclaimer
         │
         ▼
    RAGAS Evaluation           Faithfulness · Answer Relevancy · Context Precision
                               Context Recall · Hallucination Rate
```

---

## Project Structure

```
healthcare-rag/
├── config.py                  # all settings in one place (models, paths, thresholds)
├── phase1_starter_code.py     # Phase 1 entry point — run this first
├── app.py                     # Phase 6 — Streamlit UI
│
├── src/
│   ├── ingest.py              # Phase 2: multi-doc loader, chunker, embedder
│   ├── retriever.py           # Phase 3: hybrid search + re-ranker
│   ├── chain.py               # Phase 4: full RAG chain (refactored from Phase 1)
│   └── guardrails.py          # Phase 4: query classifier, confidence scorer
│
├── eval/
│   ├── dataset.json           # ground truth Q&A pairs (build in Phase 5)
│   └── evaluate.py            # RAGAS evaluation runner
│
├── docs/                      # drop your PDFs here (git-ignored)
├── chroma_db/                 # auto-created by ChromaDB (git-ignored)
│
├── environment.yml            # conda environment definition
├── .env.example               # API key template
└── .gitignore
```

---

## Setup

### Prerequisites

- [Conda](https://docs.conda.io/en/latest/miniconda.html) (Miniconda recommended)
- [Ollama](https://ollama.com) installed and running
- Python 3.11

### 1. Clone and create environment

```bash
git clone https://github.com/your-username/healthcare-rag.git
cd healthcare-rag

conda env create -f environment.yml
conda activate healthcare-rag
```

### 2. Download the LLM

```bash
ollama pull qwen3:8b
```

Ollama must be running in the background before you start the app (`ollama serve`).

**Model selection guide (CPU only, no GPU):**

| Your RAM | Model | Size | Notes |
|----------|-------|------|-------|
| 16 GB+ | `qwen3:8b` ← recommended | 5.2 GB | Strong reasoning; thinking mode disabled for clean RAG output |
| 16 GB+ | `llama3.1:8b` | 4.7 GB | Solid alternative, no thinking mode to manage |
| 8 GB | `mistral:7b` | 4.1 GB | Excellent at following complex system prompts |
| 8 GB tight | `phi3:mini` | 2.3 GB | Fast, weaker reasoning |

### 3. Configure environment

```bash
cp .env.example .env
```

No API keys required if using Ollama. The `.env` file is only needed if you later add OpenAI or a cloud embedding service.

### 4. Add documents

Drop PDF files into `./docs/`. Recommended free sources:

- [WHO Essential Medicines](https://www.who.int/publications/i/item/WHO-MHP-HPS-EML-2023.02)
- [NICE Clinical Guidelines](https://www.nice.org.uk/guidance)
- [NIH/PMC Open Access](https://pmc.ncbi.nlm.nih.gov/tools/openftlist/)
- [CDC Clinical Guidance](https://www.cdc.gov/guidelines/)

### 5. Run

```bash
python phase1_starter_code.py
```

---

## Usage

### Phase 1 — Command line

```bash
python phase1_starter_code.py
```

The script runs five built-in test queries (including emergency and refusal cases), then drops into interactive mode. Type `quit` to exit.

### Phase 6 — Streamlit app (coming)

```bash
streamlit run app.py
```

---

## Development Phases

| Phase | Week | What Gets Built | Entry Point |
|-------|------|-----------------|-------------|
| 1 | 1 | Basic RAG: single PDF → ChromaDB → Ollama → cited answer | `phase1_starter_code.py` |
| 2 | 2 | Multi-doc ingestion with metadata tagging | `src/ingest.py` |
| 3 | 3 | Hybrid search (BM25 + semantic) + CrossEncoder re-ranking | `src/retriever.py` |
| 4 | 4 | Full guardrails: query classifier, confidence scoring, citation formatter | `src/guardrails.py` |
| 5 | 5 | RAGAS evaluation suite across all 5 metrics | `eval/evaluate.py` |
| 6 | 6 | Streamlit UI: chat interface, citation cards, confidence badges | `app.py` |

---

## Ethical Guardrails

The system enforces the following at the code level, not just in documentation.

### Hard blocks (query classifier refuses and returns a message)

| Query type | Example | Action |
|------------|---------|--------|
| Medical emergency | "I'm having chest pain right now" | Refuse + redirect to emergency services |
| Patient diagnosis | "Diagnose my patient with these symptoms" | Refuse |
| Individual prescribing | "What dose should I give this patient?" | Refuse |

### Behavioural constraints (enforced in system prompt)

- Answers are generated **only** from retrieved context — never from model memory
- Every factual claim must be traceable to a cited chunk
- Sources older than 5 years trigger a temporal staleness warning
- When evidence is limited or conflicting, the system says so explicitly
- Every response ends with a non-removable disclaimer

### Mandatory disclaimer on every response

> This information is for educational purposes only and does not constitute medical advice. Always consult a qualified healthcare professional for clinical decisions.

### What this system is NOT

- Not a diagnostic engine
- Not a prescribing tool
- Not a real-time emergency resource
- Not a replacement for clinical judgment

These guardrails are enforced in code — the query classifier, system prompt, and confidence scorer (`src/guardrails.py`, Phase 4) — not just described in documentation.

---

## Evaluation (RAGAS)

The system is evaluated against five metrics. Target thresholds for this project:

| Metric | What it measures | Minimum | Target |
|--------|-----------------|---------|--------|
| Faithfulness | Are answer claims supported by the retrieved context? | 0.80 | > 0.90 |
| Answer Relevancy | Does the answer address the question? | 0.75 | > 0.85 |
| Context Precision | Are the retrieved chunks relevant (not noisy)? | 0.70 | > 0.80 |
| Context Recall | Was all necessary information retrieved? | 0.70 | > 0.80 |
| Hallucination Rate | Claims not in any source document | < 0.15 | < 0.05 |

Evaluation results are logged to `eval/results/` after each run so you can track improvement across phases.

---

## Dependencies

All managed via `environment.yml`. Key packages:

| Package | Purpose |
|---------|---------|
| `langchain` + `langchain-community` | RAG orchestration |
| `langchain-ollama` | Local LLM integration |
| `langchain-chroma` | Vector store integration |
| `sentence-transformers` | CPU-friendly embeddings |
| `chromadb` | Local vector database |
| `pypdf` + `pymupdf` | PDF loading |
| `rank-bm25` | Keyword retrieval (Phase 3) |
| `ragas` | RAG evaluation framework |
| `streamlit` | Web UI (Phase 6) |

Install:
```bash
conda env create -f environment.yml
conda activate healthcare-rag
```

---

## Limitations

- Answers are only as good as the indexed corpus — garbage in, garbage out
- CPU inference with `qwen3:8b` generates ~8–15 tokens/second (a full answer takes ~20–40 seconds)
- The system cannot access the internet or fetch updated guidelines in real time
- Medical literature contains known population biases (sex, race, age) — the system does not automatically correct for these
- RAGAS evaluation requires a manually curated ground-truth dataset to be meaningful

---

## Disclaimer

**This project is for educational and research purposes only.** It is not a certified medical device, does not meet FDA Software as a Medical Device (SaMD) requirements, and must not be used to guide clinical decisions for real patients.

---

## License

MIT License — see `LICENSE` for details.
