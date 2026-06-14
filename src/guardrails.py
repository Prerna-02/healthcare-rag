"""
Healthcare Knowledge Navigator — Phase 4
Ethical guardrails: the safety layer that wraps the RAG pipeline on BOTH sides.

INPUT guardrails (run BEFORE retrieval, to refuse unsafe queries):
  - Layer A  keyword_classify() : fast, deterministic, AUDITABLE keyword rules.
  - Layer B  llm_classify()     : an LLM intent classifier that understands MEANING,
                                  so it catches phrasings Layer A never enumerated.
  - classify_query()            : runs A, then B — returns one category.

OUTPUT guardrails (run AFTER generation, to annotate the answer):
  - compute_confidence()  : HIGH/MEDIUM/LOW from real cross-encoder relevance.
  - temporal_warnings()   : flags any cited source older than MAX_SOURCE_AGE_YEARS.
  - detect_conflicts()    : notes when the answer rests on multiple sources.
  - citation_list()       : structured citations (title, page, date, staleness, snippet).

All policy (refusal messages, the disclaimer, thresholds) lives in config.py, not
in this logic — so it's a readable, editable rulebook, not buried magic.
"""
import re

import ollama

import config

# Categories the system refuses. "general" (anything else) is allowed through.
BLOCK_CATEGORIES = set(config.REFUSAL_MESSAGES.keys())  # emergency, diagnosis, prescribing


# ── Layer A: fast deterministic keyword rules ─────────────────────────────────
# A small, high-precision set kept deliberately explicit: in a safety system you
# WANT certain phrases to ALWAYS block, instantly, with no model in the loop.
# This is the floor; Layer B generalises to everything not listed here.
_EMERGENCY_RE = re.compile(
    r"\b(can'?t breathe|chest pain|severe bleeding|unconscious|"
    r"i('?m| am)\s+(having|experiencing)\b|call (911|999|112))",
    re.IGNORECASE,
)
_DIAGNOSIS_RE = re.compile(
    r"\bdiagnose\b|\bwhat('?s| is) wrong with (me|my)\b|\bwhat do i have\b"
    r"|\bmy patient\b.*\b(has|have|presents|suffering)\b",
    re.IGNORECASE,
)
_PRESCRIBING_RE = re.compile(
    r"\bshould i (give|prescribe|start)\b|\bwhat (dose|dosage)\b.*\b(give|prescribe|this patient)\b",
    re.IGNORECASE,
)


def keyword_classify(query: str) -> str | None:
    """Layer A. Return a block category if a known pattern matches, else None."""
    if _EMERGENCY_RE.search(query):
        return "emergency"
    if _DIAGNOSIS_RE.search(query):
        return "diagnosis"
    if _PRESCRIBING_RE.search(query):
        return "prescribing"
    return None


# ── Layer B: LLM intent classifier ────────────────────────────────────────────
# Uses the ollama client directly with think=False — Qwen3 otherwise burns its
# token budget on hidden reasoning and returns nothing (verified empirically).
_CLASSIFIER_PROMPT = """You are a safety classifier for a medical knowledge assistant. Classify the query into ONE category:
- EMERGENCY: an acute medical emergency happening now (e.g. severe chest pain, can't breathe, collapse).
- DIAGNOSIS: asks you to diagnose or interpret symptoms for a specific person.
- PRESCRIBING: asks what drug or dose to give/prescribe to a specific individual.
- GENERAL: a general medical-knowledge question (definitions, guideline content, mechanisms,
  population-level recommendations). This is the safe, allowed case.
Reply with ONLY the category word.

Query: {query}
Category:"""


def llm_classify(query: str) -> str:
    """Layer B. Ask the local LLM to classify intent by MEANING. Returns one of
    emergency / diagnosis / prescribing / general (defaults to 'general')."""
    resp = ollama.chat(
        model=config.LLM_MODEL,
        messages=[{"role": "user", "content": _CLASSIFIER_PROMPT.format(query=query)}],
        think=False,
        options={"temperature": 0, "num_predict": config.CLASSIFIER_NUM_PREDICT},
    )
    text = resp.message.content.upper()
    for category in ("EMERGENCY", "DIAGNOSIS", "PRESCRIBING"):
        if category in text:
            return category.lower()
    return "general"


def classify_query(query: str, use_llm: bool = True) -> str:
    """Run Layer A, then Layer B. The two are sequential gates: a query must pass
    BOTH to be treated as 'general' (safe). Returns the category string."""
    hit = keyword_classify(query)          # Layer A — fast, guaranteed
    if hit:
        return hit
    if use_llm:
        return llm_classify(query)         # Layer B — meaning-based catch-all
    return "general"


def is_blocked(category: str) -> bool:
    return category in BLOCK_CATEGORIES


def refusal_message(category: str) -> str:
    return config.REFUSAL_MESSAGES[category]


# ── Output guardrails ─────────────────────────────────────────────────────────

def compute_confidence(scores: list[float]) -> tuple[str, float]:
    """Map average relevance (0–1, from the cross-encoder) to HIGH/MEDIUM/LOW.
    Thresholds live in config; full calibration happens with RAGAS in Phase 5."""
    if not scores:
        return "LOW", 0.0
    avg = sum(scores) / len(scores)
    if avg >= config.CONF_HIGH:
        return "HIGH", avg
    if avg >= config.CONF_MEDIUM:
        return "MEDIUM", avg
    return "LOW", avg


def temporal_warnings(docs: list) -> list[str]:
    """One warning per distinct source that is older than the freshness window."""
    warnings, seen = [], set()
    for doc in docs:
        meta = doc.metadata
        title = meta.get("source_title") or meta.get("source_filename")
        if meta.get("is_current") is False and title not in seen:
            seen.add(title)
            warnings.append(
                f"⚠️  {title} ({meta.get('publication_date')}) is over "
                f"{config.MAX_SOURCE_AGE_YEARS} years old — verify against current guidelines."
            )
    return warnings


def detect_conflicts(docs: list) -> str | None:
    """Lightweight flag: if the answer rests on chunks from 2+ distinct sources,
    surface them so the reader can check whether the sources actually agree. (A
    true semantic disagreement detector is future work — this is the honest
    first version.)"""
    titles = []
    for doc in docs:
        title = doc.metadata.get("source_title") or doc.metadata.get("source_filename")
        if title and title not in titles:
            titles.append(title)
    if len(titles) >= 2:
        joined = "; ".join(titles[:3])
        return (f"ℹ️  This answer draws on multiple sources ({joined}). "
                "If their recommendations differ, clinical context determines which applies.")
    return None


def citation_list(docs: list) -> list[dict]:
    """De-duplicated, STRUCTURED citations (one per unique source+page). Used by
    both the API (serialised to JSON) and the CLI (rendered to text). Includes a
    short snippet so a UI can offer a 'view source' expander."""
    seen, out = set(), []
    for doc in docs:
        meta = doc.metadata
        key = f"{meta.get('source_filename')}:{meta.get('page')}"
        if key in seen:
            continue
        seen.add(key)
        date = meta.get("publication_date", "")
        out.append({
            "index": len(out) + 1,
            "title": meta.get("source_title") or meta.get("source_filename"),
            "page": (meta.get("page") or 0) + 1,
            "date": "" if date == "unknown" else date,
            "source_url": meta.get("source_url", ""),
            "is_current": meta.get("is_current") is not False,
            "snippet": " ".join(doc.page_content.split())[:300],
        })
    return out
