"""
Healthcare Knowledge Navigator — Phase 5
RAGAS evaluation with a fully LOCAL judge (Qwen3 via Ollama) + HF embeddings.

For each KNOWLEDGE question we run the real Phase 4 pipeline (retrieve → generate),
then RAGAS scores: faithfulness, answer relevancy, context precision, context recall.
SAFETY questions are checked separately — they must be refused (or return the
"insufficient evidence" non-answer).

Caveat: an 8B local judge is noisier than a frontier model — treat scores as
DIRECTIONAL (good for before/after comparison), not absolute truth.

Run:
    conda activate healthcare-rag
    python -m eval.evaluate              # full set (slow on CPU: 30-60+ min)
    python -m eval.evaluate --limit 2    # quick smoke run to validate the pipeline
"""
import argparse
import json
import sys
from datetime import datetime

import ollama
import pandas as pd
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
from ragas.run_config import RunConfig

import config
from src import guardrails
from src.chain import _NO_ANSWER_MARKER, Assistant, generate

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ── A judge that can actually turn Qwen3's thinking OFF ───────────────────────
# RAGAS makes MANY judge calls. With langchain's ChatOllama (thinking ON, ~200
# hidden tokens/call) a 2-question run took CPU-hours. Routing through the ollama
# client with think=False makes each judge call ~5-10x faster.
_ROLE = {"system": "system", "human": "user", "ai": "assistant"}


class NoThinkChatOllama(BaseChatModel):
    model: str = config.LLM_MODEL
    temperature: float = 0

    @property
    def _llm_type(self) -> str:
        return "no-think-ollama"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        resp = ollama.chat(
            model=self.model,
            messages=[{"role": _ROLE.get(m.type, "user"), "content": m.content} for m in messages],
            think=False,
            options={"temperature": self.temperature},
        )
        msg = AIMessage(content=resp.message.content)
        return ChatResult(generations=[ChatGeneration(message=msg)])


DATASET_PATH = config.EVAL_DIR / "dataset.json"
RESULTS_DIR = config.EVAL_DIR / "results"
METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]


# ── Build RAGAS samples by running the real pipeline ──────────────────────────

def build_samples(assistant: Assistant, items: list) -> list:
    """For each knowledge question: retrieve with the real retriever, generate the
    real answer, and package it the way RAGAS expects."""
    samples = []
    for i, item in enumerate(items, 1):
        question = item["question"]
        print(f"  [{i}/{len(items)}] generating: {question[:55]}...")
        docs = assistant.retriever.invoke(question)
        contexts = [doc.page_content for doc in docs]
        answer, _ = generate(question, "\n\n".join(contexts))
        samples.append({
            "user_input": question,
            "retrieved_contexts": contexts,
            "response": answer,
            "reference": item["ground_truth"],
        })
    return samples


# ── Safety checks (separate from RAGAS scoring) ───────────────────────────────

def run_safety_checks(assistant: Assistant, items: list) -> pd.DataFrame:
    rows = []
    for item in items:
        question, expected = item["question"], item["expected"]
        category = guardrails.classify_query(question)
        if expected == "insufficient_evidence":
            # Not a refusal: must answer with the "insufficient evidence" non-answer.
            docs = assistant.retriever.invoke(question)
            answer, _ = generate(question, "\n\n".join(d.page_content for d in docs))
            said_no = _NO_ANSWER_MARKER in answer.lower()
            passed = said_no
            got = "insufficient_evidence" if said_no else "answered"
        else:
            got = category
            passed = category == expected
        rows.append({"question": question, "expected": expected, "got": got,
                     "pass": "✅" if passed else "❌"})
    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Only evaluate the first N knowledge questions (for a quick smoke run).")
    args = parser.parse_args()

    data = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    knowledge = data["knowledge"][: args.limit] if args.limit else data["knowledge"]
    safety = data["safety"]

    print("=" * 64)
    print(f"  Phase 5 — RAGAS evaluation (local judge: {config.LLM_MODEL})")
    print(f"  {len(knowledge)} knowledge questions, {len(safety)} safety checks")
    print("=" * 64)

    assistant = Assistant()

    # 1) Generate answers for the knowledge set.
    print("\nGenerating answers from the live pipeline...")
    samples = build_samples(assistant, knowledge)

    # 2) Score with RAGAS using the local judge + embeddings.
    print("\nScoring with RAGAS (local judge — this is the slow part)...")
    judge = LangchainLLMWrapper(NoThinkChatOllama())
    embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
        model_name=config.EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    ))
    dataset = EvaluationDataset.from_list(samples)
    # max_workers=1: one local model, avoid CPU thrash; very generous timeout
    # because each judge call generates token-by-token on CPU.
    run_config = RunConfig(max_workers=1, timeout=1800)
    result = evaluate(dataset, metrics=METRICS, llm=judge, embeddings=embeddings,
                      run_config=run_config)
    scores = result.to_pandas()

    # SAVE + PRINT THE SCORES IMMEDIATELY — they're expensive (~minutes each), so
    # nothing downstream is allowed to lose them.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    scores.to_csv(RESULTS_DIR / f"ragas-{stamp}.csv", index=False)

    metric_cols = [c for c in scores.columns if c in
                   ("faithfulness", "answer_relevancy", "context_precision", "context_recall")]
    print("\n" + "=" * 64)
    print("  RAGAS SCORES (averages)")
    print("=" * 64)
    for col in metric_cols:
        print(f"  {col:22} {scores[col].mean():.3f}")
    print(f"\n  Saved: eval/results/ragas-{stamp}.csv")

    # Safety checks run AFTER scores are safely on disk, and can't crash the run.
    print("\nRunning safety checks...")
    try:
        safety_df = run_safety_checks(assistant, safety)
        safety_df.to_csv(RESULTS_DIR / f"safety-{stamp}.csv", index=False)
        print("\n  SAFETY CHECKS")
        for _, row in safety_df.iterrows():
            print(f"  {row['pass']} expected={row['expected']:22} got={row['got']:22} | {row['question'][:45]}")
        print(f"\n  Saved: eval/results/safety-{stamp}.csv")
    except Exception as exc:  # never let a safety-check bug discard the RAGAS scores
        print(f"  ⚠️ safety checks failed ({exc}) — RAGAS scores above are still saved.")


if __name__ == "__main__":
    main()
