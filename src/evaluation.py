"""End-to-end evaluation: run test questions through the pipeline and report metrics."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

# Map expected_source labels (from test_questions.json) to keywords that appear
# in the actual citation organization/title strings returned by the pipeline.
SOURCE_KEYWORD_MAP = {
    "who hypertension": ["world health organization", "hypertension"],
    "ada diabetes": ["american diabetes association", "diabetes"],
    "aha heart failure": ["american heart association", "heart failure", "acc"],
    "cdc sti": ["centers for disease control", "cdc", "sexually transmitted"],
    "uspstf cancer screening": ["preventive services", "uspstf", "cancer screening",
                                "breast cancer", "colorectal cancer", "lung cancer", "cervical cancer"],
}

from src.config import DEFAULT_MODEL, DEFAULT_CHUNKS_PATH, PROJECT_ROOT, estimate_cost
from src.generation import generate_answer
from src.retrieval import hybrid_search, rerank, semantic_search


def evaluate_pipeline(
    test_set_path: str,
    all_chunks: list[dict] | None = None,
    prompt_version: str = "v3_few_shot",
    model: str = DEFAULT_MODEL,
    use_rerank: bool = True,
    use_hybrid: bool = True,
    top_k: int = 5,
) -> dict:
    """Run every question in the test set and aggregate metrics."""
    with open(test_set_path, "r", encoding="utf-8") as f:
        test_questions = json.load(f)

    results: list[dict] = []
    total_latency = 0.0
    total_tokens_in = 0
    total_tokens_out = 0
    total_cost = 0.0

    for q in test_questions:
        print(f"Evaluating Q{q['id']}: {q['question'][:70]}...")
        start_time = time.time()

        if use_hybrid and all_chunks:
            retrieved = hybrid_search(q["question"], all_chunks, top_k=max(top_k * 2, 10))
        else:
            retrieved = semantic_search(q["question"], top_k=max(top_k * 2, 10))

        if use_rerank:
            retrieved = rerank(q["question"], retrieved, top_k=top_k)
            # Respect Cohere Trial rate limit: 10 calls/minute = 6s between calls
            time.sleep(6)
        else:
            retrieved = retrieved[:top_k]

        response = generate_answer(
            query=q["question"],
            retrieved_chunks=retrieved,
            prompt_version=prompt_version,
            model=model,
        )

        latency = time.time() - start_time
        total_latency += latency
        total_tokens_in += response["tokens_in"]
        total_tokens_out += response["tokens_out"]
        total_cost += estimate_cost(model, response["tokens_in"], response["tokens_out"])

        expected_source = str(q.get("expected_source", "")).lower().strip()
        citations_blob = str(response["citations"]).lower()
        answer_blob = response["answer"].lower()
        combined_blob = citations_blob + " " + answer_blob

        # Use keyword map for flexible matching; fall back to direct substring match
        keywords = SOURCE_KEYWORD_MAP.get(expected_source, [expected_source])
        correct_source = bool(expected_source) and any(kw in combined_blob for kw in keywords)

        results.append(
            {
                "question_id": q["id"],
                "question": q["question"],
                "expected_source": q.get("expected_source"),
                "difficulty": q.get("difficulty"),
                "answer": response["answer"],
                "citations": response["citations"],
                "tokens_used": response["total_tokens"],
                "latency_seconds": round(latency, 2),
                "correct_source_retrieved": correct_source,
            }
        )

    n = len(test_questions) or 1
    metrics = {
        "total_questions": len(test_questions),
        "correct_source_retrieval": sum(1 for r in results if r["correct_source_retrieved"]),
        "source_accuracy_pct": round(
            100 * sum(1 for r in results if r["correct_source_retrieved"]) / n, 2
        ),
        "avg_latency_seconds": round(total_latency / n, 2),
        "total_tokens_in": total_tokens_in,
        "total_tokens_out": total_tokens_out,
        "total_tokens": total_tokens_in + total_tokens_out,
        "estimated_cost_usd": round(total_cost, 4),
        "model": model,
        "prompt_version": prompt_version,
        "use_rerank": use_rerank,
        "use_hybrid": use_hybrid,
    }
    return {"metrics": metrics, "results": results}


def main() -> None:
    test_set = os.path.join(PROJECT_ROOT, "data", "test_set", "test_questions.json")
    if not Path(test_set).exists():
        raise FileNotFoundError(f"Test set not found: {test_set}")

    all_chunks = None
    if Path(DEFAULT_CHUNKS_PATH).exists():
        with open(DEFAULT_CHUNKS_PATH, "r", encoding="utf-8") as f:
            all_chunks = json.load(f)

    report = evaluate_pipeline(
        test_set_path=test_set,
        all_chunks=all_chunks,
        prompt_version="v3_few_shot",
        model=DEFAULT_MODEL,
        use_rerank=True,
        use_hybrid=bool(all_chunks),
    )

    out_dir = os.path.join(PROJECT_ROOT, "data", "processed")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "evaluation_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n=== Metrics ===")
    for k, v in report["metrics"].items():
        print(f"  {k}: {v}")
    print(f"\nSaved full report to {out_path}")


if __name__ == "__main__":
    main()
