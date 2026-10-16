from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Limitar hilos de CPU para mantener la fluidez del sistema
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

import pandas as pd

from legal_rag_boe.indexing.evaluator import (
    BENCHMARK_DOC_CODES,
    GOLDEN_CASES,
    EvaluationCase,
    StrategyCaseResult,
    aggregate_metrics,
    evaluate_retrieval_phase,
)
from legal_rag_boe.indexing.loader import load_documents_from_json
from legal_rag_boe.indexing.rerankers import (
    BaseDocumentReranker,
    JinaRerankerV35,
)
from legal_rag_boe.indexing.strategies import (
    BaseRAGStrategy,
    RerankedRAG,
    get_canonical_strategies,
    get_default_embed_model,
)


def evaluate_reranked_strategy(
    strategy: BaseRAGStrategy,
    reranker: BaseDocumentReranker | None = None,
    candidate_k: int = 10,
    top_k: int = 3,
    cases: Sequence[EvaluationCase] = GOLDEN_CASES,
) -> list[StrategyCaseResult]:
    """Evalúa la precisión y latencia de una estrategia base combinada opcionalmente con un reranker."""
    if reranker is None:
        return evaluate_retrieval_phase(strategy, cases=cases, top_k=top_k)
    reranked_strat = RerankedRAG(
        base_strategy=strategy,
        reranker=reranker,
        candidate_k=candidate_k,
        name=f"{strategy.name} + {reranker.name}",
    )
    return evaluate_retrieval_phase(reranked_strat, cases=cases, top_k=top_k)


def run_reranker_benchmark() -> pd.DataFrame:
    """Ejecuta el benchmark sistemático comparando los 4 modelos base con y sin cada reranker."""
    indices_dir = Path("data/indices")
    documents = load_documents_from_json(json_dir="data/json", doc_codes=BENCHMARK_DOC_CODES)
    embed_model = get_default_embed_model(model_name="bge-m3", batch_size=32)

    print("--- 1. Cargando las 4 estrategias canónicas homologadas ---", flush=True)
    canonical = get_canonical_strategies(documents, embed_model=embed_model, indices_dir=indices_dir)

    print("--- 2. Inicializando Rerankers ---", flush=True)
    try:
        jina_reranker = JinaRerankerV35.get_singleton()
        print(f"-> {jina_reranker.name} cargado en dispositivo: {jina_reranker.device}.", flush=True)
    except Exception as e:
        print(f"-> Error cargando Jina Reranker: {e}", flush=True)
        jina_reranker = None

    rerankers_to_test: list[tuple[str, BaseDocumentReranker | None]] = [
        ("Sin Reranker (Base)", None),
    ]
    if jina_reranker:
        rerankers_to_test.append(("Jina-v3.5 (CUDA GPU)", jina_reranker))

    all_rows: list[dict[str, object]] = []

    print("\n--- 3. Ejecutando batería de pruebas sobre Golden Cases ---", flush=True)
    for strat_key, strat in canonical.items():
        print(f"\n[Estrategia Base: {strat.name}]", flush=True)
        for r_label, r_inst in rerankers_to_test:
            print(f"  * Evaluando con {r_label}...", flush=True)
            res = evaluate_reranked_strategy(strat, reranker=r_inst, candidate_k=10, top_k=3)
            agg = aggregate_metrics(res)
            all_rows.append({
                "Estrategia Base": strat.name,
                "Reranker": r_label,
                "Hit Rate (%)": round(agg.hit_rate * 100, 1),
                "Cobertura Hechos (%)": round(agg.mean_fact_coverage * 100, 1),
                "Tokens Contexto": round(agg.mean_tokens),
                "Latencia (ms)": round(agg.mean_latency_ms, 1),
            })
            print(
                f"    -> Hit: {agg.hit_rate * 100:.1f}% | Hechos: {agg.mean_fact_coverage * 100:.1f}% | "
                f"Tokens: {agg.mean_tokens:.0f} | Latencia: {agg.mean_latency_ms:.1f}ms",
                flush=True,
            )

    df = pd.DataFrame(all_rows)
    return df


if __name__ == "__main__":
    df_results = run_reranker_benchmark()
    print("\n=== RESUMEN COMPARATIVO DE RERANKERS ===")
    print(df_results.to_string(index=False))

