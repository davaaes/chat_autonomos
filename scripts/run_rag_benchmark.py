from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Limitar hilos de CPU para garantizar que el sistema operativo se mantenga fluido
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

from llama_index.llms.ollama import Ollama

try:
    from trulens.providers.ollama import Ollama as TruLensOllama
except ImportError:
    TruLensOllama = None  # type: ignore[misc,assignment]

import pandas as pd

from legal_rag_boe.indexing.evaluator import (
    BENCHMARK_DOC_CODES,
    GOLDEN_CASES,
    StrategyAggregateMetrics,
    StrategyCaseResult,
    TruLensRAGSession,
    aggregate_metrics,
    evaluate_retrieval_phase,
    evaluate_trulens_triad,
    get_default_trulens_provider,
    run_llm_synthesis_batch,
)
from legal_rag_boe.indexing.loader import load_documents_from_json
from legal_rag_boe.indexing.strategies import (
    ArticleLevelRAG,
    AutoMergingRAG,
    BaseRAGStrategy,
    HybridBM25RAG,
    RerankedRAG,
    SentenceWindowRAG,
    get_canonical_strategies,
    get_default_embed_model,
    get_default_reranker,
)


def run_benchmark(
    model_name: str = "qwen2.5:3b",
    skip_trulens: bool = False,
) -> tuple[list[StrategyAggregateMetrics], list[StrategyCaseResult]]:
    """Ejecuta la comparativa de estrategias RAG evaluando la Tríada de TruLens con aceleración GPU."""
    indices_dir = Path("data/indices")
    indices_dir.mkdir(parents=True, exist_ok=True)

    print("--- 1. Cargando documentos desde data/json ---", flush=True)
    documents = load_documents_from_json(json_dir="data/json", doc_codes=BENCHMARK_DOC_CODES)
    print(f"Cargados {len(documents)} documentos representativos ({', '.join(BENCHMARK_DOC_CODES)}).", flush=True)

    print("\n--- 2. Inicializando modelo de embeddings local bge-m3 (100% GPU) ---", flush=True)
    embed_model = get_default_embed_model(model_name="bge-m3", batch_size=32)

    print(f"\n--- 3. Inicializando modelo LLM ({model_name}, 100% GPU) ---", flush=True)
    llm = Ollama(
        model=model_name,
        temperature=0.1,
        request_timeout=60.0,
        additional_kwargs={
            "num_ctx": 4096,
            "num_predict": 350,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
        },
    )
    trulens_provider = get_default_trulens_provider()

    print("\n--- 4. Cargando estrategias RAG canónicas y variantes con Jina-v3.5 Reranker ---", flush=True)
    t0 = time.perf_counter()
    canonical_map = get_canonical_strategies(
        documents,
        embed_model=embed_model,
        indices_dir=indices_dir,
    )
    reranker = get_default_reranker()

    strategies: list[BaseRAGStrategy] = []
    for strat in canonical_map.values():
        strategies.append(strat)
        reranked_strat = RerankedRAG(
            base_strategy=strat,
            reranker=reranker,
            candidate_k=6,
            name=f"{strat.name} + Reranker (Jina-v3.5)",
        )
        strategies.append(reranked_strat)

    for strat in strategies:
        print(f"-> Estrategia activa: {strat.name}", flush=True)
    print(f"   Todas las estrategias cargadas en {time.perf_counter() - t0:.2f}s", flush=True)

    print("\n--- 5. Evaluando recuperación documental (Fase 1: Embeddings y Reranker) ---", flush=True)
    all_results: list[StrategyCaseResult] = []

    for strat in strategies:
        print(f"Recuperando con estrategia: {strat.name}...", flush=True)
        res = evaluate_retrieval_phase(strat, cases=GOLDEN_CASES, top_k=3)
        all_results.extend(res)

    print(f"\n--- 6. Síntesis de respuestas con {model_name} ---", flush=True)
    t_llm = time.perf_counter()
    run_llm_synthesis_batch(all_results, cases=GOLDEN_CASES, llm=llm)
    print(f"Síntesis completada en {time.perf_counter() - t_llm:.2f}s", flush=True)

    print("\n--- 7. Evaluando la Tríada RAG con TruLens ---", flush=True)
    if trulens_provider and not skip_trulens:
        t_trulens = time.perf_counter()
        evaluate_trulens_triad(all_results, cases=GOLDEN_CASES, provider=trulens_provider)
        print(f"Evaluación TruLens completada en {time.perf_counter() - t_trulens:.2f}s", flush=True)
    else:
        print("TruLens omitido o no disponible.", flush=True)

    strat_names = [s.name for s in strategies]
    aggregated: list[StrategyAggregateMetrics] = []
    print("\n=== RESUMEN POR ESTRATEGIA ===", flush=True)
    for s_name in strat_names:
        s_res = [r for r in all_results if r.strategy_name == s_name]
        agg = aggregate_metrics(s_res)
        aggregated.append(agg)
        print(
            f"  [{agg.strategy_name}]\n"
            f"    Hit: {agg.hit_rate * 100:.1f}% | Hechos: {agg.mean_fact_coverage * 100:.1f}% | "
            f"Anclaje: {agg.mean_groundedness * 100:.1f}%\n"
            f"    Tokens: {agg.mean_tokens:.0f} | Lat. Recup: {agg.mean_latency_ms:.1f}ms | "
            f"Lat. Síntesis: {agg.mean_generation_latency_ms:.1f}ms\n"
            f"    TruLens Context Rel: {agg.mean_trulens_context_relevance * 100:.1f}% | "
            f"Groundedness: {agg.mean_trulens_groundedness * 100:.1f}% | "
            f"Answer Rel: {agg.mean_trulens_answer_relevance * 100:.1f}%",
            flush=True,
        )

    return aggregated, all_results


def format_report_markdown(
    aggregated: list[StrategyAggregateMetrics],
    detailed: list[StrategyCaseResult],
) -> str:
    """Genera un informe analítico comparativo en formato Markdown con la tríada TruLens."""
    lines = [
        "# Informe de Evaluación Comparativa de Métodos RAG con TruLens",
        "",
        "## 1. Tabla Resumen de Rendimiento y Tríada de TruLens",
        "",
        "| Estrategia RAG | Hit Rate Precepto | Cobertura Hechos | Tokens Promedio | Latencia Recup. (ms) | Latencia Síntesis (ms) | Relevancia Contexto | Anclaje (Groundedness) | Relevancia Respuesta |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for m in aggregated:
        lines.append(
            f"| **{m.strategy_name}** | {m.hit_rate * 100:.1f}% | {m.mean_fact_coverage * 100:.1f}% | "
            f"{m.mean_tokens:.0f} | {m.mean_latency_ms:.1f} ms | {m.mean_generation_latency_ms:.1f} ms | {m.mean_trulens_context_relevance * 100:.1f}% | "
            f"{m.mean_trulens_groundedness * 100:.1f}% | {m.mean_trulens_answer_relevance * 100:.1f}% |"
        )

    lines.extend([
        "",
        "## 2. Comparativa Directa: Base vs. Con Reranker (Jina-v3.5)",
        "",
        "| Estrategia Base | Configuración | Hit Rate | Anclaje (Groundedness) | Rel. Respuesta | Rel. Contexto | Tokens Promedio | Lat. Recup (ms) | Lat. Síntesis (ms) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    base_names = [
        "Article-Level (Baseline)",
        "Sentence-Window (w=3)",
        "Auto-Merging [1024, 256] (r=0.5)",
        "Hybrid (Vector + BM25)",
    ]
    agg_map = {m.strategy_name: m for m in aggregated}
    for b_name in base_names:
        m_base = agg_map.get(b_name)
        m_rerank = agg_map.get(f"{b_name} + Reranker (Jina-v3.5)")
        if m_base:
            lines.append(
                f"| **{b_name}** | Base (Sin Reranker) | {m_base.hit_rate * 100:.1f}% | "
                f"{m_base.mean_trulens_groundedness * 100:.1f}% | {m_base.mean_trulens_answer_relevance * 100:.1f}% | "
                f"{m_base.mean_trulens_context_relevance * 100:.1f}% | {m_base.mean_tokens:.0f} | {m_base.mean_latency_ms:.1f} ms | {m_base.mean_generation_latency_ms:.1f} ms |"
            )
        if m_rerank:
            lines.append(
                f"| **{b_name}** | + Reranker (Jina-v3.5) | {m_rerank.hit_rate * 100:.1f}% | "
                f"{m_rerank.mean_trulens_groundedness * 100:.1f}% | {m_rerank.mean_trulens_answer_relevance * 100:.1f}% | "
                f"{m_rerank.mean_trulens_context_relevance * 100:.1f}% | {m_rerank.mean_tokens:.0f} | {m_rerank.mean_latency_ms:.1f} ms | {m_rerank.mean_generation_latency_ms:.1f} ms |"
            )

    lines.extend([
        "",
        "## 3. Análisis Detallado por Caso Práctico",
        "",
    ])

    for case in GOLDEN_CASES:
        lines.append(f"### Caso `{case.case_id}`: {case.category}")
        lines.append(f"- **Pregunta:** *{case.query}*")
        lines.append(f"- **Hechos normativos requeridos:** `{', '.join(case.expected_facts)}`")
        lines.append(f"- **Fuentes oficiales esperadas:** `{', '.join(case.expected_doc_codes)}`")
        lines.append("")
        lines.append("| Estrategia | Hit | Cobertura | Ctx Rel | Groundedness | Ans Rel | Tokens | Síntesis LLM |")
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |")

        case_results = [r for r in detailed if r.case_id == case.case_id]
        for r in case_results:
            hit_icon = "✅" if r.hit else "❌"
            clean_ans = r.generated_answer.replace("\n", " ").strip()
            if len(clean_ans) > 100:
                clean_ans = clean_ans[:97] + "..."
            lines.append(
                f"| {r.strategy_name} | {hit_icon} | {r.fact_coverage * 100:.0f}% | "
                f"{r.trulens_context_relevance * 100:.0f}% | {r.trulens_groundedness * 100:.0f}% | "
                f"{r.trulens_answer_relevance * 100:.0f}% | {r.total_tokens_approx} | {clean_ans} |"
            )
        lines.append("")

    return "\n".join(lines)


def run_trulens_leaderboard_benchmark(
    strategies: list[BaseRAGStrategy],
    llm: Ollama,
    queries: Sequence[str] | None = None,
    model_name: str = "qwen2.5:1.5b",
) -> pd.DataFrame:
    """Ejecuta la evaluación canónica de la Tríada RAG con TruLens sin requerir Golden Cases."""
    eval_queries = list(queries) if queries else [c.query for c in GOLDEN_CASES]
    session = TruLensRAGSession(model_name=model_name)
    print("\n--- Ejecutando evaluación canónica de TruLens (Tríada RAG sin etiquetas) ---", flush=True)

    for strat in strategies:
        print(f"-> Instrumentando y evaluando con TruLlama: {strat.name}...", flush=True)
        tru_recorder = session.instrument_strategy(
            strategy=strat,
            llm=llm,
            app_name="chat_autonomos",
            app_version=strat.name,
            top_k=3,
        )
        session.evaluate_queries(tru_recorder, queries=eval_queries, compute_feedbacks=True)

    leaderboard = session.get_leaderboard()
    print("\n=== TABLERO DE EXPERIMENTOS TRULENS (LEADERBOARD) ===", flush=True)
    print(leaderboard, flush=True)
    return leaderboard


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluación comparativa RAG y monitoreo con TruLens.")
    parser.add_argument("--model", default="qwen2.5:3b", help="Modelo LLM a utilizar")
    parser.add_argument("--skip-trulens", action="store_true", help="Omitir evaluación de TruLens")
    parser.add_argument("--native-trulens", action="store_true", help="Ejecutar benchmarking canónico con TruLens Leaderboard")
    parser.add_argument("--dashboard", action="store_true", help="Lanzar el dashboard interactivo de TruLens")
    args = parser.parse_args()

    if args.dashboard:
        session = TruLensRAGSession(model_name=args.model)
        print("Lanzando dashboard interactivo de TruLens en http://localhost:8501...")
        session.run_dashboard(port=8501)
    elif args.native_trulens:
        indices_dir = Path("data/indices")
        documents = load_documents_from_json(json_dir="data/json", doc_codes=BENCHMARK_DOC_CODES)
        embed_model = get_default_embed_model(model_name="bge-m3", batch_size=32)
        llm = Ollama(model=args.model, request_timeout=60.0)
        canonical_map = get_canonical_strategies(documents, embed_model=embed_model, indices_dir=indices_dir)
        run_trulens_leaderboard_benchmark(list(canonical_map.values()), llm=llm, model_name=args.model)
    else:
        agg, det = run_benchmark(
            model_name=args.model,
            skip_trulens=args.skip_trulens,
        )
        report = format_report_markdown(agg, det)
        model_tag = args.model.replace(":", "_").replace("/", "_")
        out_path = Path(f"docs/EVALUACION-RAG-{model_tag}.md")
        out_path.write_text(report, encoding="utf-8")
        Path("docs/EVALUACION-RAG-ESTRATEGIAS.md").write_text(report, encoding="utf-8")
        print(f"\nInforme guardado con éxito en: {out_path} y docs/EVALUACION-RAG-ESTRATEGIAS.md", flush=True)

