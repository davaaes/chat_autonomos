from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Sequence

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

from llama_index.core import Document
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

from legal_rag_boe.indexing.evaluator import (
    BENCHMARK_DOC_CODES,
    GOLDEN_CASES,
    StrategyAggregateMetrics,
    StrategyCaseResult,
    aggregate_metrics,
    evaluate_retrieval_phase,
    evaluate_trulens_triad,
    get_default_trulens_provider,
    run_llm_synthesis_batch,
)
from legal_rag_boe.indexing.loader import load_documents_from_json
from legal_rag_boe.indexing.strategies import (
    BaseRAGStrategy,
    get_automerging_configurations,
    get_default_embed_model,
)


def run_automerging_benchmark(
    model_name: str = "qwen2.5:3b",
    skip_trulens: bool = False,
    top_k: int = 3,
) -> tuple[list[StrategyAggregateMetrics], list[StrategyCaseResult]]:
    """Ejecuta la comparativa de configuraciones AutoMerging con la Tríada de TruLens."""
    indices_dir = Path("data/indices")
    indices_dir.mkdir(parents=True, exist_ok=True)

    print("--- 1. Cargando documentos desde data/json ---", flush=True)
    documents = load_documents_from_json(json_dir="data/json", doc_codes=BENCHMARK_DOC_CODES)
    print(f"Cargados {len(documents)} documentos ({', '.join(BENCHMARK_DOC_CODES)}).", flush=True)

    print("\n--- 2. Inicializando modelo de embeddings bge-m3 ---", flush=True)
    embed_model = get_default_embed_model(model_name="bge-m3", batch_size=32)

    print(f"\n--- 3. Inicializando LLM generador ({model_name}) y evaluador TruLens ---", flush=True)
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

    print("\n--- 4. Cargando configuraciones de Auto-Merging ---", flush=True)
    t0 = time.perf_counter()
    configs_map = get_automerging_configurations(documents, embed_model, indices_dir)
    strategies = list(configs_map.values())
    for strat in strategies:
        print(f"-> Configuración activa: {strat.name}", flush=True)
    print(f"   Configuraciones listas en {time.perf_counter() - t0:.2f}s", flush=True)

    print("\n--- 5. Fase de Recuperación Vectorial y Fusión Jerárquica ---", flush=True)
    all_results: list[StrategyCaseResult] = []
    for strat in strategies:
        print(f"Recuperando con {strat.name}...", flush=True)
        res = evaluate_retrieval_phase(strat, cases=GOLDEN_CASES, top_k=top_k)
        all_results.extend(res)

    print(f"\n--- 6. Fase de Síntesis Jurídica con {model_name} ---", flush=True)
    t_llm = time.perf_counter()
    run_llm_synthesis_batch(all_results, cases=GOLDEN_CASES, llm=llm)
    print(f"Síntesis finalizada en {time.perf_counter() - t_llm:.2f}s", flush=True)

    print("\n--- 7. Evaluación de la Tríada RAG con TruLens ---", flush=True)
    if trulens_provider and not skip_trulens:
        t_trulens = time.perf_counter()
        evaluate_trulens_triad(all_results, cases=GOLDEN_CASES, provider=trulens_provider)
        print(f"Evaluación TruLens finalizada en {time.perf_counter() - t_trulens:.2f}s", flush=True)
    else:
        print("TruLens omitido o no disponible.", flush=True)

    aggregated: list[StrategyAggregateMetrics] = []
    print("\n=== RESUMEN COMPARATIVO DE CONFIGURACIONES AUTOMERGING ===", flush=True)
    for strat in strategies:
        s_res = [r for r in all_results if r.strategy_name == strat.name]
        agg = aggregate_metrics(s_res)
        aggregated.append(agg)
        print(
            f"  [{agg.strategy_name}]\n"
            f"    Hit: {agg.hit_rate * 100:.1f}% | Hechos: {agg.mean_fact_coverage * 100:.1f}%\n"
            f"    Tokens: {agg.mean_tokens:.0f} | Lat. Recup: {agg.mean_latency_ms:.1f}ms | Lat. Síntesis: {agg.mean_generation_latency_ms:.1f}ms\n"
            f"    TruLens Ctx Rel: {agg.mean_trulens_context_relevance * 100:.1f}% | Groundedness: {agg.mean_trulens_groundedness * 100:.1f}% | Answer Rel: {agg.mean_trulens_answer_relevance * 100:.1f}%",
            flush=True,
        )

    return aggregated, all_results


def format_automerging_report(
    aggregated: list[StrategyAggregateMetrics],
    detailed: list[StrategyCaseResult],
) -> str:
    """Genera informe analítico en Markdown con la comparativa de AutoMerging y TruLens."""
    lines = [
        "# Estudio Comparativo de Configuraciones de AutoMerging con TruLens",
        "",
        "## 1. Tabla Resumen Global (Métricas de Recuperación y Tríada TruLens)",
        "",
        "| Configuración Auto-Merging | Hit Rate | Cobertura Hechos | Tokens Promedio | Lat. Recup (ms) | Lat. Síntesis (ms) | Relevancia Contexto | Anclaje (Groundedness) | Relevancia Respuesta |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for m in aggregated:
        lines.append(
            f"| **{m.strategy_name}** | {m.hit_rate * 100:.1f}% | {m.mean_fact_coverage * 100:.1f}% | "
            f"{m.mean_tokens:.0f} | {m.mean_latency_ms:.1f} ms | {m.mean_generation_latency_ms:.1f} ms | "
            f"{m.mean_trulens_context_relevance * 100:.1f}% | {m.mean_trulens_groundedness * 100:.1f}% | "
            f"{m.mean_trulens_answer_relevance * 100:.1f}% |"
        )

    lines.extend([
        "",
        "## 2. Comparativa Directa: Base vs. + Jina-v3.5 Reranker frente al Canónico",
        "",
        "| Arquitectura Jerárquica | Modo | Hit Rate | Anclaje (Groundedness) | Rel. Respuesta | Rel. Contexto | Tokens Promedio | Lat. Recup (ms) | Lat. Síntesis (ms) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    agg_map = {m.strategy_name: m for m in aggregated}
    target_bases = [
        ("AM-Canónico [1024, 512] (r=0.6)", "AM-Canónico [1024, 512] (r=0.6) + Jina-v3.5"),
        ("AM-Granularidad-Fina [1024, 256] (r=0.5)", "AM-Granularidad-Fina [1024, 256] (r=0.5) + Jina-v3.5"),
        ("AM-Jerárquico-3Niveles [1024, 512, 256] (r=0.5)", "AM-Jerárquico-3Niveles [1024, 512, 256] (r=0.5) + Jina-v3.5"),
    ]

    for base_label, rerank_label in target_bases:
        m_b = agg_map.get(base_label)
        m_r = agg_map.get(rerank_label)
        if m_b:
            lines.append(
                f"| **{base_label}** | Base (Sin Reranker) | {m_b.hit_rate * 100:.1f}% | "
                f"{m_b.mean_trulens_groundedness * 100:.1f}% | {m_b.mean_trulens_answer_relevance * 100:.1f}% | "
                f"{m_b.mean_trulens_context_relevance * 100:.1f}% | {m_b.mean_tokens:.0f} | "
                f"{m_b.mean_latency_ms:.1f} ms | {m_b.mean_generation_latency_ms:.1f} ms |"
            )
        if m_r:
            lines.append(
                f"| **{base_label}** | + Reranker (Jina-v3.5) | {m_r.hit_rate * 100:.1f}% | "
                f"{m_r.mean_trulens_groundedness * 100:.1f}% | {m_r.mean_trulens_answer_relevance * 100:.1f}% | "
                f"{m_r.mean_trulens_context_relevance * 100:.1f}% | {m_r.mean_tokens:.0f} | "
                f"{m_r.mean_latency_ms:.1f} ms | {m_r.mean_generation_latency_ms:.1f} ms |"
            )

    lines.extend([
        "",
        "## 3. Detalle por Caso Práctico (Golden Cases)",
        "",
    ])

    for case in GOLDEN_CASES:
        lines.append(f"### Caso `{case.case_id}`: {case.category}")
        lines.append(f"- **Pregunta:** *{case.query}*")
        lines.append(f"- **Hechos esperados:** `{', '.join(case.expected_facts)}`")
        lines.append(f"- **Fuentes esperadas:** `{', '.join(case.expected_doc_codes)}`")
        lines.append("")
        lines.append("| Configuración | Hit | Hechos | Ctx Rel | Groundedness | Ans Rel | Tokens | Síntesis LLM |")
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |")

        c_results = [r for r in detailed if r.case_id == case.case_id]
        for r in c_results:
            hit_ico = "✅" if r.hit else "❌"
            ans_clean = r.generated_answer.replace("\n", " ").strip()
            if len(ans_clean) > 95:
                ans_clean = ans_clean[:92] + "..."
            lines.append(
                f"| {r.strategy_name} | {hit_ico} | {r.fact_coverage * 100:.0f}% | "
                f"{r.trulens_context_relevance * 100:.0f}% | {r.trulens_groundedness * 100:.0f}% | "
                f"{r.trulens_answer_relevance * 100:.0f}% | {r.total_tokens_approx} | {ans_clean} |"
            )
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark comparativo de configuraciones AutoMerging con TruLens.")
    parser.add_argument("--model", default="qwen2.5:3b", help="Modelo LLM generador")
    parser.add_argument("--skip-trulens", action="store_true", help="Omitir evaluación de la Tríada TruLens")
    parser.add_argument("--top-k", type=int, default=3, help="Top K nodos finales devueltos")
    args = parser.parse_args()

    agg_results, det_results = run_automerging_benchmark(
        model_name=args.model,
        skip_trulens=args.skip_trulens,
        top_k=args.top_k,
    )

    report_content = format_automerging_report(agg_results, det_results)
    out_file = Path("docs/EVALUACION-AUTOMERGING-TRULENS.md")
    out_file.write_text(report_content, encoding="utf-8")
    print(f"\nInforme generado y guardado en: {out_file}", flush=True)

