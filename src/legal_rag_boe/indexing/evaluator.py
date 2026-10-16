import os
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd
from llama_index.core.llms import LLM
from llama_index.llms.ollama import Ollama

os.environ.setdefault("TRULENS_OTEL_TRACING", "0")

try:
    from trulens.apps.llamaindex import TruLlama
    from trulens.core import Feedback, TruSession
    from trulens.core.schema.record import Record
    from trulens.providers.ollama import Ollama as TruLensOllama
except ImportError:
    TruLlama = None  # type: ignore[misc,assignment]
    Feedback = None  # type: ignore[misc,assignment]
    TruSession = None  # type: ignore[misc,assignment]
    Record = None  # type: ignore[misc,assignment]
    TruLensOllama = None  # type: ignore[misc,assignment]

from legal_rag_boe.indexing.strategies import BaseRAGStrategy


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    category: str
    query: str
    expected_doc_codes: list[str]
    expected_facts: list[str]
    structure_type: str


@dataclass
class StrategyCaseResult:
    strategy_name: str
    case_id: str
    hit: bool
    fact_coverage: float
    total_tokens_approx: int
    retrieval_latency_ms: float
    groundedness_score: float
    generated_answer: str
    retrieved_sources: list[str] = field(default_factory=list)
    context_text: str = ""
    trulens_context_relevance: float = 0.0
    trulens_groundedness: float = 0.0
    trulens_answer_relevance: float = 0.0
    generation_latency_ms: float = 0.0


@dataclass
class StrategyAggregateMetrics:
    strategy_name: str
    hit_rate: float
    mean_fact_coverage: float
    mean_tokens: float
    mean_latency_ms: float
    mean_groundedness: float
    mean_trulens_context_relevance: float = 0.0
    mean_trulens_groundedness: float = 0.0
    mean_trulens_answer_relevance: float = 0.0
    mean_generation_latency_ms: float = 0.0


BENCHMARK_DOC_CODES: list[str] = [
    "DOC-01", "DOC-02", "DOC-05", "DOC-06", "DOC-08", "DOC-13", "DOC-14", "DOC-15", "DOC-16"
]


GOLDEN_CASES: list[EvaluationCase] = [
    EvaluationCase(
        case_id="Q1-RETA-TRAMOS",
        category="Seguridad Social / RETA",
        query="¿Cuál es la base mínima de cotización y la cuota mensual mínima para un autónomo en el tramo 1 con ingresos menores o iguales a 670 euros?",
        expected_doc_codes=["DOC-14", "DOC-06"],
        expected_facts=["tramo 1", "735,29", "230"],
        structure_type="tabla_tramos",
    ),
    EvaluationCase(
        case_id="Q2-IVA-MODELO303",
        category="Fiscalidad / IVA",
        query="¿Hasta qué día de enero se presenta el modelo 303 del cuarto trimestre y en qué casilla se declara el resultado de la autoliquidación?",
        expected_doc_codes=["DOC-13", "DOC-04", "DOC-05"],
        expected_facts=["30 de enero", "120"],
        structure_type="instruccion_casillas",
    ),
    EvaluationCase(
        case_id="Q3-RETA-TARIFA-PLANA",
        category="Seguridad Social / Bonificaciones",
        query="¿Cuántos meses dura inicialmente la cuota reducida de 80 euros para nuevos autónomos y qué condición de ingresos se exige para prorrogarla?",
        expected_doc_codes=["DOC-14", "DOC-06", "DOC-01"],
        expected_facts=["12 meses", "salario mínimo interprofesional", "80"],
        structure_type="norma_condicional",
    ),
    EvaluationCase(
        case_id="Q4-IRPF-SUMINISTROS",
        category="Fiscalidad / IRPF",
        query="¿Cómo y en qué porcentaje se calculan los gastos deducibles de suministros como agua, gas o electricidad si el autónomo trabaja en su vivienda habitual?",
        expected_doc_codes=["DOC-05", "DOC-11"],
        expected_facts=["30", "proporción", "superficie"],
        structure_type="formula_fiscal",
    ),
    EvaluationCase(
        case_id="Q5-ALTA-DUE-CIRCE",
        category="Trámites / Inicio de Actividad",
        query="¿Qué permite tramitar el Documento Único Electrónico (DUE) a través de CIRCE para un autónomo ante la Agencia Tributaria y la Seguridad Social?",
        expected_doc_codes=["DOC-16", "DOC-08"],
        expected_facts=["Documento Único Electrónico", "censo", "Seguridad Social"],
        structure_type="procedimiento_administrativo",
    ),
    EvaluationCase(
        case_id="Q6-LABORAL-CCC",
        category="Laboral / Contratación",
        query="¿Qué código debe solicitar un autónomo ante la Tesorería General de la Seguridad Social para poder contratar a su primer empleado y con qué modelo?",
        expected_doc_codes=["DOC-15", "DOC-06", "DOC-09"],
        expected_facts=["Código de Cuenta de Cotización", "TA.6"],
        structure_type="procedimiento_laboral",
    ),
]


def evaluate_retrieval_phase(
    strategy: BaseRAGStrategy,
    cases: Sequence[EvaluationCase] = GOLDEN_CASES,
    top_k: int = 3,
) -> list[StrategyCaseResult]:
    """Evalúa la fase de recuperación de una estrategia de forma aislada sin llamadas al LLM."""
    results: list[StrategyCaseResult] = []

    for case in cases:
        t0 = time.perf_counter()
        nodes = strategy.retrieve(case.query, top_k=top_k)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        retrieved_codes: list[str] = []
        context_parts: list[str] = []
        for n in nodes:
            doc_code = str(n.node.metadata.get("doc_code", ""))
            retrieved_codes.append(doc_code)
            context_parts.append(n.node.get_content())

        hit = any(expected in retrieved_codes for expected in case.expected_doc_codes)
        full_context = "\n---\n".join(context_parts)
        full_context_lower = full_context.lower()

        facts_found = sum(1 for fact in case.expected_facts if fact.lower() in full_context_lower)
        fact_coverage = facts_found / len(case.expected_facts) if case.expected_facts else 1.0
        total_tokens = int(len(full_context.split()) * 1.3)

        results.append(
            StrategyCaseResult(
                strategy_name=strategy.name,
                case_id=case.case_id,
                hit=hit,
                fact_coverage=fact_coverage,
                total_tokens_approx=total_tokens,
                retrieval_latency_ms=dt_ms,
                groundedness_score=0.0,
                generated_answer="",
                retrieved_sources=retrieved_codes,
                context_text=full_context,
            )
        )

    return results


def run_llm_synthesis_batch(
    results: list[StrategyCaseResult],
    cases: Sequence[EvaluationCase] = GOLDEN_CASES,
    llm: Ollama | None = None,
) -> None:
    """Ejecuta la síntesis y evaluación de anclaje de todos los contextos en una sola pasada."""
    if not llm:
        return

    case_map = {c.case_id: c for c in cases}
    for idx, res in enumerate(results, start=1):
        case = case_map.get(res.case_id)
        if not case or not res.context_text.strip():
            continue

        print(f"    [Síntesis {idx}/{len(results)}] {res.strategy_name} - {case.case_id}...", flush=True)

        prompt = (
            "Eres un asesor legal estricto para autónomos en España.\n"
            "Responde a la pregunta basándote EXCLUSIVAMENTE en el siguiente contexto.\n"
            "Si la información no aparece en el contexto, indica 'No consta en el contexto'.\n\n"
            f"Contexto:\n{res.context_text}\n\n"
            f"Pregunta: {case.query}\nRespuesta:"
        )
        t0_gen = time.perf_counter()
        try:
            raw_resp = llm.complete(prompt)
            res.generation_latency_ms = (time.perf_counter() - t0_gen) * 1000.0
            res.generated_answer = raw_resp.text.strip()
            ans_lower = res.generated_answer.lower()
            ans_facts = sum(1 for fact in case.expected_facts if fact.lower() in ans_lower)
            res.groundedness_score = ans_facts / len(case.expected_facts) if case.expected_facts else 1.0
        except Exception:
            res.generation_latency_ms = (time.perf_counter() - t0_gen) * 1000.0
            res.generated_answer = "Error en generación LLM"
            res.groundedness_score = 0.0


def evaluate_trulens_triad(
    results: list[StrategyCaseResult],
    cases: Sequence[EvaluationCase] = GOLDEN_CASES,
    provider: Any | None = None,
) -> None:
    """Evalúa la tríada RAG de TruLens calculando relevancia contextual, anclaje y relevancia de respuesta."""
    if not provider:
        return

    ctx_rel_cache: dict[tuple[str, str], float] = {}
    groundedness_cache: dict[tuple[str, str], float] = {}
    ans_rel_cache: dict[tuple[str, str], float] = {}

    case_map = {c.case_id: c for c in cases}
    for idx, res in enumerate(results, start=1):
        case = case_map.get(res.case_id)
        if not case or not res.context_text.strip() or not res.generated_answer.strip():
            continue

        print(f"    [TruLens {idx}/{len(results)}] {res.strategy_name} - {case.case_id}...", flush=True)

        c_key = (case.query, res.context_text)
        if c_key in ctx_rel_cache:
            res.trulens_context_relevance = ctx_rel_cache[c_key]
        else:
            try:
                c_score = float(provider.context_relevance(question=case.query, context=res.context_text))
                c_score = min(1.0, max(0.0, c_score))
                ctx_rel_cache[c_key] = c_score
                res.trulens_context_relevance = c_score
            except Exception:
                res.trulens_context_relevance = 0.0

        g_key = (res.context_text, res.generated_answer)
        if g_key in groundedness_cache:
            res.trulens_groundedness = groundedness_cache[g_key]
        else:
            try:
                score_g, _ = provider.groundedness_measure_with_cot_reasons(
                    source=res.context_text,
                    statement=res.generated_answer,
                )
                g_score = min(1.0, max(0.0, float(score_g)))
                groundedness_cache[g_key] = g_score
                res.trulens_groundedness = g_score
            except Exception:
                res.trulens_groundedness = 0.0

        a_key = (case.query, res.generated_answer)
        if a_key in ans_rel_cache:
            res.trulens_answer_relevance = ans_rel_cache[a_key]
        else:
            try:
                a_score = float(provider.relevance(prompt=case.query, response=res.generated_answer))
                a_score = min(1.0, max(0.0, a_score))
                ans_rel_cache[a_key] = a_score
                res.trulens_answer_relevance = a_score
            except Exception:
                res.trulens_answer_relevance = 0.0


def aggregate_metrics(results: Sequence[StrategyCaseResult]) -> StrategyAggregateMetrics:
    """Calcula las métricas agregadas promedio para una estrategia evaluada."""
    if not results:
        return StrategyAggregateMetrics(
            strategy_name="",
            hit_rate=0.0,
            mean_fact_coverage=0.0,
            mean_tokens=0.0,
            mean_latency_ms=0.0,
            mean_groundedness=0.0,
            mean_trulens_context_relevance=0.0,
            mean_trulens_groundedness=0.0,
            mean_trulens_answer_relevance=0.0,
            mean_generation_latency_ms=0.0,
        )

    name = results[0].strategy_name
    n = len(results)
    hits = sum(1 for r in results if r.hit) / n
    coverage = sum(r.fact_coverage for r in results) / n
    tokens = sum(r.total_tokens_approx for r in results) / n
    latency = sum(r.retrieval_latency_ms for r in results) / n
    gen_lat = sum(r.generation_latency_ms for r in results) / n
    groundedness = sum(r.groundedness_score for r in results) / n
    trulens_c_rel = sum(r.trulens_context_relevance for r in results) / n
    trulens_grnd = sum(r.trulens_groundedness for r in results) / n
    trulens_a_rel = sum(r.trulens_answer_relevance for r in results) / n

    return StrategyAggregateMetrics(
        strategy_name=name,
        hit_rate=hits,
        mean_fact_coverage=coverage,
        mean_tokens=tokens,
        mean_latency_ms=latency,
        mean_groundedness=groundedness,
        mean_trulens_context_relevance=trulens_c_rel,
        mean_trulens_groundedness=trulens_grnd,
        mean_trulens_answer_relevance=trulens_a_rel,
        mean_generation_latency_ms=gen_lat,
    )


def get_default_trulens_provider(model_name: str | None = None) -> Any:
    """Instancia el proveedor local de TruLens con Ollama (qwen2.5:7b)."""
    if TruLensOllama is not None:
        target_model = model_name or "qwen2.5:7b"
        return TruLensOllama(model_engine=target_model)
    return None


def get_trulens_triad_feedbacks(provider: Any) -> list[Feedback]:
    """Construye las tres funciones de feedback canónicas de la Tríada RAG con TruLens."""
    if Feedback is None or TruLlama is None:
        raise ImportError("TruLens no está instalado o disponible en el entorno.")

    f_context_relevance = (
        Feedback(provider.context_relevance, name="Context Relevance")
        .on_input()
        .on(TruLlama.select_source_nodes().node.text)
        .aggregate(np.mean)
    )

    f_groundedness = (
        Feedback(provider.groundedness_measure_with_cot_reasons, name="Groundedness")
        .on(TruLlama.select_source_nodes().node.text)
        .on_output()
    )

    f_answer_relevance = (
        Feedback(provider.relevance, name="Answer Relevance")
        .on_input()
        .on_output()
    )

    return [f_context_relevance, f_groundedness, f_answer_relevance]


class TruLensRAGSession:
    """Gestor integral de observabilidad y evaluación no supervisada de la Tríada RAG con TruLens."""

    def __init__(
        self,
        provider: Any | None = None,
        model_name: str | None = None,
        database_url: str | None = None,
    ) -> None:
        if TruSession is None:
            raise ImportError("TruLens no está instalado en el entorno.")

        self.session = TruSession(database_url=database_url) if database_url else TruSession()
        self.provider = provider or get_default_trulens_provider(model_name=model_name)
        if not self.provider:
            raise ValueError("No se pudo inicializar un proveedor válido para TruLens con Ollama.")
        self.feedbacks = get_trulens_triad_feedbacks(self.provider)

    def instrument_strategy(
        self,
        strategy: BaseRAGStrategy,
        llm: LLM,
        app_name: str = "chat_autonomos",
        app_version: str | None = None,
        top_k: int = 3,
    ) -> TruLlama:
        """Envuelve la estrategia RAG como un motor LlamaIndex instrumentado por TruLens."""
        query_engine = strategy.as_query_engine(llm=llm, top_k=top_k)
        version = app_version or strategy.name
        return TruLlama(
            query_engine,
            app_name=app_name,
            app_version=version,
            feedbacks=self.feedbacks,
        )

    def evaluate_queries(
        self,
        tru_recorder: TruLlama,
        queries: Sequence[str],
        compute_feedbacks: bool = True,
    ) -> list[Record]:
        """Ejecuta consultas abiertas sin etiquetas registrando trazas y calculando la Tríada RAG."""
        records: list[Record] = []
        for idx, query in enumerate(queries, start=1):
            print(f"    [{idx}/{len(queries)}] '{query[:50]}...' -> TruLlama + Tríada", flush=True)
            with tru_recorder as recording:
                tru_recorder.app.query(query)

            record = recording.get()
            records.append(record)
            if compute_feedbacks:
                list(
                    self.session.run_feedback_functions(
                        record=record,
                        feedback_functions=self.feedbacks,
                        app=tru_recorder,
                        wait=True,
                    )
                )

        return records

    def get_leaderboard(self) -> pd.DataFrame:
        """Obtiene el panel de control comparativo de experimentos persistido en TruLens."""
        return self.session.get_leaderboard()

    def run_dashboard(self, port: int = 8501, wait: bool = True) -> None:
        """Inicia el panel interactivo de Streamlit de TruLens para auditoría profunda."""
        proc = self.session.run_dashboard(port=port)
        if wait and proc is not None and hasattr(proc, "join"):
            try:
                proc.join()
            except KeyboardInterrupt:
                print("\nDashboard interactivo de TruLens finalizado.")

