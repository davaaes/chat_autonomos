from __future__ import annotations

from legal_rag_boe.indexing.evaluator import (
    GOLDEN_CASES,
    StrategyCaseResult,
    aggregate_metrics,
)
from legal_rag_boe.indexing.loader import load_documents_from_json


def test_load_documents_from_json_with_filter() -> None:
    docs = load_documents_from_json(json_dir="data/json", doc_codes=["DOC-14"])
    assert len(docs) == 6
    for doc in docs:
        assert doc.metadata["doc_code"] == "DOC-14"
        assert doc.metadata["authority"] == "TGSS"
        assert doc.metadata["in_scope"] is True
        assert "[TGSS |" in doc.text


def test_golden_cases_structure_validity() -> None:
    assert len(GOLDEN_CASES) == 6
    for case in GOLDEN_CASES:
        assert case.case_id.startswith("Q")
        assert len(case.expected_doc_codes) > 0
        assert len(case.expected_facts) > 0
        assert len(case.query) > 10


def test_aggregate_metrics_calculation() -> None:
    sample_results = [
        StrategyCaseResult(
            strategy_name="MockStrategy",
            case_id="Q1",
            hit=True,
            fact_coverage=1.0,
            total_tokens_approx=200,
            retrieval_latency_ms=10.0,
            groundedness_score=1.0,
            generated_answer="Respuesta 1",
            retrieved_sources=["DOC-14"],
        ),
        StrategyCaseResult(
            strategy_name="MockStrategy",
            case_id="Q2",
            hit=False,
            fact_coverage=0.5,
            total_tokens_approx=400,
            retrieval_latency_ms=20.0,
            groundedness_score=0.5,
            generated_answer="Respuesta 2",
            retrieved_sources=["DOC-01"],
        ),
    ]

    agg = aggregate_metrics(sample_results)
    assert agg.strategy_name == "MockStrategy"
    assert agg.hit_rate == 0.5
    assert agg.mean_fact_coverage == 0.75
    assert agg.mean_tokens == 300.0
    assert agg.mean_latency_ms == 15.0
    assert agg.mean_groundedness == 0.75
    assert agg.mean_trulens_context_relevance == 0.0


def test_evaluate_trulens_triad_with_mock() -> None:
    from legal_rag_boe.indexing.evaluator import evaluate_trulens_triad

    class MockProvider:
        def context_relevance(self, question: str, context: str) -> float:
            return 0.8

        def groundedness_measure_with_cot_reasons(self, source: str, statement: str) -> tuple[float, dict]:
            return 0.9, {}

        def relevance(self, prompt: str, response: str) -> float:
            return 1.0

    cases = [GOLDEN_CASES[0]]
    results = [
        StrategyCaseResult(
            strategy_name="MockRAG",
            case_id=cases[0].case_id,
            hit=True,
            fact_coverage=1.0,
            total_tokens_approx=100,
            retrieval_latency_ms=5.0,
            groundedness_score=1.0,
            generated_answer="Tramo 1 cuota 230 euros.",
            context_text="Tramo 1 ingresos 670 base 735 cuota 230.",
        )
    ]

    evaluate_trulens_triad(results, cases=cases, provider=MockProvider())  # type: ignore[arg-type]
    assert results[0].trulens_context_relevance == 0.8
    assert results[0].trulens_groundedness == 0.9
    assert results[0].trulens_answer_relevance == 1.0

    agg = aggregate_metrics(results)
    assert agg.mean_trulens_context_relevance == 0.8
    assert agg.mean_trulens_groundedness == 0.9
    assert agg.mean_trulens_answer_relevance == 1.0


def test_get_canonical_strategies_structure() -> None:
    from llama_index.core import Document
    from llama_index.core.embeddings import MockEmbedding
    from legal_rag_boe.indexing.strategies import get_canonical_strategies

    docs = [Document(text="Artículo 1. Objeto y ámbito de aplicación del RETA.", metadata={"doc_code": "DOC-14"})]
    canonical = get_canonical_strategies(docs, embed_model=MockEmbedding(embed_dim=8))  # type: ignore[arg-type]

    assert set(canonical.keys()) == {"baseline", "sentence_window", "auto_merging", "hybrid"}
    assert canonical["baseline"].name == "Article-Level (Baseline)"
    assert canonical["sentence_window"].name == "Sentence-Window (w=3)"
    assert canonical["auto_merging"].name == "Auto-Merging [1024, 256] (r=0.5)"
    assert canonical["hybrid"].name == "Hybrid (Vector + BM25)"


def test_strategy_as_retriever_and_query_engine() -> None:
    from llama_index.core import Document
    from llama_index.core.embeddings import MockEmbedding
    from llama_index.core.llms import MockLLM
    from legal_rag_boe.indexing.strategies import ArticleLevelRAG

    docs = [Document(text="Artículo 1. Objeto del RETA.", metadata={"doc_code": "DOC-14"})]
    strat = ArticleLevelRAG(docs, embed_model=MockEmbedding(embed_dim=8))  # type: ignore[arg-type]

    retriever = strat.as_retriever(top_k=1)
    nodes = retriever.retrieve("RETA")
    assert len(nodes) == 1

    query_engine = strat.as_query_engine(llm=MockLLM(), top_k=1)
    resp = query_engine.query("RETA")
    assert resp is not None


def test_trulens_triad_feedbacks_creation() -> None:
    from legal_rag_boe.indexing.evaluator import get_trulens_triad_feedbacks

    class MockTruLensProvider:
        def context_relevance(self, question: str, context: str) -> float:
            return 1.0

        def groundedness_measure_with_cot_reasons(self, source: str, statement: str) -> tuple[float, dict]:
            return 1.0, {}

        def relevance(self, prompt: str, response: str) -> float:
            return 1.0

    feedbacks = get_trulens_triad_feedbacks(MockTruLensProvider())  # type: ignore[arg-type]
    assert len(feedbacks) == 3
    names = [f.name for f in feedbacks]
    assert "Context Relevance" in names
    assert "Groundedness" in names
    assert "Answer Relevance" in names



