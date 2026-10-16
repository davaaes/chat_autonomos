from __future__ import annotations

from llama_index.core import Document
from llama_index.core.embeddings import MockEmbedding

from legal_rag_boe.indexing.evaluator import (
    EvaluationCase,
    StrategyCaseResult,
    aggregate_metrics,
    evaluate_retrieval_phase,
    evaluate_trulens_triad,
)
from legal_rag_boe.indexing.strategies import AutoMergingRAG


def test_automerging_configurations_instantiation() -> None:
    embed_model = MockEmbedding(embed_dim=8)
    docs = [
        Document(
            text="Artículo 1. Objeto y ámbito de aplicación del régimen de autónomos.",
            metadata={"doc_code": "DOC-14"},
        ),
        Document(
            text="Artículo 2. Cotización por rendimientos netos en quince tramos.",
            metadata={"doc_code": "DOC-14"},
        ),
    ]

    base_am = AutoMergingRAG(
        documents=docs,
        embed_model=embed_model,  # type: ignore[arg-type]
        chunk_sizes=[64, 32],
        ratio_threshold=0.6,
        leaf_top_k=6,
        name="AM-Base",
    )
    assert base_am.name == "AM-Base"
    assert base_am.ratio_threshold == 0.6
    assert base_am.leaf_top_k == 6

    am_agresivo = AutoMergingRAG(
        documents=[],
        embed_model=embed_model,  # type: ignore[arg-type]
        ratio_threshold=0.2,
        leaf_top_k=6,
        name="AM-Agresivo",
        base_strategy=base_am,
    )
    assert am_agresivo.ratio_threshold == 0.2
    assert am_agresivo.storage_context == base_am.storage_context

    am_estricto = AutoMergingRAG(
        documents=[],
        embed_model=embed_model,  # type: ignore[arg-type]
        ratio_threshold=0.9,
        leaf_top_k=6,
        name="AM-Estricto",
        base_strategy=base_am,
    )
    assert am_estricto.ratio_threshold == 0.9


def test_automerging_retrieval_phase_evaluation() -> None:
    embed_model = MockEmbedding(embed_dim=8)
    docs = [
        Document(
            text="Artículo 1. El alta en el censo y en la Seguridad Social se tramita con el DUE.",
            metadata={"doc_code": "DOC-16"},
        )
    ]
    strat = AutoMergingRAG(
        documents=docs,
        embed_model=embed_model,  # type: ignore[arg-type]
        chunk_sizes=[64, 32],
        ratio_threshold=0.5,
        name="AM-Test",
    )
    test_case = EvaluationCase(
        case_id="Q-TEST",
        category="Prueba",
        query="¿Qué permite tramitar el DUE ante la Seguridad Social?",
        expected_doc_codes=["DOC-16"],
        expected_facts=["DUE", "Seguridad Social"],
        structure_type="procedimiento",
    )
    results = evaluate_retrieval_phase(strat, cases=[test_case], top_k=1)
    assert len(results) == 1
    assert results[0].hit is True
    assert results[0].fact_coverage == 1.0
    assert results[0].strategy_name == "AM-Test"


def test_automerging_trulens_triad_evaluation_mock() -> None:
    class MockTruLensProvider:
        def context_relevance(self, question: str, context: str) -> float:
            return 0.85

        def groundedness_measure_with_cot_reasons(self, source: str, statement: str) -> tuple[float, dict]:
            return 0.95, {"reason": "Totalmente sustentado en el contexto"}

        def relevance(self, prompt: str, response: str) -> float:
            return 0.90

    test_case = EvaluationCase(
        case_id="Q-AM-MOCK",
        category="Test AutoMerging",
        query="¿Cuál es la base de cotización?",
        expected_doc_codes=["DOC-14"],
        expected_facts=["base"],
        structure_type="tabla",
    )
    res = StrategyCaseResult(
        strategy_name="AM-Canónico",
        case_id="Q-AM-MOCK",
        hit=True,
        fact_coverage=1.0,
        total_tokens_approx=150,
        retrieval_latency_ms=12.0,
        groundedness_score=1.0,
        generated_answer="La base de cotización mínima es de 735 euros.",
        context_text="La base de cotización mínima para el tramo 1 es de 735 euros mensuales.",
        retrieved_sources=["DOC-14"],
    )

    evaluate_trulens_triad([res], cases=[test_case], provider=MockTruLensProvider())
    assert res.trulens_context_relevance == 0.85
    assert res.trulens_groundedness == 0.95
    assert res.trulens_answer_relevance == 0.90

    agg = aggregate_metrics([res])
    assert agg.strategy_name == "AM-Canónico"
    assert agg.mean_trulens_context_relevance == 0.85
    assert agg.mean_trulens_groundedness == 0.95
    assert agg.mean_trulens_answer_relevance == 0.90


def test_automerging_target_configurations_definition() -> None:
    from pathlib import Path
    from unittest.mock import MagicMock
    from legal_rag_boe.indexing.strategies import get_automerging_configurations

    embed_model = MockEmbedding(embed_dim=8)
    mock_reranker = MagicMock()
    mock_reranker.name = "Jina-v3.5"
    docs = [Document(text="Artículo 1. Texto de prueba.", metadata={"doc_code": "DOC-14"})]
    configs = get_automerging_configurations(
        docs,
        embed_model=embed_model,  # type: ignore[arg-type]
        indices_dir=Path("data/indices"),
        reranker=mock_reranker,
    )

    expected_keys = {
        "canonico",
        "canonico_reranker",
        "granularidad_fina",
        "granularidad_fina_reranker",
        "jerarquico_3niveles",
        "jerarquico_3niveles_reranker",
    }
    assert set(configs.keys()) == expected_keys
    assert configs["canonico"].name == "AM-Canónico [1024, 512] (r=0.6)"
    assert configs["canonico_reranker"].name == "AM-Canónico [1024, 512] (r=0.6) + Jina-v3.5"
    assert configs["granularidad_fina"].name == "AM-Granularidad-Fina [1024, 256] (r=0.5)"
    assert configs["granularidad_fina_reranker"].name == "AM-Granularidad-Fina [1024, 256] (r=0.5) + Jina-v3.5"
    assert configs["jerarquico_3niveles"].name == "AM-Jerárquico-3Niveles [1024, 512, 256] (r=0.5)"
    assert configs["jerarquico_3niveles_reranker"].name == "AM-Jerárquico-3Niveles [1024, 512, 256] (r=0.5) + Jina-v3.5"


