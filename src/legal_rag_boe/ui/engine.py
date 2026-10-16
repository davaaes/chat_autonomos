from __future__ import annotations

import os
from typing import Any
from pathlib import Path

from llama_index.llms.ollama import Ollama

from legal_rag_boe.guardrails.inputs import InputGuard
from legal_rag_boe.guardrails.outputs import OutputGuard
from legal_rag_boe.guardrails.pipeline import PipelineRAGSeguro, ResultadoPipeline
from legal_rag_boe.indexing.loader import load_documents_from_json
from legal_rag_boe.indexing.strategies import (
    BaseRAGStrategy,
    RerankedRAG,
    get_canonical_strategies,
    get_default_embed_model,
)

ESTRATEGIAS_METADATA = {
    "auto_merging": {
        "id": "auto_merging",
        "nombre": "Auto-merging",
        "badge": "Recomendado",
        "descripcion": "Junta los párrafos enteros alrededor de la ley para darte la respuesta más completa y detallada. Es la mejor opción si tu consulta tiene requisitos complejos o muchas condiciones.",
        "recomendado": True,
    },
    "sentence_window": {
        "id": "sentence_window",
        "nombre": "Sentence-window",
        "badge": "Más rápido",
        "descripcion": "Es más rápido porque busca frases individuales y directas. Ideal si tienes prisa y necesitas una respuesta corta o una definición sencilla.",
        "recomendado": False,
    },
    "hybrid": {
        "id": "hybrid",
        "nombre": "Híbrido",
        "badge": "Búsqueda exacta",
        "descripcion": "Busca tanto por lo que quieres decir como por palabras exactas. Muy útil si buscas el nombre de un modelo de Hacienda (como el 037 o 303), un formulario o una casilla concreta.",
        "recomendado": False,
    },
}

RERANKER_METADATA = {
    "activo": {
        "id": "jina",
        "nombre": "Jina Reranker",
        "badge": "Mayor exactitud",
        "descripcion": "Hace una doble revisión de los textos encontrados para asegurarse de darte el dato exacto. Tarda unos segundos más, pero afina al máximo las cifras, porcentajes y condiciones.",
    },
    "inactivo": {
        "id": "none",
        "nombre": "Sin Reranker",
        "badge": "Al instante",
        "descripcion": "Te responde de inmediato con las primeras fuentes coincidentes sin hacer una segunda comprobación. Elige esta opción si buscas rapidez.",
    },
}


class RAGEngineManager:
    """Administrador centralizado de modelos, índices y pipelines RAG para la interfaz."""

    _instance: RAGEngineManager | None = None

    def __init__(
        self,
        json_dir: str | Path = "data/json",
        indices_dir: str | Path = "data/indices",
        llm_model: str = "qwen2.5:3b",
        verifier_model: str = "qwen2.5:1.5b",
        ollama_base_url: str | None = None,
    ) -> None:
        self.json_dir = Path(json_dir)
        self.indices_dir = Path(indices_dir)
        self.llm_model_name = llm_model
        self.verifier_model_name = verifier_model
        self.ollama_base_url = ollama_base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.embed_model = get_default_embed_model(base_url=self.ollama_base_url)
        self.documents = load_documents_from_json(json_dir=self.json_dir)
        self.canonical_strategies = get_canonical_strategies(
            self.documents,
            embed_model=self.embed_model,
            indices_dir=self.indices_dir,
        )
        self.llm = Ollama(
            model=self.llm_model_name,
            base_url=self.ollama_base_url,
            temperature=0.1,
            request_timeout=60.0,
            additional_kwargs={
                "num_ctx": 4096,
                "num_predict": 350,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
            },
        )
        self.verifier_llm = Ollama(
            model=self.verifier_model_name,
            base_url=self.ollama_base_url,
            request_timeout=15.0,
            temperature=0.0,
            additional_kwargs={
                "num_ctx": 2048,
                "num_predict": 70,
                "keep_alive": "30m",
            },
        )
        self.input_guard = InputGuard()
        self.output_guard = OutputGuard(
            verifier_llm=self.verifier_llm,
            enable_hallucination_check=True,
        )

    @classmethod
    def get_singleton(cls) -> RAGEngineManager:
        """Obtiene o inicializa la instancia singleton del motor RAG."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def calentar_motores(self, prewarm_llm: bool = True, prewarm_reranker: bool = True) -> None:
        """Precalienta índices, modelos de reranking y LLMs para eliminar la latencia de arranque en frío."""
        if prewarm_reranker:
            try:
                from legal_rag_boe.indexing.rerankers import JinaRerankerV35
                JinaRerankerV35.get_singleton()
            except (ImportError, RuntimeError, ConnectionError, Exception) as err:
                print(f"[AVISO] No se pudo precalentar Jina Reranker: {err}")

        if prewarm_llm:
            try:
                self.llm.complete("inicio")
                self.verifier_llm.complete("inicio")
            except (ConnectionError, RuntimeError, Exception) as err:
                print(f"[AVISO] No se pudo precalentar LLM: {err}")

    def resolver_retriever(self, strategy_id: str, use_reranker: bool) -> BaseRAGStrategy:
        """Resuelve el recuperador canónico y envuelve con reranker si está activo."""
        target_key = strategy_id if strategy_id in self.canonical_strategies else "auto_merging"
        base_strat = self.canonical_strategies[target_key]
        if use_reranker:
            return RerankedRAG(base_strategy=base_strat, candidate_k=6)
        return base_strat

    def ejecutar_consulta(
        self,
        consulta: str,
        strategy_id: str = "auto_merging",
        use_reranker: bool = True,
        top_k: int = 3,
    ) -> dict[str, Any]:
        """Ejecuta el ciclo de consulta seguro y retorna respuesta con fuentes ordenadas por relevancia descendente."""
        retriever = self.resolver_retriever(strategy_id=strategy_id, use_reranker=use_reranker)
        pipeline = PipelineRAGSeguro(
            retriever=retriever,
            llm=self.llm,
            input_guard=self.input_guard,
            output_guard=self.output_guard,
            max_reasks=1,
        )

        res: ResultadoPipeline = pipeline.ejecutar(consulta, top_k=top_k)
        fuentes_ordenadas = sorted(res.fuentes, key=lambda f: f.score, reverse=True)

        fuentes_payload: list[dict[str, Any]] = []
        for idx, f in enumerate(fuentes_ordenadas, start=1):
            fuentes_payload.append(
                {
                    "indice_visual": idx,
                    "doc_code": f.doc_code,
                    "ley": f.ley,
                    "titulo": f.titulo,
                    "texto": f.texto,
                    "url": f.url,
                    "es_principal": idx == 1,
                }
            )

        strat_info = ESTRATEGIAS_METADATA.get(strategy_id, ESTRATEGIAS_METADATA["auto_merging"])

        return {
            "consulta_original": res.consulta_original,
            "consulta_saneada": res.consulta_saneada,
            "bloqueada": res.bloqueada,
            "motivo_bloqueo": res.motivo_bloqueo,
            "reasks_utilizados": res.reasks_utilizados,
            "latencia_ms": round(res.latencia_ms, 1),
            "estrategia": {
                "id": strategy_id,
                "nombre": strat_info["nombre"],
                "badge": strat_info["badge"],
            },
            "reranker_activo": use_reranker,
            "respuesta": {
                "resumen_ejecutivo": res.respuesta.resumen_ejecutivo,
                "articulos_citados": res.respuesta.articulos_citados,
                "requisitos_o_plazos": res.respuesta.requisitos_o_plazos,
                "grado_certeza": res.respuesta.grado_certeza,
                "aviso_legal": res.respuesta.aviso_legal,
            },
            "fuentes": fuentes_payload,
        }

