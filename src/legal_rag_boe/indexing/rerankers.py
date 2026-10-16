from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Sequence

import torch
from llama_index.core.schema import NodeWithScore
from transformers import AutoModel


class BaseDocumentReranker(ABC):
    """Contrato abstracto unificado para modelos de reranking."""

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def rerank_nodes(self, query: str, nodes: Sequence[NodeWithScore], top_k: int = 3) -> list[NodeWithScore]:
        """Reordena los nodos candidatos devolviendo los top_k más relevantes."""
        raise NotImplementedError


class JinaRerankerV35(BaseDocumentReranker):
    """Reranker neuronal listwise jinaai/jina-reranker-v3.5 ejecutado en GPU (CUDA)."""

    _instance: JinaRerankerV35 | None = None

    def __init__(self, device: str | None = None) -> None:
        super().__init__(name="Jina-Reranker-v3.5 (CUDA GPU)")
        target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = target_device
        model_dtype = torch.float16 if target_device == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(
            "jinaai/jina-reranker-v3.5",
            dtype=model_dtype,
            trust_remote_code=True,
        )
        if target_device == "cuda":
            self.model.to("cuda")
        self.model.eval()

    @classmethod
    def get_singleton(cls) -> JinaRerankerV35:
        """Reutiliza la instancia del modelo en memoria para evitar recargas en GPU."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def rerank_nodes(self, query: str, nodes: Sequence[NodeWithScore], top_k: int = 3) -> list[NodeWithScore]:
        if not nodes:
            return []
        texts = [n.node.get_content()[:1500] for n in nodes]
        try:
            results = self.model.rerank(query, texts, top_n=top_k)
            scored: list[NodeWithScore] = []
            for item in results:
                idx = int(item["index"])
                score = float(item["relevance_score"])
                scored.append(NodeWithScore(node=nodes[idx].node, score=score))
            return scored
        except Exception:
            return list(nodes[:top_k])
