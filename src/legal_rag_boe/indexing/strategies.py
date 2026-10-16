from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Sequence

from pathlib import Path

from llama_index.core import Document, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.llms import LLM
from llama_index.core.node_parser import HierarchicalNodeParser, SentenceWindowNodeParser, get_leaf_nodes
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import AutoMergingRetriever, BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.retrievers.bm25 import BM25Retriever

from legal_rag_boe.indexing.rerankers import BaseDocumentReranker, JinaRerankerV35



def get_default_embed_model(
    model_name: str = "bge-m3",
    batch_size: int = 32,
    base_url: str | None = None,
) -> OllamaEmbedding:
    """Instancia el modelo de embeddings local bge-m3 optimizado con procesamiento por lotes."""
    target_base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    return OllamaEmbedding(model_name=model_name, base_url=target_base_url, embed_batch_size=batch_size)


class BaseRAGStrategy(ABC):
    """Contrato base para estrategias de indexación y recuperación documental."""

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 3) -> list[NodeWithScore]:
        """Recupera los nodos más relevantes para la consulta dada."""
        raise NotImplementedError

    def as_retriever(self, top_k: int = 3) -> BaseRetriever:
        """Expone la estrategia como un recuperador estándar de LlamaIndex."""
        return _StrategyRetrieverWrapper(self, top_k=top_k)

    def as_query_engine(self, llm: LLM | None = None, top_k: int = 3) -> BaseQueryEngine:
        """Construye un motor de consulta estándar de LlamaIndex."""
        retriever = self.as_retriever(top_k=top_k)
        return RetrieverQueryEngine.from_args(retriever=retriever, llm=llm)


class _StrategyRetrieverWrapper(BaseRetriever):
    """Adaptador que expone una estrategia RAG como BaseRetriever estándar de LlamaIndex."""

    def __init__(self, strategy: BaseRAGStrategy, top_k: int = 3) -> None:
        super().__init__()
        self._strategy = strategy
        self._top_k = top_k

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        return self._strategy.retrieve(query_bundle.query_str, top_k=self._top_k)


class ArticleLevelRAG(BaseRAGStrategy):
    """Estrategia baseline que indexa artículos completos como unidades indivisibles."""

    def __init__(
        self,
        documents: Sequence[Document],
        embed_model: OllamaEmbedding | None = None,
        name: str = "Article-Level (Baseline)",
        persist_dir: str | Path | None = None,
    ) -> None:
        super().__init__(name=name)
        self.embed_model = embed_model or get_default_embed_model()
        if persist_dir and Path(persist_dir).exists() and any(Path(persist_dir).iterdir()):
            storage_context = StorageContext.from_defaults(persist_dir=str(persist_dir))
            self.index = load_index_from_storage(storage_context, embed_model=self.embed_model)
        else:
            self.index = VectorStoreIndex.from_documents(
                documents,
                embed_model=self.embed_model,
                show_progress=False,
            )
            if persist_dir:
                Path(persist_dir).mkdir(parents=True, exist_ok=True)
                self.index.storage_context.persist(persist_dir=str(persist_dir))

    def retrieve(self, query: str, top_k: int = 3) -> list[NodeWithScore]:
        """Recupera nodos completos a nivel de artículo por similitud vectorial."""
        retriever = self.index.as_retriever(similarity_top_k=top_k)
        return retriever.retrieve(query)


class SentenceWindowRAG(BaseRAGStrategy):
    """Estrategia que desacopla el embedding de oraciones y expande a ventana contextual."""

    def __init__(
        self,
        documents: Sequence[Document],
        embed_model: OllamaEmbedding | None = None,
        window_size: int = 3,
        name: str | None = None,
        persist_dir: str | Path | None = None,
    ) -> None:
        super().__init__(name=name or f"Sentence-Window (w={window_size})")
        self.embed_model = embed_model or get_default_embed_model()
        self.window_size = window_size
        self.postprocessor = MetadataReplacementPostProcessor(target_metadata_key="window")
        if persist_dir and Path(persist_dir).exists() and any(Path(persist_dir).iterdir()):
            storage_context = StorageContext.from_defaults(persist_dir=str(persist_dir))
            self.index = load_index_from_storage(storage_context, embed_model=self.embed_model)
        else:
            self.parser = SentenceWindowNodeParser.from_defaults(
                window_size=window_size,
                window_metadata_key="window",
                original_text_metadata_key="original_text",
            )
            nodes = self.parser.get_nodes_from_documents(list(documents))
            self.index = VectorStoreIndex(nodes, embed_model=self.embed_model, show_progress=False)
            if persist_dir:
                Path(persist_dir).mkdir(parents=True, exist_ok=True)
                self.index.storage_context.persist(persist_dir=str(persist_dir))

    def retrieve(self, query: str, top_k: int = 3) -> list[NodeWithScore]:
        """Recupera oraciones relevantes y reemplaza su texto por la ventana adyacente."""
        base_retriever = self.index.as_retriever(similarity_top_k=top_k)
        raw_nodes = base_retriever.retrieve(query)
        return self.postprocessor.postprocess_nodes(raw_nodes, query_bundle=QueryBundle(query))


class AutoMergingRAG(BaseRAGStrategy):
    """Estrategia jerárquica padre-hijo con consolidación dinámica por umbral."""

    def __init__(
        self,
        documents: Sequence[Document],
        embed_model: OllamaEmbedding | None = None,
        chunk_sizes: list[int] | None = None,
        ratio_threshold: float = 0.5,
        leaf_top_k: int | None = None,
        name: str | None = None,
        base_strategy: AutoMergingRAG | None = None,
        persist_dir: str | Path | None = None,
    ) -> None:
        super().__init__(name=name or f"Auto-Merging (ratio={ratio_threshold})")
        self.embed_model = embed_model or get_default_embed_model()
        self.ratio_threshold = ratio_threshold
        self.leaf_top_k = leaf_top_k
        if base_strategy:
            self.storage_context = base_strategy.storage_context
            self.index = base_strategy.index
        elif persist_dir and Path(persist_dir).exists() and any(Path(persist_dir).iterdir()):
            self.storage_context = StorageContext.from_defaults(persist_dir=str(persist_dir))
            self.index = load_index_from_storage(self.storage_context, embed_model=self.embed_model)
        else:
            sizes = chunk_sizes or [1024, 256]
            self.parser = HierarchicalNodeParser.from_defaults(chunk_sizes=sizes)
            all_nodes = self.parser.get_nodes_from_documents(list(documents))
            leaf_nodes = get_leaf_nodes(all_nodes)

            self.docstore = SimpleDocumentStore()
            self.docstore.add_documents(all_nodes)
            self.storage_context = StorageContext.from_defaults(docstore=self.docstore)
            self.index = VectorStoreIndex(
                leaf_nodes,
                storage_context=self.storage_context,
                embed_model=self.embed_model,
                show_progress=False,
            )
            if persist_dir:
                Path(persist_dir).mkdir(parents=True, exist_ok=True)
                self.storage_context.persist(persist_dir=str(persist_dir))

    def retrieve(self, query: str, top_k: int = 3) -> list[NodeWithScore]:
        """Recupera sobre hojas y eleva al nodo padre si supera el umbral de activación."""
        leaf_k = self.leaf_top_k or max(12, top_k * 3)
        base_retriever = self.index.as_retriever(similarity_top_k=leaf_k)
        merging_retriever = AutoMergingRetriever(
            base_retriever,
            storage_context=self.storage_context,
            simple_ratio_thresh=self.ratio_threshold,
        )
        results = merging_retriever.retrieve(query)
        return results[:top_k]


class HybridBM25RAG(BaseRAGStrategy):
    """Estrategia híbrida que combina similitud densa vectorial y léxica BM25."""

    def __init__(
        self,
        documents: Sequence[Document],
        embed_model: OllamaEmbedding | None = None,
        name: str = "Hybrid (Vector + BM25)",
        vector_index: VectorStoreIndex | None = None,
        persist_dir: str | Path | None = None,
    ) -> None:
        super().__init__(name=name)
        self.embed_model = embed_model or get_default_embed_model()
        if vector_index:
            self.vector_index = vector_index
        elif persist_dir and Path(persist_dir).exists() and any(Path(persist_dir).iterdir()):
            storage_context = StorageContext.from_defaults(persist_dir=str(persist_dir))
            self.vector_index = load_index_from_storage(storage_context, embed_model=self.embed_model)
        else:
            self.vector_index = VectorStoreIndex.from_documents(
                documents,
                embed_model=self.embed_model,
                show_progress=False,
            )
            if persist_dir:
                Path(persist_dir).mkdir(parents=True, exist_ok=True)
                self.vector_index.storage_context.persist(persist_dir=str(persist_dir))

        doc_list = list(documents)
        self.bm25_retriever = BM25Retriever.from_defaults(
            nodes=doc_list,
            similarity_top_k=10,
            language="spanish",
        )

    def retrieve(self, query: str, top_k: int = 3) -> list[NodeWithScore]:
        """Fusiona candidatos densos y léxicos mediante Reciprocal Rank Fusion."""
        retrieve_k = max(10, top_k)
        dense_nodes = self.vector_index.as_retriever(similarity_top_k=retrieve_k).retrieve(query)
        if not dense_nodes or (dense_nodes[0].score is not None and dense_nodes[0].score < 0.44):
            return []

        self.bm25_retriever.similarity_top_k = retrieve_k
        bm25_nodes = self.bm25_retriever.retrieve(query)

        scores: dict[str, float] = {}
        node_map: dict[str, NodeWithScore] = {}
        rrf_k = 60

        for rank, item in enumerate(dense_nodes):
            nid = item.node.node_id
            scores[nid] = scores.get(nid, 0.0) + (1.0 / (rrf_k + rank + 1))
            node_map[nid] = item

        for rank, item in enumerate(bm25_nodes):
            nid = item.node.node_id
            scores[nid] = scores.get(nid, 0.0) + (1.0 / (rrf_k + rank + 1))
            if nid not in node_map:
                node_map[nid] = item

        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:top_k]
        fused_nodes: list[NodeWithScore] = []
        for nid in sorted_ids:
            original = node_map[nid]
            node_score = original.score if (original.score is not None and original.score > 0.05) else scores[nid]
            fused_nodes.append(NodeWithScore(node=original.node, score=node_score))
        return fused_nodes


def get_default_reranker() -> JinaRerankerV35:
    """Instancia y reutiliza el modelo homologado jinaai/jina-reranker-v3.5 en GPU."""
    return JinaRerankerV35.get_singleton()


class RerankedRAG(BaseRAGStrategy):
    """Envuelve cualquier estrategia RAG base y aplica reordenación con Jina-v3.5 sobre candidate_k candidatos."""

    def __init__(
        self,
        base_strategy: BaseRAGStrategy,
        reranker: BaseDocumentReranker | None = None,
        candidate_k: int = 12,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name or f"{base_strategy.name} + Reranker (Jina-v3.5)")
        self.base_strategy = base_strategy
        self.candidate_k = candidate_k
        self.reranker = reranker or get_default_reranker()

    def retrieve(self, query: str, top_k: int = 3) -> list[NodeWithScore]:
        """Recupera candidate_k nodos con la estrategia base y los reordena con el modelo listwise Jina."""
        retrieve_k = max(self.candidate_k, top_k)
        candidates = self.base_strategy.retrieve(query, top_k=retrieve_k)
        if not candidates:
            return []
        return self.reranker.rerank_nodes(query, candidates, top_k=top_k)


class HybridRerankRAG(RerankedRAG):
    """Estrategia híbrida (Vector + BM25) combinada con Jina-v3.5 Reranker acelerado en GPU."""

    def __init__(
        self,
        documents: Sequence[Document],
        embed_model: OllamaEmbedding | None = None,
        hybrid_strategy: HybridBM25RAG | None = None,
        reranker: BaseDocumentReranker | None = None,
        candidate_k: int = 12,
        name: str = "Hybrid + Reranker (Jina-v3.5)",
    ) -> None:
        base = hybrid_strategy or HybridBM25RAG(documents, embed_model=embed_model)
        super().__init__(
            base_strategy=base,
            reranker=reranker,
            candidate_k=candidate_k,
            name=name,
        )
        self.hybrid = base


def get_canonical_strategies(
    documents: Sequence[Document],
    embed_model: OllamaEmbedding | None = None,
    indices_dir: str | Path | None = None,
) -> dict[str, BaseRAGStrategy]:
    """Instancia el cuarteto canónico de estrategias RAG homologadas para el proyecto."""
    model = embed_model or get_default_embed_model()
    target_dir = Path(indices_dir) if indices_dir else None

    baseline = ArticleLevelRAG(
        documents,
        embed_model=model,
        name="Article-Level (Baseline)",
        persist_dir=target_dir / "article_level" if target_dir else None,
    )
    sentence_window = SentenceWindowRAG(
        documents,
        embed_model=model,
        window_size=3,
        name="Sentence-Window (w=3)",
        persist_dir=target_dir / "sentence_window_3" if target_dir else None,
    )
    auto_merging = AutoMergingRAG(
        documents,
        embed_model=model,
        chunk_sizes=[1024, 256],
        ratio_threshold=0.5,
        name="Auto-Merging [1024, 256] (r=0.5)",
        persist_dir=target_dir / "auto_merging_1024_256" if target_dir else None,
    )
    hybrid = HybridBM25RAG(
        documents,
        embed_model=model,
        name="Hybrid (Vector + BM25)",
        vector_index=baseline.index,
    )

    return {
        "baseline": baseline,
        "sentence_window": sentence_window,
        "auto_merging": auto_merging,
        "hybrid": hybrid,
    }


def get_automerging_configurations(
    documents: Sequence[Document],
    embed_model: OllamaEmbedding | None = None,
    indices_dir: str | Path | None = None,
    reranker: BaseDocumentReranker | None = None,
) -> dict[str, BaseRAGStrategy]:
    """Construye las configuraciones comparativas de AutoMerging con y sin reranker."""
    model = embed_model or get_default_embed_model()
    target_dir = Path(indices_dir) if indices_dir else None

    base_1024_512 = AutoMergingRAG(
        documents,
        embed_model=model,
        chunk_sizes=[1024, 512],
        ratio_threshold=0.6,
        leaf_top_k=12,
        name="AM-Canónico [1024, 512] (r=0.6)",
        persist_dir=target_dir / "auto_merging" if target_dir else None,
    )
    base_1024_256 = AutoMergingRAG(
        documents,
        embed_model=model,
        chunk_sizes=[1024, 256],
        ratio_threshold=0.5,
        leaf_top_k=12,
        name="AM-Granularidad-Fina [1024, 256] (r=0.5)",
        persist_dir=target_dir / "auto_merging_1024_256" if target_dir else None,
    )
    base_3niveles = AutoMergingRAG(
        documents,
        embed_model=model,
        chunk_sizes=[1024, 512, 256],
        ratio_threshold=0.5,
        leaf_top_k=15,
        name="AM-Jerárquico-3Niveles [1024, 512, 256] (r=0.5)",
        persist_dir=target_dir / "auto_merging_1024_512_256" if target_dir else None,
    )

    active_reranker = reranker or get_default_reranker()
    am_canonico_reranked = RerankedRAG(
        base_strategy=base_1024_512,
        reranker=active_reranker,
        candidate_k=6,
        name="AM-Canónico [1024, 512] (r=0.6) + Jina-v3.5",
    )
    am_fina_reranked = RerankedRAG(
        base_strategy=base_1024_256,
        reranker=active_reranker,
        candidate_k=6,
        name="AM-Granularidad-Fina [1024, 256] (r=0.5) + Jina-v3.5",
    )
    am_3niveles_reranked = RerankedRAG(
        base_strategy=base_3niveles,
        reranker=active_reranker,
        candidate_k=6,
        name="AM-Jerárquico-3Niveles [1024, 512, 256] (r=0.5) + Jina-v3.5",
    )

    return {
        "canonico": base_1024_512,
        "canonico_reranker": am_canonico_reranked,
        "granularidad_fina": base_1024_256,
        "granularidad_fina_reranker": am_fina_reranked,
        "jerarquico_3niveles": base_3niveles,
        "jerarquico_3niveles_reranker": am_3niveles_reranked,
    }
