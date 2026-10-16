from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from llama_index.core.schema import Document

from legal_rag_boe.models import DocumentBundle


def load_documents_from_json(
    json_dir: str | Path = "data/json",
    doc_codes: Sequence[str] | None = None,
    in_scope_only: bool = True,
) -> list[Document]:
    """Carga unidades documentales serializadas en JSON a objetos Document de LlamaIndex."""
    base_path = Path(json_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"Directorio no encontrado: {base_path}")

    target_files = sorted(base_path.glob("DOC-*.json"))
    if doc_codes:
        normalized_codes = {c.upper().strip() for c in doc_codes}
        target_files = [f for f in target_files if f.stem.upper() in normalized_codes]

    documents: list[Document] = []
    for file_path in target_files:
        try:
            content = file_path.read_text(encoding="utf-8")
            bundle = DocumentBundle.model_validate_json(content)
        except (json.JSONDecodeError, ValueError) as err:
            raise ValueError(f"Error procesando {file_path.name}: {err}") from err

        for article in bundle.articles:
            if in_scope_only and not article.in_scope:
                continue

            section_trail = " > ".join(article.section_path) if article.section_path else article.titulo
            header = f"[{article.authority} | {article.ley} | {article.titulo}]"
            full_text = f"{header}\n\n{article.content}"

            doc = Document(
                text=full_text,
                metadata={
                    "doc_code": article.doc_code,
                    "boe_id": article.boe_id or "",
                    "ley": article.ley,
                    "art_num": article.art_num,
                    "titulo": article.titulo,
                    "authority": article.authority,
                    "source_type": article.source_type,
                    "content_kind": article.content_kind,
                    "section_path": section_trail,
                    "url_fuente": article.url_fuente,
                    "in_scope": article.in_scope,
                },
                excluded_embed_metadata_keys=["url_fuente", "boe_id", "in_scope", "section_path"],
                excluded_llm_metadata_keys=["url_fuente", "boe_id", "in_scope", "section_path"],
                doc_id=article.id,
            )
            documents.append(doc)

    return documents
