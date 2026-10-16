from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class SourceSpec(BaseModel):
    doc_code: str
    boe_id: str | None = None
    ley: str
    alcance: str
    url_fallback: str | None = None
    source_type: str = "boe_xml"
    authority: str = "BOE"
    official: bool = True
    canonical_url: str | None = None
    extraction_profile: str = "boe_xml"
    source_urls: list[str] = Field(default_factory=list)
    local_paths: list[str] = Field(default_factory=list)


class ArticleChunk(BaseModel):
    id: str
    doc_code: str
    boe_id: str | None = None
    ley: str
    art_num: str
    titulo: str
    url_fuente: str
    content: str
    retrieval_text: str
    source_type: str = "boe_xml"
    content_kind: str = "article"
    authority: str = "BOE"
    published_at: datetime | None = None
    updated_at: datetime | None = None
    retrieved_at: datetime = Field(default_factory=_utc_now)
    page: int | None = None
    section_path: list[str] = Field(default_factory=list)
    parent_document: str | None = None
    validity_note: str | None = None
    in_scope: bool = True


class RawArtifact(BaseModel):
    doc_code: str
    boe_id: str
    source_url: str
    fetched_at: datetime = Field(default_factory=_utc_now)
    raw_text: str
    checksum: str


class DocumentBundle(BaseModel):
    source: SourceSpec
    downloaded_at: datetime = Field(default_factory=_utc_now)
    articles: list[ArticleChunk] = Field(default_factory=list)
    raw_path: Path | None = None
    cleaned_path: Path | None = None
    json_path: Path | None = None
    markdown_path: Path | None = None


class PipelineResult(BaseModel):
    generated_at: datetime = Field(default_factory=_utc_now)
    sources: list[DocumentBundle] = Field(default_factory=list)
    run_date: date = Field(default_factory=date.today)
