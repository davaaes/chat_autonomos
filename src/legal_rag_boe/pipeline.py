from __future__ import annotations

from .boe_client import BOEClient
from .config import Settings
from .models import DocumentBundle, PipelineResult, SourceSpec
from .official_ingest import extract_source
from .official_sources import PROCEDURAL_SOURCES
from .segmenter import build_article_chunks


DEFAULT_SOURCES = [
    SourceSpec(doc_code="DOC-01", boe_id="BOE-A-2007-13409", ley="Ley 20/2007", alcance="Norma completa"),
    SourceSpec(doc_code="DOC-02", boe_id="BOE-A-2015-11724", ley="RDLeg 8/2015", alcance="Artículos 305 a 333"),
    SourceSpec(doc_code="DOC-03", boe_id="BOE-A-2022-12482", ley="Real Decreto-ley 13/2022", alcance="Norma completa"),
    SourceSpec(doc_code="DOC-04", boe_id="BOE-A-2012-14696", ley="Real Decreto 1619/2012", alcance="Norma completa"),
    SourceSpec(doc_code="DOC-05", boe_id="BOE-A-2006-20764", ley="Ley 35/2006", alcance="Artículos 27 a 32 y 95 a 101"),
    SourceSpec(doc_code="DOC-06", boe_id="BOE-A-2007-6820", ley="Real Decreto 439/2007", alcance="Artículos 95 y 109 a 112"),
    SourceSpec(doc_code="DOC-07", boe_id="BOE-A-1992-28740", ley="Ley 37/1992", alcance="Artículos 4 a 28 y 90 a 114"),
    SourceSpec(doc_code="DOC-08", boe_id="BOE-A-2007-15984", ley="Real Decreto 1065/2007", alcance="Artículos 3 a 17"),
    SourceSpec(doc_code="DOC-09", boe_id="BOE-A-2015-11430", ley="RDLeg 2/2015", alcance="Artículos 1 a 60"),
    SourceSpec(doc_code="DOC-10", boe_id="BOE-A-1996-4447", ley="Real Decreto 84/1996", alcance="Artículos 1 a 40"),
    *PROCEDURAL_SOURCES,
]


def ensure_dirs(settings: Settings) -> None:
    for path in [settings.data_dir, settings.raw_dir, settings.clean_dir, settings.json_dir, settings.markdown_dir]:
        path.mkdir(parents=True, exist_ok=True)


def render_markdown(bundle: DocumentBundle) -> str:
    lines = [f"# {bundle.source.doc_code} - {bundle.source.ley}", ""]
    lines.append(f"- BOE ID: {bundle.source.boe_id}")
    lines.append(f"- Alcance: {bundle.source.alcance}")
    lines.append("")
    for article in bundle.articles:
        lines.append(f"## {article.titulo}")
        lines.append("")
        lines.append(article.content)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def render_clean_text(bundle: DocumentBundle) -> str:
    parts: list[str] = []
    for article in bundle.articles:
        parts.append(article.content)
    return "\n\n".join(parts).strip() + "\n"


def run_pipeline(settings: Settings, sources: list[SourceSpec] | None = None) -> PipelineResult:
    ensure_dirs(settings)
    client = BOEClient(settings)
    bundles: list[DocumentBundle] = []
    for source in sources or DEFAULT_SOURCES:
        try:
            if source.source_type != "boe_xml":
                articles, warnings = extract_source(
                    source,
                    settings.raw_dir,
                    settings.timeout_seconds,
                    settings.user_agent,
                )
                for warning in warnings:
                    print(f"[WARN] {source.doc_code}: {warning}")
                clean_path = settings.clean_dir / f"{source.doc_code}.txt"
                clean_path.write_text(render_clean_text(DocumentBundle(source=source, articles=articles)), encoding="utf-8")
                bundle = DocumentBundle(source=source, articles=articles, cleaned_path=clean_path)
                json_path = settings.json_dir / f"{source.doc_code}.json"
                md_path = settings.markdown_dir / f"{source.doc_code}.md"
                json_path.write_text(bundle.model_dump_json(indent=2, ensure_ascii=False), encoding="utf-8")
                md_path.write_text(render_markdown(bundle), encoding="utf-8")
                bundle.json_path = json_path
                bundle.markdown_path = md_path
                bundles.append(bundle)
                continue

            raw = client.fetch_raw_artifact(source)
            raw_path = settings.raw_dir / f"{source.doc_code}.xml"
            raw_path.write_text(raw.raw_text, encoding="utf-8")

            articles = build_article_chunks(
                doc_code=source.doc_code,
                boe_id=source.boe_id,
                ley=source.ley,
                source_url=raw.source_url,
                source_xml=raw.raw_text,
                alcance=source.alcance,
            )

            clean_path = settings.clean_dir / f"{source.doc_code}.txt"
            clean_path.write_text(render_clean_text(DocumentBundle(source=source, articles=articles)), encoding="utf-8")

            bundle = DocumentBundle(source=source, articles=articles, raw_path=raw_path, cleaned_path=clean_path)
            json_path = settings.json_dir / f"{source.doc_code}.json"
            md_path = settings.markdown_dir / f"{source.doc_code}.md"
            json_path.write_text(bundle.model_dump_json(indent=2, ensure_ascii=False), encoding="utf-8")
            md_path.write_text(render_markdown(bundle), encoding="utf-8")
            bundle.json_path = json_path
            bundle.markdown_path = md_path
            bundles.append(bundle)
        except Exception as exc:
            cached_json_path = settings.json_dir / f"{source.doc_code}.json"
            if source.source_type != "boe_xml" and cached_json_path.exists():
                try:
                    cached_bundle = DocumentBundle.model_validate_json(cached_json_path.read_text(encoding="utf-8"))
                    cached_bundle.json_path = cached_json_path
                    cached_bundle.cleaned_path = settings.clean_dir / f"{source.doc_code}.txt"
                    cached_bundle.markdown_path = settings.markdown_dir / f"{source.doc_code}.md"
                    bundles.append(cached_bundle)
                    print(f"[AVISO] {source.doc_code} fuente no disponible; reutilizando JSON previo: {exc}")
                    continue
                except Exception as cache_exc:
                    print(f"[AVISO] {source.doc_code} caché inválida: {cache_exc}")
            print(f"[AVISO] {source.doc_code} falló: {source.boe_id} -> {exc}")

    return PipelineResult(sources=bundles)
