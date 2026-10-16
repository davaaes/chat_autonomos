from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

from .models import ArticleChunk, SourceSpec


@dataclass(frozen=True)
class ExtractedUnit:
    title: str
    content: str
    location: str
    page: int | None = None
    content_kind: str = "guide_section"


@dataclass(frozen=True)
class DownloadedSource:
    url: str
    content: bytes
    content_type: str
    checksum: str


_PROFILE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "aeat_renta": ("suministro", "dietas", "vehículo", "vehiculos", "actividades económicas"),
    "aeat_iva": ("deducción", "deduccion", "vehículo", "vehiculos", "prorrata"),
    "aeat_model_instructions": ("036", "037", "303", "130", "111", "115", "modelo"),
    "importass_autonomos": ("tramo", "ingresos", "cuota", "regularización", "tarifa plana"),
    "sepe_contratos": ("indefinido", "formativo", "contrato"),
    "circe_due": ("due", "documento único electrónico", "alta", "autónomo", "tramitación"),
}

_DOC12_PAGE_RANGES = (
    (12, 17),
    (28, 29),
    (32, 34),
    (39, 40),
    (46, 47),
    (48, 48),
    (69, 74),
    (80, 81),
    (84, 85),
    (89, 93),
    (96, 98),
    (107, 113),
    (116, 117),
    (120, 124),
    (128, 131),
    (132, 134),
)

_DOC12_ALLOWED_SECTIONS = {
    "5.1", "5.3", "5.5", "5.7", "5.10", "5.12",
}


def _page_in_ranges(page_number: int) -> bool:
    return any(start <= page_number <= end for start, end in _DOC12_PAGE_RANGES)


def _filter_doc12_page(page_number: int, raw_text: str, current_section: str | None) -> tuple[str, str | None]:
    lines_to_process = raw_text.splitlines()

    if 69 <= page_number <= 98:
        kept: list[str] = []
        for line in lines_to_process:
            match = re.match(r"\s*(5\.\d+)(?:\.\d+)?\s+", line)
            if match:
                current_section = match.group(1)
            if current_section in _DOC12_ALLOWED_SECTIONS:
                kept.append(line)
        return ("\n".join(kept) if _page_in_ranges(page_number) else "", current_section)

    excluded = re.compile(
        r"modelo\s+(?:039|035|840|848)|EORI|OSS|IOSS|sociedades anónimas|sociedades de responsabilidad limitada",
        re.IGNORECASE,
    )
    if not _page_in_ranges(page_number):
        return "", current_section
    return "\n".join(line for line in lines_to_process if not excluded.search(line)), current_section


def download_source(url: str, timeout: float, user_agent: str) -> DownloadedSource:
    response = requests.get(url, timeout=timeout, headers={"User-Agent": user_agent})
    response.raise_for_status()
    content_type = response.headers.get("content-type", "").lower()
    return DownloadedSource(
        url=response.url,
        content=response.content,
        content_type=content_type,
        checksum=hashlib.sha256(response.content).hexdigest(),
    )


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _parse_markdown_sections(text: str) -> list[tuple[str, int, str, list[str]]]:
    lines = text.split("\n")
    sections: list[tuple[str, int, str, list[str]]] = []
    current_heading = ""
    current_level = 0
    current_lines: list[str] = []
    heading_stack: list[str] = []
    heading_re = re.compile(r"^(#{1,6})\s+(.+)$")

    for line in lines:
        m = heading_re.match(line.strip())
        if m:
            if current_heading or current_lines:
                body = "\n".join(current_lines).strip()
                if body:
                    sections.append((current_heading, current_level, body, list(heading_stack)))
            level = len(m.group(1))
            title = m.group(2).strip()
            current_heading = title
            current_level = level
            current_lines = []
            while heading_stack and len(heading_stack) >= level:
                heading_stack.pop()
            heading_stack.append(title)
        else:
            current_lines.append(line)

    if current_heading and current_lines:
        body = "\n".join(current_lines).strip()
        if body:
            sections.append((current_heading, current_level, body, list(heading_stack)))

    return sections


def _merge_small_sections(
    sections: list[tuple[str, int, str, list[str]]],
    min_chars: int = 150,
) -> list[tuple[str, int, str, list[str]]]:
    if not sections:
        return []
    merged: list[tuple[str, int, str, list[str]]] = []
    pending: tuple[str, int, str, list[str]] | None = None
    for heading, level, content, path in sections:
        if pending is not None:
            p_heading, p_level, p_content, p_path = pending
            combined_content = p_content + "\n\n" + content
            if len(combined_content) >= min_chars:
                merged.append((p_heading, p_level, combined_content, p_path))
                pending = None
            else:
                pending = (p_heading, p_level, combined_content, p_path)
        elif len(content) < min_chars:
            pending = (heading, level, content, path)
        else:
            merged.append((heading, level, content, path))
    if pending is not None:
        if merged:
            last_h, last_l, last_c, last_p = merged[-1]
            merged[-1] = (last_h, last_l, last_c + "\n\n" + pending[2], last_p)
        else:
            merged.append(pending)
    return merged


def _markdown_units(source: SourceSpec, file_path: Path) -> list[ExtractedUnit]:
    text = file_path.read_text(encoding="utf-8")
    raw_sections = _parse_markdown_sections(text)
    min_chars = 200 if source.doc_code == "DOC-13" else 150
    merged = _merge_small_sections(raw_sections, min_chars=min_chars)
    kind = "form_instruction" if source.doc_code == "DOC-13" else "guide_section"
    units: list[ExtractedUnit] = []
    for heading, _level, content, path in merged:
        location = " > ".join(path) if path else heading
        units.append(ExtractedUnit(
            title=heading,
            content=content,
            location=location,
            content_kind=kind,
        ))
    return units


def _html_units(source: SourceSpec, downloaded: DownloadedSource) -> list[ExtractedUnit]:
    soup = BeautifulSoup(downloaded.content, "html.parser")
    for element in soup(["script", "style", "noscript", "nav", "footer", "header", "form"]):
        element.decompose()

    units: list[ExtractedUnit] = []
    current_title = source.ley
    current_lines: list[str] = []
    headings = {"h1", "h2", "h3", "h4", "h5"}
    for element in soup.find_all([*headings, "p", "li", "table"]):
        if element.name in headings:
            if current_lines:
                units.append(ExtractedUnit(current_title, "\n".join(current_lines), current_title))
                current_lines = []
            current_title = _clean_text(element.get_text(" ", strip=True)) or current_title
            continue
        if element.name == "table":
            rows = []
            for row in element.find_all("tr"):
                cells = [_clean_text(cell.get_text(" ", strip=True)) for cell in row.find_all(["th", "td"])]
                if cells:
                    rows.append(" | ".join(cells))
            if rows:
                current_lines.append("\n".join(rows))
            continue
        text = _clean_text(element.get_text(" ", strip=True))
        if text:
            current_lines.append(text)

    if current_lines:
        units.append(ExtractedUnit(current_title, "\n".join(current_lines), current_title))

    keywords = _PROFILE_KEYWORDS.get(source.extraction_profile, ())
    filtered = [unit for unit in units if any(keyword in f"{unit.title} {unit.content}".lower() for keyword in keywords)]
    return filtered or units[:30]


def _pdf_units(source: SourceSpec, downloaded: DownloadedSource) -> list[ExtractedUnit]:
    reader = PdfReader(BytesIO(downloaded.content))
    units: list[ExtractedUnit] = []
    keywords = _PROFILE_KEYWORDS.get(source.extraction_profile, ())
    doc12_section: str | None = None
    for page_number, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        if source.extraction_profile == "aeat_actividades_filtrado":
            raw_text, doc12_section = _filter_doc12_page(page_number, raw_text, doc12_section)
        text = _clean_text(raw_text)
        if not text:
            continue
        if keywords and not any(keyword in text.lower() for keyword in keywords):
            continue
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        title = lines[0][:180] if lines else f"Página {page_number}"
        if source.doc_code == "DOC-13":
            kind = "form_instruction"
        elif source.doc_code in {"DOC-11", "DOC-12"}:
            kind = "guide_section"
        else:
            kind = "contract_fact_sheet"
        units.append(ExtractedUnit(title, "\n".join(lines), f"Página {page_number}", page_number, kind))
    if not keywords:
        return units
    return units


def _candidate_links(source: SourceSpec, downloaded: DownloadedSource) -> list[str]:
    soup = BeautifulSoup(downloaded.content, "html.parser")
    links: list[str] = []
    for anchor in soup.find_all("a", href=True):
        text = _clean_text(anchor.get_text(" ", strip=True)).lower()
        href = urljoin(downloaded.url, anchor["href"])
        is_manual_link = source.extraction_profile in {"aeat_renta", "aeat_iva"} and "/Sede/ayuda/" in href
        if ".pdf" in href.lower() or is_manual_link or any(keyword in text for keyword in _PROFILE_KEYWORDS.get(source.extraction_profile, ())):
            if href not in links:
                links.append(href)
    return links[:30]


def extract_source(source: SourceSpec, raw_dir: Path, timeout: float, user_agent: str) -> tuple[list[ArticleChunk], list[str]]:
    urls = source.source_urls or ([source.canonical_url] if source.canonical_url else [])
    warnings: list[str] = []
    units: list[tuple[ExtractedUnit, DownloadedSource]] = []

    for local_path in source.local_paths:
        path = Path(local_path)
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            warnings.append(f"{path}: archivo local no encontrado")
            continue
        content = path.read_bytes()
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            downloaded = DownloadedSource(
                url=source.canonical_url or path.as_uri(),
                content=content,
                content_type="application/pdf",
                checksum=hashlib.sha256(content).hexdigest(),
            )
            (raw_dir / f"{source.doc_code}-local.pdf").write_bytes(content)
            units.extend((unit, downloaded) for unit in _pdf_units(source, downloaded))
        elif suffix in {".md", ".markdown", ".txt"}:
            downloaded = DownloadedSource(
                url=source.canonical_url or path.as_uri(),
                content=content,
                content_type="text/markdown",
                checksum=hashlib.sha256(content).hexdigest(),
            )
            (raw_dir / f"{source.doc_code}-{path.stem}.md").write_bytes(content)
            units.extend((unit, downloaded) for unit in _markdown_units(source, path))

    if source.local_paths and units:
        urls = []

    for url in urls:
        try:
            downloaded = download_source(url, timeout, user_agent)
            suffix = ".pdf" if "pdf" in downloaded.content_type or url.lower().endswith(".pdf") else ".html"
            (raw_dir / f"{source.doc_code}-{len(units) + 1}{suffix}").write_bytes(downloaded.content)
            if suffix == ".pdf":
                parsed = _pdf_units(source, downloaded)
            else:
                parsed = _html_units(source, downloaded)
                if source.source_type == "pdf" or source.extraction_profile in {"aeat_renta", "aeat_iva"}:
                    for linked_url in _candidate_links(source, downloaded):
                        try:
                            linked = download_source(linked_url, timeout, user_agent)
                            if "pdf" in linked.content_type or linked_url.lower().endswith(".pdf"):
                                parsed.extend(_pdf_units(source, linked))
                            elif source.extraction_profile in {"aeat_renta", "aeat_iva"}:
                                parsed.extend(_html_units(source, linked))
                        except requests.RequestException as exc:
                            warnings.append(f"{linked_url}: {exc}")
            units.extend((unit, downloaded) for unit in parsed)
        except requests.RequestException as exc:
            warnings.append(f"{url}: {exc}")

    if not units:
        raise RuntimeError("No se pudo extraer ninguna unidad de las fuentes declaradas")

    chunks: list[ArticleChunk] = []
    retrieved_at = datetime.now(timezone.utc)
    for index, (unit, downloaded) in enumerate(units, start=1):
        identifier = f"{source.doc_code}-{source.extraction_profile}-{index:03d}"
        retrieval_text = " | ".join(filter(None, [source.authority, source.ley, unit.location, unit.title, unit.content[:300].replace("\n", " ")]))
        section_parts = [source.ley, *[p.strip() for p in unit.location.split(">") if p.strip()]]
        chunks.append(
            ArticleChunk(
                id=identifier,
                doc_code=source.doc_code,
                boe_id=source.boe_id,
                ley=source.ley,
                art_num=f"section-{index:03d}",
                titulo=unit.title,
                url_fuente=downloaded.url,
                content=unit.content,
                retrieval_text=retrieval_text,
                source_type=source.source_type,
                content_kind=unit.content_kind,
                authority=source.authority,
                retrieved_at=retrieved_at,
                page=unit.page,
                section_path=section_parts,
                parent_document=source.doc_code,
                validity_note="Verificar vigencia en la fuente oficial antes de aplicar",
                in_scope=True,
            )
        )
    return chunks, warnings
