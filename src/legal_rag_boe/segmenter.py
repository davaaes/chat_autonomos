from __future__ import annotations

import re
from xml.etree import ElementTree as ET

from .models import ArticleChunk


def _local_name(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _element_text(element: ET.Element) -> str:
    return " ".join(part.strip() for part in "".join(element.itertext()).split())


def _clean_article_title(raw_title: str, article_number: str) -> str:
    prefix = f"Artículo {article_number}."
    if raw_title.startswith(prefix):
        return raw_title.rstrip(".")
    return raw_title or f"Artículo {article_number}"


def parse_consolidated_xml(xml_text: str) -> list[tuple[str, str, str]]:
    root = ET.fromstring(xml_text)
    data = root.find("data")
    if data is None:
        return []

    texto = data.find("texto")
    if texto is None:
        return []

    articles: list[tuple[str, str, str]] = []
    for block in texto.findall("bloque"):
        if block.attrib.get("tipo") != "precepto":
            continue

        versions = block.findall("version")
        if not versions:
            continue

        version = versions[-1]
        heading = None
        body_lines: list[str] = []
        article_number = block.attrib.get("id", "").removeprefix("a")

        for child in list(version):
            if _local_name(child.tag) == "blockquote":
                continue
            if _local_name(child.tag) != "p":
                continue

            text = _element_text(child)
            if not text:
                continue

            class_name = child.attrib.get("class", "")
            if class_name == "articulo":
                heading = _clean_article_title(text, article_number)
                continue

            body_lines.append(text)

        if not heading:
            heading = f"Artículo {article_number}"

        content = "\n\n".join([heading, *body_lines]).strip()
        articles.append((article_number, heading, content))

    return articles


def parse_alcance_filter(alcance: str | None) -> set[str] | None:
    if not alcance or "completa" in alcance.lower():
        return None
    allowed = set()
    cleaned = re.sub(r"art[íi]culos?\s*", "", alcance, flags=re.IGNORECASE).strip()
    segments = [s.strip() for s in re.split(r"\s+y\s+|,\s*", cleaned) if s.strip()]
    for seg in segments:
        m_range = re.fullmatch(r"(\d+)\s+a\s+(\d+)", seg)
        if m_range:
            start, end = int(m_range.group(1)), int(m_range.group(2))
            for n in range(start, end + 1):
                allowed.add(str(n))
        else:
            m_single = re.fullmatch(r"(\d+)", seg)
            if m_single:
                allowed.add(m_single.group(1))
    return allowed


def is_article_in_scope(article_num: str, allowed_numbers: set[str] | None) -> bool:
    if allowed_numbers is None:
        return True
    m = re.match(r"^(\d+)", article_num)
    if not m:
        return False
    return m.group(1) in allowed_numbers


def build_article_chunks(
    *,
    doc_code: str,
    boe_id: str,
    ley: str,
    source_url: str,
    source_xml: str,
    alcance: str | None = None,
) -> list[ArticleChunk]:
    chunks: list[ArticleChunk] = []
    allowed_numbers = parse_alcance_filter(alcance)
    for num, title, content in parse_consolidated_xml(source_xml):
        if not is_article_in_scope(num, allowed_numbers):
            continue
        cleaned_content = content.strip()
        chunks.append(
            ArticleChunk(
                id=f"{doc_code}-art-{num}",
                doc_code=doc_code,
                boe_id=boe_id,
                ley=ley,
                art_num=num,
                titulo=title,
                url_fuente=source_url,
                content=cleaned_content,
                retrieval_text=f"{title}\n\n{cleaned_content}",
                in_scope=True,
            )
        )
    return chunks
