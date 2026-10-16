from __future__ import annotations

import argparse
import sys
from pathlib import Path


def extract_web_markdown(url: str, output_path: Path) -> None:
    """Descarga una página web y extrae su contenido principal en formato Markdown."""
    try:
        import trafilatura
    except ImportError:
        print("[ERROR] 'trafilatura' no está instalado. Instálalo para usar este extractor.", file=sys.stderr)
        sys.exit(1)

    print(f"Descargando contenido desde: {url}", flush=True)
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        print(f"[ERROR] No se pudo descargar contenido de la URL: {url}", file=sys.stderr)
        sys.exit(1)

    markdown = trafilatura.extract(
        downloaded,
        output_format="markdown",
        include_links=False,
        include_images=False,
    )
    if not markdown:
        print(f"[ERROR] No se pudo extraer texto relevante de la página.", file=sys.stderr)
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    print(f"Markdown guardado con éxito en: {output_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extractor de páginas web a formato Markdown.")
    parser.add_argument(
        "--url",
        default="https://sede.agenciatributaria.gob.es/Sede/todas-gestiones/impuestos-tasas/iva/modelo-303-iva-autoliquidacion_/instrucciones-2026.html",
        help="URL de la página a procesar",
    )
    parser.add_argument(
        "--output",
        default="data/pdf/doc13/modelo303.md",
        help="Ruta de destino del archivo Markdown generado",
    )
    args = parser.parse_args()

    extract_web_markdown(url=args.url, output_path=Path(args.output))