from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_settings
from .pipeline import run_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(prog="legal-rag-boe")
    parser.add_argument("--root", default=None, help="Raíz del proyecto")
    args = parser.parse_args()

    settings = load_settings(Path(args.root) if args.root else None)
    result = run_pipeline(settings)
    print(result.model_dump_json(indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
