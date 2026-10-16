from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    project_root: Path
    data_dir: Path
    raw_dir: Path
    clean_dir: Path
    json_dir: Path
    markdown_dir: Path
    boe_base_url: str
    timeout_seconds: float
    user_agent: str
    ollama_base_url: str = "http://localhost:11434"


def load_settings(project_root: Path | None = None) -> Settings:
    load_dotenv()
    root = project_root or Path(__file__).resolve().parents[2]
    data_dir = Path(os.getenv("LEGAL_RAG_DATA_DIR", root / "data"))
    return Settings(
        project_root=root,
        data_dir=data_dir,
        raw_dir=data_dir / "raw",
        clean_dir=data_dir / "clean",
        json_dir=data_dir / "json",
        markdown_dir=data_dir / "markdown",
        boe_base_url=os.getenv(
            "BOE_BASE_URL",
            "https://www.boe.es/datosabiertos/api/legislacion-consolidada/id",
        ),
        timeout_seconds=float(os.getenv("BOE_TIMEOUT_SECONDS", "30")),
        user_agent=os.getenv(
            "BOE_USER_AGENT",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) LegalRAGBOE/0.1",
        ),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )
