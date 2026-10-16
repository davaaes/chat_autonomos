from __future__ import annotations

import hashlib
from dataclasses import dataclass

import requests

from .config import Settings
from .models import RawArtifact, SourceSpec


@dataclass(frozen=True)
class FetchResponse:
    url: str
    text: str
    checksum: str
    content_type: str | None


class BOEClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": settings.user_agent, "Accept": "application/xml"})

    def build_source_url(self, boe_id: str) -> str:
        return f"{self.settings.boe_base_url}/{boe_id}"

    def fetch_consolidated_xml(self, boe_id: str) -> FetchResponse:
        url = self.build_source_url(boe_id)
        response = self.session.get(url, timeout=self.settings.timeout_seconds)
        response.raise_for_status()
        checksum = hashlib.sha256(response.content).hexdigest()
        return FetchResponse(
            url=url,
            text=response.text,
            checksum=checksum,
            content_type=response.headers.get("content-type"),
        )

    def fetch_raw_artifact(self, source: SourceSpec) -> RawArtifact:
        response = self.fetch_consolidated_xml(source.boe_id)
        return RawArtifact(
            doc_code=source.doc_code,
            boe_id=source.boe_id,
            source_url=response.url,
            raw_text=response.text,
            checksum=response.checksum,
        )
