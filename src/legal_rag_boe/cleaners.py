from __future__ import annotations

import re


HISTORICAL_PATTERNS = [
    re.compile(r"T[eé]ngase en cuenta que.*?(?=\n\n|\Z)", re.IGNORECASE | re.DOTALL),
    re.compile(r"Redacci[oó]n anterior.*?(?=\n\n|\Z)", re.IGNORECASE | re.DOTALL),
    re.compile(r"Texto consolidado.*?(?=\n\n|\Z)", re.IGNORECASE | re.DOTALL),
]


def strip_historical_noise(text: str) -> str:
    cleaned = text
    for pattern in HISTORICAL_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    return cleaned.strip()


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
