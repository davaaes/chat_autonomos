from legal_rag_boe.cleaners import strip_historical_noise


def test_strip_historical_noise_removes_previous_redaction_markers() -> None:
    text = "Artículo 1. X\n\nRedacción anterior: texto viejo\n\nTéngase en cuenta que aviso"
    cleaned = strip_historical_noise(text)
    assert "Redacción anterior" not in cleaned
    assert "Téngase en cuenta que" not in cleaned
