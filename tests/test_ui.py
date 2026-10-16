from __future__ import annotations

from unittest.mock import MagicMock, patch
from starlette.testclient import TestClient

from legal_rag_boe.ui.server import app


def test_ui_home_page() -> None:
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "Asistente Legal para Autónomos" in response.text
    assert "btn-strategy-pill" in response.text
    assert "btn-reranker-pill" in response.text
    assert "top_k=3" not in response.text
    assert "BOE Ground Truth" not in response.text


def test_ui_get_strategies() -> None:
    client = TestClient(app)
    response = client.get("/api/strategies")
    assert response.status_code == 200
    data = response.json()
    assert "estrategias" in data
    assert "rerankers" in data
    assert "auto_merging" in data["estrategias"]
    assert "sentence_window" in data["estrategias"]
    assert "hybrid" in data["estrategias"]
    assert "baseline" not in data["estrategias"]


def test_ui_chat_empty_query() -> None:
    client = TestClient(app)
    response = client.post("/api/chat", json={"consulta": ""})
    assert response.status_code == 400
    assert "error" in response.json()


def test_ui_chat_execution_with_sorted_sources() -> None:
    client = TestClient(app)

    mock_resultado = {
        "consulta_original": "¿Cuánto es la tarifa plana?",
        "consulta_saneada": "¿Cuánto es la tarifa plana?",
        "bloqueada": False,
        "motivo_bloqueo": None,
        "reasks_utilizados": 0,
        "latencia_ms": 120.5,
        "estrategia": {
            "id": "auto_merging",
            "nombre": "Auto-merging",
            "badge": "Recomendado",
        },
        "reranker_activo": True,
        "respuesta": {
            "resumen_ejecutivo": "La tarifa plana es de 80 euros mensuales.",
            "articulos_citados": ["DOC-14"],
            "requisitos_o_plazos": ["80 €/mes", "12 meses"],
            "grado_certeza": "alta",
            "aviso_legal": "AVISO PRECEPTIVO",
        },
        "fuentes": [
            {
                "indice_visual": 1,
                "doc_code": "DOC-14",
                "ley": "RETA",
                "titulo": "Tarifa Plana",
                "texto": "Cuota reducida de 80 euros durante 12 meses.",
                "url": None,
                "es_principal": True,
            },
            {
                "indice_visual": 2,
                "doc_code": "DOC-05",
                "ley": "IRPF",
                "titulo": "Art. 30",
                "texto": "Rendimientos en estimación directa.",
                "url": None,
                "es_principal": False,
            },
        ],
    }

    with patch("legal_rag_boe.ui.server.RAGEngineManager.get_singleton") as mock_get_singleton:
        mock_engine = MagicMock()
        mock_engine.ejecutar_consulta.return_value = mock_resultado
        mock_get_singleton.return_value = mock_engine

        payload = {
            "consulta": "¿Cuánto es la tarifa plana?",
            "strategy": "auto_merging",
            "use_reranker": True,
            "top_k": 3,
        }
        response = client.post("/api/chat", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["bloqueada"] is False
        assert len(data["fuentes"]) == 2
        assert data["fuentes"][0]["es_principal"] is True
        assert "score" not in data["fuentes"][0]
