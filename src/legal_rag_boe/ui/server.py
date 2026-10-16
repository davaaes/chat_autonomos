from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import os
from pathlib import Path

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response
from starlette.routing import Route

from legal_rag_boe.ui.engine import ESTRATEGIAS_METADATA, RERANKER_METADATA, RAGEngineManager

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
INDEX_HTML_PATH = TEMPLATES_DIR / "index.html"


async def handle_favicon(request: Request) -> Response:
    """Silencia la petición automática de favicon devolviendo No Content."""
    return Response(status_code=204)


async def handle_home(request: Request) -> HTMLResponse:
    """Renderiza la página principal de la interfaz de chat."""
    if not INDEX_HTML_PATH.exists():
        return HTMLResponse("<h1>Error: Plantilla index.html no encontrada</h1>", status_code=500)
    html_content = INDEX_HTML_PATH.read_text(encoding="utf-8")
    return HTMLResponse(html_content)


async def handle_get_strategies(request: Request) -> JSONResponse:
    """Retorna el catálogo de estrategias RAG y opciones de reordenación disponibles."""
    return JSONResponse(
        {
            "estrategias": ESTRATEGIAS_METADATA,
            "rerankers": RERANKER_METADATA,
        }
    )


async def handle_chat(request: Request) -> JSONResponse:
    """Procesa una consulta legal utilizando la estrategia RAG y configuración elegidas."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Cuerpo de solicitud JSON no válido."}, status_code=400)

    consulta = str(body.get("consulta", "")).strip()
    if not consulta:
        return JSONResponse({"error": "La consulta no puede estar vacía."}, status_code=400)

    strategy_id = str(body.get("strategy", "auto_merging"))
    use_reranker = bool(body.get("use_reranker", True))
    top_k = int(body.get("top_k", 3))

    try:
        engine = RAGEngineManager.get_singleton()
        resultado = engine.ejecutar_consulta(
            consulta=consulta,
            strategy_id=strategy_id,
            use_reranker=use_reranker,
            top_k=top_k,
        )
        return JSONResponse(resultado)
    except Exception as exc:
        return JSONResponse(
            {"error": f"Error interno al ejecutar el pipeline RAG: {exc}"},
            status_code=500,
        )


async def handle_health(request: Request) -> JSONResponse:
    """Comprueba el estado operativo del servicio y modelos locales."""
    return JSONResponse({"status": "ok", "app": "legal_rag_boe_ui"})


@asynccontextmanager
async def lifespan(app: Starlette) -> AsyncIterator[None]:
    """Precalienta índices y modelos de inferencia durante el arranque del servidor."""
    if not os.environ.get("PYTEST_CURRENT_TEST"):
        print("⏳ Precalentando motor RAG, índices e inferencia en GPU...", flush=True)
        try:
            engine = RAGEngineManager.get_singleton()
            engine.calentar_motores(prewarm_llm=True, prewarm_reranker=True)
            print("✅ Motor RAG e inferencia en GPU listos para responder en caliente.\n", flush=True)
        except Exception as err:
            print(f"⚠️ Aviso durante precalentamiento: {err}", flush=True)
    yield


routes = [
    Route("/", endpoint=handle_home, methods=["GET"]),
    Route("/favicon.ico", endpoint=handle_favicon, methods=["GET"]),
    Route("/api/strategies", endpoint=handle_get_strategies, methods=["GET"]),
    Route("/api/chat", endpoint=handle_chat, methods=["POST"]),
    Route("/api/health", endpoint=handle_health, methods=["GET"]),
]

middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
]

app = Starlette(debug=False, routes=routes, middleware=middleware, lifespan=lifespan)


def run_app(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Inicia el servidor uvicorn para la interfaz web."""
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    run_app()

