from __future__ import annotations

import os
import sys

# Limitar hilos de CPU para preservar la respuesta fluida del sistema
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

from legal_rag_boe.ui.server import run_app

if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    run_app(host=host, port=port)

