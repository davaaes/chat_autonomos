FROM python:3.12-slim-bookworm

# Instalar dependencias del sistema requeridas para healthcheck y librerías C/C++
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Instalar uv desde la imagen oficial
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Configuración de entorno Python y uv
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    HOST=0.0.0.0 \
    PORT=8000 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2

WORKDIR /app

# Copiar manifiestos y resolver dependencias aprovechando cache de capas de Docker
COPY pyproject.toml README.md uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Copiar código fuente, scripts y dataset / índices
COPY src/ /app/src/
COPY scripts/ /app/scripts/
COPY data/ /app/data/

# Instalar el paquete legal_rag_boe en el entorno virtual
RUN uv sync --frozen --no-dev

# Activar entorno virtual en el PATH
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

# Verificación de salud del servidor web
HEALTHCHECK --interval=20s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Comando por defecto para iniciar la interfaz Starlette
CMD ["python", "scripts/run_ui.py"]

