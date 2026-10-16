# Chat Autónomos España: Asistente Legal RAG con Guardrails y Observabilidad

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-0.12-FF6F00?logo=curseforge&logoColor=white)](https://www.llamaindex.ai/)
[![TruLens](https://img.shields.io/badge/TruLens-Observability-purple)](https://www.trulens.org/)
[![Ollama](https://img.shields.io/badge/Ollama-Local_Inference-black?logo=ollama&logoColor=white)](https://ollama.ai/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA_12-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/Docker-Compose_Ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![Tests](https://img.shields.io/badge/Tests-37%2F37_Passing-brightgreen?logo=pytest&logoColor=white)](https://pytest.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Sistema RAG (*Retrieval-Augmented Generation*) especializado en el régimen legal, fiscal y de seguridad social del trabajador autónomo en España (persona física, territorio común). 

Diseñado bajo una arquitectura **100% local (On-Premise)** para garantizar soberanía de datos y secreto profesional tributario conforme al RGPD, integrando una política estricta de cero alucinaciones (*Zero-Hallucination Policy*), validación defensiva en entrada y salida (*Guardrails*), y auditoría cuantitativa mediante la Tríada de TruLens.

---

## Ficha Técnica del Proyecto

| Dimensión | Especificación Técnica |
| :--- | :--- |
| **Dominio Jurídico** | RETA (15 tramos 2026, cuota reducida), IRPF (art. 27-32, gastos deducibles), IVA (303), censos (036/037, DUE/CIRCE) y primera contratación laboral. |
| **Corpus Ground Truth** | 513 unidades jurídicas consolidadas de la API del BOE (XML) y manuales oficiales de la AEAT y la Seguridad Social (`in_scope=True`). |
| **Estrategia RAG Óptima** | *Auto-Merging* jerárquico [1024, 256] (ratio 0.5) + Reranker listwise *Jina-v3.5* en GPU (CUDA). |
| **Tríada Asimétrica LLM** | Redactor: `qwen2.5:3b` (~2.0s) \| Auditor Guardrails: `qwen2.5:1.5b` (~450ms) \| Juez TruLens: `qwen2.5:7b` (0% coste API). |
| **Seguridad y Guardrails** | Detección de inyecciones de prompt (<0.1 ms), deflexión de consultas fuera de ámbito, enmascaramiento PII y ciclo *Re-Ask* con Pydantic v2. |
| **Métricas Clave (TruLens)** | 100.0% de anclaje normativo (*Groundedness*), 83.3% de relevancia de contexto y compresión de contexto a 471 tokens. |
| **Reproducibilidad** | Despliegue autónomo en un solo comando mediante `docker compose up` o entorno nativo ultra-rápido con `uv`. |

---

## Demostración Interactiva

![Demostración interactiva del flujo completo](docs/assets/demo_chat_flow.gif)

---

## Contexto y Valor para Producción

El asesoramiento tributario automatizado mediante modelos de lenguaje comerciales presenta dos barreras críticas:
1. **Riesgo de sanción tributaria por alucinación:** Los modelos fundacionales genéricos tienden a inventar casillas, plazos o porcentajes de deducción inexistentes en el BOE.
2. **Infracción de privacidad (RGPD):** Transmitir datos contables, censales o de facturación de contribuyentes a endpoints en la nube de terceros vulnera el deber de secreto y la confidencialidad profesional.

Este proyecto resuelve ambas limitaciones implementando un pipeline local desacoplado:
- **Indexación estructurada de fuentes primarias:** El sistema no consume resúmenes de blogs ni foros informales; opera sobre texto consolidado del BOE y resoluciones administrativas oficiales.
- **Cortocircuito semántico temprano:** Las preguntas ajenas al ámbito de autónomos (deportes, programación, recetas) o los intentos de inyección se neutralizan antes de consumir tiempo de inferencia del generador.
- **Verificación cruzada de anclaje normativo:** Un auditor ligero desacoplado examina la síntesis contra las fuentes recuperadas mediante razonamiento *Chain-of-Thought (CoT)* previo a la entrega al usuario.

---

## Interfaz de Usuario

La interfaz está construida con Starlette y Tailwind CSS, ofreciendo un diseño profesional, modo oscuro nativo, visualización de grado de certeza y desglose de fuentes oficiales con enlaces canónicos al BOE.

| Vista Principal (Modo Oscuro) | Selector de Estrategias RAG |
| :---: | :---: |
| ![Vista Principal](docs/assets/01_ui_hero.png) | ![Selector de Estrategias](docs/assets/02_strategies_selector.png) |

| Respuesta Jurídica Fundamentada con Despliegue de Fuentes del BOE |
| :---: |
| ![Respuesta Legal](docs/assets/03_legal_response.png) |

---

## Capa Defensiva de Guardrails

El flujo incorpora filtros independientes en dos fases críticas del ciclo de vida de la consulta:

```
[ Consulta de Entrada ]
         │
         ├───► PromptInjectionValidator  ───► Bloqueo preventivo de ciberseguridad
         ├───► ScopeValidator (Regex)    ───► Deflexión temprana sin consumo de LLM
         └───► PIIValidator              ───► Anonimización estricta (DNI, email, teléfono)
         │
[ Motor de Recuperación + LLM ]
         │
[ Salida Generada ]
         │
         ├───► DisclaimerValidator       ───► Inserción preceptiva del aviso legal
         ├───► CitasNormativasValidator  ───► Purga de referencias no presentes en el contexto
         └───► HallucinationValidator    ───► Auditoría CoT con modelo desacoplado (Re-Ask)
         │
[ Respuesta Validada Pydantic v2 ]
```

### Ejemplos de Validación en Tiempo Real

| Deflexión por Control de Ámbito | Bloqueo de Inyección de Prompt / Jailbreak |
| :---: | :---: |
| ![Fuera de Ámbito](docs/assets/04_guardrail_out_of_scope.png) | ![Inyección Bloqueada](docs/assets/05_guardrail_prompt_injection.png) |
| *Respuesta institucional inmediata en <0.1 ms sin invocar al LLM generador ante temas no pertinentes.* | *Bloqueo determinista ante intentos de fuga de directivas del sistema o alteración de comportamiento.* |

| Protección de Privacidad y Enmascaramiento PII |
| :---: |
| ![Sanitización PII](docs/assets/06_guardrail_pii.png) |
| *Sustitución automática de DNI/NIE, teléfonos y direcciones de correo electrónico por tokens genéricos.* |

---

## Arquitectura del Sistema: Tríada Asimétrica de Modelos

Para maximizar el rendimiento en hardware local (GPU NVIDIA RTX 3060 de 12 GB VRAM) y evitar la auto-evaluación complaciente, el sistema distribuye tareas específicas en tres modelos especializados:

```mermaid
flowchart TD
    subgraph Entrada["1. Filtros de Entrada (Input Guards)"]
        Q["Consulta del Usuario"] --> IG["InputGuard (&lt;0.1 ms)"]
        IG -->|"Inyección detectada"| Block["Bloqueo de Ciberseguridad"]
        IG -->|"Fuera de ámbito"| Deflect["Deflexión Temprana (Sin LLM)"]
        IG -->|"Consulta Válida / PII Filtrada"| Ret["Motor de Recuperación"]
    end

    subgraph Recuperacion["2. Recuperación Jerárquica y Reranking"]
        Ret --> BM["Búsqueda Léxica (Rank-BM25)"]
        Ret --> DEN["Búsqueda Densa (BGE-M3 en GPU)"]
        BM & DEN --> AM["Auto-Merging Jerárquico\n[1024, 256] (ratio 0.5)"]
        AM --> JR["Jina-Reranker-v3.5\n(Cross-Encoder Listwise en CUDA)"]
    end

    subgraph Sintesis["3. Generación y Validación de Salida"]
        JR --> GEN["Qwen 2.5 3B (Generador UI)\nInferencia GPU (~1.8s - 2.5s)"]
        GEN --> OG["OutputGuard\n(Contrato Pydantic v2)"]
        OG --> AUD["Qwen 2.5 1.5B (Auditor CoT)\nAuditoría Semántica (~450ms)"]
        AUD -->|"Fallo de Anclaje"| REASK["Ciclo Re-Ask con Crítica"]
        REASK --> GEN
        AUD -->|"Aprobado"| OUT["Interfaz Web Starlette\n(Resumen + Requisitos + Fuentes BOE)"]
    end

    subgraph Evaluacion["4. Observabilidad y Monitoreo (Offline / CI)"]
        OUT -.-> JUDGE["Qwen 2.5 7B (Juez TruLens)\nTríada RAG Reference-Free"]
    end
```

1. **Generador Principal (`qwen2.5:3b`):** Inferencia en GPU a ~1.8s - 2.5s. Balance óptimo entre tiempo de respuesta y adherencia estricta a textos normativos en español (`num_ctx=4096`, `temperature=0.1`, detección de vacíos de contexto en 0.53s).
2. **Auditor Semántico de Guardrails (`qwen2.5:1.5b`):** Modelo ligero desacoplado ejecutado en ~450 ms (`temperature=0.0`). Evalúa el anclaje de la respuesta frente al contexto recuperado sin sesgos circulares con el generador.
3. **Juez de Observabilidad (`qwen2.5:7b`):** Modelo de mayor capacidad paramétrica (7.6B vs 3.09B) utilizado en los bancos de pruebas de TruLens para auditar Relevancia de Contexto y *Groundedness* con explicaciones CoT sin coste de APIs de terceros.

---

## Evaluación Empírica con TruLens (Benchmark)

Se evaluaron las 4 arquitecturas canónicas homologadas sobre el banco de pruebas oficial de *Golden Cases* (preguntas representativas con preceptos exactos y hechos jurídicos contrastados):

| Estrategia RAG | Hit Rate Precepto | Cobertura de Hechos | Relevancia Contexto | Anclaje (*Groundedness*) | Relevancia Respuesta | Tokens Contexto | Latencia Recup. |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Auto-Merging [1024, 256] + Jina-v3.5 (Predeterminado)** | **100.0%** | **100.0%** | **83.3%** | **100.0%** | **100.0%** | **471** | **368 ms** |
| Auto-Merging [1024, 256] Base (Sin Reranker) | 100.0% | 100.0% | 77.8% | 83.3% | 83.3% | 483 | 58 ms |
| Sentence-Window (w=3) + Jina-v3.5 | 100.0% | 100.0% | 77.8% | 86.1% | 100.0% | 568 | 340 ms |
| Sentence-Window (w=3) Base | 100.0% | 100.0% | 77.8% | 83.3% | 100.0% | 487 | 62 ms |
| Hybrid (Vector BGE-M3 + BM25) | 100.0% | 100.0% | 88.9% | 94.4% | 100.0% | 892 | 75 ms |
| Article-Level Baseline | 100.0% | 100.0% | 72.2% | 85.2% | 88.9% | 1.150 | 52 ms |

### Conclusiones de Ingeniería
- **Liderazgo de Auto-Merging de Granularidad Fina:** La combinación [1024, 256] con umbral de activación `0.5` más el Reranker *Jina-v3.5* alcanza el **100.0% de anclaje normativo** (*Groundedness*) con el mínimo consumo de contexto del benchmark (**471 tokens promedio**, una reducción del 30.5% frente al baseline por artículos).
- **Impacto del Reranking Listwise:** Discrimina preceptos numéricos de alta precisión (por ejemplo, tramo 1 de cotización RETA con base de 735,29 € frente a disposiciones transitorias generales), mitigando fusiones prematuras en el árbol jerárquico.

---

## Dataset Ground Truth y Capa de Datos (`data/`)

El corpus se compone de **513 unidades jurídicas atómicas pertinentes** (`in_scope=True`) estructuradas en dos capas complementarias:

### Capa 1: Normativa Consolidada del BOE (XML oficial)
- **DOC-01:** Ley 20/2007 del Estatuto del Trabajo Autónomo (LETA).
- **DOC-02:** Real Decreto-ley 13/2022 (Reforma del sistema de cotización en función de los rendimientos netos).
- **DOC-03:** Real Decreto 2064/1995 (Reglamento General de Cotización de la Seguridad Social).
- **DOC-04:** Real Decreto 84/1996 (Reglamento de Inscripción y Afiliación a la Seguridad Social).
- **DOC-05:** Ley 35/2006 del IRPF (Rendimientos de actividades económicas, arts. 27 a 32).
- **DOC-06:** Real Decreto Legislativo 8/2015 (Texto Refundido de la Ley General de la Seguridad Social).
- **DOC-07:** Ley 37/1992 del Impuesto sobre el Valor Añadido (IVA).
- **DOC-08:** Real Decreto 1065/2007 (Reglamento General de Gestión e Inspección Tributaria y censos/DUE).
- **DOC-09:** Real Decreto Legislativo 2/2015 (Estatuto de los Trabajadores - Contratación laboral).
- **DOC-10:** Ley 58/2003 General Tributaria (Infracciones, sanciones y prescripción).

### Capa 2: Procedimientos Administrativos Oficiales (AEAT y Seguridad Social)
- **DOC-11:** Manual Práctico de Renta e IRPF de la Agencia Tributaria (Gastos deducibles, suministros y afectación parcial de vivienda).
- **DOC-12:** Manual Práctico de IVA de la Agencia Tributaria (Regímenes de estimación, deducciones y rectificaciones).
- **DOC-13:** Guías Oficiales de Modelos Tributarios (036/037 alta censal, 130 pagos fraccionados, 303 trimestral IVA, 111 y 115 retenciones).
- **DOC-14:** Guía de Cotización RETA (Tablas definitivas de 15 tramos 2026, cuota reducida de 80€ y regularización anual).
- **DOC-15:** Guía de Primera Contratación Laboral (Inscripción de empresa TGSS, solicitud de código CCC modelo TA.6, alta previa de trabajadores y registro en el SEPE).
- **DOC-16:** Ventanilla Única y Sistema CIRCE (Documento Único Electrónico DUE y Puntos de Atención al Emprendedor PAE).

---

## Despliegue y Reproducibilidad

### Método 1: Contenedores con Docker Compose (Recomendado)

El archivo `docker-compose.yml` aprovisiona automáticamente el entorno completo: servidor Ollama, descarga desatendida de modelos e inicio de la aplicación web con soporte de GPU:

```bash
# 1. Clonar el repositorio
git clone https://github.com/davaaes/chat_autonomos.git
cd chat_autonomos

# 2. Levantar el stack completo (Ollama + Descarga de Modelos + Servidor Starlette)
docker compose up
```

La interfaz estará operativa de inmediato en `http://localhost:8000`.

> **Despliegue con Ollama preexistente en el host:**  
> Si ya ejecutas Ollama en tu máquina local y deseas reutilizar los modelos sin descargas adicionales dentro del contenedor:
> ```bash
> docker compose -f docker-compose.host-ollama.yml up
> ```

---

### Método 2: Ejecución Local Nativa (con `uv`)

#### Prerrequisitos
- [uv](https://github.com/astral-sh/uv) instalado en el sistema.
- [Ollama](https://ollama.ai/) en ejecución local con los modelos base:
  ```bash
  ollama pull qwen2.5:3b
  ollama pull qwen2.5:1.5b
  ollama pull bge-m3
  ```

#### Comandos de Inicio
```bash
# 1. Sincronizar dependencias del entorno virtual
uv sync

# 2. Iniciar el servidor web interactivo
uv run python scripts/run_ui.py
```

Accede desde cualquier navegador a `http://localhost:8000`.

---

## Suite de Pruebas Automatizadas

El proyecto cuenta con **37 pruebas unitarias y de integración** que validan el parsing de documentos, los filtros defensivos de guardrails, los esquemas Pydantic v2, la carga de índices y los endpoints REST:

```bash
uv run pytest -v
```

```text
tests/test_automerging_rag.py ....               [ 10%]
tests/test_cleaners.py .                         [ 13%]
tests/test_guardrails.py ...................     [ 64%]
tests/test_indexing_rag.py .......               [ 83%]
tests/test_segmenter.py ..                       [ 89%]
tests/test_ui.py ....                            [100%]

======================= 37 passed in 25.83s =======================
```

---

## Estructura del Repositorio

```text
chat_autonomos/
├── data/
│   ├── clean/                   # Documentos normativos depurados
│   ├── indices/                 # Índices vectoriales persistidos en disco
│   ├── json/                    # 513 documentos atómicos con metadatos de procedencia
│   └── pdf/                     # Manuales oficiales de la AEAT y la Seguridad Social
├── docs/
│   └── assets/                  # Recursos visuales de la interfaz y demostración animada (GIF)
├── scripts/
│   ├── run_ui.py                # Punto de entrada de la interfaz web
│   ├── run_rag_benchmark.py     # Banco de pruebas comparativo de estrategias RAG
│   ├── benchmark_automerging.py # Estudio experimental de granularidades AutoMerging
│   └── generate_ui_assets.py    # Generador automatizado de capturas con Playwright
├── src/
│   └── legal_rag_boe/
│       ├── boe_client.py        # Cliente HTTP de la API oficial del BOE
│       ├── config.py            # Gestión centralizada de configuración
│       ├── guardrails/          # Capa defensiva (Input/Output guards, PII, Re-Ask)
│       ├── indexing/            # Estrategias LlamaIndex, BGE-M3, Rerankers, TruLens
│       └── ui/                  # Servidor Starlette, endpoints REST y SPA Tailwind
├── tests/                       # 37 pruebas unitarias con pytest
├── Dockerfile                   # Imagen de producción multi-capa optimizada con uv
├── docker-compose.yml           # Stack autónomo con servicio Ollama
├── NOTICE                       # Atribución formal de modelos de terceros y BOE
├── pyproject.toml               # Manifiesto de dependencias y configuración del paquete
└── README.md                    # Documentación técnica del proyecto
```

---

## Licencia y Atribución de Modelos Upstream

El código fuente de este proyecto está licenciado bajo los términos de la licencia **[MIT](LICENSE)**.

Los pesos de los modelos de inteligencia artificial y embeddings utilizados en este proyecto se descargan bajo demanda desde sus registros oficiales y están sujetos a sus respectivas licencias originales de distribución (detalladas en el archivo [NOTICE](NOTICE)):

| Componente del Pipeline | Modelo Utilizado | Proveedor / Autor | Licencia Upstream | Ámbito de Uso |
| :--- | :--- | :--- | :--- | :--- |
| **Embeddings Densos** | [`bge-m3`](https://huggingface.co/BAAI/bge-m3) | BAAI | **[MIT](https://huggingface.co/BAAI/bge-m3/blob/main/LICENSE)** | Libre (académico y comercial) |
| **Auditor de Guardrails** | [`qwen2.5:1.5b`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | Alibaba Cloud (Qwen) | **[Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)** | Libre (académico y comercial) |
| **Juez de Evaluación** | [`qwen2.5:7b`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | Alibaba Cloud (Qwen) | **[Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)** | Libre (académico y comercial) |
| **Redactor / Generador UI** | [`qwen2.5:3b`](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) | Alibaba Cloud (Qwen) | **[Qwen Research](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/LICENSE)** | Libre para desarrollo, portfolio e investigación |
| **Reranker Neuronal** | [`jina-reranker-v3.5`](https://huggingface.co/jinaai/jina-reranker-v3.5) | Jina AI | **[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)** | Libre para uso no comercial y académico |

> **Cumplimiento Legal y Desacoplamiento Comercial:**  
> - El repositorio **no redistribuye binarios ni pesos de modelos propietarios** en Git (protegido en `.gitignore`).
> - La inclusión de `jina-reranker-v3.5` bajo **CC BY-NC 4.0** cumple estrictamente los términos de atribución y uso no comercial para este proyecto de código abierto y portfolio.
> - Para despliegues comerciales enterprise donde no aplique CC BY-NC 4.0, el sistema cuenta con desacoplamiento nativo: basta con activar el modo **"Sin Reranker"** en la interfaz, sustituirlo por un reranker Apache 2.0 (`bge-reranker-large`), y promover como redactor a `qwen2.5:7b` (Apache 2.0).
> - La reutilización de textos normativos del BOE se realiza conforme a la Ley 37/2007 de reutilización de la información del sector público.

---

## Descargo de Responsabilidad

La información proporcionada por este asistente tiene finalidad estrictamente orientativa y de divulgación técnica basada en fuentes primarias oficiales del BOE y manuales de la Agencia Tributaria. No constituye asesoramiento legal, contable ni tributario vinculante en los términos de la Ley 58/2003 General Tributaria.
