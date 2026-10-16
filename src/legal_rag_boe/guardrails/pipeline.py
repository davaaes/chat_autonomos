from __future__ import annotations

import json
import re
import time
from typing import Any

from llama_index.core.llms import LLM
from pydantic import BaseModel, Field

from legal_rag_boe.guardrails.core import OnFailAction
from legal_rag_boe.guardrails.inputs import InputGuard, ScopeValidator
from legal_rag_boe.guardrails.outputs import DISCLAIMER_PRECEPTIVO, OutputGuard, RespuestaLegalValidada
from legal_rag_boe.indexing.strategies import BaseRAGStrategy, RerankedRAG


class FuenteDetallada(BaseModel):
    """Metadatos y fragmento textual de una fuente oficial recuperada."""

    id: str = ""
    doc_code: str = ""
    ley: str = ""
    titulo: str = ""
    score: float = 0.0
    texto: str = ""
    url: str | None = None


class ResultadoPipeline(BaseModel):
    """Resultado completo de la ejecución del pipeline seguro de consulta legal."""

    respuesta: RespuestaLegalValidada
    consulta_original: str
    consulta_saneada: str
    nodos_recuperados: list[str] = Field(
        default_factory=list,
        description="Textos o identificadores de los nodos recuperados por el motor RAG.",
    )
    fuentes: list[FuenteDetallada] = Field(
        default_factory=list,
        description="Detalle estructurado de cada fuente oficial consultada.",
    )
    bloqueada: bool = False
    motivo_bloqueo: str | None = None
    reasks_utilizados: int = 0
    latencia_ms: float = 0.0


PROMPT_SISTEMA_GUARDRAILS = """Eres un asistente jurídico especializado exclusivamente en autónomos personas físicas en España.
Tu cometido es responder con precisión y rigor basándote ÚNICAMENTE en el contexto normativo oficial proporcionado.

DIRECTIVAS INMUTABLES:
1. No inventes artículos, plazos ni números que no consten en las fuentes oficiales.
2. Si un dato no consta en el contexto, indícalo explícitamente en el resumen ejecutivo.
3. Responde OBLIGATORIAMAMENTE en un único bloque JSON conforme al siguiente contrato formal:

{{
  "resumen_ejecutivo": "Explicación directa, rigurosa y clara para el usuario.",
  "articulos_citados": ["Norma o artículo oficial citado, ej. Ley 35/2006 art. 30"],
  "requisitos_o_plazos": ["Condición, casilla o plazo específico"],
  "grado_certeza": "alta",
  "aviso_legal": "AVISO: Información de carácter orientativo y divulgativo basada en normativa oficial del BOE. No constituye asesoramiento legal o tributario vinculante ni dictamen profesional."
}}

CONTEXTO NORMATIVO RECUPERADO:
{contexto}

CONSULTA DEL AUTÓNOMO:
{consulta}
"""

PROMPT_REASK = """Tu respuesta anterior no superó los controles de seguridad jurídica y anclaje normativo.
Detalle del error detectado:
{error_detalle}

Respuesta anterior errónea:
{respuesta_previa}

Contexto normativo oficial:
{contexto}

Corrige el fallo respetando estrictamente el contexto oficial sin inventar datos, normas ni cifras.
Genera EXCLUSIVAMENTE el objeto JSON válido con el esquema indicado."""


class PipelineRAGSeguro:
    """Orquestador end-to-end de RAG seguro con filtros de entrada, salida y ciclo Re-Ask."""

    def __init__(
        self,
        retriever: BaseRAGStrategy,
        llm: Any,
        input_guard: InputGuard | None = None,
        output_guard: OutputGuard | None = None,
        verifier_llm: Any = None,
        enable_hallucination_guard: bool = False,
        max_reasks: int = 2,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.input_guard = input_guard or InputGuard()
        if output_guard is not None:
            self.output_guard = output_guard
        else:
            self.output_guard = OutputGuard(
                verifier_llm=verifier_llm,
                enable_hallucination_check=enable_hallucination_guard or (verifier_llm is not None),
            )
        self.max_reasks = max_reasks

    def ejecutar(self, consulta_usuario: str, top_k: int = 3) -> ResultadoPipeline:
        """Ejecuta el ciclo de seguridad completo: Input Guard -> RAG -> LLM -> Output Guard."""
        t0 = time.perf_counter()

        val_in = self.input_guard.validate(consulta_usuario)
        if not val_in.validation_passed and any(
            act in (OnFailAction.REFRAIN, OnFailAction.EXCEPTION) for act in val_in.failed_actions
        ):
            latencia = (time.perf_counter() - t0) * 1000.0
            error_msg = val_in.error_messages[0] if val_in.error_messages else "Consulta bloqueada."
            texto_deflexion = str(val_in.validated_output)
            return ResultadoPipeline(
                respuesta=RespuestaLegalValidada(
                    resumen_ejecutivo=texto_deflexion,
                    articulos_citados=[],
                    requisitos_o_plazos=[],
                    grado_certeza="baja",
                    aviso_legal=DISCLAIMER_PRECEPTIVO,
                ),
                consulta_original=consulta_usuario,
                consulta_saneada=texto_deflexion,
                nodos_recuperados=[],
                bloqueada=True,
                motivo_bloqueo=error_msg,
                reasks_utilizados=0,
                latencia_ms=latencia,
            )

        consulta_saneada = str(val_in.validated_output)

        retrieved_nodes = self.retriever.retrieve(consulta_saneada, top_k=top_k)
        es_reranker = any(n.score is not None and n.score < 0.0 for n in retrieved_nodes) or isinstance(
            self.retriever, RerankedRAG
        )
        umbral_relevancia = 0.05 if es_reranker else 0.45

        context_parts = []
        node_summaries = []
        fuentes: list[FuenteDetallada] = []
        for i, n in enumerate(retrieved_nodes, start=1):
            score_val = float(n.score) if n.score is not None else 0.0
            if score_val < umbral_relevancia:
                continue
            doc_id = str(n.node.metadata.get("doc_code", "DOC"))
            ley = str(n.node.metadata.get("ley", ""))
            art = str(n.node.metadata.get("titulo", ""))
            texto_nodo = n.node.get_content().strip()
            if not texto_nodo:
                continue
            header = f"[{doc_id} | {ley} | {art}]"
            context_parts.append(f"--- FUENTE {i}: {header} ---\n{texto_nodo}")
            node_summaries.append(f"{header} (Score: {score_val:.4f})")
            fuentes.append(
                FuenteDetallada(
                    id=f"f-{i}",
                    doc_code=doc_id,
                    ley=ley,
                    titulo=art,
                    score=score_val,
                    texto=texto_nodo,
                    url=n.node.metadata.get("canonical_url"),
                )
            )

        if not fuentes:
            latencia = (time.perf_counter() - t0) * 1000.0
            return ResultadoPipeline(
                respuesta=RespuestaLegalValidada(
                    resumen_ejecutivo=ScopeValidator.RESPUESTA_DEFLEXION,
                    articulos_citados=[],
                    requisitos_o_plazos=[],
                    grado_certeza="baja",
                    aviso_legal=DISCLAIMER_PRECEPTIVO,
                ),
                consulta_original=consulta_usuario,
                consulta_saneada=consulta_saneada,
                nodos_recuperados=[],
                fuentes=[],
                bloqueada=False,
                motivo_bloqueo="Esta consulta no se adapta al ámbito temático del asesor.",
                reasks_utilizados=0,
                latencia_ms=latencia,
            )

        context_str = "\n\n".join(context_parts)

        prompt = PROMPT_SISTEMA_GUARDRAILS.format(
            contexto=context_str,
            consulta=consulta_saneada,
        )

        resp = self.llm.complete(prompt)
        raw_text = resp.text if hasattr(resp, "text") else str(resp)

        reasks_count = 0
        meta = {"context": context_str, "consulta": consulta_saneada}
        modelo_validado, val_out = self.output_guard.validate(raw_text, metadata=meta)

        while modelo_validado is None and reasks_count < self.max_reasks:
            reasks_count += 1
            error_det = val_out.error_messages[0] if val_out.error_messages else "Formato inválido"
            reask_prompt = PROMPT_REASK.format(
                error_detalle=error_det,
                respuesta_previa=raw_text,
                contexto=context_str,
            )
            resp = self.llm.complete(reask_prompt)
            raw_text = resp.text if hasattr(resp, "text") else str(resp)
            modelo_validado, val_out = self.output_guard.validate(raw_text, metadata=meta)

        if modelo_validado is None:
            if isinstance(val_out.validated_output, RespuestaLegalValidada):
                modelo_validado = val_out.validated_output
                modelo_validado.grado_certeza = "baja"
            else:
                texto_limpio = self._extraer_resumen_limpio(raw_text)
                citas = self._extraer_lista_campo(raw_text, "articulos_citados")
                plazos = self._extraer_lista_campo(raw_text, "requisitos_o_plazos")
                modelo_validado = RespuestaLegalValidada(
                    resumen_ejecutivo=texto_limpio,
                    articulos_citados=citas,
                    requisitos_o_plazos=plazos,
                    grado_certeza="baja",
                    aviso_legal=DISCLAIMER_PRECEPTIVO,
                )

        latencia = (time.perf_counter() - t0) * 1000.0
        return ResultadoPipeline(
            respuesta=modelo_validado,
            consulta_original=consulta_usuario,
            consulta_saneada=consulta_saneada,
            nodos_recuperados=node_summaries,
            fuentes=fuentes,
            bloqueada=False,
            motivo_bloqueo=None,
            reasks_utilizados=reasks_count,
            latencia_ms=latencia,
        )

    @staticmethod
    def _extraer_resumen_limpio(raw_text: str) -> str:
        if not raw_text or not raw_text.strip():
            return "No se pudo estructurar una respuesta con las fuentes disponibles."

        texto = raw_text.strip()
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", texto)
        candidate = match.group(1).strip() if match else texto
        b_start = candidate.find("{")
        b_end = candidate.rfind("}")
        if b_start != -1 and b_end != -1 and b_end > b_start:
            sub = re.sub(r",\s*([}\]])", r"\1", candidate[b_start : b_end + 1])
            try:
                d = json.loads(sub)
                if isinstance(d, dict) and d.get("resumen_ejecutivo"):
                    return str(d["resumen_ejecutivo"]).strip()
            except json.JSONDecodeError:
                d = None

        regex_resumen = re.search(r'"resumen_ejecutivo"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', texto)
        if regex_resumen:
            try:
                return regex_resumen.group(1).encode("utf-8").decode("unicode_escape").strip()
            except (UnicodeDecodeError, UnicodeEncodeError):
                return regex_resumen.group(1).strip()

        if texto.startswith("{") or texto.startswith("```"):
            lineas_utiles = [
                l.strip()
                for l in texto.splitlines()
                if not re.match(r"^[{}[\]\",`]|^\s*\"[a-zA-Z_]+\"\s*:", l.strip())
            ]
            candidato_texto = " ".join(lineas_utiles).strip()
            if candidato_texto:
                return candidato_texto
            return "No se pudo estructurar una respuesta con suficiente certeza a partir de las fuentes oficiales."

        return texto

    @staticmethod
    def _extraer_lista_campo(raw_text: str, campo: str) -> list[str]:
        if not raw_text:
            return []
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_text)
        candidate = match.group(1).strip() if match else raw_text
        b_start = candidate.find("{")
        b_end = candidate.rfind("}")
        if b_start != -1 and b_end != -1 and b_end > b_start:
            try:
                d = json.loads(re.sub(r",\s*([}\]])", r"\1", candidate[b_start : b_end + 1]))
                if isinstance(d, dict) and isinstance(d.get(campo), list):
                    return [str(item).strip() for item in d[campo] if str(item).strip()]
            except json.JSONDecodeError:
                return []
        return []
