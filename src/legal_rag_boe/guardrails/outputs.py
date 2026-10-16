from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

from legal_rag_boe.guardrails.core import (
    FailResult,
    Guard,
    OnFailAction,
    PassResult,
    ValidationOutcome,
    ValidationResult,
    Validator,
)

DISCLAIMER_PRECEPTIVO = (
    "AVISO: Información de carácter orientativo y divulgativo basada en normativa oficial del BOE. "
    "No constituye asesoramiento legal o tributario vinculante ni dictamen profesional."
)


class RespuestaLegalValidada(BaseModel):
    """Contrato formal tipado de la respuesta final del asistente legal."""

    resumen_ejecutivo: str = Field(
        ...,
        description="Explicación clara, directa y formal de la respuesta legal para el autónomo.",
    )
    articulos_citados: list[str] = Field(
        default_factory=list,
        description="Artículos, leyes o modelos oficiales que sustentan la respuesta.",
    )
    requisitos_o_plazos: list[str] = Field(
        default_factory=list,
        description="Condiciones, casillas, plazos o importes cuantitativos clave.",
    )
    grado_certeza: Literal["alta", "media", "baja"] = Field(
        default="alta",
        description="Grado de anclaje normativo contrastado con las fuentes recuperadas.",
    )
    aviso_legal: str = Field(
        default=DISCLAIMER_PRECEPTIVO,
        description="Cláusula inmutable de exención de responsabilidad obligatoria.",
    )

    @field_validator("requisitos_o_plazos", "articulos_citados", mode="before")
    @classmethod
    def normalizar_listas(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            v_str = v.strip()
            return [v_str] if v_str else []
        if isinstance(v, list):
            return [str(item).strip() for item in v if str(item).strip()]
        return []

    @field_validator("grado_certeza", mode="before")
    @classmethod
    def normalizar_certeza(cls, v: Any) -> str:
        if isinstance(v, str):
            v_clean = v.strip().lower()
            if "alta" in v_clean:
                return "alta"
            if "media" in v_clean:
                return "media"
            if "baja" in v_clean:
                return "baja"
        return "baja"

    @field_validator("resumen_ejecutivo", mode="before")
    @classmethod
    def normalizar_resumen(cls, v: Any) -> str:
        if not isinstance(v, str):
            return str(v) if v is not None else ""
        texto = v.strip()
        if texto.startswith("```") or (texto.startswith("{") and "resumen_ejecutivo" in texto):
            candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", texto, flags=re.MULTILINE).strip()
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict) and "resumen_ejecutivo" in parsed:
                    return str(parsed["resumen_ejecutivo"]).strip()
            except Exception:
                match = re.search(r'"resumen_ejecutivo"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', candidate)
                if match:
                    try:
                        return match.group(1).encode("utf-8").decode("unicode_escape").strip()
                    except Exception:
                        return match.group(1).strip()
        return texto


class DisclaimerValidator(Validator):
    """Garantiza la presencia estricta e inmutable de la cláusula de descargo legal."""

    def __init__(self, on_fail: OnFailAction | str = OnFailAction.FIX) -> None:
        super().__init__(on_fail=on_fail)

    def validate(self, value: Any, metadata: dict[str, Any] | None = None) -> ValidationResult:
        if isinstance(value, RespuestaLegalValidada):
            texto_aviso = value.aviso_legal.strip()
            if not texto_aviso or "asesoramiento legal" not in texto_aviso.lower():
                value.aviso_legal = DISCLAIMER_PRECEPTIVO
                return FailResult(
                    error_message="Aviso legal ausente o modificado; se ha restablecido el texto preceptivo.",
                    fix_value=value,
                    action=self.on_fail,
                )
            return PassResult()
        return PassResult()


class ArticulosCitadosValidator(Validator):
    """Verifica y normaliza la lista de citas y preceptos jurídicos."""

    def __init__(self, on_fail: OnFailAction | str = OnFailAction.FIX) -> None:
        super().__init__(on_fail=on_fail)

    def validate(self, value: Any, metadata: dict[str, Any] | None = None) -> ValidationResult:
        if isinstance(value, RespuestaLegalValidada):
            citas_unicas = list(dict.fromkeys(c.strip() for c in value.articulos_citados if c.strip()))
            if len(citas_unicas) != len(value.articulos_citados):
                value.articulos_citados = citas_unicas
                return FailResult(
                    error_message="Se eliminaron citas duplicadas o vacías.",
                    fix_value=value,
                    action=self.on_fail,
                )
        return PassResult()


class CitasNormativasValidator(Validator):
    """Verifica que las citas normativas figuren en el contexto documental recuperado."""

    STOPWORDS_CITAS = {
        "doc", "ley", "art", "articulo", "artículo", "real", "decreto", "orden", "modelo",
        "para", "los", "las", "del", "con", "por", "sobre", "reglamento", "estatuto",
    }
    REGEX_DOC = re.compile(r"doc-\d+", flags=re.IGNORECASE)
    REGEX_SPLIT = re.compile(r"[\s,;./\-_]+")

    def __init__(self, on_fail: OnFailAction | str = OnFailAction.FIX) -> None:
        super().__init__(on_fail=on_fail)

    def validate(self, value: Any, metadata: dict[str, Any] | None = None) -> ValidationResult:
        if not isinstance(value, RespuestaLegalValidada) or not value.articulos_citados:
            return PassResult()

        contexto = str(metadata.get("context", "") or metadata.get("contexto", "")).lower() if metadata else ""
        if not contexto:
            return PassResult()

        citas_validas: list[str] = []
        citas_sospechosas: list[str] = []

        for cita in value.articulos_citados:
            cita_limpia = cita.strip()
            doc_matches = self.REGEX_DOC.findall(cita_limpia.lower())
            if doc_matches:
                if all(dm in contexto for dm in doc_matches):
                    citas_validas.append(cita_limpia)
                else:
                    citas_sospechosas.append(cita_limpia)
                continue

            tokens = [
                t for t in self.REGEX_SPLIT.split(cita_limpia.lower())
                if len(t) > 2 and t not in self.STOPWORDS_CITAS
            ]
            if tokens and any(token in contexto for token in tokens):
                citas_validas.append(cita_limpia)
            else:
                citas_sospechosas.append(cita_limpia)

        if citas_sospechosas:
            if self.on_fail == OnFailAction.FIX:
                value.articulos_citados = citas_validas
                return FailResult(
                    error_message=f"Citas no encontradas en el contexto eliminadas: {', '.join(citas_sospechosas)}",
                    fix_value=value,
                    action=self.on_fail,
                )
            return FailResult(
                error_message=f"Citas jurídicas no respaldadas en el contexto: {', '.join(citas_sospechosas)}",
                fix_value=value,
                action=self.on_fail,
            )

        return PassResult()


class HallucinationValidator(Validator):
    """Verifica el anclaje normativo de la respuesta mediante auditoría con modelo ligero."""

    PROMPT_AUDITORIA = """Contexto normativo:
\"\"\"{contexto}\"\"\"

Afirmación a auditar:
\"\"\"{afirmacion}\"\"\"

Instrucción:
Determina si la afirmación está respaldada de forma rigurosa por el contexto normativo oficial.
Reglas:
- Fraseos equivalentes o sinónimos de conceptos legales se consideran VÁLIDOS.
- Cifras distintas, plazos erróneos, modelos tributarios equivocados u obligaciones no mencionadas son MOTIVO DE FALLO.

Formato de salida requerido:
MOTIVO: <explicación en una frase>
VEREDICTO: <PASS o FAIL>"""

    def __init__(
        self,
        llm: Any = None,
        model_name: str = "qwen2.5:1.5b",
        on_fail: OnFailAction | str = OnFailAction.REASK,
    ) -> None:
        super().__init__(on_fail=on_fail)
        self.model_name = model_name
        self._llm = llm

    @property
    def llm(self) -> Any:
        if self._llm is None:
            from llama_index.llms.ollama import Ollama

            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            self._llm = Ollama(
                model=self.model_name,
                base_url=base_url,
                request_timeout=15.0,
                temperature=0.0,
                additional_kwargs={
                    "num_ctx": 2048,
                    "num_predict": 70,
                    "keep_alive": "30m",
                },
            )
        return self._llm

    def validate(self, value: Any, metadata: dict[str, Any] | None = None) -> ValidationResult:
        if not isinstance(value, RespuestaLegalValidada):
            return PassResult()

        contexto = str(metadata.get("context", "") or metadata.get("contexto", "")).strip() if metadata else ""
        if not contexto:
            return PassResult()

        partes_afirmacion = [value.resumen_ejecutivo] + value.requisitos_o_plazos
        afirmacion = " ".join(p.strip() for p in partes_afirmacion if p.strip())
        if not afirmacion:
            return PassResult()

        afirmacion_lower = afirmacion.lower()
        declaracion_ausencia = (
            "no consta" in afirmacion_lower
            or "no se encuentra en el contexto" in afirmacion_lower
            or "no figura en el contexto" in afirmacion_lower
            or "no se menciona en las fuentes" in afirmacion_lower
            or "no fue posible encontrar" in afirmacion_lower
        )
        if declaracion_ausencia and not value.articulos_citados:
            return PassResult()

        prompt = self.PROMPT_AUDITORIA.format(contexto=contexto, afirmacion=afirmacion)

        try:
            resp = self.llm.complete(prompt)
            texto_resp = resp.text if hasattr(resp, "text") else str(resp)
            texto_limpio = texto_resp.strip().upper()

            es_fallo = False
            motivo = "Afirmación no respaldada en las fuentes normativas."

            if (
                "VEREDICTO: FAIL" in texto_limpio
                or "VERDICT: FAIL" in texto_limpio
                or texto_limpio.startswith("FAIL")
                or texto_limpio.startswith("YES")
            ):
                es_fallo = True
                for line in texto_resp.splitlines():
                    line_upper = line.upper()
                    if line_upper.startswith("MOTIVO:") or line_upper.startswith("REASON:"):
                        motivo = line.split(":", 1)[1].strip()
                        break

            if es_fallo:
                if self.on_fail == OnFailAction.FIX:
                    value.grado_certeza = "baja"
                    return FailResult(
                        error_message=f"Alucinación detectada: {motivo}",
                        fix_value=value,
                        action=self.on_fail,
                    )
                return FailResult(
                    error_message=f"Alucinación detectada: {motivo}",
                    fix_value=value,
                    action=self.on_fail,
                )
        except Exception as err:
            return FailResult(
                error_message=f"Fallo en auditoría de alucinaciones: {err}",
                fix_value=value,
                action=OnFailAction.FIX,
            )

        return PassResult()


class OutputGuard:
    """Orquestador de salida que procesa el texto en bruto del LLM y garantiza el contrato Pydantic."""

    JSON_BLOCK_REGEX = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```")

    def __init__(
        self,
        target_schema: type[BaseModel] = RespuestaLegalValidada,
        validators: list[Validator] | None = None,
        verifier_llm: Any = None,
        enable_hallucination_check: bool = False,
    ) -> None:
        self.target_schema = target_schema
        if validators is not None:
            active_validators = list(validators)
        else:
            active_validators = [
                DisclaimerValidator(),
                ArticulosCitadosValidator(),
                CitasNormativasValidator(),
            ]
            if enable_hallucination_check or verifier_llm is not None:
                active_validators.append(HallucinationValidator(llm=verifier_llm))
        self.guard = Guard(active_validators)

    def parse_raw_text(self, raw_text: str) -> tuple[dict[str, Any] | None, str | None]:
        """Extrae el objeto JSON desde bloques markdown o texto llano."""
        text = raw_text.strip()
        match = self.JSON_BLOCK_REGEX.search(text)
        candidate = match.group(1).strip() if match else text

        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed, None
            return None, f"El JSON debe ser un objeto/diccionario, se obtuvo {type(parsed).__name__}"
        except json.JSONDecodeError as err:
            brace_start = candidate.find("{")
            brace_end = candidate.rfind("}")
            if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
                sub_candidate = candidate[brace_start : brace_end + 1]
                try:
                    parsed = json.loads(sub_candidate)
                    if isinstance(parsed, dict):
                        return parsed, None
                except json.JSONDecodeError:
                    cleaned = re.sub(r",\s*([}\]])", r"\1", sub_candidate)
                    try:
                        parsed = json.loads(cleaned)
                        if isinstance(parsed, dict):
                            return parsed, None
                    except json.JSONDecodeError as clean_err:
                        err = clean_err
            return None, f"Error de sintaxis JSON: {err}"

    def validate(
        self,
        raw_text_or_model: str | RespuestaLegalValidada,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[RespuestaLegalValidada | None, ValidationOutcome]:
        """Convierte y valida la salida frente a Pydantic y las reglas de descargo legal."""
        if isinstance(raw_text_or_model, RespuestaLegalValidada):
            model_instance = raw_text_or_model
            outcome = self.guard.validate(model_instance, metadata=metadata)
            if not outcome.validation_passed and OnFailAction.REASK in outcome.failed_actions:
                return None, outcome
            return outcome.validated_output, outcome

        parsed_dict, parse_err = self.parse_raw_text(raw_text_or_model)
        if parse_err or parsed_dict is None:
            fail_outcome = ValidationOutcome(
                raw_input=raw_text_or_model,
                validated_output=None,
                validation_passed=False,
                error_messages=[parse_err or "No se pudo extraer JSON válido de la respuesta."],
                failed_actions=[OnFailAction.REASK],
            )
            return None, fail_outcome

        try:
            model_instance = self.target_schema.model_validate(parsed_dict)
        except ValidationError as val_err:
            fail_outcome = ValidationOutcome(
                raw_input=raw_text_or_model,
                validated_output=None,
                validation_passed=False,
                error_messages=[str(val_err)],
                failed_actions=[OnFailAction.REASK],
            )
            return None, fail_outcome

        outcome = self.guard.validate(model_instance, metadata=metadata)
        if not outcome.validation_passed and OnFailAction.REASK in outcome.failed_actions:
            return None, outcome
        validated_model: RespuestaLegalValidada = outcome.validated_output
        return validated_model, outcome

