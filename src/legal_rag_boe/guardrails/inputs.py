from __future__ import annotations

import re
from typing import Any

from legal_rag_boe.guardrails.core import (
    FailResult,
    Guard,
    OnFailAction,
    PassResult,
    ValidationOutcome,
    ValidationResult,
    Validator,
)


class PromptInjectionValidator(Validator):
    """Detecta intentos de saltarse las restricciones del sistema o manipular el contexto."""

    PATRONES_HOSTILES = [
        r"\bignora\s+(?:todas\s+)?(?:las\s+)?(?:instrucciones|reglas|normas|órdenes)\b",
        r"\bignore\s+(?:all\s+)?(?:previous\s+)?(?:instructions|rules|prompts)\b",
        r"\bolvida\s+(?:todas\s+)?(?:las\s+)?(?:instrucciones|reglas)\b",
        r"\bforget\s+(?:all\s+)?(?:previous\s+)?(?:instructions|rules)\b",
        r"\b(?:actúa\s+como|act\s+as|simula\s+que\s+eres|pretend\s+you\s+are)\b",
        r"\b(?:dan\s+mode|developer\s+mode|jailbreak)\b",
        r"\b(?:muestra|revela|dime)\s+(?:tu\s+)?(?:system\s+)?prompt\b",
        r"\b(?:show|reveal)\s+(?:your\s+)?(?:system\s+)?prompt\b",
        r"<\|im_start\|>|<\|im_end\|>|\[INST\]|\[/INST\]|\bsystem:\s*",
    ]
    _COMBINED_REGEX = re.compile("|".join(f"(?:{p})" for p in PATRONES_HOSTILES), flags=re.IGNORECASE)

    def __init__(self, on_fail: OnFailAction | str = OnFailAction.EXCEPTION) -> None:
        super().__init__(on_fail=on_fail)

    def validate(self, value: Any, metadata: dict[str, Any] | None = None) -> ValidationResult:
        texto = str(value)
        if self._COMBINED_REGEX.search(texto):
            return FailResult(
                error_message="Consulta bloqueada por riesgo de seguridad e inyección de prompt.",
                fix_value="Consulta bloqueada: violación de directivas de seguridad.",
                action=self.on_fail,
            )
        return PassResult()


class PIIValidator(Validator):
    """Detecta y anonimiza identificadores fiscales, cuentas bancarias y datos personales."""

    REGEX_DNI_NIE = re.compile(r"\b[XYZxyz]?\d{7,8}[A-Za-z]\b")
    REGEX_IBAN = re.compile(r"\bES\d{2}[ \-]?(?:\d{4}[ \-]?){5}\b")
    REGEX_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
    REGEX_TELEFONO = re.compile(r"(?:\+34|0034)?[ \-]?[6789]\d{2}[ \-]?\d{3}[ \-]?\d{3}\b")

    def __init__(self, on_fail: OnFailAction | str = OnFailAction.FIX) -> None:
        super().__init__(on_fail=on_fail)

    def validate(self, value: Any, metadata: dict[str, Any] | None = None) -> ValidationResult:
        texto = str(value)
        saneado = texto

        if "@" in texto:
            saneado = self.REGEX_EMAIL.sub("[EMAIL ANONIMIZADO]", saneado)
        if any(c.isdigit() for c in texto):
            if "es" in texto.lower():
                saneado = self.REGEX_IBAN.sub("[IBAN ANONIMIZADO]", saneado)
            saneado = self.REGEX_DNI_NIE.sub("[DNI/NIE ANONIMIZADO]", saneado)
            saneado = self.REGEX_TELEFONO.sub("[TELÉFONO ANONIMIZADO]", saneado)

        if saneado != texto:
            return FailResult(
                error_message="Datos personales identificables (PII) detectados y anonimizados.",
                fix_value=saneado,
                action=self.on_fail,
            )
        return PassResult()


class ScopeValidator(Validator):
    """Garantiza que la consulta pertenezca estrictamente al ámbito de autónomos persona física."""

    PATRONES_FUERA_DE_ALCANCE = [
        (
            r"\b(?:sociedad\s+limitada|s\.l\.|sociedad\s+anónima|s\.a\.|impuesto\s+sobre\s+sociedades|capital\s+social\s+mínimo|reparto\s+de\s+dividendos\s+de\s+sl)\b",
            "sociedades mercantiles",
        ),
        (
            r"\b(?:país\s+vasco|euskadi|bizkaia|vizcaya|gipuzkoa|guipúzcoa|araba|álava|navarra|concierto\s+económico|convenio\s+económico|hacienda\s+foral)\b",
            "regímenes forales",
        ),
        (
            r"\b(?:delito\s+penal|código\s+penal|pena\s+de\s+prisión|cárcel|estafa\s+procesal|blanqueo\s+de\s+capitales)\b",
            "derecho penal",
        ),
        (
            r"\b(?:fútbol|champions(?:\s+league)?|real\s+madrid|barcelona|atletico\s+de\s+madrid|laliga|baloncesto|nba|tenis|roland\s+garros|mundial\s+de\s+fútbol)\b",
            "deportes y entretenimiento",
        ),
        (
            r"\b(?:números?\s+primos?|raíz\s+cuadrada|teorema\s+de\s+pitágoras|ecuación\s+de\s+segundo\s+grado|fotosíntesis|distancia\s+(?:a\s+la|de\s+la)\s+luna)\b",
            "matemáticas y ciencia general",
        ),
        (
            r"\b(?:receta\s+de|cómo\s+hacer\s+una?\s+tarta|tarta\s+de\s+manzana|ingredientes\s+para\s+cocinar|receta\s+culinaria)\b",
            "cocina y gastronomía",
        ),
        (
            r"\b(?:película|serie\s+de\s+netflix|actores|actrices|cantante|concierto\s+musical|eurovisión)\b",
            "cultura y espectáculos",
        ),
        (
            r"\b(?:javascript|python|css|html|script\s+bash|c\+\+|algoritmo\s+de\s+ordenación)\b",
            "desarrollo de software",
        ),
    ]
    _COMPILED = tuple(
        (re.compile(p, flags=re.IGNORECASE), motivo) for p, motivo in PATRONES_FUERA_DE_ALCANCE
    )

    RESPUESTA_DEFLEXION = (
        "Esta consulta queda fuera del alcance del asesor. "
        "El asistente está especializado exclusivamente en fiscalidad, trámites y normativa para autónomos en España. "
        "No se han encontrado fuentes normativas oficiales aplicables a este tema."
    )

    def __init__(self, on_fail: OnFailAction | str = OnFailAction.REFRAIN) -> None:
        super().__init__(on_fail=on_fail)

    def validate(self, value: Any, metadata: dict[str, Any] | None = None) -> ValidationResult:
        texto = str(value)
        for pattern, motivo in self._COMPILED:
            if pattern.search(texto):
                return FailResult(
                    error_message=f"Consulta fuera de alcance ({motivo}).",
                    fix_value=self.RESPUESTA_DEFLEXION,
                    action=self.on_fail,
                )
        return PassResult()


class InputGuard:
    """Orquestador de seguridad de entrada que ejecuta filtros de inyección, PII y alcance."""

    def __init__(
        self,
        injection_action: OnFailAction | str = OnFailAction.REFRAIN,
        pii_action: OnFailAction | str = OnFailAction.FIX,
        scope_action: OnFailAction | str = OnFailAction.REFRAIN,
    ) -> None:
        self.guard = Guard([
            PromptInjectionValidator(on_fail=injection_action),
            PIIValidator(on_fail=pii_action),
            ScopeValidator(on_fail=scope_action),
        ])

    def validate(self, query: str) -> ValidationOutcome:
        """Sanea y valida la consulta del usuario antes del acceso al motor RAG."""
        return self.guard.validate(query)
