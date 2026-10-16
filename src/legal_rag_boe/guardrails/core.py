from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any


class OnFailAction(str, Enum):
    """Acciones deterministas a ejecutar ante un fallo de validación."""

    EXCEPTION = "exception"
    REASK = "reask"
    FIX = "fix"
    FILTER = "filter"
    REFRAIN = "refrain"


class ValidationResult:
    """Resultado base devuelto por cualquier validador."""

    def __init__(self, passed: bool, error_message: str | None = None) -> None:
        self.passed = passed
        self.error_message = error_message


class PassResult(ValidationResult):
    """Indica que el valor supera la validación sin incidencias."""

    def __init__(self) -> None:
        super().__init__(passed=True, error_message=None)


class FailResult(ValidationResult):
    """Indica incumplimiento de política con acción correctiva asociada."""

    def __init__(
        self,
        error_message: str,
        fix_value: Any = None,
        action: OnFailAction | str = OnFailAction.EXCEPTION,
    ) -> None:
        super().__init__(passed=False, error_message=error_message)
        self.fix_value = fix_value
        self.action = OnFailAction(action) if isinstance(action, str) else action


class Validator(ABC):
    """Clase base abstracta para validadores de entrada y salida."""

    def __init__(self, on_fail: OnFailAction | str = OnFailAction.EXCEPTION) -> None:
        self.on_fail = OnFailAction(on_fail) if isinstance(on_fail, str) else on_fail

    @abstractmethod
    def validate(self, value: Any, metadata: dict[str, Any] | None = None) -> ValidationResult:
        """Aplica la regla de validación sobre el valor proporcionado."""
        raise NotImplementedError


class ValidationOutcome:
    """Balance consolidado tras ejecutar una cadena de validadores en un Guard."""

    def __init__(
        self,
        raw_input: Any,
        validated_output: Any,
        validation_passed: bool,
        error_messages: list[str] | None = None,
        failed_actions: list[OnFailAction] | None = None,
    ) -> None:
        self.raw_input = raw_input
        self.validated_output = validated_output
        self.validation_passed = validation_passed
        self.error_messages = error_messages or []
        self.failed_actions = failed_actions or []


class Guard:
    """Orquestador de seguridad que ejecuta cadenas de validadores deterministas."""

    def __init__(self, validators: list[Validator] | None = None) -> None:
        self.validators: list[Validator] = list(validators) if validators else []

    def use(self, validator: Validator) -> Guard:
        """Añade un validador a la cadena de ejecución."""
        self.validators.append(validator)
        return self

    def validate(self, value: Any, metadata: dict[str, Any] | None = None) -> ValidationOutcome:
        """Evalúa el valor por cada validador aplicando la acción ante fallo configurada."""
        current_value = value
        all_passed = True
        errors: list[str] = []
        actions: list[OnFailAction] = []

        for val in self.validators:
            result = val.validate(current_value, metadata=metadata)
            if not result.passed:
                all_passed = False
                fail_res = result if isinstance(result, FailResult) else FailResult(result.error_message or "Error")
                action = fail_res.action or val.on_fail
                errors.append(fail_res.error_message or "Fallo de validación.")
                actions.append(action)

                if action == OnFailAction.EXCEPTION:
                    raise ValueError(f"Guardrail Exception: {fail_res.error_message}")
                if action == OnFailAction.FIX and fail_res.fix_value is not None:
                    current_value = fail_res.fix_value
                elif action == OnFailAction.REFRAIN:
                    current_value = fail_res.fix_value

        return ValidationOutcome(
            raw_input=value,
            validated_output=current_value,
            validation_passed=all_passed,
            error_messages=errors,
            failed_actions=actions,
        )

