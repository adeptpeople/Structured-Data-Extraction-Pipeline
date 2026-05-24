from __future__ import annotations

import jsonschema
from jsonschema import Draft202012Validator

from extraction_pipeline.extraction.tool_schema import EXTRACTION_JSON_SCHEMA
from extraction_pipeline.schemas.validation import (
    FailureMode,
    RetryClassification,
    ValidationResult,
)

_VALIDATOR = Draft202012Validator(EXTRACTION_JSON_SCHEMA)


def validate_json_schema(raw_dict: dict, attempt: int = 1) -> ValidationResult:
    errors = list(_VALIDATOR.iter_errors(raw_dict))
    if not errors:
        return ValidationResult.success(attempt)

    messages = [e.message for e in errors]
    return ValidationResult.failure(
        modes=[FailureMode.json_schema_violation],
        messages=messages,
        classification=RetryClassification.retry_resolvable,
        attempt=attempt,
    )
