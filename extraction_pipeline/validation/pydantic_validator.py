from __future__ import annotations

import pydantic

from extraction_pipeline.schemas.extraction import ExtractedDocument
from extraction_pipeline.schemas.validation import (
    FailureMode,
    RetryClassification,
    ValidationResult,
)


def validate_pydantic(raw_dict: dict, attempt: int = 1) -> tuple[ValidationResult, ExtractedDocument | None]:
    try:
        doc = ExtractedDocument.model_validate(raw_dict)
        return ValidationResult.success(attempt), doc
    except pydantic.ValidationError as exc:
        messages = [f"{e['loc']}: {e['msg']}" for e in exc.errors()]
        return (
            ValidationResult.failure(
                modes=[FailureMode.pydantic_constraint],
                messages=messages,
                classification=RetryClassification.retry_resolvable,
                attempt=attempt,
            ),
            None,
        )
