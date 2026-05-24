from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel


class FailureMode(str, Enum):
    json_schema_violation = "json_schema_violation"
    pydantic_constraint = "pydantic_constraint"
    semantic_rule = "semantic_rule"
    missing_required_field = "missing_required_field"
    oversized_document = "oversized_document"
    malformed_json = "malformed_json"


class RetryClassification(str, Enum):
    retry_resolvable = "retry_resolvable"
    non_resolvable = "non_resolvable"


class ValidationResult(BaseModel):
    passed: bool
    failure_modes: list[FailureMode] = []
    error_messages: list[str] = []
    classification: Optional[RetryClassification] = None
    attempt_number: int = 0

    @classmethod
    def success(cls, attempt: int = 1) -> "ValidationResult":
        return cls(passed=True, attempt_number=attempt)

    @classmethod
    def failure(
        cls,
        modes: list[FailureMode],
        messages: list[str],
        classification: RetryClassification,
        attempt: int = 1,
    ) -> "ValidationResult":
        return cls(
            passed=False,
            failure_modes=modes,
            error_messages=messages,
            classification=classification,
            attempt_number=attempt,
        )


class ExtractionMetrics(BaseModel):
    document_id: str
    attempt_count: int
    validation_failures: int
    retry_successes: int
    non_resolvable_failures: int
    final_status: str
    processing_time_ms: int = 0
    retry_count: int = 0
    validation_errors: list[str] = []
    review_required: bool = False
    review_reason: list[str] = []
    confidence_distribution: dict = {}
