"""
Stateless semantic rule checks applied after JSON Schema and Pydantic validation pass.
Each rule returns (passed: bool, error_message: str | None).
"""
from __future__ import annotations

import re
from typing import Optional

from extraction_pipeline.schemas.extraction import ExtractedDocument
from extraction_pipeline.schemas.validation import (
    FailureMode,
    RetryClassification,
    ValidationResult,
)

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _check_date_format(doc: ExtractedDocument) -> Optional[str]:
    if doc.publication_date is not None and not _ISO_DATE_RE.match(doc.publication_date):
        return (
            f"publication_date '{doc.publication_date}' must be ISO 8601 format YYYY-MM-DD. "
            f"If only month/year known use YYYY-MM-01. If year only use YYYY-01-01."
        )
    return None


def _check_other_detail_conditional(doc: ExtractedDocument) -> Optional[str]:
    from extraction_pipeline.schemas.extraction import DocumentType
    if doc.document_type == DocumentType.other and not doc.document_type_other_detail:
        return "document_type_other_detail must be non-null when document_type is 'other'."
    if doc.document_type != DocumentType.other and doc.document_type_other_detail is not None:
        return (
            "document_type_other_detail must be null when document_type is not 'other'. "
            f"Got '{doc.document_type_other_detail}'."
        )
    return None


def _check_currency_with_total(doc: ExtractedDocument) -> Optional[str]:
    if doc.invoice_total is not None and doc.currency is None:
        return "currency must be provided when invoice_total is non-null."
    return None


def _check_confidence_completeness(doc: ExtractedDocument) -> Optional[str]:
    scores = doc.confidence_scores
    field_pairs = [
        ("title", doc.title, scores.title),
        ("author", doc.author, scores.author),
        ("publication_date", doc.publication_date, scores.publication_date),
        ("invoice_total", doc.invoice_total, scores.invoice_total),
        ("citations", doc.citations, scores.citations),
    ]
    missing = []
    for field, value, conf in field_pairs:
        if value is not None and conf is None:
            missing.append(field)
    if missing:
        return f"Missing confidence scores for extracted fields: {missing}. Set a score for each non-null field."
    return None


_RULES = [
    _check_date_format,
    _check_other_detail_conditional,
    _check_currency_with_total,
    _check_confidence_completeness,
]


def validate_semantic(doc: ExtractedDocument, attempt: int = 1) -> ValidationResult:
    errors: list[str] = []
    for rule in _RULES:
        msg = rule(doc)
        if msg:
            errors.append(msg)

    if not errors:
        return ValidationResult.success(attempt)

    return ValidationResult.failure(
        modes=[FailureMode.semantic_rule],
        messages=errors,
        classification=RetryClassification.retry_resolvable,
        attempt=attempt,
    )
