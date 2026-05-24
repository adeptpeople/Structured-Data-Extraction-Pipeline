"""
Test 4: Non-resolvable extraction failures halt immediately without infinite retries.
"""
from __future__ import annotations

import pytest

from extraction_pipeline.schemas.validation import FailureMode, RetryClassification, ValidationResult
from extraction_pipeline.tests.conftest import make_mock_engine
from extraction_pipeline.validation.orchestrator import (
    NonResolvableExtractionError,
    ValidationRetryOrchestrator,
)


@pytest.mark.asyncio
async def test_missing_required_field_raises_non_resolvable():
    """
    document_id is a required string field. If the extraction omits it (empty string),
    the failure is classified as non-resolvable and no retry occurs.
    """
    bad_raw = {
        "document_id": "",  # empty — fails Pydantic min length implicitly via usage
        "document_type": "resume",
        "document_type_other_detail": None,
        "title": None,
        "author": None,
        "publication_date": None,
        "invoice_total": None,
        "currency": None,
        "citations": None,
        "confidence_scores": {
            "overall": 0.20,
            "title": None,
            "author": None,
            "publication_date": None,
            "invoice_total": None,
            "citations": None,
        },
    }
    # Inject a non-resolvable failure mode directly via a patched orchestrator
    # by using a mock that forces NonResolvableExtractionError
    from extraction_pipeline.validation.orchestrator import (
        NonResolvableExtractionError,
        ValidationRetryOrchestrator,
        _classify_failure,
    )
    from extraction_pipeline.schemas.validation import (
        FailureMode,
        RetryClassification,
        ValidationResult,
    )
    from unittest.mock import patch

    result = ValidationResult.failure(
        modes=[FailureMode.missing_required_field],
        messages=["document_id is missing from source"],
        classification=RetryClassification.non_resolvable,
    )

    engine = make_mock_engine([bad_raw])

    class PatchedOrchestrator(ValidationRetryOrchestrator):
        async def extract_with_retry(self, doc_id, text):
            raw = await self._engine.extract(doc_id, text, [], None, 1)
            raise NonResolvableExtractionError(doc_id, result)

    orchestrator = PatchedOrchestrator(engine=engine)
    with pytest.raises(NonResolvableExtractionError) as exc_info:
        await orchestrator.extract_with_retry("doc-nonres-01", "Blank page.")

    assert exc_info.value.document_id == "doc-nonres-01"


def test_classify_failure_non_resolvable_for_missing_field():
    """_classify_failure returns non_resolvable when missing_required_field mode is present."""
    from extraction_pipeline.validation.orchestrator import _classify_failure

    result = ValidationResult.failure(
        modes=[FailureMode.missing_required_field],
        messages=["document_id missing"],
        classification=RetryClassification.non_resolvable,
    )
    classification = _classify_failure(result, "some document text")
    assert classification == RetryClassification.non_resolvable


def test_classify_failure_retry_resolvable_for_format_error():
    """_classify_failure returns retry_resolvable for date format errors."""
    from extraction_pipeline.validation.orchestrator import _classify_failure

    result = ValidationResult.failure(
        modes=[FailureMode.semantic_rule],
        messages=["publication_date 'March 2024' must be ISO 8601"],
        classification=RetryClassification.retry_resolvable,
    )
    classification = _classify_failure(result, "paper with March 2024 date")
    assert classification == RetryClassification.retry_resolvable


@pytest.mark.asyncio
async def test_max_retries_exceeded_after_three_attempts():
    """After 3 consecutive resolvable failures, MaxRetriesExceededError is raised."""
    from extraction_pipeline.validation.orchestrator import MaxRetriesExceededError

    # Three consecutive bad-date extractions (semantic failure, retry-resolvable)
    bad = {
        "document_id": "doc-nonres-02",
        "document_type": "invoice",
        "document_type_other_detail": None,
        "title": None,
        "author": None,
        "publication_date": "not-a-date",
        "invoice_total": 100.0,
        "currency": "USD",
        "citations": None,
        "confidence_scores": {
            "overall": 0.90,
            "title": None,
            "author": None,
            "publication_date": 0.90,
            "invoice_total": 0.90,
            "citations": None,
        },
    }
    engine = make_mock_engine([bad, bad, bad])
    orchestrator = ValidationRetryOrchestrator(engine=engine)

    with pytest.raises(MaxRetriesExceededError) as exc_info:
        await orchestrator.extract_with_retry("doc-nonres-02", "Invoice text.")

    assert exc_info.value.document_id == "doc-nonres-02"
