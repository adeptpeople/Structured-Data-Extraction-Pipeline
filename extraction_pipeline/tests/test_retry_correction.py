"""
Test 3: Validation retry loop corrects bad extraction output.
Simulates a model returning an invalid date on attempt 1, then correcting it on attempt 2.
"""
from __future__ import annotations

import pytest

from extraction_pipeline.tests.conftest import make_mock_engine
from extraction_pipeline.validation.orchestrator import ValidationRetryOrchestrator


def _raw_with_bad_date(doc_id: str) -> dict:
    return {
        "document_id": doc_id,
        "document_type": "research_paper",
        "document_type_other_detail": None,
        "title": "Test Paper",
        "author": "Jane Smith",
        "publication_date": "March 2024",  # bad format — not ISO 8601
        "invoice_total": None,
        "currency": None,
        "citations": None,
        "confidence_scores": {
            "overall": 0.85,
            "title": 0.99,
            "author": 0.99,
            "publication_date": 0.85,
            "invoice_total": None,
            "citations": None,
        },
    }


def _raw_with_good_date(doc_id: str) -> dict:
    raw = _raw_with_bad_date(doc_id)
    raw["publication_date"] = "2024-03-01"  # corrected to ISO 8601
    return raw


@pytest.mark.asyncio
async def test_retry_corrects_bad_date_format():
    """
    Attempt 1 returns a non-ISO date → semantic validation fails.
    Attempt 2 returns a valid ISO date → validation passes.
    Assert attempt_count == 2 and final date is correct.
    """
    engine = make_mock_engine([
        _raw_with_bad_date("doc-retry-01"),
        _raw_with_good_date("doc-retry-01"),
    ])
    orchestrator = ValidationRetryOrchestrator(engine=engine)
    doc, metrics = await orchestrator.extract_with_retry("doc-retry-01", "Sample research text.")

    assert doc.publication_date == "2024-03-01"
    assert metrics.attempt_count == 2
    assert metrics.retry_count == 1
    assert metrics.validation_failures == 1
    assert metrics.retry_successes == 1
    assert metrics.final_status == "success"


@pytest.mark.asyncio
async def test_retry_corrects_missing_currency():
    """
    invoice_total present but currency=None on attempt 1.
    Attempt 2 provides both → passes.
    """
    bad = {
        "document_id": "doc-retry-02",
        "document_type": "invoice",
        "document_type_other_detail": None,
        "title": None,
        "author": None,
        "publication_date": "2024-01-15",
        "invoice_total": 999.99,
        "currency": None,  # missing — semantic rule violation
        "citations": None,
        "confidence_scores": {
            "overall": 0.90,
            "title": None,
            "author": None,
            "publication_date": 0.99,
            "invoice_total": 0.90,
            "citations": None,
        },
    }
    good = dict(bad)
    good["currency"] = "EUR"

    engine = make_mock_engine([bad, good])
    orchestrator = ValidationRetryOrchestrator(engine=engine)
    doc, metrics = await orchestrator.extract_with_retry("doc-retry-02", "Invoice text...")

    assert doc.currency == "EUR"
    assert metrics.attempt_count == 2


@pytest.mark.asyncio
async def test_retry_succeeds_on_second_attempt_with_correct_output():
    """Ensure retry_successes counter increments only on corrected retries."""
    engine = make_mock_engine([
        _raw_with_bad_date("doc-retry-03"),
        _raw_with_good_date("doc-retry-03"),
    ])
    orchestrator = ValidationRetryOrchestrator(engine=engine)
    _, metrics = await orchestrator.extract_with_retry("doc-retry-03", "Text.")
    assert metrics.retry_successes == 1
