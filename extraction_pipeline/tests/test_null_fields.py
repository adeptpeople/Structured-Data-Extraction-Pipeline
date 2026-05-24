"""
Test 1: Missing fields return null — no hallucination.
Verifies that absent document data produces None values, not fabricated strings.
"""
from __future__ import annotations

import pytest

from extraction_pipeline.schemas.extraction import DocumentType, ExtractedDocument
from extraction_pipeline.tests.conftest import make_extracted_document, make_mock_engine
from extraction_pipeline.validation.orchestrator import ValidationRetryOrchestrator


@pytest.mark.asyncio
async def test_missing_title_returns_none():
    """A document without a title must produce title=None."""
    raw = {
        "document_id": "doc-null-01",
        "document_type": "invoice",
        "document_type_other_detail": None,
        "title": None,
        "author": None,
        "publication_date": "2024-01-15",
        "invoice_total": 1000.0,
        "currency": "USD",
        "citations": None,
        "confidence_scores": {
            "overall": 0.93,
            "title": None,
            "author": None,
            "publication_date": 0.99,
            "invoice_total": 0.93,
            "citations": None,
        },
    }
    engine = make_mock_engine([raw])
    orchestrator = ValidationRetryOrchestrator(engine=engine)
    doc, metrics = await orchestrator.extract_with_retry("doc-null-01", "Invoice text...")

    assert doc.title is None
    assert doc.author is None
    assert doc.citations is None
    assert metrics.final_status == "success"


@pytest.mark.asyncio
async def test_missing_date_returns_none():
    """A document without a date must produce publication_date=None."""
    raw = {
        "document_id": "doc-null-02",
        "document_type": "resume",
        "document_type_other_detail": None,
        "title": None,
        "author": "Jane Doe",
        "publication_date": None,
        "invoice_total": None,
        "currency": None,
        "citations": None,
        "confidence_scores": {
            "overall": 0.92,
            "title": None,
            "author": 0.92,
            "publication_date": None,
            "invoice_total": None,
            "citations": None,
        },
    }
    engine = make_mock_engine([raw])
    orchestrator = ValidationRetryOrchestrator(engine=engine)
    doc, _ = await orchestrator.extract_with_retry("doc-null-02", "Resume text...")

    assert doc.publication_date is None
    assert doc.invoice_total is None
    assert doc.currency is None


@pytest.mark.asyncio
async def test_all_optional_fields_null_is_valid():
    """All optional fields can simultaneously be null for a minimal document."""
    raw = {
        "document_id": "doc-null-03",
        "document_type": "other",
        "document_type_other_detail": "blank page scan",
        "title": None,
        "author": None,
        "publication_date": None,
        "invoice_total": None,
        "currency": None,
        "citations": None,
        "confidence_scores": {
            "overall": 0.30,
            "title": None,
            "author": None,
            "publication_date": None,
            "invoice_total": None,
            "citations": None,
        },
    }
    engine = make_mock_engine([raw])
    orchestrator = ValidationRetryOrchestrator(engine=engine)
    doc, _ = await orchestrator.extract_with_retry("doc-null-03", "Blank page.")

    assert doc.document_type == "other"
    assert doc.title is None
    assert doc.author is None
    assert doc.publication_date is None
    assert doc.invoice_total is None


def test_pydantic_model_null_fields_validate():
    """Pydantic model accepts null for all optional fields."""
    doc = make_extracted_document(
        document_id="doc-null-04",
        title=None,
        author=None,
        publication_date=None,
        invoice_total=None,
        currency=None,
        citations=None,
        title_conf=None,
        author_conf=None,
        date_conf=None,
        total_conf=None,
        citations_conf=None,
        overall_confidence=0.95,
    )
    assert doc.title is None
    assert doc.author is None
    assert doc.publication_date is None
    assert doc.citations is None
    assert doc.invoice_total is None
