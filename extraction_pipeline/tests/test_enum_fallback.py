"""
Test 2: Enum + other_detail fallback.
Verifies correct handling of document_type='other' and detail field requirements.
"""
from __future__ import annotations

import pytest
import pydantic

from extraction_pipeline.schemas.extraction import DocumentType, ExtractedDocument
from extraction_pipeline.tests.conftest import make_mock_engine
from extraction_pipeline.validation.orchestrator import ValidationRetryOrchestrator


def _make_other_raw(doc_id: str, detail: str) -> dict:
    return {
        "document_id": doc_id,
        "document_type": "other",
        "document_type_other_detail": detail,
        "title": None,
        "author": None,
        "publication_date": "2023-05-01",
        "invoice_total": None,
        "currency": None,
        "citations": None,
        "confidence_scores": {
            "overall": 0.85,
            "title": None,
            "author": None,
            "publication_date": 0.85,
            "invoice_total": None,
            "citations": None,
        },
    }


@pytest.mark.asyncio
async def test_other_type_with_detail_passes():
    """document_type='other' with a non-null other_detail is valid."""
    raw = _make_other_raw("doc-enum-01", "government filing")
    engine = make_mock_engine([raw])
    orchestrator = ValidationRetryOrchestrator(engine=engine)
    doc, _ = await orchestrator.extract_with_retry("doc-enum-01", "Government filing content.")

    assert doc.document_type == DocumentType.other
    assert doc.document_type_other_detail == "government filing"


@pytest.mark.asyncio
async def test_other_type_without_detail_fails_validation():
    """document_type='other' with null other_detail must fail Pydantic validation."""
    raw = _make_other_raw("doc-enum-02", None)  # type: ignore[arg-type]
    raw["document_type_other_detail"] = None
    with pytest.raises(pydantic.ValidationError):
        ExtractedDocument.model_validate(raw)


@pytest.mark.asyncio
async def test_non_other_type_with_detail_fails_semantic():
    """document_type='invoice' with a non-null other_detail must fail semantic validation."""
    from extraction_pipeline.validation.semantic_rules import validate_semantic
    from extraction_pipeline.schemas.extraction import ExtractedDocument

    raw = {
        "document_id": "doc-enum-03",
        "document_type": "invoice",
        "document_type_other_detail": "should not be here",
        "title": None,
        "author": None,
        "publication_date": "2024-01-01",
        "invoice_total": 500.0,
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
    doc = ExtractedDocument.model_validate(raw)
    result = validate_semantic(doc)
    assert not result.passed
    assert any("other_detail" in m or "other" in m for m in result.error_messages)


def test_all_valid_enum_values_accepted():
    """All six enum values must be accepted."""
    valid_types = ["invoice", "contract", "research_paper", "resume", "medical_record"]
    for doc_type in valid_types:
        raw = {
            "document_id": f"doc-{doc_type}",
            "document_type": doc_type,
            "document_type_other_detail": None,
            "title": None,
            "author": None,
            "publication_date": None,
            "invoice_total": None,
            "currency": None,
            "citations": None,
            "confidence_scores": {
                "overall": 0.90,
                "title": None,
                "author": None,
                "publication_date": None,
                "invoice_total": None,
                "citations": None,
            },
        }
        doc = ExtractedDocument.model_validate(raw)
        assert doc.document_type.value == doc_type


def test_invalid_enum_value_rejected():
    """An unknown document_type must raise a Pydantic validation error."""
    raw = {
        "document_id": "doc-bad-enum",
        "document_type": "newspaper",
        "document_type_other_detail": None,
        "title": None,
        "author": None,
        "publication_date": None,
        "invoice_total": None,
        "currency": None,
        "citations": None,
        "confidence_scores": {"overall": 0.90},
    }
    with pytest.raises(pydantic.ValidationError):
        ExtractedDocument.model_validate(raw)
