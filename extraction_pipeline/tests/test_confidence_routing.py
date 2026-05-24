"""
Test 8: Confidence-based human review routing.
Verifies that documents route to the correct tier based on overall confidence,
and that field-level escalation works for critical fields.
"""
from __future__ import annotations

import pytest

from extraction_pipeline.routing.confidence_router import ConfidenceRouter
from extraction_pipeline.schemas.extraction import (
    ConfidenceScores,
    DocumentType,
    ExtractedDocument,
)
from extraction_pipeline.schemas.review import ReviewRoute


def _make_doc(
    overall: float,
    invoice_total_conf: float | None = None,
    doc_type: str = "research_paper",
    invoice_total: float | None = None,
) -> ExtractedDocument:
    return ExtractedDocument(
        document_id="doc-route-test",
        document_type=DocumentType(doc_type),
        document_type_other_detail=None,
        title="Test Document",
        author="Jane Smith",
        publication_date="2024-01-01",
        invoice_total=invoice_total,
        currency="USD" if invoice_total else None,
        citations=None,
        confidence_scores=ConfidenceScores(
            overall=overall,
            title=0.99,
            author=0.99,
            publication_date=overall,
            invoice_total=invoice_total_conf,
            citations=None,
        ),
    )


def test_high_confidence_auto_approves():
    """overall=0.95 → auto_approve."""
    router = ConfidenceRouter()
    doc = _make_doc(overall=0.95)
    item = router.route(doc)
    assert item.route == ReviewRoute.auto_approve
    assert item.flagged_fields == []


def test_medium_confidence_qa_sample():
    """overall=0.75 → qa_sample."""
    router = ConfidenceRouter()
    doc = _make_doc(overall=0.75)
    item = router.route(doc)
    assert item.route == ReviewRoute.qa_sample


def test_low_confidence_human_review():
    """overall=0.55 → human_review."""
    router = ConfidenceRouter()
    doc = _make_doc(overall=0.55)
    item = router.route(doc)
    assert item.route == ReviewRoute.human_review


def test_boundary_at_auto_approve_threshold():
    """overall=0.90 exactly → auto_approve."""
    router = ConfidenceRouter()
    doc = _make_doc(overall=0.90)
    item = router.route(doc)
    assert item.route == ReviewRoute.auto_approve


def test_boundary_just_below_auto_approve():
    """overall=0.8999 → qa_sample."""
    router = ConfidenceRouter()
    doc = _make_doc(overall=0.8999)
    item = router.route(doc)
    assert item.route == ReviewRoute.qa_sample


def test_field_level_escalation_triggers_qa_sample():
    """
    overall=0.92 (would auto-approve) but invoice_total confidence=0.60
    → escalated to qa_sample with flagged_fields=['invoice_total'].
    """
    router = ConfidenceRouter()
    doc = _make_doc(
        overall=0.92,
        invoice_total_conf=0.60,
        doc_type="invoice",
        invoice_total=4592.50,
    )
    item = router.route(doc)
    assert item.route == ReviewRoute.qa_sample
    assert "invoice_total" in item.flagged_fields


def test_field_level_escalation_does_not_trigger_for_high_field_conf():
    """
    overall=0.92 and invoice_total confidence=0.95 → no escalation → auto_approve.
    """
    router = ConfidenceRouter()
    doc = _make_doc(
        overall=0.92,
        invoice_total_conf=0.95,
        doc_type="invoice",
        invoice_total=100.0,
    )
    item = router.route(doc)
    assert item.route == ReviewRoute.auto_approve
    assert item.flagged_fields == []


def test_route_field_level_returns_per_field_routes():
    """route_field_level returns a dict keyed by field name with correct routes."""
    router = ConfidenceRouter()
    doc = _make_doc(overall=0.80, invoice_total_conf=0.55, invoice_total=500.0)
    field_routes = router.route_field_level(doc)

    assert "title" in field_routes
    assert field_routes["title"] == ReviewRoute.auto_approve  # 0.99
    assert field_routes["invoice_total"] == ReviewRoute.human_review  # 0.55


def test_review_queue_item_carries_extracted_data():
    """ReviewQueueItem.extracted_data is a non-empty dict."""
    router = ConfidenceRouter()
    doc = _make_doc(overall=0.50)
    item = router.route(doc)
    assert isinstance(item.extracted_data, dict)
    assert item.extracted_data["document_id"] == "doc-route-test"
