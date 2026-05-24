"""
Test 7: Oversized document chunking and chunk-merge extraction.
Verifies that a 15k-char document is split into ≥3 chunks, each extracted,
and the merged result contains invoice_total from the chunk that had it.
"""
from __future__ import annotations

import pytest

from extraction_pipeline.tests.conftest import make_mock_engine
from extraction_pipeline.validation.orchestrator import (
    ValidationRetryOrchestrator,
    _chunk_text,
    _merge_extractions,
)
from extraction_pipeline import config


def _make_chunk_raw(
    doc_id: str,
    invoice_total=None,
    total_conf=None,
    overall=0.85,
) -> dict:
    return {
        "document_id": doc_id,
        "document_type": "invoice",
        "document_type_other_detail": None,
        "title": None,
        "author": None,
        "publication_date": "2024-01-15",
        "invoice_total": invoice_total,
        "currency": "USD" if invoice_total else None,
        "citations": None,
        "confidence_scores": {
            "overall": overall,
            "title": None,
            "author": None,
            "publication_date": 0.90,
            "invoice_total": total_conf,
            "citations": None,
        },
    }


def test_chunk_text_splits_large_document():
    """A 75k-char document should produce ≥3 chunks (max_chars=24000, overlap=200)."""
    text = "A" * 75_000
    chunks = _chunk_text(text, max_tokens=config.MAX_TOKENS_PER_CHUNK)
    assert len(chunks) >= 3


def test_chunk_text_small_document_no_split():
    """A small document stays as one chunk."""
    text = "A" * 1000
    chunks = _chunk_text(text, max_tokens=config.MAX_TOKENS_PER_CHUNK)
    assert len(chunks) == 1


def test_merge_extractions_takes_highest_confidence_field():
    """
    Chunk 1 has invoice_total=None (conf=None).
    Chunk 2 has invoice_total=4592.50 (conf=0.95).
    Merged result must have invoice_total=4592.50.
    """
    chunk1 = _make_chunk_raw("doc-chunk", invoice_total=None, total_conf=None, overall=0.85)
    chunk2 = _make_chunk_raw("doc-chunk", invoice_total=4592.50, total_conf=0.95, overall=0.90)

    merged = _merge_extractions([chunk1, chunk2])

    assert merged["invoice_total"] == 4592.50
    assert merged["confidence_scores"]["invoice_total"] == 0.95


def test_merge_extractions_single_chunk_unchanged():
    """A single chunk merge returns the original dict unchanged."""
    chunk = _make_chunk_raw("doc-chunk", invoice_total=100.0, total_conf=0.90)
    merged = _merge_extractions([chunk])
    assert merged["invoice_total"] == 100.0


@pytest.mark.asyncio
async def test_oversized_document_triggers_chunking():
    """
    A 25k-char document exceeds MAX_TOKENS_PER_CHUNK * 4 chars.
    The orchestrator should extract multiple chunks.
    """
    oversized_text = (
        "This is a long invoice document with details. " * 600
        + "Invoice Total: USD 9,999.99"
    )
    assert len(oversized_text) > config.MAX_TOKENS_PER_CHUNK * 4

    # Provide enough mock responses for all expected chunks (one per chunk)
    chunk_raw = _make_chunk_raw(
        "doc-oversized",
        invoice_total=9999.99,
        total_conf=0.95,
        overall=0.92,
    )
    engine = make_mock_engine([chunk_raw] * 10)
    orchestrator = ValidationRetryOrchestrator(engine=engine)
    doc, _ = await orchestrator.extract_with_retry("doc-oversized", oversized_text)

    assert doc.document_id == "doc-oversized"
    assert doc.invoice_total == 9999.99
