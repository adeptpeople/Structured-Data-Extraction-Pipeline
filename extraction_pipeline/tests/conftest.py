"""Shared fixtures and mock factories for all tests. No API key required."""
from __future__ import annotations

import json
from typing import Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from extraction_pipeline.schemas.extraction import (
    ConfidenceScores,
    DocumentType,
    ExtractedDocument,
)


# ── Sample document texts ────────────────────────────────────────────────────

INVOICE_TEXT = """\
INVOICE
Invoice #: INV-2024-0892
Date: 2024-07-15
Bill To: Acme Corporation
Services rendered: Cloud infrastructure consulting
Invoice Total: USD 4,592.50
Payment due within 30 days.
"""

RESEARCH_PAPER_TEXT = """\
Quantum Entanglement in Distributed Systems
Authors: Dr. Sarah Chen, Prof. Michael Torres
Published: March 15, 2024 in Journal of Quantum Computing
Abstract: This paper examines the role of quantum entanglement.
References:
[1] Einstein A et al. Phys Rev. 1935;47:777.
"""

SPARSE_TEXT = """\
This document contains minimal metadata.
The content discusses various topics without identifying the author or date.
"""

CONTRACT_TEXT = """\
SERVICE AGREEMENT
This Service Agreement is entered into as of 14 February 2025
by and between TechCorp Inc. and ClientCo Ltd.
WHEREAS, Provider desires to provide software development services.
"""

AMBIGUOUS_TEXT = """\
GOVERNMENT FILING — FORM 10-K
Filing Entity: Acme Corp
Fiscal Year End: December 31, 2023
This filing does not include invoice information.
"""

LONG_TEXT = ("A" * 100 + "\n") * 300  # ~30k chars, will trigger chunking


# ── Raw extraction dict factories ────────────────────────────────────────────

def make_raw_extraction(
    document_id: str = "doc-001",
    document_type: str = "invoice",
    document_type_other_detail=None,
    title=None,
    author=None,
    publication_date="2024-07-15",
    invoice_total=4592.50,
    currency="USD",
    citations=None,
    overall_confidence=0.95,
    title_conf=None,
    author_conf=None,
    date_conf=0.99,
    total_conf=0.97,
    citations_conf=None,
) -> dict:
    return {
        "document_id": document_id,
        "document_type": document_type,
        "document_type_other_detail": document_type_other_detail,
        "title": title,
        "author": author,
        "publication_date": publication_date,
        "invoice_total": invoice_total,
        "currency": currency,
        "citations": citations,
        "confidence_scores": {
            "overall": overall_confidence,
            "title": title_conf,
            "author": author_conf,
            "publication_date": date_conf,
            "invoice_total": total_conf,
            "citations": citations_conf,
        },
    }


def make_extracted_document(**kwargs) -> ExtractedDocument:
    raw = make_raw_extraction(**kwargs)
    return ExtractedDocument.model_validate(raw)


# ── Mock ExtractionEngine factories ──────────────────────────────────────────

def make_mock_engine(return_values: list[dict]) -> MagicMock:
    """
    Returns a mock ExtractionEngine whose extract() returns each dict in sequence.
    If the sequence is exhausted, raises StopAsyncIteration.
    """
    engine = MagicMock()
    call_count = {"n": 0}

    async def mock_extract(*args, **kwargs):
        idx = call_count["n"]
        call_count["n"] += 1
        if idx >= len(return_values):
            raise RuntimeError("mock engine exhausted")
        result = return_values[idx]
        if isinstance(result, Exception):
            raise result
        return result

    engine.extract = mock_extract
    return engine


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def invoice_raw() -> dict:
    return make_raw_extraction()


@pytest.fixture
def research_paper_raw() -> dict:
    return make_raw_extraction(
        document_id="doc-002",
        document_type="research_paper",
        title="Quantum Entanglement in Distributed Systems",
        author="Dr. Sarah Chen",
        publication_date="2024-03-15",
        invoice_total=None,
        currency=None,
        citations=["Einstein A et al. Phys Rev. 1935;47:777."],
        overall_confidence=0.95,
        title_conf=0.99,
        author_conf=0.99,
        date_conf=0.97,
        total_conf=None,
        citations_conf=0.95,
    )
