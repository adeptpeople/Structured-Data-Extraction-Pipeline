"""
Test 6: Batch failure recovery by custom_id.
Uses a mock batch client to simulate 2 successes + 1 failure, then verifies
that recover_failed() re-submits only the failed custom_id.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from extraction_pipeline.batch.batch_processor import BatchProcessor
from extraction_pipeline.schemas.batch import BatchRequest


def _make_batch_requests(n: int) -> list[BatchRequest]:
    return [
        BatchRequest(
            custom_id=f"doc-{i:03d}",
            document_id=f"doc-{i:03d}",
            document_text=f"Invoice #{i} Total: USD {100 * i}.00",
        )
        for i in range(1, n + 1)
    ]


def _make_mock_client(
    primary_output: list[dict],
    recovery_output: list[dict],
) -> MagicMock:
    """Build a mock openai.AsyncOpenAI client that returns controlled batch results."""
    client = MagicMock()
    call_count = {"n": 0}

    async def mock_create(**kwargs):
        batch = MagicMock()
        batch.id = f"batch-{call_count['n']:03d}"
        call_count["n"] += 1
        return batch

    async def mock_retrieve(batch_id: str):
        batch = MagicMock()
        batch.id = batch_id
        batch.status = "completed"
        batch.output_file_id = f"file-{batch_id}"
        return batch

    async def mock_file_content(file_id: str):
        is_recovery = call_count["n"] > 1
        output = recovery_output if is_recovery else primary_output
        content = MagicMock()
        content.text = "\n".join(json.dumps(row) for row in output)
        return content

    client.batches.create = AsyncMock(side_effect=mock_create)
    client.batches.retrieve = AsyncMock(side_effect=mock_retrieve)
    client.files.content = AsyncMock(side_effect=mock_file_content)
    return client


def _make_success_entry(custom_id: str, invoice_total: float) -> dict:
    args = {
        "document_id": custom_id,
        "document_type": "invoice",
        "document_type_other_detail": None,
        "title": None,
        "author": None,
        "publication_date": "2024-01-15",
        "invoice_total": invoice_total,
        "currency": "USD",
        "citations": None,
        "confidence_scores": {
            "overall": 0.95,
            "title": None,
            "author": None,
            "publication_date": 0.99,
            "invoice_total": 0.95,
            "citations": None,
        },
    }
    return {
        "custom_id": custom_id,
        "response": {
            "body": {
                "choices": [{
                    "message": {
                        "tool_calls": [{
                            "function": {"arguments": json.dumps(args)}
                        }]
                    }
                }]
            }
        }
    }


def _make_failure_entry(custom_id: str, code: str = "server_error") -> dict:
    return {
        "custom_id": custom_id,
        "error": {"code": code, "message": "Internal server error"},
    }


@pytest.mark.asyncio
async def test_batch_identifies_failed_custom_ids():
    """Primary batch: 2 success + 1 failure. Verify failed_custom_ids = ['doc-003']."""
    primary_output = [
        _make_success_entry("doc-001", 100.0),
        _make_success_entry("doc-002", 200.0),
        _make_failure_entry("doc-003"),
    ]
    recovery_output = [
        _make_success_entry("doc-003", 300.0),
    ]

    mock_client = _make_mock_client(primary_output, recovery_output)
    processor = BatchProcessor(client=mock_client)

    docs = _make_batch_requests(3)
    batch_id = await processor.submit_batch(docs)
    job = await processor.poll_until_complete(batch_id)

    assert job.completed_count == 2
    assert job.failed_count == 1
    assert "doc-003" in job.failed_custom_ids


@pytest.mark.asyncio
async def test_recover_failed_resubmits_only_failed_docs():
    """recover_failed() calls submit_batch only with the failed document."""
    submitted_custom_ids: list[str] = []

    async def capture_submit(docs: list[BatchRequest]) -> str:
        submitted_custom_ids.extend(d.custom_id for d in docs)
        return "batch-recovery-001"

    processor = BatchProcessor(client=MagicMock())
    processor.submit_batch = capture_submit  # type: ignore[method-assign]

    docs = _make_batch_requests(3)
    doc_map = {d.custom_id: d for d in docs}

    recovery_id = await processor.recover_failed(["doc-003"], doc_map)

    assert recovery_id == "batch-recovery-001"
    assert submitted_custom_ids == ["doc-003"]


@pytest.mark.asyncio
async def test_recover_failed_with_no_failures_returns_none():
    """If there are no failed custom_ids, recover_failed returns None."""
    processor = BatchProcessor(client=MagicMock())
    result = await processor.recover_failed([], {})
    assert result is None
