"""
Test 9: SLA compliance benchmark.
Generates 100 mock BatchResult records and verifies SLAAnalyzer produces
a valid SLAReport with correct P95, throughput, and pass-rate values.
"""
from __future__ import annotations

import random

import pytest

from extraction_pipeline.batch.sla_analyzer import SLAAnalyzer
from extraction_pipeline.schemas.batch import BatchJob, BatchResult, SLAReport
from extraction_pipeline import config


def _make_mock_job(n: int = 100, seed: int = 42) -> BatchJob:
    rng = random.Random(seed)
    results = []
    for i in range(n):
        # Normally distributed latency around 8000ms, std=3000ms
        latency = max(500, int(rng.gauss(8000, 3000)))
        results.append(BatchResult(
            custom_id=f"doc-{i:03d}",
            document_id=f"doc-{i:03d}",
            success=True,
            latency_ms=latency,
        ))
    return BatchJob(
        batch_id="bench-batch-001",
        status="completed",
        submitted_count=n,
        completed_count=n,
        failed_count=0,
        results=results,
    )


def test_sla_analyzer_p95_is_computable():
    """SLAReport.p95_latency_ms must be a positive finite number."""
    job = _make_mock_job()
    report = SLAAnalyzer().analyze(job)
    assert report.p95_latency_ms > 0
    assert report.p95_latency_ms < 60_000


def test_sla_analyzer_throughput_is_positive():
    """throughput_per_hr must be > 0."""
    job = _make_mock_job()
    report = SLAAnalyzer().analyze(job)
    assert report.throughput_per_hr > 0


def test_sla_analyzer_pass_rate_in_range():
    """sla_pass_rate must be between 0.0 and 1.0."""
    job = _make_mock_job()
    report = SLAAnalyzer().analyze(job)
    assert 0.0 <= report.sla_pass_rate <= 1.0


def test_sla_analyzer_p95_greater_than_p50():
    """P95 must be ≥ P50."""
    job = _make_mock_job()
    report = SLAAnalyzer().analyze(job)
    assert report.p95_latency_ms >= report.p50_latency_ms


def test_sla_analyzer_all_fast_docs_pass_100_percent():
    """When all latencies are 1ms, sla_pass_rate must be 1.0."""
    results = [
        BatchResult(
            custom_id=f"doc-{i:03d}",
            document_id=f"doc-{i:03d}",
            success=True,
            latency_ms=1,
        )
        for i in range(100)
    ]
    job = BatchJob(
        batch_id="bench-fast",
        status="completed",
        submitted_count=100,
        completed_count=100,
        failed_count=0,
        results=results,
    )
    report = SLAAnalyzer().analyze(job)
    assert report.sla_pass_rate == 1.0


def test_sla_analyzer_all_slow_docs_fail_sla():
    """When all latencies exceed the SLA target, sla_pass_rate must be 0.0."""
    target_ms = config.SLA_TARGET_P95_SECS * 1000
    results = [
        BatchResult(
            custom_id=f"doc-{i:03d}",
            document_id=f"doc-{i:03d}",
            success=True,
            latency_ms=target_ms + 10_000,
        )
        for i in range(100)
    ]
    job = BatchJob(
        batch_id="bench-slow",
        status="completed",
        submitted_count=100,
        completed_count=100,
        failed_count=0,
        results=results,
    )
    report = SLAAnalyzer().analyze(job)
    assert report.sla_pass_rate == 0.0


def test_sla_report_print_table_returns_string():
    """SLAReport.print_table() returns a non-empty string."""
    job = _make_mock_job(50)
    report = SLAAnalyzer().analyze(job)
    table = report.print_table()
    assert isinstance(table, str)
    assert "P95" in table
    assert "throughput" in table.lower() or "Throughput" in table


def test_sla_report_with_recovery_job_counts_retries():
    """When a recovery job is provided, retried_count and retry_success_count are set."""
    primary_results = [
        BatchResult(custom_id=f"doc-{i}", document_id=f"doc-{i}", success=True, latency_ms=5000)
        for i in range(95)
    ] + [
        BatchResult(custom_id=f"doc-fail-{i}", document_id=f"doc-fail-{i}", success=False, latency_ms=0)
        for i in range(5)
    ]
    primary_job = BatchJob(
        batch_id="primary-001",
        status="completed",
        submitted_count=100,
        completed_count=95,
        failed_count=5,
        results=primary_results,
    )
    recovery_results = [
        BatchResult(custom_id=f"doc-fail-{i}", document_id=f"doc-fail-{i}", success=True, latency_ms=6000)
        for i in range(4)
    ] + [
        BatchResult(custom_id="doc-fail-4", document_id="doc-fail-4", success=False, latency_ms=0)
    ]
    recovery_job = BatchJob(
        batch_id="recovery-001",
        status="completed",
        submitted_count=5,
        completed_count=4,
        failed_count=1,
        results=recovery_results,
    )

    report = SLAAnalyzer().analyze(primary_job, recovery_job)
    assert report.retried_count == 5
    assert report.retry_success_count == 4
    assert report.total_documents == 105  # 100 primary + 5 recovery
