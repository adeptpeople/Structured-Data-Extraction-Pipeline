from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class BatchRequest(BaseModel):
    custom_id: str
    document_text: str
    document_id: str


class BatchResult(BaseModel):
    custom_id: str
    document_id: str
    success: bool
    extracted_data: Optional[dict] = None
    error_message: Optional[str] = None
    latency_ms: int = 0
    retry_count: int = 0
    error_type: Optional[str] = None


class BatchJob(BaseModel):
    batch_id: str
    status: str
    submitted_count: int
    completed_count: int = 0
    failed_count: int = 0
    results: list[BatchResult] = []
    failed_custom_ids: list[str] = []
    recovery_batch_id: Optional[str] = None


class SLAReport(BaseModel):
    batch_id: str
    total_documents: int
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    avg_latency_ms: float
    throughput_per_hr: float
    sla_pass_rate: float
    sla_target_p95_ms: float
    submitted_count: int = 0
    completed_count: int = 0
    failed_count: int = 0
    retried_count: int = 0
    retry_success_count: int = 0

    def print_table(self) -> str:
        lines = [
            f"\n{'='*55}",
            f"  SLA BENCHMARK REPORT — Batch {self.batch_id[:8]}",
            f"{'='*55}",
            f"  Total documents    : {self.total_documents}",
            f"  Submitted          : {self.submitted_count}",
            f"  Completed          : {self.completed_count}",
            f"  Failed             : {self.failed_count}",
            f"  Retried            : {self.retried_count}",
            f"  Retry successes    : {self.retry_success_count}",
            f"{'─'*55}",
            f"  P50 latency        : {self.p50_latency_ms:.0f} ms",
            f"  P95 latency        : {self.p95_latency_ms:.0f} ms",
            f"  P99 latency        : {self.p99_latency_ms:.0f} ms",
            f"  Avg latency        : {self.avg_latency_ms:.0f} ms",
            f"  SLA target (P95)   : {self.sla_target_p95_ms:.0f} ms",
            f"  SLA pass rate      : {self.sla_pass_rate*100:.1f}%",
            f"  Throughput/hr      : {self.throughput_per_hr:.1f} docs",
            f"{'='*55}\n",
        ]
        return "\n".join(lines)
