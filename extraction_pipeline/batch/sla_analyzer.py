"""SLA analysis using pandas. Computes P50/P95/P99, throughput/hr, and pass rate."""
from __future__ import annotations

import pandas as pd

from extraction_pipeline import config
from extraction_pipeline.schemas.batch import BatchJob, BatchResult, SLAReport


class SLAAnalyzer:
    def analyze(self, job: BatchJob, recovery_job: BatchJob | None = None) -> SLAReport:
        all_results: list[BatchResult] = list(job.results)
        retry_count = 0
        retry_success = 0

        if recovery_job:
            all_results.extend(recovery_job.results)
            retry_count = len(recovery_job.results)
            retry_success = sum(1 for r in recovery_job.results if r.success)

        successful = [r for r in all_results if r.success and r.latency_ms > 0]
        if not successful:
            return SLAReport(
                batch_id=job.batch_id,
                total_documents=len(all_results),
                p50_latency_ms=0.0,
                p95_latency_ms=0.0,
                p99_latency_ms=0.0,
                avg_latency_ms=0.0,
                throughput_per_hr=0.0,
                sla_pass_rate=0.0,
                sla_target_p95_ms=config.SLA_TARGET_P95_SECS * 1000,
                submitted_count=job.submitted_count,
                completed_count=job.completed_count,
                failed_count=job.failed_count,
                retried_count=retry_count,
                retry_success_count=retry_success,
            )

        df = pd.DataFrame([{"latency_ms": r.latency_ms} for r in successful])
        target_ms = config.SLA_TARGET_P95_SECS * 1000

        p50 = df["latency_ms"].quantile(0.50)
        p95 = df["latency_ms"].quantile(0.95)
        p99 = df["latency_ms"].quantile(0.99)
        avg = df["latency_ms"].mean()
        total_ms = df["latency_ms"].sum()
        throughput = (len(successful) / total_ms * 3_600_000) if total_ms > 0 else 0.0
        pass_rate = float((df["latency_ms"] <= target_ms).mean())

        return SLAReport(
            batch_id=job.batch_id,
            total_documents=len(all_results),
            p50_latency_ms=float(p50),
            p95_latency_ms=float(p95),
            p99_latency_ms=float(p99),
            avg_latency_ms=float(avg),
            throughput_per_hr=float(throughput),
            sla_pass_rate=pass_rate,
            sla_target_p95_ms=target_ms,
            submitted_count=job.submitted_count,
            completed_count=job.completed_count,
            failed_count=job.failed_count,
            retried_count=retry_count,
            retry_success_count=retry_success,
        )
