"""AnalyticsWriter — writes SLA metrics and field-level analytics to PostgreSQL."""
from __future__ import annotations

from extraction_pipeline.schemas.batch import SLAReport


class AnalyticsWriter:
    def __init__(self, pool=None) -> None:
        self._pool = pool

    async def _get_pool(self):
        if self._pool is None:
            from extraction_pipeline.storage.db import get_pool
            self._pool = await get_pool()
        return self._pool

    async def write_sla_report(self, report: SLAReport, window: str = "batch") -> None:
        try:
            pool = await self._get_pool()
            await pool.execute(
                """
                INSERT INTO sla_metrics
                    (batch_id, measurement_window, p50_latency_ms, p95_latency_ms,
                     p99_latency_ms, throughput_per_hr, sla_pass_rate, total_docs)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                report.batch_id,
                window,
                report.p50_latency_ms,
                report.p95_latency_ms,
                report.p99_latency_ms,
                report.throughput_per_hr,
                report.sla_pass_rate,
                report.total_documents,
            )
        except Exception:
            pass
