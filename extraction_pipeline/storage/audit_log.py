"""AuditLogger — writes extraction run records to PostgreSQL."""
from __future__ import annotations

import json
from typing import Optional

from extraction_pipeline.schemas.extraction import ExtractedDocument
from extraction_pipeline.schemas.validation import ExtractionMetrics


class AuditLogger:
    def __init__(self, pool=None) -> None:
        self._pool = pool

    async def _get_pool(self):
        if self._pool is None:
            from extraction_pipeline.storage.db import get_pool
            self._pool = await get_pool()
        return self._pool

    async def log_extraction(
        self,
        doc: ExtractedDocument,
        metrics: ExtractionMetrics,
        route: str,
        batch_id: Optional[str] = None,
    ) -> Optional[int]:
        try:
            pool = await self._get_pool()
            row = await pool.fetchrow(
                """
                INSERT INTO extraction_runs
                    (document_id, document_type, attempt_number, retry_count,
                     success, route, overall_confidence, latency_ms, batch_id, custom_id)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                RETURNING id
                """,
                doc.document_id,
                doc.document_type.value,
                metrics.attempt_count,
                metrics.retry_count,
                metrics.final_status == "success",
                route,
                doc.confidence_scores.overall,
                metrics.processing_time_ms,
                batch_id,
                doc.document_id,
            )
            return row["id"] if row else None
        except Exception:
            return None  # Degrade gracefully when DB unavailable

    async def log_field_scores(
        self,
        doc: ExtractedDocument,
        extraction_run_id: Optional[int] = None,
    ) -> None:
        try:
            pool = await self._get_pool()
            scores = doc.confidence_scores.model_dump()
            rows = []
            for field, conf in scores.items():
                if field == "overall":
                    continue
                field_val = getattr(doc, field, None)
                rows.append((
                    extraction_run_id,
                    doc.document_id,
                    field,
                    conf,
                    field_val is None,
                ))
            await pool.executemany(
                """
                INSERT INTO field_confidence_scores
                    (extraction_run_id, document_id, field_name, confidence, is_null)
                VALUES ($1, $2, $3, $4, $5)
                """,
                rows,
            )
        except Exception:
            pass
