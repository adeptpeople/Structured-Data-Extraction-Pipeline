"""asyncpg connection pool manager."""
from __future__ import annotations

from typing import Optional

try:
    import asyncpg
    _ASYNCPG_AVAILABLE = True
except ImportError:
    _ASYNCPG_AVAILABLE = False

from extraction_pipeline import config

_pool: Optional[object] = None


async def get_pool():
    global _pool
    if not _ASYNCPG_AVAILABLE:
        raise RuntimeError("asyncpg not installed. Run: pip install asyncpg")
    if _pool is None:
        _pool = await asyncpg.create_pool(config.DATABASE_URL)
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS extraction_runs (
    id              BIGSERIAL PRIMARY KEY,
    document_id     TEXT NOT NULL,
    document_type   TEXT,
    attempt_number  SMALLINT NOT NULL DEFAULT 1,
    retry_count     SMALLINT NOT NULL DEFAULT 0,
    success         BOOLEAN NOT NULL,
    failure_mode    TEXT,
    retry_class     TEXT,
    route           TEXT NOT NULL,
    overall_confidence REAL,
    latency_ms      INTEGER,
    batch_id        TEXT,
    custom_id       TEXT,
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS field_confidence_scores (
    id                BIGSERIAL PRIMARY KEY,
    extraction_run_id BIGINT,
    document_id       TEXT NOT NULL,
    field_name        TEXT NOT NULL,
    confidence        REAL,
    is_null           BOOLEAN NOT NULL DEFAULT FALSE,
    created_at        TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS review_decisions (
    id                BIGSERIAL PRIMARY KEY,
    document_id       TEXT NOT NULL,
    extraction_run_id BIGINT,
    reviewer_id       TEXT,
    decision          TEXT NOT NULL,
    corrections       JSONB,
    flagged_fields    TEXT[],
    reviewed_at       TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS batch_jobs (
    id              BIGSERIAL PRIMARY KEY,
    batch_id        TEXT UNIQUE NOT NULL,
    status          TEXT NOT NULL,
    document_count  INTEGER NOT NULL,
    success_count   INTEGER DEFAULT 0,
    failure_count   INTEGER DEFAULT 0,
    recovery_batch_id TEXT,
    submitted_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at    TIMESTAMPTZ,
    p95_latency_ms  REAL,
    throughput_per_hr REAL
);

CREATE TABLE IF NOT EXISTS sla_metrics (
    id                  BIGSERIAL PRIMARY KEY,
    batch_id            TEXT,
    measurement_window  TEXT,
    p50_latency_ms      REAL,
    p95_latency_ms      REAL,
    p99_latency_ms      REAL,
    throughput_per_hr   REAL,
    sla_pass_rate       REAL,
    total_docs          INTEGER,
    created_at          TIMESTAMPTZ DEFAULT now()
);
"""
