"""FastAPI application for human review queue management and analytics dashboard."""
from __future__ import annotations

from fastapi import FastAPI, HTTPException

from extraction_pipeline.review.models import (
    DashboardMetrics,
    ReviewQueueResponse,
    ReviewSubmitRequest,
    ReviewSubmitResponse,
)
from extraction_pipeline.review.queue import ReviewQueueManager
from extraction_pipeline.schemas.review import ReviewRoute

app = FastAPI(title="Extraction Review API", version="1.0.0")
_queue = ReviewQueueManager()


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/queue/human-review", response_model=list[ReviewQueueResponse])
async def get_human_review_queue(limit: int = 20) -> list[ReviewQueueResponse]:
    """Return lowest-confidence-first items requiring human review."""
    items = await _queue.dequeue_batch(ReviewRoute.human_review, count=limit)
    return [ReviewQueueResponse(**i.model_dump()) for i in items]


@app.get("/queue/qa-sample", response_model=list[ReviewQueueResponse])
async def get_qa_sample_queue(limit: int = 20) -> list[ReviewQueueResponse]:
    """Return items routed to QA sampling."""
    items = await _queue.dequeue_batch(ReviewRoute.qa_sample, count=limit)
    return [ReviewQueueResponse(**i.model_dump()) for i in items]


@app.post("/review/{document_id}", response_model=ReviewSubmitResponse)
async def submit_review(document_id: str, body: ReviewSubmitRequest) -> ReviewSubmitResponse:
    """Submit a review decision for a document."""
    # Remove from both queues (document may be in either)
    for route in [ReviewRoute.human_review, ReviewRoute.qa_sample]:
        await _queue.remove(route, document_id)
    return ReviewSubmitResponse(document_id=document_id, decision=body.decision)


@app.get("/analytics/dashboard", response_model=DashboardMetrics)
async def get_dashboard() -> DashboardMetrics:
    """Return aggregated extraction quality metrics. Stub implementation."""
    return DashboardMetrics(
        total_extractions=0,
        success_rate=0.0,
        auto_approved_rate=0.0,
        qa_sample_rate=0.0,
        human_review_rate=0.0,
        avg_confidence=0.0,
        avg_latency_ms=0.0,
        p95_latency_ms=0.0,
        validation_failure_rate=0.0,
        retry_success_rate=0.0,
    )
