from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from extraction_pipeline.schemas.review import ReviewDecision, ReviewRoute


class ReviewQueueResponse(BaseModel):
    document_id: str
    route: ReviewRoute
    overall_confidence: float
    flagged_fields: list[str]
    extracted_data: dict


class ReviewSubmitRequest(BaseModel):
    decision: ReviewDecision
    corrections: Optional[dict] = None
    reviewer_id: Optional[str] = None


class ReviewSubmitResponse(BaseModel):
    document_id: str
    decision: ReviewDecision
    acknowledged: bool = True


class DashboardMetrics(BaseModel):
    total_extractions: int
    success_rate: float
    auto_approved_rate: float
    qa_sample_rate: float
    human_review_rate: float
    avg_confidence: float
    avg_latency_ms: float
    p95_latency_ms: float
    validation_failure_rate: float
    retry_success_rate: float
