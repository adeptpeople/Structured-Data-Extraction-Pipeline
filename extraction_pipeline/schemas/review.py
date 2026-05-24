from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel


class ReviewRoute(str, Enum):
    auto_approve = "auto_approve"
    qa_sample = "qa_sample"
    human_review = "human_review"


class ReviewQueueItem(BaseModel):
    document_id: str
    route: ReviewRoute
    overall_confidence: float
    flagged_fields: list[str] = []
    extracted_data: dict = {}


class ReviewDecision(str, Enum):
    approve = "approve"
    reject = "reject"
    correct = "correct"


class ReviewSubmission(BaseModel):
    document_id: str
    decision: ReviewDecision
    corrections: Optional[dict] = None
    reviewer_id: Optional[str] = None
