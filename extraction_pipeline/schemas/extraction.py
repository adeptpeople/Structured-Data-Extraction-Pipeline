from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class DocumentType(str, Enum):
    invoice = "invoice"
    contract = "contract"
    research_paper = "research_paper"
    resume = "resume"
    medical_record = "medical_record"
    other = "other"


class ConfidenceScores(BaseModel):
    overall: float = Field(ge=0.0, le=1.0)
    title: Optional[float] = Field(None, ge=0.0, le=1.0)
    author: Optional[float] = Field(None, ge=0.0, le=1.0)
    publication_date: Optional[float] = Field(None, ge=0.0, le=1.0)
    invoice_total: Optional[float] = Field(None, ge=0.0, le=1.0)
    citations: Optional[float] = Field(None, ge=0.0, le=1.0)


class ExtractedDocument(BaseModel):
    document_id: str
    document_type: DocumentType
    document_type_other_detail: Optional[str] = None
    title: Optional[str] = None
    author: Optional[str] = None
    publication_date: Optional[str] = None
    invoice_total: Optional[float] = None
    currency: Optional[str] = None
    citations: Optional[list[str]] = None
    confidence_scores: ConfidenceScores

    @model_validator(mode="after")
    def other_type_requires_detail(self) -> "ExtractedDocument":
        if self.document_type == DocumentType.other and not self.document_type_other_detail:
            raise ValueError(
                "document_type_other_detail is required when document_type is 'other'"
            )
        return self

    def get_field_confidence(self, field: str) -> Optional[float]:
        return getattr(self.confidence_scores, field, None)
