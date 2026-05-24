"""
ConfidenceRouter — routes extracted documents to auto_approve, qa_sample, or human_review
based on overall confidence and field-level escalation rules.
"""
from __future__ import annotations

from extraction_pipeline import config
from extraction_pipeline.schemas.extraction import ExtractedDocument
from extraction_pipeline.schemas.review import ReviewQueueItem, ReviewRoute


class ConfidenceRouter:
    def route(self, doc: ExtractedDocument) -> ReviewQueueItem:
        overall = doc.confidence_scores.overall
        flagged_fields: list[str] = []

        # Primary routing by overall score
        if overall >= config.CONFIDENCE_AUTO_APPROVE:
            route = ReviewRoute.auto_approve
        elif overall >= config.CONFIDENCE_QA_SAMPLE:
            route = ReviewRoute.qa_sample
        else:
            route = ReviewRoute.human_review

        # Field-level escalation: escalate to qa_sample if a critical field is low-confidence
        if route == ReviewRoute.auto_approve:
            for field in config.CRITICAL_FIELDS:
                field_conf = doc.get_field_confidence(field)
                if field_conf is not None and field_conf < config.CONFIDENCE_QA_SAMPLE:
                    flagged_fields.append(field)
            if flagged_fields:
                route = ReviewRoute.qa_sample

        return ReviewQueueItem(
            document_id=doc.document_id,
            route=route,
            overall_confidence=overall,
            flagged_fields=flagged_fields,
            extracted_data=doc.model_dump(),
        )

    def route_field_level(self, doc: ExtractedDocument) -> dict[str, ReviewRoute]:
        """Return per-field routing for fine-grained review assignment."""
        result: dict[str, ReviewRoute] = {}
        scores = doc.confidence_scores.model_dump()
        for field, conf in scores.items():
            if conf is None or field == "overall":
                continue
            if conf >= config.CONFIDENCE_AUTO_APPROVE:
                result[field] = ReviewRoute.auto_approve
            elif conf >= config.CONFIDENCE_QA_SAMPLE:
                result[field] = ReviewRoute.qa_sample
            else:
                result[field] = ReviewRoute.human_review
        return result
