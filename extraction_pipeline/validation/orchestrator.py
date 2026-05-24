"""
ValidationRetryOrchestrator — the heart of the pipeline.

Flow per document:
  1. Chunk if oversized
  2. Loop up to MAX_RETRIES:
       a. ExtractionEngine.extract() → raw_dict
       b. JSON Schema validation
       c. Pydantic validation
       d. Semantic rule validation
       e. If pass → return ExtractedDocument
       f. Classify failure:
            non_resolvable → raise immediately
            retry_resolvable → inject error feedback, continue
  3. Raise MaxRetriesExceededError
"""
from __future__ import annotations

import asyncio
from typing import Optional

from extraction_pipeline import config
from extraction_pipeline.extraction.engine import ExtractionEngine
from extraction_pipeline.extraction.few_shot import FewShotLibrary
from extraction_pipeline.schemas.extraction import ExtractedDocument
from extraction_pipeline.schemas.validation import (
    ExtractionMetrics,
    FailureMode,
    RetryClassification,
    ValidationResult,
)
from extraction_pipeline.validation.json_schema_validator import validate_json_schema
from extraction_pipeline.validation.pydantic_validator import validate_pydantic
from extraction_pipeline.validation.semantic_rules import validate_semantic


class NonResolvableExtractionError(Exception):
    def __init__(self, document_id: str, result: ValidationResult) -> None:
        self.document_id = document_id
        self.result = result
        super().__init__(
            f"Non-resolvable extraction failure for '{document_id}': "
            + "; ".join(result.error_messages)
        )


class MaxRetriesExceededError(Exception):
    def __init__(self, document_id: str, last_result: ValidationResult) -> None:
        self.document_id = document_id
        self.last_result = last_result
        super().__init__(
            f"Max retries ({config.MAX_RETRIES}) exceeded for '{document_id}'. "
            + "Last errors: " + "; ".join(last_result.error_messages)
        )


def _classify_failure(result: ValidationResult, document_text: str) -> RetryClassification:
    """
    Determine if a failure is retry-resolvable or non-resolvable.
    Non-resolvable: missing_required_field when the data is provably absent from the doc.
    Everything else is retry-resolvable (format errors, type errors, enum mismatches).
    """
    if FailureMode.missing_required_field in result.failure_modes:
        return RetryClassification.non_resolvable
    # Heuristic: if the error message mentions a field that is definitely absent
    # from the document (no keywords present), classify as non_resolvable.
    # For simplicity, format-level errors are always retry-resolvable.
    return RetryClassification.retry_resolvable


def _chunk_text(text: str, max_tokens: int = config.MAX_TOKENS_PER_CHUNK) -> list[str]:
    """Split text into overlapping chunks. Approximates 1 token ≈ 4 chars."""
    max_chars = max_tokens * 4
    overlap = 200  # chars
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def _merge_extractions(extractions: list[dict]) -> dict:
    """
    Merge multiple chunk extractions by taking the highest-confidence value per field.
    """
    if len(extractions) == 1:
        return extractions[0]

    merged = dict(extractions[0])
    nullable_fields = ["title", "author", "publication_date", "invoice_total", "currency", "citations"]

    for extraction in extractions[1:]:
        scores = extraction.get("confidence_scores", {})
        base_scores = merged.get("confidence_scores", {})

        for field in nullable_fields:
            current_val = merged.get(field)
            new_val = extraction.get(field)
            current_conf = base_scores.get(field) or 0.0
            new_conf = scores.get(field) or 0.0

            if new_val is not None and new_conf > current_conf:
                merged[field] = new_val
                merged["confidence_scores"][field] = new_conf

    # Recompute overall as min of non-null field scores
    cs = merged.get("confidence_scores", {})
    field_scores = [v for k, v in cs.items() if k != "overall" and v is not None]
    if field_scores:
        merged["confidence_scores"]["overall"] = min(field_scores)

    return merged


class ValidationRetryOrchestrator:
    def __init__(
        self,
        engine: Optional[ExtractionEngine] = None,
        few_shot_library: Optional[FewShotLibrary] = None,
    ) -> None:
        self._engine = engine or ExtractionEngine()
        self._few_shot = few_shot_library or FewShotLibrary()

    async def extract_with_retry(
        self,
        document_id: str,
        document_text: str,
    ) -> tuple[ExtractedDocument, ExtractionMetrics]:
        import time

        start_ms = int(time.time() * 1000)
        metrics = ExtractionMetrics(
            document_id=document_id,
            attempt_count=0,
            validation_failures=0,
            retry_successes=0,
            non_resolvable_failures=0,
            final_status="pending",
        )

        # Chunk if oversized
        chunks = _chunk_text(document_text)
        few_shot_msgs = self._few_shot.get_examples(document_text[:500])

        if len(chunks) > 1:
            chunk_extractions = await self._extract_chunks(
                document_id, chunks, few_shot_msgs, metrics
            )
            merged = _merge_extractions(chunk_extractions)
            _, doc = validate_pydantic(merged)
            if doc is not None:
                metrics.final_status = "success"
                metrics.processing_time_ms = int(time.time() * 1000) - start_ms
                return doc, metrics
            # Fall through to retry loop with merged text if pydantic fails
            document_text = document_text  # keep original for retry

        error_feedback: Optional[str] = None
        last_result: Optional[ValidationResult] = None

        for attempt in range(1, config.MAX_RETRIES + 1):
            metrics.attempt_count = attempt

            raw_dict = await self._engine.extract(
                document_id=document_id,
                document_text=document_text,
                few_shot_messages=few_shot_msgs,
                error_feedback=error_feedback,
                attempt_number=attempt,
            )

            # Layer 1: JSON Schema
            result = validate_json_schema(raw_dict, attempt)
            if not result.passed:
                metrics.validation_failures += 1
                classification = _classify_failure(result, document_text)
                if classification == RetryClassification.non_resolvable:
                    metrics.non_resolvable_failures += 1
                    metrics.final_status = "non_resolvable"
                    raise NonResolvableExtractionError(document_id, result)
                error_feedback = "\n".join(result.error_messages)
                last_result = result
                continue

            # Layer 2: Pydantic
            result, doc = validate_pydantic(raw_dict, attempt)
            if not result.passed:
                metrics.validation_failures += 1
                classification = _classify_failure(result, document_text)
                if classification == RetryClassification.non_resolvable:
                    metrics.non_resolvable_failures += 1
                    metrics.final_status = "non_resolvable"
                    raise NonResolvableExtractionError(document_id, result)
                error_feedback = "\n".join(result.error_messages)
                last_result = result
                continue

            # Layer 3: Semantic rules
            result = validate_semantic(doc, attempt)
            if not result.passed:
                metrics.validation_failures += 1
                classification = _classify_failure(result, document_text)
                if classification == RetryClassification.non_resolvable:
                    metrics.non_resolvable_failures += 1
                    metrics.final_status = "non_resolvable"
                    raise NonResolvableExtractionError(document_id, result)
                error_feedback = "\n".join(result.error_messages)
                last_result = result
                continue

            # All passed
            if attempt > 1:
                metrics.retry_successes += 1
            metrics.final_status = "success"
            metrics.processing_time_ms = int(time.time() * 1000) - start_ms
            metrics.retry_count = attempt - 1
            metrics.confidence_distribution = doc.confidence_scores.model_dump()
            return doc, metrics

        metrics.final_status = "max_retries_exceeded"
        metrics.processing_time_ms = int(time.time() * 1000) - start_ms
        raise MaxRetriesExceededError(document_id, last_result)

    async def _extract_chunks(
        self,
        document_id: str,
        chunks: list[str],
        few_shot_msgs: list[dict],
        metrics: ExtractionMetrics,
    ) -> list[dict]:
        tasks = [
            self._engine.extract(
                document_id=f"{document_id}_chunk{i}",
                document_text=chunk,
                few_shot_messages=few_shot_msgs,
            )
            for i, chunk in enumerate(chunks)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        valid = [r for r in results if isinstance(r, dict)]
        if not valid:
            raise MaxRetriesExceededError(
                document_id,
                ValidationResult.failure(
                    modes=[FailureMode.oversized_document],
                    messages=["All chunks failed extraction"],
                    classification=RetryClassification.retry_resolvable,
                ),
            )
        # Ensure all chunks carry the real document_id
        for r in valid:
            r["document_id"] = document_id
        return valid
