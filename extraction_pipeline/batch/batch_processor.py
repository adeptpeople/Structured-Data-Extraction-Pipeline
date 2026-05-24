"""
BatchProcessor — submits 100-document batches via the OpenAI Message Batches API,
polls for completion, and recovers failed jobs by custom_id.
"""
from __future__ import annotations

import json
import time
from typing import Optional

import openai

from extraction_pipeline import config
from extraction_pipeline.batch.batch_poller import BatchPoller
from extraction_pipeline.extraction.engine import ExtractionEngine
from extraction_pipeline.extraction.few_shot import FewShotLibrary
from extraction_pipeline.schemas.batch import BatchJob, BatchRequest, BatchResult


class BatchProcessor:
    def __init__(
        self,
        client: Optional[openai.AsyncOpenAI] = None,
        engine: Optional[ExtractionEngine] = None,
        few_shot_library: Optional[FewShotLibrary] = None,
    ) -> None:
        self._client = client or openai.AsyncOpenAI(api_key=config.OPENAI_API_KEY)
        self._engine = engine or ExtractionEngine(self._client)
        self._few_shot = few_shot_library or FewShotLibrary()
        self._poller = BatchPoller(self._client)

    async def submit_batch(self, documents: list[BatchRequest]) -> str:
        """Build and submit a Message Batches API batch. Returns the OpenAI batch_id."""
        requests = []
        for doc in documents:
            few_shot_msgs = self._few_shot.get_examples(doc.document_text[:500])
            body = self._engine.build_batch_request_body(
                doc.document_id, doc.document_text, few_shot_msgs
            )
            requests.append({
                "custom_id": doc.custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            })

        batch = await self._client.batches.create(
            requests=requests,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        return batch.id

    async def poll_until_complete(self, batch_id: str) -> BatchJob:
        """Poll the batch until terminal, then parse per-custom_id results."""
        batch = await self._poller.poll(batch_id)

        results: list[BatchResult] = []
        failed_ids: list[str] = []

        if batch.status == "completed" and batch.output_file_id:
            file_content = await self._client.files.content(batch.output_file_id)
            for line in file_content.text.strip().split("\n"):
                if not line.strip():
                    continue
                entry = json.loads(line)
                custom_id = entry.get("custom_id", "unknown")
                t0 = time.time()

                if entry.get("error"):
                    failed_ids.append(custom_id)
                    results.append(BatchResult(
                        custom_id=custom_id,
                        document_id=custom_id,
                        success=False,
                        error_message=str(entry["error"]),
                        latency_ms=0,
                        error_type=entry["error"].get("code", "unknown"),
                    ))
                    continue

                response_body = entry.get("response", {}).get("body", {})
                choices = response_body.get("choices", [])
                if not choices:
                    failed_ids.append(custom_id)
                    results.append(BatchResult(
                        custom_id=custom_id,
                        document_id=custom_id,
                        success=False,
                        error_message="No choices in response",
                        latency_ms=0,
                        error_type="empty_response",
                    ))
                    continue

                tool_calls = choices[0].get("message", {}).get("tool_calls", [])
                if not tool_calls:
                    failed_ids.append(custom_id)
                    results.append(BatchResult(
                        custom_id=custom_id,
                        document_id=custom_id,
                        success=False,
                        error_message="No tool_calls in response",
                        latency_ms=0,
                        error_type="no_tool_call",
                    ))
                    continue

                extracted = json.loads(tool_calls[0]["function"]["arguments"])
                latency = int((time.time() - t0) * 1000)
                results.append(BatchResult(
                    custom_id=custom_id,
                    document_id=custom_id,
                    success=True,
                    extracted_data=extracted,
                    latency_ms=latency,
                ))

        completed = sum(1 for r in results if r.success)
        failed = len(failed_ids)

        return BatchJob(
            batch_id=batch_id,
            status=batch.status,
            submitted_count=len(results),
            completed_count=completed,
            failed_count=failed,
            results=results,
            failed_custom_ids=failed_ids,
        )

    async def recover_failed(
        self,
        failed_custom_ids: list[str],
        original_documents: dict[str, BatchRequest],
    ) -> Optional[str]:
        """Re-submit failed jobs as a new mini-batch. Returns new batch_id or None."""
        if not failed_custom_ids:
            return None
        failed_docs = [
            original_documents[cid]
            for cid in failed_custom_ids
            if cid in original_documents
        ]
        if not failed_docs:
            return None
        return await self.submit_batch(failed_docs)

    async def process_batch(
        self, documents: list[BatchRequest]
    ) -> tuple[BatchJob, Optional[BatchJob]]:
        """
        Full pipeline: submit → poll → recover failed.
        Returns (primary_job, recovery_job | None).
        """
        batch_id = await self.submit_batch(documents)
        job = await self.poll_until_complete(batch_id)

        recovery_job: Optional[BatchJob] = None
        if job.failed_custom_ids:
            doc_map = {d.custom_id: d for d in documents}
            recovery_batch_id = await self.recover_failed(job.failed_custom_ids, doc_map)
            if recovery_batch_id:
                job.recovery_batch_id = recovery_batch_id
                recovery_job = await self.poll_until_complete(recovery_batch_id)

        return job, recovery_job
