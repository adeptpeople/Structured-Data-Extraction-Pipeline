"""Async batch status poller with exponential backoff."""
from __future__ import annotations

import asyncio
from typing import Optional

import openai

from extraction_pipeline import config


class BatchPoller:
    def __init__(self, client: Optional[openai.AsyncOpenAI] = None) -> None:
        self._client = client or openai.AsyncOpenAI(api_key=config.OPENAI_API_KEY)

    async def poll(self, batch_id: str) -> object:
        """Poll until the batch reaches a terminal state. Returns the final batch object."""
        interval = config.BATCH_POLL_INTERVAL_SECS
        terminal_statuses = {"completed", "failed", "cancelled", "expired"}

        while True:
            batch = await self._client.batches.retrieve(batch_id)
            if batch.status in terminal_statuses:
                return batch
            await asyncio.sleep(interval)
            interval = min(interval * 1.5, config.BATCH_POLL_MAX_SECS)
