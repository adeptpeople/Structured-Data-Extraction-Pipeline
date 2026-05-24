"""
Redis-backed review queue using sorted sets.
Score = overall_confidence (ascending = lowest confidence reviewed first).
"""
from __future__ import annotations

import json
from typing import Optional

from extraction_pipeline.schemas.review import ReviewQueueItem, ReviewRoute

try:
    import redis.asyncio as aioredis
    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False


_QUEUE_KEYS = {
    ReviewRoute.human_review: "queue:human_review",
    ReviewRoute.qa_sample: "queue:qa_sample",
}


class ReviewQueueManager:
    def __init__(self, redis_url: str = "redis://localhost:6379/0") -> None:
        self._redis_url = redis_url
        self._client: Optional[object] = None

    async def _get_client(self):
        if not _REDIS_AVAILABLE:
            raise RuntimeError("redis package not installed. Run: pip install redis")
        if self._client is None:
            self._client = aioredis.from_url(self._redis_url)
        return self._client

    async def enqueue(self, item: ReviewQueueItem) -> None:
        key = _QUEUE_KEYS.get(item.route)
        if key is None:
            return  # auto_approve doesn't queue
        client = await self._get_client()
        payload = json.dumps(item.model_dump())
        await client.zadd(key, {payload: item.overall_confidence})

    async def dequeue_batch(
        self, route: ReviewRoute, count: int = 20
    ) -> list[ReviewQueueItem]:
        key = _QUEUE_KEYS.get(route)
        if key is None:
            return []
        client = await self._get_client()
        items = await client.zrange(key, 0, count - 1, withscores=False)
        return [ReviewQueueItem(**json.loads(i)) for i in items]

    async def remove(self, route: ReviewRoute, document_id: str) -> None:
        key = _QUEUE_KEYS.get(route)
        if key is None:
            return
        client = await self._get_client()
        items = await client.zrange(key, 0, -1)
        for item_bytes in items:
            data = json.loads(item_bytes)
            if data.get("document_id") == document_id:
                await client.zrem(key, item_bytes)
                break
