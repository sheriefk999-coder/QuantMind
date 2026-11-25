"""Event bus. Uses Redis pub/sub when configured, otherwise in-process fan-out.

Every module publishes what it does (regime change, opinion, veto, fill...) so
the dashboard can stream it and the audit log can persist it.
"""
from __future__ import annotations

import asyncio
import json
from collections import deque
from datetime import datetime, timezone
from typing import Any

from .config import settings


class EventBus:
    def __init__(self, history: int = 500):
        self._subscribers: set[asyncio.Queue] = set()
        self.history: deque[dict] = deque(maxlen=history)
        self._redis = None
        if settings.redis_url:
            try:
                import redis  # type: ignore

                self._redis = redis.Redis.from_url(settings.redis_url)
            except Exception:
                self._redis = None

    def publish(self, topic: str, payload: dict[str, Any]) -> dict:
        event = {"ts": datetime.now(timezone.utc).isoformat(), "topic": topic, "payload": payload}
        self.history.append(event)
        for q in list(self._subscribers):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass
        if self._redis is not None:
            try:
                self._redis.publish("quantmind.events", json.dumps(event, default=str))
            except Exception:
                pass
        return event

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)


bus = EventBus()
