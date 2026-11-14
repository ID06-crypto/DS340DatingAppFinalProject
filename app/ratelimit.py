"""A small in-memory rate limiter. It assumes one server process."""

import threading
import time
from collections import defaultdict, deque


class RateLimiter:
    def __init__(self, limit: int, window_seconds: float) -> None:
        self.limit = limit
        self.window = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        """Record an attempt, and return whether it's within the limit."""
        now = time.monotonic()
        with self._lock:
            hits = self._hits[key]
            while hits and hits[0] <= now - self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                return False
            hits.append(now)
            return True

    def exceeded(self, key: str) -> bool:
        """Whether the limit is already used up, without recording anything."""
        now = time.monotonic()
        with self._lock:
            hits = self._hits[key]
            while hits and hits[0] <= now - self.window:
                hits.popleft()
            return len(hits) >= self.limit

    def record(self, key: str) -> None:
        """Count one attempt, such as a failed password."""
        with self._lock:
            self._hits[key].append(time.monotonic())
