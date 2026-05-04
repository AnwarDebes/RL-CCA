"""Replay buffer for AlphaZero-style multi-iteration training.

Standard FIFO buffer with size cap; supports two sampling modes:

  * uniform: sample uniformly with replacement (the default).
  * recent-weighted: more weight on recent entries (controllable
    half-life - useful when the policy improves rapidly and old data
    is stale).

Each entry is a self-contained dict matching the trajectory-entry
format produced by `play_one_cc_game` and the kingmaker self-play loop.

The buffer is **thread-safe** for single-writer/multiple-reader usage
(worker writes while training reads). A simple lock guards the deque.
"""

from __future__ import annotations

import math
import random
import threading
from collections import deque
from typing import Any, Dict, Iterable, List, Optional


class ReplayBuffer:
    """FIFO buffer with capped size and optional recent-weighted sampling."""

    def __init__(self, capacity: int = 100_000) -> None:
        self.capacity = capacity
        self._buf: deque = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)

    def add(self, entry: Dict[str, Any]) -> None:
        """Add a single trajectory entry."""
        with self._lock:
            self._buf.append(entry)

    def add_many(self, entries: Iterable[Dict[str, Any]]) -> None:
        with self._lock:
            for e in entries:
                self._buf.append(e)

    def sample(
        self,
        n: int,
        mode: str = "uniform",
        half_life: int = 50_000,
        rng: Optional[random.Random] = None,
    ) -> List[Dict[str, Any]]:
        """Draw n entries.

        Args:
            n: how many to sample.
            mode: "uniform" or "recent_weighted".
            half_life: in 'recent_weighted' mode, the index distance after
                which an entry's weight halves. Larger = more uniform.
            rng: optional random.Random for reproducibility.
        """
        rng = rng or random
        with self._lock:
            sz = len(self._buf)
            if sz == 0:
                return []
            if mode == "uniform":
                return [rng.choice(self._buf) for _ in range(n)]
            if mode == "recent_weighted":
                # Newest item has weight 1; weights decay exponentially with
                # distance from newest.
                weights = [
                    2 ** (-(sz - 1 - i) / max(1, half_life))
                    for i in range(sz)
                ]
                # rng.choices accepts weights kw
                return rng.choices(list(self._buf), weights=weights, k=n)
            raise ValueError(f"unknown sampling mode: {mode}")

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()

    def state_dict(self) -> Dict[str, Any]:
        """Serialise enough to restore the buffer (entries kept verbatim)."""
        with self._lock:
            return dict(capacity=self.capacity, entries=list(self._buf))

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        with self._lock:
            self.capacity = state["capacity"]
            self._buf = deque(state["entries"], maxlen=self.capacity)
