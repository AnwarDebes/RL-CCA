"""RAM-based replay buffer (v3 - single-scalar value)."""

import numpy as np
import torch
from typing import Dict, Optional

from config import Config


class ReplayBuffer:
    """In-memory replay buffer. Pre-allocates numpy arrays for zero-copy sampling.
    Stores a single-scalar value target per entry (was 4-component in v2)."""

    def __init__(self, capacity: int = Config.REPLAY_BUFFER_SIZE):
        self.capacity = capacity
        self.states = np.zeros(
            (capacity, Config.NUM_CHANNELS, Config.GRID_SIZE, Config.GRID_SIZE),
            dtype=np.float16,
        )
        self.policies = np.zeros((capacity, Config.ACTION_SPACE), dtype=np.float16)
        self.values = np.zeros(capacity, dtype=np.float32)         # scalar
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.legal_masks = np.zeros((capacity, Config.ACTION_SPACE), dtype=np.bool_)
        self.game_ids = np.zeros(capacity, dtype=np.int32)
        self.step_indices = np.zeros(capacity, dtype=np.int16)
        self.priorities = np.ones(capacity, dtype=np.float32)

        self.position = 0
        self.size = 0

    def add(
        self,
        state: np.ndarray,
        policy: np.ndarray,
        value: float,
        reward: float,
        action: int,
        legal_mask: np.ndarray,
        game_id: int = 0,
        step: int = 0,
    ):
        idx = self.position % self.capacity
        self.states[idx] = state.astype(np.float16)
        self.policies[idx] = policy.astype(np.float16)
        self.values[idx] = float(value)
        self.rewards[idx] = reward
        self.actions[idx] = action
        self.legal_masks[idx] = legal_mask.astype(np.bool_)
        self.game_ids[idx] = game_id
        self.step_indices[idx] = step
        self.position += 1
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, prioritized: bool = False) -> Dict[str, torch.Tensor]:
        if batch_size > self.size:
            batch_size = self.size

        if prioritized and self.size > 0:
            probs = self.priorities[:self.size] / self.priorities[:self.size].sum()
            indices = np.random.choice(self.size, batch_size, p=probs, replace=False)
        else:
            indices = np.random.randint(0, self.size, batch_size)

        return {
            "states": torch.from_numpy(self.states[indices].astype(np.float32)),
            "policies": torch.from_numpy(self.policies[indices].astype(np.float32)),
            "values": torch.from_numpy(self.values[indices]),  # [B] scalar
            "rewards": torch.from_numpy(self.rewards[indices]),
            "actions": torch.from_numpy(self.actions[indices].astype(np.int64)),
            "legal_masks": torch.from_numpy(self.legal_masks[indices]),
            "game_ids": self.game_ids[indices],
            "step_indices": self.step_indices[indices],
            "indices": indices,
        }

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        self.priorities[indices] = priorities

    def __len__(self):
        return self.size
