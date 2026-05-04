"""v3 replay buffer - adds aux fields: per-player value vector, n_players,
opp_action (next opponent's move), opp_legal_mask, plies_remaining.

Layout matches v2 ReplayBuffer (numpy ring) so sampling stays cheap.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from config import Config


MAX_PLAYERS = 6


class ReplayBufferV3:
    def __init__(self, capacity: int = Config.REPLAY_BUFFER_SIZE):
        self.capacity = capacity
        # core (same as v2)
        self.states = np.zeros(
            (capacity, Config.NUM_CHANNELS, Config.GRID_SIZE, Config.GRID_SIZE),
            dtype=np.float16,
        )
        self.policies = np.zeros((capacity, Config.ACTION_SPACE), dtype=np.float16)
        self.values = np.zeros(capacity, dtype=np.float32)         # current-player slot
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.legal_masks = np.zeros((capacity, Config.ACTION_SPACE), dtype=np.bool_)
        self.game_ids = np.zeros(capacity, dtype=np.int32)
        self.step_indices = np.zeros(capacity, dtype=np.int16)
        self.priorities = np.ones(capacity, dtype=np.float32)
        self.players = np.zeros(capacity, dtype=np.int8)           # current seat (0..5)
        # v3 extras
        self.value_vec = np.zeros((capacity, MAX_PLAYERS), dtype=np.float32)  # full vec
        self.n_players = np.zeros(capacity, dtype=np.int8)
        self.opp_action = np.full(capacity, -1, dtype=np.int32)
        self.opp_legal_masks = np.zeros((capacity, Config.ACTION_SPACE), dtype=np.bool_)
        self.plies_remaining = np.zeros(capacity, dtype=np.float32)
        self.plies_valid = np.zeros(capacity, dtype=np.bool_)

        self.position = 0
        self.size = 0

    def add(
        self, *,
        state, policy, value, action, legal_mask,
        value_vec, n_players, player,
        opp_action=-1, opp_legal_mask=None,
        plies_remaining=0.0, plies_valid=False,
        game_id=0, step=0,
    ):
        idx = self.position % self.capacity
        self.states[idx] = state.astype(np.float16)
        self.policies[idx] = policy.astype(np.float16)
        self.values[idx] = float(value)
        self.actions[idx] = int(action)
        self.legal_masks[idx] = legal_mask.astype(np.bool_)
        self.game_ids[idx] = int(game_id)
        self.step_indices[idx] = int(step)
        self.players[idx] = int(player)
        self.value_vec[idx] = np.asarray(value_vec, dtype=np.float32)
        self.n_players[idx] = int(n_players)
        self.opp_action[idx] = int(opp_action)
        if opp_legal_mask is not None:
            self.opp_legal_masks[idx] = opp_legal_mask.astype(np.bool_)
        else:
            self.opp_legal_masks[idx] = False
        self.plies_remaining[idx] = float(plies_remaining)
        self.plies_valid[idx] = bool(plies_valid)
        self.position += 1
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        if batch_size > self.size:
            batch_size = self.size
        indices = np.random.randint(0, self.size, batch_size)
        return {
            "states": torch.from_numpy(self.states[indices].astype(np.float32)),
            "policies": torch.from_numpy(self.policies[indices].astype(np.float32)),
            "values": torch.from_numpy(self.values[indices]),
            "actions": torch.from_numpy(self.actions[indices].astype(np.int64)),
            "legal_masks": torch.from_numpy(self.legal_masks[indices]),
            "players": torch.from_numpy(self.players[indices].astype(np.int64)),
            "value_vec": torch.from_numpy(self.value_vec[indices]),
            "n_players": torch.from_numpy(self.n_players[indices].astype(np.int64)),
            "opp_action": torch.from_numpy(self.opp_action[indices].astype(np.int64)),
            "opp_legal_mask": torch.from_numpy(self.opp_legal_masks[indices]),
            "plies_target": torch.from_numpy(self.plies_remaining[indices]),
            "plies_valid": torch.from_numpy(self.plies_valid[indices]),
            "indices": indices,
        }

    def __len__(self):
        return self.size
