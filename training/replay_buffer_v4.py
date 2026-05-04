"""v4 replay buffer - adds aux fields for v4 heads (score_margin, pin_final).

Inherits storage layout from v3 buffer; adds:
  - score_margin_target: [MAX_PLAYERS] per-seat margin in roughly [-1, 1]
  - pin_final_target:    [NUM_PIECES] int (distance bucket of own pieces at game end)
  - pin_final_valid:     bool
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from config import Config


MAX_PLAYERS = Config.MAX_PLAYERS
PIN_BUCKETS = Config.PIN_FINAL_BUCKETS_V4


class ReplayBufferV4:
    def __init__(self, capacity: int = Config.REPLAY_BUFFER_SIZE):
        self.capacity = capacity
        self.states = np.zeros(
            (capacity, Config.NUM_CHANNELS, Config.GRID_SIZE, Config.GRID_SIZE),
            dtype=np.float16,
        )
        self.policies = np.zeros((capacity, Config.ACTION_SPACE), dtype=np.float16)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.legal_masks = np.zeros((capacity, Config.ACTION_SPACE), dtype=np.bool_)
        self.players = np.zeros(capacity, dtype=np.int8)
        self.value_vec = np.zeros((capacity, MAX_PLAYERS), dtype=np.float32)
        self.n_players = np.zeros(capacity, dtype=np.int8)
        self.opp_action = np.full(capacity, -1, dtype=np.int32)
        self.opp_legal_masks = np.zeros((capacity, Config.ACTION_SPACE), dtype=np.bool_)
        self.plies_remaining = np.zeros(capacity, dtype=np.float32)
        self.plies_valid = np.zeros(capacity, dtype=np.bool_)
        # v4 extras
        self.score_margin = np.zeros((capacity, MAX_PLAYERS), dtype=np.float32)
        self.pin_final = np.zeros((capacity, Config.NUM_PIECES), dtype=np.int8)
        self.pin_final_valid = np.zeros(capacity, dtype=np.bool_)
        self.position = 0
        self.size = 0

    def add(
        self, *,
        state, policy, value, action, legal_mask,
        value_vec, n_players, player,
        opp_action=-1, opp_legal_mask=None,
        plies_remaining=0.0, plies_valid=False,
        score_margin=None, pin_final=None, pin_final_valid=False,
    ):
        idx = self.position % self.capacity
        self.states[idx] = state.astype(np.float16)
        self.policies[idx] = policy.astype(np.float16)
        self.values[idx] = float(value)
        self.actions[idx] = int(action)
        self.legal_masks[idx] = legal_mask.astype(np.bool_)
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
        if score_margin is not None:
            self.score_margin[idx] = np.asarray(score_margin, dtype=np.float32)
        if pin_final is not None:
            self.pin_final[idx] = np.asarray(pin_final, dtype=np.int8)
        self.pin_final_valid[idx] = bool(pin_final_valid)
        self.position += 1
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        if batch_size > self.size:
            batch_size = self.size
        idx = np.random.randint(0, self.size, batch_size)
        return {
            "states": torch.from_numpy(self.states[idx].astype(np.float32)),
            "policies": torch.from_numpy(self.policies[idx].astype(np.float32)),
            "values": torch.from_numpy(self.values[idx]),
            "actions": torch.from_numpy(self.actions[idx].astype(np.int64)),
            "legal_masks": torch.from_numpy(self.legal_masks[idx]),
            "players": torch.from_numpy(self.players[idx].astype(np.int64)),
            "value_vec": torch.from_numpy(self.value_vec[idx]),
            "n_players": torch.from_numpy(self.n_players[idx].astype(np.int64)),
            "opp_action": torch.from_numpy(self.opp_action[idx].astype(np.int64)),
            "opp_legal_mask": torch.from_numpy(self.opp_legal_masks[idx]),
            "plies_target": torch.from_numpy(self.plies_remaining[idx]),
            "plies_valid": torch.from_numpy(self.plies_valid[idx]),
            "score_margin_target": torch.from_numpy(self.score_margin[idx]),
            "pin_final_target": torch.from_numpy(self.pin_final[idx].astype(np.int64)),
            "pin_final_valid": torch.from_numpy(self.pin_final_valid[idx]),
            "indices": idx,
        }

    def __len__(self):
        return self.size
