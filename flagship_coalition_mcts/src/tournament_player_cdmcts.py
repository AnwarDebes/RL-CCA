"""Tournament player wrapping CD-MCTS for deployment in the existing
nexus tournament infrastructure.

Drop-in replacement for `NexusTournamentPlayerV4` (same interface):

    player = NexusTournamentPlayerCDMCTS(model_path="checkpoints/cdmcts_cc_final.pt")
    player.set_color(my_color, all_player_colors)
    move = player.choose_move(state_pins, legal_moves, ...)

Key differences vs v4 baseline:
  * Uses CD-MCTS (PL value head + coalition belief + EXP-IX selector +
    vector backup) instead of scalar PUCT.
  * Time budgeting: same adaptive simulation count (opening / midgame /
    endgame) and hard wall-clock cap.
  * Subtree reuse is currently NOT implemented (it requires CD-MCTS-side
    `advance_root` analogous to v4's; pending future work - falls back to
    fresh-tree per move).
  * Coalition_weight at inference: 0.5 (matches training default). Can
    be set lower (e.g. 0.0) to disable coalition penalty if undesired.

Robust fallbacks (matches v4 player):
  * On exception: fall back to raw policy network (no MCTS).
  * On second exception: fall back to HeuristicAgent.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import torch

_NEXUS_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if _NEXUS_ROOT not in sys.path:
    sys.path.insert(0, _NEXUS_ROOT)

from config import Config
from core.action_space import decode_action_to_server, get_legal_actions
from core.board import HexBoard
from core.game_env import GameEnv

from .cc_runner import CCMCTSEvaluator, build_cc_network
from .games.chinese_checkers import ChineseCheckersGame
from .mcts import run_mcts


class NexusTournamentPlayerCDMCTS:
    """CD-MCTS tournament player. Same interface as v4."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        coalition_weight: float = 0.5,
        sim_budget_opening: Optional[int] = None,
        sim_budget_midgame: Optional[int] = None,
        sim_budget_endgame: Optional[int] = None,
        hard_budget_sec: Optional[float] = None,
        channels: int = 64,
        num_blocks: int = 4,
        hidden_dim: int = 128,
    ) -> None:
        self.device = device
        self.coalition_weight = coalition_weight
        self.sim_budget_opening = sim_budget_opening or getattr(
            Config, "MCTS_INF_SIMS_OPENING_V4", 48
        )
        self.sim_budget_midgame = sim_budget_midgame or getattr(
            Config, "MCTS_INF_SIMS_MIDGAME_V4", 32
        )
        self.sim_budget_endgame = sim_budget_endgame or getattr(
            Config, "MCTS_INF_SIMS_ENDGAME_V4", 16
        )
        self.hard_budget_sec = hard_budget_sec or getattr(
            Config, "MCTS_INF_HARD_BUDGET_SEC", 0.45
        )

        self.network = build_cc_network(
            num_players_max=6, channels=channels,
            num_blocks=num_blocks, hidden_dim=hidden_dim,
        )
        if model_path is not None and os.path.exists(model_path):
            try:
                from .checkpoint import load_checkpoint
                load_checkpoint(model_path, self.network, strict=False)
                print(f"[CDMCTS player] loaded {model_path}")
            except Exception as e:
                print(f"[CDMCTS player] load failed: {e}; using untrained net")
        self.network.eval()
        self.network.to(device)

        # Per-game state
        self.color: Optional[str] = None
        self.all_player_colors: Optional[List[str]] = None
        self._board: Optional[HexBoard] = None
        # Cached MCTS root (for subtree reuse across moves)
        self._cached_root = None
        self._cached_action_history: List[int] = []

    def set_color(self, color: str, all_player_colors: Optional[List[str]] = None) -> None:
        self.color = color
        self.all_player_colors = all_player_colors
        self._board = None  # rebuilt lazily per move
        self._cached_root = None
        self._cached_action_history = []

    def _adaptive_sims(self, plies: int) -> int:
        if plies <= 10:
            return self.sim_budget_opening
        if plies <= 60:
            return self.sim_budget_midgame
        return self.sim_budget_endgame

    def _build_env(self, state_pins: Dict[str, List[int]]) -> GameEnv:
        if self._board is None:
            self._board = HexBoard()
        # Determine number of players from state_pins.
        present = [c for c in self.all_player_colors if c in state_pins]
        n = len(present)
        env = GameEnv(board=self._board, num_players=n)
        env.colors = list(present)
        # Place pieces from state_pins
        for p, c in enumerate(present):
            env.pieces[p] = list(state_pins[c])
        # Set current player from self.color
        if self.color in present:
            env.current_player = present.index(self.color)
        else:
            env.current_player = 0
        # Reset move count to 0 (we don't know exact ply at the server level
        # in this minimal wrapper; tournament infra would fill this in).
        return env

    def choose_move(
        self,
        state_pins: Dict[str, List[int]],
        legal_moves: List[Dict],
        plies: int = 0,
        time_remaining: float = 60.0,
    ) -> Dict:
        """Choose a move. Returns one of `legal_moves` (a dict matching
        the tournament server's move-spec convention).

        Robust fallbacks:
          1. CD-MCTS: full PL + coalition + EXP-IX
          2. Raw policy: argmax over network's policy head (no MCTS)
          3. Heuristic agent
        """
        try:
            return self._cdmcts_move(state_pins, legal_moves, plies, time_remaining)
        except Exception as e:
            print(f"[CDMCTS player] CD-MCTS failed: {e}; falling back to raw policy")
            try:
                return self._policy_move(state_pins, legal_moves)
            except Exception as e2:
                print(f"[CDMCTS player] raw policy failed: {e2}; falling back to heuristic")
                return self._heuristic_move(state_pins, legal_moves)

    def _cdmcts_move(
        self,
        state_pins: Dict[str, List[int]],
        legal_moves: List[Dict],
        plies: int,
        time_remaining: float,
    ) -> Dict:
        env = self._build_env(state_pins)
        sims = self._adaptive_sims(plies)
        evaluator = CCMCTSEvaluator(self.network)
        game = ChineseCheckersGame()

        t0 = time.time()
        # Run MCTS, but cap by hard wall-clock budget.
        sims_per_chunk = max(1, sims // 4)
        sims_run = 0
        from .mcts import _build_root, run_simulation
        from .subtree_reuse import reuse_or_rebuild
        # Try subtree reuse from prior move's tree.
        root = reuse_or_rebuild(
            self._cached_root, self._cached_action_history,
            env, evaluator, game,
        )
        # If reuse succeeded, clear the action history (we've consumed it).
        if root is not self._cached_root:
            self._cached_root = None
        self._cached_action_history = []
        rng = np.random.default_rng(int(time.time() * 1e6) & 0x7fffffff)
        while sims_run < sims and (time.time() - t0) < self.hard_budget_sec:
            chunk = min(sims_per_chunk, sims - sims_run)
            for _ in range(chunk):
                run_simulation(root, evaluator, game, rng, coalition_weight=self.coalition_weight)
            sims_run += chunk
        from .cce_selector import policy_at_root
        pi = policy_at_root(root.selector_state, temperature=1.0)
        # Pick the most-visited action
        legal_actions = root.legal_actions
        action_idx = int(np.argmax(pi))
        action = legal_actions[action_idx]
        # Cache for next move's subtree reuse: stash this root and remember
        # we played `action`. Next move, the caller is expected to call
        # advance_with_opponent_action(opp_action) before choose_move so we
        # know the full path.
        self._cached_root = root
        self._cached_action_history = [action]
        return decode_action_to_server(action, env)

    def advance_with_opponent_action(self, opponent_raw_action: int) -> None:
        """Inform the player of an opponent's actual action so subtree
        reuse can descend through it on the next move.

        Tournament infra should call this between our moves whenever the
        opponent acts. If the opponent's action wasn't expanded in our
        tree, the next choose_move will rebuild from scratch (graceful).
        """
        if self._cached_root is None:
            return
        self._cached_action_history.append(opponent_raw_action)

    def _policy_move(self, state_pins, legal_moves):
        """Argmax raw network policy over legal actions."""
        env = self._build_env(state_pins)
        from .cc_runner import CCMCTSEvaluator
        ev = CCMCTSEvaluator(self.network)
        out = ev.evaluate(env)
        legal = ChineseCheckersGame.legal_actions(env)
        if not legal:
            raise RuntimeError("no legal actions")
        sub = out.prior_policy[legal]
        a = legal[int(np.argmax(sub))]
        return decode_action_to_server(a, env)

    def _heuristic_move(self, state_pins, legal_moves):
        from training.heuristic_agent import HeuristicAgent
        env = self._build_env(state_pins)
        agent = HeuristicAgent(env.board)
        a = agent.choose_move(env, env.current_player)
        return decode_action_to_server(a, env)
