"""MCTS v4 - BATCHED LEAF EVALUATION variant.

Drop-in replacement for `GumbelMCTSv4` with proper GPU utilization via
batched leaf evaluation + virtual loss.

The original mcts_v4.py does ONE forward pass per simulation (batch=1),
which leaves the GPU mostly idle for a 7.5M-parameter network. This
variant collects up to BATCH_SIZE leaves before doing a single batched
forward, then distributes results back to each leaf for backup.

Correctness invariants
----------------------
1. With `batch_size=1`, behavior is identical to `GumbelMCTSv4` (modulo
   internal virtual-loss ops that cancel out for a single descent).
   Proven by `tests/test_mcts_v4_batched_equivalence.py`.

2. Virtual loss is added on descent, removed on backup. Each leaf
   collected in a batch has exactly one v-loss application along its
   path, removed when its backup completes.

3. The batched forward processes `len(batch)` states. Network output's
   batch-dim is sliced one entry per leaf; per-state dictionary fields
   are extracted in order.

4. Root expansion (initial) is still a single forward - the batch
   pattern only kicks in for in-search rollouts.

Implementation details
----------------------
* Reuses NodeV4 structure from mcts_v4 (no schema change).
* Reuses action helpers and mcts.utils.
* Adds `batch_size: int = Config.MCTS_BATCH_LEAVES` to constructor.
* New method `_collect_leaf_with_vloss` performs a single descent with
  virtual loss applied along the path. Returns (path, leaf_node).
* New method `_batched_forward` runs one forward over a list of nodes.
* New method `_simulate_batch` collects a queue of root-actions into
  batches of size up to `batch_size` and processes them.

Risk mitigation
---------------
* The original `mcts_v4.py` is NOT modified - this file is additive.
* If something goes wrong, swap back is one import-line change.
* Equivalence tests run on CPU with no network calls actually crossing
  to GPU (small CPU model), so they can run during Phase 2 GPU training.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from config import Config
from core.action_space import get_legal_actions
from core.game_env import GameEnv
from mcts.mcts_v4 import NodeV4  # reuse Node schema
from mcts.utils import (
    compute_improved_policy,
    completed_q_score,
    gumbel_top_k,
    normalize_q_values,
    sample_gumbel,
    sequential_halving_rounds,
)


MAX_PLAYERS = Config.MAX_PLAYERS


class GumbelMCTSv4Batched:
    """Batched variant of GumbelMCTSv4. Same external API.

    `batch_size` controls how many leaves are collected before each
    network forward. `batch_size=1` reverts to per-sim behaviour.
    """

    def __init__(
        self,
        network,
        device: torch.device,
        num_simulations: int = 32,
        m: int = 8,
        c_scale: float = 2.5,
        add_root_noise: bool = False,
        batch_size: Optional[int] = None,
    ):
        self.network = network
        self.device = device
        self.num_simulations = num_simulations
        self.m = m
        self.c_scale = c_scale
        self.add_root_noise = add_root_noise
        self.batch_size = batch_size if batch_size is not None else (
            getattr(Config, "MCTS_BATCH_LEAVES", 8)
        )

    @torch.no_grad()
    def search(
        self,
        env: GameEnv,
        root: Optional[NodeV4] = None,
    ) -> Tuple[int, np.ndarray, np.ndarray, NodeV4]:
        """Run MCTS from `env`. Returns (best_action, improved_policy, root_value_vec, root)."""
        self.network.eval()
        player = env.current_player
        num_players = getattr(env, "num_players", 2)

        # 1. Root initialization (single forward - small overhead)
        if root is None or root.state_env is None:
            root = NodeV4(state_env=env.clone(), player=player,
                          num_players=num_players)
            self._expand_and_evaluate_single(root)
        elif not root.is_expanded:
            root.state_env = env.clone()
            self._expand_and_evaluate_single(root)

        legal_actions = get_legal_actions(env.get_legal_mask(player))
        if not legal_actions:
            return 0, np.zeros(Config.ACTION_SPACE), np.zeros(MAX_PLAYERS), root
        if len(legal_actions) == 1:
            policy = np.zeros(Config.ACTION_SPACE)
            policy[legal_actions[0]] = 1.0
            return legal_actions[0], policy, root.value_vec_sum.copy(), root

        priors = np.array([root.children[a].prior for a in legal_actions])
        log_priors_full = np.full(Config.ACTION_SPACE, -30.0)
        for i, a in enumerate(legal_actions):
            log_priors_full[a] = math.log(max(priors[i], 1e-8))

        # 2. Gumbel-Top-k root candidates
        actual_m = min(self.m, len(legal_actions))
        candidates = gumbel_top_k(log_priors_full, legal_actions, actual_m)
        gumbel_noise = {a: sample_gumbel(1)[0] for a in candidates}

        # 3. Sequential halving - but accumulate root_actions into a batch queue
        remaining = list(candidates)
        budget = self.num_simulations
        rounds = sequential_halving_rounds(actual_m)
        sims_per_round = max(actual_m, budget // max(1, rounds))

        for round_idx in range(rounds):
            if len(remaining) <= 1:
                break

            # Build the queue of root_actions to simulate this round
            sims_each = max(1, sims_per_round // len(remaining))
            extra = sims_per_round - sims_each * len(remaining)
            queue: List[int] = []
            for i, a in enumerate(remaining):
                count = sims_each + (1 if i < extra else 0)
                queue.extend([a] * count)

            # Process queue in batches
            self._run_simulation_queue(root, queue)

            # Eliminate bottom half by completed-Q + Gumbel noise
            scores = {}
            for a in remaining:
                child = root.children[a]
                q_a = child.q_for(player)
                scores[a] = (gumbel_noise.get(a, 0.0)
                             + log_priors_full[a]
                             + self.c_scale * q_a)
            remaining.sort(key=lambda a: scores[a], reverse=True)
            half = max(1, len(remaining) // 2)
            remaining = remaining[:half]

        best_action = remaining[0]

        # 4. Improved policy from visit counts
        visits = np.zeros(Config.ACTION_SPACE, dtype=np.float64)
        for a in legal_actions:
            visits[a] = root.children[a].visit_count
        if visits.sum() == 0:
            improved_policy = np.zeros(Config.ACTION_SPACE)
            for a in legal_actions:
                improved_policy[a] = 1.0 / len(legal_actions)
        else:
            improved_policy = visits / visits.sum()

        return (
            int(best_action), improved_policy,
            root.value_vec_sum / max(1, root.visit_count), root,
        )

    # ------------------------------------------------------------------
    # Batched-leaf search core
    # ------------------------------------------------------------------

    def _run_simulation_queue(self, root: NodeV4, queue: List[int]) -> None:
        """Process all simulations in `queue` in batches of size up to
        self.batch_size."""
        bs = max(1, self.batch_size)
        i = 0
        while i < len(queue):
            slice_ = queue[i : i + bs]
            self._simulate_batch(root, slice_)
            i += bs

    def _simulate_batch(self, root: NodeV4, root_actions: List[int]) -> None:
        """Collect leaves for each root_action, batch-evaluate, backup all."""
        # Each entry: (path, leaf_node, has_been_stepped, is_terminal, value_vec_or_None)
        descents: List[Tuple[List[Tuple[NodeV4, Optional[int]]], NodeV4]] = []
        for ra in root_actions:
            descent = self._descend_with_vloss(root, ra)
            if descent is None:
                continue
            descents.append(descent)

        if not descents:
            return

        # Separate terminal vs non-terminal (terminal needs no forward)
        nonterm_idx: List[int] = []
        terminal_value_vecs: Dict[int, np.ndarray] = {}
        nonterm_states = []
        nonterm_masks = []
        nonterm_seats = []

        for idx, (path, leaf) in enumerate(descents):
            if leaf.is_terminal:
                terminal_value_vecs[idx] = self._terminal_value_vec(leaf.state_env)
            else:
                nonterm_idx.append(idx)
                env = leaf.state_env
                pl = env.current_player
                nonterm_states.append(env.get_state_tensor(pl))
                nonterm_masks.append(env.get_legal_mask(pl))
                nonterm_seats.append(pl)

        # Batched forward for non-terminals
        nonterm_value_vecs: Dict[int, np.ndarray] = {}
        if nonterm_states:
            states_t = torch.stack(nonterm_states, dim=0).to(self.device)
            masks_t = torch.stack(nonterm_masks, dim=0).to(self.device)
            seats_t = torch.tensor(nonterm_seats, device=self.device)
            try:
                out = self.network(states_t, masks_t, current_seat=seats_t)
            except TypeError:
                out = self.network(states_t, masks_t)
            priors_batch = out["policy"].cpu().numpy()
            if "value_vec" in out:
                vv_batch = out["value_vec"].cpu().numpy()
            else:
                v_batch = self.network.aggregate_value(out["value"]).cpu().numpy()
                vv_batch = None

            for j, leaf_idx in enumerate(nonterm_idx):
                _path, leaf = descents[leaf_idx]
                env_l = leaf.state_env
                pl = env_l.current_player
                priors = priors_batch[j]
                value_vec = np.zeros(MAX_PLAYERS, dtype=np.float64)
                if vv_batch is not None:
                    vv = vv_batch[j]
                    value_vec[: len(vv)] = vv
                else:
                    value_vec[pl] = float(v_batch[j])
                # Expand the leaf
                legal = get_legal_actions(env_l.get_legal_mask(pl))
                if not legal:
                    leaf.is_terminal = True
                else:
                    action_priors = {a: float(priors[a]) for a in legal}
                    leaf.expand(legal, action_priors)
                nonterm_value_vecs[leaf_idx] = value_vec

        # Backup all descents (with vloss removal)
        for idx, (path, leaf) in enumerate(descents):
            if idx in terminal_value_vecs:
                value_vec = terminal_value_vecs[idx]
            else:
                value_vec = nonterm_value_vecs[idx]
            self._backup_with_vloss_removal(path, leaf, value_vec)

    def _descend_with_vloss(
        self, root: NodeV4, root_action: int,
    ) -> Optional[Tuple[List[Tuple[NodeV4, Optional[int]]], NodeV4]]:
        """Descend from root via root_action, applying virtual_loss along the
        way. Returns (path, leaf_node) where leaf is unexpanded-non-terminal,
        terminal, or already-fully-explored.

        path is the list of (node, action_taken_from_node). The leaf is
        included separately (not in path).

        Virtual loss has been applied to root and every internal node on
        the descent path. The caller must call `_backup_with_vloss_removal`
        to undo it.
        """
        # Apply v-loss to root
        root.add_virtual_loss()
        path: List[Tuple[NodeV4, Optional[int]]] = [(root, root_action)]

        # Step into root child
        node = root.children[root_action]

        # If this child is unexpanded and not terminal: leaf found
        if not node.is_expanded and not node.is_terminal:
            sim_env = root.state_env.clone()
            _, done = sim_env.step(root_action)
            node.state_env = sim_env
            if done:
                node.is_terminal = True
            node.add_virtual_loss()
            return path, node

        # Else descend deeper (possibly through already-expanded interior)
        node.add_virtual_loss()
        current = node
        while current.is_expanded and not current.is_terminal and current.children:
            best_action = self._select_child(current)
            path.append((current, best_action))
            next_node = current.children[best_action]

            if not next_node.is_expanded and not next_node.is_terminal:
                sim_env = current.state_env.clone()
                _, done = sim_env.step(best_action)
                next_node.state_env = sim_env
                if done:
                    next_node.is_terminal = True
                next_node.add_virtual_loss()
                return path, next_node

            current = next_node
            current.add_virtual_loss()

        # Reached a terminal-along-the-way (rare)
        return path, current

    def _backup_with_vloss_removal(
        self,
        path: List[Tuple[NodeV4, Optional[int]]],
        leaf: NodeV4,
        value_vec: np.ndarray,
    ) -> None:
        """Apply value_vec to every node on the path AND the leaf, while
        removing virtual loss applied during descent."""
        # path: list of (node, action). leaf is the final leaf node.
        for node, _ in path:
            node.value_vec_sum += value_vec
            node.visit_count += 1
            node.remove_virtual_loss()
        # Leaf gets backup + v-loss removal too
        leaf.value_vec_sum += value_vec
        leaf.visit_count += 1
        leaf.remove_virtual_loss()

    def _select_child(self, node: NodeV4) -> int:
        """Same as mcts_v4._select_child."""
        legal = list(node.children.keys())
        priors = {a: node.children[a].prior for a in legal}
        q_values = {a: node.children[a].q_for(node.player)
                    for a in legal if node.children[a].visit_count > 0}
        prior_arr = np.array([priors[a] for a in legal])
        prior_arr = prior_arr / max(1e-9, prior_arr.sum())
        q_norm = normalize_q_values(q_values, legal)
        log_p = np.log(prior_arr + 1e-9)
        c = self.c_scale
        visits = np.array([node.children[a].visit_count for a in legal], dtype=np.float64)
        total_visits = visits.sum()
        improved_logits = log_p + c * np.array([q_norm.get(a, 0.5) for a in legal])
        improved = np.exp(improved_logits - improved_logits.max())
        improved = improved / improved.sum()
        score = improved - visits / (1.0 + total_visits)
        return legal[int(np.argmax(score))]

    def _expand_and_evaluate_single(self, node: NodeV4) -> np.ndarray:
        """Single-leaf forward (for root expansion only)."""
        env = node.state_env
        player = env.current_player
        state_t = env.get_state_tensor(player).unsqueeze(0).to(self.device)
        mask_t = env.get_legal_mask(player).unsqueeze(0).to(self.device)
        seat_t = torch.tensor([player], device=self.device)
        try:
            out = self.network(state_t, mask_t, current_seat=seat_t)
        except TypeError:
            out = self.network(state_t, mask_t)
        priors = out["policy"][0].cpu().numpy()
        value_vec = np.zeros(MAX_PLAYERS, dtype=np.float64)
        if "value_vec" in out:
            vv = out["value_vec"][0].cpu().numpy()
            value_vec[: len(vv)] = vv
        else:
            v = self.network.aggregate_value(out["value"])[0].item()
            value_vec[player] = v
        legal = get_legal_actions(env.get_legal_mask(player))
        if not legal:
            node.is_terminal = True
            return value_vec
        action_priors = {a: float(priors[a]) for a in legal}
        node.expand(legal, action_priors)
        # Backup root expansion
        node.value_vec_sum += value_vec
        node.visit_count += 1
        return value_vec

    @staticmethod
    def _terminal_value_vec(env: GameEnv) -> np.ndarray:
        from core import teacher_score as ts
        vec = np.zeros(MAX_PLAYERS, dtype=np.float64)
        for p in range(env.num_players):
            vec[p] = ts.normalized_value_target(env.compute_final_score(p))
        return vec
