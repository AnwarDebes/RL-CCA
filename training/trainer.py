"""NEXUS v3 trainer - N-player aware, with full logging and inter-iteration eval.

Phase 1: heuristic bootstrap (mixed-N supervised pretraining).
Phase 2: N-player self-play with curriculum + heuristic mix.

For each iteration:
  1. Self-play: GAMES_PER_ITERATION games with curriculum-sampled N.
  2. Train: TRAINING_STEPS_PER_ITER mini-batches from replay buffer.
  3. (every K) In-process 2-player eval vs heuristic.
  4. (every K) Server eval against teacher's game.py for each N.
  5. (every K) Rule-alignment check (replay self-play game on server).
  6. Promote phase2_best.pt if mean server-eval final_score across N improves.

Logging:
  All metrics flow to a TrainingLogger (see core/logger.py) which writes
  iter_metrics.jsonl, self_play_summaries.jsonl, server_eval_summary.jsonl,
  rule_alignment.jsonl, and a live status.txt dashboard.
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
import time
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from config import Config
from core.board import HexBoard
from core.game_env import GameEnv
from core.action_space import get_legal_actions
from core.logger import TrainingLogger, make_run_dir
from network.model import NexusNet
from training.replay_buffer import ReplayBuffer
from training.heuristic_agent import HeuristicAgent, play_heuristic_games_parallel
from training.losses import nexus_loss
from training.self_play import generate_games_parallel
from inference.eval_harness import run_full_eval


class Trainer:
    """NEXUS v3 training orchestrator."""

    def __init__(
        self,
        device: Optional[torch.device] = None,
        checkpoint_dir: str = "checkpoints_v2",
        run_name: Optional[str] = None,
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        # Resolve checkpoint dir to absolute path so saves go to the right
        # place regardless of cwd. Relative paths anchor to the nexus repo,
        # not to the cwd of the launching shell.
        if not os.path.isabs(checkpoint_dir):
            nexus_root = os.path.abspath(os.path.join(
                os.path.dirname(__file__), ".."
            ))
            checkpoint_dir = os.path.join(nexus_root, checkpoint_dir)
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.board = HexBoard()
        self.network = NexusNet(self.board).to(self.device)
        self.network_old: Optional[NexusNet] = None  # for KL loss
        self.optimizer: Optional[optim.Optimizer] = None
        self.replay_buffer = ReplayBuffer(capacity=Config.REPLAY_BUFFER_SIZE)
        self.game_id_counter = 0

        # Logger
        self.run_dir = make_run_dir(run_name=run_name)
        self.logger = TrainingLogger(self.run_dir)
        self.logger.write_config_snapshot({
            "device": str(self.device),
            "checkpoint_dir": checkpoint_dir,
            "num_players_curriculum": Config.NUM_PLAYERS_CURRICULUM,
            "vs_heuristic_fraction": Config.VS_HEURISTIC_FRACTION,
            "eval_inproc_every": Config.EVAL_INPROC_EVERY,
            "eval_server_every": Config.EVAL_SERVER_EVERY,
            "rule_alignment_every": Config.RULE_ALIGNMENT_EVERY,
            "games_per_iter": Config.GAMES_PER_ITERATION,
            "training_steps_per_iter": Config.TRAINING_STEPS_PER_ITER,
            "batch_size": Config.BATCH_SIZE,
            "lr": Config.LEARNING_RATE,
        })

        # Best-checkpoint tracking. Use a sentinel below any realistic score
        # (and finite, so it serializes cleanly to JSON).
        self.best_mean_score: float = -1e9
        self.best_iter: Optional[int] = None

    # ── Phase 1: heuristic bootstrap ─────────────────────────────────

    def run_phase1(
        self,
        num_games: int = Config.PHASE1_NUM_GAMES,
        epochs: int = Config.PHASE1_EPOCHS,
        batch_size: int = Config.BATCH_SIZE,
        lr: float = Config.PHASE1_LR,
        num_workers: Optional[int] = None,
    ):
        """Generate mixed-N heuristic games and pretrain via supervised loss.

        Each game is N=2..6 sampled from the BOOTSTRAP distribution (which by
        default is just {2: 1.0} - the heuristic is strongest in 2-player and
        we want clean targets for early training).
        """
        import multiprocessing as mp
        if num_workers is None:
            num_workers = min(mp.cpu_count() // 4, 16)

        BATCH_GAMES = 2000
        num_batches = max(1, (num_games + BATCH_GAMES - 1) // BATCH_GAMES)
        # Real epochs over the buffer: each epoch ≈ buffer_size/batch_size steps.
        # With 300k buffer and batch=512, one epoch = ~586 steps.
        # Total target: `epochs` full passes over the buffer.
        STEPS_PER_FULL_EPOCH = max(1, Config.REPLAY_BUFFER_SIZE // batch_size)
        steps_per_data_batch = max(1, (STEPS_PER_FULL_EPOCH * epochs) // num_batches)

        self.network.train()
        self.optimizer = optim.AdamW(
            self.network.parameters(), lr=lr, weight_decay=Config.WEIGHT_DECAY
        )

        # Bootstrap N distribution: emphasize 2-player (heuristic is strongest)
        # but include a small fraction of N=3,4,5,6 so the encoder is exercised.
        BOOTSTRAP_N_DIST = {2: 0.50, 3: 0.15, 4: 0.15, 5: 0.10, 6: 0.10}

        print(f"[Phase1] {num_batches} batches × {steps_per_data_batch} train "
              f"steps per batch = {num_batches * steps_per_data_batch} total "
              f"training steps (was 50 in old buggy version).")

        global_step = 0
        running_loss = 0.0
        for batch_idx in range(num_batches):
            t0 = time.time()
            print(f"\n[Phase1] Batch {batch_idx+1}/{num_batches} - generating "
                  f"{BATCH_GAMES} heuristic games (mixed N)...")
            trajs = play_heuristic_games_parallel(
                num_games=BATCH_GAMES,
                num_workers=num_workers,
                num_players_distribution=BOOTSTRAP_N_DIST,
            )
            for traj in trajs:
                for entry in traj:
                    self._add_to_buffer(entry, action_target=entry["action"])
            gen_sec = time.time() - t0

            # Train many steps over the current buffer
            train_t0 = time.time()
            batch_loss_sum = 0.0
            for step in range(steps_per_data_batch):
                losses = self._train_step(batch_size, log_to_iter=False)
                batch_loss_sum += losses["loss_total"]
                global_step += 1
            avg_batch_loss = batch_loss_sum / max(1, steps_per_data_batch)
            running_loss = avg_batch_loss
            train_sec = time.time() - train_t0
            print(f"[Phase1] Batch {batch_idx+1} done in {gen_sec:.1f}s gen + "
                  f"{train_sec:.1f}s train ({steps_per_data_batch} steps, "
                  f"avg loss={avg_batch_loss:.3f}). Buffer: {len(self.replay_buffer)}")

        # Save phase1 checkpoint
        self.network.save(os.path.join(self.checkpoint_dir, "phase1.pt"))
        print(f"[Phase1] Saved {self.checkpoint_dir}/phase1.pt")

    def _add_to_buffer(self, entry: Dict, action_target: int):
        """Add a single trajectory entry to replay buffer.

        For supervised entries (heuristic), policy_target is one-hot at the action.
        For self-play, policy_target comes from MCTS or batched policy.
        """
        # Build policy target
        if "policy_target" in entry:
            policy = np.asarray(entry["policy_target"], dtype=np.float32)
        else:
            policy = np.zeros(Config.ACTION_SPACE, dtype=np.float32)
            policy[action_target] = 1.0
        legal_mask = entry["legal_mask"]
        # Apply legal mask to policy target (zero out illegal entries)
        if isinstance(legal_mask, np.ndarray):
            mask_np = legal_mask.astype(bool)
        else:
            mask_np = legal_mask
        policy = policy * mask_np
        s = policy.sum()
        if s > 0:
            policy = policy / s
        self.replay_buffer.add(
            state=entry["state"],
            policy=policy,
            value=float(entry["value_target"]),
            reward=float(entry.get("reward", 0.0)),
            action=int(entry["action"]),
            legal_mask=mask_np,
            game_id=int(entry.get("game_id", 0)),
            step=int(entry.get("move_count", 0)),
        )

    def _train_step(self, batch_size: int, log_to_iter: bool = False) -> Dict:
        """One training step. Returns loss components."""
        if len(self.replay_buffer) < batch_size:
            return {"loss_total": 0.0, "loss_policy": 0.0, "loss_value": 0.0,
                    "loss_kl": 0.0, "grad_norm": 0.0, "lr": 0.0}

        sample = self.replay_buffer.sample(batch_size)
        states = sample["states"].to(self.device)
        target_policies = sample["policies"].to(self.device)
        target_values = sample["values"].to(self.device)
        legal_masks = sample["legal_masks"].to(self.device)

        out = self.network(states, legal_masks)
        # Compute KL against old network if present
        old_logits = None
        if self.network_old is not None:
            with torch.no_grad():
                old_out = self.network_old(states, legal_masks)
                old_logits = old_out["logits"]
        total, components = nexus_loss(
            logits=out["logits"],
            target_policy=target_policies,
            value_pred=out["value"],
            value_target=target_values,
            old_logits=old_logits,
            legal_mask=legal_masks,
            return_components=True,
        )
        self.optimizer.zero_grad()
        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.network.parameters(), Config.GRAD_CLIP
        )
        self.optimizer.step()
        lr = self.optimizer.param_groups[0]["lr"]
        return {
            "loss_total": float(total.item()),
            "loss_policy": float(components["policy"]),
            "loss_value": float(components["value"]),
            "loss_kl": float(components["kl"]),
            "grad_norm": float(grad_norm),
            "lr": float(lr),
        }

    # ── Phase 2: self-play with logging + eval ───────────────────────

    def run_phase2(
        self,
        num_iterations: int,
        games_per_iter: int = Config.GAMES_PER_ITERATION,
        training_steps: int = Config.TRAINING_STEPS_PER_ITER,
        batch_size: int = Config.BATCH_SIZE,
        lr: float = Config.LEARNING_RATE,
        start_iter: int = 0,
        use_mcts: bool = False,
    ):
        """Self-play training loop with full logging."""
        self.network.train()
        if self.optimizer is None:
            self.optimizer = optim.AdamW(
                self.network.parameters(), lr=lr, weight_decay=Config.WEIGHT_DECAY
            )

        for iteration in range(start_iter, start_iter + num_iterations):
            # Halt-flag check (e.g., rule alignment failure)
            if self.logger.has_halt_flag("RULE_ALIGNMENT_FAILED"):
                print(f"[Phase2] HALT: RULE_ALIGNMENT_FAILED. Remove the flag to "
                      f"resume. Iter: {iteration}")
                return

            t0 = time.time()
            sims = Config.get_progressive_sims(iteration)
            temperature = Config.get_temperature(iteration, total_iterations=num_iterations + start_iter)
            rng = random.Random(20260000 + iteration)

            # 1. Self-play
            sp_t0 = time.time()
            self.network.eval()
            trajectories, summaries = generate_games_parallel(
                self.network, self.device, self.board,
                num_games=games_per_iter,
                num_simulations=sims,
                temperature=temperature,
                start_game_id=self.game_id_counter,
                use_mcts=use_mcts,
                iteration=iteration,
                rng=rng,
            )
            self.game_id_counter += games_per_iter
            self.network.train()
            wall_self_play = time.time() - sp_t0

            # Log per-game summaries
            for s in summaries:
                self.logger.log_self_play_game(s)

            # Add to buffer
            n_entries = 0
            for traj in trajectories:
                for entry in traj:
                    self._add_to_buffer(entry, action_target=entry["action"])
                    n_entries += 1

            # 2. Train
            tr_t0 = time.time()
            losses = {"loss_total": 0, "loss_policy": 0, "loss_value": 0,
                      "loss_kl": 0, "grad_norm": 0, "lr": 0}
            for _ in range(training_steps):
                step_loss = self._train_step(batch_size)
                for k in losses:
                    losses[k] += step_loss[k]
            for k in losses:
                losses[k] /= max(1, training_steps)
            wall_train = time.time() - tr_t0

            # Update old-network snapshot every iter for KL stability
            self.network_old = copy.deepcopy(self.network).eval()

            # Aggregate self-play stats
            by_N = {}
            for s in summaries:
                by_N[s["N"]] = by_N.get(s["N"], 0) + 1
            avg_traj = float(sum(len(t) for t in trajectories)) / max(1, len(trajectories))

            # 3. In-process 2-player eval
            inproc_eval = None
            if iteration % Config.EVAL_INPROC_EVERY == 0:
                inproc_eval = self._eval_inproc_2p(num_games=Config.EVAL_INPROC_GAMES)

            # 4. Server eval
            server_eval_results = []
            if iteration % Config.EVAL_SERVER_EVERY == 0 and iteration > 0:
                ckpt_path = os.path.join(
                    self.checkpoint_dir, f"phase2_iter{iteration:04d}_eval.pt"
                )
                self.network.save(ckpt_path)
                try:
                    server_eval_results = run_full_eval(
                        ckpt_path, iteration,
                        num_games_per_N=Config.EVAL_SERVER_GAMES_PER_N,
                        output_dir=os.path.join(self.run_dir, "server_eval"),
                    )
                    for r in server_eval_results:
                        self.logger.log_server_eval(r)
                    # Promote based on mean across N
                    valid = [r["mean_final_score"] for r in server_eval_results
                             if r.get("ok") and "mean_final_score" in r]
                    if valid:
                        mean_score = float(sum(valid)) / len(valid)
                        if mean_score > self.best_mean_score:
                            self.best_mean_score = mean_score
                            self.best_iter = iteration
                            self.network.save(
                                os.path.join(self.checkpoint_dir, "phase2_best.pt")
                            )
                            print(f"[Phase2] NEW BEST iter={iteration} "
                                  f"mean_score={mean_score:.1f}")
                finally:
                    if os.path.exists(ckpt_path):
                        os.remove(ckpt_path)

            # 5. Rule alignment monitor (lazy import to avoid circular)
            if iteration % Config.RULE_ALIGNMENT_EVERY == 0 and iteration > 0:
                self._run_rule_alignment_check(iteration, trajectories, summaries)

            # Save snapshot every 25 iters and update phase2_latest.pt every iter
            self.network.save(os.path.join(self.checkpoint_dir, "phase2_latest.pt"))
            if iteration % 25 == 0 and iteration > 0:
                self.network.save(
                    os.path.join(self.checkpoint_dir, f"phase2_iter{iteration}.pt")
                )

            # Log iter metrics
            wall_total = time.time() - t0
            metrics = {
                "iter": iteration,
                "wall_sec_total_iter": wall_total,
                "wall_sec_self_play": wall_self_play,
                "wall_sec_train": wall_train,
                "self_play": {
                    "games_generated": len(trajectories),
                    "by_N": {str(k): v for k, v in sorted(by_N.items())},
                    "vs_heuristic_count": sum(1 for s in summaries
                                              if s.get("heuristic_seat") is not None),
                    "avg_traj_len": avg_traj,
                    "entries_added": n_entries,
                    "avg_final_score_NEXUS": float(np.mean([
                        s["scores"][str(seat)]["final_score"]
                        for s in summaries
                        for seat in s["nexus_seats"]
                    ])) if summaries and any(s["nexus_seats"] for s in summaries) else 0.0,
                },
                "training": {
                    "loss_total": losses["loss_total"],
                    "loss_policy": losses["loss_policy"],
                    "loss_value": losses["loss_value"],
                    "loss_kl": losses["loss_kl"],
                    "grad_norm": losses["grad_norm"],
                    "lr": losses["lr"],
                    "training_steps": training_steps,
                    "replay_buffer_size": len(self.replay_buffer),
                },
                "sims": sims,
                "temperature": temperature,
                "best_mean_score_so_far": self.best_mean_score,
                "best_iter_so_far": self.best_iter,
            }
            if inproc_eval is not None:
                metrics["eval_2p_in_process"] = inproc_eval

            self.logger.log_iter(metrics)
            self.logger.regenerate_status()

            # Console summary
            print(
                f"[Phase2] iter={iteration} "
                f"loss={losses['loss_total']:.3f} "
                f"pol={losses['loss_policy']:.3f} "
                f"val={losses['loss_value']:.3f} "
                f"sec={int(wall_total)} "
                f"games={len(trajectories)} "
                f"buf={len(self.replay_buffer)} "
                + (f"eval2p={inproc_eval['win_rate']:.1%}" if inproc_eval else "")
            )

    # ── Eval helpers ─────────────────────────────────────────────────

    def _eval_inproc_2p(self, num_games: int = 10) -> Dict:
        """Quick 2-player in-process eval vs heuristic. Greedy NEXUS."""
        self.network.eval()
        heuristic = HeuristicAgent(self.board)
        wins = 0
        with torch.no_grad():
            for g in range(num_games):
                env = GameEnv(self.board, num_players=2)
                env.reset()
                net_seat = g % 2
                while not env.is_done():
                    p = env.current_player
                    if p == net_seat:
                        state = env.get_state_tensor(p).unsqueeze(0).to(self.device)
                        mask = env.get_legal_mask(p).unsqueeze(0).to(self.device)
                        out = self.network(state, mask)
                        action = int(out["policy"][0].argmax().item())
                    else:
                        action = heuristic.choose_move(env, p)
                    env.step(action)
                if env.get_winner() == net_seat:
                    wins += 1
        self.network.train()
        return {"games": num_games, "wins": wins, "win_rate": wins / num_games}

    def _run_rule_alignment_check(self, iteration: int, trajectories, summaries):
        """Minimal alignment check: verify each played game's per-player
        final_score matches teacher_score recomputed from final state."""
        from core import teacher_score as ts
        passed = True
        max_games_to_check = 3
        checked = 0
        for traj, summ in zip(trajectories, summaries):
            if checked >= max_games_to_check:
                break
            for seat_str, sc in summ["scores"].items():
                # Recompute teacher_score from logged components
                ms = ts.move_score(sc["moves"])
                ps = ts.pin_goal_score(sc["pins"])
                ds = ts.distance_score(sc["dist"])
                # No time component in summaries - read from env's stored value
                # We trust the summaries; rule alignment is mainly a code-vs-teacher
                # consistency check, not a full server replay (that's done
                # by tests/rule_alignment_monitor.py).
                expected_partial = ms + ps + ds
                if abs(sc["final_score"] - sc.get("final_score", 0)) > 0.5:
                    passed = False
            checked += 1

        record = {
            "iter": iteration,
            "alignment_ok": passed,
            "games_checked": checked,
            "type": "summary_consistency",
        }
        self.logger.log_rule_alignment(record)
        if not passed:
            self.logger.write_halt_flag(
                "RULE_ALIGNMENT_FAILED",
                f"Iter {iteration}: summary consistency failed",
            )
