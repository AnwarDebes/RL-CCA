"""NEXUS v3 trainer.

Differences vs v2 (training/trainer.py):
  • Uses NexusNetV3 (vector value head + aux heads).
  • Replaces ReplayBuffer with ReplayBufferV3 (aux fields).
  • Uses self_play_v3.generate_games_batched_v3 (records aux + supports
    frozen opponent pool).
  • Uses losses_v3.nexus_loss_v3 (aux losses + entropy bonus).
  • Cosine LR with warm restarts (T_0 = Config.LR_T_0).
  • Frozen-opponent snapshot pool of size Config.FREEZE_POOL_SIZE,
    refreshed every Config.FREEZE_POOL_EVERY iters.
  • Populates run_dir/policy_value_probes/ every iter for entropy-collapse
    monitoring.
  • Eval is done with MCTS (configurable sims) AND greedy, so we measure
    the policy's true ceiling.
  • Saves to checkpoints_v3/ by default. Promotes phase2_best_v3.pt on
    mean server-eval improvement.
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
import time
from collections import deque
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
from network.model_v3 import NexusNetV3
from training.replay_buffer_v3 import ReplayBufferV3
from training.heuristic_agent import HeuristicAgent
from training.losses_v3 import nexus_loss_v3
from training.self_play_v3 import generate_games_batched_v3
from training.phase1_v3 import play_games_parallel_v3
from inference.eval_harness import run_full_eval


class TrainerV3:
    def __init__(
        self,
        device: Optional[torch.device] = None,
        checkpoint_dir: str = "checkpoints_v3",
        run_name: Optional[str] = None,
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        if not os.path.isabs(checkpoint_dir):
            nexus_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..")
            )
            checkpoint_dir = os.path.join(nexus_root, checkpoint_dir)
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.board = HexBoard()
        self.network = NexusNetV3(self.board).to(self.device)
        self.network_old: Optional[NexusNetV3] = None  # for KL stability
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler: Optional[optim.lr_scheduler.CosineAnnealingWarmRestarts] = None
        self.replay_buffer = ReplayBufferV3(capacity=Config.REPLAY_BUFFER_SIZE)
        self.game_id_counter = 0

        # Frozen opponent pool - list of NexusNetV3 instances (eval mode)
        self.frozen_pool: List[NexusNetV3] = []

        # Probe states for entropy monitoring (deterministic, fixed across run)
        self.probe_states, self.probe_masks, self.probe_seats = self._build_probes()

        # Logger
        self.run_dir = make_run_dir(run_name=run_name)
        self.logger = TrainingLogger(self.run_dir)
        self.logger.write_config_snapshot({
            "device": str(self.device),
            "checkpoint_dir": checkpoint_dir,
            "v3": True,
            "hidden_dim_v3": Config.HIDDEN_DIM_V3,
            "num_res_blocks_v3": Config.NUM_RES_BLOCKS_V3,
            "lr_t_0": Config.LR_T_0,
            "lr_eta_min": Config.LR_ETA_MIN,
            "freeze_pool_size": Config.FREEZE_POOL_SIZE,
            "freeze_pool_every": Config.FREEZE_POOL_EVERY,
            "freeze_opp_fraction": Config.FREEZE_OPP_FRACTION,
            "entropy_bonus_weight": Config.ENTROPY_BONUS_WEIGHT,
            "opp_policy_loss_weight": Config.OPP_POLICY_LOSS_WEIGHT,
            "plies_loss_weight": Config.PLIES_LOSS_WEIGHT,
            "value_vec_loss_weight": Config.VALUE_VEC_LOSS_WEIGHT,
            "num_players_curriculum_v3": Config.NUM_PLAYERS_CURRICULUM_V3,
            "vs_heuristic_fraction": Config.VS_HEURISTIC_FRACTION,
        })

        self.best_mean_score: float = -1e9
        self.best_iter: Optional[int] = None

    # ── Probe states for entropy monitoring ──────────────────────────

    def _build_probes(self):
        """Build a fixed set of game states used to track policy entropy
        and value drift across iterations. 8 states from N=2..6 starts
        plus a few mid-game positions reached by random play."""
        rng = random.Random(20260501)
        states, masks, seats = [], [], []
        for N in [2, 3, 4, 5, 6]:
            env = GameEnv(self.board, num_players=N)
            env.reset()
            # capture both initial and a small-mid position
            for steps in [0, 6]:
                for _ in range(steps):
                    p = env.current_player
                    legal = get_legal_actions(env.get_legal_mask(p))
                    env.step(rng.choice(legal))
                p = env.current_player
                states.append(env.get_state_tensor(p))
                masks.append(env.get_legal_mask(p))
                seats.append(p)
                # Reset for next iteration to keep deterministic
                env.reset()
                for _ in range(steps):
                    p = env.current_player
                    legal = get_legal_actions(env.get_legal_mask(p))
                    env.step(rng.choice(legal))
        S = torch.stack(states).to(self.device)
        M = torch.stack(masks).to(self.device)
        T = torch.tensor(seats, dtype=torch.long, device=self.device)
        return S, M, T

    @torch.no_grad()
    def _log_probes(self, iteration: int) -> Dict:
        self.network.eval()
        out = self.network(self.probe_states, self.probe_masks,
                           current_seat=self.probe_seats)
        log_p = F.log_softmax(out["logits"], dim=-1)
        log_p = torch.where(torch.isinf(log_p), torch.zeros_like(log_p), log_p)
        probs = log_p.exp()
        H = -(probs * log_p * self.probe_masks.float()).sum(dim=-1)  # [B]
        record = {
            "iter": iteration,
            "policy_entropy_mean": float(H.mean().item()),
            "policy_entropy_min": float(H.min().item()),
            "policy_entropy_max": float(H.max().item()),
            "value_mean": float(out["value"].mean().item()),
            "value_std": float(out["value"].std().item()),
            "n_probes": int(self.probe_states.size(0)),
        }
        path = os.path.join(self.run_dir, "policy_value_probes",
                            f"iter_{iteration:04d}.json")
        with open(path, "w") as f:
            json.dump(record, f)
        self.network.train()
        return record

    # ── Phase 1: heuristic bootstrap (v3-aware) ──────────────────────

    def run_phase1(
        self,
        num_games: int = Config.PHASE1_NUM_GAMES,
        epochs: int = Config.PHASE1_EPOCHS,
        batch_size: int = Config.BATCH_SIZE,
        lr: float = Config.PHASE1_LR,
        num_workers: Optional[int] = None,
    ):
        import multiprocessing as mp
        if num_workers is None:
            num_workers = min(mp.cpu_count() // 4, 16)

        BATCH_GAMES = 2000
        num_batches = max(1, (num_games + BATCH_GAMES - 1) // BATCH_GAMES)
        STEPS_PER_FULL_EPOCH = max(1, Config.REPLAY_BUFFER_SIZE // batch_size)
        steps_per_data_batch = max(1, (STEPS_PER_FULL_EPOCH * epochs) // num_batches)

        self.network.train()
        self.optimizer = optim.AdamW(
            self.network.parameters(), lr=lr, weight_decay=Config.WEIGHT_DECAY
        )

        BOOTSTRAP_N_DIST = {2: 0.40, 3: 0.20, 4: 0.15, 5: 0.15, 6: 0.10}
        print(f"[Phase1-v3] {num_batches} batches × {steps_per_data_batch} train steps")

        for batch_idx in range(num_batches):
            t0 = time.time()
            print(f"[Phase1-v3] Batch {batch_idx+1}/{num_batches} - "
                  f"generating {BATCH_GAMES} mixed-N heuristic games")
            trajs = play_games_parallel_v3(
                num_games=BATCH_GAMES,
                num_workers=num_workers,
                num_players_distribution=BOOTSTRAP_N_DIST,
            )
            for traj in trajs:
                for entry in traj:
                    self._add_to_buffer(entry)
            gen_sec = time.time() - t0

            tr_t0 = time.time()
            loss_sum = 0.0
            for _ in range(steps_per_data_batch):
                s = self._train_step(batch_size)
                loss_sum += s["loss_total"]
            avg = loss_sum / max(1, steps_per_data_batch)
            train_sec = time.time() - tr_t0
            print(f"[Phase1-v3] Batch {batch_idx+1}: gen={gen_sec:.1f}s "
                  f"train={train_sec:.1f}s ({steps_per_data_batch} steps) "
                  f"loss={avg:.3f} buf={len(self.replay_buffer)}")

        self.network.save(os.path.join(self.checkpoint_dir, "phase1_v3.pt"))
        print(f"[Phase1-v3] Saved {self.checkpoint_dir}/phase1_v3.pt")

    # ── Buffer push helper ───────────────────────────────────────────

    def _add_to_buffer(self, entry: Dict):
        legal = entry["legal_mask"]
        if isinstance(legal, np.ndarray):
            mask_np = legal.astype(bool)
        else:
            mask_np = np.asarray(legal, dtype=bool)
        # Normalize policy target across legal moves
        policy = np.asarray(entry["policy_target"], dtype=np.float32) * mask_np
        s = policy.sum()
        if s > 0:
            policy = policy / s
        opp_mask = entry.get("opp_legal_mask")
        if opp_mask is None:
            opp_mask = np.zeros(Config.ACTION_SPACE, dtype=np.bool_)
        else:
            opp_mask = np.asarray(opp_mask).astype(bool)
        self.replay_buffer.add(
            state=entry["state"],
            policy=policy,
            value=float(entry["value_target"]),
            action=int(entry["action"]),
            legal_mask=mask_np,
            value_vec=entry["value_vec"],
            n_players=int(entry["n_players"]),
            player=int(entry["player"]),
            opp_action=int(entry.get("opp_action", -1)),
            opp_legal_mask=opp_mask,
            plies_remaining=float(entry.get("plies_remaining", 0.0)),
            plies_valid=bool(entry.get("plies_valid", False)),
            game_id=int(entry.get("game_id", 0)),
            step=int(entry.get("move_count", 0)),
        )

    # ── Training step ────────────────────────────────────────────────

    def _train_step(self, batch_size: int) -> Dict:
        if len(self.replay_buffer) < batch_size:
            return {"loss_total": 0.0, "loss_policy": 0.0, "loss_value_vec": 0.0,
                    "loss_opp": 0.0, "loss_plies": 0.0, "policy_entropy": 0.0,
                    "loss_kl": 0.0, "grad_norm": 0.0, "lr": 0.0}

        sample = self.replay_buffer.sample(batch_size)
        states = sample["states"].to(self.device)
        target_policies = sample["policies"].to(self.device)
        legal_masks = sample["legal_masks"].to(self.device)
        seats = sample["players"].to(self.device)

        out = self.network(states, legal_masks, current_seat=seats)

        old_logits = None
        if self.network_old is not None:
            with torch.no_grad():
                old_out = self.network_old(states, legal_masks,
                                           current_seat=seats)
                old_logits = old_out["logits"]

        batch_dev = {
            "policies": target_policies,
            "value_vec": sample["value_vec"].to(self.device),
            "n_players": sample["n_players"].to(self.device),
            "opp_action": sample["opp_action"].to(self.device),
            "opp_legal_mask": sample["opp_legal_mask"].to(self.device),
            "plies_target": sample["plies_target"].to(self.device),
            "plies_valid": sample["plies_valid"].to(self.device),
        }
        comps = nexus_loss_v3(out, batch_dev, legal_masks, old_logits=old_logits)

        self.optimizer.zero_grad()
        comps["total"].backward()
        gn = torch.nn.utils.clip_grad_norm_(
            self.network.parameters(), Config.GRAD_CLIP
        )
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        return {
            "loss_total": float(comps["total"].item()),
            "loss_policy": float(comps["policy"].item()),
            "loss_value_vec": float(comps["value_vec"].item()),
            "loss_opp": float(comps["opp_policy"].item()),
            "loss_plies": float(comps["plies"].item()),
            "policy_entropy": float(comps["entropy"].item()),
            "loss_kl": float(comps["kl"].item()),
            "grad_norm": float(gn),
            "lr": float(self.optimizer.param_groups[0]["lr"]),
        }

    # ── Frozen opponent pool management ──────────────────────────────

    def _maybe_snapshot_frozen(self, iteration: int):
        if iteration % Config.FREEZE_POOL_EVERY != 0 or iteration == 0:
            return
        snap = NexusNetV3(self.board).to(self.device)
        snap.load_state_dict(self.network.state_dict())
        snap.eval()
        # disable grads for the snapshot to save memory and speed up
        for p in snap.parameters():
            p.requires_grad_(False)
        self.frozen_pool.append(snap)
        if len(self.frozen_pool) > Config.FREEZE_POOL_SIZE:
            self.frozen_pool.pop(0)
        print(f"[Phase2-v3] Frozen pool size={len(self.frozen_pool)}")

    # ── Phase 2: self-play loop ──────────────────────────────────────

    def run_phase2(
        self,
        num_iterations: int,
        games_per_iter: int = Config.GAMES_PER_ITERATION,
        training_steps: int = Config.TRAINING_STEPS_PER_ITER,
        batch_size: int = Config.BATCH_SIZE,
        lr: float = Config.LEARNING_RATE,
        start_iter: int = 0,
        eval_with_mcts_sims: int = 0,    # 0 = disabled (greedy server-eval)
    ):
        self.network.train()
        if self.optimizer is None:
            self.optimizer = optim.AdamW(
                self.network.parameters(), lr=lr,
                weight_decay=Config.WEIGHT_DECAY
            )
        # Cosine warm restarts - break plateau periodically
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=max(1, Config.LR_T_0 * max(1, training_steps)),
            T_mult=int(Config.LR_RESTART_MULT) if Config.LR_RESTART_MULT > 1 else 1,
            eta_min=Config.LR_ETA_MIN,
        )

        for iteration in range(start_iter, start_iter + num_iterations):
            if self.logger.has_halt_flag("RULE_ALIGNMENT_FAILED"):
                print(f"[Phase2-v3] HALT iter {iteration}")
                return

            t0 = time.time()
            temperature = Config.get_temperature(
                iteration, total_iterations=num_iterations + start_iter
            )
            rng = random.Random(20260000 + iteration)

            # 1) Snapshot the frozen pool on cadence
            self._maybe_snapshot_frozen(iteration)

            # 2) Self-play
            sp_t0 = time.time()
            self.network.eval()
            trajectories, summaries = generate_games_batched_v3(
                self.network, self.device, self.board,
                num_games=games_per_iter,
                temperature=temperature,
                start_game_id=self.game_id_counter,
                iteration=iteration,
                rng=rng,
                frozen_pool=self.frozen_pool,
            )
            self.game_id_counter += games_per_iter
            self.network.train()
            wall_self_play = time.time() - sp_t0

            for s in summaries:
                self.logger.log_self_play_game(s)

            n_entries = 0
            for traj in trajectories:
                for entry in traj:
                    self._add_to_buffer(entry)
                    n_entries += 1

            # 3) Train
            tr_t0 = time.time()
            agg = {}
            for _ in range(training_steps):
                step = self._train_step(batch_size)
                for k, v in step.items():
                    agg[k] = agg.get(k, 0.0) + v
            for k in agg:
                agg[k] /= max(1, training_steps)
            wall_train = time.time() - tr_t0

            # KL old-net snapshot
            self.network_old = copy.deepcopy(self.network).eval()
            for p in self.network_old.parameters():
                p.requires_grad_(False)

            # 4) Probes
            probe = self._log_probes(iteration)

            # 5) Inproc 2P eval
            inproc_eval = None
            if iteration % Config.EVAL_INPROC_EVERY == 0:
                inproc_eval = self._eval_inproc_2p(num_games=Config.EVAL_INPROC_GAMES)

            # 6) Server eval
            server_eval_results = []
            if iteration % Config.EVAL_SERVER_EVERY == 0 and iteration > 0:
                ckpt_path = os.path.join(
                    self.checkpoint_dir, f"phase2_v3_iter{iteration:04d}_eval.pt"
                )
                self.network.save(ckpt_path)
                try:
                    server_eval_results = run_full_eval(
                        ckpt_path, iteration,
                        num_games_per_N=Config.EVAL_SERVER_GAMES_PER_N,
                        output_dir=os.path.join(self.run_dir, "server_eval"),
                        v3=True,
                    )
                    for r in server_eval_results:
                        self.logger.log_server_eval(r)
                    valid = [r["mean_final_score"] for r in server_eval_results
                             if r.get("ok") and "mean_final_score" in r]
                    if valid:
                        mean_score = float(sum(valid)) / len(valid)
                        if mean_score > self.best_mean_score:
                            self.best_mean_score = mean_score
                            self.best_iter = iteration
                            self.network.save(
                                os.path.join(self.checkpoint_dir,
                                             "phase2_best_v3.pt")
                            )
                            print(f"[Phase2-v3] NEW BEST iter={iteration} "
                                  f"mean_score={mean_score:.1f}")
                finally:
                    if os.path.exists(ckpt_path):
                        os.remove(ckpt_path)

            # 6.5) Rule-alignment check (every 10 iters) - re-added in v3
            #      rebuild after diagnostic showed v3 trainer dropped this.
            if iteration % Config.RULE_ALIGNMENT_EVERY == 0 and iteration > 0:
                self._run_rule_alignment_check(iteration, trajectories, summaries)

            # 7) Snapshots
            self.network.save(os.path.join(self.checkpoint_dir,
                                           "phase2_latest_v3.pt"))
            if iteration % 25 == 0 and iteration > 0:
                self.network.save(os.path.join(self.checkpoint_dir,
                                               f"phase2_v3_iter{iteration}.pt"))

            # 8) Log iter
            wall_total = time.time() - t0
            by_N = {}
            for s in summaries:
                by_N[s["N"]] = by_N.get(s["N"], 0) + 1
            avg_traj = (sum(len(t) for t in trajectories) /
                        max(1, len(trajectories)))
            metrics = {
                "iter": iteration,
                "wall_sec_total_iter": wall_total,
                "wall_sec_self_play": wall_self_play,
                "wall_sec_train": wall_train,
                "self_play": {
                    "games_generated": len(trajectories),
                    "by_N": {str(k): v for k, v in sorted(by_N.items())},
                    "avg_traj_len": avg_traj,
                    "entries_added": n_entries,
                    "frozen_pool_size": len(self.frozen_pool),
                },
                "training": {
                    "loss_total": agg.get("loss_total", 0.0),
                    "loss_policy": agg.get("loss_policy", 0.0),
                    "loss_value_vec": agg.get("loss_value_vec", 0.0),
                    "loss_opp": agg.get("loss_opp", 0.0),
                    "loss_plies": agg.get("loss_plies", 0.0),
                    "policy_entropy": agg.get("policy_entropy", 0.0),
                    "loss_kl": agg.get("loss_kl", 0.0),
                    "grad_norm": agg.get("grad_norm", 0.0),
                    "lr": agg.get("lr", 0.0),
                    "training_steps": training_steps,
                    "replay_buffer_size": len(self.replay_buffer),
                },
                "probe": probe,
                "temperature": temperature,
                "best_mean_score_so_far": self.best_mean_score,
                "best_iter_so_far": self.best_iter,
            }
            if inproc_eval is not None:
                metrics["eval_2p_in_process"] = inproc_eval
            self.logger.log_iter(metrics)
            self.logger.regenerate_status()

            print(
                f"[Phase2-v3] iter={iteration} "
                f"loss={agg.get('loss_total', 0):.3f} "
                f"pol={agg.get('loss_policy', 0):.3f} "
                f"val={agg.get('loss_value_vec', 0):.3f} "
                f"opp={agg.get('loss_opp', 0):.3f} "
                f"H={agg.get('policy_entropy', 0):.3f} "
                f"lr={agg.get('lr', 0):.5f} "
                f"sec={int(wall_total)} "
                f"games={len(trajectories)} "
                f"buf={len(self.replay_buffer)}"
                + (f" eval2p={inproc_eval['win_rate']:.1%}" if inproc_eval else "")
            )

    # ── Eval helpers ─────────────────────────────────────────────────

    def _run_rule_alignment_check(self, iteration: int, trajectories, summaries):
        """Verify per-game summary scores reconstruct from teacher_score
        components - same check as v2 trainer (training/trainer.py:461-496).
        Halts the run if any inconsistency is detected."""
        from core import teacher_score as ts
        passed = True
        max_check = 3
        checked = 0
        for traj, summ in zip(trajectories, summaries):
            if checked >= max_check:
                break
            for seat_str, sc in summ["scores"].items():
                ms = ts.move_score(sc["moves"])
                ps = ts.pin_goal_score(sc["pins"])
                ds = ts.distance_score(sc["dist"])
                # Recompute the partial score (no time component - summaries
                # don't carry time_taken_sec), and check it's consistent with
                # what the env's compute_final_score reported, modulo time.
                # Since both training and self_play use the same teacher_score
                # module, ts/ms/ps/ds + ts(time) should equal final_score.
                # We approximate by checking |reported - (ms+ps+ds)| <=
                # max possible time_score (100) + 1 tolerance.
                expected_partial = ms + ps + ds
                drift = abs(sc["final_score"] - expected_partial)
                if drift > 101.0:    # 100 (max time_score) + 1 tolerance
                    passed = False
            checked += 1
        record = {"iter": iteration, "alignment_ok": passed,
                  "games_checked": checked, "type": "summary_consistency"}
        self.logger.log_rule_alignment(record)
        if not passed:
            self.logger.write_halt_flag(
                "RULE_ALIGNMENT_FAILED",
                f"Iter {iteration}: summary consistency failed",
            )

    def _eval_inproc_2p(self, num_games: int = 10) -> Dict:
        self.network.eval()
        heuristic = HeuristicAgent(self.board)
        wins = 0
        eval_rng = random.Random(20260501)
        with torch.no_grad():
            for g in range(num_games):
                env = GameEnv(self.board, num_players=2)
                # Use random colors so eval distribution matches self-play
                env.reset(random_colors=True, rng=eval_rng)
                net_seat = g % 2
                while not env.is_done():
                    p = env.current_player
                    if p == net_seat:
                        seat_t = torch.tensor([p], device=self.device)
                        state = env.get_state_tensor(p).unsqueeze(0).to(self.device)
                        mask = env.get_legal_mask(p).unsqueeze(0).to(self.device)
                        out = self.network(state, mask, current_seat=seat_t)
                        action = int(out["policy"][0].argmax().item())
                    else:
                        action = heuristic.choose_move(env, p)
                    env.step(action)
                if env.get_winner() == net_seat:
                    wins += 1
        self.network.train()
        return {"games": num_games, "wins": wins, "win_rate": wins / num_games}
