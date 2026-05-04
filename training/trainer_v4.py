"""NEXUS v4 trainer.

Differences vs v3:
  - Uses NexusNetV4 (NBT backbone, score_margin + pin_final aux heads)
  - Uses ReplayBufferV4 (aux fields)
  - Uses self_play_v4 (MCTS-improved policy targets, NOT raw argmax)
  - AdamW with parameter-group split (no decay on biases/LayerNorm)
  - bf16 mixed precision when supported
  - EMA copy of weights for eval and best-checkpoint promotion
  - Frozen opponent pool snapshots EMA weights, not raw weights
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
from network.model_v4 import NexusNetV4
from training.replay_buffer_v4 import ReplayBufferV4
from training.heuristic_agent import HeuristicAgent
from training.losses_v4 import nexus_loss_v4
from training.self_play_v4 import generate_games_with_mcts
from training.self_play_v4_parallel import generate_games_parallel
from training.self_play_v4_server import generate_games_inference_server
import os as _os
_PARALLEL_WORKERS = int(_os.environ.get("NEXUS_SELFPLAY_WORKERS", "0"))
_INFERENCE_SERVER_WORKERS = int(_os.environ.get("NEXUS_INFERENCE_SERVER", "0"))
_INFERENCE_BATCH_MAX = int(_os.environ.get("NEXUS_INFERENCE_BATCH", "32"))
from inference.eval_harness import run_full_eval


def _split_param_groups(model):
    """Param-group split: weight decay on weights, no decay on biases/norms."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() < 2 or any(k in name.lower() for k in ("norm", "bias")):
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": Config.WEIGHT_DECAY},
        {"params": no_decay, "weight_decay": 0.0},
    ]


class EMAModel:
    """Polyak averaging - maintains an exponentially-weighted-average copy of
    model parameters. Used for eval and best-checkpoint promotion (more
    stable than raw weights, esp. with cosine LR restarts)."""

    def __init__(self, model: torch.nn.Module, decay: float = Config.EMA_DECAY_V4):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[k].copy_(v.detach())

    def copy_to(self, model: torch.nn.Module):
        model.load_state_dict(self.shadow)


class TrainerV4:
    def __init__(
        self,
        device: Optional[torch.device] = None,
        checkpoint_dir: str = "checkpoints_v4",
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
        self.network = NexusNetV4(self.board).to(self.device)
        self.network_old: Optional[NexusNetV4] = None
        self.ema = EMAModel(self.network)
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
        self.replay_buffer = ReplayBufferV4(capacity=Config.REPLAY_BUFFER_SIZE)
        self.game_id_counter = 0
        self.frozen_pool: List[NexusNetV4] = []

        # Mixed precision
        self.use_bf16 = (
            Config.USE_BF16_AMP and torch.cuda.is_available() and
            torch.cuda.is_bf16_supported()
        )

        self.run_dir = make_run_dir(run_name=run_name)
        self.logger = TrainingLogger(self.run_dir)
        self.logger.write_config_snapshot({
            "device": str(self.device),
            "v4": True,
            "use_bf16": self.use_bf16,
            "hidden_dim_v4": Config.HIDDEN_DIM_V4,
            "num_res_blocks_v4": Config.NUM_RES_BLOCKS_V4,
            "nbt_bottleneck_v4": Config.NBT_BOTTLENECK_V4,
            "mcts_train_sims": Config.MCTS_TRAIN_SIMS_V4,
            "mcts_train_m": Config.MCTS_TRAIN_M_V4,
            "ema_decay": Config.EMA_DECAY_V4,
        })

        self.best_mean_score = -1e9
        self.best_iter: Optional[int] = None
        self._eval_net = NexusNetV4(self.board).to(self.device)  # for EMA eval

    # ── Buffer push ──────────────────────────────────────────────────

    def _add_to_buffer(self, entry: Dict):
        legal = entry["legal_mask"]
        mask_np = np.asarray(legal).astype(bool)
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
            score_margin=entry.get("score_margin"),
            pin_final=entry.get("pin_final"),
            pin_final_valid=bool(entry.get("pin_final_valid", False)),
        )

    # ── Train step (with bf16 AMP) ───────────────────────────────────

    def _train_step(self, batch_size: int) -> Dict:
        if len(self.replay_buffer) < batch_size:
            return {"loss_total": 0.0, "lr": 0.0, "grad_norm": 0.0}

        sample = self.replay_buffer.sample(batch_size)
        states = sample["states"].to(self.device, non_blocking=True)
        target_policies = sample["policies"].to(self.device, non_blocking=True)
        legal_masks = sample["legal_masks"].to(self.device, non_blocking=True)
        seats = sample["players"].to(self.device, non_blocking=True)

        ctx = (
            torch.amp.autocast("cuda", dtype=torch.bfloat16)
            if self.use_bf16 else
            torch.amp.autocast("cuda", enabled=False)
        )

        with ctx:
            out = self.network(states, legal_masks, current_seat=seats)
            old_logits = None
            if self.network_old is not None:
                with torch.no_grad():
                    old_out = self.network_old(states, legal_masks, current_seat=seats)
                    old_logits = old_out["logits"]
            batch_dev = {
                "policies": target_policies,
                "value_vec": sample["value_vec"].to(self.device, non_blocking=True),
                "n_players": sample["n_players"].to(self.device, non_blocking=True),
                "opp_action": sample["opp_action"].to(self.device, non_blocking=True),
                "opp_legal_mask": sample["opp_legal_mask"].to(self.device, non_blocking=True),
                "plies_target": sample["plies_target"].to(self.device, non_blocking=True),
                "plies_valid": sample["plies_valid"].to(self.device, non_blocking=True),
                "score_margin_target": sample["score_margin_target"].to(self.device, non_blocking=True),
                "pin_final_target": sample["pin_final_target"].to(self.device, non_blocking=True),
                "pin_final_valid": sample["pin_final_valid"].to(self.device, non_blocking=True),
            }
            comps = nexus_loss_v4(out, batch_dev, legal_masks, old_logits=old_logits)
            total = comps["total"]

        # bf16 doesn't need GradScaler
        self.optimizer.zero_grad()
        total.backward()
        gn = torch.nn.utils.clip_grad_norm_(
            self.network.parameters(), Config.GRAD_CLIP
        )
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        # Update EMA
        self.ema.update(self.network)

        return {
            "loss_total": float(total.item()),
            "loss_policy": float(comps["policy"].item()),
            "loss_value_vec": float(comps["value_vec"].item()),
            "loss_opp": float(comps["opp_policy"].item() if hasattr(comps["opp_policy"], 'item') else comps["opp_policy"]),
            "loss_plies": float(comps["plies"].item() if hasattr(comps["plies"], 'item') else comps["plies"]),
            "loss_score_margin": float(comps["score_margin"].item()),
            "loss_pin_final": float(comps["pin_final"].item() if hasattr(comps["pin_final"], 'item') else comps["pin_final"]),
            "policy_entropy": float(comps["entropy"].item()),
            "loss_kl": float(comps["kl"].item()),
            "grad_norm": float(gn),
            "lr": float(self.optimizer.param_groups[0]["lr"]),
        }

    # ── Frozen pool snapshot ─────────────────────────────────────────

    def _maybe_snapshot_frozen(self, iteration: int):
        if iteration % Config.FREEZE_POOL_EVERY != 0 or iteration == 0:
            return
        snap = NexusNetV4(self.board).to(self.device)
        # Snapshot the EMA weights for stability
        snap.load_state_dict(self.ema.shadow)
        snap.eval()
        for p in snap.parameters():
            p.requires_grad_(False)
        self.frozen_pool.append(snap)
        if len(self.frozen_pool) > Config.FREEZE_POOL_SIZE:
            self.frozen_pool.pop(0)
        print(f"[Phase2-v4] Frozen pool size={len(self.frozen_pool)}")

    # ── Phase 2: self-play with MCTS-improved targets ────────────────

    def run_phase2(
        self,
        num_iterations: int,
        games_per_iter: int = Config.GAMES_PER_ITERATION,
        training_steps: int = Config.TRAINING_STEPS_PER_ITER,
        batch_size: int = Config.BATCH_SIZE,
        lr: float = Config.LEARNING_RATE,
        start_iter: int = 0,
    ):
        self.network.train()
        if self.optimizer is None:
            self.optimizer = optim.AdamW(
                _split_param_groups(self.network), lr=lr,
            )
        if self.scheduler is None:
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=max(1, Config.LR_T_0 * max(1, training_steps)),
                T_mult=1,
                eta_min=Config.LR_ETA_MIN,
            )
        # Apply resumed optim/sched state, if any
        resume_optim = getattr(self, "_resume_optim_state", None)
        if resume_optim is not None:
            self.optimizer.load_state_dict(resume_optim)
            self._resume_optim_state = None
        resume_sched = getattr(self, "_resume_sched_state", None)
        if resume_sched is not None:
            self.scheduler.load_state_dict(resume_sched)
            self._resume_sched_state = None

        for iteration in range(start_iter, start_iter + num_iterations):
            if self.logger.has_halt_flag("RULE_ALIGNMENT_FAILED"):
                print(f"[Phase2-v4] HALT iter {iteration}")
                return

            t0 = time.time()
            temperature = Config.get_temperature(
                iteration, total_iterations=num_iterations + start_iter
            )
            rng = random.Random(20260503 + iteration)

            self._maybe_snapshot_frozen(iteration)

            # 1) Self-play with MCTS-improved targets
            sp_t0 = time.time()
            self.network.eval()
            if _INFERENCE_SERVER_WORKERS > 0:
                print(f"[Phase2-v4] Self-play iter={iteration} via "
                      f"INFERENCE SERVER + {_INFERENCE_SERVER_WORKERS} CPU workers "
                      f"(games={games_per_iter}, batch_max={_INFERENCE_BATCH_MAX})",
                      flush=True)
                trajectories, summaries = generate_games_inference_server(
                    self.network, self.device, self.board,
                    num_games=games_per_iter,
                    iteration=iteration,
                    rng=rng,
                    num_simulations=Config.MCTS_TRAIN_SIMS_V4,
                    m=Config.MCTS_TRAIN_M_V4,
                    temperature=temperature,
                    start_game_id=self.game_id_counter,
                    frozen_pool=None,
                    num_workers=_INFERENCE_SERVER_WORKERS,
                    batch_max=_INFERENCE_BATCH_MAX,
                )
            elif _PARALLEL_WORKERS > 0:
                print(f"[Phase2-v4] Self-play iter={iteration} via "
                      f"{_PARALLEL_WORKERS} parallel workers "
                      f"(games={games_per_iter})", flush=True)
                trajectories, summaries = generate_games_parallel(
                    self.network, self.device, self.board,
                    num_games=games_per_iter,
                    iteration=iteration,
                    rng=rng,
                    num_simulations=Config.MCTS_TRAIN_SIMS_V4,
                    m=Config.MCTS_TRAIN_M_V4,
                    temperature=temperature,
                    start_game_id=self.game_id_counter,
                    frozen_pool=None,  # not yet supported in parallel path
                    num_workers=_PARALLEL_WORKERS,
                )
            else:
                trajectories, summaries = generate_games_with_mcts(
                    self.network, self.device, self.board,
                    num_games=games_per_iter,
                    iteration=iteration,
                    rng=rng,
                    num_simulations=Config.MCTS_TRAIN_SIMS_V4,
                    m=Config.MCTS_TRAIN_M_V4,
                    temperature=temperature,
                    start_game_id=self.game_id_counter,
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

            # 2) Train
            tr_t0 = time.time()
            agg = {}
            for _ in range(training_steps):
                step = self._train_step(batch_size)
                for k, v in step.items():
                    agg[k] = agg.get(k, 0.0) + v
            for k in agg:
                agg[k] /= max(1, training_steps)
            wall_train = time.time() - tr_t0

            # KL old-net (every iter)
            self.network_old = copy.deepcopy(self.network).eval()
            for p in self.network_old.parameters():
                p.requires_grad_(False)

            # 3) Inproc 2P eval (use EMA weights)
            inproc_eval = None
            if iteration % Config.EVAL_INPROC_EVERY == 0:
                inproc_eval = self._eval_inproc_2p_ema()

            # 4) Server eval (every K iters; use EMA weights)
            server_eval_results = []
            if iteration % Config.EVAL_SERVER_EVERY == 0 and iteration > 0:
                self.ema.copy_to(self._eval_net)
                ckpt_path = os.path.join(
                    self.checkpoint_dir, f"phase2_v4_iter{iteration:04d}_eval.pt"
                )
                self._eval_net.save(ckpt_path)
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
                        ms = float(sum(valid)) / len(valid)
                        if ms > self.best_mean_score:
                            self.best_mean_score = ms
                            self.best_iter = iteration
                            self._eval_net.save(
                                os.path.join(self.checkpoint_dir, "phase2_best_v4.pt")
                            )
                            print(f"[Phase2-v4] NEW BEST iter={iteration} mean_score={ms:.1f}")
                finally:
                    if os.path.exists(ckpt_path):
                        os.remove(ckpt_path)

            # 5) Snapshot
            self.network.save(os.path.join(self.checkpoint_dir,
                                           "phase2_latest_v4.pt"))
            if iteration % 25 == 0 and iteration > 0:
                self.network.save(os.path.join(self.checkpoint_dir,
                                               f"phase2_v4_iter{iteration}.pt"))

            # 5.5) Full-state checkpoint every 5 iters for crash recovery
            if iteration % 5 == 0 and iteration > 0:
                self.save_full_state(os.path.join(self.checkpoint_dir,
                                                   "phase2_resume_v4.pt"))

            # 6) Log
            wall_total = time.time() - t0
            by_N = {}
            for s in summaries:
                by_N[s["N"]] = by_N.get(s["N"], 0) + 1
            avg_traj = (sum(len(t) for t in trajectories)
                        / max(1, len(trajectories)))
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
                    "loss_score_margin": agg.get("loss_score_margin", 0.0),
                    "loss_pin_final": agg.get("loss_pin_final", 0.0),
                    "policy_entropy": agg.get("policy_entropy", 0.0),
                    "loss_kl": agg.get("loss_kl", 0.0),
                    "grad_norm": agg.get("grad_norm", 0.0),
                    "lr": agg.get("lr", 0.0),
                    "training_steps": training_steps,
                    "replay_buffer_size": len(self.replay_buffer),
                },
                "temperature": temperature,
                "best_mean_score_so_far": self.best_mean_score,
                "best_iter_so_far": self.best_iter,
            }
            if inproc_eval is not None:
                metrics["eval_2p_in_process"] = inproc_eval
            self.logger.log_iter(metrics)
            self.logger.regenerate_status()

            print(
                f"[Phase2-v4] iter={iteration} "
                f"loss={agg.get('loss_total', 0):.3f} "
                f"pol={agg.get('loss_policy', 0):.3f} "
                f"val={agg.get('loss_value_vec', 0):.3f} "
                f"H={agg.get('policy_entropy', 0):.3f} "
                f"lr={agg.get('lr', 0):.5f} "
                f"sec={int(wall_total)} "
                f"games={len(trajectories)} "
                f"buf={len(self.replay_buffer)}"
                + (f" eval2p={inproc_eval['win_rate']:.1%}" if inproc_eval else "")
            )

    # ── Full state save/load for crash recovery ─────────────────────

    def save_full_state(self, path: str):
        """Save EVERYTHING needed to resume: weights, EMA, optimizer,
        scheduler, replay buffer, frozen pool weights, counters, best_score.

        Call this every iter to make crash recovery cheap.
        """
        import pickle
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            "network_state_dict": self.network.state_dict(),
            "ema_shadow": {k: v.cpu().clone() for k, v in self.ema.shadow.items()},
            "optimizer_state_dict": (self.optimizer.state_dict()
                                     if self.optimizer is not None else None),
            "scheduler_state_dict": (self.scheduler.state_dict()
                                     if self.scheduler is not None else None),
            "frozen_pool_state_dicts": [
                snap.state_dict() for snap in self.frozen_pool
            ],
            "best_mean_score": self.best_mean_score,
            "best_iter": self.best_iter,
            "game_id_counter": self.game_id_counter,
            "buffer_position": self.replay_buffer.position,
            "buffer_size": self.replay_buffer.size,
        }
        torch.save(state, path)
        # Buffer arrays separately (large, np.save is fast)
        buf_dir = path + "_buffer"
        os.makedirs(buf_dir, exist_ok=True)
        for name in ["states", "policies", "values", "actions", "legal_masks",
                     "players", "value_vec", "n_players", "opp_action",
                     "opp_legal_masks", "plies_remaining", "plies_valid",
                     "score_margin", "pin_final", "pin_final_valid"]:
            np.save(os.path.join(buf_dir, name + ".npy"),
                    getattr(self.replay_buffer, name))

    def load_full_state(self, path: str):
        """Restore everything from save_full_state. Call BEFORE run_phase2."""
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(state["network_state_dict"])
        for k, v in state["ema_shadow"].items():
            self.ema.shadow[k] = v.to(self.device)
        # Optimizer/scheduler are constructed in run_phase2; we apply state there.
        self._resume_optim_state = state.get("optimizer_state_dict")
        self._resume_sched_state = state.get("scheduler_state_dict")
        # Frozen pool
        self.frozen_pool = []
        for sd in state.get("frozen_pool_state_dicts", []):
            snap = NexusNetV4(self.board).to(self.device)
            snap.load_state_dict(sd)
            snap.eval()
            for p in snap.parameters():
                p.requires_grad_(False)
            self.frozen_pool.append(snap)
        self.best_mean_score = state.get("best_mean_score", -1e9)
        self.best_iter = state.get("best_iter")
        self.game_id_counter = state.get("game_id_counter", 0)
        # Buffer
        buf_dir = path + "_buffer"
        if os.path.isdir(buf_dir):
            for name in ["states", "policies", "values", "actions", "legal_masks",
                         "players", "value_vec", "n_players", "opp_action",
                         "opp_legal_masks", "plies_remaining", "plies_valid",
                         "score_margin", "pin_final", "pin_final_valid"]:
                arr_path = os.path.join(buf_dir, name + ".npy")
                if os.path.exists(arr_path):
                    setattr(self.replay_buffer, name, np.load(arr_path))
            self.replay_buffer.position = state.get("buffer_position", 0)
            self.replay_buffer.size = state.get("buffer_size", 0)
        print(f"[Trainer] Loaded full state from {path}: "
              f"buf={len(self.replay_buffer)}, best_iter={self.best_iter}, "
              f"frozen_pool={len(self.frozen_pool)}")

    def _eval_inproc_2p_ema(self, num_games: int = Config.EVAL_INPROC_GAMES) -> Dict:
        """Inproc 2P eval using EMA weights for stability."""
        self.ema.copy_to(self._eval_net)
        self._eval_net.eval()
        heuristic = HeuristicAgent(self.board)
        wins = 0
        eval_rng = random.Random(20260503)
        with torch.no_grad():
            for g in range(num_games):
                env = GameEnv(self.board, num_players=2)
                env.reset(random_colors=True, rng=eval_rng)
                net_seat = g % 2
                while not env.is_done():
                    p = env.current_player
                    if p == net_seat:
                        seat_t = torch.tensor([p], device=self.device)
                        state = env.get_state_tensor(p).unsqueeze(0).to(self.device)
                        mask = env.get_legal_mask(p).unsqueeze(0).to(self.device)
                        out = self._eval_net(state, mask, current_seat=seat_t)
                        action = int(out["policy"][0].argmax().item())
                    else:
                        action = heuristic.choose_move(env, p)
                    env.step(action)
                if env.get_winner() == net_seat:
                    wins += 1
        return {"games": num_games, "wins": wins, "win_rate": wins / num_games}
