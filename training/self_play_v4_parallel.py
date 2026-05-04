"""Parallel self-play wrapper for Phase 2 v4.

Launches K worker subprocesses (CUDA-safe spawn). Each worker loads the
network from a passed state_dict, runs generate_games_with_mcts on a
slice of games, and returns trajectories + summaries via mp.Queue.

Frozen-pool participation is disabled in the parallel path for now
(simpler; can be re-added by passing snapshot state_dicts to workers).
"""
from __future__ import annotations

import os
import random
from typing import Dict, List, Optional, Tuple

import torch
import torch.multiprocessing as mp

from config import Config


def _worker_proc(worker_id: int, state_dict_cpu, n_games: int,
                 iteration: int, num_simulations: int, m: int,
                 temperature: float, start_game_id: int, seed: int,
                 result_queue) -> None:
    """Run inside spawned subprocess. CUDA context is fresh."""
    try:
        # Each worker gets its own CUDA context on the same GPU
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_num_threads(2)  # avoid CPU oversubscription across workers

        from core.board import HexBoard
        from network.model_v4 import NexusNetV4
        from training.self_play_v4 import generate_games_with_mcts

        device = torch.device("cuda")
        board = HexBoard()
        network = NexusNetV4(board).to(device)
        network.load_state_dict(state_dict_cpu)
        network.eval()

        rng = random.Random(seed)
        trajectories, summaries = generate_games_with_mcts(
            network, device, board,
            num_games=n_games,
            iteration=iteration,
            rng=rng,
            num_simulations=num_simulations,
            m=m,
            temperature=temperature,
            start_game_id=start_game_id,
            frozen_pool=None,
        )
        result_queue.put((worker_id, trajectories, summaries, None))
    except Exception as e:
        import traceback
        result_queue.put((worker_id, None, None,
                          f"{e}\n{traceback.format_exc()}"))


def generate_games_parallel(
    network,
    device: torch.device,
    board,
    num_games: int,
    iteration: int,
    rng: random.Random,
    num_simulations: int = Config.MCTS_TRAIN_SIMS_V4,
    m: int = Config.MCTS_TRAIN_M_V4,
    temperature: float = 1.0,
    start_game_id: int = 0,
    frozen_pool: Optional[List] = None,
    num_workers: int = 8,
) -> Tuple[List, List]:
    """Spawn K workers, each running a slice of games on the GPU."""
    num_workers = max(1, min(num_workers, num_games))
    base = num_games // num_workers
    rem = num_games % num_workers
    splits = [base + (1 if i < rem else 0) for i in range(num_workers)]

    # Move state_dict to CPU for spawn pickling
    state_dict_cpu = {k: v.detach().cpu() for k, v in network.state_dict().items()}

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    procs = []
    cur_gid = start_game_id
    for w in range(num_workers):
        n = splits[w]
        if n <= 0:
            continue
        seed = rng.randrange(2**31)
        p = ctx.Process(
            target=_worker_proc,
            args=(w, state_dict_cpu, n, iteration,
                  num_simulations, m, temperature, cur_gid, seed,
                  result_queue),
            daemon=False,
        )
        p.start()
        procs.append(p)
        cur_gid += n

    trajectories: List = [None] * len(procs)
    summaries: List = [None] * len(procs)
    errors: List = []
    for _ in procs:
        wid, traj, summ, err = result_queue.get()
        if err is not None:
            errors.append(f"[worker {wid}] {err}")
        else:
            trajectories[wid] = traj
            summaries[wid] = summ

    for p in procs:
        p.join(timeout=60)
        if p.is_alive():
            p.terminate()

    if errors:
        raise RuntimeError("Parallel self-play worker error(s):\n" +
                           "\n---\n".join(errors))

    flat_traj = []
    flat_summ = []
    for t in trajectories:
        if t is not None:
            flat_traj.extend(t)
    for s in summaries:
        if s is not None:
            flat_summ.extend(s)
    return flat_traj, flat_summ
