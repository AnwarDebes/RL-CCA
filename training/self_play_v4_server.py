"""Self-play orchestrator using cross-game inference server + work-stealing queue.

Spawns 1 inference server (GPU) + K CPU workers. Each worker pulls the next
game-id from a shared work queue (work-stealing) - eliminates pre-assignment
imbalance from variable N (player count) per game.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import torch
import torch.multiprocessing as mp

from config import Config
from training.inference_server import (
    InferenceClient, start_inference_server, _SHUTDOWN,
)


_QUEUE_DONE = "__QUEUE_DONE__"


def _cpu_worker_proc(worker_id: int, iteration: int,
                     num_simulations: int, m: int, temperature: float,
                     seed: int, work_queue, req_queue, reply_queue,
                     result_queue) -> None:
    """CPU-only worker: pulls game-ids from work_queue, runs one game per pull.

    Returns (worker_id, trajectories_list, summaries_list, error)
    where trajectories_list = list of per-game traj lists.
    """
    try:
        torch.set_num_threads(2)

        from core.board import HexBoard
        from training.self_play_v4 import generate_games_with_mcts

        device = torch.device("cpu")
        board = HexBoard()
        client = InferenceClient(req_queue, reply_queue, worker_id)
        rng = random.Random(seed)

        all_trajectories = []
        all_summaries = []

        while True:
            try:
                gid = work_queue.get(timeout=0.5)
            except Exception:
                continue
            if gid == _QUEUE_DONE:
                break
            traj, summ = generate_games_with_mcts(
                client, device, board,
                num_games=1,
                iteration=iteration,
                rng=rng,
                num_simulations=num_simulations,
                m=m,
                temperature=temperature,
                start_game_id=gid,
                frozen_pool=None,
            )
            all_trajectories.extend(traj)
            all_summaries.extend(summ)

        result_queue.put((worker_id, all_trajectories, all_summaries, None))
    except Exception as e:
        import traceback
        result_queue.put((worker_id, None, None,
                          f"{e}\n{traceback.format_exc()}"))


def generate_games_inference_server(
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
    frozen_pool=None,
    num_workers: int = 16,
    batch_max: int = 32,
) -> Tuple[List, List]:
    """K CPU workers (work-stealing) + 1 GPU inference server."""
    num_workers = max(1, min(num_workers, num_games))

    state_dict_cpu = {k: v.detach().cpu() for k, v in network.state_dict().items()}

    server_proc, req_queue, reply_queues, shutdown_event = start_inference_server(
        state_dict_cpu, num_workers=num_workers,
        batch_max=batch_max, wait_us=500,
    )

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    work_queue = ctx.Queue()

    # Populate work queue with game-ids; sentinels signal end
    for gid in range(start_game_id, start_game_id + num_games):
        work_queue.put(gid)
    for _ in range(num_workers):
        work_queue.put(_QUEUE_DONE)

    procs = []
    try:
        for w in range(num_workers):
            seed = rng.randrange(2**31)
            p = ctx.Process(
                target=_cpu_worker_proc,
                args=(w, iteration, num_simulations, m, temperature,
                      seed, work_queue, req_queue, reply_queues[w],
                      result_queue),
                daemon=False,
            )
            p.start()
            procs.append(p)

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
            p.join(timeout=120)
            if p.is_alive():
                p.terminate()

        if errors:
            raise RuntimeError(
                "Inference-server worker error(s):\n" +
                "\n---\n".join(errors)
            )

        flat_traj = []
        flat_summ = []
        for t in trajectories:
            if t is not None:
                flat_traj.extend(t)
        for s in summaries:
            if s is not None:
                flat_summ.extend(s)
        return flat_traj, flat_summ
    finally:
        shutdown_event.set()
        try:
            req_queue.put(_SHUTDOWN, block=False)
        except Exception:
            pass
        server_proc.join(timeout=10)
        if server_proc.is_alive():
            server_proc.terminate()
