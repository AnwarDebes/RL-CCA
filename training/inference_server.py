"""Cross-game batched inference server for Phase 2 v4 self-play.

Architecture:
  - 1 server process owns the GPU + network. Drains a request queue,
    stacks states into a batch, runs ONE forward, distributes responses.
  - K CPU worker processes run MCTS + env stepping. They send leaf states
    via mp.Queue and block on a per-worker reply queue.

This breaks bs=1 latency-bound forwards; with K=16 workers, each forward
batches ~16 states → ~30x compute per kernel launch → real GPU saturation.

Public API:
  - start_inference_server(state_dict_cpu, ...) → (server_proc, req_queue, reply_queues)
  - InferenceClient(req_queue, reply_queue, worker_id) - drop-in for `network`
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.multiprocessing as mp
import numpy as np


_SHUTDOWN = "__SHUTDOWN__"


class InferenceClient:
    """Drop-in replacement for the network in MCTS - sends requests to server."""

    def __init__(self, req_queue, reply_queue, worker_id: int):
        self.req_queue = req_queue
        self.reply_queue = reply_queue
        self.worker_id = worker_id
        self.device = torch.device("cpu")  # MCTS does .to(device) - keep on CPU

    def __call__(self, state_t, mask_t, current_seat=None):
        # state_t: [1, C, H, W] cpu; mask_t: [1, A] cpu; current_seat: [1] cpu int
        state_np = state_t.detach().cpu().numpy()
        mask_np = mask_t.detach().cpu().numpy()
        seat_int = int(current_seat[0]) if current_seat is not None else 0

        self.req_queue.put((self.worker_id, state_np, mask_np, seat_int))
        policy_np, value_vec_np = self.reply_queue.get()

        return {
            "policy": torch.from_numpy(policy_np).unsqueeze(0),
            "value_vec": torch.from_numpy(value_vec_np).unsqueeze(0),
        }

    def eval(self):
        return self

    def train(self):
        return self


def _server_loop(state_dict_cpu, num_workers: int,
                 req_queue, reply_queues,
                 ready_event, shutdown_event,
                 batch_max: int = 32, wait_us: int = 500):
    """Server process: own the GPU, batch requests, respond.

    batch_max: max states per forward (cap).
    wait_us: time to wait for additional requests after first arrives (microseconds).
    """
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        from core.board import HexBoard
        from network.model_v4 import NexusNetV4

        device = torch.device("cuda")
        board = HexBoard()
        network = NexusNetV4(board).to(device).eval()
        network.load_state_dict(state_dict_cpu)

        ready_event.set()

        wait_s = wait_us / 1_000_000.0
        while not shutdown_event.is_set():
            # Block on first request
            try:
                first = req_queue.get(timeout=0.5)
            except Exception:
                continue
            if first == _SHUTDOWN:
                break

            batch = [first]
            # Quick drain for additional pending requests up to batch_max
            t_end = time.time() + wait_s
            while len(batch) < batch_max and time.time() < t_end:
                try:
                    item = req_queue.get_nowait()
                    if item == _SHUTDOWN:
                        shutdown_event.set()
                        break
                    batch.append(item)
                except Exception:
                    # Empty for now; tight spin briefly
                    if time.time() >= t_end:
                        break
            # Greedy drain everything pending up to batch_max
            while len(batch) < batch_max:
                try:
                    item = req_queue.get_nowait()
                    if item == _SHUTDOWN:
                        shutdown_event.set()
                        break
                    batch.append(item)
                except Exception:
                    break

            # Forward as a batch
            worker_ids = [b[0] for b in batch]
            states = np.stack([b[1][0] for b in batch], axis=0)
            masks = np.stack([b[2][0] for b in batch], axis=0)
            seats = np.array([b[3] for b in batch], dtype=np.int64)

            state_t = torch.from_numpy(states).pin_memory().to(device, non_blocking=True)
            mask_t = torch.from_numpy(masks).pin_memory().to(device, non_blocking=True)
            seat_t = torch.from_numpy(seats).pin_memory().to(device, non_blocking=True)

            with torch.no_grad():
                try:
                    out = network(state_t, mask_t, current_seat=seat_t)
                except TypeError:
                    out = network(state_t, mask_t)

            policy = out["policy"].detach().cpu().numpy()
            value_vec = out["value_vec"].detach().cpu().numpy()

            for i, wid in enumerate(worker_ids):
                reply_queues[wid].put((policy[i], value_vec[i]))
    except Exception as e:
        import traceback
        print(f"[INFERENCE SERVER ERROR] {e}\n{traceback.format_exc()}",
              flush=True)
        # Try to unblock workers with poison
        for q in reply_queues:
            try:
                q.put((np.zeros(1), np.zeros(1)))
            except Exception:
                pass
        raise


def start_inference_server(state_dict_cpu, num_workers: int,
                           batch_max: int = 32, wait_us: int = 500):
    """Spawn server process. Returns (proc, req_queue, reply_queues, shutdown_event, ready_event)."""
    ctx = mp.get_context("spawn")
    req_queue = ctx.Queue(maxsize=num_workers * 4)
    reply_queues = [ctx.Queue(maxsize=2) for _ in range(num_workers)]
    ready_event = ctx.Event()
    shutdown_event = ctx.Event()

    proc = ctx.Process(
        target=_server_loop,
        args=(state_dict_cpu, num_workers,
              req_queue, reply_queues,
              ready_event, shutdown_event,
              batch_max, wait_us),
        daemon=False,
    )
    proc.start()
    # Wait until network is loaded on GPU
    if not ready_event.wait(timeout=120):
        proc.terminate()
        raise RuntimeError("Inference server failed to ready within 120s")
    return proc, req_queue, reply_queues, shutdown_event
