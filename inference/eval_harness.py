"""Evaluation harness - run NEXUS vs N-1 heuristic opponents on the teacher's
actual `game.py` server. Used during training (every Config.EVAL_SERVER_EVERY
iters) and at final evaluation.

Reuses scripts/test_n_player.py infrastructure; this is a programmatic API.
"""

from __future__ import annotations

import json
import os
import random
import socket
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional

NEXUS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEACHER_DIR = os.path.join(
    NEXUS_DIR, "RLChineseCheckers", "multi system single machine minimal"
)

if NEXUS_DIR not in sys.path:
    sys.path.insert(0, NEXUS_DIR)

from core.board import HexBoard

HOST, PORT = "127.0.0.1", 50555


# ── RPC ──────────────────────────────────────────────────────────


def _rpc(payload: Dict[str, Any]) -> Dict[str, Any]:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(15.0)
    try:
        s.connect((HOST, PORT))
    except Exception as e:
        return {"ok": False, "error": str(e)}
    s.sendall(json.dumps(payload).encode("utf-8"))
    chunks = []
    while True:
        try:
            data = s.recv(1_000_000)
            if not data:
                break
            chunks.append(data)
            try:
                json.loads(b"".join(chunks).decode("utf-8"))
                break
            except json.JSONDecodeError:
                continue
        except socket.timeout:
            break
    s.close()
    try:
        return json.loads(b"".join(chunks).decode("utf-8"))
    except Exception:
        return {"ok": False, "error": "no-response"}


# ── Heuristic client (in-process) ────────────────────────────────


_BOARD = None


def _board() -> HexBoard:
    global _BOARD
    if _BOARD is None:
        _BOARD = HexBoard()
    return _BOARD


def _heuristic_pick(my_color: str, my_pin_positions: List[int],
                    legal_moves: Dict[Any, List[int]]):
    """Distance-to-goal + hop bonus + goal-landing bonus."""
    b = _board()
    best_score = float("-inf")
    best_move = None
    for pid_str, dests in legal_moves.items():
        pin_id = int(pid_str)
        if pin_id >= len(my_pin_positions):
            continue
        piece_pos = my_pin_positions[pin_id]
        for dest in dests:
            d_before = b.min_distance_to_goal(piece_pos, my_color)
            d_after = b.min_distance_to_goal(dest, my_color)
            score = (d_before - d_after) * 10.0
            hop_len = b.axial_distance(piece_pos, dest)
            if hop_len > 1:
                score += hop_len * 5.0
            if b.is_in_goal(dest, my_color):
                score += 20.0
            if score > best_score:
                best_score = score
                best_move = (pin_id, dest)
    if best_move:
        return best_move
    for pid_str, dests in legal_moves.items():
        if dests:
            return int(pid_str), dests[0]
    return None


def _heuristic_client_loop(name: str, expected_N: int,
                           stop_evt: threading.Event,
                           rng: random.Random):
    r = _rpc({"op": "join", "player_name": name})
    if not r.get("ok"):
        return
    game_id = r["game_id"]
    player_id = r["player_id"]
    colour = r["colour"]

    # Wait until all N players have joined
    for _ in range(120):
        if stop_evt.is_set():
            return
        st = _rpc({"op": "get_state", "game_id": game_id}).get("state", {})
        if len(st.get("players", [])) >= expected_N:
            break
        if st.get("status") == "PLAYING":
            break
        time.sleep(0.2)

    _rpc({"op": "start", "game_id": game_id, "player_id": player_id})

    while not stop_evt.is_set():
        st = _rpc({"op": "get_state", "game_id": game_id})
        if not st.get("ok"):
            time.sleep(0.05)
            continue
        state = st["state"]
        if state["status"] == "FINISHED":
            return
        if state.get("current_turn_colour") == colour and state["status"] == "PLAYING":
            lr = _rpc({"op": "get_legal_moves", "game_id": game_id, "player_id": player_id})
            legal = lr.get("legal_moves", {})
            movable = [(pid, mv) for pid, mv in legal.items() if mv]
            if not movable:
                time.sleep(0.05)
                continue
            my_pins = state.get("pins", {}).get(colour, [])
            pick = _heuristic_pick(colour, my_pins, legal)
            if pick is None:
                pid, dests = rng.choice(movable)
                pid, dest = int(pid), rng.choice(dests)
            else:
                pid, dest = pick
            _rpc({"op": "move", "game_id": game_id, "player_id": player_id,
                  "pin_id": int(pid), "to_index": int(dest)})
        time.sleep(0.02)


# ── Server lifecycle ─────────────────────────────────────────────


def _start_teacher_server() -> subprocess.Popen:
    proc = subprocess.Popen(
        ["python3", "game.py"],
        cwd=TEACHER_DIR,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(1.5)
    proc.stdin.write(b"create\n")
    proc.stdin.flush()
    for _ in range(40):
        sr = _rpc({"op": "status"})
        if sr.get("ok") and len(sr.get("games", [])) >= 1:
            return proc
        time.sleep(0.25)
    return proc


def _stop_teacher_server(proc: subprocess.Popen):
    try:
        proc.terminate()
        proc.wait(timeout=3)
    except Exception:
        proc.kill()


# ── Single match ─────────────────────────────────────────────────


def run_single_match(N: int, model_path: str, name: str = "NEXUS",
                     game_log_dir: Optional[str] = None,
                     hard_cap_sec: float = 90.0,
                     rng_seed: int = 0) -> Dict[str, Any]:
    """Run one match: 1 NEXUS + (N-1) heuristic opponents on teacher's server.

    Returns a dict with: status (ok/timeout-no-finish/error), final_state,
    nexus_score (final_score/dist/pins/moves), nexus_rank.
    """
    rng = random.Random(rng_seed)
    proc = _start_teacher_server()
    if not proc:
        return {"ok": False, "error": "server-start-failed"}

    stop_evt = threading.Event()
    threads = []
    for i in range(N - 1):
        t = threading.Thread(
            target=_heuristic_client_loop,
            args=(f"H{i+1}", N, stop_evt, rng),
            daemon=True,
        )
        t.start()
        threads.append(t)
        time.sleep(0.15)

    # Spawn NEXUS via play_server.py subprocess (real path)
    env = os.environ.copy()
    env["PYTHONPATH"] = NEXUS_DIR
    log_path = None
    log_file = None
    if game_log_dir:
        os.makedirs(game_log_dir, exist_ok=True)
        log_path = os.path.join(game_log_dir, f"N{N}_{name}.log")
        log_file = open(log_path, "w")
    nexus = subprocess.Popen(
        [
            os.path.join(NEXUS_DIR, "venv", "bin", "python"),
            os.path.join(NEXUS_DIR, "scripts", "play_server.py"),
            "--model", model_path,
            "--name", name,
            "--auto-start",
            "--device", "cuda",
        ],
        cwd=NEXUS_DIR,
        stdout=(log_file or subprocess.DEVNULL),
        stderr=subprocess.STDOUT,
        env=env,
    )

    start = time.time()
    finished = False
    final_state = None
    while time.time() - start < hard_cap_sec:
        sr = _rpc({"op": "status"})
        if sr.get("ok"):
            for g in sr.get("games", []):
                if g["status"] == "FINISHED" and len(g["players"]) == N:
                    full = _rpc({"op": "get_state", "game_id": g["game_id"]})
                    final_state = full.get("state")
                    finished = True
                    break
            if finished:
                break
        time.sleep(0.5)

    stop_evt.set()
    try:
        nexus.terminate()
        nexus.wait(timeout=5)
    except Exception:
        nexus.kill()
    if log_file:
        log_file.close()
    _stop_teacher_server(proc)

    if not finished:
        return {"ok": False, "error": "timeout-no-finish",
                "log_path": log_path}

    # Parse final state for NEXUS
    nexus_score = None
    ranked = sorted(
        final_state["players"],
        key=lambda p: (p.get("score") or {}).get("final_score", 0),
        reverse=True,
    )
    nexus_rank = None
    for i, p in enumerate(ranked):
        if p["name"] == name:
            nexus_score = p.get("score") or {}
            nexus_rank = i + 1
            break

    return {
        "ok": True,
        "N": N,
        "rank": nexus_rank,
        "nexus_score": nexus_score,
        "total_moves": final_state["move_count"],
        "log_path": log_path,
    }


# ── Multi-game eval (one cell of the eval grid) ──────────────────


def run_eval_cell(model_path: str, N: int, num_games: int,
                  iter_n: int, output_dir: Optional[str] = None) -> Dict[str, Any]:
    """Run num_games matches at player count N. Returns aggregate summary."""
    games_dir = None
    if output_dir:
        games_dir = os.path.join(output_dir, f"iter_{iter_n:04d}", f"N{N}")
        os.makedirs(games_dir, exist_ok=True)

    started = time.time()
    ranks = []
    scores = []
    moves = []
    for g in range(num_games):
        result = run_single_match(
            N=N, model_path=model_path, name="NEXUS",
            game_log_dir=games_dir, rng_seed=iter_n * 1000 + g,
        )
        if not result.get("ok"):
            continue
        # Defensive: nexus_score may be None if the player wasn't found in
        # the final state (e.g., port conflict with another concurrent eval).
        nexus_score = result.get("nexus_score") or {}
        scores.append(nexus_score.get("final_score", 0.0))
        if result.get("rank") is not None:
            ranks.append(result["rank"])
        moves.append(result.get("total_moves", 0))

    elapsed = time.time() - started

    if not scores:
        return {"iter": iter_n, "N": N, "ok": False,
                "games": num_games, "wall_sec": elapsed,
                "error": "all-games-failed"}
    if not ranks:
        ranks = [N] * len(scores)  # if rank unknown, assume worst case

    return {
        "iter": iter_n,
        "N": N,
        "ok": True,
        "games": len(scores),
        "nexus_ranks": ranks,
        "mean_rank": float(sum(ranks)) / len(ranks),
        "mean_final_score": float(sum(scores)) / len(scores),
        "min_final_score": float(min(scores)),
        "max_final_score": float(max(scores)),
        "moves_avg": float(sum(moves)) / len(moves) if moves else 0.0,
        "wall_sec": elapsed,
    }


# ── Full eval pass across N ──────────────────────────────────────


def run_full_eval(model_path: str, iter_n: int,
                  num_games_per_N: int = 3,
                  Ns: List[int] = (2, 3, 4, 5, 6),
                  output_dir: Optional[str] = None,
                  v3: bool = False) -> List[Dict[str, Any]]:
    """v3 flag is informational only - play_server auto-detects."""
    _ = v3
    """Run server eval for each N and return list of summaries."""
    results = []
    for N in Ns:
        print(f"  [eval] iter={iter_n} N={N} ({num_games_per_N} games)...")
        r = run_eval_cell(model_path, N, num_games_per_N, iter_n, output_dir)
        results.append(r)
        print(f"  [eval] iter={iter_n} N={N} -> "
              f"mean_score={r.get('mean_final_score', 0):.1f} "
              f"({r.get('wall_sec', 0):.1f}s)")
    return results
