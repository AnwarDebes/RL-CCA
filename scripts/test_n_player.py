#!/usr/bin/env python3
"""End-to-end N-player verification.

Starts the teacher's game.py server, then for N in {2,3,4,5,6}:
  - Creates a game on the server
  - Spawns one NEXUS player (subprocess via play_server.py)
  - Spawns N-1 in-process random players
  - Waits for game to finish (or hit 60s game-time limit)
  - Reports scores

Random players move immediately (no random delay) so we get many moves
within the 60s game-time limit instead of timing out after 5-12 moves.
"""
import json
import os
import random
import socket
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
NEXUS_DIR = os.path.dirname(HERE)
TEACHER_DIR = os.path.join(NEXUS_DIR, "RLChineseCheckers", "multi system single machine minimal")
HOST, PORT = "127.0.0.1", 50555

sys.path.insert(0, NEXUS_DIR)
from core.board import HexBoard  # noqa: E402

_BOARD = HexBoard()


def heuristic_pick(my_color, my_pin_positions, legal_moves):
    """Same scoring as inference.tournament_player._heuristic_move.

    Distance-to-goal reduction + hop bonus + goal-landing bonus.
    """
    best_score = float('-inf')
    best_move = None
    for pid_str, dests in legal_moves.items():
        pin_id = int(pid_str)
        if pin_id >= len(my_pin_positions):
            continue
        piece_pos = my_pin_positions[pin_id]
        for dest in dests:
            d_before = _BOARD.min_distance_to_goal(piece_pos, my_color)
            d_after = _BOARD.min_distance_to_goal(dest, my_color)
            score = (d_before - d_after) * 10.0
            hop_len = _BOARD.axial_distance(piece_pos, dest)
            if hop_len > 1:
                score += hop_len * 5.0
            if _BOARD.is_in_goal(dest, my_color):
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


# ── RPC helpers ────────────────────────────────────────────────

def rpc(payload):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(15.0)
    try:
        s.connect((HOST, PORT))
    except Exception as e:
        return {"ok": False, "error": f"connect-failed: {e}"}
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
    raw = b"".join(chunks)
    if not raw:
        return {"ok": False, "error": "no-response"}
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception as e:
        return {"ok": False, "error": f"bad-json: {e}"}


# ── Random player (in-process thread) ──────────────────────────

def opp_player_loop(name, expected_N, stop_evt, log_lines, mode="random"):
    r = rpc({"op": "join", "player_name": name})
    if not r.get("ok"):
        log_lines.append(f"[{name}] JOIN FAILED: {r.get('error')}")
        return
    game_id = r["game_id"]
    player_id = r["player_id"]
    colour = r["colour"]
    log_lines.append(f"[{name}] joined as {colour}")

    # Wait until ALL N players have joined before sending START
    # (sending start prematurely sets lock_joining and locks others out)
    for _ in range(120):
        if stop_evt.is_set():
            return
        st = rpc({"op": "get_state", "game_id": game_id}).get("state", {})
        if len(st.get("players", [])) >= expected_N:
            break
        if st.get("status") == "PLAYING":
            break
        time.sleep(0.25)

    rpc({"op": "start", "game_id": game_id, "player_id": player_id})
    log_lines.append(f"[{name}] sent START")

    while not stop_evt.is_set():
        st = rpc({"op": "get_state", "game_id": game_id})
        if not st.get("ok"):
            time.sleep(0.2)
            continue
        state = st["state"]
        if state["status"] == "FINISHED":
            return
        if state.get("current_turn_colour") == colour and state["status"] == "PLAYING":
            lr = rpc({"op": "get_legal_moves", "game_id": game_id, "player_id": player_id})
            legal = lr.get("legal_moves", {})
            movable = [(pid, mv) for pid, mv in legal.items() if mv]
            if not movable:
                time.sleep(0.1)
                continue
            if mode == "heuristic":
                my_pins = state.get("pins", {}).get(colour, [])
                pick = heuristic_pick(colour, my_pins, legal)
                if pick is None:
                    pid, dests = random.choice(movable)
                    dest = random.choice(dests)
                    pid = int(pid)
                else:
                    pid, dest = pick
            else:
                pid, dests = random.choice(movable)
                dest = random.choice(dests)
                pid = int(pid)
            rpc({"op": "move", "game_id": game_id, "player_id": player_id,
                 "pin_id": int(pid), "to_index": int(dest)})
        time.sleep(0.05)


# ── Run one N-player match ─────────────────────────────────────

def start_server():
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
    # Poll until create has registered (status RPC returns 1 game)
    for _ in range(40):
        sr = rpc({"op": "status"})
        if sr.get("ok") and len(sr.get("games", [])) >= 1:
            return proc
        time.sleep(0.25)
    return proc


def stop_server(proc):
    try:
        proc.terminate()
        proc.wait(timeout=3)
    except Exception:
        proc.kill()


def run_match(N, model_path, opp_mode="random"):
    print(f"\n{'='*60}")
    print(f" N = {N} players  (1 NEXUS + {N-1} {opp_mode})")
    print('='*60)

    # Fresh server per match - avoids stale state between runs
    server_proc = start_server()
    print(f"[server] fresh, pid={server_proc.pid}")

    stop_evt = threading.Event()
    log = []
    threads = []
    for i in range(N - 1):
        prefix = "H" if opp_mode == "heuristic" else "R"
        t = threading.Thread(
            target=opp_player_loop,
            args=(f"{prefix}{i+1}", N, stop_evt, log),
            kwargs={"mode": opp_mode},
            daemon=True,
        )
        t.start()
        threads.append(t)
        time.sleep(0.15)  # stagger so colors get assigned in order

    # Spawn NEXUS via play_server.py (subprocess so we exercise the real path)
    env = os.environ.copy()
    env["PYTHONPATH"] = NEXUS_DIR
    nexus_log_path = os.path.join("/tmp", f"nexus_n{N}.log")
    nexus_log = open(nexus_log_path, "w")
    nexus = subprocess.Popen(
        [
            os.path.join(NEXUS_DIR, "venv", "bin", "python"),
            os.path.join(NEXUS_DIR, "scripts", "play_server.py"),
            "--model", model_path,
            "--name", "NEXUS",
            "--auto-start",
            "--device", "cuda",
        ],
        cwd=NEXUS_DIR,
        stdout=nexus_log,
        stderr=subprocess.STDOUT,
        env=env,
    )

    # Poll for game finish or hard cap
    start = time.time()
    game_id = None
    last_print = start
    finished = False
    final_state = None
    while time.time() - start < 90:  # hard cap = 90s (server limit is 60s)
        # Find a PLAYING/READY game on the server
        sr = rpc({"op": "status"})
        if sr.get("ok"):
            for g in sr["games"]:
                if g["status"] in ("PLAYING", "FINISHED") and len(g["players"]) == N:
                    game_id = g["game_id"]
                    if g["status"] == "FINISHED":
                        full = rpc({"op": "get_state", "game_id": game_id})
                        final_state = full.get("state")
                        finished = True
                        break
            if finished:
                break

        # progress print every 5s
        if time.time() - last_print > 5:
            if game_id:
                full = rpc({"op": "get_state", "game_id": game_id}).get("state", {})
                print(f"  [t={int(time.time()-start)}s] status={full.get('status')} "
                      f"moves={full.get('move_count')} "
                      f"turn={full.get('current_turn_colour')}")
            last_print = time.time()
        time.sleep(0.5)

    stop_evt.set()
    try:
        nexus.terminate()
        nexus.wait(timeout=5)
    except Exception:
        nexus.kill()
    nexus_log.close()

    if not finished:
        print(f"  RESULT: HUNG / NOT FINISHED within 90s - see {nexus_log_path}")
        stop_server(server_proc)
        return {"N": N, "ok": False, "reason": "timeout-no-finish"}

    print(f"  RESULT: FINISHED in {int(time.time()-start)}s, {final_state['move_count']} moves")
    print(f"  Turn order: {final_state.get('turn_order')}")
    print(f"  Scores:")
    nexus_score = None
    for pl in final_state["players"]:
        sc = pl.get("score") or {}
        marker = " <- NEXUS" if pl["name"] == "NEXUS" else ""
        print(f"    {pl['name']:8s} ({pl['colour']:11s}): "
              f"final={sc.get('final_score',0):.1f} "
              f"pins={sc.get('pin_goal_score',0):.0f} "
              f"dist={sc.get('distance_score',0):.1f} "
              f"moves={sc.get('moves',0)}{marker}")
        if pl["name"] == "NEXUS":
            nexus_score = sc

    # Rank NEXUS
    ranked = sorted(
        final_state["players"],
        key=lambda p: (p.get("score") or {}).get("final_score", 0),
        reverse=True,
    )
    nexus_rank = next(i + 1 for i, p in enumerate(ranked) if p["name"] == "NEXUS")
    print(f"  NEXUS rank: {nexus_rank}/{N}")

    stop_server(server_proc)
    return {"N": N, "ok": True, "rank": nexus_rank, "score": nexus_score,
            "log": nexus_log_path}


def main():
    model = os.path.join(NEXUS_DIR, "checkpoints", "phase2_best.pt")
    opp_mode = "random"
    args = sys.argv[1:]
    if args and args[0] in ("random", "heuristic"):
        opp_mode = args.pop(0)
    if args:
        model = args[0]
    print(f"Model: {model}")
    print(f"Opponent mode: {opp_mode}")
    print(f"Teacher game.py at: {TEACHER_DIR}")

    results = []
    for N in [2, 3, 4, 5, 6]:
        r = run_match(N, model, opp_mode=opp_mode)
        results.append(r)
        time.sleep(1.0)

    print("\n\n" + "="*60)
    print(" SUMMARY")
    print("="*60)
    for r in results:
        if r["ok"]:
            sc = r["score"] or {}
            print(f"  N={r['N']}: rank {r['rank']}/{r['N']}, "
                  f"final={sc.get('final_score',0):.1f}, "
                  f"pins={sc.get('pin_goal_score',0):.0f}, "
                  f"moves={sc.get('moves',0)}")
        else:
            print(f"  N={r['N']}: {r['reason']}")


if __name__ == "__main__":
    main()
