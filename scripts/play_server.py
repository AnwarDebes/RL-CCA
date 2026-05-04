#!/usr/bin/env python3
"""NEXUS tournament client - connects to the teacher's game server.

Usage:
  python scripts/play_server.py [--model checkpoints/phase2_best.pt]
                                [--host 127.0.0.1] [--port 50555]
                                [--name NEXUS]

The server protocol uses one-shot TCP connections (not persistent).
Each RPC call opens a new socket, sends JSON, receives JSON, closes.
"""
import sys
sys.path.insert(0, '/home/coder/nexus')

import argparse
import json
import os
import socket
import time

from inference.tournament_player import NexusTournamentPlayer


# ── JSON-RPC helpers (one-shot TCP, matching teacher's server) ───

def rpc(payload: dict, host: str = "127.0.0.1", port: int = 50555) -> dict:
    """Send JSON to server and receive JSON reply."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(10.0)
    try:
        s.connect((host, port))
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
            # Try parsing - if valid JSON, we have the full response
            try:
                json.loads(b''.join(chunks).decode("utf-8"))
                break
            except json.JSONDecodeError:
                continue
        except socket.timeout:
            break
    s.close()

    raw = b''.join(chunks)
    if not raw:
        return {"ok": False, "error": "no-response"}
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception as e:
        return {"ok": False, "error": f"bad-json: {e}"}


# ── Main loop ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NEXUS tournament player")
    parser.add_argument("--model", default="checkpoints/phase2_best.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50555)
    parser.add_argument("--name", default="NEXUS")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--auto-start", action="store_true",
                        help="Auto-send START without waiting for Enter")
    args = parser.parse_args()

    # Load model
    model_path = args.model if os.path.exists(args.model) else None
    if model_path:
        print(f"[NEXUS] Loading model from {model_path}")
    else:
        print(f"[NEXUS] WARNING: No model found at {args.model}, using heuristic only")

    player = NexusTournamentPlayer(model_path=model_path, device=args.device)
    RPC = lambda payload: rpc(payload, args.host, args.port)

    # JOIN
    print(f"[NEXUS] Joining as '{args.name}'...")
    r = RPC({"op": "join", "player_name": args.name})
    if not r.get("ok"):
        print(f"[NEXUS] JOIN ERROR: {r.get('error')}")
        return

    game_id = r["game_id"]
    player_id = r["player_id"]
    my_colour = r["colour"]
    print(f"[NEXUS] Joined game {game_id[:8]}... as {my_colour}")

    # Wait for READY_TO_START
    while True:
        st = RPC({"op": "get_state", "game_id": game_id})
        status = st.get("state", {}).get("status", "")
        if status in ("READY_TO_START", "PLAYING"):
            break
        print("[NEXUS] Waiting for players...")
        time.sleep(0.5)

    # Send START
    if args.auto_start:
        print("[NEXUS] Auto-starting...")
    else:
        input("[NEXUS] Press ENTER to send START...")
    RPC({"op": "start", "game_id": game_id, "player_id": player_id})
    print("[NEXUS] Sent START")

    # Wait for PLAYING
    while True:
        st = RPC({"op": "get_state", "game_id": game_id})
        if st.get("state", {}).get("status") == "PLAYING":
            break
        time.sleep(0.5)

    # Get all player colors
    state = st["state"]
    all_colors = [p["colour"] for p in state["players"]]
    player.set_color(my_colour, all_colors)

    print(f"[NEXUS] === GAME STARTED ===")
    print(f"[NEXUS] Players: {all_colors}")
    print(f"[NEXUS] Turn order: {state.get('turn_order', [])}")
    print(f"[NEXUS] My color: {my_colour}, goal: {player.opp_color} zone")

    last_move_seen = 0
    total_time_spent = 0.0

    while True:
        st = RPC({"op": "get_state", "game_id": game_id})
        if not st.get("ok"):
            print(f"[NEXUS] Error: {st.get('error')}")
            time.sleep(0.5)
            continue

        state = st["state"]

        # Timeout notice
        if state.get("turn_timeout_notice"):
            print(f"[NEXUS] TIMEOUT: {state['turn_timeout_notice']}")

        # Game finished?
        if state["status"] == "FINISHED":
            print(f"\n[NEXUS] === GAME FINISHED ===")
            for pl in state["players"]:
                sc = pl.get("score")
                if sc:
                    marker = " <-- ME" if pl["colour"] == my_colour else ""
                    print(
                        f"  {pl['name']} ({pl['colour']}): "
                        f"{sc['final_score']:.1f} "
                        f"[time={sc['time_score']:.1f}, "
                        f"moves({sc['moves']})={sc['move_score']:.1f}, "
                        f"pins={sc['pin_goal_score']:.1f}, "
                        f"dist={sc['distance_score']:.1f}]{marker}"
                    )
            print(f"[NEXUS] Total thinking time: {total_time_spent:.2f}s")
            break

        # Show moves
        if state["move_count"] > last_move_seen:
            mv = state.get("last_move")
            if mv:
                marker = " (ME)" if mv["colour"] == my_colour else ""
                print(
                    f"  Move {state['move_count']}: {mv['by']} ({mv['colour']}) "
                    f"{mv['from']}->{mv['to']} [{mv['move_ms']:.1f}ms]{marker}"
                )
            last_move_seen = state["move_count"]

        # Our turn?
        if state.get("current_turn_colour") == my_colour and state["status"] == "PLAYING":
            move_start = time.time()

            # Get legal moves from server
            legal_req = RPC({
                "op": "get_legal_moves",
                "game_id": game_id,
                "player_id": player_id
            })

            if not legal_req.get("ok"):
                print(f"[NEXUS] Error getting legal moves: {legal_req.get('error')}")
                time.sleep(0.5)
                continue

            legal_moves = legal_req.get("legal_moves", {})
            movable = {pid: moves for pid, moves in legal_moves.items() if moves}

            if not movable:
                print("[NEXUS] No legal moves available!")
                time.sleep(0.5)
                continue

            # Get current board state
            state_pins = state.get("pins", {})

            # Choose move
            pin_id, dest = player.choose_move(state_pins, legal_moves)

            move_time = time.time() - move_start
            total_time_spent += move_time
            remaining = player.time_manager.remaining_total()

            print(
                f"  [NEXUS] -> pin {pin_id} to cell {dest} "
                f"({move_time:.3f}s, total={total_time_spent:.1f}s, "
                f"budget_left={remaining:.1f}s)"
            )

            # Send move
            mv_resp = RPC({
                "op": "move",
                "game_id": game_id,
                "player_id": player_id,
                "pin_id": pin_id,
                "to_index": dest
            })

            if not mv_resp.get("ok"):
                print(f"[NEXUS] Move rejected: {mv_resp.get('error')}")
                # Fallback: try heuristic
                print("[NEXUS] Trying heuristic fallback...")
                pin_id, dest = player._heuristic_move(state_pins, legal_moves)
                mv_resp = RPC({
                    "op": "move",
                    "game_id": game_id,
                    "player_id": player_id,
                    "pin_id": pin_id,
                    "to_index": dest
                })
                if not mv_resp.get("ok"):
                    print(f"[NEXUS] CRITICAL: Heuristic also rejected: {mv_resp.get('error')}")
            else:
                if mv_resp.get("status") == "WIN":
                    print(f"[NEXUS] *** WE WON! *** {mv_resp.get('msg')}")
                elif mv_resp.get("status") == "DRAW":
                    print(f"[NEXUS] DRAW: {mv_resp.get('msg')}")

        time.sleep(0.3)  # poll interval


if __name__ == "__main__":
    main()
