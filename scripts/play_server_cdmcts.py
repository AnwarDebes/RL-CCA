#!/usr/bin/env python3
"""CD-MCTS tournament client - connects to teacher's game server.

Same protocol as play_server.py but uses NexusTournamentPlayerCDMCTS
(the flagship CD-MCTS player) instead of the v2 player. Run after
training a CD-MCTS network on Chinese Checkers.

Usage:
  python scripts/play_server_cdmcts.py [--model checkpoints/cdmcts_cc/final.pt]
                                       [--host 127.0.0.1] [--port 50555]
                                       [--name CDMCTS]
                                       [--coalition-weight 0.5]

This file imports the JSON-RPC helpers from the existing play_server.py
to avoid code duplication.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, '/home/coder/nexus')

# Reuse the existing play_server.py helpers (JSON-RPC machinery).
# This avoids duplicating ~150 lines of identical protocol code.
from scripts.play_server import rpc

from flagship_coalition_mcts.src.tournament_player_cdmcts import (
    NexusTournamentPlayerCDMCTS,
)


def main():
    parser = argparse.ArgumentParser(description="CD-MCTS tournament player")
    parser.add_argument("--model", default="checkpoints/cdmcts_cc/final.pt")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50555)
    parser.add_argument("--name", default="CDMCTS")
    parser.add_argument("--device", default="cpu",
                        help="cpu or cuda (cpu is fine for inference; "
                             "the training-time choice doesn't bind here)")
    parser.add_argument("--coalition-weight", type=float, default=0.5)
    parser.add_argument("--auto-start", action="store_true")
    args = parser.parse_args()

    # Load model
    model_path = args.model if os.path.exists(args.model) else None
    if model_path:
        print(f"[CDMCTS] Loading model from {model_path}")
    else:
        print(f"[CDMCTS] No model at {args.model}; using untrained network "
              f"(will fall back to heuristic on each move)")

    player = NexusTournamentPlayerCDMCTS(
        model_path=model_path,
        device=args.device,
        coalition_weight=args.coalition_weight,
    )
    RPC = lambda payload: rpc(payload, args.host, args.port)

    # JOIN
    print(f"[CDMCTS] Joining as '{args.name}'...")
    r = RPC({"op": "join", "player_name": args.name})
    if not r.get("ok"):
        print(f"[CDMCTS] JOIN ERROR: {r.get('error')}")
        return

    game_id = r["game_id"]
    player_id = r["player_id"]
    my_colour = r["colour"]
    print(f"[CDMCTS] Joined game {game_id[:8]}... as {my_colour}")

    # Wait for READY_TO_START
    while True:
        st = RPC({"op": "get_state", "game_id": game_id})
        status = st.get("state", {}).get("status", "")
        if status in ("READY_TO_START", "PLAYING"):
            break
        time.sleep(0.5)

    if args.auto_start:
        print("[CDMCTS] Auto-starting...")
    else:
        input("[CDMCTS] Press ENTER to send START...")
    RPC({"op": "start", "game_id": game_id, "player_id": player_id})
    print("[CDMCTS] Sent START")

    while True:
        st = RPC({"op": "get_state", "game_id": game_id})
        if st.get("state", {}).get("status") == "PLAYING":
            break
        time.sleep(0.5)

    state = st["state"]
    all_colors = [p["colour"] for p in state["players"]]
    player.set_color(my_colour, all_colors)
    print(f"[CDMCTS] === GAME STARTED ===")
    print(f"[CDMCTS] Players: {all_colors}")
    print(f"[CDMCTS] My color: {my_colour}")

    last_move_seen = 0
    last_move_was_mine = False
    total_time_spent = 0.0

    while True:
        st = RPC({"op": "get_state", "game_id": game_id})
        if not st.get("ok"):
            time.sleep(0.5)
            continue
        state = st["state"]

        # Game finished?
        if state["status"] == "FINISHED":
            print(f"\n[CDMCTS] === GAME FINISHED ===")
            for pl in state["players"]:
                sc = pl.get("score")
                if sc:
                    marker = " <-- ME" if pl["colour"] == my_colour else ""
                    print(
                        f"  {pl['name']} ({pl['colour']}): "
                        f"{sc['final_score']:.1f}{marker}"
                    )
            print(f"[CDMCTS] Total thinking time: {total_time_spent:.2f}s")
            break

        # Show moves; track opponent's last move for subtree reuse.
        if state["move_count"] > last_move_seen:
            mv = state.get("last_move")
            if mv:
                is_mine = (mv["colour"] == my_colour)
                marker = " (ME)" if is_mine else ""
                print(
                    f"  Move {state['move_count']}: {mv['by']} ({mv['colour']}) "
                    f"{mv['from']}->{mv['to']} [{mv['move_ms']:.1f}ms]{marker}"
                )
                # Inform player of opponent's move for subtree reuse.
                if not is_mine:
                    # Convert (from, to) to a raw action id. The conversion
                    # depends on the action_space encoding; we use the same
                    # encode helper as in core/action_space.py.
                    try:
                        from core.action_space import encode_action_from_server
                        opp_action = encode_action_from_server(
                            mv["from"], mv["to"], my_colour, all_colors,
                        )
                        if opp_action is not None:
                            player.advance_with_opponent_action(opp_action)
                    except Exception as e:
                        # If encoding fails, the next choose_move just
                        # rebuilds from scratch - graceful.
                        pass
            last_move_seen = state["move_count"]

        # Our turn?
        if state.get("current_turn_colour") == my_colour and state["status"] == "PLAYING":
            move_start = time.time()
            legal_req = RPC({
                "op": "get_legal_moves",
                "game_id": game_id,
                "player_id": player_id,
            })
            if not legal_req.get("ok"):
                time.sleep(0.5)
                continue
            legal_moves = legal_req.get("legal_moves", {})
            movable = {pid: moves for pid, moves in legal_moves.items() if moves}
            if not movable:
                print("[CDMCTS] No legal moves available!")
                time.sleep(0.5)
                continue

            # Build state_pins from the server's state representation.
            state_pins = {p["colour"]: p["pin_positions"] for p in state["players"]}
            plies = state.get("move_count", 0)

            try:
                move = player.choose_move(
                    state_pins,
                    legal_moves,
                    plies=plies,
                    time_remaining=state.get("time_remaining", 60.0),
                )
            except Exception as e:
                print(f"[CDMCTS] choose_move failed: {e}; passing turn.")
                time.sleep(1.0)
                continue

            elapsed = time.time() - move_start
            total_time_spent += elapsed

            r = RPC({
                "op": "make_move",
                "game_id": game_id,
                "player_id": player_id,
                "pin_id": move.get("pin_id"),
                "to": move.get("to"),
            })
            if not r.get("ok"):
                print(f"[CDMCTS] make_move failed: {r.get('error')}")
                time.sleep(0.5)
        else:
            time.sleep(0.2)


if __name__ == "__main__":
    main()
