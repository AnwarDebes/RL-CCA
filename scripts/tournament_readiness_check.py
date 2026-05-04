#!/usr/bin/env python3
"""Tournament readiness check - run this anytime to confirm the agent
is good to ship. Exits 0 if ready, 1 if not.

Checks:
  1. The chosen checkpoint loads without error
  2. Forward pass produces valid output
  3. Greedy game (self vs self) ends with a winner (proves argmax works)
  4. Round-trip action encoding has 0 errors
  5. State encoder works for all N=2..6
  6. Network's policy never outputs illegal actions
  7. Tournament_player can load + choose_move with mock state
"""
from __future__ import annotations
import os
import sys
import time

NEXUS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, NEXUS)

import argparse
import random
import torch
import numpy as np

from config import Config
from core.action_space import (
    encode_action, decode_action,
    build_legal_mask_from_server, decode_action_to_server,
)
from core.board import HexBoard
from core.game_env import GameEnv
from core.state_encoder import StateEncoder
from network.model import NexusNet
from network.model_v3 import NexusNetV3


def _load_auto(path: str, device):
    sd = torch.load(path, map_location="cpu", weights_only=True)
    is_v3 = any(k.startswith("backbone.res_blocks_a.") for k in sd.keys())
    if is_v3:
        m = NexusNetV3().to(device)
        m.load_state_dict(sd)
        return m, True
    m = NexusNet.load(path, str(device))
    return m, False


def colored(s, color):
    code = {"green": "\033[32m", "red": "\033[31m", "yellow": "\033[33m",
            "bold": "\033[1m", "reset": "\033[0m"}
    return f"{code.get(color, '')}{s}{code['reset']}"


def check(name, fn):
    sys.stdout.write(f"  {name:50s}")
    sys.stdout.flush()
    try:
        result = fn()
        if result is True or result is None:
            print(colored("✓ PASS", "green"))
            return True
        else:
            print(colored(f"✗ FAIL: {result}", "red"))
            return False
    except Exception as e:
        print(colored(f"✗ ERROR: {e}", "red"))
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(NEXUS, "checkpoints_v2", "phase2_best.pt"))
    args = ap.parse_args()

    print(colored(f"=== Tournament Readiness Check ({time.strftime('%Y-%m-%d %H:%M:%S')}) ===", "bold"))
    print(f"Model: {args.model}\n")

    if not os.path.exists(args.model):
        print(colored(f"FATAL: model not found: {args.model}", "red"))
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    board = HexBoard()
    failed = 0

    # 1. Checkpoint loads (auto-detects v2 vs v3)
    network = [None]
    is_v3_holder = [False]
    def _load():
        net, is_v3 = _load_auto(args.model, device)
        net.eval()
        network[0] = net
        is_v3_holder[0] = is_v3
        return True
    if not check("[1] Checkpoint loads", _load):
        failed += 1
    net = network[0]
    is_v3 = is_v3_holder[0]
    if net is not None:
        print(f"  Detected: {'v3 (NexusNetV3)' if is_v3 else 'v2 (NexusNet)'}")

    if net is None:
        print(colored("\nCannot continue without model.", "red"))
        return 1

    # 2. Forward pass produces valid output
    def _forward():
        s = torch.zeros(1, Config.NUM_CHANNELS, 17, 17, device=device)
        m = torch.ones(1, 1210, dtype=torch.bool, device=device)
        with torch.no_grad():
            out = net(s, m)
        if out["policy"].shape != (1, 1210): return f"bad policy shape {out['policy'].shape}"
        if out["value"].shape != (1,): return f"bad value shape {out['value'].shape}"
        if not (-1 <= float(out["value"][0]) <= 1): return f"value out of range: {float(out['value'][0])}"
        return True
    if not check("[2] Forward pass valid output", _forward):
        failed += 1

    # 3. Greedy game ends with winner (argmax works)
    def _greedy():
        env = GameEnv(board, num_players=2)
        env.reset()
        moves = 0
        while not env.is_done() and moves < 1500:
            p = env.current_player
            s = env.get_state_tensor(p).unsqueeze(0).to(device)
            m = env.get_legal_mask(p).unsqueeze(0).to(device)
            with torch.no_grad():
                out = net(s, m)
            action = int(out["policy"][0].argmax())
            env.step(action)
            moves += 1
        if env.winner is None:
            return f"greedy game timed out at {moves} moves with no winner (argmax broken)"
        return True
    if not check("[3] Greedy 2-player game has winner", _greedy):
        failed += 1

    # 4. Round-trip action encoding (server <-> nexus)
    def _round_trip():
        rng = random.Random(42)
        for _ in range(200):
            pin_positions = rng.sample(range(121), 10)
            server_legal = {str(pid): rng.sample([c for c in range(121) if c != pin_positions[pid]], rng.randint(1, 4))
                            for pid in range(10) if rng.random() > 0.3}
            if not server_legal:
                continue
            mask = build_legal_mask_from_server(server_legal, pin_positions)
            for action in mask.nonzero(as_tuple=False).squeeze(-1).tolist():
                pin_id, dest = decode_action_to_server(action, pin_positions)
                if str(pin_id) not in server_legal or dest not in server_legal[str(pin_id)]:
                    return f"round-trip mismatch on action {action}"
        return True
    if not check("[4] Action encoding round-trip", _round_trip):
        failed += 1

    # 5. State encoder works for all N
    def _all_n():
        for N in [2, 3, 4, 5, 6]:
            env = GameEnv(board, num_players=N)
            env.reset()
            t = env.get_state_tensor()
            if t.shape != (Config.NUM_CHANNELS, 17, 17):
                return f"N={N} bad encoder shape {t.shape}"
        return True
    if not check("[5] State encoder works for N=2..6", _all_n):
        failed += 1

    # 6. Policy never outputs illegal actions (100 random tests)
    def _legal():
        rng = random.Random(123)
        errors = 0
        for _ in range(100):
            mask = torch.zeros(1, 1210, dtype=torch.bool, device=device)
            n_legal = rng.randint(1, 30)
            legal_idx = rng.sample(range(1210), n_legal)
            for i in legal_idx:
                mask[0, i] = True
            s = torch.randn(1, Config.NUM_CHANNELS, 17, 17, device=device)
            with torch.no_grad():
                out = net(s, mask)
            if float(out["policy"][0][~mask[0]].sum()) > 1e-5:
                errors += 1
        if errors > 0:
            return f"{errors}/100 trials had illegal probability mass"
        return True
    if not check("[6] Policy mass on illegal actions = 0", _legal):
        failed += 1

    # 7. TournamentPlayer end-to-end
    def _tp():
        from inference.tournament_player import NexusTournamentPlayer
        tp = NexusTournamentPlayer(model_path=args.model, device=str(device))
        tp.set_color("red", ["red", "blue"])
        # Mock server state
        state_pins = {
            "red": sorted(board.start_zones["red"])[:10],
            "blue": sorted(board.start_zones["blue"])[:10],
        }
        # Mock legal moves: every red pin can move to one cell ahead
        legal_moves = {str(i): [pos - 17] for i, pos in enumerate(state_pins["red"]) if (pos - 17) >= 0}
        if not legal_moves:
            legal_moves = {"0": [100]}
        pin_id, dest = tp.choose_move(state_pins, legal_moves)
        if not (0 <= pin_id < 10):
            return f"bad pin_id {pin_id}"
        return True
    if not check("[7] TournamentPlayer.choose_move works", _tp):
        failed += 1

    print()
    if failed == 0:
        print(colored(f"=== ALL CHECKS PASSED - agent is tournament-ready ===", "green"))
        return 0
    else:
        print(colored(f"=== {failed} CHECK(S) FAILED - agent NOT ready ===", "red"))
        return 1


if __name__ == "__main__":
    sys.exit(main())
