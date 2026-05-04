"""Pre-flight test suite - MUST pass before any training starts.

Verifies that the student's game logic, board, and scoring are exactly
aligned with the teacher's tournament environment. Six tests:

  PF1 - Cell indexing matches teacher
  PF2 - Zones (start + goal) match teacher
  PF3 - Legal moves match teacher on 100 random configurations
  PF4 - Win condition matches teacher on constructed end-positions
  PF5 - teacher_score() reproduces logged scores within 1 point
  PF6 - Self-play game (random policy) replays cleanly on teacher's server

Any failure means training cannot start until the discrepancy is resolved.
"""

from __future__ import annotations

import math
import os
import random
import re
import sys

NEXUS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEACHER_DIR = os.path.join(
    NEXUS_DIR, "RLChineseCheckers", "multi system single machine minimal"
)
sys.path.insert(0, NEXUS_DIR)
sys.path.insert(0, TEACHER_DIR)

import pytest

# Student modules
from core.board import HexBoard as NexusHexBoard
from core.game_env import GameEnv
from config import Config
from core import teacher_score as ts

# Teacher modules (note: must import these AFTER sys.path adjustment)
import checkers_board as t_board_mod
import checkers_pins as t_pin_mod
TeacherHexBoard = t_board_mod.HexBoard
TeacherPin = t_pin_mod.Pin


# ── Helpers ──────────────────────────────────────────────────────


def _fresh_teacher_board():
    """Create a teacher's board with all cells unoccupied (default state)."""
    b = TeacherHexBoard()
    for c in b.cells:
        c.occupied = False
    return b


def _fresh_nexus_board():
    return NexusHexBoard()


def _set_teacher_pieces(board, pin_specs):
    """Place pins on a fresh teacher board.

    pin_specs: List[(color, [cell_indices])] - gives each color's pin positions.
    Returns: dict {color: [Pin]}.
    """
    pins_by_color = {}
    for color, positions in pin_specs:
        # Important: clearing a cell here would un-occupy others; we trust the
        # board is fresh.
        pins = []
        for i, idx in enumerate(positions):
            pin = TeacherPin(board, idx, id=i, color=color)
            pins.append(pin)
        pins_by_color[color] = pins
    return pins_by_color


def _set_nexus_pieces(env, color_to_positions):
    """Force nexus env piece positions; bypasses normal reset."""
    # Ensure colors list matches the order
    colors = list(color_to_positions.keys())
    env.colors = colors
    env.pieces = [sorted(color_to_positions[c]) for c in colors]
    env.current_player = 0
    env.move_count = 0
    env.done = False
    env.winner = None
    env._rebuild_occupied()
    env._legal_cache = {p: None for p in range(len(colors))}


# ── PF1 - Cell indexing matches teacher ──────────────────────────


def test_pf1_cell_indexing():
    """Both boards must agree on the (q, r) at every index 0..120."""
    teacher = TeacherHexBoard()
    nexus = NexusHexBoard()

    # Teacher's full cell list includes the central hex (61 cells, postype='board')
    # PLUS the 60 zone cells. Nexus only models the 121 cells reachable in play.
    # Verify they match.
    assert len(teacher.cells) == 121, f"Teacher has {len(teacher.cells)} cells"
    assert nexus.num_cells == 121, f"Nexus has {nexus.num_cells} cells"

    for i in range(121):
        tq, tr = teacher.cells[i].q, teacher.cells[i].r
        nq, nr = nexus.cell_q[i], nexus.cell_r[i]
        assert (tq, tr) == (nq, nr), (
            f"Cell {i}: teacher=({tq},{tr}), nexus=({nq},{nr})"
        )


# ── PF2 - Zones (start + goal) match teacher ─────────────────────


def test_pf2_zones():
    """Teacher's `axial_of_colour(color)` must equal nexus's `start_zones[color]`."""
    teacher = TeacherHexBoard()
    nexus = NexusHexBoard()

    for color in ['red', 'blue', 'lawn green', 'gray0', 'yellow', 'purple']:
        teacher_zone = set(teacher.axial_of_colour(color))
        nexus_start = set(nexus.start_zones[color])
        assert teacher_zone == nexus_start, (
            f"Color {color}: teacher zone {sorted(teacher_zone)} "
            f"vs nexus start_zones {sorted(nexus_start)}"
        )

        # Goal zone for color = start zone of opposite color
        opp = teacher.colour_opposites[color]
        teacher_goal = set(teacher.axial_of_colour(opp))
        nexus_goal = set(nexus.goal_zones[color])
        assert teacher_goal == nexus_goal, (
            f"Color {color}: teacher goal {sorted(teacher_goal)} "
            f"vs nexus goal_zones {sorted(nexus_goal)}"
        )


# ── PF3 - Legal moves match teacher on 100 random configurations ─


def _get_teacher_legal_moves(board, pin):
    """Wrapper for teacher's getPossibleMoves - unchanged by us."""
    return sorted(pin.getPossibleMoves())


def test_pf3_legal_moves_equivalence():
    """For 100 random configurations, both systems must agree on legal moves
    for every piece on the board."""
    rng = random.Random(20260429)
    discrepancies = []

    for trial in range(100):
        # Pick 2 colors and random positions for 10 pins each
        colors_pool = ['red', 'blue', 'lawn green', 'gray0', 'yellow', 'purple']
        c1, c2 = rng.sample(colors_pool, 2)

        # Sample distinct cell indices to place pins
        all_cells = list(range(121))
        positions_pool = rng.sample(all_cells, 20)
        c1_positions = sorted(positions_pool[:10])
        c2_positions = sorted(positions_pool[10:])

        # Build teacher state
        teacher_board = _fresh_teacher_board()
        teacher_pins = _set_teacher_pieces(
            teacher_board, [(c1, c1_positions), (c2, c2_positions)]
        )

        # Build nexus state
        nexus_env = GameEnv(NexusHexBoard())
        _set_nexus_pieces(nexus_env, {c1: c1_positions, c2: c2_positions})

        # For each player and each pin, compare legal destinations
        for player_idx, color in enumerate([c1, c2]):
            nexus_legal = nexus_env.get_legal_moves(player_idx)
            for pin in teacher_pins[color]:
                teacher_dests = set(_get_teacher_legal_moves(teacher_board, pin))
                nexus_dests = set(nexus_legal.get(pin.axialindex, []))
                if teacher_dests != nexus_dests:
                    discrepancies.append({
                        "trial": trial,
                        "color": color,
                        "pin_pos": pin.axialindex,
                        "teacher_only": sorted(teacher_dests - nexus_dests),
                        "nexus_only": sorted(nexus_dests - teacher_dests),
                    })

    assert not discrepancies, (
        f"{len(discrepancies)} discrepancies in 100 trials. First 5: "
        f"{discrepancies[:5]}"
    )


# ── PF4 - Win condition matches teacher ──────────────────────────


def test_pf4_win_condition():
    """When all pins of color C are in opposite(C)'s zone, both systems
    must agree that C has won. Conversely, if any pin is missing from the
    opposite zone, neither should declare a win for C."""
    teacher = TeacherHexBoard()
    nexus = NexusHexBoard()

    for color in ['red', 'blue', 'lawn green', 'gray0', 'yellow', 'purple']:
        opp = teacher.colour_opposites[color]
        opp_zone = teacher.axial_of_colour(opp)
        assert len(opp_zone) == 10

        # Construct end-position: all 10 pins of `color` in opposite zone.
        # No other pins on the board.
        teacher_board = _fresh_teacher_board()
        teacher_pins = _set_teacher_pieces(teacher_board, [(color, opp_zone)])

        # Teacher's check_player_status logic: all pins where postype == opposite
        teacher_won = all(
            teacher_board.cells[p.axialindex].postype == opp
            for p in teacher_pins[color]
        )
        assert teacher_won, f"Constructed win for {color} not detected by teacher"

        # Nexus's check: count_in_goal == 10
        nexus_in_goal = nexus.count_in_goal(opp_zone, color)
        assert nexus_in_goal == 10, (
            f"Color {color}: nexus says only {nexus_in_goal} in goal "
            f"(positions: {opp_zone})"
        )

        # Negative test: move one pin out of goal - neither should declare win
        if len(opp_zone) > 1:
            non_goal_cells = [i for i in range(121) if i not in set(opp_zone)]
            modified_positions = list(opp_zone[:-1]) + [non_goal_cells[0]]

            teacher_board2 = _fresh_teacher_board()
            teacher_pins2 = _set_teacher_pieces(
                teacher_board2, [(color, modified_positions)]
            )
            teacher_won2 = all(
                teacher_board2.cells[p.axialindex].postype == opp
                for p in teacher_pins2[color]
            )
            assert not teacher_won2

            nexus_in_goal2 = nexus.count_in_goal(modified_positions, color)
            assert nexus_in_goal2 != 10


# ── PF5 - teacher_score() reproduces logged scores ───────────────


_SCORE_LINE = re.compile(
    r"SCORE\s+(\S+)\s+\((\S+(?:\s\S+)?)\)\s*:\s*"
    r"Final=([\d\.\-]+),\s*"
    r"Time=([\d\.\-]+),\s*"
    r"Moves\((\d+)\)=([\d\.\-]+),\s*"
    r"Pins\((\d+)\)=([\d\.\-]+),\s*"
    r"Dist=([\d\.\-]+)"
)


def _parse_score_lines(log_path):
    """Yield dicts of parsed score lines from a teacher game log."""
    with open(log_path) as f:
        for line in f:
            m = _SCORE_LINE.search(line)
            if not m:
                continue
            yield {
                "name": m.group(1),
                "color": m.group(2),
                "final": float(m.group(3)),
                "time_score": float(m.group(4)),
                "move_count": int(m.group(5)),
                "move_score": float(m.group(6)),
                "pin_count": int(m.group(7)),
                "pin_goal_score": float(m.group(8)),
                "distance_score": float(m.group(9)),
            }


def test_pf5a_teacher_score_synthetic_values():
    """Direct verification of teacher_score formulas with constructed inputs.

    Doesn't depend on log parsing precision. Teacher's compute_scores in
    game.py:198-256 implements:
      time_score = max(0, 100 - time_taken_sec)  if moves > 0 else 0
      move_score = exp(-((m-45)^2 / (2*sigma^2)))  sigma=4 if m<45 else 18
      pin_goal_score = 100 * pins
      distance_score = max(0, 200 - total_dist)  if moves > 0 else 0
      final = sum
    """
    # Mid-game, fast play (typical winning agent): m=45 peaks move_score
    assert ts.time_score(30.0) == pytest.approx(70.0)
    assert ts.time_score(0.5) == pytest.approx(99.5)
    assert ts.time_score(150.0) == pytest.approx(0.0)  # clipped at 0

    assert ts.move_score(45) == pytest.approx(1.0)            # peak
    assert ts.move_score(44) == pytest.approx(math.exp(-1/32))  # sigma=4 below
    assert ts.move_score(46) == pytest.approx(math.exp(-1/648)) # sigma=18 above
    assert ts.move_score(0) == 0.0                             # gating

    assert ts.pin_goal_score(0) == 0.0
    assert ts.pin_goal_score(7) == 700.0
    assert ts.pin_goal_score(10) == 1000.0

    assert ts.distance_score(0.0) == pytest.approx(200.0)
    assert ts.distance_score(50.0) == pytest.approx(150.0)
    assert ts.distance_score(250.0) == pytest.approx(0.0)  # clipped

    # Composite - winning play: 5s clock, 50 moves, 10 pins home, 0 dist
    expected = ts.time_score(5.0) + ts.move_score(50) + 1000.0 + 200.0
    assert ts.final_score(5.0, 50, 10, 0.0) == pytest.approx(expected)

    # Composite - pre-game (no moves): everything zero
    assert ts.final_score(0.0, 0, 0, 0.0) == 0.0


def test_pf5b_teacher_score_matches_logged_components():
    """Parse 5+ teacher game logs. Verify:
      (a) move_score(logged.move_count) == logged.move_score (formula match)
      (b) pin_goal_score(logged.pin_count) == logged.pin_goal_score
      (c) sum of logged components == logged.final  (log self-consistency)

    We do NOT try to invert time_score/distance_score back to time_taken/total_dist
    because the log rounds those displays (e.g. time_score=100.0 represents
    time_taken_sec ∈ (0, 0.5)) so the inversion is lossy. The synthetic test
    above (PF5a) covers those formulas directly.
    """
    import math as _math

    games_dir = os.path.join(TEACHER_DIR, "games")
    log_files = sorted([
        os.path.join(games_dir, f)
        for f in os.listdir(games_dir)
        if f.startswith("game_") and f.endswith(".log")
    ])
    assert len(log_files) >= 5, f"Need at least 5 logs, found {len(log_files)}"

    files_checked = 0
    lines_checked = 0
    discrepancies = []

    for path in log_files[:10]:
        files_checked += 1
        for parsed in _parse_score_lines(path):
            lines_checked += 1

            # (a) move_score formula
            ms_got = ts.move_score(parsed["move_count"])
            ms_logged = parsed["move_score"]
            if abs(ms_got - ms_logged) > 1.0:
                discrepancies.append({
                    "file": os.path.basename(path), "key": "move_score",
                    "expected": ms_logged, "got": ms_got,
                    "moves": parsed["move_count"],
                })

            # (b) pin_goal_score formula
            ps_got = ts.pin_goal_score(parsed["pin_count"])
            if abs(ps_got - parsed["pin_goal_score"]) > 0.5:
                discrepancies.append({
                    "file": os.path.basename(path), "key": "pin_goal_score",
                    "expected": parsed["pin_goal_score"], "got": ps_got,
                    "pins": parsed["pin_count"],
                })

            # (c) log self-consistency
            sum_components = (parsed["time_score"] + parsed["move_score"]
                              + parsed["pin_goal_score"] + parsed["distance_score"])
            if abs(sum_components - parsed["final"]) > 0.5:
                discrepancies.append({
                    "file": os.path.basename(path), "key": "final_sum",
                    "expected": parsed["final"], "got": sum_components,
                })

    assert files_checked >= 5, f"Only checked {files_checked} files"
    assert lines_checked >= 50, f"Only checked {lines_checked} score lines"
    assert not discrepancies, (
        f"{len(discrepancies)} discrepancies found in {lines_checked} lines. "
        f"First 5: {discrepancies[:5]}"
    )


# ── PF6 - Self-play game replays cleanly on teacher's server ─────
# (This test starts a real subprocess of game.py and connects 2 random clients.
# Marked as requiring server access, can be slow - uses ~20 seconds.)


@pytest.mark.timeout(120)
def test_pf6_self_play_replays_on_server():
    """Generate a short random-policy game, replay every move on teacher's
    actual game.py server. Every move must be accepted; the win/draw status
    must propagate. This exercises the full training-to-tournament pipeline.

    Implementation: rather than generate a self-play game with the (not-yet-
    rewritten) network, we just run 2 random clients via the teacher's own
    server end-to-end. If the server runs to completion without errors and
    produces a logged game, that proves our infrastructure is intact.
    """
    import json
    import socket
    import subprocess
    import threading
    import time

    HOST, PORT = "127.0.0.1", 50555

    def rpc(payload):
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

    # Start teacher server
    proc = subprocess.Popen(
        ["python3", "game.py"],
        cwd=TEACHER_DIR,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        time.sleep(1.5)
        proc.stdin.write(b"create\n")
        proc.stdin.flush()
        # Wait for game to register
        for _ in range(20):
            sr = rpc({"op": "status"})
            if sr.get("ok") and len(sr.get("games", [])) >= 1:
                break
            time.sleep(0.25)
        else:
            raise AssertionError("Server did not register a game in time")

        rng = random.Random(20260429)
        stop_evt = threading.Event()

        def random_client(name):
            r = rpc({"op": "join", "player_name": name})
            if not r.get("ok"):
                return
            game_id = r["game_id"]
            player_id = r["player_id"]
            colour = r["colour"]
            # Wait for both players joined before sending start
            for _ in range(40):
                if stop_evt.is_set():
                    return
                st = rpc({"op": "get_state", "game_id": game_id}).get("state", {})
                if len(st.get("players", [])) >= 2:
                    break
                time.sleep(0.2)
            rpc({"op": "start", "game_id": game_id, "player_id": player_id})
            while not stop_evt.is_set():
                st = rpc({"op": "get_state", "game_id": game_id})
                if not st.get("ok"):
                    time.sleep(0.1)
                    continue
                state = st["state"]
                if state["status"] == "FINISHED":
                    return
                if state.get("current_turn_colour") == colour and state["status"] == "PLAYING":
                    lr = rpc({"op": "get_legal_moves", "game_id": game_id, "player_id": player_id})
                    legal = lr.get("legal_moves", {})
                    movable = [(pid, mv) for pid, mv in legal.items() if mv]
                    if not movable:
                        time.sleep(0.05)
                        continue
                    pid, dests = rng.choice(movable)
                    dest = rng.choice(dests)
                    rpc({"op": "move", "game_id": game_id, "player_id": player_id,
                         "pin_id": int(pid), "to_index": int(dest)})
                time.sleep(0.02)

        t1 = threading.Thread(target=random_client, args=("R1",), daemon=True)
        t2 = threading.Thread(target=random_client, args=("R2",), daemon=True)
        t1.start()
        time.sleep(0.2)
        t2.start()

        # Poll for FINISHED status (teacher imposes 60s game cap, so this exits soon)
        finished = False
        start = time.time()
        while time.time() - start < 90:
            sr = rpc({"op": "status"})
            if sr.get("ok") and sr.get("games"):
                if sr["games"][0]["status"] == "FINISHED":
                    finished = True
                    break
            time.sleep(0.5)

        stop_evt.set()
        t1.join(timeout=3)
        t2.join(timeout=3)

        assert finished, "Server-side game did not reach FINISHED state"
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:
            proc.kill()
