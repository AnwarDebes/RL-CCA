#!/usr/bin/env python3
"""Deep stress test for pin_id <-> piece_id mapping.

Plays 30 full games using the teacher's actual board/pin classes.
Every NEXUS move is validated against teacher's getPossibleMoves().
Every encode/decode roundtrip is verified.
"""
import sys, os
sys.path.insert(0, '/home/coder/nexus')
sys.path.insert(0, '/home/coder/nexus/RLChineseCheckers/multi system single machine minimal')

from checkers_board import HexBoard as TeacherBoard
from checkers_pins import Pin
from core.action_space import (
    encode_action, decode_action,
    build_legal_mask_from_server, decode_action_to_server,
    get_legal_actions,
)
import random

# We don't use NexusTournamentPlayer here to avoid slow model inference.
# Instead we test the MAPPING LOGIC directly with random moves through the mask.

class Mute:
    def write(self, s): pass
    def flush(self): pass

def main():
    total_moves = 0
    total_errors = 0
    total_divs = 0

    for g in range(30):
        pairs = [('red','blue'),('lawn green','gray0'),('yellow','purple')]
        c1, c2 = pairs[g % 3]
        if g % 2 == 1:
            c1, c2 = c2, c1

        tb = TeacherBoard()
        tp = {}
        for c in [c1, c2]:
            idxs = tb.axial_of_colour(c)[:10]
            tp[c] = [Pin(tb, idxs[i], id=i, color=c) for i in range(10)]

        turns = [c1, c2]
        ti = 0
        gm = 0
        ge = 0
        gd = 0

        for _ in range(200):
            cc = turns[ti % 2]

            # Server state
            pin_positions = [p.axialindex for p in tp[cc]]
            sorted_positions = sorted(pin_positions)
            if pin_positions != sorted_positions:
                gd += 1

            # Server legal moves
            sl = {}
            has_moves = False
            for i, pin in enumerate(tp[cc]):
                m = pin.getPossibleMoves()
                sl[str(i)] = m
                if m:
                    has_moves = True
            if not has_moves:
                break

            # Build NEXUS legal mask from server data
            mask = build_legal_mask_from_server(sl, pin_positions)
            legal_actions = get_legal_actions(mask)

            if not legal_actions:
                break

            # Pick random legal action from mask
            action = random.choice(legal_actions)

            # Decode to server format
            pid, dst = decode_action_to_server(action, pin_positions)

            # VALIDATE: is this move legal per teacher?
            if pid < 0 or pid >= 10:
                print(f'Game {g}: INVALID pid={pid}')
                ge += 1
                break

            teacher_legal = tp[cc][pid].getPossibleMoves()
            if dst not in teacher_legal:
                print(f'Game {g} move {gm}: ILLEGAL! pid={pid} dst={dst}')
                print(f'  pin at cell {tp[cc][pid].axialindex}')
                print(f'  teacher legal: {teacher_legal}')
                print(f'  pin_positions: {pin_positions}')
                print(f'  sorted:        {sorted_positions}')
                ge += 1
                break

            # VALIDATE roundtrip
            piece_id, dest2 = decode_action(action)
            sp = sorted(pin_positions)
            expected_cell = sp[piece_id]
            expected_pid = pin_positions.index(expected_cell)
            if expected_pid != pid or dest2 != dst:
                print(f'Game {g}: ROUNDTRIP FAIL')
                ge += 1
                break

            # Apply move on teacher board
            old_stdout = sys.stdout
            sys.stdout = Mute()
            ok = tp[cc][pid].placePin(dst)
            sys.stdout = old_stdout

            if not ok:
                print(f'Game {g}: placePin FAILED')
                ge += 1
                break

            gm += 1
            ti += 1

            # Check win
            opp = tb.colour_opposites[cc]
            if all(tb.cells[p.axialindex].postype == opp for p in tp[cc]):
                break

        total_moves += gm
        total_errors += ge
        total_divs += gd
        s = 'OK' if ge == 0 else '** FAIL **'
        print(f'Game {g:2d} ({c1:12s} vs {c2:12s}): {gm:3d} moves, '
              f'diverged={gd:3d} turns, {s}')

    print()
    print('='*60)
    print(f'TOTAL: {total_moves} moves validated')
    print(f'ERRORS: {total_errors}')
    print(f'TURNS WITH SORT DIVERGENCE: {total_divs}')
    print(f'  (turns where pin_id != piece_id, i.e. mapping exercised)')
    print()
    if total_errors == 0:
        print('VERDICT: PIN ID MAPPING IS 100% CORRECT')
    else:
        print('VERDICT: *** MAPPING HAS BUGS ***')
    print('='*60)


if __name__ == '__main__':
    main()
