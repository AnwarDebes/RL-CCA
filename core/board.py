"""Hex board representation for Chinese Checkers (star board, 121 cells).

Optimized with precomputed distance lookup tables and grid mappings.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple

from config import Config


class HexBoard:
    """
    Chinese Checkers star board with 121 cells in axial coordinates.

    The star board is a hexagon of radius 4 (61 cells) plus 6 triangular
    tips of 10 cells each = 121 total cells.

    All distance lookups use precomputed tables - O(1) per query.
    """

    def __init__(self):
        # Core data - flat lists for speed
        self.cell_q: List[int] = []
        self.cell_r: List[int] = []
        self.cell_gx: List[int] = []   # precomputed grid x
        self.cell_gy: List[int] = []   # precomputed grid y
        self.index_of: Dict[Tuple[int, int], int] = {}
        self.neighbors: List[List[int]] = []  # neighbors[cell_idx] = [nbr_indices]
        self.goal_zones: Dict[str, Set[int]] = {}
        self.start_zones: Dict[str, Set[int]] = {}

        # Precomputed lookup tables (built once, used millions of times)
        self._dist_table: List[List[int]] = []  # _dist_table[i][j] = hex distance
        self._min_dist_to_goal: Dict[str, List[int]] = {}  # color -> [121 distances]
        self._goal_set: Dict[str, Set[int]] = {}  # fast membership test

        self._build_board()
        self._compute_neighbors()
        self._define_zones()
        self._precompute_distances()

    def _build_board(self):
        valid_coords = set()
        R = Config.BOARD_RADIUS  # 4

        # Central hexagon
        for q in range(-R, R + 1):
            for r in range(-R, R + 1):
                if max(abs(q), abs(r), abs(q + r)) <= R:
                    valid_coords.add((q, r))

        # 6 triangles
        for k in range(1, R + 1):
            # Top (red start): r < -R
            r = -(R + k)
            for q in range(k, R + 1):
                valid_coords.add((q, r))
            # Bottom (blue start): r > R
            r = R + k
            for q in range(-R, -k + 1):
                valid_coords.add((q, r))
            # Top-right (lawn green): q > R
            q = R + k
            for r in range(-R, -k + 1):
                valid_coords.add((q, r))
            # Bottom-left (gray0): q < -R
            q = -(R + k)
            for r in range(k, R + 1):
                valid_coords.add((q, r))
            # Bottom-right (yellow): s < -R
            s_val = -(R + k)
            for r in range(k, R + 1):
                valid_coords.add((-r - s_val, r))
            # Top-left (purple): s > R
            s_val = R + k
            for r in range(-R, -k + 1):
                valid_coords.add((-r - s_val, r))

        # Sort by (r, q) to match server ordering
        sorted_coords = sorted(valid_coords, key=lambda c: (c[1], c[0]))

        for idx, (q, r) in enumerate(sorted_coords):
            self.cell_q.append(q)
            self.cell_r.append(r)
            self.cell_gx.append(q + 8)
            self.cell_gy.append(r + 8)
            self.index_of[(q, r)] = idx

    def _compute_neighbors(self):
        n = len(self.cell_q)
        self.neighbors = [[] for _ in range(n)]
        for idx in range(n):
            q, r = self.cell_q[idx], self.cell_r[idx]
            for dq, dr in Config.DIRECTIONS:
                nq, nr = q + dq, r + dr
                nbr = self.index_of.get((nq, nr))
                if nbr is not None:
                    self.neighbors[idx].append(nbr)

    def _define_zones(self):
        R = Config.BOARD_RADIUS
        zones = {c: set() for c in Config.COLOR_OPPOSITES}

        for idx in range(len(self.cell_q)):
            q, r = self.cell_q[idx], self.cell_r[idx]
            s = -q - r
            if max(abs(q), abs(r), abs(s)) <= R:
                continue
            if r < -R:
                zones['blue'].add(idx)       # top triangle = blue (matches teacher)
            elif r > R:
                zones['red'].add(idx)        # bottom triangle = red (matches teacher)
            elif q > R:
                zones['lawn green'].add(idx)
            elif q < -R:
                zones['gray0'].add(idx)
            elif s < -R:
                zones['purple'].add(idx)     # bottom-right = purple (matches teacher)
            elif s > R:
                zones['yellow'].add(idx)     # top-left = yellow (matches teacher)

        self.start_zones = zones
        self.goal_zones = {
            color: self.start_zones[opp]
            for color, opp in Config.COLOR_OPPOSITES.items()
        }
        self._goal_set = {c: set(g) for c, g in self.goal_zones.items()}

    def _precompute_distances(self):
        """Precompute ALL pairwise distances and per-color min-to-goal distances.

        This turns every distance query from O(goals) to O(1).
        121 * 121 = 14,641 entries - trivial memory.
        """
        n = len(self.cell_q)

        # Pairwise distance table
        self._dist_table = [[0] * n for _ in range(n)]
        for i in range(n):
            qi, ri, si = self.cell_q[i], self.cell_r[i], -self.cell_q[i] - self.cell_r[i]
            for j in range(i + 1, n):
                qj, rj, sj = self.cell_q[j], self.cell_r[j], -self.cell_q[j] - self.cell_r[j]
                d = max(abs(qi - qj), abs(ri - rj), abs(si - sj))
                self._dist_table[i][j] = d
                self._dist_table[j][i] = d

        # Per-color min distance to goal - the hot path for reward computation
        for color in Config.COLOR_OPPOSITES:
            goal_list = list(self.goal_zones[color])
            dists = [0] * n
            for idx in range(n):
                dists[idx] = min(self._dist_table[idx][g] for g in goal_list)
            self._min_dist_to_goal[color] = dists

    # ── Fast lookups (O(1)) ──────────────────────────────────────────

    def axial_distance(self, idx1: int, idx2: int) -> int:
        return self._dist_table[idx1][idx2]

    def min_distance_to_goal(self, cell_idx: int, color: str) -> int:
        return self._min_dist_to_goal[color][cell_idx]

    def sum_distances_to_goal(self, piece_positions: List[int], color: str) -> int:
        table = self._min_dist_to_goal[color]
        return sum(table[p] for p in piece_positions)

    def is_in_goal(self, cell_idx: int, color: str) -> bool:
        return cell_idx in self._goal_set[color]

    def count_in_goal(self, piece_positions: List[int], color: str) -> int:
        gs = self._goal_set[color]
        return sum(1 for p in piece_positions if p in gs)

    def get_neighbors(self, cell_idx: int) -> List[int]:
        return self.neighbors[cell_idx]

    def grid_pos(self, cell_idx: int) -> Tuple[int, int]:
        return self.cell_gx[cell_idx], self.cell_gy[cell_idx]

    def axial_to_grid(self, cell_idx: int) -> Tuple[int, int]:
        return self.cell_gx[cell_idx], self.cell_gy[cell_idx]

    def grid_to_cell_idx(self, gx: int, gy: int) -> Optional[int]:
        return self.index_of.get((gx - 8, gy - 8))

    @property
    def num_cells(self) -> int:
        return len(self.cell_q)

    def get_valid_cell_indices(self) -> List[int]:
        return list(range(len(self.cell_q)))

    def get_valid_grid_positions(self) -> List[Tuple[int, int]]:
        return list(zip(self.cell_gx, self.cell_gy))
