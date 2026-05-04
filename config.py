"""NEXUS v2 - All hyperparameters centralized."""

import math


class Config:
    # === Board ===
    BOARD_RADIUS = 4
    NUM_CELLS = 121
    NUM_PIECES = 10
    GRID_SIZE = 17
    # v3-rebuild encoder: 22 base channels + 10 per-opponent (5 opps × {pieces, goal})
    # Fixes the "all opponents unioned" information loss that capped v2 N=6 at ~840.
    NUM_CHANNELS = 32
    NUM_OPP_SLOTS = 5            # max opponents (since N=6 → 5 opponents)

    # === Axial directions (q, r) for hex grid ===
    DIRECTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]

    # === Color mappings ===
    COLOR_OPPOSITES = {
        'red': 'blue',
        'blue': 'red',
        'lawn green': 'gray0',
        'gray0': 'lawn green',
        'yellow': 'purple',
        'purple': 'yellow',
    }

    # === Network (v2 - kept for backward compat) ===
    HIDDEN_DIM = 128
    NUM_RES_BLOCKS = 4
    NUM_TRANSFORMER_BLOCKS = 2
    TRANSFORMER_HEADS = 4
    TRANSFORMER_FFN_DIM = 512
    ACTION_SPACE = 1210  # 10 pieces * 121 cells
    OPP_GRU_HIDDEN = 64
    OPP_GRU_EMBED = 32
    AUX_DIM = 64

    # === Network v3 ===
    HIDDEN_DIM_V3 = 160
    NUM_RES_BLOCKS_V3 = 8        # 4 before transformer + 4 after
    TRANSFORMER_HEADS_V3 = 4
    TRANSFORMER_FFN_DIM_V3 = 640
    MAX_PLAYERS = 6              # value-vector head dim

    # === Network v4 - KataGo NBT-style + bigger ===
    HIDDEN_DIM_V4 = 256
    NUM_RES_BLOCKS_V4 = 16       # 8 before transformer + 8 after, NBT bottleneck
    NBT_BOTTLENECK_V4 = 128      # bottleneck channels in nested-bottleneck-tower
    TRANSFORMER_HEADS_V4 = 8
    TRANSFORMER_FFN_DIM_V4 = 1024
    PIN_FINAL_BUCKETS_V4 = 5     # distance-to-goal buckets for per-pin aux head
    # MCTS at training & inference.
    # Training: 8 sims (Gumbel-AlphaZero paper: works at 2-16 sims due to
    #           sequential halving + completed-Q). Keeps iters tractable.
    # Inference: tournament-time has ~0.45s budget; 32 sims ≈ 280ms.
    # Training sims bumped from 8 to 16 with 14-day budget - doubles policy
    # improvement quality per move (Gumbel-AZ paper: clear gains in 8→16 range).
    MCTS_TRAIN_SIMS_V4 = 8
    MCTS_TRAIN_M_V4 = 4              # Gumbel candidates at training
    MCTS_INF_SIMS_OPENING_V4 = 48
    MCTS_INF_SIMS_MIDGAME_V4 = 32
    MCTS_INF_SIMS_ENDGAME_V4 = 16
    MCTS_INF_M_V4 = 8                # Gumbel candidates at inference
    # Phase 2 v4 self-play: 32 games per iter (vs 64 in v3) since each game
    # is much slower with MCTS. Half the games × 2 iters = same total budget.
    GAMES_PER_ITERATION_V4 = 32
    MCTS_INF_HARD_BUDGET_SEC = 0.45      # per-move wall-clock cap
    MCTS_VIRTUAL_LOSS = 1.0
    MCTS_BATCH_LEAVES = 8                # batched-leaf collection
    # Loss weights (v4)
    SCORE_MARGIN_LOSS_WEIGHT = 0.5
    PIN_FINAL_LOSS_WEIGHT = 0.3
    # EMA / mixed precision / weight decay split
    EMA_DECAY_V4 = 0.999
    # Empirically, bf16 with this v4 architecture (16 NBT + transformer + 6 heads)
    # was 8× SLOWER than fp32 (12s/step vs 1.5s/step). Probably because each
    # small layer triggers dtype-conversion overhead that dwarfs the matmul.
    # Disabled until we figure out a clean fix.
    USE_BF16_AMP = False

    # === MCTS ===
    GUMBEL_M = 8  # keeps actual sim count close to budget
    C_PUCT = 1.5
    DIRICHLET_ALPHA = 0.3
    DIRICHLET_FRAC = 0.25

    # === Training ===
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    BATCH_SIZE = 512
    REPLAY_BUFFER_SIZE = 300_000  # ~5 GB at float16 storage
    GRAD_CLIP = 1.0

    # Loss weights
    POLICY_LOSS_WEIGHT = 1.0
    VALUE_LOSS_WEIGHT = 1.0
    CONSISTENCY_LOSS_WEIGHT = 0.5
    KL_LOSS_WEIGHT = 0.3

    # === v3 loss + training extras ===
    # Auxiliary heads (KataGo-style): drop-in additive losses
    OPP_POLICY_LOSS_WEIGHT = 0.15
    PLIES_LOSS_WEIGHT = 0.10
    VALUE_VEC_LOSS_WEIGHT = 0.5    # per-player value vector
    ENTROPY_BONUS_WEIGHT = 0.005   # -beta * H(pi); prevents entropy collapse
    # LR schedule: cosine with warm restarts (T_0 in iters, eta_min)
    LR_T_0 = 100
    LR_ETA_MIN = 5e-5
    LR_RESTART_MULT = 1.0
    # Frozen opponent pool (anti-plateau): keep the last K snapshots and
    # sample one of them for FREEZE_OPP_FRACTION of seats per game.
    FREEZE_POOL_SIZE = 5
    FREEZE_POOL_EVERY = 25         # snapshot cadence (iters)
    FREEZE_OPP_FRACTION = 0.25     # of self-play seats use a frozen opp
    # FPU reduction (Leela-style): in MCTS, unvisited children get parent_value - FPU
    MCTS_FPU = 0.25
    MCTS_USE_DIRICHLET = False     # Gumbel root selection makes Dirichlet redundant

    # === DiscoRL ===
    DISCO_START_ITER = 200
    DISCO_ALPHA_INITIAL = 0.1
    DISCO_ALPHA_MAX = 0.3
    DISCO_EVAL_ITER = 300
    DISCO_ELO_THRESHOLD = 20

    # === Self-play ===
    GAMES_PER_ITERATION = 64
    NUM_WORKERS = 24  # 25% of 96 cores on shared server
    TEMPERATURE_INITIAL = 1.0   # high exploration early - needed for self-imitation
    TEMPERATURE_FINAL = 0.3     # still some stochasticity at end (not greedy)
    TRAINING_STEPS_PER_ITER = 50

    # === N-player support ===
    # Curriculum: start narrow, broaden over time (keys = iter threshold)
    # value: dict {N: weight} sampled per game.
    NUM_PLAYERS_CURRICULUM = {
        0:   {2: 1.00},                                      # warmup, 2-player only
        50:  {2: 0.50, 3: 0.50},                              # introduce 3-player
        150: {2: 0.30, 3: 0.30, 4: 0.40},                     # introduce 4-player
        300: {2: 0.10, 3: 0.15, 4: 0.20, 5: 0.25, 6: 0.30},   # full N=2..6
    }
    # v3 curriculum: train ALL N from iter 0 (no warm-up bias).
    # v3-attempt-1 diagnostic: N=6 had 0/1344 games at iter 20 because the old
    # curriculum started at 100% N=2 and only added N=6 at iter 150 - so the
    # eval at iter 10 measured a model that had never played 6-player. Fixed
    # by training all N from the start. Slight skew toward N=2 in the first
    # ~30 iters because N=2 games finish faster (more training signal per sec).
    NUM_PLAYERS_CURRICULUM_V3 = {
        0:   {2: 0.30, 3: 0.175, 4: 0.175, 5: 0.175, 6: 0.175},  # all N from start
        30:  {2: 0.20, 3: 0.20,  4: 0.20,  5: 0.20,  6: 0.20},   # uniform after warmup
    }
    # Fraction of self-play games where ONE seat is replaced by HeuristicAgent
    # to anchor the agent against the strongest non-RL baseline.
    VS_HEURISTIC_FRACTION = 0.30

    # === Inter-iteration testing ===
    # Run inexpensive in-process eval every K iters (2-player vs heuristic)
    EVAL_INPROC_EVERY = 5
    EVAL_INPROC_GAMES = 10
    # Run server eval (against teacher's game.py) every K iters, for each N
    EVAL_SERVER_EVERY = 10
    EVAL_SERVER_GAMES_PER_N = 10
    # Run rule-alignment check every K iters
    RULE_ALIGNMENT_EVERY = 10

    # === Progressive simulation schedule (realistic for sequential GPU) ===
    SIM_SCHEDULE = {0: 32, 300: 64, 700: 200}

    # === Phase 1 ===
    PHASE1_NUM_GAMES = 50000
    PHASE1_EPOCHS = 50  # more epochs for stronger Phase 1 convergence
    PHASE1_LR = 0.002
    PHASE1_LABEL_SMOOTHING = 0.03
    PHASE1_MIN_WIN_RATE = 0.60  # minimum vs heuristic before proceeding to Phase 2

    # === Phase 3 ===
    PHASE3_POPULATION_SIZE = 20
    PHASE3_ELO_K = 32
    PHASE3_MATCH_GAMES = 50

    # === Tournament ===
    TOTAL_TIME_BUDGET = 55.0
    PER_MOVE_BUDGET = 8.0

    # === Value aggregation for MCTS ===
    # Components: [v_win, v_pins, v_moves, v_dist]
    # v_pins (pins in goal) maps to tournament's dominant pin_goal_score (1000 pts max)
    # v_dist maps to tournament's distance_score (200 pts max)
    # v_moves maps to tournament's move_score (~1 pt max, negligible)
    VALUE_WIN_WEIGHT = 0.45
    VALUE_PINS_WEIGHT = 0.25
    VALUE_MOVES_WEIGHT = 0.0   # negligible in tournament scoring (~1 pt)
    VALUE_DIST_WEIGHT = 0.30

    @staticmethod
    def get_progressive_sims(iteration):
        """Return simulation count based on training iteration."""
        sims = 32
        for thresh, s in sorted(Config.SIM_SCHEDULE.items()):
            if iteration >= thresh:
                sims = s
        return sims

    @staticmethod
    def get_num_players_distribution(iteration, v3: bool = False):
        """Return {N: weight} distribution over player counts at this iteration."""
        curriculum = (Config.NUM_PLAYERS_CURRICULUM_V3 if v3
                      else Config.NUM_PLAYERS_CURRICULUM)
        dist = curriculum[0]
        for thresh, d in sorted(curriculum.items()):
            if iteration >= thresh:
                dist = d
        return dist

    @staticmethod
    def sample_num_players(iteration, rng=None, v3: bool = False):
        """Sample a player count for this iteration's curriculum."""
        import random as _r
        dist = Config.get_num_players_distribution(iteration, v3=v3)
        ns = list(dist.keys())
        weights = [dist[n] for n in ns]
        if rng is None:
            rng = _r
        return rng.choices(ns, weights=weights, k=1)[0]

    @staticmethod
    def get_temperature(iteration, total_iterations=1000):
        """Anneal temperature from initial to final."""
        frac = min(1.0, iteration / max(1, total_iterations))
        return Config.TEMPERATURE_INITIAL + frac * (
            Config.TEMPERATURE_FINAL - Config.TEMPERATURE_INITIAL
        )
