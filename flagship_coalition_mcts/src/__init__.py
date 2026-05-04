"""Coalition-Distributional MCTS - public API.

Top-level imports for the flagship subproject. Use these instead of
deep-importing from submodules.
"""

__version__ = "0.1.0"
__title__ = "Coalition-Distributional MCTS"
__author__ = "(see paper)"

# Network components
from .network import CDMCTSEvaluator, CDMCTSNetwork, MLPEncoder, cdmcts_loss
from .cnn_encoder import CCCNNEncoder

# Device selection
from .device import gpu_status_string, select_device

# Tree visualization / debugging
from .tree_debug import format_stats, print_tree, tree_stats

# Heads
from .plackett_luce import (
    PlackettLuceHead,
    log_likelihood as pl_log_likelihood,
    placement_marginals_exact,
    sample_ranking,
    winner_marginal,
)
from .coalition_head import (
    CoalitionHead,
    coalition_log_probs,
    coalition_marginal_alignment,
    coalition_entropy,
)

# MCTS
from .cce_selector import SelectorState, hedge_distribution, select_action, update_regrets
from .mcts import NetworkOutput, Node, run_mcts
from .baseline_mcts import (
    ScalarEvaluator, ScalarNetworkOutput, run_mcts_scalar,
)
from .nn_cce_baseline import (
    NNCCEEvaluator, NNCCENetworkOutput, run_mcts_nncce,
)
from .subtree_reuse import advance_root, advance_root_by_raw_action, reuse_or_rebuild

# Utilities
from .replay_buffer import ReplayBuffer
from .checkpoint import (
    CHECKPOINT_VERSION, CheckpointBundle,
    save_checkpoint, load_checkpoint, list_checkpoints, latest_checkpoint,
)
from .head_to_head import HeadToHeadResult, head_to_head, permutation_test
from .results_table import RunRecord, aggregate, format_latex, format_markdown, load_run
from .summarize_results import _short_summary as summarize_one_result
from .find_best_seed import _extract_metric as extract_metric
from .exploitability import cce_gap, exploitability, expected_utility_under_profile
from .model_summary import count_parameters, summary, total_params

# CC integration
from .cc_runner import CCMCTSEvaluator, build_cc_evaluator, build_cc_network, play_one_cc_game

# Tournament deployment
try:
    from .tournament_player_cdmcts import NexusTournamentPlayerCDMCTS
except ImportError:
    # Tournament player needs nexus core/* - defer if unavailable
    NexusTournamentPlayerCDMCTS = None

__all__ = [
    "CDMCTSEvaluator", "CDMCTSNetwork", "MLPEncoder", "cdmcts_loss",
    "CCCNNEncoder",
    "gpu_status_string", "select_device",
    "format_stats", "print_tree", "tree_stats",
    "PlackettLuceHead", "pl_log_likelihood",
    "placement_marginals_exact", "sample_ranking", "winner_marginal",
    "CoalitionHead", "coalition_log_probs", "coalition_marginal_alignment",
    "coalition_entropy",
    "SelectorState", "hedge_distribution", "select_action", "update_regrets",
    "NetworkOutput", "Node", "run_mcts",
    "ScalarEvaluator", "ScalarNetworkOutput", "run_mcts_scalar",
    "NNCCEEvaluator", "NNCCENetworkOutput", "run_mcts_nncce",
    "advance_root", "advance_root_by_raw_action", "reuse_or_rebuild",
    "ReplayBuffer",
    "CHECKPOINT_VERSION", "CheckpointBundle",
    "save_checkpoint", "load_checkpoint", "list_checkpoints", "latest_checkpoint",
    "HeadToHeadResult", "head_to_head", "permutation_test",
    "RunRecord", "aggregate", "format_latex", "format_markdown", "load_run",
    "cce_gap", "exploitability", "expected_utility_under_profile",
    "count_parameters", "summary", "total_params",
    "CCMCTSEvaluator", "build_cc_evaluator", "build_cc_network", "play_one_cc_game",
    "NexusTournamentPlayerCDMCTS",
]
