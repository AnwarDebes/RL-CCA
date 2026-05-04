"""CMAZ - Component-Mixed AlphaZero - public API."""

__version__ = "0.1.0"
__title__ = "CMAZ - Component-Mixed AlphaZero"
__author__ = "(see paper)"

from .monotonic_mixer import ComponentValueHead, MonotonicMixer
from .network import (
    CMAZEncoder, CMAZEvaluator, CMAZNetwork, cmaz_loss,
)
from .cmaz_mcts import (
    CMAZNetworkOutput, CMAZNode, run_mcts_cmaz, run_simulation_cmaz,
)

# Game adapters
try:
    from .cc_adapter import (
        CMAZCCEvaluator, build_cmaz_cc_network, play_one_cmaz_cc_game,
    )
    from .kingmaker_adapter import (
        KingmakerCMAZEvaluator, build_cmaz_kingmaker_network,
        kingmaker_features_for_cmaz, kingmaker_score_components,
    )
    from .halma_adapter import (
        HalmaCMAZEvaluator, build_cmaz_halma_network, halma_score_components,
    )
except ImportError:
    # Adapters need nexus core/* - defer if unavailable.
    CMAZCCEvaluator = None
    build_cmaz_cc_network = None

__all__ = [
    "ComponentValueHead", "MonotonicMixer",
    "CMAZEncoder", "CMAZEvaluator", "CMAZNetwork", "cmaz_loss",
    "CMAZNetworkOutput", "CMAZNode", "run_mcts_cmaz", "run_simulation_cmaz",
    "CMAZCCEvaluator", "build_cmaz_cc_network", "play_one_cmaz_cc_game",
    "KingmakerCMAZEvaluator", "build_cmaz_kingmaker_network",
    "kingmaker_features_for_cmaz", "kingmaker_score_components",
    "HalmaCMAZEvaluator", "build_cmaz_halma_network", "halma_score_components",
]
