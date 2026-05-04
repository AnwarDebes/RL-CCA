"""Model architecture summary utility.

Prints a paper-appendix-quality summary of the network's structure:
total parameters, per-layer parameter counts, output dimension at each
stage. Used to populate the paper's "Architecture details" appendix.

Works on any nn.Module with named_parameters; specialised friendliness
for CDMCTSNetwork, CMAZNetwork, and WreathCCNetwork.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict

import torch
import torch.nn as nn


def count_parameters(module: nn.Module) -> Dict[str, int]:
    """Returns {name: param_count} aggregated by top-level module name."""
    out = defaultdict(int)
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        top = name.split(".")[0]
        out[top] += param.numel()
    return dict(out)


def total_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def summary(module: nn.Module, name: str = "Network") -> str:
    """Multi-line summary string."""
    lines = []
    lines.append(f"=== {name} ===")
    lines.append(f"Total trainable parameters: {total_params(module):,}")
    counts = count_parameters(module)
    if counts:
        lines.append("Per top-level submodule:")
        for n, c in sorted(counts.items(), key=lambda x: -x[1]):
            pct = 100.0 * c / max(1, total_params(module))
            lines.append(f"  {n:<24} {c:>10,}   ({pct:.1f}%)")
    return "\n".join(lines)


def latex_table(module: nn.Module, name: str = "Network") -> str:
    """LaTeX-ready table of parameter counts per submodule."""
    counts = count_parameters(module)
    lines = []
    lines.append("\\begin{table}[h]")
    lines.append(f"\\caption{{Parameter counts for {name}.}}")
    lines.append("\\begin{tabular}{lr}")
    lines.append("\\toprule")
    lines.append("Component & Parameters \\\\")
    lines.append("\\midrule")
    for n, c in sorted(counts.items(), key=lambda x: -x[1]):
        n_clean = n.replace("_", "\\_")
        lines.append(f"{n_clean} & {c:,} \\\\")
    lines.append("\\midrule")
    lines.append(f"Total & {total_params(module):,} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def main():
    """CLI: print summaries for all three subprojects' default networks."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", choices=["text", "latex"], default="text")
    args = ap.parse_args()

    # Lazy imports - keep this module light if only count_parameters is needed.
    from .cnn_encoder import CCCNNEncoder
    from .network import CDMCTSNetwork, MLPEncoder

    print("Building default-config networks for summary...\n")
    formatter = latex_table if args.format == "latex" else summary

    # Flagship CD-MCTS with CC CNN encoder
    enc1 = CCCNNEncoder(in_channels=32, channels=64, num_blocks=4, out_dim=128)
    cd_mcts = CDMCTSNetwork(encoder=enc1, action_space_size=1210, max_players=6)
    print(formatter(cd_mcts, "CD-MCTS (flagship, CC config)"))
    print()

    # Flagship CD-MCTS with kingmaker MLP encoder
    enc2 = MLPEncoder(input_dim=12, hidden_dim=24, num_layers=2)
    cd_mcts_km = CDMCTSNetwork(encoder=enc2, action_space_size=4, max_players=3)
    print(formatter(cd_mcts_km, "CD-MCTS (flagship, kingmaker config)"))
    print()

    try:
        from decomposed_mcts.src.cc_adapter import build_cmaz_cc_network
        cmaz = build_cmaz_cc_network(channels=64, num_blocks=4, hidden_dim=128)
        print(formatter(cmaz, "CMAZ (workshop, CC config)"))
        print()
    except ImportError:
        pass

    try:
        from equivariant_net.src.wreath_network import WreathCCNetwork
        wreath = WreathCCNetwork(
            spatial_channels=16, spatial_blocks=3, spatial_out=128,
            seat_hidden=32, seat_blocks=2,
        )
        print(formatter(wreath, "Wreath equivariant (workshop, CC config)"))
    except ImportError:
        pass


if __name__ == "__main__":
    main()
