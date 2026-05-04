"""Training-history visualisation: produces paper-quality loss + metric plots.

Reads `history.json` files produced by the training scripts and emits
matplotlib plots showing:
  * Total loss over iterations
  * Per-component loss breakdown (policy / pl / coalition / value)
  * Generation time per batch (for monitoring training-pipeline health)

Skips gracefully if matplotlib is unavailable.

Usage:
    python -m flagship_coalition_mcts.src.training_visualize \
        --history checkpoints/cdmcts_cc_seed0/history.json \
        --output plots/training_seed0.png
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List


def _load_history(path: str) -> List[dict]:
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "history" in data:
        return data["history"]
    raise ValueError(f"Unrecognised history format in {path}")


def plot_history(
    history: List[dict],
    output: str = "training_history.png",
    title: str = "Training history",
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")  # no GUI backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; falling back to text dump")
        for h in history:
            print(h)
        return

    iters = [h.get("iter", i) for i, h in enumerate(history)]
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Top: total loss + per-component
    ax = axes[0]
    if any("avg_total" in h for h in history):
        ax.plot(iters, [h.get("avg_total") for h in history], label="total", lw=2)
    for key in ("avg_policy", "avg_pl", "avg_coalition", "avg_value", "avg_components"):
        vals = [h.get(key) for h in history if key in h]
        if vals and any(v is not None for v in vals):
            xs = [iters[i] for i, h in enumerate(history) if key in h]
            ax.plot(xs, vals, label=key.replace("avg_", ""), alpha=0.7)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title(f"{title} - losses")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom: gen vs train time
    ax = axes[1]
    gen_secs = [h.get("gen_sec", 0) for h in history]
    train_secs = [h.get("train_sec", 0) for h in history]
    ax.bar(iters, gen_secs, label="gen", alpha=0.6)
    ax.bar(iters, train_secs, bottom=gen_secs, label="train", alpha=0.6)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Seconds")
    ax.set_title(f"{title} - pipeline timing")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output, dpi=120)
    print(f"Wrote plot to {output}")


def text_summary(history: List[dict]) -> str:
    """Compact text-only summary if no plotting backend."""
    lines = ["Iteration  Loss   Policy   Other components   Gen(s)  Train(s)"]
    for h in history:
        i = h.get("iter", 0)
        loss = h.get("avg_total", float("nan"))
        pol = h.get("avg_policy", float("nan"))
        gen = h.get("gen_sec", 0)
        train = h.get("train_sec", 0)
        other = ", ".join(
            f"{k.replace('avg_', '')}={v:.3f}"
            for k, v in h.items() if k.startswith("avg_") and k not in ("avg_total", "avg_policy")
        )
        lines.append(f"  {i:>4}    {loss:>5.3f}  {pol:>5.3f}   {other}   {gen:>5.0f}   {train:>5.0f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", required=True)
    ap.add_argument("--output", default="training_history.png")
    ap.add_argument("--title", default="Training history")
    ap.add_argument("--text-only", action="store_true",
                    help="Print text summary instead of plotting")
    args = ap.parse_args()

    if not os.path.exists(args.history):
        print(f"ERROR: {args.history} not found")
        return 1

    history = _load_history(args.history)
    if args.text_only:
        print(text_summary(history))
    else:
        plot_history(history, output=args.output, title=args.title)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
