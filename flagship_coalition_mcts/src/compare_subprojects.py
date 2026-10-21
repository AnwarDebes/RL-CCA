"""Side-by-side comparison of the three research subprojects.

Generates a comparison table (Markdown or LaTeX) showing what each
subproject contributes, what is shared, and how they relate to prior
work.
"""

from __future__ import annotations


COMPARISON_TABLE = [
    # (feature, CD-MCTS, CMAZ, wreath equivariant, prior art)
    ("Main idea",
     "Plackett-Luce + coalition + EXP-IX",
     "QMIX-style mixer in MCTS",
     "Wreath C6 ≀ S_N for star-hex games",
     "-"),
    ("Value head",
     "vector PL theta + coalition + scalar",
     "K per-component values + mixer Q",
     "scalar (per current player)",
     "AZ: scalar; KataGo: 2-vec; Petosa: N-vec"),
    ("Selection rule",
     "EXP-IX no-regret",
     "PUCT (vanilla)",
     "PUCT (vanilla)",
     "AZ: PUCT; CFR: regret-matching"),
    ("MCTS backup",
     "vector placement marginal (N x N)",
     "vector per-component (K)",
     "scalar",
     "AZ: scalar; MO-MCTS: vector"),
    ("Convergence target",
     "CCE of N-player meta-game",
     "AZ-style (no formal claim)",
     "AZ-style (no formal claim)",
     "AZ: Nash (2p zero-sum only)"),
    ("Inference-time tuning",
     "coalition_weight tunable",
     "mixer override (utility re-weighting)",
     "-",
     "-"),
    ("Equivariance",
     "-",
     "-",
     "C6 spatial × S_N seat (wreath)",
     "HexaConv: p6m; FGNN: D2"),
    ("Key experiment",
     "kingmaker H2H + CCE-gap",
     "inference-override sweep",
     "bit-identical seat permutation",
     "-"),
    ("Game testbeds",
     "kingmaker, halma, real CC",
     "kingmaker, halma, real CC",
     "real CC (with hex symmetry)",
     "-"),
    ("Theory",
     "CCE convergence (conjecture)",
     "-",
     "-",
     "-"),
]


def render_markdown() -> str:
    headers = ["Feature", "CD-MCTS", "CMAZ", "Wreath equivariant", "Prior art"]
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in COMPARISON_TABLE:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def render_latex() -> str:
    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("\\caption{Side-by-side comparison of the three research subprojects.}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{lllll}")
    lines.append("\\toprule")
    lines.append("Feature & CD-MCTS & CMAZ & Wreath & Prior art \\\\")
    lines.append("\\midrule")
    for row in COMPARISON_TABLE:
        cells = [c.replace("&", "\\&").replace("_", "\\_") for c in row]
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", choices=["markdown", "latex"], default="markdown")
    args = ap.parse_args()
    if args.format == "markdown":
        print(render_markdown())
    else:
        print(render_latex())


if __name__ == "__main__":
    main()
