"""flagship_coalition_mcts - CD-MCTS research subproject.

Run `python -m flagship_coalition_mcts <subcommand>` for entry points.
"""

import sys


HELP = """
Usage: python -m flagship_coalition_mcts <subcommand>

Subcommands:
  summary           Print architecture summary (params per submodule).
  compare           Print subproject-comparison table.
  smoke             Run the full integration smoke test on real CC.
  test              Run the unit tests.
  --help            Show this message.

Examples:
  python -m flagship_coalition_mcts summary --format latex
  python -m flagship_coalition_mcts smoke --num-players 2

For detailed deployment instructions see flagship_coalition_mcts/DEPLOY.md.
For experiment recipes see flagship_coalition_mcts/docs/PHASE2_EXPERIMENT_BLUEPRINT.md.
"""


def main():
    args = sys.argv[1:]
    if not args or args[0] in ("--help", "-h"):
        print(HELP)
        return 0
    cmd = args[0]
    rest = args[1:]
    if cmd == "summary":
        from flagship_coalition_mcts.src.model_summary import main as m
        sys.argv = ["model_summary"] + rest
        return m()
    if cmd == "compare":
        from flagship_coalition_mcts.src.compare_subprojects import main as m
        sys.argv = ["compare"] + rest
        return m()
    if cmd == "smoke":
        from flagship_coalition_mcts.experiments.full_integration_demo import main as m
        sys.argv = ["smoke"] + rest
        return m()
    if cmd == "test":
        import subprocess
        return subprocess.call([
            sys.executable, "-m", "pytest",
            "flagship_coalition_mcts/tests/", "--tb=short",
        ])
    print(f"Unknown subcommand: {cmd}")
    print(HELP)
    return 1


if __name__ == "__main__":
    sys.exit(main())
