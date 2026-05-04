"""decomposed_mcts - CMAZ research subproject.

Run `python -m decomposed_mcts <subcommand>` for entry points.
"""

import sys


HELP = """
Usage: python -m decomposed_mcts <subcommand>

Subcommands:
  override-demo     Inference-time override demo (untrained network).
  train-kingmaker   Train CMAZ on kingmaker, then demo override sweep.
  test              Run unit tests.
  --help            Show this message.
"""


def main():
    args = sys.argv[1:]
    if not args or args[0] in ("--help", "-h"):
        print(HELP)
        return 0
    cmd = args[0]
    rest = args[1:]
    if cmd == "override-demo":
        from decomposed_mcts.experiments.inference_override_demo import main as m
        sys.argv = ["override-demo"] + rest
        return m()
    if cmd == "train-kingmaker":
        from decomposed_mcts.experiments.train_and_demo_kingmaker import main as m
        sys.argv = ["train-kingmaker"] + rest
        return m()
    if cmd == "test":
        import subprocess
        return subprocess.call([
            sys.executable, "-m", "pytest", "decomposed_mcts/tests/", "--tb=short",
        ])
    print(f"Unknown subcommand: {cmd}")
    print(HELP)
    return 1


if __name__ == "__main__":
    sys.exit(main())
