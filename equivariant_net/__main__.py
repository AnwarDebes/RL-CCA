"""equivariant_net - wreath equivariant research subproject.

Run `python -m equivariant_net <subcommand>` for entry points.
"""

import sys


HELP = """
Usage: python -m equivariant_net <subcommand>

Subcommands:
  test              Run unit tests (verifies bit-identical equivariance).
  --help            Show this message.
"""


def main():
    args = sys.argv[1:]
    if not args or args[0] in ("--help", "-h"):
        print(HELP)
        return 0
    cmd = args[0]
    if cmd == "test":
        import subprocess
        return subprocess.call([
            sys.executable, "-m", "pytest",
            "equivariant_net/tests/test_seat_equivariant.py",
            "equivariant_net/tests/test_c6_spatial.py",
            "equivariant_net/tests/test_wreath_fuse.py",
            "equivariant_net/tests/test_cc_wreath_encoder.py",
            "equivariant_net/tests/test_wreath_network.py",
            "equivariant_net/tests/test_cc_runner.py",
            "equivariant_net/tests/test_public_api.py",
            "--tb=short",
        ])
    print(f"Unknown subcommand: {cmd}")
    print(HELP)
    return 1


if __name__ == "__main__":
    sys.exit(main())
