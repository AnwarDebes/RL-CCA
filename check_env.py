#!/usr/bin/env python3
"""Environment sanity check.

Verifies the Python environment has everything needed to run the
research subprojects. Useful as a first command after checking out
the codebase or onboarding a new contributor.

Usage: ./venv/bin/python check_env.py
"""

import importlib
import os
import sys


REQUIRED = [
    ("torch", "PyTorch - networks, autograd"),
    ("numpy", "Numerical arrays"),
    ("pytest", "Test runner"),
]
OPTIONAL = [
    ("matplotlib", "Plotting (used by training_visualize.py)"),
    ("escnn", "Production-grade equivariant CNNs (not currently used)"),
]


def check(modname: str, description: str, optional: bool = False) -> bool:
    try:
        m = importlib.import_module(modname)
        version = getattr(m, "__version__", "?")
        marker = "[ok]" if not optional else "[opt-ok]"
        print(f"  {marker} {modname:<14} v{version:<10} {description}")
        return True
    except ImportError:
        marker = "[FAIL]" if not optional else "[opt-missing]"
        print(f"  {marker} {modname:<14} {'':<11} {description}")
        return optional  # Optional missing is not a fatal failure


def main():
    print("=" * 60)
    print("Environment check for nexus research subprojects")
    print("=" * 60)
    print(f"\nPython: {sys.version.split()[0]}")
    print(f"Executable: {sys.executable}")

    print("\nRequired packages:")
    all_ok = all(check(m, d) for m, d in REQUIRED)

    print("\nOptional packages:")
    for m, d in OPTIONAL:
        check(m, d, optional=True)

    # Subproject importability
    print("\nSubproject importability:")
    nexus_root = os.path.dirname(os.path.abspath(__file__))
    if nexus_root not in sys.path:
        sys.path.insert(0, nexus_root)

    for pkg in ["flagship_coalition_mcts.src",
                "decomposed_mcts.src",
                "equivariant_net.src"]:
        try:
            m = importlib.import_module(pkg)
            v = getattr(m, "__version__", "?")
            print(f"  [ok] {pkg} (v{v})")
        except Exception as e:
            print(f"  [FAIL] {pkg}: {e}")
            all_ok = False

    # Existing nexus core modules (needed for CC integration)
    print("\nExisting nexus core modules:")
    for pkg in ["core.board", "core.game_env", "config"]:
        try:
            importlib.import_module(pkg)
            print(f"  [ok] {pkg}")
        except Exception as e:
            print(f"  [FAIL] {pkg}: {e}")
            all_ok = False

    # GPU availability
    print("\nGPU support:")
    try:
        import torch
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            print(f"  [ok] CUDA available - {n} device(s)")
            for i in range(n):
                name = torch.cuda.get_device_name(i)
                mem = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"       device {i}: {name} ({mem:.1f} GB)")
            free, total = torch.cuda.mem_get_info()
            free_gb = free / 1e9
            total_gb = total / 1e9
            print(f"       device 0 free memory: {free_gb:.1f} / {total_gb:.1f} GB")
            if free_gb < 1.0:
                print(f"       WARN: GPU memory tight - v4 RL training may be active")
        else:
            print("  [opt-missing] CUDA not available (CPU-only mode)")
            print("       Most experiments still work but slower")
    except Exception as e:
        print(f"  GPU check failed: {e}")

    print("\n" + "=" * 60)
    if all_ok:
        print("ALL CHECKS PASSED - environment is ready.")
        print("Try: make test-fast  (or)  make smoke")
    else:
        print("SOME CHECKS FAILED - see above.")
    print("=" * 60)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
