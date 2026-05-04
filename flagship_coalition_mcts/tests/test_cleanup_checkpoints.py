"""Tests for the checkpoint cleanup utility."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import pytest


def _make_fake_checkpoints(d: str, iters: list, extras: list = None) -> None:
    """Create empty .pt files for testing."""
    for it in iters:
        path = os.path.join(d, f"iter_{it:04d}.pt")
        with open(path, "wb") as f:
            f.write(b"x" * 1024)  # 1 KB
    for name in extras or []:
        path = os.path.join(d, name)
        with open(path, "wb") as f:
            f.write(b"x" * 1024)


def test_dry_run_does_not_delete():
    nexus_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    venv = os.path.join(nexus_root, "venv", "bin", "python")
    if not os.path.exists(venv):
        venv = sys.executable
    with tempfile.TemporaryDirectory() as d:
        _make_fake_checkpoints(d, list(range(1, 21)), ["final.pt"])
        before = sorted(os.listdir(d))
        # Dry run
        subprocess.run(
            [venv, os.path.join(nexus_root, "scripts", "cleanup_checkpoints.py"),
             "--dir", d, "--keep-every", "10"],
            check=True, capture_output=True,
        )
        after = sorted(os.listdir(d))
        assert before == after, "dry run should not delete files"


def test_apply_deletes_correct_files():
    nexus_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    venv = os.path.join(nexus_root, "venv", "bin", "python")
    if not os.path.exists(venv):
        venv = sys.executable
    with tempfile.TemporaryDirectory() as d:
        # Iters 1..15, with --keep-every 5 should keep 5, 10, 15
        # Plus final.pt
        _make_fake_checkpoints(d, list(range(1, 16)), ["final.pt"])
        subprocess.run(
            [venv, os.path.join(nexus_root, "scripts", "cleanup_checkpoints.py"),
             "--dir", d, "--keep-every", "5", "--apply"],
            check=True, capture_output=True,
        )
        remaining = sorted(os.listdir(d))
        # Should keep iter_0005, iter_0010, iter_0015 (latest), final.pt
        assert "iter_0005.pt" in remaining
        assert "iter_0010.pt" in remaining
        assert "iter_0015.pt" in remaining
        assert "final.pt" in remaining
        # Should NOT keep iter_0001..iter_0009 except multiples of 5
        assert "iter_0001.pt" not in remaining
        assert "iter_0007.pt" not in remaining


def test_handles_empty_directory():
    nexus_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    venv = os.path.join(nexus_root, "venv", "bin", "python")
    if not os.path.exists(venv):
        venv = sys.executable
    with tempfile.TemporaryDirectory() as d:
        result = subprocess.run(
            [venv, os.path.join(nexus_root, "scripts", "cleanup_checkpoints.py"),
             "--dir", d],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "No .pt files" in result.stdout


def test_missing_directory_fails():
    nexus_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    venv = os.path.join(nexus_root, "venv", "bin", "python")
    if not os.path.exists(venv):
        venv = sys.executable
    result = subprocess.run(
        [venv, os.path.join(nexus_root, "scripts", "cleanup_checkpoints.py"),
         "--dir", "/nonexistent/path/that/does/not/exist"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "not a directory" in result.stdout
