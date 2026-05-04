"""Tests for the device-selection utility."""

from __future__ import annotations

import pytest

from flagship_coalition_mcts.src.device import gpu_status_string, select_device


def test_select_cpu_explicit():
    assert select_device(prefer="cpu") == "cpu"


def test_select_auto_returns_cpu_or_cuda():
    """Auto mode returns one of the two strings."""
    dev = select_device(prefer="auto")
    assert dev in ("cpu", "cuda")


def test_cuda_with_huge_min_free_falls_back_to_cpu():
    """If we demand 1 TB free, even with a GPU, we fall back to CPU."""
    dev = select_device(prefer="auto", min_free_gb=1024.0)
    assert dev == "cpu"


def test_cuda_explicit_raises_if_no_cuda():
    """Asking for cuda when none is available raises (or returns cuda
    if it IS available)."""
    import torch
    if torch.cuda.is_available():
        # Should succeed
        dev = select_device(prefer="cuda")
        assert dev == "cuda"
    else:
        with pytest.raises(RuntimeError):
            select_device(prefer="cuda")


def test_status_string_returns_non_empty():
    s = gpu_status_string()
    assert isinstance(s, str)
    assert len(s) > 0


def test_select_unknown_prefer_treated_as_auto():
    """Unknown prefer values shouldn't crash - they fall through to default behavior.
    (The function's contract is that prefer ∈ {auto, cpu, cuda}; this
    test documents the actual behavior on out-of-spec input.)"""
    # Currently any non-cpu-non-cuda prefer behaves as auto (returns
    # cuda if available). We just verify it returns something valid.
    dev = select_device(prefer="random_garbage")
    assert dev in ("cpu", "cuda")
