"""Device selection utilities.

Centralises CPU/GPU choice logic so training scripts and the tournament
player can use a consistent rule. Falls back gracefully when GPU isn't
available or memory is too tight (e.g., during v4 RL training).

Usage:
    from flagship_coalition_mcts.src.device import select_device
    dev = select_device()  # "cuda" if available + has memory; else "cpu"

Override:
    dev = select_device(prefer="cpu")
    dev = select_device(prefer="cuda", min_free_gb=4.0)
"""

from __future__ import annotations

from typing import Optional


def select_device(
    prefer: str = "auto",
    min_free_gb: float = 1.5,
) -> str:
    """Return the device string to use.

    Args:
        prefer: "auto" (default - pick the best available), "cuda"
            (require GPU, raise if not available), or "cpu" (force CPU).
        min_free_gb: minimum free GPU memory to consider CUDA usable.
            If less than this is free (e.g., v4 RL training is hot),
            fall back to CPU.

    Returns:
        "cuda" or "cpu".
    """
    import torch
    if prefer == "cpu":
        return "cpu"
    cuda_available = torch.cuda.is_available()
    if prefer == "cuda" and not cuda_available:
        raise RuntimeError("CUDA requested but not available")
    if prefer in ("auto", "cuda") and cuda_available:
        # Check free memory before committing to GPU.
        try:
            free_b, _total_b = torch.cuda.mem_get_info()
            free_gb = free_b / 1e9
            if free_gb >= min_free_gb:
                return "cuda"
            # Not enough free memory.
            if prefer == "cuda":
                # Explicit cuda request - surface the issue rather than fall back silently.
                raise RuntimeError(
                    f"CUDA requested but only {free_gb:.1f} GB free "
                    f"(need {min_free_gb:.1f} GB). v4 RL training may be active."
                )
            # auto: fall back to CPU.
        except Exception:
            # mem_get_info missing on older torch - assume usable
            return "cuda"
    return "cpu"


def gpu_status_string() -> str:
    """One-line summary of GPU availability + memory."""
    import torch
    if not torch.cuda.is_available():
        return "CPU only (no CUDA)"
    n = torch.cuda.device_count()
    name = torch.cuda.get_device_name(0)
    try:
        free, total = torch.cuda.mem_get_info()
        return f"{n} GPU(s); {name}; {free/1e9:.1f}/{total/1e9:.1f} GB free"
    except Exception:
        total = torch.cuda.get_device_properties(0).total_memory
        return f"{n} GPU(s); {name}; {total/1e9:.1f} GB total"
