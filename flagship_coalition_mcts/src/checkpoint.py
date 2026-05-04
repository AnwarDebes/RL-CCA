"""Checkpoint save/load utilities for CD-MCTS training.

Bundles network state, optimizer state, replay buffer state, training
iteration count, and free-form metadata into a single torch-pickle file.
Supports versioning via a `version` field - future incompatible changes
bump the version and provide migration hooks.

Why this matters
----------------
Production training runs must be **resumable** - a 24h training run that
crashes at hour 23 must restart from hour 23, not hour 0. Without a
proper checkpoint utility, every long run is at risk.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from .replay_buffer import ReplayBuffer


CHECKPOINT_VERSION = 1


@dataclass
class CheckpointBundle:
    """In-memory representation of a checkpoint."""

    version: int
    iter_idx: int
    network_state: Dict[str, Any]
    optimizer_state: Optional[Dict[str, Any]] = None
    replay_buffer_state: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


def save_checkpoint(
    path: str,
    network: torch.nn.Module,
    iter_idx: int,
    optimizer: Optional[torch.optim.Optimizer] = None,
    replay_buffer: Optional[ReplayBuffer] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a CheckpointBundle to disk.

    Atomic save: writes to .tmp first, then renames. So a crash mid-save
    cannot corrupt the previous checkpoint.
    """
    bundle = dict(
        version=CHECKPOINT_VERSION,
        iter_idx=iter_idx,
        network_state=network.state_dict(),
        optimizer_state=optimizer.state_dict() if optimizer is not None else None,
        replay_buffer_state=replay_buffer.state_dict() if replay_buffer is not None else None,
        metadata=metadata or {},
    )
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(bundle, tmp)
    os.replace(tmp, path)


def load_checkpoint(
    path: str,
    network: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    replay_buffer: Optional[ReplayBuffer] = None,
    strict: bool = True,
) -> CheckpointBundle:
    """Load a checkpoint and apply it to the given objects.

    Returns the parsed bundle (without state-dicts mutated by the load
    operations - the originals are kept for inspection).

    If strict=False, accepts older versions and tries to migrate.
    """
    bundle_dict = torch.load(path, weights_only=False, map_location="cpu")
    version = bundle_dict.get("version", 0)
    if version != CHECKPOINT_VERSION:
        if strict:
            raise ValueError(
                f"checkpoint version {version} != current {CHECKPOINT_VERSION}; "
                f"pass strict=False to attempt loading"
            )
        # No migrations defined yet - just warn and continue.

    network.load_state_dict(bundle_dict["network_state"])
    if optimizer is not None and bundle_dict.get("optimizer_state") is not None:
        optimizer.load_state_dict(bundle_dict["optimizer_state"])
    if replay_buffer is not None and bundle_dict.get("replay_buffer_state") is not None:
        replay_buffer.load_state_dict(bundle_dict["replay_buffer_state"])

    return CheckpointBundle(
        version=version,
        iter_idx=bundle_dict.get("iter_idx", 0),
        network_state=bundle_dict["network_state"],
        optimizer_state=bundle_dict.get("optimizer_state"),
        replay_buffer_state=bundle_dict.get("replay_buffer_state"),
        metadata=bundle_dict.get("metadata"),
    )


def list_checkpoints(directory: str) -> list:
    """Return sorted list of (iter_idx, path) for all checkpoints in dir."""
    if not os.path.isdir(directory):
        return []
    out = []
    for fname in os.listdir(directory):
        if not fname.endswith(".pt"):
            continue
        path = os.path.join(directory, fname)
        try:
            bundle = torch.load(path, weights_only=False, map_location="cpu")
            out.append((bundle.get("iter_idx", -1), path))
        except Exception:
            continue
    out.sort()
    return out


def latest_checkpoint(directory: str) -> Optional[str]:
    """Path of the latest-iteration checkpoint in `directory`, or None."""
    cps = list_checkpoints(directory)
    return cps[-1][1] if cps else None
