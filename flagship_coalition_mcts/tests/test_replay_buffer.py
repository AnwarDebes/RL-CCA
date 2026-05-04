"""Tests for the replay buffer."""

from __future__ import annotations

import random

from flagship_coalition_mcts.src.replay_buffer import ReplayBuffer


def test_add_and_len():
    buf = ReplayBuffer(capacity=10)
    assert len(buf) == 0
    buf.add({"x": 1})
    assert len(buf) == 1


def test_capacity_cap_drops_oldest():
    buf = ReplayBuffer(capacity=3)
    for i in range(5):
        buf.add({"i": i})
    assert len(buf) == 3
    # Sample many; the oldest entries (0, 1) must be gone
    samples = buf.sample(100, mode="uniform")
    keys = {s["i"] for s in samples}
    assert keys.issubset({2, 3, 4})


def test_uniform_sampling_returns_correct_count():
    buf = ReplayBuffer(capacity=100)
    for i in range(50):
        buf.add({"i": i})
    samples = buf.sample(20)
    assert len(samples) == 20


def test_recent_weighted_biases_toward_recent():
    """Recent-weighted should pick the newest entries more often than uniform."""
    buf = ReplayBuffer(capacity=200)
    for i in range(200):
        buf.add({"i": i})
    rng = random.Random(0)
    samples = buf.sample(2000, mode="recent_weighted", half_life=20, rng=rng)
    # Mean sample index should be > 100 (more recent than uniform's expected 100)
    mean = sum(s["i"] for s in samples) / len(samples)
    assert mean > 130, f"recent_weighted mean = {mean}, expected > 130"


def test_state_dict_roundtrip():
    buf = ReplayBuffer(capacity=10)
    for i in range(5):
        buf.add({"i": i})
    sd = buf.state_dict()
    buf2 = ReplayBuffer(capacity=1)
    buf2.load_state_dict(sd)
    assert len(buf2) == 5
    assert buf2.capacity == 10


def test_clear():
    buf = ReplayBuffer(capacity=10)
    buf.add_many([{"i": i} for i in range(5)])
    buf.clear()
    assert len(buf) == 0


def test_sample_from_empty_returns_empty():
    buf = ReplayBuffer(capacity=10)
    out = buf.sample(5)
    assert out == []
