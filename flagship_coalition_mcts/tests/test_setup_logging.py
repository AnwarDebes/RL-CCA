"""Tests for the unified logging utility."""

from __future__ import annotations

import logging
import os
import tempfile
import time

from flagship_coalition_mcts.src.setup_logging import LogProgress, setup_logging


def test_setup_logging_returns_logger():
    log = setup_logging("test_tag", log_dir=None, also_stdout=False)
    assert isinstance(log, logging.Logger)
    assert log.name == "test_tag"


def test_setup_logging_writes_to_file():
    with tempfile.TemporaryDirectory() as d:
        log = setup_logging("test_file", log_dir=d, also_stdout=False)
        log.info("hello world")
        # Force flush
        for h in log.handlers:
            h.flush()
        path = os.path.join(d, "test_file.log")
        assert os.path.exists(path)
        with open(path) as f:
            content = f.read()
        assert "hello world" in content
        assert "[INFO]" in content
        assert "[test_file]" in content


def test_setup_logging_replaces_handlers_on_repeat():
    log1 = setup_logging("test_repeat", log_dir=None, also_stdout=False)
    n1 = len(log1.handlers)
    log2 = setup_logging("test_repeat", log_dir=None, also_stdout=False)
    n2 = len(log2.handlers)
    assert n1 == n2  # not accumulating handlers


def test_logprogress_ticks_after_interval():
    log = setup_logging("test_progress", log_dir=None, also_stdout=False)
    seen = []

    class Capture(logging.Handler):
        def emit(self, record):
            seen.append(record.getMessage())

    h = Capture()
    log.addHandler(h)
    with LogProgress(log, interval=0.0) as p:
        # interval=0 means every tick logs
        p.tick(1, "loss=0.5")
        p.tick(2, "loss=0.4")
    assert len(seen) >= 1
    assert any("loss=0.5" in s for s in seen)


def test_logprogress_does_not_tick_before_interval():
    log = setup_logging("test_progress2", log_dir=None, also_stdout=False)
    seen = []

    class Capture(logging.Handler):
        def emit(self, record):
            seen.append(record.getMessage())

    log.addHandler(Capture())
    with LogProgress(log, interval=10.0) as p:
        # Several ticks within 1ms of each other
        p.tick(1)
        p.tick(2)
        p.tick(3)
    assert len(seen) == 0
