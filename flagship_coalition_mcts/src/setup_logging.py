"""Unified logging utility for training scripts.

Provides a consistent log format across CD-MCTS, CMAZ, and wreath
training scripts so that history JSON files and stdout dumps look the
same. Also handles file-and-stdout dual logging for long runs.

Usage:
    from flagship_coalition_mcts.src.setup_logging import setup_logging
    log = setup_logging("cdmcts_cc_seed0", log_dir="logs/")
    log.info("Starting iteration 1")

Format:
    [TIMESTAMP] [LEVEL] [TAG] message
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional


def setup_logging(
    tag: str,
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
    also_stdout: bool = True,
) -> logging.Logger:
    """Configure a logger for a training run.

    Args:
        tag: short identifier (used as logger name and in log lines).
        log_dir: directory for log files; created if missing. None disables
            file logging.
        level: minimum log level.
        also_stdout: also stream to stdout.
    """
    logger = logging.getLogger(tag)
    logger.setLevel(level)
    # Remove any prior handlers (e.g. from a previous setup_logging call
    # in the same process).
    for h in logger.handlers[:]:
        logger.removeHandler(h)

    fmt = logging.Formatter(
        f"[%(asctime)s] [%(levelname)s] [{tag}] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if also_stdout:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        path = os.path.join(log_dir, f"{tag}.log")
        fh = logging.FileHandler(path)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    # Prevent double-logging when libraries also configure root.
    logger.propagate = False
    return logger


class LogProgress:
    """Context manager that emits a progress line every N seconds.

    Useful for long iterations where step-level prints are too noisy
    but silence makes the training look hung.

        with LogProgress(logger, interval=30) as p:
            for step in range(10000):
                ... do work ...
                p.tick(step, step_loss)
    """

    def __init__(self, logger: logging.Logger, interval: float = 30.0) -> None:
        self.logger = logger
        self.interval = interval
        self._last_t = 0.0

    def __enter__(self):
        import time
        self._last_t = time.time()
        return self

    def __exit__(self, *args):
        return False

    def tick(self, step: int, *info) -> None:
        import time
        now = time.time()
        if now - self._last_t >= self.interval:
            self._last_t = now
            msg = f"step {step}"
            if info:
                msg += " " + " ".join(str(x) for x in info)
            self.logger.info(msg)
