"""Regression tests for launch scripts to prevent the -u-flag bug from returning.

Bug 4 in BUGS_FOUND.md: missing `-u` causes stdout buffering and loss of
live training visibility. This test enforces that any script that
invokes `python script.py` and redirects stdout to a file uses `-u`.
"""

from __future__ import annotations

import os
import re

import pytest


_NEXUS_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)


def _check_script_has_unbuffered(path: str) -> None:
    """Assert that any python invocation that redirects to a file uses -u."""
    with open(path) as f:
        content = f.read()
    # Find lines that pipe to a file and invoke python
    lines = content.splitlines()
    in_command = False
    cur_cmd = ""
    for line in lines:
        # Build up multi-line commands
        if line.rstrip().endswith("\\"):
            cur_cmd += line.rstrip("\\").strip() + " "
            in_command = True
            continue
        if in_command:
            cur_cmd += line.strip()
            in_command = False
            line_to_check = cur_cmd
            cur_cmd = ""
        else:
            line_to_check = line
        # Now check the (possibly multi-line) command.
        # Pattern: invokes python AND redirects to a file (>"...").
        # Also matches: > $LOG, >> $LOG, etc.
        if re.search(r"python.*>(>?)\s*[\$\"\w/_.-]+\s*(?:2>&1)?\s*&?", line_to_check):
            if "python" in line_to_check:
                # Check for -u flag
                if not re.search(r"python\s+-u", line_to_check):
                    raise AssertionError(
                        f"{path}: redirects python output to file without "
                        f"`-u` flag - Bug 4 risk. Line:\n  {line_to_check}"
                    )


def test_launch_phase2_uses_unbuffered():
    path = os.path.join(_NEXUS_ROOT, "scripts", "launch_phase2.sh")
    if not os.path.exists(path):
        pytest.skip(f"{path} not found")
    _check_script_has_unbuffered(path)


def test_restart_phase2_uses_unbuffered():
    path = os.path.join(_NEXUS_ROOT, "scripts", "restart_phase2.sh")
    if not os.path.exists(path):
        pytest.skip(f"{path} not found")
    _check_script_has_unbuffered(path)


def test_helper_correctly_flags_missing_u():
    """The helper itself must reject scripts without -u."""
    import tempfile
    bad_script = """#!/bin/bash
nohup ./venv/bin/python script.py > log.txt 2>&1 &
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
        f.write(bad_script)
        path = f.name
    try:
        with pytest.raises(AssertionError):
            _check_script_has_unbuffered(path)
    finally:
        os.unlink(path)


def test_helper_accepts_dash_u():
    import tempfile
    good_script = """#!/bin/bash
nohup ./venv/bin/python -u script.py > log.txt 2>&1 &
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
        f.write(good_script)
        path = f.name
    try:
        _check_script_has_unbuffered(path)  # should not raise
    finally:
        os.unlink(path)
