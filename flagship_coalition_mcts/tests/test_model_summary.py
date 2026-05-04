"""Tests for the model summary utility."""

from __future__ import annotations

import torch
import torch.nn as nn

from flagship_coalition_mcts.src.model_summary import (
    count_parameters, latex_table, summary, total_params,
)


def _build_module():
    return nn.Sequential(
        nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 2),
    )


def test_total_params_correct():
    m = _build_module()
    expected = 4 * 8 + 8 + 8 * 2 + 2  # weight + bias for each linear
    assert total_params(m) == expected


def test_count_parameters_groups_by_top_module():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(4, 8)
            self.b = nn.Linear(8, 2)
        def forward(self, x):
            return self.b(self.a(x))
    m = Net()
    counts = count_parameters(m)
    assert "a" in counts
    assert "b" in counts
    assert counts["a"] == 4 * 8 + 8
    assert counts["b"] == 8 * 2 + 2


def test_summary_returns_non_empty_string():
    m = _build_module()
    s = summary(m, name="Test")
    assert "Test" in s
    assert "Total" in s


def test_latex_table_has_required_structure():
    m = _build_module()
    s = latex_table(m, name="Test")
    assert "\\begin{tabular}" in s
    assert "\\toprule" in s
    assert "\\bottomrule" in s
    assert "\\end{tabular}" in s


def test_summary_with_no_grad_params_excluded():
    """Parameters with requires_grad=False should not be counted."""
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(4, 8)
            self.b = nn.Linear(8, 2)
            for p in self.b.parameters():
                p.requires_grad = False
        def forward(self, x):
            return self.b(self.a(x))
    m = Net()
    expected = 4 * 8 + 8  # only `a`
    assert total_params(m) == expected
    counts = count_parameters(m)
    assert "a" in counts
    assert "b" not in counts
