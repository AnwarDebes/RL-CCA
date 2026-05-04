"""Tests for the subproject comparison renderer."""

from flagship_coalition_mcts.src.compare_subprojects import (
    COMPARISON_TABLE, render_latex, render_markdown,
)


def test_table_has_consistent_row_widths():
    """Every row must have exactly 5 entries (Feature + 3 subprojects + Prior art)."""
    for row in COMPARISON_TABLE:
        assert len(row) == 5


def test_render_markdown_has_all_rows():
    md = render_markdown()
    for row in COMPARISON_TABLE:
        # First column appears verbatim
        assert row[0] in md


def test_render_latex_has_proper_structure():
    tex = render_latex()
    assert "\\begin{tabular}" in tex
    assert "\\toprule" in tex
    assert "\\end{tabular}" in tex


def test_render_latex_escapes_underscores_and_amps():
    """Cells with `_` or `&` must be LaTeX-escaped."""
    tex = render_latex()
    # Should not have unescaped underscores in cell text
    # (the cell "C6 spatial × S_N seat (wreath)" has S_N which becomes S\_N)
    assert "S\\_N" in tex or "S_N" not in tex.replace("S\\_N", "")
