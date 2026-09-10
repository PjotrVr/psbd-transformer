"""Notebooks must not restate a protocol constant they could import.

A notebook that writes `TARGET_SHIFT = 0.6` beside a codebase whose rule is 0.8 is
not a style problem. Notebook 14 did exactly that and reported TPR@1%FPR of 0.507
where PSBD's own rule gives 0.633, and nothing failed, because a literal is always
a valid float. The number that reached a table was wrong by 0.126 and the notebook
ran clean.

0.6 is a real constant in this codebase, it is just a different one: it belongs to
SHIFT_MATCH_TARGETS, the grid for comparing placements at matched perturbation
strength. Two protocols, both legitimate, and a bare literal cannot say which one
it meant. An import can.
"""

import json
import re
from pathlib import Path

import pytest

from defences.decision import ADAPTIVE_SHIFT_TARGET, HEADLINE_QUANTILE

NOTEBOOK_DIRECTORY = Path("notebooks")

# Names a notebook must import rather than assign a literal to, mapped to the
# constant in defences.decision that owns the value.
PROTOCOL_CONSTANTS = {
    "TARGET_SHIFT": ("ADAPTIVE_SHIFT_TARGET", ADAPTIVE_SHIFT_TARGET),
    "SIGMA_TARGET": ("ADAPTIVE_SHIFT_TARGET", ADAPTIVE_SHIFT_TARGET),
    "HEADLINE_QUANTILE": ("HEADLINE_QUANTILE", HEADLINE_QUANTILE),
}

LITERAL_ASSIGNMENT = re.compile(
    r"^\s*(" + "|".join(PROTOCOL_CONSTANTS) + r")\s*=\s*([0-9.]+)\s*$", re.MULTILINE
)


def notebook_paths():
    return sorted(NOTEBOOK_DIRECTORY.glob("*.ipynb"))


def code_source(path):
    notebook = json.loads(path.read_text())
    return "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


@pytest.mark.parametrize("path", notebook_paths(), ids=lambda p: p.stem)
def test_protocol_constants_are_imported_not_restated(path):
    offences = [
        f"{name} = {value} (import {PROTOCOL_CONSTANTS[name][0]} from defences.decision, "
        f"which is {PROTOCOL_CONSTANTS[name][1]})"
        for name, value in LITERAL_ASSIGNMENT.findall(code_source(path))
    ]
    assert not offences, f"{path.name} restates a protocol constant: {offences}"


def test_the_two_shift_targets_are_actually_different():
    """Guards the premise: if these ever coincide, this whole test is vacuous."""
    from defences.decision import SHIFT_MATCH_TARGETS

    placement_targets = set(SHIFT_MATCH_TARGETS) - {ADAPTIVE_SHIFT_TARGET}
    assert placement_targets, (
        "SHIFT_MATCH_TARGETS no longer holds a value distinct from "
        "ADAPTIVE_SHIFT_TARGET, so confusing the 2 protocols is no longer possible "
        "and this test proves nothing"
    )
