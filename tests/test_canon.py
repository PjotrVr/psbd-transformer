"""The 2 placements every table names, held together with the basis declaration.

defences.decision names the recommended and the published placement as
constants, configs/psbd_basis.json declares the same 2 by note. Either could be
edited alone, so this test is what makes them 1 fact.
"""

import json
import os

from defences.decision import (
    ADAPTIVE_SHIFT_TARGET,
    HEADLINE_QUANTILE,
    PLACEMENT_MATCH_TARGET,
    PSBD_QUANTILES,
    PUBLISHED_PLACEMENT,
    RECOMMENDED_PLACEMENT,
)

import pytest

# Part of the fast tier: see pyproject's fast marker.
pytestmark = pytest.mark.fast

# Absolute, so the test does not depend on pytest being invoked from the root.
BASIS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs",
    "psbd_basis.json",
)


def basis_entry_noted(fragment: str) -> dict:
    with open(BASIS_PATH) as handle:
        basis = json.load(handle)["basis"]
    matches = [entry for entry in basis if fragment in entry.get("note", "")]
    assert len(matches) == 1, (
        f"expected 1 basis entry noted {fragment!r}, got {matches}"
    )
    return matches[0]


def test_the_recommended_placement_is_the_one_the_basis_declares():
    entry = basis_entry_noted("the recommended deployment config")
    assert entry["id"] == RECOMMENDED_PLACEMENT
    assert (entry["position"], entry["operator"]) == (
        "before_attention_norm",
        "token_mask",
    )


def test_the_published_placement_is_dropout_after_the_residual_add():
    entry = basis_entry_noted("the published ConvNet placement")
    assert entry["id"] == PUBLISHED_PLACEMENT
    assert (entry["position"], entry["operator"]) == (
        "post_residual",
        "dropout",
    )


def read_declaration() -> dict:
    with open(BASIS_PATH) as handle:
        return json.load(handle)


def test_the_bars_the_panel_declares_are_the_ones_the_code_reads():
    """The 2 bars a cell is admitted by, held to the declaration.

    CLAUDE.md says the constants every table reads are held to
    configs/psbd_basis.json by this file, and until now only the 2 placement
    strings were. A table generator was found reading its own ASR bar of 0.5
    against the declared 0.85, which admitted cells the panel excludes.
    """
    declaration = read_declaration()
    assert declaration["asr_bar"] == 0.85
    assert declaration["clean_accuracy_drop_bar"] == -0.05


def test_the_2_rate_rules_are_the_canonical_targets():
    """The deployable rule and the comparison device, and that they differ.

    A single target would collapse the 2 rules into 1 and silently change every
    cross-placement comparison in the paper.
    """
    assert ADAPTIVE_SHIFT_TARGET == 0.8
    assert PLACEMENT_MATCH_TARGET == 0.6
    assert ADAPTIVE_SHIFT_TARGET != PLACEMENT_MATCH_TARGET


def test_the_quantile_ladder_carries_the_reported_operating_points():
    """The headline quantile, and the 2 false-positive budgets the paper reports.

    The paper reports AUROC with TPR at a 10% and a 20% false-positive budget, so
    both quantiles have to stay on the ladder or those columns lose their source.
    """
    assert HEADLINE_QUANTILE == 0.25
    assert HEADLINE_QUANTILE in PSBD_QUANTILES
    assert 0.10 in PSBD_QUANTILES
    assert 0.20 in PSBD_QUANTILES
    assert PSBD_QUANTILES == tuple(sorted(PSBD_QUANTILES))
