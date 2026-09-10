"""The 2 placements every table names, held together with the basis declaration.

defences.decision names the recommended and the published placement as
constants, configs/psbd_basis.json declares the same 2 by note. Either could be
edited alone, so this test is what makes them 1 fact.
"""

import json

from defences.decision import PUBLISHED_PLACEMENT, RECOMMENDED_PLACEMENT

BASIS_PATH = "configs/psbd_basis.json"


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
    assert (entry["position_config"], entry["perturbation"]) == (
        "before_attention_norm",
        "token_mask",
    )


def test_the_published_placement_is_dropout_after_the_residual_add():
    entry = basis_entry_noted("the published ConvNet placement")
    assert entry["id"] == PUBLISHED_PLACEMENT
    assert (entry["position_config"], entry["perturbation"]) == (
        "post_residual",
        "dropout",
    )
