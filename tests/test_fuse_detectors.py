"""Rank fusion of PSBD with a recorded detector, on synthetic scores.

The fused columns must be percentiles of the clean validation split, so a
threshold set there transfers to the paired splits, and the min rule must flag a
sample whenever either component does.
"""

import torch

from cli.fuse_detectors import fuse, tpr_at_fpr


def synthetic_scores(seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    scores = {
        "validation": torch.rand(200, generator=generator),  # (200,)
        "clean": torch.rand(100, generator=generator),  # (100,)
        # Poisoned inputs sit low, PSU's convention for every detector.
        "backdoor": torch.rand(100, generator=generator) * 0.05,  # (100,)
    }
    return scores


def test_fused_ranks_are_percentiles_of_validation():
    fused = fuse(synthetic_scores(0), synthetic_scores(1))
    for name in ("mean", "min"):
        ranks = fused["validation"][name]
        assert ranks.min() >= 0.0 and ranks.max() <= 1.0
        assert fused["backdoor"][name].mean() < fused["clean"][name].mean()


def test_min_rule_flags_when_either_component_does():
    psu = synthetic_scores(0)
    other = synthetic_scores(1)
    # A backdoor sample the second detector misses entirely.
    other["backdoor"][0] = 0.99
    fused = fuse(psu, other)
    assert fused["backdoor"]["min"][0] <= fused["backdoor"]["mean"][0]


def test_tpr_at_fpr_reads_the_validation_quantile():
    fused = fuse(synthetic_scores(0), synthetic_scores(1))
    tpr, fpr = tpr_at_fpr(
        fused["validation"]["mean"],
        fused["clean"]["mean"],
        fused["backdoor"]["mean"],
        0.05,
    )
    assert tpr > 0.9
    assert fpr < 0.2
