"""What each operator actually does to a tensor, tested so a broken one fails.

Written after a mutation audit found that token_mask, channel_mask, droppath and
gain_scale could each be replaced by the identity function with the whole suite
still passing. token_mask at before_attention_norm is the recommended deployment
configuration, so its behaviour being untested was the largest hole in the suite.

Every test here is written to fail under at least one specific wrong
implementation, and the wrong implementation it targets is named in its
docstring. A test that passes for the identity function is not a test.

These target psbd.operators, the tree that survives the rewrite. The older
tests/test_perturbations.py covers the same operators through defences.*.
"""

import pytest
import torch

from psbd.operators import build_perturbation

# A batch large enough that a masking fraction is measurable rather than noise.
BATCH = 64
TOKENS = 16
CHANNELS = 32
RATE = 0.5


@pytest.fixture
def activation() -> torch.Tensor:
    """A (batch, tokens, channels) activation with no zeros of its own.

    Every test below reads "is 0" as "was removed", so an input that could
    contain a genuine 0 would make that reading unsound.
    """
    generator = torch.Generator().manual_seed(0)
    values = torch.randn(BATCH, TOKENS, CHANNELS, generator=generator)

    return torch.where(values.abs() < 0.1, torch.full_like(values, 0.5), values)


def perturb(name: str, activation: torch.Tensor, rate: float = RATE, seed: int = 0):
    torch.manual_seed(seed)
    module = build_perturbation(name)(rate).train()

    return module(activation)


def test_token_mask_removes_whole_tokens_and_nothing_partial(activation):
    """Fails for the identity, and for elementwise dropout.

    The defining property is granularity: a token is either entirely present or
    entirely absent. Elementwise dropout zeroes scattered entries within a token,
    which this rejects.
    """
    output = perturb("token_mask", activation)

    zeroed_entries = output == 0
    fully_zeroed = zeroed_entries.all(dim=2)
    partially_zeroed = zeroed_entries.any(dim=2) & ~fully_zeroed

    assert not partially_zeroed.any(), "a token was partly masked, so this is not a token mask"
    assert fully_zeroed.any(), "no token was masked at all, so nothing was perturbed"

    # Token 0 is the class token and is never masked on a rank-3 activation.
    assert not fully_zeroed[:, 0].any(), "the class token was masked"

    # The masked fraction must track the rate over the maskable tokens.
    masked_fraction = fully_zeroed[:, 1:].float().mean().item()
    assert masked_fraction == pytest.approx(RATE, abs=0.08)


def test_channel_mask_removes_whole_channels_shared_across_tokens(activation):
    """Fails for the identity, and for elementwise dropout.

    The defining property is that a masked channel is masked at EVERY token of
    that sample. Elementwise dropout masks a channel at some tokens and not
    others, which this rejects.
    """
    output = perturb("channel_mask", activation)

    zeroed_entries = output == 0
    channel_zeroed_everywhere = zeroed_entries.all(dim=1)
    channel_zeroed_somewhere = zeroed_entries.any(dim=1)

    assert torch.equal(channel_zeroed_everywhere, channel_zeroed_somewhere), (
        "a channel was masked at some tokens but not others, so it is not shared"
    )
    assert channel_zeroed_everywhere.any(), "no channel was masked at all"

    masked_fraction = channel_zeroed_everywhere.float().mean().item()
    assert masked_fraction == pytest.approx(RATE, abs=0.08)


def test_channel_masks_differ_between_samples(activation):
    """Fails for an implementation that draws one mask for the whole batch.

    A batch-shared mask would make a sample's perturbation depend on which other
    samples shared its batch, which is the coupling that audit finding A2
    identified in the Gaussian operator.
    """
    output = perturb("channel_mask", activation)
    channel_zeroed = (output == 0).all(dim=1)  # (batch, channels)

    distinct_masks = {tuple(row.tolist()) for row in channel_zeroed}
    assert len(distinct_masks) > 1, "every sample got the same mask, so it is batch-shared"


def test_droppath_is_all_or_nothing_per_sample(activation):
    """Fails for the identity, for elementwise dropout, and for a token mask.

    The property is that a sample is either untouched or entirely zeroed. The
    older test asserted (fully_zeroed OR not_fully_zeroed), which is a tautology
    true of every possible implementation.
    """
    output = perturb("droppath", activation)

    zeroed_entries = output == 0
    fully_zeroed = zeroed_entries.flatten(1).all(dim=1)
    any_zeroed = zeroed_entries.flatten(1).any(dim=1)

    assert torch.equal(fully_zeroed, any_zeroed), (
        "a sample was partly zeroed, so this is not a per-sample drop"
    )
    assert fully_zeroed.any(), "no sample was dropped"
    assert not fully_zeroed.all(), "every sample was dropped"

    dropped_fraction = fully_zeroed.float().mean().item()
    assert dropped_fraction == pytest.approx(RATE, abs=0.15)


def test_gain_scale_multiplies_every_entry_by_one_plus_the_rate(activation):
    """Fails for the identity, and for any operator that removes anything."""
    rate = 0.4
    output = perturb("gain_scale", activation, rate=rate)

    assert torch.allclose(output, activation * (1.0 + rate), atol=1e-6)
    assert not (output == 0).any(), "gain_scale must remove nothing"


def test_gain_scale_is_deterministic_across_passes(activation):
    """Its registry entry claims determinism, and PSU's pass count depends on it."""
    module = build_perturbation("gain_scale")(0.4).train()

    assert torch.equal(module(activation), module(activation))


@pytest.mark.parametrize(
    "name", ["dropout", "token_mask", "channel_mask", "droppath"]
)
def test_inverted_scaling_preserves_the_mean_on_a_shifted_input(name, activation):
    """Fails when the 1 / (1 - rate) survivor scaling is removed.

    The older version of this test used a zero-mean input, where dropping the
    scaling multiplies a mean of approximately 0 by the keep probability and
    leaves it approximately 0. Shifting the input away from 0 is what gives the
    assertion something to detect: without the scaling the mean falls by the drop
    rate, which at 0.5 is a factor of 2.
    """
    shifted = activation + 3.0

    # droppath decides per sample rather than per entry, so a single pass keeps a
    # Binomial(batch, rate) fraction of the mean and its spread is far wider than
    # the other operators'. It needs more repeats to resolve the same effect, and
    # 24 leaves it about 1.8 standard errors from the truth, which is noise rather
    # than a missing scale factor.
    repeats = 400 if name == "droppath" else 64
    means = torch.stack(
        [perturb(name, shifted, seed=seed).mean() for seed in range(repeats)]
    )
    relative_error = ((means.mean() - shifted.mean()) / shifted.mean()).abs().item()
    standard_error = (means.std() / repeats**0.5 / shifted.mean()).abs().item()

    # Removing the survivor scaling multiplies the mean by the keep probability,
    # so at rate 0.5 the error would be 0.5. The tolerance sits far below that and
    # comfortably above the sampling noise, so the test detects the real failure
    # without depending on a lucky draw.
    assert relative_error < max(0.02, 4 * standard_error), (
        f"{name} moved the mean by {relative_error:.4f} "
        f"(standard error {standard_error:.4f}), so inverted scaling is missing"
    )


@pytest.mark.parametrize(
    "name", ["dropout", "token_mask", "channel_mask", "droppath", "gaussian", "rademacher"]
)
def test_every_stochastic_operator_is_inert_in_eval_mode(name, activation):
    """A probe that fires outside its sweep contaminates the baseline it is measured against."""
    module = build_perturbation(name)(RATE).eval()

    assert torch.equal(module(activation), activation)


@pytest.mark.parametrize(
    "name",
    ["dropout", "token_mask", "channel_mask", "droppath", "gaussian", "rademacher", "gain_scale"],
)
def test_a_zero_rate_is_the_identity(name, activation):
    """Rate 0 is the no-perturbation control, and every sweep relies on it."""
    module = build_perturbation(name)(0.0).train()

    assert torch.equal(module(activation), activation)


@pytest.mark.parametrize("name", ["token_mask", "channel_mask", "droppath", "dropout"])
def test_masking_more_removes_more(name, activation):
    """Fails for any implementation whose output ignores the rate."""
    fractions = [
        (perturb(name, activation, rate=rate, seed=1) == 0).float().mean().item()
        for rate in (0.1, 0.5, 0.9)
    ]

    assert fractions == sorted(fractions), f"{name} removed {fractions} at rates 0.1, 0.5, 0.9"
    assert fractions[-1] > fractions[0] + 0.3, f"{name} barely responded to the rate"


def test_the_shape_is_always_preserved(activation):
    """Every operator is a drop-in for nn.Dropout, so none may change the shape."""
    for name in ("dropout", "token_mask", "channel_mask", "droppath", "gaussian", "rademacher", "gain_scale"):
        assert perturb(name, activation).shape == activation.shape, name


class TestMultiProbeInversionFragility:
    """A probe an adaptive attacker has inverted poisons the min-rank union.

    The union exists as defence in depth: evading one probe should not evade the
    detector. But the union takes the most extreme evidence across probes, so a
    probe whose ranking has been reversed contributes confident wrong evidence
    rather than merely useless evidence, and the union inherits it.

    Measured on a real evasive checkpoint, the probed position falls to AUROC
    0.023 while an unprobed position still reads 0.913, and the union of the 2
    lands below the unprobed position alone. These tests pin the mechanism on
    synthetic data so the behaviour is documented rather than rediscovered.
    """

    @staticmethod
    def _probe(seed, clean_shift, backdoor_shift, count=600):
        generator = torch.Generator().manual_seed(seed)
        validation = torch.randn(count, generator=generator)
        clean = torch.randn(count, generator=generator) + clean_shift
        backdoor = torch.randn(count, generator=generator) + backdoor_shift
        return validation, clean, backdoor

    def test_a_union_of_healthy_probes_stays_healthy(self):
        from psbd.decision import multi_probe_detection

        first = self._probe(0, 0.0, -2.0)
        second = self._probe(1, 0.0, -2.0)
        union = multi_probe_detection(
            [first[0], second[0]], [first[1], second[1]], [first[2], second[2]], 0.25
        )

        assert union["auroc"] > 0.85, "2 healthy probes should still detect"

    def test_one_inverted_probe_drags_the_union_to_chance(self):
        """The finding: an inverted probe is worse than a useless one."""
        from psbd.decision import detection_report, multi_probe_detection

        healthy = self._probe(0, 0.0, -2.0)
        inverted = self._probe(1, -2.0, 0.0)

        healthy_alone = detection_report(*healthy, 0.25)["auroc"]
        inverted_alone = detection_report(*inverted, 0.25)["auroc"]
        union = multi_probe_detection(
            [healthy[0], inverted[0]],
            [healthy[1], inverted[1]],
            [healthy[2], inverted[2]],
            0.25,
        )

        assert healthy_alone > 0.85
        assert inverted_alone < 0.15, "the second probe must actually be inverted"
        assert union["auroc"] < 0.6, (
            f"union reached {union['auroc']:.3f}; an inverted probe should collapse it"
        )
        assert union["auroc"] < healthy_alone - 0.2, (
            "the union must be measurably worse than the healthy probe alone"
        )

    def test_a_useless_probe_costs_far_less_than_an_inverted_one(self):
        """Distinguishes 'carries no signal' from 'carries reversed signal'."""
        from psbd.decision import multi_probe_detection

        healthy = self._probe(0, 0.0, -2.0)
        useless = self._probe(1, 0.0, 0.0)
        inverted = self._probe(2, -2.0, 0.0)

        with_useless = multi_probe_detection(
            [healthy[0], useless[0]], [healthy[1], useless[1]], [healthy[2], useless[2]], 0.25
        )["auroc"]
        with_inverted = multi_probe_detection(
            [healthy[0], inverted[0]], [healthy[1], inverted[1]], [healthy[2], inverted[2]], 0.25
        )["auroc"]

        assert with_useless > with_inverted + 0.15, (
            f"useless {with_useless:.3f} should cost far less than inverted {with_inverted:.3f}"
        )


class TestMedianRankUnion:
    """The median reduction, added because the min union adopts an inverted probe.

    Same construction as TestMultiProbeInversionFragility, so the two classes read
    against each other: those tests pin the failure, these pin the fix and its
    limit.
    """

    @staticmethod
    def _probe(seed, clean_shift, backdoor_shift, count=800):
        generator = torch.Generator().manual_seed(seed)
        validation = torch.randn(count, generator=generator)
        clean = torch.randn(count, generator=generator) + clean_shift
        backdoor = torch.randn(count, generator=generator) + backdoor_shift
        return validation, clean, backdoor

    def _union(self, probes, reduction):
        from psbd.decision import multi_probe_detection

        return multi_probe_detection(
            [p[0] for p in probes],
            [p[1] for p in probes],
            [p[2] for p in probes],
            0.25,
            reduction=reduction,
        )["auroc"]

    def test_median_costs_nothing_when_every_probe_is_healthy(self):
        healthy = [self._probe(seed, 0.0, -2.0) for seed in range(5)]

        assert self._union(healthy, "median") >= self._union(healthy, "min") - 0.01

    def test_median_survives_a_single_inverted_probe_and_min_does_not(self):
        probes = [self._probe(0, 0.0, -2.0), self._probe(1, 0.0, -2.0)]
        probes.append(self._probe(9, -2.0, 0.0))  # inverted by an adaptive attacker

        by_min = self._union(probes, "min")
        by_median = self._union(probes, "median")

        assert by_median > by_min + 0.15, (
            f"median {by_median:.3f} should clearly beat min {by_min:.3f} here"
        )
        assert by_median > 0.75, "median should still detect with a majority healthy"

    def test_median_fails_once_the_attacker_owns_the_majority(self):
        """The honest limit. A majority vote is only as good as its majority."""
        probes = [self._probe(seed, 0.0, -2.0) for seed in range(2)]
        probes += [self._probe(20 + seed, -2.0, 0.0) for seed in range(3)]

        assert self._union(probes, "median") < 0.5, (
            "with 3 of 5 inverted the median should follow the attacker"
        )

    def test_an_unknown_reduction_is_refused(self):
        from psbd.scores import multi_probe_score

        probe = self._probe(0, 0.0, -2.0)
        with pytest.raises(ValueError, match="Unknown probe reduction"):
            multi_probe_score([probe[1]], [probe[0]], reduction="mean")

    def test_auroc_is_reported_inside_every_rule_block(self):
        """A caller reading a rule block must not find None where the number is."""
        from psbd.decision import multi_probe_detection

        probes = [self._probe(seed, 0.0, -2.0) for seed in range(3)]
        report = multi_probe_detection(
            [p[0] for p in probes], [p[1] for p in probes], [p[2] for p in probes], 0.25
        )

        for rule, block in report["by_rule"].items():
            assert block.get("auroc") is not None, f"{rule} block has no auroc"
            assert block["auroc"] == pytest.approx(report["auroc"]), (
                "auroc is threshold free, so every block must agree with the top level"
            )
