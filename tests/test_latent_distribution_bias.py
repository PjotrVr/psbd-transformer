"""What the latent distribution statistics do at the width and sample count they
are actually run at, rather than at the toy sizes the unit tests use.

test_latent_distribution.py checks each statistic against a constructed answer,
but it does so at 4 to 8 dimensions with hundreds of samples. The tools are run
at 768 dimensions with 1000 samples, where several of them stop estimating what
their docstring says. Three of the statistics fit their own direction, covariance
or spectrum on the same samples they are then evaluated on, and the resulting
bias scales with dim / num_samples, so it is invisible at 8 dimensions and
dominant at 768.

These tests pin that behaviour at the real operating point. They are not asserting
that the bias is desirable. They exist so that the bias is a documented, measured
quantity rather than a surprise inside a published number, and so that a later fix
announces itself by failing here instead of silently changing a figure.

Every case below has a known answer by construction: the two populations either
differ by nothing at all, or differ by a controlled amount.
"""

import math

import numpy as np
import pytest
import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms_v2
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import VisionTransformer
from torchvision.models.swin_transformer import SwinTransformer

from analysis.direction import backdoor_direction, project_onto_direction
from analysis.distribution import (
    COVARIANCE_SHRINKAGE,
    effective_rank,
    layer_distribution_row,
    layer_distribution_table,
    mahalanobis_distances,
    separation_auroc,
    standardized_mean_shift,
)
from analysis.features import (
    _as_token_sequence,
    extract_layer_features,
    transformer_blocks,
)
from models.backbones import network_core
from models.positions import BLOCK_TYPES, POSITION_REGISTRY, resolve_targets

# ViT-B/16's residual width and the default sample count in cases.load_latent_case.
DEPLOYED_DIM = 768
DEPLOYED_SAMPLES = 1000

# Swin-S block output widths, layers 1 to 24, from torchvision's stage layout.
SWIN_S_BLOCK_WIDTHS = [96] * 2 + [192] * 2 + [384] * 18 + [768] * 2


def paired_null(num_samples: int, dim: int, seed: int, trigger_scale: float = 0.3):
    """A trigger that perturbs every sample but writes no consistent direction.

    This is the null every one of these statistics is implicitly tested against
    when a benign checkpoint is used as the negative control: the trigger moves
    the representation, but it moves it nowhere in particular.
    """
    generator = torch.Generator().manual_seed(seed)
    clean = torch.randn(num_samples, dim, generator=generator)
    delta = trigger_scale * torch.randn(num_samples, dim, generator=generator)

    return clean, clean + delta


def unpaired_null(num_samples: int, dim: int, seed: int):
    """Two independent draws from one distribution, so the truth is exactly chance.

    Harsher than paired_null and closer to what a deep layer looks like once a
    strong trigger has decorrelated the 2 populations. It is the case where
    fitting the direction on the scored samples does the most damage.
    """
    generator = torch.Generator().manual_seed(seed)
    clean = torch.randn(num_samples, dim, generator=generator)
    backdoor = torch.randn(num_samples, dim, generator=generator)

    return clean, backdoor


def fitted_separation(clean: torch.Tensor, backdoor: torch.Tensor) -> float:
    """The statistic exactly as layer_distribution_row computes it."""
    direction = backdoor_direction(clean, backdoor)
    auroc = separation_auroc(
        project_onto_direction(clean, direction),
        project_onto_direction(backdoor, direction),
    )
    return auroc


def held_out_separation(
    clean: torch.Tensor, backdoor: torch.Tensor, folds: int = 4, seed: int = 0
) -> float:
    """The same statistic with the direction fitted off the evaluated samples.

    The only difference from fitted_separation is which samples the direction is
    estimated on, so any gap between the two is the cost of reusing the samples.
    """
    num_samples = clean.shape[0]
    order = torch.randperm(num_samples, generator=torch.Generator().manual_seed(seed))

    aurocs = []
    for fold in range(folds):
        held = order[fold * num_samples // folds : (fold + 1) * num_samples // folds]
        fit = torch.ones(num_samples, dtype=torch.bool)
        fit[held] = False

        direction = backdoor_direction(clean[fit], backdoor[fit])
        aurocs.append(
            separation_auroc(
                project_onto_direction(clean[held], direction),
                project_onto_direction(backdoor[held], direction),
            )
        )

    return float(np.mean(aurocs))


def test_separation_auroc_is_optimistically_biased_at_the_deployed_width():
    """The direction is fitted on the samples the AUROC is then measured on.

    On two populations that differ by nothing but zero-mean noise the honest
    answer is 0.5. At 768 dimensions and 1000 samples the fitted-direction
    estimate reads close to 0.58 instead, while the identical statistic with the
    direction fitted off the evaluated samples returns 0.5. The gap is entirely
    the reuse of the samples.
    """
    clean, backdoor = paired_null(DEPLOYED_SAMPLES, DEPLOYED_DIM, seed=101)

    fitted = fitted_separation(clean, backdoor)
    honest = held_out_separation(clean, backdoor)

    assert honest == pytest.approx(0.5, abs=0.03), (
        "the leakage-free estimator must recover the truth on null data"
    )
    assert fitted > 0.55, (
        "the reported statistic is expected to be inflated here; if this now "
        "passes at 0.5 the circularity has been fixed and this test should be "
        "replaced by a calibration test"
    )
    assert fitted - honest > 0.05


def test_the_separation_bias_is_invisible_at_the_width_the_unit_tests_use():
    """Why the existing suite does not catch it: the bias scales with dim / n.

    At 6 dimensions the inflation is inside the tolerance the unit tests already
    allow, so a test built at that width certifies nothing about the 768
    dimensional case it is standing in for.
    """
    narrow_clean, narrow_backdoor = paired_null(DEPLOYED_SAMPLES, 6, seed=102)
    wide_clean, wide_backdoor = paired_null(DEPLOYED_SAMPLES, DEPLOYED_DIM, seed=102)

    narrow_bias = fitted_separation(narrow_clean, narrow_backdoor) - 0.5
    wide_bias = fitted_separation(wide_clean, wide_backdoor) - 0.5

    assert narrow_bias < 0.02
    assert wide_bias > 10 * narrow_bias


def test_standardized_shift_has_a_positive_floor_on_data_with_no_shift():
    """Cohen's d must be 0 when the populations have the same mean.

    The statistic projects onto the fitted mean-difference direction, so the
    numerator is the norm of that direction by construction and can never be
    negative. On null data at the deployed width it reports a moderate effect.
    """
    clean, backdoor = paired_null(DEPLOYED_SAMPLES, DEPLOYED_DIM, seed=103)

    shift = standardized_mean_shift(clean, backdoor)
    assert shift > 0.2, "expected the known null floor at 768 dimensions"


def test_standardized_shift_can_never_report_a_negative_effect():
    """The sign carries no information, because the direction points at the answer.

    Swapping the two populations returns the same magnitude, so a reader cannot
    use the sign to tell "no effect" from "effect in the other direction".
    """
    generator = torch.Generator().manual_seed(104)
    clean = torch.randn(300, 20, generator=generator)
    backdoor = clean - 1.5

    forward = standardized_mean_shift(clean, backdoor)
    reversed_order = standardized_mean_shift(backdoor, clean)

    assert forward > 0
    assert forward == pytest.approx(reversed_order, rel=1e-4)


def test_cross_fitting_removes_the_width_dependent_null_floor():
    """Swin quadruples its width across the stack, and the in-sample floor rose with it.

    Feed identical null data at each Swin-S stage width. Nothing differs between
    the layers, so a layer profile must be flat. Fitting the direction on the same
    samples it scores made it rise instead, which turned Swin's stage widths into
    an apparent depth trend in the exact shape a backdoor appearing with depth
    would produce. Cross-fitting is what makes the profile flat.
    """
    in_sample, cross_fitted = [], []
    for width in (96, 192, 384, 768):
        clean, backdoor = unpaired_null(DEPLOYED_SAMPLES, width, seed=105)
        direction = backdoor_direction(clean, backdoor)
        in_sample.append(
            separation_auroc(
                project_onto_direction(clean, direction),
                project_onto_direction(backdoor, direction),
            )
        )
        cross_fitted.append(layer_distribution_row(clean, backdoor)["separation_auroc"])

    # The bug, kept as the contrast that justifies the fix.
    assert in_sample == sorted(in_sample), "the in-sample floor is monotone in width"
    assert in_sample[-1] - in_sample[0] > 0.03

    # The fix: no width dependence survives, and every width sits at chance.
    assert max(cross_fitted) - min(cross_fitted) < 0.08, (
        f"a width-dependent floor remains: {cross_fitted}"
    )
    for width, auroc in zip((96, 192, 384, 768), cross_fitted):
        assert auroc == pytest.approx(0.5, abs=0.06), f"width {width} is off chance"


def test_cross_fitted_separation_is_stable_in_sample_count():
    """Two cases run at different sample counts must be comparable.

    load_latent_case takes samples as a free keyword, so nothing stops one case
    being measured at 200 and another at 5000. In-sample separation fell from
    0.66 to 0.53 across that range on pure null data, which exceeds most real
    effects. Cross-fitted separation does not move, so the comparison is sound.

    effective_rank still moves, and deliberately so: the sample covariance is
    biased downward by roughly the Marchenko-Pastur factor. That is why
    rank_ratio, where the bias cancels, is the column to quote.
    """
    rows = {
        n: layer_distribution_row(*unpaired_null(n, DEPLOYED_DIM, seed=106))
        for n in (200, 1000, 5000)
    }

    aurocs = [rows[n]["separation_auroc"] for n in (200, 1000, 5000)]
    for n, auroc in zip((200, 1000, 5000), aurocs):
        assert auroc == pytest.approx(0.5, abs=0.08), f"n={n} is off chance at {auroc}"
    assert max(aurocs) - min(aurocs) < 0.12, f"still sample-count dependent: {aurocs}"

    # The remaining, documented, sample-count dependence.
    ranks = [rows[n]["effective_rank_clean"] for n in (200, 1000, 5000)]
    assert ranks == sorted(ranks), "effective rank must still rise with n"
    assert ranks[-1] > 2 * ranks[0]

    # And the ratio that cancels it.
    ratios = [rows[n]["rank_ratio"] for n in (200, 1000, 5000)]
    for ratio in ratios:
        assert ratio == pytest.approx(1.0, abs=0.15)


def test_effective_rank_is_far_below_the_true_rank_at_the_deployed_sample_count():
    """Participation ratio of a sample covariance, against Marchenko-Pastur.

    Isotropic noise fills all 768 directions, so the population answer is 768.
    The sample estimate is dim / (1 + dim / num_samples), which is 434 at the
    default sample count. The formula itself is right; the number is simply not
    the population quantity, and the gap is a function of the sample count.
    """
    generator = torch.Generator().manual_seed(107)
    features = torch.randn(DEPLOYED_SAMPLES, DEPLOYED_DIM, generator=generator)

    measured = effective_rank(features)
    predicted = DEPLOYED_DIM / (1 + DEPLOYED_DIM / DEPLOYED_SAMPLES)

    assert measured == pytest.approx(predicted, rel=0.02)
    assert measured < 0.6 * DEPLOYED_DIM


def test_rank_ratio_cancels_the_sample_count_bias_that_effective_rank_carries():
    """The ratio is the trustworthy form, because both halves share dim and n.

    This is the reason rank_ratio may be read across layers while the raw ranks
    beside it may not.
    """
    for num_samples in (500, 1000, 4000):
        clean, backdoor = paired_null(num_samples, DEPLOYED_DIM, seed=108)
        row = layer_distribution_row(clean, backdoor)

        assert row["effective_rank_clean"] < 0.9 * DEPLOYED_DIM
        assert row["rank_ratio"] == pytest.approx(1.0, abs=0.05)


def true_covariance_distances(
    reference: torch.Tensor, query: torch.Tensor
) -> torch.Tensor:
    """Mahalanobis with the covariance known exactly, for data built as identity.

    The reference sets below are standard normal, so the population covariance is
    the identity and the correct distance is the plain euclidean one from the
    reference mean. Comparing against this isolates the estimation error from the
    quantity being estimated.
    """
    distances = (query - reference.mean(dim=0, keepdim=True)).pow(2).sum(dim=1).sqrt()
    return distances


def test_estimated_covariance_inflates_the_mahalanobis_separation_at_768_dimensions():
    """The covariance is estimated from the very reference it then scores.

    The query is paired with the reference here, which is the real usage, so this
    is not a from-nothing artifact: there is a genuine separation to find. The
    estimation error exaggerates it. At 768 dimensions and 1000 samples a true
    separation of about 0.88 is reported as a saturated 1.00.
    """
    generator = torch.Generator().manual_seed(109)
    clean = torch.randn(DEPLOYED_SAMPLES, DEPLOYED_DIM, generator=generator)
    backdoor = clean + 0.3 * torch.randn(
        DEPLOYED_SAMPLES, DEPLOYED_DIM, generator=generator
    )

    reported = separation_auroc(
        mahalanobis_distances(clean, clean), mahalanobis_distances(clean, backdoor)
    )
    truth = separation_auroc(
        true_covariance_distances(clean, clean),
        true_covariance_distances(clean, backdoor),
    )

    assert truth < 0.92, "the correct answer is a strong but unsaturated separation"
    assert reported > 0.99, "the reported answer saturates"
    assert reported - truth > 0.08


def test_the_mahalanobis_inflation_is_driven_by_dimension_over_sample_count():
    """Confirms the cause, and that the toy test sizes cannot see it.

    The same trigger at 96 dimensions, and the same trigger at 768 dimensions
    with 5 times the samples, are both estimated almost correctly. Only the
    deployed combination is badly inflated.
    """

    def inflation(num_samples: int, dim: int) -> float:
        generator = torch.Generator().manual_seed(110)
        clean = torch.randn(num_samples, dim, generator=generator)
        backdoor = clean + 0.3 * torch.randn(num_samples, dim, generator=generator)
        reported = separation_auroc(
            mahalanobis_distances(clean, clean), mahalanobis_distances(clean, backdoor)
        )
        truth = separation_auroc(
            true_covariance_distances(clean, clean),
            true_covariance_distances(clean, backdoor),
        )
        return reported - truth

    deployed = inflation(DEPLOYED_SAMPLES, DEPLOYED_DIM)
    narrow = inflation(DEPLOYED_SAMPLES, 96)
    well_sampled = inflation(5 * DEPLOYED_SAMPLES, DEPLOYED_DIM)

    assert deployed > 0.08
    assert narrow < 0.05
    assert well_sampled < deployed / 2


def test_the_reported_clean_distance_is_depressed_because_it_is_in_sample():
    """plot_distance_distribution scores the clean set against its own covariance.

    The clean curve is therefore drawn low, which widens the visible gap to the
    backdoor curve beyond the real one. The gap is not invented, it is roughly
    doubled.
    """
    generator = torch.Generator().manual_seed(111)
    clean = torch.randn(DEPLOYED_SAMPLES, DEPLOYED_DIM, generator=generator)
    backdoor = clean + 0.3 * torch.randn(
        DEPLOYED_SAMPLES, DEPLOYED_DIM, generator=generator
    )

    reported_gap = (
        mahalanobis_distances(clean, backdoor).median()
        - mahalanobis_distances(clean, clean).median()
    ).item()
    true_gap = (
        true_covariance_distances(clean, backdoor).median()
        - true_covariance_distances(clean, clean).median()
    ).item()

    assert reported_gap > 1.8 * true_gap
    # The clean curve, not the backdoor curve, is the one drawn in the wrong place.
    assert mahalanobis_distances(clean, clean).median() < 0.95 * (
        true_covariance_distances(clean, clean).median()
    )


def test_an_unpaired_query_is_separated_completely_from_an_identical_population():
    """The limit the paired case approaches once a trigger decorrelates the point.

    Not the default usage, but it is what the statistic does the moment the
    backdoor representation stops tracking its clean counterpart, so a strong
    trigger is scored on a scale whose null is nowhere near 0.5.
    """
    generator = torch.Generator().manual_seed(112)
    reference = torch.randn(DEPLOYED_SAMPLES, DEPLOYED_DIM, generator=generator)
    unpaired = torch.randn(DEPLOYED_SAMPLES, DEPLOYED_DIM, generator=generator)

    in_sample = mahalanobis_distances(reference, reference)
    out_of_sample = mahalanobis_distances(reference, unpaired)

    assert separation_auroc(in_sample, out_of_sample) > 0.99
    assert out_of_sample.median() > 1.4 * in_sample.median()


def test_shrinkage_suppresses_a_shift_into_a_narrow_clean_direction():
    """The trigger writes off the clean manifold, and shrinkage discounts exactly that.

    A 3 sigma move is a 3 sigma move whichever direction it happens along. With
    the covariance pulled 10 percent toward a scaled identity, the same move is
    reported as larger than 3 along a wide clean direction and far smaller than 3
    along a narrow one, which is backwards for an off-manifold trigger.
    """
    dim = 256
    generator = torch.Generator().manual_seed(120)
    per_direction_scale = torch.logspace(0, -2, dim)
    reference = torch.randn(20000, dim, generator=generator) * per_direction_scale

    reported = {}
    for axis, name in ((0, "widest"), (dim - 1, "narrowest")):
        query = torch.zeros(1, dim)
        query[0, axis] = 3.0 * per_direction_scale[axis]
        reported[name] = mahalanobis_distances(reference, query).item()

    assert reported["widest"] > 3.0
    assert reported["narrowest"] < 1.5, (
        "a genuine 3 sigma off-manifold move is reported as under half its size"
    )

    # The attenuation is the shrinkage itself, not estimation noise: an eigenvalue
    # far below the mean is replaced by roughly COVARIANCE_SHRINKAGE times the mean.
    mean_eigenvalue = (per_direction_scale.double() ** 2).mean()
    narrow_eigenvalue = per_direction_scale[-1].double() ** 2
    shrunk = (
        1 - COVARIANCE_SHRINKAGE
    ) * narrow_eigenvalue + COVARIANCE_SHRINKAGE * mean_eigenvalue
    assert (narrow_eigenvalue / shrunk).sqrt().item() < 0.2


def test_swin_mean_reduction_is_the_tensor_the_trained_head_pools():
    """The Swin analogue of ViT's class token, checked against torchvision's head.

    SwinTransformer.forward is norm, permute to (batch, channels, height, width),
    AdaptiveAvgPool2d(1), flatten. Averaging the flattened spatial grid must give
    the identical tensor, or the Swin features are not what the classifier reads.
    """
    torch.manual_seed(0)
    swin = SwinTransformer(
        patch_size=[4, 4],
        embed_dim=8,
        depths=[2, 1],
        num_heads=[2, 2],
        window_size=[4, 4],
        num_classes=4,
    ).eval()

    with torch.no_grad():
        stack_output = swin.features(torch.rand(3, 3, 32, 32))
        normed = swin.norm(stack_output)
        head_input = swin.flatten(swin.avgpool(swin.permute(normed)))
        ours = _as_token_sequence(normed).mean(dim=1)

    assert torch.equal(ours, head_input)


def test_a_layer_index_names_the_same_block_to_a_probe_and_to_the_analysis():
    """The contract features.py claims in its module docstring, checked on both.

    A probe restricted to block_range (i + 1, i + 1) and feature layer i + 1 must
    refer to the same module object, or a placement result and a feature profile
    cannot be read against each other.
    """
    torch.manual_seed(0)
    builders = {
        "vit": lambda: VisionTransformer(
            image_size=32,
            patch_size=16,
            num_layers=3,
            num_heads=2,
            hidden_dim=16,
            mlp_dim=32,
            num_classes=4,
        ),
        "swin": lambda: SwinTransformer(
            patch_size=[4, 4],
            embed_dim=8,
            depths=[2, 1],
            num_heads=[2, 2],
            window_size=[4, 4],
            num_classes=4,
        ),
    }

    for architecture, build in builders.items():
        model = nn.Sequential(transforms_v2.Resize((32, 32)), build())
        core = network_core(model)
        blocks = transformer_blocks(core, architecture)
        spec = POSITION_REGISTRY[architecture]["before_attention_norm"]

        for index in range(len(blocks)):
            probed = resolve_targets(
                model,
                core,
                spec,
                BLOCK_TYPES[architecture],
                block_range=(index + 1, index + 1),
            )
            expected = (
                blocks[index]
                if spec.submodule_name == ""
                else blocks[index].get_submodule(spec.submodule_name)
            )
            assert len(probed) == 1
            assert probed[0] is expected, (
                f"{architecture} block {index + 1} disagrees between the probe "
                "registry and the feature extractor"
            )


def test_the_block_walk_agrees_whether_it_starts_at_the_wrapper_or_the_core():
    """resolve_targets walks the wrapper, transformer_blocks walks the core."""
    torch.manual_seed(0)
    model = nn.Sequential(
        transforms_v2.Resize((32, 32)),
        SwinTransformer(
            patch_size=[4, 4],
            embed_dim=8,
            depths=[2, 1],
            num_heads=[2, 2],
            window_size=[4, 4],
            num_classes=4,
        ),
    )

    from_core = transformer_blocks(network_core(model), "swin")
    from_wrapper = [
        module for module in model.modules() if isinstance(module, BLOCK_TYPES["swin"])
    ]

    assert len(from_core) == len(from_wrapper)
    assert all(a is b for a, b in zip(from_core, from_wrapper))


def test_swin_block_widths_follow_the_stage_layout_the_statistics_assume():
    """Pins that a Swin feature table really does change width mid-stack."""
    torch.manual_seed(0)
    model = nn.Sequential(
        transforms_v2.Resize((32, 32)),
        SwinTransformer(
            patch_size=[2, 2],
            embed_dim=8,
            depths=[2, 2, 1],
            num_heads=[2, 2, 2],
            window_size=[4, 4],
            num_classes=4,
        ),
    ).eval()

    generator = torch.Generator().manual_seed(1)
    loader = DataLoader(
        TensorDataset(
            torch.rand(4, 3, 32, 32, generator=generator),
            torch.zeros(4, dtype=torch.long),
        ),
        batch_size=4,
        shuffle=False,
    )
    features = extract_layer_features(
        model, loader, torch.device("cpu"), use_bfloat16=False
    )

    widths = [features[layer].shape[1] for layer in sorted(features) if layer > 0]
    assert widths == [8, 8, 16, 16, 32]


def test_extraction_removes_its_hooks_when_the_loader_itself_raises():
    """A leaked hook would silently contaminate every later extraction."""
    torch.manual_seed(0)
    network = VisionTransformer(
        image_size=32,
        patch_size=16,
        num_layers=2,
        num_heads=2,
        hidden_dim=16,
        mlp_dim=32,
        num_classes=4,
    )
    model = nn.Sequential(transforms_v2.Resize((32, 32)), network)

    class ExplodingLoader:
        def __iter__(self):
            yield torch.rand(2, 3, 32, 32), torch.zeros(2, dtype=torch.long)
            raise RuntimeError("loader exploded")

    with pytest.raises(RuntimeError, match="loader exploded"):
        extract_layer_features(
            model, ExplodingLoader(), torch.device("cpu"), use_bfloat16=False
        )

    for block in transformer_blocks(network, "vit"):
        assert not block._forward_hooks
        assert not block._forward_pre_hooks


def test_a_layer_with_fewer_than_two_samples_reports_absent_rather_than_raising():
    """One row defines no distribution, and must not lose every other layer's row.

    Constant layers were already guarded. A layer with a single sample raised out
    of shrunk_covariance instead, so a single tiny layer discarded the whole
    table.
    """
    single = torch.randn(1, 8)

    row = layer_distribution_row(single, single)
    for key in ("separation_auroc", "mahalanobis_median", "effective_rank_clean"):
        assert math.isnan(row[key]), f"{key} should be undefined on 1 sample"

    table = layer_distribution_table(
        {0: single, 1: torch.randn(60, 8)}, {0: single, 1: torch.randn(60, 8)}
    )
    assert len(table) == 2, "the tiny layer must not remove the usable one"
    assert not math.isnan(table[1]["separation_auroc"])


def test_the_statistics_are_cpu_only_although_extraction_is_device_parameterized():
    """Documents that a caller holding non-CPU features gets an error, not a number.

    extract_layer_features moves its output to the CPU, so the intended flow is
    safe. A caller who assembles features itself is not, and the failure is loud
    rather than silent, which is the behaviour worth pinning.
    """
    reference = torch.randn(20, 4, device="meta")
    query = torch.randn(5, 4, device="meta")

    with pytest.raises(RuntimeError):
        mahalanobis_distances(reference, query)


def test_bfloat16_rounding_leaves_the_mean_based_statistics_intact():
    """cases.py hardcodes use_bfloat16=True, so production features are rounded.

    The rounding costs a lot of accuracy on an individual paired difference, but
    it is close to zero mean, so the direction and everything derived from it
    survives. Pinned because the opposite conclusion would invalidate every
    number the module produces on a GPU.
    """
    generator = torch.Generator().manual_seed(121)
    clean = torch.randn(DEPLOYED_SAMPLES, DEPLOYED_DIM, generator=generator) * 4.0
    delta = 0.05 * torch.randn(DEPLOYED_SAMPLES, DEPLOYED_DIM, generator=generator)
    delta[:, :16] += 0.1
    backdoor = clean + delta

    rounded_clean = clean.to(torch.bfloat16).float()
    rounded_backdoor = backdoor.to(torch.bfloat16).float()

    exact = layer_distribution_row(clean, backdoor)
    rounded = layer_distribution_row(rounded_clean, rounded_backdoor)

    assert rounded["separation_auroc"] == pytest.approx(
        exact["separation_auroc"], abs=0.01
    )
    assert rounded["standardized_shift"] == pytest.approx(
        exact["standardized_shift"], rel=0.05
    )
    assert rounded["direction_norm"] == pytest.approx(exact["direction_norm"], rel=0.05)
