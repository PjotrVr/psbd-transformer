"""The latent distribution tools, checked against cases whose answer is known.

Every statistic here is a number that will end up in a figure, so each one is
tested against data constructed to have a specific answer rather than against a
regression pin. A pin only says the code did not change; these say it is right.

The Swin cases exist because per-layer feature extraction was ViT-only, and its
token reduction silently returned an image row rather than a class token on
Swin's (batch, height, width, channels) activations. That is the failure mode
worth a permanent test: a wrong number with a plausible shape.
"""

import math

import matplotlib
import pytest
import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms_v2
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import VisionTransformer
from torchvision.models.swin_transformer import SwinTransformer

from psbd.analysis.distribution import (
    local_intrinsic_dimensionality,
    has_spread,
    class_centroids,
    effective_rank,
    layer_distribution_row,
    layer_distribution_table,
    mahalanobis_distances,
    plot_distance_distribution,
    plot_embedding_scatter,
    plot_layer_profile,
    plot_projection_histogram,
    separation_auroc,
    shrunk_covariance,
    standardized_mean_shift,
    target_class_alignment,
)
from psbd.analysis.features import (
    default_reduction,
    detect_model_architecture,
    extract_layer_features,
    transformer_blocks,
)

# Figures are built in tests on nodes with no display.
matplotlib.use("Agg")

SYNTHETIC_CLASSES = 4


@pytest.fixture(scope="module")
def tiny_vit() -> nn.Module:
    """A ViT with the real block structure at tiny width, 2 blocks."""
    torch.manual_seed(0)
    network = VisionTransformer(
        image_size=32,
        patch_size=16,
        num_layers=2,
        num_heads=2,
        hidden_dim=16,
        mlp_dim=32,
        num_classes=SYNTHETIC_CLASSES,
    )
    return nn.Sequential(transforms_v2.Resize((32, 32)), network)


@pytest.fixture(scope="module")
def tiny_swin() -> nn.Module:
    """A Swin with the real 2-stage structure at tiny width, 3 blocks total.

    Widths double at the stage boundary exactly as Swin-S does, so the test sees
    the per-layer dimension change that a ViT never produces.
    """
    torch.manual_seed(0)
    network = SwinTransformer(
        patch_size=[4, 4],
        embed_dim=8,
        depths=[2, 1],
        num_heads=[2, 2],
        window_size=[4, 4],
        num_classes=SYNTHETIC_CLASSES,
    )
    return nn.Sequential(transforms_v2.Resize((32, 32)), network)


@pytest.fixture(scope="module")
def image_loader() -> DataLoader:
    generator = torch.Generator().manual_seed(1)
    images = torch.rand(8, 3, 32, 32, generator=generator)
    labels = torch.zeros(8, dtype=torch.long)
    return DataLoader(TensorDataset(images, labels), batch_size=4, shuffle=False)


def test_architecture_is_detected_from_the_blocks_a_model_contains(tiny_vit, tiny_swin):
    assert detect_model_architecture(tiny_vit[1]) == "vit"
    assert detect_model_architecture(tiny_swin[1]) == "swin"


def test_block_counts_match_the_configured_depths(tiny_vit, tiny_swin):
    assert len(transformer_blocks(tiny_vit[1], "vit")) == 2
    assert len(transformer_blocks(tiny_swin[1], "swin")) == 3


def test_default_reduction_follows_what_the_head_reads():
    assert default_reduction("vit") == "cls"
    assert default_reduction("swin") == "mean"


def test_swin_features_are_extracted_at_every_block(tiny_swin, image_loader):
    """The extraction that used to raise on Swin now returns one row per block."""
    features = extract_layer_features(
        tiny_swin, image_loader, torch.device("cpu"), use_bfloat16=False
    )

    assert sorted(features) == [0, 1, 2, 3]
    for layer, tensor in features.items():
        assert tensor.shape[0] == 8, f"layer {layer} lost samples"
        assert tensor.dim() == 2, f"layer {layer} was not reduced to (samples, dim)"

    # Stage 2 doubles the width, so the last block must be wider than the first.
    assert features[3].shape[1] == 2 * features[1].shape[1]


def test_class_token_reduction_is_refused_on_swin(tiny_swin, image_loader):
    """Swin has no class token, so 'cls' must raise instead of taking an image row."""
    with pytest.raises(ValueError, match="classification token"):
        extract_layer_features(
            tiny_swin,
            image_loader,
            torch.device("cpu"),
            use_bfloat16=False,
            reduction="cls",
        )


def test_vit_layer_zero_is_the_first_block_input(tiny_vit, image_loader):
    """Layer 0 must be the tensor entering block 1, whichever module it is read from."""
    captured = []
    handle = tiny_vit[1].encoder.layers.register_forward_pre_hook(
        lambda _module, inputs: captured.append(inputs[0][:, 0, :].detach().float())
    )
    with torch.inference_mode():
        for images, _ in image_loader:
            tiny_vit(images)
    handle.remove()

    features = extract_layer_features(
        tiny_vit, image_loader, torch.device("cpu"), use_bfloat16=False
    )
    assert torch.equal(torch.cat(captured), features[0])


def test_extraction_leaves_no_hooks_behind(tiny_swin, image_loader):
    extract_layer_features(
        tiny_swin, image_loader, torch.device("cpu"), use_bfloat16=False
    )

    for block in transformer_blocks(tiny_swin[1], "swin"):
        assert not block._forward_hooks
        assert not block._forward_pre_hooks


def test_identical_populations_are_not_separable():
    generator = torch.Generator().manual_seed(2)
    scores = torch.randn(500, generator=generator)

    assert separation_auroc(scores, scores) == pytest.approx(0.5, abs=0.01)


def test_disjoint_populations_are_perfectly_separable():
    clean = torch.zeros(100)
    backdoor = torch.ones(100)

    assert separation_auroc(clean, backdoor) == pytest.approx(1.0)


def test_separation_auroc_is_reported_unflipped():
    """A score that separates the wrong way reads below 0.5 rather than being fixed."""
    clean = torch.ones(100)
    backdoor = torch.zeros(100)

    assert separation_auroc(clean, backdoor) == pytest.approx(0.0)


def test_standardized_shift_recovers_a_known_cohens_d():
    """Two unit-variance populations 2 apart have d = 2 along their mean difference."""
    generator = torch.Generator().manual_seed(3)
    clean = torch.randn(4000, 6, generator=generator)
    backdoor = torch.randn(4000, 6, generator=generator)
    backdoor[:, 0] += 2.0

    assert standardized_mean_shift(clean, backdoor) == pytest.approx(2.0, abs=0.1)


def test_effective_rank_of_a_single_direction_is_one():
    generator = torch.Generator().manual_seed(4)
    direction = torch.randn(1, 12, generator=generator)
    amplitudes = torch.randn(300, 1, generator=generator)

    assert effective_rank(amplitudes @ direction) == pytest.approx(1.0, abs=0.01)


def test_effective_rank_of_isotropic_noise_is_the_dimension():
    generator = torch.Generator().manual_seed(5)
    dimension = 8
    features = torch.randn(20000, dimension, generator=generator)

    assert effective_rank(features) == pytest.approx(dimension, rel=0.05)


def test_shrinkage_leaves_an_identity_covariance_unchanged():
    """Shrinkage pulls toward a scaled identity, so identity input is a fixed point."""
    generator = torch.Generator().manual_seed(6)
    features = torch.randn(5000, 4, generator=generator)
    covariance = shrunk_covariance(features)

    assert torch.allclose(covariance, torch.eye(4, dtype=covariance.dtype), atol=0.1)


def test_mahalanobis_matches_euclidean_on_whitened_data():
    """With identity covariance the metric reduces to distance from the mean."""
    generator = torch.Generator().manual_seed(7)
    reference = torch.randn(20000, 3, generator=generator)
    query = torch.tensor([[3.0, 0.0, 0.0], [0.0, 4.0, 0.0]])

    distances = mahalanobis_distances(reference, query)
    assert distances[0] == pytest.approx(3.0, abs=0.1)
    assert distances[1] == pytest.approx(4.0, abs=0.1)


def test_mahalanobis_discounts_a_high_variance_direction():
    """A shift along a wide direction is less unusual than the same shift along a narrow one."""
    generator = torch.Generator().manual_seed(8)
    reference = torch.randn(20000, 2, generator=generator)
    reference[:, 0] *= 10.0
    query = torch.tensor([[5.0, 0.0], [0.0, 5.0]])

    distances = mahalanobis_distances(reference, query)
    assert distances[0] < distances[1]


def test_class_centroids_average_within_each_class():
    features = torch.tensor([[0.0, 0.0], [2.0, 2.0], [10.0, 10.0]])
    labels = torch.tensor([0, 0, 1])

    centroids = class_centroids(features, labels)
    assert torch.equal(centroids[0], torch.tensor([1.0, 1.0]))
    assert torch.equal(centroids[1], torch.tensor([10.0, 10.0]))


def test_alignment_is_one_when_the_trigger_shifts_toward_the_target_class():
    clean = torch.tensor([[0.0, 0.0], [0.0, 0.0], [4.0, 0.0], [4.0, 0.0]])
    labels = torch.tensor([1, 1, 0, 0])
    # The clean mean is (2, 0) and the target centroid is (4, 0), so the target
    # direction is +x. Shifting the whole population along +x aligns exactly.
    backdoor = clean + torch.tensor([5.0, 0.0])

    alignment = target_class_alignment(clean, backdoor, labels, target_label=0)
    assert alignment == pytest.approx(1.0, abs=1e-5)


def test_alignment_is_zero_when_the_trigger_shifts_orthogonally():
    clean = torch.tensor([[0.0, 0.0], [0.0, 0.0], [4.0, 0.0], [4.0, 0.0]])
    labels = torch.tensor([1, 1, 0, 0])
    backdoor = clean + torch.tensor([0.0, 5.0])

    alignment = target_class_alignment(clean, backdoor, labels, target_label=0)
    assert alignment == pytest.approx(0.0, abs=1e-5)


def test_alignment_is_undefined_when_the_target_class_is_absent():
    clean = torch.randn(10, 3)
    labels = torch.ones(10, dtype=torch.long)

    assert torch.isnan(
        torch.tensor(target_class_alignment(clean, clean, labels, target_label=7))
    )


def test_a_layer_row_carries_every_statistic():
    generator = torch.Generator().manual_seed(9)
    clean = torch.randn(200, 5, generator=generator)
    backdoor = clean + torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])

    row = layer_distribution_row(clean, backdoor)
    for key in (
        "direction_norm",
        "separation_auroc",
        "standardized_shift",
        "cka",
        "effective_rank_clean",
        "effective_rank_backdoor",
        "mahalanobis_median",
    ):
        assert key in row, f"missing {key}"
        assert not torch.isnan(torch.tensor(row[key])), f"{key} is nan"

    # The target statistic appears only when labels and a target are supplied.
    assert "target_alignment" not in row


def test_a_shifted_population_separates_and_an_unshifted_one_does_not():
    generator = torch.Generator().manual_seed(10)
    clean = torch.randn(400, 6, generator=generator)
    unshifted = torch.randn(400, 6, generator=generator)
    shifted = clean + torch.tensor([6.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    assert layer_distribution_row(clean, unshifted)[
        "separation_auroc"
    ] == pytest.approx(0.5, abs=0.05)
    assert layer_distribution_row(clean, shifted)["separation_auroc"] > 0.99


def test_the_table_has_one_row_per_shared_layer():
    generator = torch.Generator().manual_seed(11)
    clean = {layer: torch.randn(50, 4, generator=generator) for layer in (0, 1, 2)}
    backdoor = {layer: torch.randn(50, 4, generator=generator) for layer in (1, 2, 3)}

    table = layer_distribution_table(clean, backdoor)
    assert [row["layer"] for row in table] == [1, 2]


def test_the_table_carries_target_alignment_when_labels_are_given():
    generator = torch.Generator().manual_seed(12)
    clean = {0: torch.randn(60, 4, generator=generator)}
    backdoor = {0: torch.randn(60, 4, generator=generator)}
    labels = torch.randint(0, 3, (60,), generator=generator)

    table = layer_distribution_table(clean, backdoor, labels, target_label=1)
    assert "target_alignment" in table[0]


@pytest.mark.parametrize("method", ["pca"])
def test_plot_helpers_return_a_figure(method):
    generator = torch.Generator().manual_seed(13)
    clean = torch.randn(80, 5, generator=generator)
    backdoor = clean + 1.0
    labels = torch.randint(0, 3, (80,), generator=generator)
    table = layer_distribution_table({0: clean, 1: clean}, {0: backdoor, 1: backdoor})

    figures = [
        plot_layer_profile(table),
        plot_projection_histogram(clean, backdoor),
        plot_distance_distribution(clean, backdoor),
        plot_embedding_scatter(clean, backdoor, method=method),
        plot_embedding_scatter(clean, backdoor, method=method, clean_labels=labels),
    ]
    for figure in figures:
        assert isinstance(figure, matplotlib.figure.Figure)
        matplotlib.pyplot.close(figure)


def test_an_unknown_projection_method_is_refused():
    features = torch.randn(20, 4)
    with pytest.raises(ValueError, match="Unknown projection method"):
        plot_embedding_scatter(features, features, method="tsne")


def test_mahalanobis_is_undefined_against_a_reference_with_no_spread():
    """A constant reference population is a point, so the distance has no meaning.

    This is not hypothetical. A ViT's layer 0 under the cls reduction is the class
    token before any block has run, a learned constant identical for every image,
    so the first row of every real distribution table hits this case.
    """
    reference = torch.ones(50, 4)
    query = torch.randn(7, 4)

    distances = mahalanobis_distances(reference, query)
    assert distances.shape == (7,)
    assert torch.isnan(distances).all()


def test_a_constant_layer_produces_a_row_rather_than_an_exception():
    """A degenerate layer must still yield a row, so one bad layer cannot lose a table."""
    constant = torch.ones(40, 4)
    varied = torch.randn(40, 4)

    row = layer_distribution_row(constant, varied)
    assert set(row) >= {"direction_norm", "separation_auroc", "mahalanobis_median"}


def test_rank_ratio_is_one_when_neither_population_collapses():
    generator = torch.Generator().manual_seed(14)
    clean = torch.randn(4000, 6, generator=generator)
    backdoor = torch.randn(4000, 6, generator=generator)

    assert layer_distribution_row(clean, backdoor)["rank_ratio"] == pytest.approx(
        1.0, abs=0.1
    )


def test_rank_ratio_falls_when_the_trigger_collapses_the_representation():
    """A backdoor confined to one direction occupies far fewer than the clean data."""
    generator = torch.Generator().manual_seed(15)
    clean = torch.randn(2000, 8, generator=generator)
    direction = torch.randn(1, 8, generator=generator)
    backdoor = torch.randn(2000, 1, generator=generator) @ direction

    assert layer_distribution_row(clean, backdoor)["rank_ratio"] < 0.2


def test_has_spread_distinguishes_a_constant_population_from_a_varying_one():
    assert not has_spread(torch.ones(20, 5))
    assert has_spread(torch.randn(20, 5))


def test_a_degenerate_layer_reports_similarity_as_absent_not_as_collapse():
    """A constant layer must not read as cka 0, which aggregates as a total collapse.

    ViT's layer 0 under the cls reduction is exactly this case, so without the
    guard every ViT summary reports a minimum cka of 0 whether or not the model is
    backdoored, and the benign control looks identical to a backdoor.
    """
    constant = torch.ones(40, 6)
    row = layer_distribution_row(
        constant, constant, torch.zeros(40, dtype=torch.long), 0
    )

    assert math.isnan(row["cka"]), "constant features must not report a real cka"
    assert math.isnan(row["target_alignment"])
    assert math.isnan(row["rank_ratio"])
    # The paired difference is genuinely 0 here, so it is reported as 0.
    assert row["direction_norm"] == pytest.approx(0.0)


def test_a_degenerate_layer_does_not_poison_a_minimum_over_layers():
    """The reason the guard exists: pandas-style min over layers must skip it."""
    generator = torch.Generator().manual_seed(16)
    clean = {0: torch.ones(60, 6), 1: torch.randn(60, 6, generator=generator)}
    backdoor = {0: torch.ones(60, 6), 1: torch.randn(60, 6, generator=generator)}

    table = layer_distribution_table(clean, backdoor)
    real_cka = [row["cka"] for row in table if not math.isnan(row["cka"])]
    assert len(real_cka) == 1, "only the varying layer should report a cka"


def test_lid_recovers_a_known_subspace_dimension():
    """A population on a 3-dimensional subspace inside 64 dims must read near 3.

    The point of the estimator is that it reports the dimension of the manifold
    the data lies on rather than the dimension of the space it is embedded in, so
    a wrong implementation that reads the ambient dimension fails here loudly.
    """
    generator = torch.Generator().manual_seed(0)
    basis = torch.randn(3, 64, generator=generator)
    population = torch.randn(4000, 3, generator=generator) @ basis

    estimate = local_intrinsic_dimensionality(population, population).median().item()
    assert estimate == pytest.approx(3.0, abs=1.0), f"read {estimate}, expected near 3"


def test_lid_rises_with_the_true_dimension():
    """The estimator is biased upward at finite k, so order is what is asserted."""
    generator = torch.Generator().manual_seed(1)
    estimates = [
        local_intrinsic_dimensionality(
            torch.randn(4000, dim, generator=generator),
            torch.randn(4000, dim, generator=generator),
        )
        .median()
        .item()
        for dim in (2, 5, 8)
    ]

    assert estimates == sorted(estimates)
    assert estimates[-1] > estimates[0] + 3.0


def test_lid_excludes_a_query_from_being_its_own_neighbour():
    """A point matches itself at distance 0, which would make the log diverge."""
    generator = torch.Generator().manual_seed(2)
    population = torch.randn(500, 6, generator=generator)

    estimate = local_intrinsic_dimensionality(population, population)
    assert torch.isfinite(estimate).all(), "self-match was not excluded"
    assert (estimate > 0).all(), "LID must be positive"


def test_lid_needs_more_reference_samples_than_neighbours():
    with pytest.raises(ValueError, match="reference samples"):
        local_intrinsic_dimensionality(torch.randn(5, 4), torch.randn(10, 4))


def test_a_layer_row_carries_lid_for_both_populations():
    """LID is reported alongside the rank ratio because they can disagree in sign."""
    generator = torch.Generator().manual_seed(3)
    clean = torch.randn(400, 6, generator=generator)
    backdoor = clean + torch.randn(1, 6, generator=generator) * 0.5

    row = layer_distribution_row(clean, backdoor)
    for key in ("lid_clean", "lid_backdoor", "lid_ratio"):
        assert key in row
        assert not math.isnan(row[key]), f"{key} is nan on a healthy layer"
