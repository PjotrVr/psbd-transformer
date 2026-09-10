"""Beatrix's Gram algebra, band fitting and jackknife, pinned on CPU without a checkpoint.

The Gram features are checked by hand on a 2 by 2 token matrix, the band on a
class-structured bank whose right answer is known by construction, the overflow
guard at the order bound the port refuses, the jackknife against a fold refitted
by hand and the whole pipeline on the synthetic fixture for shape, finiteness
and determinism. The last test runs the released Feature_Correlations class on
the same tensors and requires agreement to 1e-5 after the single constant the 2
normalisations differ by.

The synthetic fixture's inner ViT is random-init on random-noise images, so its
class-conditional statistics have little to condition on and 9 of its 10
classes fall back to the pooled band. No test here asks Beatrix to separate on
it. What it reads there is recorded in docs/detectors/beatrix.md.
"""

import ast
import math
import os

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models.swin_transformer import SwinTransformer

from detectors import beatrix
from experiments.preflight.synthetic import build_backdoored_model, build_splits

DEVICE = torch.device("cpu")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFERENCE_FILE = os.path.join(
    REPO_ROOT, "third_party", "Beatrix", "defenses", "Beatrix", "Beatrix.py"
)


def class_structured_bank(
    per_class: tuple[int, ...],
    tokens_per_matrix: int = 64,
    dim: int = 6,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gaussian token matrices with a fixed mixing matrix per class, (N, tokens, dim) float16 and (N,) long.

    Each class has its own second-moment structure, so the Gram entries of its
    members concentrate around class-specific values and the band fitted on 1
    class excludes the members of another.
    """
    generator = torch.Generator().manual_seed(seed)

    matrices = []
    labels = []
    for class_index, count in enumerate(per_class):
        mixing = torch.eye(dim) + 0.8 * torch.randn(
            dim, dim, generator=generator
        )  # (dim, dim)
        noise = torch.randn(
            count, tokens_per_matrix, dim, generator=generator
        )  # (count, tokens, dim)
        matrices.append(noise @ mixing)  # (count, tokens, dim)
        labels.append(torch.full((count,), class_index, dtype=torch.long))

    tokens = torch.cat(matrices).to(torch.float16)  # (N, tokens, dim)
    predicted = torch.cat(labels)  # (N,)
    return tokens, predicted


def test_gram_features_by_hand_on_a_2_by_2_token_matrix():
    tokens = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])  # (1, 2, 2)

    features = beatrix.gram_features(tokens, (1, 2, 3, 4))  # (1, 12)

    # G^1 = v^T v over the 2 tokens is [[1 + 9, 2 + 12], [., 4 + 16]].
    assert features[0, :3].tolist() == [10.0, 14.0, 20.0]
    # G^2 squares first, [[1 + 81, 4 + 144], [., 16 + 256]], then takes the root.
    expected_second = [math.sqrt(82.0), math.sqrt(148.0), math.sqrt(272.0)]
    assert features[0, 3:6].tolist() == pytest.approx(expected_second, rel=1e-6)

    expected = []
    for power in (1, 2, 3, 4):
        powered = [
            [value**power for value in token] for token in ([1.0, 2.0], [3.0, 4.0])
        ]
        for i, j in ((0, 0), (0, 1), (1, 1)):
            entry = sum(row[i] * row[j] for row in powered)
            expected.append(math.copysign(abs(entry) ** (1.0 / power), entry))
    assert features.shape == (1, 12)
    assert features[0].tolist() == pytest.approx(expected, rel=1e-6)


def test_the_signed_root_keeps_the_sign_of_a_negative_entry():
    tokens = torch.tensor([[[-1.0, 2.0]]])  # (1, 1, 2), a single token

    features = beatrix.gram_features(tokens, (1, 2, 3))  # (1, 9)

    # With 1 token, G^p_ij = (v_i v_j)^p, so an odd order returns v_i v_j with
    # its sign and an even order returns |v_i v_j|.
    expected = [1.0, -2.0, 4.0, 1.0, 2.0, 4.0, 1.0, -2.0, 4.0]
    assert features[0].tolist() == pytest.approx(expected, rel=1e-5)


def test_gram_features_refuses_the_inputs_that_would_be_silently_wrong():
    with pytest.raises(ValueError, match="expected"):
        beatrix.gram_features(torch.zeros(4, 3))
    with pytest.raises(TypeError, match="float32 or float64"):
        beatrix.gram_features(torch.zeros(1, 4, 3, dtype=torch.bfloat16))


def test_a_query_from_another_cluster_deviates_more_than_a_member():
    tokens, predicted = class_structured_bank((30, 30, 30), tokens_per_matrix=200)
    bands = beatrix.fit_class_bands(tokens, predicted, 3, beatrix.PAPER_POWERS, DEVICE)

    fresh, _ = class_structured_bank((5, 5, 5), tokens_per_matrix=200, seed=1)
    against_class_1 = torch.ones(5, dtype=torch.long)  # (5,)
    class_0_queries = beatrix.deviations_of_tokens(
        fresh[:5], against_class_1, bands, DEVICE
    )  # (5,)
    class_1_queries = beatrix.deviations_of_tokens(
        fresh[5:10], against_class_1, bands, DEVICE
    )  # (5,)

    assert bands.counts.tolist() == [30, 30, 30]
    assert bands.pooled_lower is None and bands.pooled_class_count == 0
    assert class_0_queries.min() > class_1_queries.max()


def test_the_pooled_band_stands_in_for_a_class_under_the_minimum():
    tokens, predicted = class_structured_bank((30, 30, 30, 3))

    bands = beatrix.fit_class_bands(tokens, predicted, 4, beatrix.PAPER_POWERS, DEVICE)

    assert bands.counts.tolist() == [30, 30, 30, 3]
    assert bands.pooled_class_count == 1
    assert bands.pooled_lower is not None and bands.pooled_upper is not None
    assert torch.equal(bands.lower_by_class[3], bands.pooled_lower)
    assert torch.equal(bands.upper_by_class[3], bands.pooled_upper)
    assert not torch.equal(bands.lower_by_class[0], bands.pooled_lower)
    assert bands.lower_by_class.shape == (4, bands.num_features)


def test_the_pooled_band_equals_a_band_fitted_on_every_reference_at_once():
    """Column chunking must reproduce the unchunked fit to the bit.

    The Gram batch size is held at 5 on both sides, since a matmul over a
    different batch count may take a different kernel path and round
    differently, which is a property of BLAS rather than of the chunking.
    """
    tokens, _ = class_structured_bank((12,))

    unchunked = beatrix.fit_band_over_tokens(
        tokens, beatrix.PAPER_POWERS, DEVICE, batch_size=5
    )
    chunked = beatrix.fit_band_over_tokens(
        tokens, beatrix.PAPER_POWERS, DEVICE, batch_size=5, chunk_bytes=12 * 4 * 7
    )

    assert torch.equal(chunked[0], unchunked[0])
    assert torch.equal(chunked[1], unchunked[1])


def test_the_overflow_guard_raises_at_order_8_and_not_at_order_4():
    """A Gram entry is a token value to the power 2p, so 1e4 gives 1e32 at p = 4
    and 1e64 at p = 8 against a float32 ceiling of 3.4e38."""
    tokens = torch.full((2, 3, 4), 1e4)  # (batch, tokens, dim)

    finite = beatrix.gram_features(tokens, beatrix.PAPER_POWERS)
    assert torch.isfinite(finite).all()

    with pytest.raises(FloatingPointError, match="non-finite Gram entry"):
        beatrix.gram_features(tokens, beatrix.OFFICIAL_CODE_POWERS)


def test_the_jackknife_scores_every_reference_out_of_its_own_fold():
    tokens, predicted = class_structured_bank((30, 30, 30))

    jackknife = beatrix.jackknife_deviations(
        tokens, predicted, 3, beatrix.PAPER_POWERS, DEVICE
    )  # (90,)

    assert jackknife.shape == (90,) and jackknife.dtype == torch.float32
    assert torch.isfinite(jackknife).all()
    assert jackknife.max() > 0, "every score at 0 would make the comparison vacuous"

    # Fold 0 is the first 18 rows by position. Refitting on the other 72 and
    # scoring those 18 must reproduce the jackknife's values exactly.
    other_folds = beatrix.fit_class_bands(
        tokens[18:], predicted[18:], 3, beatrix.PAPER_POWERS, DEVICE
    )
    by_hand = beatrix.deviations_of_tokens(
        tokens[:18], predicted[:18], other_folds, DEVICE
    )
    assert torch.equal(jackknife[:18], by_hand)

    in_sample_bands = beatrix.fit_class_bands(
        tokens, predicted, 3, beatrix.PAPER_POWERS, DEVICE
    )
    in_sample = beatrix.deviations_of_tokens(tokens, predicted, in_sample_bands, DEVICE)
    assert not torch.equal(jackknife, in_sample)


def test_the_jackknife_refuses_fewer_references_than_folds():
    tokens, predicted = class_structured_bank((3,))
    with pytest.raises(ValueError, match="cannot be cut"):
        beatrix.jackknife_deviations(tokens, predicted, 1, beatrix.PAPER_POWERS, DEVICE)


@pytest.fixture(scope="module")
def synthetic_case():
    model = build_backdoored_model()
    loaders = build_splits(num_samples=128, batch_size=32)
    return model, loaders


def test_end_to_end_on_the_synthetic_fixture(synthetic_case):
    model, loaders = synthetic_case

    tokens, predicted = beatrix.collect_reference_tokens(
        model, loaders["validation"], DEVICE, layer=1, use_bfloat16=False
    )
    assert tokens.shape == (128, 17, 32) and tokens.dtype == torch.float16
    assert predicted.shape == (128,) and predicted.dtype == torch.long

    bands = beatrix.fit_class_bands(tokens, predicted, 10, beatrix.PAPER_POWERS, DEVICE)
    validation = beatrix.deviation_scores(
        beatrix.jackknife_deviations(
            tokens, predicted, 10, beatrix.PAPER_POWERS, DEVICE
        )
    )
    clean = beatrix.beatrix_scores(
        model, loaders["clean"], DEVICE, 1, bands, use_bfloat16=False
    )
    backdoor = beatrix.beatrix_scores(
        model, loaders["backdoor"], DEVICE, 1, bands, use_bfloat16=False
    )

    for scores in (validation, clean, backdoor):
        assert scores.shape == (128,) and scores.dtype == torch.float32
        assert torch.isfinite(scores).all()
        assert scores.max() <= 0, "a negated deviation is never positive"

    again = beatrix.beatrix_scores(
        model, loaders["clean"], DEVICE, 1, bands, use_bfloat16=False
    )
    assert torch.equal(clean, again)


def test_the_default_layer_follows_the_architecture(synthetic_case):
    model, _ = synthetic_case
    assert beatrix.default_feature_layer(model) == 9
    assert beatrix.default_feature_layer(model, "swin") == 22
    with pytest.raises(ValueError, match="no Beatrix feature layer"):
        beatrix.default_feature_layer(model, "resnet")


def test_a_swin_grid_capture_flattens_to_tokens():
    grid = torch.randn(2, 3, 3, 5)  # (batch, height, width, channels)

    tokens = beatrix.captured_token_matrix({22: grid}, 22)  # (2, 9, 5)

    assert tokens.shape == (2, 9, 5) and tokens.dtype == torch.float16
    assert torch.equal(tokens, grid.reshape(2, 9, 5).to(torch.float16))
    features = beatrix.gram_features(tokens.float())
    assert features.shape == (2, 4 * beatrix.gram_entry_count(5))


def test_collect_reference_tokens_reads_a_swin_stage_through_its_grid():
    torch.manual_seed(0)
    swin = SwinTransformer(
        patch_size=[4, 4],
        embed_dim=8,
        depths=[2, 1],
        num_heads=[2, 2],
        window_size=[4, 4],
        num_classes=4,
    ).eval()
    loader = DataLoader(
        TensorDataset(torch.rand(6, 3, 32, 32), torch.zeros(6, dtype=torch.long)),
        batch_size=4,
    )

    tokens, predicted = beatrix.collect_reference_tokens(
        swin, loader, DEVICE, layer=3, use_bfloat16=False
    )

    # Block 3 is the single block of stage 2, a 4 by 4 grid of 16 channels, so
    # the capture arrives as (batch, 4, 4, 16) and leaves as (batch, 16, 16).
    assert tokens.shape == (6, 16, 16) and predicted.shape == (6,)


def load_official_feature_correlations():
    """The released Feature_Correlations class, executed out of its module.

    Beatrix.py imports config, classifier_models and skimage at the top, none of
    which exist here, so the class source is cut out with ast and run in a
    namespace holding the 3 names its body uses.
    """
    if not os.path.exists(REFERENCE_FILE):
        pytest.skip("third_party/Beatrix is not checked out")

    source = open(REFERENCE_FILE).read()
    class_node = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ClassDef) and node.name == "Feature_Correlations"
    )
    namespace = {"torch": torch, "np": np, "F": F}
    exec(ast.get_source_segment(source, class_node), namespace)
    return namespace["Feature_Correlations"]


def test_matches_the_official_feature_correlations_to_1e_5():
    Feature_Correlations = load_official_feature_correlations()
    generator = torch.Generator().manual_seed(0)
    reference = torch.randn(
        40, 8, 5, 5, generator=generator
    )  # (N, channels, height, width)
    # Drawn off the reference distribution so the deviations are not all 0.
    queries = 1.5 * torch.randn(6, 8, 5, 5, generator=generator) + 0.5  # (6, 8, 5, 5)

    official = Feature_Correlations([1, 2, 3, 4], "mad")
    official.train([reference])
    theirs = torch.from_numpy(official.get_deviations_([queries])).squeeze(1)  # (6,)

    # Spatial positions are the tokens and channels the feature axis, so the
    # Gram contracts over the same axis in both implementations.
    reference_tokens = reference.flatten(2).transpose(1, 2)  # (40, 25, 8)
    query_tokens = queries.flatten(2).transpose(1, 2)  # (6, 25, 8)
    lower, upper = beatrix.fit_gram_band(beatrix.gram_features(reference_tokens))
    ours = beatrix.gram_deviation(
        beatrix.gram_features(query_tokens),
        lower,
        upper,
        beatrix.gram_entry_count(8),
        4,
    )  # (6,)

    # Eq. (14) divides by n(n + 1) / 2 = 36 entries per order. The released code
    # flattens the full triu'd 8 by 8 matrix, whose 28 zeroed entries add 0 to
    # the sum, then divides by n^2 / 2 = 32, so the 2 differ by 36 / 32 exactly.
    rescaled = ours * beatrix.gram_entry_count(8) / (8 * 8 / 2)
    assert ours.max() > 0, "all-zero deviations would make the agreement vacuous"
    assert official.num_feature == 32.0
    assert torch.allclose(rescaled, theirs.float(), atol=1e-5)
