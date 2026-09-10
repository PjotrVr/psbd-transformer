"""TED's rank statistic against the notebook's own logic, then the rest of the port.

first_same_class_rank is the 1 piece of detectors.ted with a reference it must
equal: cell 14 of third_party/ted/TED.ipynb sorts the bank by distance, walks the
predicted labels and takes .index of the query's label. That logic is
reimplemented here in numpy and the port must agree to the integer, with and
without the notebook's ranking_array[1:] self exclusion. The outlier model has no
reference, since pyod is not installed and the port replaces its PCA detector, so
those tests pin direction and finiteness only.

The end-to-end tests run on the synthetic fixture and report the AUROC rather
than asserting it. The fixture's random-weight model spreads its predictions over
few classes that depend on the seed, so 2 seeds are read: the default, where the
target class never appears in the bank, plus 1 where it does.
"""

import numpy as np
import pytest
import torch

from defences.decision import HEADLINE_QUANTILE, detection_report
from detectors import ted
from experiments.preflight.synthetic import (
    NUM_CLASSES,
    TARGET_CLASS,
    build_backdoored_model,
    build_splits,
)

DEVICE = torch.device("cpu")

# The default fixture model, seed 0, never predicts TARGET_CLASS on a clean input,
# so its bank holds no target-class row and every triggered input is scored by
# the absent-class rule alone. Seed 14 predicts 3 classes on noise with about a
# quarter of its bank in the target class, which exercises the rank dynamics.
DEFAULT_SEED = 0
MECHANISM_SEED = 14

FIXTURE_SAMPLES = 1024


def notebook_ranks(queries, query_labels, bank, bank_labels, drop_nearest):
    """Cell 14 of TED.ipynb in numpy: sort by distance, walk the labels, .index.

    drop_nearest on is getDefenseRegion, which slices ranking_array[1:] because
    the bank row itself sorts first. Off is getLayerRegionDistance for outside
    queries. A label absent from the walked list gives None, the sample the
    notebook drops.
    """
    ranks = []
    for query, label in zip(queries, query_labels):
        distances = np.sqrt(((bank - query[None, :]) ** 2).sum(axis=1))  # (bank_size,)
        order = np.argsort(distances, kind="stable")
        if drop_nearest:
            order = order[1:]
        walked = [int(bank_labels[index]) for index in order]
        ranks.append(walked.index(int(label)) if int(label) in walked else None)
    return ranks


def notebook_ranks_with_absent_as_bank_size(*arguments):
    """The notebook's ranks with the port's convention for a dropped sample."""
    bank_size = arguments[2].shape[0]
    ranks = [bank_size if rank is None else rank for rank in notebook_ranks(*arguments)]
    return ranks


@pytest.fixture
def small_bank():
    generator = torch.Generator().manual_seed(0)
    bank = torch.randn(30, 5, generator=generator)  # (bank_size, features)
    bank_labels = torch.randint(0, 3, (30,), generator=generator)  # (bank_size,)
    queries = torch.randn(7, 5, generator=generator)  # (batch, features)
    query_labels = torch.randint(0, 3, (7,), generator=generator)  # (batch,)
    return bank, bank_labels, queries, query_labels


def test_outside_queries_match_the_notebook_to_the_integer(small_bank):
    bank, bank_labels, queries, query_labels = small_bank
    expected = notebook_ranks_with_absent_as_bank_size(
        queries.numpy(), query_labels.numpy(), bank.numpy(), bank_labels.numpy(), False
    )

    ranks = ted.first_same_class_rank(queries, query_labels, bank, bank_labels)

    assert ranks.dtype == torch.long and ranks.shape == (7,)
    assert ranks.tolist() == expected


def test_a_bank_row_ranks_itself_first_unless_excluded(small_bank):
    bank, bank_labels, _, _ = small_bank
    own_rows = bank[:7]  # (7, features), exact copies of bank rows
    own_labels = bank_labels[:7]
    expected_excluded = notebook_ranks_with_absent_as_bank_size(
        own_rows.numpy(), own_labels.numpy(), bank.numpy(), bank_labels.numpy(), True
    )

    included = ted.first_same_class_rank(own_rows, own_labels, bank, bank_labels)
    excluded = ted.first_same_class_rank(
        own_rows, own_labels, bank, bank_labels, exclude_self=torch.arange(7)
    )

    assert included.tolist() == [0] * 7, "a copy of a bank row is its own nearest"
    assert excluded.tolist() == expected_excluded
    assert excluded.tolist() != included.tolist(), (
        "excluding the row must move at least 1 rank off 0 under this seed"
    )


def test_a_singleton_class_excluded_from_itself_gets_the_bank_size():
    bank = torch.randn(10, 4, generator=torch.Generator().manual_seed(1))
    bank_labels = torch.tensor([0] * 9 + [1])
    lone_row = bank[9:10]  # (1, features), the only row of class 1

    excluded = ted.first_same_class_rank(
        lone_row, torch.tensor([1]), bank, bank_labels, exclude_self=torch.tensor([9])
    )
    included = ted.first_same_class_rank(lone_row, torch.tensor([1]), bank, bank_labels)

    assert excluded.item() == 10 and included.item() == 0


def test_a_label_absent_from_the_bank_gets_the_bank_size(small_bank):
    bank, bank_labels, queries, _ = small_bank
    absent_labels = torch.full((7,), 5)  # (batch,), no bank row carries 5

    ranks = ted.first_same_class_rank(queries, absent_labels, bank, bank_labels)

    assert ranks.tolist() == [30] * 7


def test_a_mixed_exclusion_vector_masks_only_the_named_rows(small_bank):
    bank, bank_labels, queries, query_labels = small_bank
    mixed_queries = torch.cat([bank[:3], queries[:4]])  # (7, features)
    mixed_labels = torch.cat([bank_labels[:3], query_labels[:4]])
    exclude_self = torch.tensor([0, 1, 2, -1, -1, -1, -1])
    expected = notebook_ranks_with_absent_as_bank_size(
        bank[:3].numpy(),
        bank_labels[:3].numpy(),
        bank.numpy(),
        bank_labels.numpy(),
        True,
    ) + notebook_ranks_with_absent_as_bank_size(
        queries[:4].numpy(),
        query_labels[:4].numpy(),
        bank.numpy(),
        bank_labels.numpy(),
        False,
    )

    ranks = ted.first_same_class_rank(
        mixed_queries, mixed_labels, bank, bank_labels, exclude_self=exclude_self
    )

    assert ranks.tolist() == expected


def test_a_bank_not_divisible_by_the_chunk_gives_the_same_ranks(small_bank):
    bank, bank_labels, queries, query_labels = small_bank
    expected = notebook_ranks_with_absent_as_bank_size(
        queries.numpy(), query_labels.numpy(), bank.numpy(), bank_labels.numpy(), False
    )

    chunked = ted.first_same_class_rank(
        queries, query_labels, bank, bank_labels, chunk_size=7
    )
    whole = ted.first_same_class_rank(
        queries, query_labels, bank, bank_labels, chunk_size=30
    )

    assert 30 % 7 != 0
    assert chunked.tolist() == expected == whole.tolist()


def test_a_float16_bank_is_ranked_in_float32(small_bank):
    bank, bank_labels, queries, query_labels = small_bank
    half_bank = bank.to(torch.float16)
    expected = notebook_ranks_with_absent_as_bank_size(
        queries.numpy(),
        query_labels.numpy(),
        half_bank.float().numpy(),
        bank_labels.numpy(),
        False,
    )

    ranks = ted.first_same_class_rank(queries, query_labels, half_bank, bank_labels)

    assert ranks.tolist() == expected


def test_a_feature_width_mismatch_is_refused(small_bank):
    bank, bank_labels, queries, query_labels = small_bank
    with pytest.raises(AssertionError):
        ted.first_same_class_rank(queries[:, :4], query_labels, bank, bank_labels)


def clustered_bank(generator, spacings, per_class=30, classes=3, dim=5):
    """A ReferenceBank whose classes are Gaussian clusters that tighten with depth.

    Layer l places class c at spacings[l] times the c-th unit vector with unit
    noise, so deeper layers separate the classes more, the evolution the paper
    describes for a clean sample.
    """
    labels = torch.arange(classes).repeat_interleave(per_class)  # (bank_size,)
    features = {
        layer: (
            torch.eye(classes, dim)[labels] * spacing
            + torch.randn(classes * per_class, dim, generator=generator)
        ).to(torch.float16)  # (bank_size, dim)
        for layer, spacing in enumerate(spacings)
    }
    bank = ted.ReferenceBank(
        features=features,
        predicted=labels,
        layers=tuple(range(len(spacings))),
        architecture="vit",
        source_indices=torch.arange(classes * per_class),
        source_size=classes * per_class,
    )
    return bank


def query_trajectories(bank, cluster, label, count, generator, spacings, dim=5):
    """(count, num_layers) ranks of queries drawn from 1 cluster under 1 label."""
    labels = torch.full((count,), label)
    ranks_by_layer = [
        ted.first_same_class_rank(
            torch.eye(3, dim)[cluster] * spacing
            + torch.randn(count, dim, generator=generator),
            labels,
            bank.features[layer],
            bank.predicted,
        )
        for layer, spacing in enumerate(spacings)
    ]
    trajectories = torch.stack(ranks_by_layer, dim=1).float()  # (count, num_layers)
    return trajectories


def test_a_query_from_the_wrong_cluster_scores_as_an_outlier():
    """Cluster A's features under cluster B's label is what a triggered input is."""
    generator = torch.Generator().manual_seed(0)
    spacings = (1.0, 2.0, 4.0)
    bank = clustered_bank(generator, spacings)
    model = ted.fit_trajectory_model(ted.leave_one_out_trajectories(bank))

    mislabelled = query_trajectories(bank, 0, 1, 40, generator, spacings)
    faithful = query_trajectories(bank, 1, 1, 40, generator, spacings)
    outlier = ted.outlier_scores(mislabelled, model)
    benign = ted.outlier_scores(faithful, model)

    assert outlier.mean() > benign.mean()
    assert (outlier > benign.median()).float().mean() >= 0.9


def test_a_3_sigma_trajectory_exceeds_the_benign_99th_percentile():
    generator = torch.Generator().manual_seed(0)
    mixing = torch.tensor(
        [
            [3.0, 1.0, 0.0, 0.0],
            [0.0, 2.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.5],
            [0.0, 0.0, 0.0, 4.0],
        ]
    )
    benign = torch.randn(4000, 4, generator=generator) @ mixing + 5.0  # (N, 4)
    model = ted.fit_trajectory_model(benign)

    benign_scores = ted.outlier_scores(benign, model)
    shifted = (model.mean + 3.0 * model.std).view(1, -1).float()  # (1, 4)
    shifted_score = ted.outlier_scores(shifted, model)

    assert shifted_score.item() > torch.quantile(benign_scores, 0.99).item()


def test_a_constant_layer_fits_and_scores_without_nan():
    generator = torch.Generator().manual_seed(0)
    benign = torch.randn(200, 3, generator=generator) * 2.0 + 3.0
    benign[:, 1] = 5.0  # a layer whose clean rank never varies

    model = ted.fit_trajectory_model(benign)
    benign_scores = ted.outlier_scores(benign, model)
    deviant = benign[:1].clone()
    deviant[0, 1] = 6.0
    deviant_score = ted.outlier_scores(deviant, model)

    assert torch.isfinite(model.precision).all() and torch.isfinite(model.std).all()
    assert model.std[1].item() == 1.0, "a constant layer keeps its raw rank units"
    assert torch.isfinite(benign_scores).all()
    assert torch.isfinite(deviant_score).all()
    assert deviant_score.item() > benign_scores.max().item()


def test_fewer_than_2_trajectories_are_refused():
    with pytest.raises(ValueError, match="at least 2"):
        ted.fit_trajectory_model(torch.zeros(1, 3))


def test_classes_without_reference_lists_the_absent_ones():
    bank = clustered_bank(torch.Generator().manual_seed(0), (1.0, 2.0))

    assert ted.classes_without_reference(bank, 5) == [3, 4]
    assert ted.classes_without_reference(bank, 3) == []


@pytest.fixture(scope="module")
def mechanism_case():
    model = build_backdoored_model(MECHANISM_SEED)
    loaders = build_splits(num_samples=FIXTURE_SAMPLES, batch_size=64)
    return model, loaders


def fixture_scores(model, loaders, reduction):
    """The registry's fit and score sequence on the fixture, by split."""
    bank = ted.collect_reference_bank(
        model, loaders["validation"], DEVICE, False, reduction
    )
    trajectory_model = ted.fit_trajectory_model(ted.leave_one_out_trajectories(bank))
    scores = {
        "validation": ted.ted_scores(
            model,
            loaders["validation"],
            DEVICE,
            bank,
            trajectory_model,
            reduction,
            False,
            loader_is_bank_source=True,
        ),
        "clean": ted.ted_scores(
            model, loaders["clean"], DEVICE, bank, trajectory_model, reduction, False
        ),
        "backdoor": ted.ted_scores(
            model, loaders["backdoor"], DEVICE, bank, trajectory_model, reduction, False
        ),
    }
    return bank, scores


def test_leave_one_out_differs_from_in_sample_and_matches_the_loader_path(
    mechanism_case,
):
    model, loaders = mechanism_case
    bank = ted.collect_reference_bank(
        model, loaders["validation"], DEVICE, False, "cls"
    )

    leave_one_out = ted.leave_one_out_trajectories(bank)  # (bank_size, 3)
    in_sample = torch.stack(
        [
            ted.first_same_class_rank(
                bank.features[layer],
                bank.predicted,
                bank.features[layer],
                bank.predicted,
            )
            for layer in bank.layers
        ],
        dim=1,
    ).float()
    trajectory_model = ted.fit_trajectory_model(leave_one_out)
    scores = ted.outlier_scores(leave_one_out, trajectory_model)

    via_loader, _ = ted.rank_trajectories(
        model,
        loaders["validation"],
        DEVICE,
        bank,
        "cls",
        False,
        loader_is_bank_source=True,
    )  # (FIXTURE_SAMPLES, 3)
    at_bank_rows = via_loader[bank.source_indices]  # (bank_size, 3)

    assert leave_one_out.shape == (bank.size, 3)
    assert torch.equal(in_sample, torch.zeros_like(in_sample)), (
        "in sample every bank row finds itself at rank 0"
    )
    assert not torch.equal(leave_one_out, in_sample)
    assert torch.isfinite(scores).all() and scores.shape == (bank.size,)
    assert scores.unique().numel() > 1
    # Queries and bank rows share the float16 rounding, so the loader path and the
    # bank path see identical numbers and must agree to the integer.
    assert torch.equal(at_bank_rows, leave_one_out)


def test_a_loader_of_another_length_is_refused_as_the_bank_source(mechanism_case):
    model, loaders = mechanism_case
    bank = ted.collect_reference_bank(
        model, loaders["validation"], DEVICE, False, "cls"
    )
    shorter = build_splits(num_samples=64, batch_size=64)["clean"]

    with pytest.raises(ValueError, match="source loader"):
        ted.rank_trajectories(
            model, shorter, DEVICE, bank, "cls", False, loader_is_bank_source=True
        )


@pytest.mark.parametrize("reduction", ["cls", "flatten"])
def test_end_to_end_on_the_fixture_where_the_target_class_is_in_the_bank(
    mechanism_case, reduction
):
    model, loaders = mechanism_case

    bank, scores = fixture_scores(model, loaders, reduction)
    _, again = fixture_scores(model, loaders, reduction)
    report = detection_report(
        scores["validation"], scores["clean"], scores["backdoor"], HEADLINE_QUANTILE
    )

    assert (bank.predicted == TARGET_CLASS).sum() > 0, (
        "the seed must put the target in the bank"
    )
    assert TARGET_CLASS not in ted.classes_without_reference(bank, NUM_CLASSES)
    for split, values in scores.items():
        assert values.shape == (FIXTURE_SAMPLES,), split
        assert values.dtype == torch.float32, split
        assert torch.isfinite(values).all(), split
        assert torch.equal(values, again[split]), split
    print(
        f"\nTED fixture seed {MECHANISM_SEED} reduction={reduction}: bank "
        f"{bank.size} rows, {int((bank.predicted == TARGET_CLASS).sum())} "
        f"of the target class, AUROC {report['auroc']:.3f}, TPR {report['tpr']:.3f} "
        f"at FPR {report['fpr']:.3f}, direction {report['direction']}"
    )


@pytest.mark.parametrize("reduction", ["cls", "flatten"])
def test_the_default_fixture_is_scored_by_the_absent_class_rule(reduction):
    """Seed 0 never predicts the target class on noise, so the bank has no row for
    it and every triggered input ranks at the bank size on every layer. The AUROC
    that results is real but tests the absent-class rule rather than the rank
    dynamics, which is why the fixture verdict is read from MECHANISM_SEED."""
    model = build_backdoored_model(DEFAULT_SEED)
    loaders = build_splits(num_samples=FIXTURE_SAMPLES, batch_size=64)

    bank, scores = fixture_scores(model, loaders, reduction)
    backdoor_trajectories, predicted = ted.rank_trajectories(
        model, loaders["backdoor"], DEVICE, bank, reduction, False
    )
    report = detection_report(
        scores["validation"], scores["clean"], scores["backdoor"], HEADLINE_QUANTILE
    )

    assert TARGET_CLASS in ted.classes_without_reference(bank, NUM_CLASSES)
    assert (predicted == TARGET_CLASS).all()
    assert torch.equal(
        backdoor_trajectories,
        torch.full_like(backdoor_trajectories, float(bank.size)),
    )
    assert all(torch.isfinite(values).all() for values in scores.values())
    print(
        f"\nTED fixture seed {DEFAULT_SEED} reduction={reduction}: bank "
        f"{bank.size} rows, 0 of the target class, AUROC "
        f"{report['auroc']:.3f} by the absent-class rule alone"
    )


def test_swin_shaped_activations_flow_through_as_token_sequence():
    grid = torch.randn(2, 3, 3, 5)  # (batch, height, width, channels)

    pooled = ted.reduce_activation(grid, "mean", has_class_token=False)
    flattened = ted.reduce_activation(grid, "flatten", has_class_token=False)
    tokens = ted.reduce_activation(torch.randn(2, 17, 32), "cls", has_class_token=True)

    expected_pooled = grid.reshape(2, 9, 5).mean(dim=1).to(torch.float16)  # (2, 5)
    assert pooled.dtype == ted.BANK_DTYPE
    assert torch.equal(pooled, expected_pooled)
    assert flattened.shape == (2, 45)
    assert tokens.shape == (2, 32)
    with pytest.raises(ValueError, match="class"):
        ted.reduce_activation(grid, "cls", has_class_token=False)
    with pytest.raises(ValueError, match="unknown"):
        ted.reduce_activation(grid, "max", has_class_token=False)


def test_a_feature_that_overflows_the_bank_dtype_is_refused():
    huge = torch.full((1, 4, 8), 1e5)  # (batch, tokens, dim), past float16's range

    with pytest.raises(ValueError, match="overflows"):
        ted.reduce_activation(huge, "flatten", has_class_token=True)
