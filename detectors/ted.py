"""TED: topological evolution dynamics (Mo et al., IEEE S&P 2024).

Paper: "Robust Backdoor Detection for Deep Learning via Topological Evolution
Dynamics", arXiv:2312.02673. The statistic is Section V-B and Algorithm 2, which
carries no numbered equation of its own, so its lines are cited instead.

    original form, Algorithm 2
        lines 2 to 7      S_1 = ... = S_c = {}
                          stack m samples of X_i in S_i, i = 1..c
                          [h_l(x)]_{l=1}^{N} = forward x through f
        lines 8 to 14     for i = 1..|S|:
                              j = argmax_{k in [1, c]} f(x_i)_k
                              for l = 1..N:
                                  S_sorted = sort_by_distance(d, h_l(.), S, x_i)
                                  x_nn     = get_nearest_neighbor(d, h_l(.), S_j - x_i, x_i)
                                  K_l^(i)  = get_rank(S_sorted, x_nn)
                              record [K_l^(i)]_{l=1}^{N}
        lines 15 to 16    M   = PCA({ [K_l^(i)]_{l=1}^{N} }_{i=1}^{|S|}, alpha)
                          tau = M.get_detect_threshold(alpha)
        lines 17 to 21    x in X_test is malicious if M(x) > tau

    descriptive form
        predicted_class  = the label the model gives the query
        rank[layer]      = position of the nearest bank row carrying
                           predicted_class, in the bank sorted by euclidean
                           distance to the query at that layer, the query's own
                           row excluded when the query is a bank row
        trajectory       = rank over every considered layer
        ted_score(query) = outlier score of trajectory under a model fitted on
                           the bank's own trajectories

S is the bank of stored clean samples and S_j its members the model predicts as
class j, X_i the clean samples of class i, m the stored samples per class, h_l
the representation at layer l, f the classifier, d the euclidean distance, N the
number of considered layers, c the number of classes and alpha the reject rate
that fixes tau. The threshold is replaced by the shared quantile rule of
defences.decision, so alpha plays no part here.

Mechanism. A clean input sits among clean inputs of its predicted class at every
depth, so the nearest bank row of that class is near the front of the sorted bank
at every layer and the ranks stay small and consistent. A poisoned input reaches
the target class only through its trigger, so in the early and middle layers its
nearest neighbours belong to its source class and the first target-class row is
ranked far back. The sequence of ranks over depth is the feature, and a poisoned
input's sequence is an outlier among the clean sequences. Ranks are invariant to
any positive rescaling of a layer's activations, so the scale drift across a
residual stream never enters the statistic.

Data requirement: the clean validation split, labelled, since the bank keeps only
the samples the model classifies correctly. Forward-pass cost: 1 per input, plus
1 euclidean distance matrix against the bank at each of the N layers, and a fixed
cost of 2 passes over the validation split, 1 for the bank and 1 for its
leave-one-out scores.

Deviations from the paper, each recorded in full in docs/detectors/ted.md:

  1. The considered layers are the residual stream at every block boundary, 13
     on ViT-B/16 and 25 on Swin-S, in place of every Conv2d, ReLU and Linear
     output.
  2. The outlier model is a squared Mahalanobis distance on standardised
     trajectories, replacing pyod's PCA detector, which is not installed. The
     direction of the score is preserved and no equivalence is claimed.
  3. The bank is every correctly classified sample of the shared 2000-sample
     split, without the notebook's cap of 1000.
  4. A query whose predicted class has no bank row gets rank equal to the bank
     size at every layer, where the notebook drops the sample.
  5. Features are rounded to float16 for the bank and for every query, and
     distances are taken in float32.
  6. Bank rows are ranked with their own row excluded, the notebook's
     ranking_array[1:], and the validation scores are those leave-one-out values.
  7. Tokens are flattened by default, as the notebook flattens every activation,
     with cls and mean as recorded alternatives.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from analysis.features import (
    as_token_sequence,
    captured_layers,
    detect_model_architecture,
    transformer_blocks,
)
from defences.inference import forward_probs
from models.backbones import network_core

# The notebook's fetch_activation flattens every hooked activation with
# view(batch, -1), so every token of every layer takes part in the distance.
DEFAULT_REDUCTION = "flatten"

# DEFENSE_TRAIN_SIZE in the notebook, the cap it draws at random from the
# correctly classified samples. The paper states 20 per class on CIFAR-10 and
# MNIST, 1000 in total on GTSRB and 200 per label on PubFig and ImageNet. Kept as
# a record only, the bank here is uncapped.
PAPER_DEFENSE_SET_SIZE = 1000

# Storage dtype of the bank and of every query feature. A flattened ViT-B/16
# layer is 197 tokens by 768 channels, 151296 values per sample, so 13 layers of
# about 1900 references are 7.5 GB in float16 and would be 15 GB in float32.
# Distances are still taken in float32, chunk by chunk, so the rounding touches
# the stored values only.
BANK_DTYPE = torch.float16

# Added to the diagonal of the trajectory covariance before the pseudo-inverse.
# Ranks are integers and a layer where the clean bank is nearly constant makes
# the covariance singular, which the floor turns into a large, finite penalty.
COVARIANCE_FLOOR = 1e-6

# A layer whose clean ranks spread less than this keeps its raw rank units, as
# sklearn's StandardScaler inside pyod's PCA does for a zero-variance feature.
STANDARD_DEVIATION_FLOOR = 1e-6

# Rows cast to float32 per cdist call, on the bank side and on the query side.
# 256 rows of flattened ViT-B/16 features are 155 MB in float32, so the 2 cast
# chunks and the (rows, bank_size) distance matrix are the largest transients of
# a scoring pass, where casting the whole bank at once would be 1.15 GB a layer.
DISTANCE_CHUNK = 256


@dataclass(frozen=True, eq=False)
class ReferenceBank:
    """S of Algorithm 2 with its features at every considered layer.

    features[layer] is (bank_size, features_at_layer) in BANK_DTYPE on the
    scoring device, predicted is (bank_size,) long, the label the model gave each
    row, which equals its loader label since only correctly classified samples
    are kept. source_indices is (bank_size,) long, the position of each row in
    the loader it was collected from. source_size is that loader's length, so a
    later pass over the same loader can exclude a sample's own row.

    On ViT-B/16 under the flatten reduction the bank is about 7.5 GB and lives on
    the device for the whole scoring run. Dropping every reference to the bank
    frees it, which is when a caller should let it go.
    """

    features: dict[int, torch.Tensor]
    predicted: torch.Tensor
    layers: tuple[int, ...]
    architecture: str
    source_indices: torch.Tensor
    source_size: int

    def __post_init__(self) -> None:
        bank_size = self.predicted.numel()
        assert self.predicted.shape == (bank_size,)
        assert self.source_indices.shape == (bank_size,)
        assert tuple(self.features) == self.layers
        for layer in self.layers:
            assert self.features[layer].shape[0] == bank_size, layer

    @property
    def size(self) -> int:
        """|S|, the number of stored references."""
        count = self.predicted.numel()
        return count


@dataclass(frozen=True)
class TrajectoryModel:
    """The clean trajectory distribution an outlier score is measured against.

    mean and std are (num_layers,) float64, precision is (num_layers,
    num_layers) float64, the floored pseudo-inverse of the covariance of the
    standardised clean trajectories.
    """

    mean: torch.Tensor
    std: torch.Tensor
    precision: torch.Tensor

    def __post_init__(self) -> None:
        num_layers = self.mean.numel()
        assert self.std.shape == (num_layers,)
        assert self.precision.shape == (num_layers, num_layers)


def reduce_activation(
    activation: torch.Tensor, reduction: str, has_class_token: bool
) -> torch.Tensor:
    """A raw block activation as (batch, features) in BANK_DTYPE under 1 token reduction.

    The same 3 reductions and the same class-token guard as
    analysis.features._reduce_tokens, repeated here because that helper is
    private to its module. Swin's (batch, height, width, channels) grid is
    flattened to a token axis first, so "flatten" and "mean" read both
    architectures alike and "cls" refuses the architecture that has no class
    token rather than returning its first grid row.

    Bank rows and queries both come through here, so a bank member scored through
    its own loader carries the same rounded values as its stored row, given a
    reproducible forward, and the self-exclusion mask removes exactly its own
    distance. A feature that overflows BANK_DTYPE would make every distance
    against it infinite, so the cast is checked rather than trusted.
    """
    # The cast precedes the reduction because a mean over Swin's 3136 first-stage
    # tokens in bfloat16 loses about 2 decimal digits against float32.
    tokens = as_token_sequence(activation).float()  # (batch, tokens, dim)

    if reduction == "cls":
        if not has_class_token:
            raise ValueError(
                "reduction 'cls' needs a classification token and this "
                "architecture has none, so use 'mean' or 'flatten'"
            )
        reduced = tokens[:, 0, :]  # (batch, dim)
    elif reduction == "mean":
        reduced = tokens.mean(dim=1)  # (batch, dim)
    elif reduction == "flatten":
        reduced = tokens.flatten(1)  # (batch, tokens * dim)
    else:
        raise ValueError(f"unknown token reduction {reduction!r}")

    rounded = reduced.to(BANK_DTYPE)  # (batch, features)
    if not torch.isfinite(rounded).all():
        raise ValueError(
            f"a reduced feature overflows {BANK_DTYPE}, so every distance against "
            "it would be infinite. Use the 'cls' or 'mean' reduction, or a wider "
            "bank dtype."
        )
    return rounded


@torch.inference_mode()
def collect_reference_bank(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_bfloat16: bool = True,
    reduction: str = DEFAULT_REDUCTION,
    architecture: str | None = None,
) -> ReferenceBank:
    """Algorithm 2 lines 2 to 7: the clean bank with its features at every layer.

    Membership follows cell 4 of the notebook, a sample is kept when the model's
    prediction equals its loader label, so the bank holds no sample the model
    already misreads. The considered layers are every hook point of
    captured_layers, 0 for the input of the first block and 1 to num_blocks for
    each block's output. Features are reduced by reduction and stored in
    BANK_DTYPE on device, layer by layer, so the peak during collection is the
    bank plus 1 layer's batches rather than 2 copies of the bank.
    """
    model.eval()
    core = network_core(model)
    resolved_architecture = (
        architecture if architecture is not None else detect_model_architecture(core)
    )
    has_class_token = resolved_architecture == "vit"
    num_blocks = len(transformer_blocks(core, resolved_architecture))
    layers = tuple(range(num_blocks + 1))

    feature_batches: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    predicted_batches: list[torch.Tensor] = []
    index_batches: list[torch.Tensor] = []
    served = 0
    with captured_layers(model, layers, resolved_architecture) as captured:
        for images, labels in loader:
            probs = forward_probs(
                model, images, device, use_bfloat16
            )  # (batch, num_classes)
            predicted = probs.argmax(dim=1)  # (batch,)
            correct = predicted == labels.to(device).long()  # (batch,)
            positions = torch.arange(
                served, served + images.size(0), device=device
            )  # (batch,)
            served += images.size(0)

            for layer in layers:
                reduced = reduce_activation(
                    captured[layer], reduction, has_class_token
                )  # (batch, features_at_layer)
                feature_batches[layer].append(
                    reduced[correct]
                )  # (correct, features_at_layer)
            predicted_batches.append(predicted[correct])
            index_batches.append(positions[correct])

    if served == 0:
        raise ValueError("the loader served no samples, so no bank can be built")

    features: dict[int, torch.Tensor] = {}
    for layer in layers:
        features[layer] = torch.cat(feature_batches[layer])  # (bank_size, features)
        feature_batches[layer] = []

    predicted = torch.cat(predicted_batches).long()  # (bank_size,)
    source_indices = torch.cat(index_batches).long()  # (bank_size,)
    if predicted.numel() == 0:
        raise ValueError(
            "no sample of the loader is classified correctly, so the bank is "
            "empty and every rank would be undefined"
        )

    bank = ReferenceBank(
        features, predicted, layers, resolved_architecture, source_indices, served
    )
    return bank


def classes_without_reference(bank: ReferenceBank, num_classes: int) -> list[int]:
    """The classes no bank row is predicted as, so their queries take the absent-class rank.

    Every query the model sends to 1 of these classes ranks at the bank size on
    every layer (deviation 4) and is scored as an extreme outlier whether or not
    it carries a trigger. On Tiny ImageNet the shared split holds about 10
    samples per class, so a class can drop out of the bank once its misclassified
    samples are removed. A record of this list beside a run's scores says how many
    of its flags came from the rule rather than from the rank dynamics.
    """
    present = torch.zeros(num_classes, dtype=torch.bool)  # (num_classes,)
    present[bank.predicted.cpu()] = True
    absent = torch.nonzero(~present).flatten().tolist()
    return absent


def own_row_of_source_sample(bank: ReferenceBank) -> torch.Tensor:
    """(source_size,) long: the bank row each sample of the bank's loader is, else -1.

    Handed to first_same_class_rank when the loader being scored is the loader
    the bank came from, so every bank member is ranked against the bank without
    itself, line 12 of Algorithm 2, while the misclassified rest are ranked
    against the whole bank.
    """
    rows = torch.full((bank.source_size,), -1, dtype=torch.long)  # (source_size,)
    rows[bank.source_indices.cpu()] = torch.arange(bank.size)
    return rows


def first_same_class_rank(
    queries: torch.Tensor,
    query_labels: torch.Tensor,
    bank: torch.Tensor,
    bank_labels: torch.Tensor,
    exclude_self: torch.Tensor | None = None,
    chunk_size: int = DISTANCE_CHUNK,
) -> torch.Tensor:
    """Algorithm 2 lines 11 to 13 at 1 layer for a batch of queries, (batch,) long.

    queries is (batch, features) and bank (bank_size, features) in any float
    dtype, query_labels (batch,) and bank_labels (bank_size,) long. exclude_self
    is None or (batch,) long naming the bank row each query is a copy of, with -1
    for a query that is no bank row. The named row is masked to an infinite
    distance, which is what the notebook's ranking_array[1:] does for a bank row
    ranked against its own bank.

    The rank is the 0-based position of the first bank row carrying the query's
    label in the bank sorted by euclidean distance to the query, so a rank of 0
    means the nearest row already has that label. A query whose label appears
    on no bank row gets the bank size, the position just past the sorted bank,
    where the notebook drops the sample. Distances are taken in float32 over
    chunk_size bank rows at a time, so the bank may stay in a narrower dtype.
    """
    batch, feature_width = queries.shape
    bank_size, bank_width = bank.shape
    assert feature_width == bank_width, (feature_width, bank_width)
    assert query_labels.shape == (batch,), query_labels.shape
    assert bank_labels.shape == (bank_size,), bank_labels.shape

    # cdist expands the squared distance through a matmul at these sizes, the
    # same expansion torchmetrics' pairwise_euclidean_distance uses in the
    # notebook, so the rounding of the 2 agrees in kind.
    distances = torch.empty(
        batch, bank_size, device=queries.device, dtype=torch.float32
    )  # (batch, bank_size)
    query_rows = queries.float()  # (batch, features)
    for start in range(0, bank_size, chunk_size):
        stop = min(start + chunk_size, bank_size)
        bank_rows = bank[start:stop].float()  # (chunk, features)
        distances[:, start:stop] = torch.cdist(query_rows, bank_rows)  # (batch, chunk)

    if exclude_self is not None:
        assert exclude_self.shape == (batch,), exclude_self.shape
        is_bank_row = exclude_self >= 0  # (batch,)
        owners = torch.arange(batch, device=queries.device)[is_bank_row]  # (bank_rows,)
        distances[owners, exclude_self[is_bank_row]] = float("inf")

    same_class = query_labels.view(-1, 1) == bank_labels.view(
        1, -1
    )  # (batch, bank_size)
    nearest_same_class = torch.where(
        same_class, distances, torch.full_like(distances, float("inf"))
    ).amin(dim=1)  # (batch,)

    # The sorted position of the nearest same-class row is the number of rows
    # strictly closer than it, which reads the sort without materialising it. A
    # row tied with it at the same distance sorts after it here, as the notebook's
    # unstable sort may or may not do, and exact ties do not occur in practice.
    closer = (distances < nearest_same_class.view(-1, 1)).sum(dim=1)  # (batch,)
    has_same_class = torch.isfinite(nearest_same_class)  # (batch,)
    ranks = torch.where(
        has_same_class, closer, torch.full_like(closer, bank_size)
    )  # (batch,)
    return ranks


@torch.inference_mode()
def rank_trajectories(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    bank: ReferenceBank,
    reduction: str = DEFAULT_REDUCTION,
    use_bfloat16: bool = True,
    loader_is_bank_source: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Algorithm 2 lines 9 to 14 for every sample a loader serves.

    Returns (trajectories, predicted). trajectories is (N, num_layers) float32
    on CPU holding the rank at each of the bank's layers in loader order.
    predicted is (N,) long, the label the model gave each sample. reduction must
    be the reduction the bank was collected under, since the feature widths must
    agree, and first_same_class_rank asserts that they do.

    loader_is_bank_source says the loader is the 1 the bank was collected from,
    served in the same order, so each bank member is ranked with its own row
    excluded and the pass yields leave-one-out trajectories for the bank and
    ordinary trajectories for the misclassified rest. A loader of another length
    under that flag is refused, since the exclusion is by position.
    """
    model.eval()
    has_class_token = bank.architecture == "vit"
    own_rows = own_row_of_source_sample(bank) if loader_is_bank_source else None

    trajectory_batches: list[torch.Tensor] = []
    predicted_batches: list[torch.Tensor] = []
    served = 0
    with captured_layers(model, bank.layers, bank.architecture) as captured:
        for images, _ in loader:
            probs = forward_probs(
                model, images, device, use_bfloat16
            )  # (batch, num_classes)
            predicted = probs.argmax(dim=1)  # (batch,)
            batch = images.size(0)
            exclude_self = (
                own_rows[served : served + batch].to(device)
                if own_rows is not None
                else None
            )  # (batch,) or None
            served += batch

            ranks_by_layer = [
                first_same_class_rank(
                    reduce_activation(captured[layer], reduction, has_class_token),
                    predicted,
                    bank.features[layer],
                    bank.predicted,
                    exclude_self,
                )
                for layer in bank.layers
            ]
            trajectory_batches.append(
                torch.stack(ranks_by_layer, dim=1).cpu()
            )  # (batch, num_layers)
            predicted_batches.append(predicted.cpu())

    if loader_is_bank_source and served != bank.source_size:
        raise ValueError(
            f"the loader served {served} samples where the bank came from "
            f"{bank.source_size}, so it cannot be the bank's source loader"
        )
    if not trajectory_batches:
        empty = (torch.empty(0, len(bank.layers)), torch.empty(0, dtype=torch.long))
        return empty

    trajectories = torch.cat(trajectory_batches).float()  # (N, num_layers)
    predicted = torch.cat(predicted_batches).long()  # (N,)
    return trajectories, predicted


def leave_one_out_trajectories(bank: ReferenceBank) -> torch.Tensor:
    """Algorithm 2 lines 8 to 14 for the bank itself, (bank_size, num_layers) float32.

    Line 12 takes the nearest neighbour from S_j minus x_i, so a bank row never
    counts itself, which the notebook's getDefenseRegion realises by dropping the
    first entry of the sorted bank. These are the clean trajectories the outlier
    model is fitted on. Queries are read in chunks as well as the bank, so the
    largest transient is 2 float32 chunks plus a (chunk, bank_size) matrix.
    """
    own_rows = torch.arange(bank.size, device=bank.predicted.device)  # (bank_size,)

    ranks_by_layer: list[torch.Tensor] = []
    for layer in bank.layers:
        features = bank.features[layer]  # (bank_size, features_at_layer)
        layer_ranks = [
            first_same_class_rank(
                features[start : start + DISTANCE_CHUNK],
                bank.predicted[start : start + DISTANCE_CHUNK],
                features,
                bank.predicted,
                own_rows[start : start + DISTANCE_CHUNK],
            )
            for start in range(0, bank.size, DISTANCE_CHUNK)
        ]
        ranks_by_layer.append(torch.cat(layer_ranks))  # (bank_size,)

    trajectories = (
        torch.stack(ranks_by_layer, dim=1).float().cpu()
    )  # (bank_size, num_layers)
    return trajectories


def fit_trajectory_model(benign: torch.Tensor) -> TrajectoryModel:
    """The clean trajectory distribution, fitted on (N, num_layers) benign ranks.

    Replaces the PCA outlier detector of Algorithm 2 line 15, which the notebook
    takes from pyod with standardization on, weighted on, n_components 'mle' and
    every component selected.

    pyod's score, pyod.models.pca.PCA.decision_function at the current master
        z(x)     = scaler_.transform(x), the fitted StandardScaler
        score(x) = sum(cdist(z(x), selected_components_) / selected_w_components_)
                 = sum_{j=1}^{k} || z(x) - v_j ||_2 / w_j
        v_j      = row j of components_, a unit principal axis read as a point
        w_j      = explained_variance_ratio_[j], since weighted is True
        k        = n_components_, every component when none is deselected

    this port
        z(x)     = (x - mean) / std, with std 1 where it is under the floor
        score(x) = z(x)^T (Sigma + epsilon I)^+ z(x)
        Sigma    = covariance of z over the benign trajectories
        epsilon  = COVARIANCE_FLOOR

    pyod's quantity is a weighted sum of distances from a standardised point to
    each eigenvector treated as a point, so it is no reconstruction error. Both
    scores grow with the standardised distance from the clean centre, which is
    the direction Algorithm 2 line 19 relies on, and nothing more is claimed.

    The fit runs in float64 because the pseudo-inverse's default cutoff in
    float32 sits above the floor and would discard exactly the directions the
    floor exists to keep.
    """
    assert benign.dim() == 2, benign.shape
    num_samples, num_layers = benign.shape
    if num_samples < 2:
        raise ValueError(
            f"{num_samples} benign trajectories cannot define a covariance, so the "
            "bank needs at least 2 correctly classified samples"
        )

    benign64 = benign.double()  # (N, num_layers)
    mean = benign64.mean(dim=0)  # (num_layers,)
    spread = benign64.std(dim=0, unbiased=False)  # (num_layers,)
    std = torch.where(
        spread <= STANDARD_DEVIATION_FLOOR, torch.ones_like(spread), spread
    )  # (num_layers,)

    standardised = (benign64 - mean) / std  # (N, num_layers)
    covariance = (
        standardised.T @ standardised / (num_samples - 1)
    )  # (num_layers, num_layers)
    floored = covariance + COVARIANCE_FLOOR * torch.eye(
        num_layers, dtype=torch.float64
    )  # (num_layers, num_layers)
    precision = torch.linalg.pinv(floored, hermitian=True)  # (num_layers, num_layers)

    model = TrajectoryModel(mean, std, precision)
    return model


def outlier_scores(trajectories: torch.Tensor, model: TrajectoryModel) -> torch.Tensor:
    """M(x) of Algorithm 2 line 19 per trajectory, (N,) float32, high for outliers.

    trajectories is (N, num_layers) in any float dtype. Not negated, so this is
    the quantity the paper thresholds and can be read against its box plots.
    """
    assert trajectories.dim() == 2, trajectories.shape
    assert trajectories.shape[1] == model.mean.numel(), trajectories.shape

    standardised = (trajectories.double() - model.mean) / model.std  # (N, num_layers)
    squared_mahalanobis = torch.einsum(
        "nl,lk,nk->n", standardised, model.precision, standardised
    )  # (N,)

    scores = squared_mahalanobis.float()  # (N,)
    return scores


def ted_scores(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    bank: ReferenceBank,
    trajectory_model: TrajectoryModel,
    reduction: str = DEFAULT_REDUCTION,
    use_bfloat16: bool = True,
    loader_is_bank_source: bool = False,
) -> torch.Tensor:
    """TED score per sample, shape (N,) float32 on CPU, low meaning poisoned.

    Negated at this boundary. Algorithm 2 line 19 flags a sample whose outlier
    score exceeds tau, so the paper's statistic is high for poisoned, the
    opposite of PSU's convention, and returning it unnegated would produce a
    well-formed, exactly inverted detector. loader_is_bank_source is set only
    when loader is the loader the bank was collected from, so its members get
    their leave-one-out ranks.
    """
    trajectories, _ = rank_trajectories(
        model, loader, device, bank, reduction, use_bfloat16, loader_is_bank_source
    )
    if trajectories.numel() == 0:
        return torch.empty(0)

    outliers = outlier_scores(trajectories, trajectory_model)  # (N,)
    if not torch.isfinite(outliers).all():
        raise ValueError(
            "a TED outlier score is not finite, which a floored covariance and "
            "integer ranks cannot produce unless a feature overflowed"
        )

    scores = -outliers  # (N,), low means poisoned
    return scores
