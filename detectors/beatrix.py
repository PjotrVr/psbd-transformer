"""Beatrix: class-conditional Gram-matrix deviation (Ma et al., NDSS 2023).

Paper: "The 'Beatrix' Resurrections: Robust Backdoor Detection via Gram
Matrices", arXiv:2209.11715. Feature modelling is Section IV-A, Equations (8)
and (9). The deviation measurement is Section IV-B, Equations (10) to (14), and
the threshold determination paragraph of the same section is the 5-fold
jackknife the validation split is scored with here.

    original form
        G       = v v^T                                                 Eq. (8)
        G^p     = ( v^p (v^p)^T )^(1/p)                                 Eq. (9)
        s       = [ vec(G^1), vec(G^2), ..., vec(G^P) ]
        s~_j    = median({ s_ij, i in 1..|X_t| })                       Eq. (10)
        MAD_j   = median({ |s_ij - s~_j|, i in 1..|X_t| })              Eq. (11)
        delta_j = 0                     if min <= s^_j <= max           Eq. (12), (13)
                = (min - s^_j) / min    if s^_j <= min
                = (s^_j - max) / max    if max <= s^_j
        with      min = s~_j - k MAD_j,  max = s~_j + k MAD_j
        delta   = 2 / (n (n + 1) P) * sum_j delta_j                     Eq. (14)

    descriptive form
        gram[p]        = signed p-th root of (tokens^p)^T (tokens^p), a
                         (dim, dim) matrix contracted over the token axis
        gram_features  = the upper triangles of gram[1] to gram[P], concatenated
        lower, upper   = per-entry median minus and plus band_width * MAD over
                         the clean references the model put in the same class
        deviation      = mean over entries of the relative excess below lower
                         or above upper

v is the feature representation at layer l, n channels by m positions on the
paper's ConvNet, vec the upper triangle with the diagonal, P the order bound, X_t
the clean samples the model assigns to class t, s^ the query's feature vector and
k the band scale factor, 10.

Mechanism. A trigger drives the prediction to the target class through
activations the clean members of that class never show, so the query's Gram
entries, the second moments of its tokens across feature dimensions, fall
outside the band the class's clean references span, and the deviation grows
with how many entries fall outside and by how much. A clean input of the class
lands inside the band on nearly every entry and its deviation stays near 0.

Data requirement: the clean validation split, without its labels. The paper
budgets 30 labelled images per class, and this port groups the shared split by
the model's predicted label as both released implementations do, so the labels
are never read. Forward-pass cost: 1 per input plus the Gram algebra, and a
fixed cost of 1 pass over the validation split for the bands.

Deviations from the paper, each recorded in full in docs/detectors/beatrix.md:

  1. P = 4, the value Section V-A fixes after its order ablation. The released
     code runs orders 1 to 8, which overflows float32 on a ViT residual stream.
  2. The Gram is taken at the output of block 9 of 12 on ViT-B/16 and block 22
     of 24 on Swin-S, the depth of the released hook on the input of layer4, and
     contracts over the tokens so it is (dim, dim), the analogue of the paper's
     channel Gram. The class token is included as every spatial position is on
     a ConvNet.
  3. The reference set is the shared 2000-sample split grouped by predicted
     label, about 10 per class on Tiny, 20 on CIFAR-100, 46 on GTSRB and 200
     on CIFAR-10 against the paper's 30, and a class under MIN_CLASS_SAMPLES
     takes the pooled band over every reference.
  4. The sum is divided by n(n + 1) P / 2 as Eq. (14) writes it. The released
     code divides by n^2 P / 2, a constant factor on every score.
  5. The validation split is scored out of fit by the released 5-fold
     jackknife, never against bands it helped fit.
  6. Activations leave the model as float16, the reference bank's dtype, and
     are cast to float32 before any Gram, on the reference and the query side
     alike. The bank is held on the CPU and moves to the device 1 class at a
     time.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from analysis.features import (
    as_token_sequence,
    captured_layers,
    detect_model_architecture,
)
from defences.inference import forward_logits
from models.backbones import network_core

# p in 1..P of Eq. (9). Section V-A fixes P = 4 after the order ablation, and on
# a ViT it is also a float32 range limit: a Gram entry sums 197 products of 2
# p-th powers, so a residual-stream value of 1000 contributes 1e24 at p = 4 and
# 1e48 at p = 8, past the 3.4e38 float32 holds.
PAPER_POWERS: tuple[int, ...] = (1, 2, 3, 4)

# What the released code runs, order_list=np.arange(1, 9) in BEAT_detector at
# Beatrix.py:410. Kept so a number produced with it is recorded rather than
# inherited, and unused by default because of the overflow above.
OFFICIAL_CODE_POWERS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8)

# k in Eq. (13), "set to 10 in our case", written as mads * 10 in minmax_mad at
# Beatrix.py:328.
MAD_BAND = 10.0

# get_deviations_ at Beatrix.py:365 divides by |min + 1e-6| rather than by
# |min|, which keeps an entry whose band edge sits at exactly 0 from dividing
# by 0.
RELATIVE_EPSILON = 1e-6

# Section V-A budgets "30 images per class as a clean dataset for the defender",
# clean_data_perclass = 30 at Beatrix.py:415. The shared split gives 10 to 200.
PAPER_CLEAN_PER_CLASS = 30

# Below this many references a class's median and MAD are read off too few
# points to bound anything, so the pooled band over every reference stands in.
MIN_CLASS_SAMPLES = 5

# T in Section IV-B's threshold determination, step = 5 in threshold_determine
# at Beatrix.py:390, which holds out each fifth of the references by position.
JACKKNIFE_FOLDS = 5

# The captured_layers index read per architecture, 0 being the input of block 0
# and k the output of block k. The released hook reads the input of layer4 at
# Beatrix.py:57, 3 quarters of the way through a ResNet. Block 9 of 12 on
# ViT-B/16 sits at the same depth. Block 22 of 24 on Swin-S is the last block
# of its third stage, the last point before the final patch merge.
FEATURE_LAYER_BY_ARCHITECTURE: dict[str, int] = {"vit": 9, "swin": 22}

# Token matrices go through the Gram algebra this many at a time, so no more
# than 1 batch of Gram vectors exists at once. 64 of them at dim 768 and P = 4
# are 302 MB of float32.
GRAM_BATCH_SIZE = 64

# A band fit holds a (references, columns) float32 block plus 1 temporary of
# the same size for the MAD. Columns are chunked to keep that block under this
# many bytes, which turns the pooled fit over 2000 references into about 9
# passes.
FIT_CHUNK_BYTES = 1 << 30


def default_feature_layer(model: nn.Module, architecture: str | None = None) -> int:
    """The captured_layers index Beatrix reads on this model, from FEATURE_LAYER_BY_ARCHITECTURE."""
    resolved = (
        architecture
        if architecture is not None
        else detect_model_architecture(network_core(model))
    )
    if resolved not in FEATURE_LAYER_BY_ARCHITECTURE:
        raise ValueError(
            f"no Beatrix feature layer is declared for architecture {resolved!r}, "
            f"known: {sorted(FEATURE_LAYER_BY_ARCHITECTURE)}"
        )

    layer = FEATURE_LAYER_BY_ARCHITECTURE[resolved]
    return layer


def gram_entry_count(dim: int) -> int:
    """n(n + 1) / 2, the length of the upper triangle of a (dim, dim) Gram matrix."""
    count = dim * (dim + 1) // 2
    return count


def gram_features(
    tokens: torch.Tensor, powers: tuple[int, ...] = PAPER_POWERS
) -> torch.Tensor:
    """Eq. (9) at every order in powers, vectorised, (batch, num_powers * dim (dim + 1) / 2).

    tokens is (batch, tokens, dim) in float32 or float64. The Gram of each order
    contracts over the token axis, so it is (dim, dim) and reads the correlation
    between feature dimensions across positions, which is what the paper's
    channel-by-channel Gram reads on a ConvNet. The upper triangle including the
    diagonal is kept in torch.triu_indices order and the orders are concatenated
    in the order given.

    Half precision is refused rather than promoted, since a bfloat16 Gram sums
    197 products at 8 bits of mantissa and the caller is meant to have cast
    already. A non-finite entry raises: at p = 8 a residual-stream dimension in
    the hundreds overflows float32, which is 1 reason PAPER_POWERS stops at 4.
    """
    if tokens.dim() != 3:
        raise ValueError(
            f"expected (batch, tokens, dim), got shape {tuple(tokens.shape)}"
        )
    if tokens.dtype not in (torch.float32, torch.float64):
        raise TypeError(
            f"gram_features needs float32 or float64 tokens, got {tokens.dtype}"
        )

    dim = tokens.shape[2]
    row_index, column_index = torch.triu_indices(
        dim, dim, device=tokens.device
    )  # (entries,) each

    per_order = []
    for power in powers:
        powered = tokens**power  # (batch, tokens, dim)
        gram = powered.transpose(1, 2) @ powered  # (batch, dim, dim)
        # The signed root maps every order back to the scale of the first, so
        # the MAD band of a high order is of the same width as the others.
        rooted = gram.sign() * gram.abs() ** (1.0 / power)  # (batch, dim, dim)
        per_order.append(rooted[:, row_index, column_index])  # (batch, entries)

    features = torch.cat(per_order, dim=1)  # (batch, num_powers * entries)
    if not torch.isfinite(features).all():
        raise FloatingPointError(
            f"non-finite Gram entry at powers {powers}: a token value raised to "
            "twice the order overflowed float32, so lower the order bound"
        )
    return features


def fit_gram_band(
    features: torch.Tensor, band_width: float = MAD_BAND
) -> tuple[torch.Tensor, torch.Tensor]:
    """Eq. (10), (11) and the [min, max] of Eq. (13), as (lower, upper), each (features,).

    features is (references, features), 1 row per clean reference of a class.
    torch.median returns the lower of the 2 middle values on an even count, the
    convention the released get_median_mad inherits, so a band fitted here on
    the same rows equals the reference's to the bit.
    """
    if features.dim() != 2:
        raise ValueError(
            f"expected (references, features), got shape {tuple(features.shape)}"
        )
    if features.shape[0] == 0:
        raise ValueError("no reference rows, so Eq. (10) has no median")

    median = features.median(dim=0).values  # (features,)
    mad = (features - median).abs().median(dim=0).values  # (features,)

    lower = median - band_width * mad  # (features,)
    upper = median + band_width * mad  # (features,)
    return lower, upper


def gram_deviation(
    features: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    gram_entries: int,
    num_powers: int,
) -> torch.Tensor:
    """Eq. (12) to (14), the deviation of each row of features, (batch,), high for poisoned.

    lower and upper are (features,) for a shared band or (batch, features) for a
    band gathered per row, and the feature count must equal gram_entries *
    num_powers, the n(n + 1) P / 2 of Eq. (14) the sum is divided by. The
    denominators follow the released code, |min + 1e-6| rather than |min|.
    """
    num_features = gram_entries * num_powers
    widths = (features.shape[-1], lower.shape[-1], upper.shape[-1])
    if any(width != num_features for width in widths):
        raise ValueError(
            f"features, lower and upper have widths {widths}, expected "
            f"{gram_entries} Gram entries times {num_powers} orders = {num_features}"
        )

    below = (
        F.relu(lower - features) / (lower + RELATIVE_EPSILON).abs()
    )  # (batch, features)
    above = (
        F.relu(features - upper) / (upper + RELATIVE_EPSILON).abs()
    )  # (batch, features)

    deviation = (below + above).sum(dim=1) / num_features  # (batch,)
    return deviation


def captured_token_matrix(
    captured: dict[int, torch.Tensor], layer: int
) -> torch.Tensor:
    """The latest capture at layer as (batch, tokens, dim) float16, still on its device.

    A Swin capture arrives as (batch, height, width, channels) and its grid is
    the token axis. float16 halves the reference bank and loses nothing from a
    bfloat16 forward, whose 8 bits of mantissa it holds with 3 to spare. The
    query side goes through the same cast so a fitted band and a scored query
    are read off identically rounded values. A value past float16's range raises
    here, naming the layer, rather than surfacing as a non-finite Gram entry 2
    functions later.
    """
    tokens = as_token_sequence(captured[layer])  # (batch, tokens, dim)
    half = tokens.detach().to(torch.float16)  # (batch, tokens, dim)
    if not torch.isfinite(half).all():
        raise FloatingPointError(
            f"layer {layer} activation does not fit float16, so its Gram cannot be "
            "taken from the float16 reference bank"
        )
    return half


@torch.inference_mode()
def collect_reference_tokens(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    layer: int,
    use_bfloat16: bool = True,
    architecture: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The split's token matrices and predicted labels, (N, tokens, dim) float16 on the CPU and (N,) long.

    1 forward per image, the pass the Gram statistics and the predicted label
    both come from. The bank is 2000 x 197 x 768 halves on ViT-B/16, 605 MB, and
    lives on the CPU so a class at a time can be moved to the device for its
    band. Rows are in loader order.
    """
    model.eval()

    token_batches = []
    predicted_batches = []
    with captured_layers(model, (layer,), architecture) as captured:
        for images, _ in loader:
            logits = forward_logits(
                model, images, device, use_bfloat16
            )  # (batch, num_classes)
            predicted_batches.append(logits.argmax(dim=1).cpu())  # (batch,)
            token_batches.append(
                captured_token_matrix(captured, layer).cpu()
            )  # (batch, tokens, dim)

    if not token_batches:
        empty = (
            torch.empty(0, 0, 0, dtype=torch.float16),
            torch.empty(0, dtype=torch.long),
        )
        return empty

    tokens = torch.cat(token_batches)  # (N, tokens, dim)
    predicted = torch.cat(predicted_batches).long()  # (N,)
    return tokens, predicted


@dataclass(frozen=True, eq=False)
class ClassBands:
    """The fitted state Beatrix scores against, 1 [min, max] band per predicted class.

    lower_by_class and upper_by_class are (num_classes, features) float32 on the
    scoring device, every row filled, since a query may be predicted as any
    class. A class with fewer than min_class_samples references holds the pooled
    band over every reference, which pooled_lower and pooled_upper keep as
    (features,) and which stay None when no class needed them. counts is
    (num_classes,) long, the references per predicted class, so a record can
    say how many classes fell back. powers and dim fix the feature layout the
    bands were fitted on, so a query of another width is refused.
    """

    lower_by_class: torch.Tensor
    upper_by_class: torch.Tensor
    pooled_lower: torch.Tensor | None
    pooled_upper: torch.Tensor | None
    counts: torch.Tensor
    min_class_samples: int
    powers: tuple[int, ...]
    dim: int

    @property
    def num_features(self) -> int:
        count = gram_entry_count(self.dim) * len(self.powers)
        return count

    @property
    def pooled_class_count(self) -> int:
        count = int((self.counts < self.min_class_samples).sum().item())
        return count


def fit_band_over_tokens(
    tokens: torch.Tensor,
    powers: tuple[int, ...],
    device: torch.device,
    batch_size: int = GRAM_BATCH_SIZE,
    chunk_bytes: int = FIT_CHUNK_BYTES,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The Eq. (10) to (13) band over a set of token matrices, (lower, upper), each (features,) on device.

    tokens is (references, tokens, dim) float16 on the CPU. The Gram vectors of
    every reference never exist at once: the band is fitted in column chunks
    sized by chunk_bytes, and each chunk recomputes the Gram of every reference
    batch and keeps only its columns. A class of 46 references fits in 1 chunk
    and 1 pass. The pooled band over 2000 references takes about 9 passes, in
    place of the 9.4 GB the full matrix would take.
    """
    if tokens.dim() != 3:
        raise ValueError(
            f"expected (references, tokens, dim), got shape {tuple(tokens.shape)}"
        )
    num_references, _, dim = tokens.shape
    if num_references == 0:
        raise ValueError("no reference token matrices, so Eq. (10) has no median")

    num_features = gram_entry_count(dim) * len(powers)
    bytes_per_column = num_references * torch.finfo(torch.float32).bits // 8
    columns_per_chunk = max(1, chunk_bytes // bytes_per_column)

    lower = torch.empty(num_features, device=device)  # (features,)
    upper = torch.empty(num_features, device=device)  # (features,)
    for start in range(0, num_features, columns_per_chunk):
        stop = min(start + columns_per_chunk, num_features)

        column_blocks = []
        for batch in tokens.split(batch_size):
            features = gram_features(
                batch.to(device).float(), powers
            )  # (batch, features)
            # A slice keeps its whole parent alive, so the block is copied out
            # and the parent freed before the next batch's Gram is taken.
            column_blocks.append(features[:, start:stop].clone())  # (batch, chunk)
        chunk_features = torch.cat(column_blocks)  # (references, chunk)

        lower[start:stop], upper[start:stop] = fit_gram_band(chunk_features)

    return lower, upper


def fit_class_bands(
    tokens: torch.Tensor,
    predicted: torch.Tensor,
    num_classes: int,
    powers: tuple[int, ...] = PAPER_POWERS,
    device: torch.device = torch.device("cpu"),
    min_class_samples: int = MIN_CLASS_SAMPLES,
) -> ClassBands:
    """A band per predicted class from the reference bank, the state Beatrix scores against.

    tokens is (N, tokens, dim) float16 on the CPU and predicted (N,) long, the
    pair collect_reference_tokens returns. Eq. (10) groups by class t, which the
    released code reads as the model's own prediction on the clean image, so a
    query is compared with the clean inputs the model puts where it put the
    query. A class under min_class_samples takes the pooled band, which is
    fitted only when some class needs it.
    """
    if tokens.dim() != 3:
        raise ValueError(f"expected (N, tokens, dim), got shape {tuple(tokens.shape)}")
    if predicted.shape != (tokens.shape[0],):
        raise ValueError(
            f"predicted has shape {tuple(predicted.shape)} for {tokens.shape[0]} "
            "token matrices"
        )
    counts = torch.bincount(predicted.long(), minlength=num_classes)  # (num_classes,)
    if counts.numel() != num_classes:
        raise ValueError(
            f"a predicted label reaches {counts.numel() - 1} but num_classes is "
            f"{num_classes}"
        )

    dim = tokens.shape[2]
    num_features = gram_entry_count(dim) * len(powers)
    lower_by_class = torch.empty(
        num_classes, num_features, device=device
    )  # (num_classes, features)
    upper_by_class = torch.empty(
        num_classes, num_features, device=device
    )  # (num_classes, features)

    pooled_lower, pooled_upper = None, None
    if bool((counts < min_class_samples).any()):
        pooled_lower, pooled_upper = fit_band_over_tokens(tokens, powers, device)

    for class_index in range(num_classes):
        if counts[class_index] < min_class_samples:
            lower_by_class[class_index] = pooled_lower
            upper_by_class[class_index] = pooled_upper
            continue

        members = tokens[predicted == class_index]  # (members, tokens, dim)
        lower_by_class[class_index], upper_by_class[class_index] = fit_band_over_tokens(
            members, powers, device
        )

    bands = ClassBands(
        lower_by_class=lower_by_class,
        upper_by_class=upper_by_class,
        pooled_lower=pooled_lower,
        pooled_upper=pooled_upper,
        counts=counts,
        min_class_samples=min_class_samples,
        powers=tuple(powers),
        dim=dim,
    )
    return bands


def _deviation_of_batch(
    tokens: torch.Tensor, predicted: torch.Tensor, bands: ClassBands
) -> torch.Tensor:
    """Eq. (14) for 1 batch, (batch,), each row against the band of its predicted class.

    tokens is (batch, tokens, dim) float32 and predicted (batch,) long, both on
    the bands' device. The Gram vectors of a query exist only here and are
    freed with the batch.
    """
    if tokens.shape[2] != bands.dim:
        raise ValueError(
            f"token width {tokens.shape[2]} differs from the {bands.dim} the bands "
            "were fitted on"
        )

    features = gram_features(tokens, bands.powers)  # (batch, features)
    lower = bands.lower_by_class[predicted]  # (batch, features)
    upper = bands.upper_by_class[predicted]  # (batch, features)

    deviation = gram_deviation(
        features, lower, upper, gram_entry_count(bands.dim), len(bands.powers)
    )  # (batch,)
    return deviation


def deviations_of_tokens(
    tokens: torch.Tensor,
    predicted: torch.Tensor,
    bands: ClassBands,
    device: torch.device,
    batch_size: int = GRAM_BATCH_SIZE,
) -> torch.Tensor:
    """Eq. (14) for every row of a token bank, (N,) float32 on the CPU, high for poisoned.

    tokens is (N, tokens, dim) float16 on the CPU and predicted (N,) long. The
    bank is moved to device a batch at a time. The jackknife scores its held-out
    folds through this, and a test can score a hand-built bank without a model.
    """
    batch_deviations = []
    for token_batch, predicted_batch in zip(
        tokens.split(batch_size), predicted.split(batch_size)
    ):
        deviation = _deviation_of_batch(
            token_batch.to(device).float(), predicted_batch.to(device), bands
        )  # (batch,)
        batch_deviations.append(deviation.cpu())

    if not batch_deviations:
        return torch.empty(0)

    deviations = torch.cat(batch_deviations).float()  # (N,)
    return deviations


def jackknife_deviations(
    tokens: torch.Tensor,
    predicted: torch.Tensor,
    num_classes: int,
    powers: tuple[int, ...] = PAPER_POWERS,
    device: torch.device = torch.device("cpu"),
    folds: int = JACKKNIFE_FOLDS,
    min_class_samples: int = MIN_CLASS_SAMPLES,
) -> torch.Tensor:
    """Out-of-fit Eq. (14) for every reference, (N,) float32 on the CPU, high for poisoned.

    threshold_determine at Beatrix.py:388 cuts the references into folds
    contiguous blocks by position, fits the bands on the other blocks, scores
    the held-out block and pools the results to set the threshold. The same
    happens here, with block boundaries at i * folds // N so every reference is
    scored exactly once where the released code drops the remainder of an
    uneven cut. A sample deviates less from a band it helped fit, so scoring the
    validation split in sample would set the threshold too tight.

    A contiguous cut holds out a random fifth only when the rows arrive in
    random order. The released code shuffles before cutting, and the shared
    split is a randperm of the test set served without reshuffling, so the cut
    is random there. A class-sorted loader would hold out whole classes and
    score them against the pooled band.
    """
    num_references = tokens.shape[0]
    if num_references < folds:
        raise ValueError(
            f"{num_references} references cannot be cut into {folds} folds"
        )

    fold_ids = torch.arange(num_references) * folds // num_references  # (N,)
    deviations = torch.empty(num_references)  # (N,)
    for fold in range(folds):
        held_out = fold_ids == fold  # (N,)
        bands = fit_class_bands(
            tokens[~held_out],
            predicted[~held_out],
            num_classes,
            powers,
            device,
            min_class_samples,
        )
        deviations[held_out] = deviations_of_tokens(
            tokens[held_out], predicted[held_out], bands, device
        )

    _require_finite(deviations, "jackknife deviation")
    return deviations


@torch.inference_mode()
def beatrix_deviations(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    layer: int,
    bands: ClassBands,
    use_bfloat16: bool = True,
    architecture: str | None = None,
) -> torch.Tensor:
    """Eq. (14) per sample of a loader, (N,) float32 on the CPU, high for poisoned.

    1 forward per batch gives both the predicted label, which picks the band,
    and the layer activation the Gram is taken from. Rows are in loader order.

    Not negated. Sign correction happens once, in deviation_scores, so this
    stays the paper's statistic and can be compared to a published number.
    """
    model.eval()

    batch_deviations = []
    with captured_layers(model, (layer,), architecture) as captured:
        for images, _ in loader:
            logits = forward_logits(
                model, images, device, use_bfloat16
            )  # (batch, num_classes)
            predicted = logits.argmax(dim=1)  # (batch,)
            tokens = captured_token_matrix(
                captured, layer
            ).float()  # (batch, tokens, dim)

            deviation = _deviation_of_batch(tokens, predicted, bands)  # (batch,)
            batch_deviations.append(deviation.cpu())

    if not batch_deviations:
        return torch.empty(0)

    deviations = torch.cat(batch_deviations).float()  # (N,)
    _require_finite(deviations, "deviation")
    return deviations


def deviation_scores(deviations: torch.Tensor) -> torch.Tensor:
    """Beatrix deviations as scores in the shared convention, (N,), low meaning poisoned.

    The module's single negation. The paper flags an input whose deviation
    exceeds a percentile of the benign deviations, so its statistic is high for
    poisoned, the opposite of PSU's convention. Returning it unnegated would
    produce a well-formed, exactly inverted detector. The jackknife scores of the
    validation split and the scores of every other loader both pass through
    here, so the 2 cannot disagree in sign.
    """
    scores = -deviations.float()  # (N,), low means poisoned
    return scores


def beatrix_scores(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    layer: int,
    bands: ClassBands,
    use_bfloat16: bool = True,
    architecture: str | None = None,
) -> torch.Tensor:
    """Beatrix score per sample, shape (N,), low meaning poisoned."""
    deviations = beatrix_deviations(
        model, loader, device, layer, bands, use_bfloat16, architecture
    )
    if deviations.numel() == 0:
        return torch.empty(0)

    scores = deviation_scores(deviations)  # (N,)
    return scores


def _require_finite(values: torch.Tensor, what: str) -> None:
    """Refuse a non-finite statistic, which would otherwise sort to 1 end of every ranking."""
    if not torch.isfinite(values).all():
        count = int((~torch.isfinite(values)).sum().item())
        raise FloatingPointError(
            f"{count} non-finite {what} values out of {values.numel()}"
        )
