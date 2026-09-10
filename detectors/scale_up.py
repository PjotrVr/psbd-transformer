"""SCALE-UP: scaled prediction consistency (Guo et al., ICLR 2023).

Paper: "SCALE-UP: An Efficient Black-box Input-level Backdoor Detection via
Analyzing Scaled Prediction Consistency", arXiv:2302.03251. The statistic is
Section 4.2, Equation (2). The data-limited variant is Section 4.3, Equations
(3) and (4).

    original form
        SPC(x)  = ( sum_{n in S} I{ C(n * x) = C(x) } ) / |S|          Eq. (2)

        mu_i    = E_{x in X_i}[ SPC(x) ]                               Eq. (3)
        sigma_i = sqrt( E_{x in X_i}[ (SPC(x) - mu_i)^2 ] )            Eq. (3)

        NSPC(x) = ( SPC(x) - mu_yhat ) / sigma_yhat,  yhat = C(x)      Eq. (4)

    descriptive form
        spc(image)  = fraction of amplified copies whose predicted label
                      equals the label predicted on the unamplified image
        nspc(image) = (spc(image) - clean_mean[predicted_class])
                      / clean_std[predicted_class]

S is the set of integer scaling factors, C the classifier's predicted label and
X_i the defender's benign samples of class i. The paper's rule is "backdoor if
SPC(x) > T".

Mechanism. Amplifying every pixel toward saturation destroys the class evidence of
a benign image, so its predicted label moves. A trigger is high-contrast and
survives the amplification, so a poisoned image keeps being classified as the
target and its predictions stay consistent.

2 variants, both implemented. Data-free is Eq. (2) alone, the paper's headline
method and the default here. Data-limited standardizes it by Eqs. (3) and (4)
against per-class clean statistics, which the paper presents as an optional
refinement.

Data requirement: none for data-free. The clean validation split for
data-limited, where the paper budgets 100 benign samples per class.
Forward-pass cost: |S| + 1 per input, 6 at the default scaling set.

Deviations from the paper, each recorded in full in docs/detector-ports.md:

  1. The scaling set defaults to the paper's S = {3, 5, 7, 9, 11}. The released
     code uses range(1, 12), a different statistic on a different support, kept
     as OFFICIAL_CODE_SCALES so a number produced with it is recorded rather
     than inherited.
  2. Per-class statistics come from the shared 2000-sample split, about 20 per
     class on CIFAR-100. A class below MIN_CLASS_SAMPLES falls back to the pooled
     statistics, a case the paper never needed to specify.
  3. Consistency is always measured against C(x), the model's own prediction, as
     Eq. (2) says. The grouping into X_i uses the validation sample's true class,
     as Eq. (3) says.
  4. No input noise. The released code adds 0.02 * U[0, 1) before scaling, which
     appears in no equation.
  5. No prediction-correctness masking. Eligibility is owned by
     attacks.poisoning.AttackSuccessSet, so the detector scores every sample the
     loader serves.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .strip import normalization_buffers

from defences.inference import forward_probs

# S in Eq. (2), as printed in Section 4.2.
PAPER_SCALES: tuple[int, ...] = (3, 5, 7, 9, 11)

# What the authors' released code actually runs, range(1, 12). Kept available and
# named so a number produced with it can never be mistaken for a paper number.
OFFICIAL_CODE_SCALES: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)

# Below this many clean validation samples, a class's own Eq. (3) statistics are
# too noisy to divide by, so the pooled statistics stand in. The paper budgets
# 100 per class and never meets this case.
MIN_CLASS_SAMPLES = 5

# Guards Eq. (4) against a class whose clean SPC never varies, which makes
# sigma_i exactly 0 and the quotient infinite.
STD_FLOOR = 1e-6


def amplify_pixels(
    images: torch.Tensor,
    factor: int,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """n * x constrained to [0, 1], returned in normalized space, same shape in.

    Section 4.2 states "we constrain n * x in [0, 1] during the multiplication
    process", so the multiply and the clip both belong in pixel space. The loaders
    here deliver normalized tensors, so the round trip is denormalize, multiply,
    clip, renormalize.

    Scaling the normalized tensor directly would amplify the dataset mean as
    though it were signal. The clip, where the method's whole nonlinearity lives,
    would land in the wrong place.
    """
    pixels = (images * std + mean).clamp(0.0, 1.0)  # (batch, C, H, W)
    amplified = (pixels * float(factor)).clamp(0.0, 1.0)

    renormalized = (amplified - mean) / std
    return renormalized


@torch.inference_mode()
def spc_scores(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mean: tuple[float, ...],
    std: tuple[float, ...],
    scales: tuple[int, ...] = PAPER_SCALES,
    use_bfloat16: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Raw SPC per sample plus the 2 things the data-limited variant needs.

    Returns (spc, predicted_labels, true_labels), each (N,). spc is Eq. (2) in
    [0, 1], high for poisoned. predicted_labels is C(x) and true_labels is
    whatever the loader served, which the caller needs to group Eq. (3) by class.

    Not negated. Sign correction happens once, at the scoring boundary, so this
    stays the paper's statistic and can be compared to a published number directly.
    """
    model.eval()

    spc_batches = []
    predicted_batches = []
    label_batches = []
    for images, labels in loader:
        images = images.to(device)  # (batch, C, H, W)
        mean_tensor, std_tensor = normalization_buffers(
            mean, std, images.device, images.dtype
        )

        baseline_probs = forward_probs(model, images, device, use_bfloat16)
        baseline_labels = baseline_probs.argmax(dim=1)  # (batch,) = C(x)

        agreements = torch.zeros(images.size(0), device=device)  # (batch,)
        for factor in scales:
            amplified = amplify_pixels(images, factor, mean_tensor, std_tensor)
            probs = forward_probs(model, amplified, device, use_bfloat16)
            agreements += (probs.argmax(dim=1) == baseline_labels).float()

        spc_batches.append((agreements / len(scales)).cpu())  # Eq. (2)
        predicted_batches.append(baseline_labels.cpu())
        label_batches.append(labels.cpu().long())

    if not spc_batches:
        empty = (torch.empty(0), torch.empty(0, dtype=torch.long))
        return empty[0], empty[1], empty[1]

    spc = torch.cat(spc_batches).float()  # (N,)
    predicted_labels = torch.cat(predicted_batches).long()  # (N,)
    true_labels = torch.cat(label_batches).long()  # (N,)
    return spc, predicted_labels, true_labels


def fit_class_spc_statistics(
    validation_spc: torch.Tensor,
    validation_labels: torch.Tensor,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Eq. (3): per-class mean and standard deviation of clean SPC, each (num_classes,).

    validation_labels groups the samples into X_i, the defender's benign samples
    of class i, which Eq. (3) defines by true class rather than by prediction.

    2 fallbacks to the pooled statistics, neither of which the paper specifies
    because neither arises at its budget of 100 samples per class. Too few
    samples: the shared split gives roughly 20 per class on CIFAR-100 and can give
    0 for a class it missed, and dividing by the standard deviation of 1 sample
    manufactures a score out of nothing. Degenerate spread: SPC lives on a grid of
    1/|S|, so a class whose few clean samples all landed on the same grid point
    has sigma_i exactly 0, an artifact of quantization rather than low variance,
    and Eq. (4) would return enormous z-scores for that class alone.
    """
    pooled_mean = validation_spc.mean()
    pooled_std = validation_spc.std(unbiased=True)

    class_means = pooled_mean.repeat(num_classes).clone()  # (num_classes,)
    class_stds = pooled_std.repeat(num_classes).clone()  # (num_classes,)

    for class_index in range(num_classes):
        members = validation_spc[validation_labels == class_index]
        if members.numel() < MIN_CLASS_SAMPLES:
            continue

        class_means[class_index] = members.mean()
        spread = members.std(unbiased=True)
        if spread > STD_FLOOR:
            class_stds[class_index] = spread

    class_stds = class_stds.clamp_min(STD_FLOOR)
    return class_means, class_stds


# Folds for standardising the fitting set against itself. 2 keeps every class's
# per-fold count as large as the shared split allows while still holding out
# every sample from the statistics it is scored against.
CROSS_FIT_FOLDS = 2


def cross_fitted_validation_scores(
    spc: torch.Tensor,
    predicted_labels: torch.Tensor,
    true_labels: torch.Tensor,
    num_classes: int,
    folds: int = CROSS_FIT_FOLDS,
) -> torch.Tensor:
    """Data-limited scores for the fitting set itself, shape (N,), low meaning poisoned.

    The clean validation split is both the set Eq. (3) is fitted on and the set
    the detection threshold is read from. A sample standardised against
    statistics it helped fit sits closer to the class mean than a fresh sample
    would, so an in-sample threshold is too tight and the achieved false positive
    rate on the paired clean split exceeds the budget. Each fold is therefore
    standardised against the statistics of the other folds, by position, and the
    result is negated here exactly as scale_up_scores negates at its boundary.
    """
    assert spc.shape == predicted_labels.shape == true_labels.shape, (
        spc.shape,
        predicted_labels.shape,
        true_labels.shape,
    )
    fold_of = torch.arange(spc.numel()) % folds  # (N,)
    nspc = torch.empty(spc.numel(), dtype=torch.float32)  # (N,)
    for fold in range(folds):
        held_out = fold_of == fold  # (N,)
        class_means, class_stds = fit_class_spc_statistics(
            spc[~held_out], true_labels[~held_out], num_classes
        )
        nspc[held_out] = standardize_spc(
            spc[held_out], predicted_labels[held_out], class_means, class_stds
        ).float()

    scores = -nspc  # (N,), low means poisoned
    return scores


def standardize_spc(
    spc: torch.Tensor,
    predicted_labels: torch.Tensor,
    class_means: torch.Tensor,
    class_stds: torch.Tensor,
) -> torch.Tensor:
    """Eq. (4): NSPC per sample, shape (N,), still high for poisoned.

    Indexed by the predicted label yhat = C(x), exactly as Eq. (4) writes it, so a
    sample is compared against the clean SPC distribution of the class the model
    put it in rather than the class it truly belongs to. That is the only reading
    available at inference time, where no true label exists.
    """
    means = class_means[predicted_labels]  # (N,)
    stds = class_stds[predicted_labels]  # (N,)

    nspc = (spc - means) / stds
    return nspc


@torch.inference_mode()
def scale_up_scores(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mean: tuple[float, ...],
    std: tuple[float, ...],
    scales: tuple[int, ...] = PAPER_SCALES,
    use_bfloat16: bool = True,
    class_means: torch.Tensor | None = None,
    class_stds: torch.Tensor | None = None,
) -> torch.Tensor:
    """SCALE-UP score per sample, shape (N,), low meaning poisoned.

    Data-free when class_means and class_stds are None, which is Eq. (2) alone and
    the paper's headline method. Data-limited when both are supplied, which adds
    the Eq. (4) standardization against the clean statistics fitted by
    fit_class_spc_statistics.

    Negated at this boundary. SCALE-UP's rule is "backdoor if SPC(x) > T", so the
    paper's statistic is high for poisoned, the opposite of PSU's convention, and
    returning it unnegated would produce a well-formed, exactly inverted detector.
    """
    spc, predicted_labels, _ = spc_scores(
        model, loader, device, mean, std, scales, use_bfloat16
    )
    if spc.numel() == 0:
        return torch.empty(0)

    supplies_statistics = class_means is not None and class_stds is not None
    if supplies_statistics:
        spc = standardize_spc(spc, predicted_labels, class_means, class_stds)

    scores = -spc.float()  # (N,), low means poisoned
    return scores
