"""SCALE-UP: scaled prediction consistency (Guo et al., ICLR 2023).

Paper: "SCALE-UP: An Efficient Black-box Input-level Backdoor Detection via
Analyzing Scaled Prediction Consistency", arXiv:2302.03251, OpenReview
o0LFPcoFKnr. The statistic is Section 4.2, Equation (2). The data-limited variant
is Section 4.3, Equations (3) and (4).

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

| Symbol | Meaning |
|---|---|
| x | the suspicious input, pixel values in [0, 1] |
| C(.) | the deployed classifier's predicted label, argmax only |
| S | the scaling set, a set of integer multipliers |
| n | one scaling factor, n in S |
| I{.} | indicator, 1 when the condition holds |
| SPC(x) | scaled prediction consistency, Eq. (2) |
| X_i | the defender's benign samples of class i |
| mu_i, sigma_i | mean and standard deviation of SPC over X_i, Eq. (3) |
| yhat | C(x), the predicted label of the unamplified input |
| NSPC(x) | the standardized statistic of the data-limited variant, Eq. (4) |
| T | the detection threshold, "backdoor if SPC(x) > T" |

Mechanism. Amplifying every pixel toward saturation destroys the class evidence
of a benign image, so its predicted label moves. A trigger is high-contrast and
survives the amplification, so a poisoned image keeps being classified as the
attacker's target and its predictions stay consistent. Paper Theorem 1 proves the
limiting case for an RBF kernel regressor at a 50% poisoning rate.

Two variants, both implemented:

  data-free (the default here). Eq. (2) alone. The paper's headline contribution
    and the claim its abstract makes, since it needs only query access and the
    predicted label.
  data-limited. Eq. (2) standardized by Eqs. (3) and (4) against per-class clean
    statistics. Presented as an optional refinement, worth roughly +0.005 AUROC
    on average in the paper's own Tables 1 and 2.

Data requirement: data-free needs 0 clean samples. Data-limited needs the clean
validation split (the paper budgets 100 benign samples per class).
Forward-pass cost: |S| + 1 per input, 6 at the default scaling set. |S| amplified
passes for Eq. (2) plus 1 unamplified pass for C(x). The paper counts only the
|S| extra queries, since C(x) is the prediction the deployment was going to make
anyway.

Deviations from the paper, all stated rather than silently absorbed:

  1. Scaling set. The paper writes S = {3, 5, 7, 9, 11} in Section 4.2, but
     introduces it with "e.g." and never restates it in the experimental
     settings. The authors' released code uses range(1, 12), so S = {1, ..., 11},
     which puts n = 1 inside the average where it is trivially consistent and
     lifts the floor of SPC from 0 to 1/11. Those are 2 different statistics on 2
     different supports. This port defaults to the paper's set and offers the
     code's set as OFFICIAL_CODE_SCALES, so whichever produced a number is
     recorded rather than inherited.
  2. Per-class statistics on a shared budget. Eq. (3) is per class and the paper
     budgets 100 benign samples per class. Every method here shares the single
     2000-sample clean validation split, which on CIFAR-100 is about 20 samples
     per class. A class with fewer than MIN_CLASS_SAMPLES falls back to the
     pooled mean and standard deviation, which the paper does not specify. Both
     published third-party ports sidestep this by using 1 global mean and
     standard deviation for every class, which is a different method from Eq. (4).
  3. Calibration reference. Eq. (2) always measures consistency against C(x),
     the model's own prediction. Both third-party ports instead compare against
     the ground-truth validation label when computing Eq. (3), which disagrees
     with Eq. (2) exactly on the samples the model gets wrong. This port uses
     C(x) throughout. The grouping into X_i uses the validation sample's true
     class, which is what Eq. (3) says.
  4. No input noise. The authors' released code adds 0.02 * U[0, 1) to every test
     sample before scaling. That appears in no equation, has no ablation, and is
     not reproduced here.
  5. No prediction-correctness masking. The third-party ports drop every sample
     whose prediction disagrees with its dataset label, which on the poisoned
     side keeps only successfully attacked images. This project already owns that
     decision through psbd.poisoning.AttackSuccessSet, so the detector scores
     every sample the loader serves and the eligibility rule stays in one place.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..inference import forward_probs

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

    Skipping the round trip and scaling the normalized tensor directly would
    amplify the dataset mean as though it were signal, and the clip, which is
    where the whole nonlinearity of the method lives, would land at the wrong
    place. Of the 3 published implementations only backdoor-toolbox gets this
    ordering right.
    """
    pixels = (images * std + mean).clamp(0.0, 1.0)  # (batch, C, H, W)
    amplified = (pixels * float(factor)).clamp(0.0, 1.0)

    renormalized = (amplified - mean) / std
    return renormalized


def _normalization_buffers(
    mean: tuple[float, ...],
    std: tuple[float, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The dataset statistics shaped to broadcast over (batch, C, H, W)."""
    mean_tensor = torch.tensor(mean, device=device, dtype=dtype).view(1, -1, 1, 1)
    std_tensor = torch.tensor(std, device=device, dtype=dtype).view(1, -1, 1, 1)
    return mean_tensor, std_tensor


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
    """Raw SPC per sample plus the two things the data-limited variant needs.

    Returns (spc, predicted_labels, true_labels), each (N,). spc is Eq. (2) in
    [0, 1], HIGH for poisoned. predicted_labels is C(x) and true_labels is
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
        mean_tensor, std_tensor = _normalization_buffers(
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

    Two fallbacks to the pooled statistics, neither of which the paper specifies
    because neither arises at its budget of 100 samples per class.

      Too few samples. The shared 2000-sample budget gives roughly 20 per class on
        CIFAR-100 and can give 0 for a class the split happened to miss. Dividing
        by the standard deviation of 1 sample manufactures a score out of nothing.

      Degenerate spread. SPC is a fraction over |S| scaled copies, so it lives on
        a grid of 1/|S| and takes only |S| + 1 distinct values. A class whose 20
        clean samples all landed on the same grid point has sigma_i exactly 0,
        which is not a measurement of low variance but an artifact of quantization
        against a small sample. Left alone it is clamped to STD_FLOOR and Eq. (4)
        returns z-scores of order 1e5 for that class alone, which does not change
        AUROC but makes the score scale meaningless and any absolute threshold
        unusable. Observed on vit_cifar100 with the shared budget, not a
        hypothetical.
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


def standardize_spc(
    spc: torch.Tensor,
    predicted_labels: torch.Tensor,
    class_means: torch.Tensor,
    class_stds: torch.Tensor,
) -> torch.Tensor:
    """Eq. (4): NSPC per sample, shape (N,), still HIGH for poisoned.

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

    NEGATED at this boundary. SCALE-UP's decision rule is "backdoor if SPC(x) > T",
    so the paper's statistic is HIGH for poisoned, the opposite of PSU's
    convention. Returning it unnegated would produce a confident, well-formed,
    exactly-inverted detector, which is the single most likely bug in this file.
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
