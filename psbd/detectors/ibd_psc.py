"""IBD-PSC: parameter-oriented scaling consistency (Hou et al., ICML 2024).

Paper: "IBD-PSC: Input-level Backdoor Detection via Parameter-oriented Scaling
Consistency", arXiv:2405.09786, PMLR v235 hou24a. Model amplification is
Section 4.3, Equation (2). Adaptive layer selection is Section 4.3, Equation (3)
and Algorithm 1. The score is Section 4.4, Equation (4).

    original form
        F_hat^omega_k = FC . f_hat^omega_L . ... . f_hat^omega_{L-k+1}
                        . f_{L-k} . ... . f_1                          Eq. (2)
        with  gamma_hat = omega * gamma,  beta_hat = omega * beta

        eta = (1 / |D_r|) * sum_{(x,y) in D_r}
              I( argmax( F_hat^omega_k(x) ) != y )                     Eq. (3)

        k   = smallest k in 1..L with eta > xi                         Algorithm 1

        PSC(x) = (1 / n) * sum_{i=k}^{k+n-1} F_hat^omega_i(x)_{y'}     Eq. (4)
        with  y' = argmax( F(x) )

    descriptive form
        amplified_model(i) = the model with the last i normalization layers'
                             scale and shift both multiplied by omega
        clean_error(i)     = top-1 error of amplified_model(i) on the clean
                             validation split
        start_count        = smallest i whose clean_error exceeds xi
        psc(image)         = mean over the ensemble of the softmax probability
                             each amplified model assigns to the label the
                             UNAMPLIFIED model predicted

| Symbol | Meaning | Default |
|---|---|---|
| F | the deployed, unmodified model, softmax output | |
| f_i | the i-th hidden layer | |
| L | number of amplifiable normalization layers | architecture-dependent |
| gamma, beta | a normalization layer's scale and shift | |
| omega | the amplification factor | 1.5 |
| k | number of layers amplified in the first ensemble member | from Algorithm 1 |
| n | ensemble size | 5 |
| xi | clean top-1 error rate that triggers the break in Algorithm 1 | 0.6 |
| eta | clean top-1 error rate of one amplified model, Eq. (3) | |
| D_r | the defender's local benign set | 100 samples in the paper |
| y' | argmax F(x), the unamplified model's own prediction | |
| T | detection threshold, "poisoned if PSC(x) > T" | 0.9 |

Mechanism. Amplifying the affine parameters of the layers nearest the head
inflates every logit, which pushes a benign prediction off its true class because
the class evidence is a comparison between similarly sized logits. A backdoor
maps its trigger to the target class through a far larger margin, so a poisoned
input keeps its predicted label and its probability under the same amplification.
PSC therefore measures how well an amplified model retains the ORIGINAL model's
decision, not how well the ensemble members agree with each other.

Data requirement: needs the clean validation split, WITH labels, for Algorithm 1.
The paper budgets 100 benign samples. Nothing else uses clean data. Labels are
required because Eq. (3) is a top-1 error rate against ground truth.
Forward-pass cost: n + 1 per input, 6 at the default ensemble size. n amplified
passes for Eq. (4) plus 1 unamplified pass for y'. Layer selection is a fixed
one-off cost of up to L passes over the validation split, not a per-input cost.

The ViT deviation, which is the substantive one:

  The paper scales BatchNorm2d and only BatchNorm2d. Its released code in
  BackdoorBox matches, filtering on isinstance(module, torch.nn.BatchNorm2d). A
  ViT or a Swin contains no BatchNorm at all, so count_BN_layers returns 0,
  sorted_indices is empty, Algorithm 1 loops over range(1, 0), and start_index
  comes back None. IBD-PSC as published is not runnable on this project's
  architectures.

  This port amplifies nn.LayerNorm instead. The substitution is exact rather than
  approximate at the level of what Eq. (2) does to a layer's output:

        omega*gamma * x_hat + omega*beta
      = omega * (gamma * x_hat + beta)
      = omega * (the layer's original output)

  which holds for LayerNorm exactly as it holds for BatchNorm, because both
  normalize first and apply the affine map second. What does NOT carry over is
  the architectural claim behind the choice: the paper's L is a count of BN
  layers, one per convolutional stage, whereas a ViT block contains 2 LayerNorms
  sitting on the 2 branch inputs of a residual stream. Amplifying one of those
  scales a branch, not the stream, so the effect on the logits is weaker per
  layer than a BN scaling is in a ConvNet. Algorithm 1 absorbs that difference by
  construction, since it selects k from measured clean error rather than from a
  fixed depth, but the resulting k is not comparable to a published k.

Further deviations, all stated rather than silently absorbed:

  1. Layer count in the ensemble. Eq. (4) sums over i = k .. k+n-1 amplified
     layers. The released code amplifies sorted_indices[:layer_index+1], so
     k+1 .. k+n, 1 more layer at every position than the equation. This port
     follows the equation.
  2. Algorithm 1's range. The paper loops i = 1 to L. The released code loops
     range(1, layer_num), so it never tests the all-layers configuration and
     returns None when the error rate never crosses xi, which then crashes
     downstream with no message. This port follows the paper, tests i = 1..L, and
     falls back to k = L when no i crosses, which is the value Algorithm 1 holds
     at loop exit.
  3. Ensemble clamping. When k is close to L the window k..k+n-1 runs past L,
     which is undefined. The paper does not address it because its BN counts are
     large. This port keeps only the members with i <= L, so a late k gives a
     smaller ensemble rather than an invalid one, and records how many members
     were actually used.
  4. Data budget. The paper states 100 benign samples. The released demo uses
     2000. This port uses the shared clean validation split, so every method here
     sees the same data and none is advantaged.
  5. No deep copies. The released code calls copy.deepcopy(self.model) once per
     ensemble member per batch, which is 5 full model copies per batch. This port
     writes the amplified parameters in place and restores them from saved
     clones, which is numerically exact and does not allocate a second model.
"""

from contextlib import contextmanager

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..inference import forward_probs

# Section 5.1 defaults, identical in the paper and in the released code.
DEFAULT_SCALING_FACTOR = 1.5  # omega
DEFAULT_ENSEMBLE_SIZE = 5  # n
DEFAULT_ERROR_THRESHOLD = 0.6  # xi
DEFAULT_DETECTION_THRESHOLD = 0.9  # T, unused here since scoring is threshold-free


def amplifiable_norm_layers(model: nn.Module) -> list[nn.LayerNorm]:
    """Every affine LayerNorm in the model, DEEPEST FIRST, the Eq. (2) targets.

    Definition order from named_modules is reversed, which is what the released
    code does with list(reversed(range(layer_num))), so element 0 is the layer
    nearest the classifier head and taking the first i elements is "the last i
    layers" of Eq. (2).

    Definition order equals execution order for both torchvision ViT and Swin, so
    the reversal really is depth ordering here. It would not be for an
    architecture that declares its modules out of forward order, which is a latent
    trap the released code shares.

    A LayerNorm constructed with elementwise_affine=False has no gamma or beta to
    scale, so Eq. (2) is undefined for it and it is skipped rather than counted.
    """
    affine_norms = [
        module
        for module in model.modules()
        if isinstance(module, nn.LayerNorm) and module.weight is not None
    ]
    if not affine_norms:
        raise ValueError(
            "no affine LayerNorm modules found, so IBD-PSC has nothing to amplify. "
            "The published method targets BatchNorm2d and this port targets "
            "LayerNorm; a model with neither cannot be scored."
        )

    deepest_first = list(reversed(affine_norms))
    return deepest_first


@contextmanager
def amplified_parameters(
    ordered_layers: list[nn.LayerNorm], scaling_factor: float
):
    """Hold every layer's gamma and beta, restoring them exactly on exit.

    Yields a setter taking a layer count, which writes the Eq. (2) amplification
    for the first `count` layers of ordered_layers and leaves the rest at their
    loaded values. Calling it repeatedly walks the ensemble without ever
    compounding, because each call writes from the saved originals rather than
    multiplying what is already there. Restoring by copy_ from a clone is exact,
    unlike multiplying back by 1 / omega, which would leave float drift in the
    deployed model after scoring.
    """
    saved = [
        (layer.weight.detach().clone(), layer.bias.detach().clone())
        for layer in ordered_layers
    ]

    @torch.no_grad()
    def amplify_first(count: int) -> None:
        for position, (layer, (weight, bias)) in enumerate(zip(ordered_layers, saved)):
            factor = scaling_factor if position < count else 1.0
            layer.weight.copy_(weight * factor)
            layer.bias.copy_(bias * factor)

    try:
        yield amplify_first
    finally:
        with torch.no_grad():
            for layer, (weight, bias) in zip(ordered_layers, saved):
                layer.weight.copy_(weight)
                layer.bias.copy_(bias)


@torch.inference_mode()
def clean_error_rate(
    model: nn.Module,
    validation_loader: DataLoader,
    device: torch.device,
    use_bfloat16: bool,
) -> float:
    """Eq. (3): top-1 error of the currently amplified model on the clean split.

    Ground truth comes from the loader, which is why Algorithm 1 needs a LABELLED
    benign set while the score itself needs none.
    """
    model.eval()

    wrong = 0
    total = 0
    for images, labels in validation_loader:
        probs = forward_probs(model, images, device, use_bfloat16)  # (batch, classes)
        predicted = probs.argmax(dim=1).cpu()
        wrong += int((predicted != labels.cpu().long()).sum().item())
        total += labels.size(0)

    if total == 0:
        raise ValueError("clean validation split is empty, so Eq. (3) is undefined")

    error_rate = wrong / total
    return error_rate


def select_start_layer_count(
    model: nn.Module,
    validation_loader: DataLoader,
    device: torch.device,
    ordered_layers: list[nn.LayerNorm],
    scaling_factor: float = DEFAULT_SCALING_FACTOR,
    error_threshold: float = DEFAULT_ERROR_THRESHOLD,
    use_bfloat16: bool = True,
) -> tuple[int, list[float]]:
    """Algorithm 1: the smallest layer count whose clean error exceeds xi.

    Returns (k, error_rates), where error_rates[i - 1] is the Eq. (3) value
    measured at i amplified layers, up to and including the one that crossed. The
    trace is returned rather than discarded because k is the one data-dependent
    quantity in the method and a k of 1, or a k equal to L, both mean the
    amplification was mis-scaled for this architecture rather than that the model
    is unusual.

    This is the layer-selection rule the operator port in psbd.operators.GainScale
    does not have. Without it the amplification is applied everywhere at a fixed
    rate, which asks a different question: Algorithm 1 exists to find the depth at
    which benign accuracy starts collapsing, and scores at that depth, because
    that is where benign and poisoned confidence separate most.
    """
    total_layers = len(ordered_layers)
    error_rates: list[float] = []

    with amplified_parameters(ordered_layers, scaling_factor) as amplify_first:
        for count in range(1, total_layers + 1):
            amplify_first(count)
            error_rate = clean_error_rate(
                model, validation_loader, device, use_bfloat16
            )
            error_rates.append(error_rate)
            if error_rate > error_threshold:
                return count, error_rates

    # Algorithm 1 assigns k = i before testing, so the value it holds after a loop
    # that never breaks is L. The released code instead returns None here and
    # crashes on the next call.
    return total_layers, error_rates


@torch.inference_mode()
def psc_scores(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    ordered_layers: list[nn.LayerNorm],
    start_layer_count: int,
    scaling_factor: float = DEFAULT_SCALING_FACTOR,
    ensemble_size: int = DEFAULT_ENSEMBLE_SIZE,
    use_bfloat16: bool = True,
) -> torch.Tensor:
    """Eq. (4): raw PSC per sample, shape (N,), HIGH for poisoned.

    y' is taken from the unamplified model once per batch and never recomputed per
    ensemble member, which is what makes this a measure of retained agreement with
    the deployed model rather than of internal ensemble agreement.

    Not negated. Sign correction happens once, in ibd_psc_scores, so this stays
    the paper's statistic and can be compared to a published number directly.
    """
    model.eval()
    total_layers = len(ordered_layers)
    member_counts = [
        count
        for count in range(start_layer_count, start_layer_count + ensemble_size)
        if count <= total_layers
    ]
    if not member_counts:
        raise ValueError(
            f"start_layer_count {start_layer_count} leaves no ensemble member at or "
            f"below the {total_layers} available layers"
        )

    batch_scores = []
    with amplified_parameters(ordered_layers, scaling_factor) as amplify_first:
        for images, _ in loader:
            images = images.to(device)  # (batch, C, H, W)

            amplify_first(0)
            baseline_probs = forward_probs(model, images, device, use_bfloat16)
            original_labels = baseline_probs.argmax(dim=1)  # (batch,) = y'

            retained = torch.zeros(images.size(0), device=device)  # (batch,)
            for count in member_counts:
                amplify_first(count)
                probs = forward_probs(model, images, device, use_bfloat16)
                retained += probs.gather(1, original_labels.view(-1, 1)).squeeze(1)

            batch_scores.append((retained / len(member_counts)).cpu())

    if not batch_scores:
        return torch.empty(0)

    psc = torch.cat(batch_scores).float()  # (N,)
    return psc


def ibd_psc_scores(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    ordered_layers: list[nn.LayerNorm],
    start_layer_count: int,
    scaling_factor: float = DEFAULT_SCALING_FACTOR,
    ensemble_size: int = DEFAULT_ENSEMBLE_SIZE,
    use_bfloat16: bool = True,
) -> torch.Tensor:
    """IBD-PSC score per sample, shape (N,), low meaning poisoned.

    NEGATED at this boundary. The paper's rule is "poisoned if PSC(x) > T", so its
    statistic is HIGH for poisoned, the opposite of PSU's convention. Returning it
    unnegated would produce a confident, well-formed, exactly-inverted detector.
    """
    psc = psc_scores(
        model,
        loader,
        device,
        ordered_layers,
        start_layer_count,
        scaling_factor,
        ensemble_size,
        use_bfloat16,
    )
    if psc.numel() == 0:
        return torch.empty(0)

    scores = -psc.float()  # (N,), low means poisoned
    return scores
