"""IBD-PSC: parameter-oriented scaling consistency (Hou et al., ICML 2024).

Paper: "IBD-PSC: Input-level Backdoor Detection via Parameter-oriented Scaling
Consistency", arXiv:2405.09786. Model amplification is Section 4.3, Equation (2),
adaptive layer selection is Equation (3) and Algorithm 1, and the score is
Section 4.4, Equation (4).

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
                             unamplified model predicted

Defaults follow Section 5.1: omega 1.5, n 5, xi 0.6. L is the number of
amplifiable normalization layers and depends on the architecture.

Mechanism. Amplifying the affine parameters of the layers nearest the head
inflates every logit, which pushes a benign prediction off its class because the
class evidence is a comparison between similarly sized logits. A backdoor maps its
trigger to the target through a far larger margin, so a poisoned input keeps its
label and its probability. PSC measures how well an amplified model retains the
original model's decision, not how well the ensemble members agree.

Data requirement: the clean validation split, with labels, for Algorithm 1, since
Eq. (3) is a top-1 error against ground truth. The score itself needs none.
Forward-pass cost: n + 1 per input, 6 at the default ensemble size, plus a fixed
cost of up to L passes over the validation split for layer selection.

The substantive deviation is that this port amplifies LayerNorm. The paper and
its released code scale BatchNorm2d only, and a ViT or Swin has none, so the
published method is not runnable on these architectures. The substitution is
exact at the level of what Eq. (2) does to a layer's output, since both
normalize first and apply the affine map second:

      omega*gamma * x_hat + omega*beta = omega * (gamma * x_hat + beta)

What does not carry over is the depth claim: a ViT block holds 2 LayerNorms on
the 2 branch inputs of a residual stream, so amplifying one scales a branch
rather than the stream and the effect per layer is weaker than a BatchNorm
scaling in a ConvNet. Algorithm 1 absorbs that by selecting k from measured clean
error, but the resulting k is not comparable to a published k.

Further deviations, each recorded in full in docs/detector-ports.md:

  1. The ensemble sums over k..k+n-1 as Eq. (4) writes it. The released code
     amplifies 1 more layer at every position.
  2. Algorithm 1 tests i = 1..L and falls back to k = L when no i crosses xi,
     the value the algorithm holds at loop exit. The released code never tests
     the all-layers case and returns None.
  3. When the window k..k+n-1 runs past L, only the members with i <= L are
     kept, so a late k gives a smaller ensemble rather than an invalid one.
  4. The clean data is the shared validation split, so every method here sees
     the same budget.
  5. Amplified parameters are written in place and restored from saved clones,
     which is exact and allocates no second model. The released code deep-copies
     the model once per ensemble member per batch.
"""

from contextlib import contextmanager

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from defences.inference import forward_probs

# Section 5.1 defaults, identical in the paper and in the released code.
DEFAULT_SCALING_FACTOR = 1.5  # omega
DEFAULT_ENSEMBLE_SIZE = 5  # n
DEFAULT_ERROR_THRESHOLD = 0.6  # xi
DEFAULT_DETECTION_THRESHOLD = 0.9  # T, unused here since scoring is threshold-free


def amplifiable_norm_layers(model: nn.Module) -> list[nn.LayerNorm]:
    """Every affine LayerNorm in the model, deepest first, the Eq. (2) targets.

    Definition order is reversed, as the released code does, so element 0 is the
    layer nearest the head and the first i elements are "the last i layers" of
    Eq. (2). Definition order equals execution order for torchvision's ViT and
    Swin, so the reversal is depth ordering here, which would not hold for an
    architecture that declares its modules out of forward order. A LayerNorm with
    elementwise_affine=False has nothing to scale and is skipped.
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
def amplified_parameters(ordered_layers: list[nn.LayerNorm], scaling_factor: float):
    """A context holding every layer's gamma and beta, restored exactly on exit.

    Yields a setter taking a layer count, which writes the Eq. (2) amplification
    for the first count layers and leaves the rest at their loaded values. Each
    call writes from the saved originals, so walking the ensemble never compounds,
    and restoring by copy from a clone is exact where multiplying back by 1 / omega
    would leave float drift in the deployed model.
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

    Ground truth comes from the loader, which is why Algorithm 1 needs a labelled
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

    Returns (k, error_rates), where error_rates[i - 1] is the Eq. (3) value at i
    amplified layers, up to and including the one that crossed. The trace comes
    back because k is the single data-dependent quantity in the method, and a k
    of 1 or of L means the amplification was mis-scaled for the architecture.

    This is the layer-selection rule the operator port defences.operators.GainScale
    does not have. Algorithm 1 finds the depth at which benign accuracy starts
    collapsing and scores there, because that is where benign and poisoned
    confidence separate most.
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
    """Eq. (4): raw PSC per sample, shape (N,), high for poisoned.

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

    Negated at this boundary. The paper's rule is "poisoned if PSC(x) > T", so its
    statistic is high for poisoned, the opposite of PSU's convention, and returning
    it unnegated would produce a well-formed, exactly inverted detector.
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
