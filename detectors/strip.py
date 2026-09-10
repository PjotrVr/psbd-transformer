"""STRIP: STRong Intentional Perturbation (Gao et al., ACSAC 2019).

Paper: "STRIP: A Defence Against Trojan Attacks on Deep Neural Networks",
arXiv:1902.06531. The statistic is Section IV-D, Equations (2), (3) and (4).

    original form
        H_n   = - sum_{i=1}^{M} y_i * log2(y_i)                     Eq. (2)
        H_sum = sum_{n=1}^{N} H_n                                   Eq. (3)
        H     = (1 / N) * H_sum                                     Eq. (4)

    descriptive form
        blend_entropy = entropy of the prediction on a superimposed copy
        strip_score   = mean blend_entropy over the N superimposed copies

N is the number of clean images superimposed on each input, M the number of
classes and y_i the softmax probability of class i on a superimposed copy.

Mechanism. A clean input's class evidence is destroyed once a second image is
laid over it, so predictions scatter and entropy is high. A trigger survives the
superimposition and keeps dragging the prediction to the target class, so entropy
stays low. The paper flags an input whose H falls below a percentile of the clean
entropy distribution, the same quantile-of-clean-validation rule
defences.decision applies here.

Data requirement: N clean images, drawn from the shared clean validation split.
Forward-pass cost: N per input. The paper defaults to N = 100 and later reports
10 as sufficient. This port uses 8, the value every recorded number in this repo
was produced with.

Deviations from the paper, each recorded in full in docs/detectors/strip.md:

  1. Entropy in nats rather than bits, a constant factor that moves no ranking,
     AUROC or quantile position, only the printed threshold.
  2. Superimposition is a sum in [0, 1] pixel space followed by saturation,
     matching cv2.addWeighted on uint8 arrays. The loader delivers normalized
     tensors, so the round trip is denormalize, add, saturate, renormalize.
  3. The overlay set is fixed across every scored input, which removes overlay
     choice as a source of per-sample variance.
"""

import torch
import torch.nn as nn
from lightning import seed_everything
from torch.utils.data import DataLoader

from defences.inference import forward_probs

# Guards the log when a class gets probability 0 after softmax underflow.
PROBABILITY_FLOOR = 1e-12

# N in Eq. (3). The paper's own default is 100, reduced to 10 in its later
# analysis. 8 is this repo's standing choice and is kept so the ported numbers
# stay comparable to everything already recorded.
DEFAULT_NUM_OVERLAYS = 8


def blend_entropy(probs: torch.Tensor) -> torch.Tensor:
    """Shannon entropy per row, in nats, numerically safe at p=0, shape (N,).

    original form
        H(p) = - sum_c p_c * log(p_c)
    descriptive form
        entropy = negative sum over classes of probability times its log
    """
    entropy = -(probs.clamp_min(PROBABILITY_FLOOR).log() * probs).sum(dim=1)  # (N,)
    return entropy


def normalization_buffers(
    mean: tuple[float, ...],
    std: tuple[float, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The dataset statistics as (1, channels, 1, 1) tensors, broadcastable over a batch.

    Shared by every detector that leaves normalized space to work on pixels, so
    the denormalize, perturb, renormalize round trip is written once.
    """
    mean_tensor = torch.tensor(mean, device=device, dtype=dtype).view(1, -1, 1, 1)
    std_tensor = torch.tensor(std, device=device, dtype=dtype).view(1, -1, 1, 1)
    return mean_tensor, std_tensor


@torch.inference_mode()
def strip_scores(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    overlay_images: torch.Tensor,
    mean: tuple[float, ...],
    std: tuple[float, ...],
    use_bfloat16: bool = True,
    seed: int = 0,
    num_overlays: int = DEFAULT_NUM_OVERLAYS,
) -> torch.Tensor:
    """STRIP entropy per sample, shape (N,), low meaning poisoned.

    overlay_images is a (>=num_overlays, C, H, W) batch of clean images, normally
    the output of collect_overlay_batch on the clean validation split.

    Returned as the raw entropy, not negated. STRIP's claim is that a triggered
    input has low entropy under superimposition, and low already means poisoned in
    the shared convention, so negating would invert the detector.
    """
    model.eval()
    seed_everything(seed)
    overlays = overlay_images[:num_overlays].to(device)  # (num_overlays, C, H, W)

    batch_scores = []
    for images, _ in loader:
        images = images.to(device)  # (batch, C, H, W)
        mean_tensor, std_tensor = normalization_buffers(
            mean, std, images.device, images.dtype
        )
        pixels = (images * std_tensor + mean_tensor).clamp(0.0, 1.0)
        overlay_pixels = (overlays * std_tensor + mean_tensor).clamp(0.0, 1.0)
        summed_entropy = torch.zeros(images.size(0), device=device)  # (batch,)
        for overlay in overlay_pixels:
            superimposed = (pixels + overlay.unsqueeze(0)).clamp(0.0, 1.0)
            blended = (superimposed - mean_tensor) / std_tensor
            probs = forward_probs(model, blended, device, use_bfloat16)
            summed_entropy += blend_entropy(probs)
        batch_scores.append((summed_entropy / len(overlays)).cpu())  # Eq. (4)

    if not batch_scores:
        return torch.empty(0)

    scores = torch.cat(batch_scores).float()  # (N,)
    return scores


@torch.inference_mode()
def collect_overlay_batch(loader: DataLoader, count: int, seed: int) -> torch.Tensor:
    """The first count clean images as the superimposition set, (count, C, H, W).

    Taken from the clean validation split the defender already holds for
    thresholding, so STRIP is given exactly the same data budget as every other
    method here and none gets an advantage from seeing more.
    """
    seed_everything(seed)

    collected = []
    gathered = 0
    for images, _ in loader:
        collected.append(images)
        gathered += images.size(0)
        if gathered >= count:
            break

    if not collected:
        return torch.empty(0)

    overlays = torch.cat(collected)[:count]  # (count, C, H, W)
    return overlays
