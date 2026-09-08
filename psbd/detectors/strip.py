"""STRIP: STRong Intentional Perturbation (Gao et al., ACSAC 2019).

Paper: "STRIP: A Defence Against Trojan Attacks on Deep Neural Networks",
arXiv:1902.06531. The statistic is Section IV-D, Equations (2), (3) and (4).

    original form
        H_n   = - sum_{i=1}^{M} y_i * log2(y_i)                     Eq. (2)
        H_sum = sum_{n=1}^{N} H_n                                   Eq. (3)
        H     = (1 / N) * H_sum                                     Eq. (4)

    descriptive form
        blend_entropy = entropy of the prediction on one superimposed copy
        strip_score   = mean blend_entropy over the N superimposed copies

| Symbol | Meaning |
|---|---|
| x | the suspicious input |
| x^{p_n} | the n-th perturbed copy, x superimposed with a clean image |
| N | number of superimposed copies per input |
| M | number of classes |
| y_i | softmax probability of class i on the perturbed copy |
| H_n | entropy of the n-th perturbed copy, Eq. (2) |
| H | the normalized entropy, the detection statistic, Eq. (4) |

Mechanism. A clean input's class evidence is destroyed once a second image is
laid on top of it, so predictions scatter across classes and entropy is high. A
trigger survives the superimposition and keeps dragging the prediction to the
attacker's target class, so entropy stays low. The paper's decision rule flags an
input whose H falls below a percentile of the clean-input entropy distribution
chosen at a target false rejection rate (1% FRR in their experiments), which is
the same quantile-of-clean-validation rule psbd.decision applies here.

Data requirement: needs N clean samples, drawn from the shared clean validation
split so no method sees more data than another.
Forward-pass cost: N per input, the number of overlays. The paper's default is
N = 100 and it later reports N = 10 as sufficient. This port defaults to 8, which
is what the pre-existing implementation in defences/baselines.py used and what
every already-recorded number in this repo was produced with.

Deviations from the paper, all stated rather than silently absorbed:

  1. Entropy in nats, not bits. Eq. (2) uses log2 and this uses the natural log.
     The 2 differ by the constant factor log(2), so no ranking, no AUROC, and no
     quantile position changes. Only the printed threshold value is scaled.
  2. Superimposition is a sum in [0, 1] PIXEL space, then saturated, matching
     cv2.addWeighted(background, 1, overlay, 1, 0) on uint8 arrays: both weights
     1, and OpenCV saturating-casts the result at 255. An earlier version of this
     port summed the 2 already-normalized tensors instead. That is not the same
     operation: adding in normalized space gives
     (p1 - m)/s + (p2 - m)/s = (p1 + p2 - 2m)/s, which is the pixel-space sum
     displaced by a further -m/s per channel (2.43 units on CIFAR-10 channel 0),
     and it also skipped the saturation the reference performs. The loader
     delivers normalized tensors, so the round trip is denormalize, add,
     saturate, renormalize.
  3. The overlay set is fixed across all scored inputs rather than resampled per
     input. Every input then faces the same perturbation set, which removes
     overlay choice as a source of per-sample variance in the comparison.
"""

import torch
import torch.nn as nn
from lightning import seed_everything
from torch.utils.data import DataLoader

from ..inference import forward_probs

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

    Returned as the raw entropy, NOT negated. STRIP's whole claim is that a
    triggered input has LOW entropy under superimposition, and the shared
    detection convention is that LOW means poisoned, so the 2 already agree and
    negating would invert the detector. That mistake was made in this repo first
    and showed up as AUROC 0.000, perfect separation with the sign reversed, which
    is what the two-sided field in detection_report exists to surface.
    """
    model.eval()
    seed_everything(seed)
    overlays = overlay_images[:num_overlays].to(device)  # (num_overlays, C, H, W)

    batch_scores = []
    for images, _ in loader:
        images = images.to(device)  # (batch, C, H, W)
        mean_tensor, std_tensor = _normalization_buffers(
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
def collect_overlay_batch(
    loader: DataLoader, count: int, seed: int
) -> torch.Tensor:
    """The first `count` clean images as the superimposition set, (count, C, H, W).

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
