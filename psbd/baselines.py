"""Baseline backdoor-input detectors, for comparison against PSBD.

PSBD is an input-level, test-time detector: it takes one suspicious input and
decides whether it carries a trigger, using only the model and a clean validation
set. A comparison is only meaningful against detectors that solve the same problem
under the same assumptions, so the 2 here were chosen for that reason rather than
for being the most cited.

STRIP (Gao et al., 2019). Superimpose the suspicious input with N clean images and
measure the entropy of the resulting predictions. A clean input's prediction is
fragile under superimposition, so entropy is high. A triggered input keeps being
dragged to the target class by the trigger that survives blending, so entropy is
low. Same decision direction as PSU (low means poisoned) and the same threshold
convention, which makes it the closest available comparison.

Confidence (no citation, it is the null model). The no-dropout probability of the
predicted class, nothing else. Included because it costs 1 forward pass and an
adversarial review found it beats PSBD on the benign control, so any method
claiming to detect backdoors should be shown to beat it.

Deliberately NOT included: Spectral Signatures, Activation Clustering, and SCAn all
score a whole poisoned TRAINING set by clustering its representations. They need
access to the full training pool and produce a partition, not a per-input decision,
so putting their numbers in the same table as PSBD's would compare 2 different
tasks.

Both detectors here return a score per sample where LOW means poisoned, matching
PSU, so psbd.decision.detection_report applies unchanged.
"""

import torch
import torch.nn as nn
from lightning import seed_everything
from torch.utils.data import DataLoader

from .inference import forward_probs

# Guards the log when a class gets probability 0 after softmax underflow.
PROBABILITY_FLOOR = 1e-12


@torch.inference_mode()
def confidence_scores(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_bfloat16: bool,
) -> torch.Tensor:
    """The no-dropout probability of the predicted class, one per sample, shape (N,).

    The null baseline. If a detector cannot beat this, its extra machinery is not
    earning anything.

    Negated so that low means poisoned, matching PSU's convention: a backdoored
    model is typically MORE confident on a triggered input, so raw confidence
    points the other way and would read as an inverted detector.
    """
    model.eval()

    scores = []
    for images, _ in loader:
        probs = forward_probs(model, images, device, use_bfloat16)  # (batch, classes)
        scores.append(-probs.max(dim=1).values.cpu())

    return torch.cat(scores).float() if scores else torch.empty(0)


def _entropy(probs: torch.Tensor) -> torch.Tensor:
    """Shannon entropy per row, in nats, numerically safe at p=0.

    original form
        H(p) = - sum_c p_c * log(p_c)
    descriptive form
        entropy = negative sum over classes of probability times its log
    """
    entropy = -(probs.clamp_min(PROBABILITY_FLOOR).log() * probs).sum(dim=1)  # (N,)
    return entropy


@torch.inference_mode()
def strip_scores(
    model: nn.Module,
    loader: DataLoader,
    overlay_images: torch.Tensor,
    device: torch.device,
    use_bfloat16: bool,
    seed: int,
    num_overlays: int = 8,
) -> torch.Tensor:
    """STRIP entropy per sample, shape (N,), low meaning poisoned.

    For each suspicious input, superimpose it with num_overlays clean images drawn
    from the defender's own clean set, and average the prediction entropy over
    those blends. A clean input's class evidence is destroyed by superimposition,
    so its predictions scatter and entropy is high. A trigger survives blending and
    keeps pulling the prediction to the target class, so entropy stays low.

    Superimposition is a plain sum, as in the original: the paper perturbs by
    adding a second image rather than by alpha-blending, and the resulting
    out-of-range pixel values are part of why clean predictions become unstable.

    overlay_images is a (num_overlays, C, H, W) batch taken from the clean
    validation split, data a defender is already assumed to hold. Reseeding here
    makes the overlay assignment reproducible. The overlays themselves are fixed
    across all samples so every input faces the same perturbation set, which
    removes overlay choice as a source of per-sample variance.

    Returned as the raw entropy, NOT negated. STRIP's whole claim is that a
    triggered input has LOW entropy under superimposition, and the shared detection
    convention is that LOW means poisoned, so the 2 already agree and negating
    would invert the detector. That mistake was made here first and showed up as
    AUROC 0.000, perfect separation with the sign reversed, which is what the
    two-sided field in detection_report exists to surface.
    """
    model.eval()
    seed_everything(seed)
    overlays = overlay_images[:num_overlays].to(device)  # (num_overlays, C, H, W)

    scores = []
    for images, _ in loader:
        images = images.to(device)  # (batch, C, H, W)
        total = torch.zeros(images.size(0), device=device)  # (batch,)
        for overlay in overlays:
            blended = images + overlay.unsqueeze(0)
            probs = forward_probs(model, blended, device, use_bfloat16)
            total += _entropy(probs)
        scores.append((total / len(overlays)).cpu())

    return torch.cat(scores).float() if scores else torch.empty(0)


@torch.inference_mode()
def collect_overlay_batch(loader: DataLoader, count: int, seed: int) -> torch.Tensor:
    """The first `count` clean images, as STRIP's superimposition set, (count, C, H, W).

    Taken from the clean validation split the defender already holds for
    thresholding, so STRIP is given exactly the same data budget as PSBD and
    neither method gets an advantage from seeing more.
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

    overlays = torch.cat(collected)[:count]
    return overlays
