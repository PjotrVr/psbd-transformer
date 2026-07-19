"""Forward passes for PSBD: the no-dropout baseline cache and the PSU score.

Prediction Shift Uncertainty per the PSBD paper, Equation 2:

    original form
        phi_PSU(x) = P_c(x; theta) - (1/k) * sum_{i=1..k} P_c(x; p, theta_i')
        with c = argmax_c P(x; theta)

    descriptive form
        psu(x) = prob_no_dropout(argmax_class) - mean_over_k_passes(
                     prob_with_dropout(argmax_class))

A low PSU means the confidence in the no-dropout prediction barely moves under
dropout, which flags the sample as likely poisoned.
"""

from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import seed_everything
from torch.utils.data import DataLoader


def _autocast_context(device: torch.device, use_bfloat16: bool):
    """Run the forward pass in bfloat16 without downcasting stored scores."""
    if use_bfloat16 and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def forward_probs(
    model: nn.Module,
    images: torch.Tensor,
    device: torch.device,
    use_bfloat16: bool,
) -> torch.Tensor:
    """Softmax probabilities in float32 regardless of autocast dtype."""
    with _autocast_context(device, use_bfloat16):
        logits = model(images.to(device))
    return F.softmax(logits.float(), dim=1)


def enable_dropout_modules(model: nn.Module) -> None:
    """Put every Dropout into train mode so it samples a fresh mask per pass."""
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


@torch.inference_mode()
def build_baseline_cache(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_bfloat16: bool,
) -> list[dict]:
    """Precompute no-dropout probabilities and argmax labels once per split.

    Caching avoids recomputing the deterministic baseline for every dropout
    rate in the sweep, which is the dominant cost saving across the run.
    Assumes dropout is already off when called.
    """
    model.eval()
    cache: list[dict] = []
    for images, _ in loader:
        probs = forward_probs(model, images, device, use_bfloat16)
        cache.append({"probs": probs.cpu(), "labels": probs.argmax(dim=1).cpu()})
    return cache


@torch.inference_mode()
def compute_psu_and_shift(
    model: nn.Module,
    loader: DataLoader,
    baseline_cache: list[dict],
    device: torch.device,
    forward_passes: int,
    use_bfloat16: bool,
    seed: int,
) -> tuple[torch.Tensor, float]:
    """Return per-sample PSU scores (float32) and the dataset shift ratio.

    Shift ratio is the fraction of the k dropout passes whose argmax differs
    from the no-dropout argmax, averaged over the split. Reseeding here makes
    the sampled dropout masks reproducible and identical across the clean,
    backdoor, and validation calls, which keeps their comparison paired.
    """
    enable_dropout_modules(model)
    seed_everything(seed)

    per_sample_scores: list[torch.Tensor] = []
    shift_count = 0
    total_pass_samples = 0

    for (images, _), cache_row in zip(loader, baseline_cache):
        images = images.to(device)
        baseline_probs = cache_row["probs"].to(device)
        baseline_labels = cache_row["labels"].to(device)

        dropout_probs = []
        for _ in range(forward_passes):
            probs = forward_probs(model, images, device, use_bfloat16)
            shift_count += (probs.argmax(dim=1) != baseline_labels).sum().item()
            dropout_probs.append(probs)
        total_pass_samples += images.size(0) * forward_passes

        mean_dropout_probs = torch.stack(dropout_probs, dim=0).mean(dim=0)
        confidence_drop = baseline_probs - mean_dropout_probs
        scores = confidence_drop.gather(1, baseline_labels.view(-1, 1)).squeeze(1)
        per_sample_scores.append(scores.cpu())

    scores = torch.cat(per_sample_scores) if per_sample_scores else torch.empty(0)
    shift_ratio = shift_count / max(total_pass_samples, 1)
    return scores.float(), shift_ratio
