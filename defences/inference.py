"""Forward passes for PSBD: the no-dropout baseline cache and the stochastic passes.

Prediction Shift Uncertainty per the PSBD paper, Equation 2:

    original form
        phi_PSU(x) = P_c(x; theta) - (1/k) * sum_{i=1..k} P_c(x; p, theta_i')
        with c = argmax_c P(x; theta)

    descriptive form
        psu(x) = prob_no_dropout(argmax_class) - mean_over_k_passes(
                     prob_with_dropout(argmax_class))

A low PSU means the confidence in the no-dropout prediction barely moves under
dropout, which flags the sample as likely poisoned. The subtraction itself lives
in defences.psbd_metrics, on the CPU side; this module only produces the two
forward-pass ingredients it needs.

Nothing here ever touches the model's own dropout modules. The perturbation comes
entirely from modules plugged in by defences.dropout, which live in hook closures
outside the model tree and are explicitly left in train mode, so model.eval()
keeps every built-in dropout at its natural identity while the probe still fires.
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


@torch.inference_mode()
def build_baseline_cache(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_bfloat16: bool,
) -> list[dict]:
    """Precompute the no-dropout state of one split, once.

    Caching avoids recomputing the deterministic baseline for every dropout rate
    in the sweep, which is the dominant cost saving across the run. Must be
    called before any position is plugged; the sweep entrypoint guarantees that
    by building every baseline before its rate loop starts.

    Three tensors per batch. probs and its argmax are what PSU is measured
    against. loader_labels is what the loader asked for, which on the backdoor
    split is the attack-success label, so comparing it to the argmax recovers
    per-sample whether the trigger actually flipped this image. That matters
    whenever ASR is well below 1: a triggered image the model classifies
    correctly is behaviourally clean, and scoring it as a detection positive
    penalises the detector for the attack's failure.
    """
    model.eval()
    cache: list[dict] = []
    for images, labels in loader:
        probs = forward_probs(model, images, device, use_bfloat16)
        cache.append(
            {
                "probs": probs.cpu(),
                "labels": probs.argmax(dim=1).cpu(),
                "loader_labels": labels.cpu().long(),
            }
        )
    return cache


@torch.inference_mode()
def compute_dropout_pass_probs(
    model: nn.Module,
    loader: DataLoader,
    baseline_labels: torch.Tensor,
    device: torch.device,
    forward_passes: int,
    use_bfloat16: bool,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Raw per-pass evidence, both shaped (forward_passes, N).

    Returns (probs, argmax):
      probs   float32, the probability assigned to the baseline-argmax class c
      argmax  int16, the class each dropout pass actually predicted

    Raw, not reduced: collapsing the k passes to one score here would throw away
    the ability to recompute PSU under a different aggregation (median instead of
    mean, a different k) without rerunning the GPU pass. Saving one float per pass
    per sample keeps that open at trivial disk cost.

    argmax is saved because PSU alone cannot express the paper's own mechanism.
    Three things need it and none are recoverable from probs: the shift ratio
    sigma (paper Eq. PS, the fraction of passes whose prediction changed), the
    adaptive dropout-rate rule which is defined on sigma, and the central claim
    that clean samples which shift, shift specifically to the target class. int16
    is safe for every dataset here; tiny is the largest at 200 classes.

    baseline_labels is the flat (N,) no-dropout argmax class per sample, in the
    same shuffle=False order the loader serves, so a running offset pairs each
    batch to its labels without re-batching.

    The perturbation comes from dropout modules plugged in by hooks (see
    defences.dropout), which are already in train mode, so this never toggles the
    model's own dropout. model.eval() keeps every existing dropout at its natural
    identity. Reseeding here fixes the mask sequence, so rerunning the same
    (split, position, rate) reproduces the same masks exactly. Across two splits
    of different length the sequences agree only up to the shorter one's batch
    count, which is why clean and backdoor pairing is done by sample index at
    analysis time, not by assuming shared masks.
    """
    model.eval()
    seed_everything(seed)

    prob_batches: list[torch.Tensor] = []
    argmax_batches: list[torch.Tensor] = []
    offset = 0
    for images, _ in loader:
        batch_size = images.size(0)
        images = images.to(device)
        labels = baseline_labels[offset : offset + batch_size].to(device)
        offset += batch_size

        prob_columns = []
        argmax_columns = []
        for _ in range(forward_passes):
            probs = forward_probs(model, images, device, use_bfloat16)
            selected = probs.gather(1, labels.view(-1, 1)).squeeze(1)
            prob_columns.append(selected.cpu())
            argmax_columns.append(probs.argmax(dim=1).to(torch.int16).cpu())
        prob_batches.append(torch.stack(prob_columns, dim=0))
        argmax_batches.append(torch.stack(argmax_columns, dim=0))

    if not prob_batches:
        return (
            torch.empty(forward_passes, 0),
            torch.empty(forward_passes, 0, dtype=torch.int16),
        )
    return (
        torch.cat(prob_batches, dim=1).float(),
        torch.cat(argmax_batches, dim=1),
    )
