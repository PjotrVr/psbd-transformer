"""Forward passes for PSBD: the no-probe baseline and the perturbed passes.

Prediction Shift Uncertainty per the PSBD paper, Equation 2:

    original form
        phi_PSU(x) = P_c(x; theta) - (1/k) * sum_{i=1..k} P_c(x; p, theta_i')
        with c = argmax_c P(x; theta)

    descriptive form
        psu(x) = prob_no_dropout(argmax_class)
                 - mean_over_k_passes(prob_with_dropout(argmax_class))

A low PSU means the confidence in the no-dropout prediction barely moves under
the perturbation, which flags the sample as likely poisoned. The subtraction lives
in defences.scores, on the CPU side. This module produces the 2 forward-pass
ingredients it needs and writes nothing itself.

Nothing here touches the model's own dropout modules. The perturbation comes from
modules plugged in by models.positions, which live in hook closures outside the
model tree and are left in train mode, so model.eval() keeps every built-in
dropout at its identity while the probe still fires.
"""

from contextlib import contextmanager, nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import seed_everything
from torch.utils.data import DataLoader

from models.positions import activate_model_dropout, restore_model_dropout


def forward_logits(
    model: nn.Module,
    images: torch.Tensor,
    device: torch.device,
    use_bfloat16: bool,
) -> torch.Tensor:
    """Logits, (batch, num_classes), float32 whatever autocast did.

    images is a (batch, channels, height, width) batch, moved to device here.
    Not under no_grad. A caller that needs the gradient of a logit with respect
    to the input or to a captured activation differentiates through this call,
    which is what the gradient-based detectors do.
    """
    with _autocast_context(device, use_bfloat16):
        logits = model(images.to(device))  # (batch, num_classes), bf16 under autocast

    logits_float32 = logits.float()  # (batch, num_classes)
    return logits_float32


def forward_probs(
    model: nn.Module,
    images: torch.Tensor,
    device: torch.device,
    use_bfloat16: bool,
) -> torch.Tensor:
    """Softmax probabilities, (batch, num_classes), float32 whatever autocast did.

    images is a (batch, channels, height, width) batch, moved to device here.
    """
    logits = forward_logits(model, images, device, use_bfloat16)  # (batch, num_classes)
    probs = F.softmax(logits, dim=1)  # (batch, num_classes)
    return probs


@contextmanager
def frozen_parameters(model: nn.Module):
    """A context in which no parameter requires a gradient, restored exactly on exit.

    A detector that optimises an input mask or differentiates a logit with respect
    to an activation wants the backward pass to stop at the activations. With
    every parameter frozen autograd builds no weight-gradient graph, so the
    backward costs about a third less, and no parameter can accumulate a .grad
    that a later optimiser step would consume by mistake. Flags are saved per
    parameter, so a model that already had some frozen gets them back as they were.
    """
    saved = [(parameter, parameter.requires_grad) for parameter in model.parameters()]
    for parameter, _ in saved:
        parameter.requires_grad_(False)
    try:
        yield
    finally:
        for parameter, flag in saved:
            parameter.requires_grad_(flag)


@torch.inference_mode()
def build_baseline_cache(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_bfloat16: bool,
) -> list[dict]:
    """The no-perturbation state of a split, computed once for every rate to share.

    Must run before any position is plugged, which the sweep guarantees by building
    every baseline before its rate loop starts.

    3 tensors per batch, as 1 dict per batch in loader order. probs is the
    (batch, num_classes) softmax and labels its (batch,) argmax, which are what
    PSU is measured against. loader_labels is the (batch,) label the loader asked
    for, on the backdoor split the attack-success label, so comparing it to the
    argmax says per sample whether the trigger actually flipped the image. A
    triggered image the model still classifies correctly is behaviourally clean,
    and scoring it as a positive would penalise the detector for the attack's
    failure.
    """
    model.eval()

    cache: list[dict] = []
    for images, labels in loader:
        probs = forward_probs(
            model, images, device, use_bfloat16
        )  # (batch, num_classes)
        cache.append(
            {
                "probs": probs.cpu(),
                "labels": probs.argmax(dim=1).cpu(),  # (batch,)
                "loader_labels": labels.cpu().long(),  # (batch,)
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
    model_dropout: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The raw per-pass evidence as (probs, argmax), both shaped (passes, n).

    probs is float32, the probability each perturbed pass gave the baseline-argmax
    class. argmax is int16, the class each pass actually predicted. Both are kept
    raw rather than reduced so PSU can be recomputed under another aggregation
    without a GPU rerun. argmax is what the shift ratio, the adaptive rate rule and
    the shift-target claim all need, and none of them is recoverable from probs.

    baseline_labels is the (n,) no-perturbation argmax in the loader's own
    shuffle=False order, so a running offset pairs each batch to its labels.
    model_dropout, when set, switches the model's own dropouts on so the probe
    stacks on top of them. Removal then compounds and the nominal rate stops
    describing the disturbance.

    Reseeding fixes the mask sequence, so the same (split, position, rate)
    reproduces the same masks. Splits of different length agree only up to the
    shorter one's batch count, which is why clean and backdoor rows are paired by
    sample index at analysis time rather than by assuming shared masks.
    """
    model.eval()
    # Activation must follow eval(), which would otherwise put the model's own
    # dropouts straight back to identity and silently produce a complete,
    # plausible, unstacked result.
    restore = activate_model_dropout(model, model_dropout) if model_dropout else []
    seed_everything(seed)

    try:
        prob_batches: list[torch.Tensor] = []
        argmax_batches: list[torch.Tensor] = []
        offset = 0
        for images, _ in loader:
            batch_size = images.size(0)
            images = images.to(device)
            labels = baseline_labels[offset : offset + batch_size].to(
                device
            )  # (batch,)
            offset += batch_size

            prob_columns = []
            argmax_columns = []
            for _ in range(forward_passes):
                probs = forward_probs(
                    model, images, device, use_bfloat16
                )  # (batch, num_classes)
                selected = probs.gather(1, labels.view(-1, 1)).squeeze(1)  # (batch,)
                prob_columns.append(selected.cpu())
                argmax_columns.append(
                    probs.argmax(dim=1).to(torch.int16).cpu()
                )  # (batch,)

            prob_batches.append(torch.stack(prob_columns, dim=0))  # (passes, batch)
            argmax_batches.append(torch.stack(argmax_columns, dim=0))  # (passes, batch)
    finally:
        restore_model_dropout(restore)

    if not prob_batches:
        empty = (
            torch.empty(forward_passes, 0),  # (passes, 0)
            torch.empty(forward_passes, 0, dtype=torch.int16),  # (passes, 0)
        )
        return empty

    per_pass = (
        torch.cat(prob_batches, dim=1).float(),  # (passes, n)
        torch.cat(argmax_batches, dim=1),  # (passes, n)
    )
    return per_pass


def _autocast_context(device: torch.device, use_bfloat16: bool):
    """The autocast context for the forward pass, so stored scores stay float32."""
    if use_bfloat16 and device.type == "cuda":
        autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return autocast
    no_autocast = nullcontext()
    return no_autocast
