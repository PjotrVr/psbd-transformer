"""An adaptive attacker that trains against PSBD.

A defence is only interesting if it survives an adversary who knows it. PSBD flags
samples whose prediction shift under dropout is LOW, so an attacker who controls
training can add a penalty that pushes the poisoned samples' shift up onto the
clean distribution, keeping the backdoor while removing the statistic the defence
reads.

    original form
        L = L_CE(f(x), y)
            + lambda * ReLU( mean_{x in C} PSU(x) - mean_{x in P} PSU(x) )

    descriptive form
        loss = cross_entropy
               + weight * relu(mean_clean_psu - mean_poisoned_psu)

L_CE is the classifier's cross-entropy on the batch, lambda the weight on the
penalty, C the clean samples in the batch, P the poisoned ones and PSU(x) the
prediction shift uncertainty of sample x under the probe.

The hinge is deliberate. A symmetric abs(mean_C - mean_P) would also punish
poisoned PSU for rising above clean, which costs capacity for no evasion benefit
and risks overshooting into a separation the defence could read from the other
side.

PSU here is the fractional form, matching the defender's preferred score:

    original form
        PSU_ratio(x) = 1 - mean_{i=1..k} P_c(x; p, theta'_i) / P_c(x; theta)
    descriptive form
        psu_ratio = 1 - mean_over_passes(prob_with_probe(c)) / prob_no_probe(c)

c is an index and is detached, so it carries no gradient. Everything else is
differentiable, because a dropout mask is a constant in the backward pass, so the
penalty reaches theta through both terms.

The PSU computation runs in eval mode so the defender and attacker measure the
same statistic. On Swin, train mode leaves stochastic depth active, which gives
the attacker a noisier, differently centred PSU than the defender reads.

See docs/plans/adaptive-attacker-and-dropout-stacking.md for the threat model, the
success criteria and the transfer test that is the actual point.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from defences.operators import build_operator
from models.positions import DROPOUT_CONFIGS, plug_dropout, unplug_dropout

# Guards the division when the model gives its own predicted class almost no
# probability, which would otherwise make the ratio diverge.
BASE_PROBABILITY_FLOOR = 1e-6


class FlaggedPoisonedSet(Dataset):
    """A poisoned training set that also reports which samples are poisoned.

    The attacker legitimately knows this: it chose the poison indices. It exists
    only inside the training loss and must never reach the defender's side of any
    evaluation, which is why this wrapper is confined to the adaptive path rather
    than added to PoisonedTrainingSet, whose 2-tuple contract every other caller
    depends on.
    """

    def __init__(self, inner):
        self.inner = inner
        if not hasattr(inner, "poison_indices"):
            raise TypeError(
                f"{type(inner).__name__} has no poison_indices, so per-sample "
                "poison flags cannot be produced"
            )

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, index: int):
        image, label = self.inner[index]
        is_poisoned = int(index in self.inner.poison_indices)
        return image, label, is_poisoned


def psu_for_batch(
    model: nn.Module,
    images: torch.Tensor,
    probe: dict,
    passes: int,
    logits: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sample fractional PSU under the probe, differentiable w.r.t. the model.

    Returns (psu, logits), both carrying grad history: psu is (batch,) and logits
    is (batch, num_classes). The logits are handed back so the caller reuses the
    unperturbed forward for cross-entropy. Every retained forward is a full
    ViT-B/16 activation graph, and computing the clean pass twice is enough on its
    own to run a batch of 64 out of memory on a 40 GB card.

    The model is switched to eval mode for the PSU computation so the attacker
    measures the same statistic as the defender. On Swin this disables stochastic
    depth, and on ViT (dropout=0) it is a no-op. The probe modules are outside the
    model tree and explicitly set to train mode by plug_dropout, so they sample
    masks regardless.

    The probe is plugged and unplugged around the stochastic passes, in a finally,
    because a probe left attached leaks into the next step's clean forward pass
    and compounds across steps, so the model would be trained against a
    perturbation nobody recorded.
    """
    was_training = model.training
    model.eval()

    if logits is None:
        logits = model(images)  # (batch, num_classes)
    tracked = logits.argmax(dim=1).detach()  # (batch,)
    probs = F.softmax(logits, dim=1)  # (batch, num_classes)
    base = probs.gather(1, tracked.view(-1, 1)).squeeze(1)  # (batch,)

    names = DROPOUT_CONFIGS.get(probe["position"], (probe["position"],))
    factory = {name: build_operator(probe["operator"]) for name in names}
    handles = plug_dropout(model, probe["architecture"], names, factory, probe["rate"])
    try:
        dropped = torch.zeros_like(base)  # (batch,)
        for _ in range(passes):
            perturbed = F.softmax(model(images), dim=1)  # (batch, num_classes)
            dropped = dropped + perturbed.gather(1, tracked.view(-1, 1)).squeeze(1)
        dropped = dropped / passes  # (batch,)
    finally:
        unplug_dropout(handles)

    if was_training:
        model.train()

    base_clamped = base.clamp_min(BASE_PROBABILITY_FLOOR)

    psu = (base - dropped) / base_clamped  # (batch,)
    return psu, logits


@torch.inference_mode()
def calibrate_probe_rate(
    model: nn.Module,
    val_loader: DataLoader,
    probe_config: dict,
    device: torch.device,
    target_sigma: float = 0.6,
    candidate_rates: tuple[float, ...] = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5),
) -> float:
    """Find the probe rate whose clean-validation shift ratio is closest to target.

    Shift ratio is the fraction of samples whose argmax changes under the
    perturbation. This matches the defender's sigma-matching rule
    (defences.decision.select_rate_at_matched_shift), so the attacker optimizes at the
    same perturbation strength the defender would choose.

    Runs a forward pass per candidate rate over the validation set, a small cost
    beside a 15-epoch training run.
    """
    model.eval()
    architecture = probe_config["architecture"]
    position = probe_config["position"]
    operator = probe_config["operator"]

    names = DROPOUT_CONFIGS.get(position, (position,))
    factory = {name: build_operator(operator) for name in names}

    baseline_argmax = []
    all_images = []
    for images, _ in val_loader:
        images = images.to(device)  # (batch, C, H, W)
        all_images.append(images)
        logits = model(images)  # (batch, num_classes)
        baseline_argmax.append(logits.argmax(dim=1).cpu())  # (batch,)
    baseline_argmax = torch.cat(baseline_argmax)  # (N,)

    shift_by_rate = {}
    for rate in candidate_rates:
        handles = plug_dropout(model, architecture, names, factory, rate)
        try:
            perturbed_argmax = []
            for images in all_images:
                logits = model(images)  # (batch, num_classes)
                perturbed_argmax.append(logits.argmax(dim=1).cpu())  # (batch,)
            perturbed_argmax = torch.cat(perturbed_argmax)  # (N,)
            shifted = (perturbed_argmax != baseline_argmax).float().mean().item()
            shift_by_rate[rate] = shifted
        finally:
            unplug_dropout(handles)

    usable = {r: s for r, s in shift_by_rate.items() if s is not None}
    if not usable:
        raise RuntimeError("no candidate rate produced a usable shift ratio")

    chosen = min(sorted(usable), key=lambda r: abs(usable[r] - target_sigma))
    print(
        f"calibrated probe rate: {chosen} "
        f"(sigma={usable[chosen]:.3f}, target={target_sigma})"
    )
    for r in sorted(shift_by_rate):
        print(f"  rate={r:.3f} sigma={shift_by_rate[r]:.3f}")
    return chosen


def evasion_penalty(psu: torch.Tensor, is_poisoned: torch.Tensor) -> torch.Tensor:
    """Hinge on the group means. Zero when poisoned shift already matches clean.

    Returns a 0-dim tensor with grad history when either group is missing from the
    batch, so a batch that happens to contain no poisoned sample contributes
    nothing rather than producing a nan that silently poisons the running loss. At
    a 1% poison rate and batch 128 that is a common case, not an edge case.
    """
    poisoned = is_poisoned.bool()  # (batch,)
    if poisoned.all() or not poisoned.any():
        no_penalty = psu.sum() * 0.0
        return no_penalty

    hinge = torch.relu(psu[~poisoned].mean() - psu[poisoned].mean())
    return hinge


def evasive_update(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    is_poisoned: torch.Tensor,
    criterion,
    optimizer,
    probe: dict,
    weight: float,
    passes: int,
) -> tuple[torch.Tensor, dict]:
    """A single optimizer step on cross-entropy plus the evasion hinge."""
    optimizer.zero_grad(set_to_none=True)
    psu, logits = psu_for_batch(model, images, probe, passes)
    penalty = evasion_penalty(psu, is_poisoned)
    loss = criterion(logits, labels) + weight * penalty
    loss.backward()
    optimizer.step()

    poisoned = is_poisoned.bool()  # (batch,)
    # Both group means are reported beside the gap. The cheapest way to close
    # the gap is to drag clean shift down to meet poisoned rather than raise
    # poisoned, which changes the whole model rather than hiding a backdoor, and
    # the gap alone cannot tell the 2 apart.
    stats = {
        "psu_clean": float(psu[~poisoned].mean())
        if (~poisoned).any()
        else float("nan"),
        "psu_poisoned": float(psu[poisoned].mean()) if poisoned.any() else float("nan"),
        "penalty": float(penalty.detach()),
    }
    batch_loss = loss.detach()
    return batch_loss, stats


def train_one_epoch_evasive(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    optimizer,
    device: torch.device,
    probe: dict,
    weight: float,
    passes: int,
) -> tuple[float, dict]:
    """An epoch of adaptive training, returning mean loss and mean diagnostics.

    The loader must yield 3-tuples, so it has to be built over FlaggedPoisonedSet.
    """
    model.train()
    total_loss, batches = 0.0, 0
    sums = {"psu_clean": 0.0, "psu_poisoned": 0.0, "penalty": 0.0}
    counts = dict.fromkeys(sums, 0)

    for images, labels, is_poisoned in loader:
        images = images.to(device)
        labels = labels.to(device).long()
        is_poisoned = is_poisoned.to(device)
        loss, stats = evasive_update(
            model,
            images,
            labels,
            is_poisoned,
            criterion,
            optimizer,
            probe,
            weight,
            passes,
        )
        total_loss += float(loss)
        batches += 1
        # A nan marks a group missing from this batch and is left out of the mean.
        for key, value in stats.items():
            if value == value:
                sums[key] += value
                counts[key] += 1

    means = {k: (sums[k] / counts[k] if counts[k] else float("nan")) for k in sums}

    mean_loss = total_loss / max(batches, 1)
    return mean_loss, means
