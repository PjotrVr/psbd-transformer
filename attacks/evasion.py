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

A second objective, `psu_mean`, implements the adaptive attacker the PSBD
paper itself describes (Li, Chen, Liu, Wang, arXiv 2406.05826, Appendix
"Resistance to Potential Adaptive Attacks"). Selected with
`probe["objective"] = "psu_mean"`, default `"psu_gap_hinge"`, the objective
above. The 2 canonical names state what each loss computes: `psu_gap_hinge`
penalises the gap between the poisoned and the clean mean fractional PSU
under the probe, `psu_mean` adds alpha times the mean absolute PSU over every
training sample to the backdoor loss. The old names `"hinge"` and
`"psbd_paper"` are still accepted (`normalize_evasion_objective`), so an
already-queued job script naming either keeps running unchanged.

    original form
        L = (1 - alpha) L_bd + alpha L_ada
        L_ada = phi_PSU(x), x in D^c union D^b
        phi_PSU(x) = P_c(x; theta) - (1/k) sum_{i=1}^{k} P_c(x; p, theta'_i)

    descriptive form
        loss = (1 - alpha) * cross_entropy + alpha * mean_over_batch(psu_absolute)
        psu_absolute(x) = prob_no_probe(c) - mean_over_passes(prob_with_probe(c))

L_bd is the paper's own name for the cross entropy loss on the poisoned training
set, D^c and D^b are its clean and poisoned partitions, phi_PSU is PSBD's own
Prediction Shift Uncertainty (sec/4_method.tex) and alpha is the weight, read
off `--evade-weight` for this objective rather than the additive coefficient it
is for the hinge.

D^c union D^b is the whole training set, so L_ada is a mean over the entire
batch, both clean and poisoned samples together, with no reference to
is_poisoned at all. That is the mechanism the paper's own text describes. The
paper aims for benign samples to carry a low PSU value themselves, so the
defender's threshold can no longer separate the 2 groups by that statistic.

2 deviations from the paper's own recipe, both recorded here:

1. phi_PSU is the ABSOLUTE PSU, a plain probability difference, not the
   fractional ratio (`psu_for_batch`) the rest of this module and
   `defences.scores` use as the headline statistic. This module implements both
   forms side by side, `psu_for_batch` for the psu_gap_hinge objective and
   `absolute_psu_for_batch` for `psu_mean`, so each objective matches the form
   its own source specifies.
2. The paper computes L_ada every 50 iterations to save compute. This computes
   it every batch instead, which is what the psu_gap_hinge objective already does and
   keeps both objectives on the same training loop, at the cost of running the
   k extra probe passes more often than the paper's own recipe.

A third path, multi-probe evasion, extends the psu_gap_hinge objective (`--evade-probes`,
`cli.train_backdoor`) from 1 probe to several, so the attacker trains against the
defence's own min-rank union (`defences.decision.multi_probe_auroc`,
`docs/hypothesis/H41-multi-probe-defence.md`) rather than a single placement it
could deploy a second probe against.

    original form
        L = L_CE(f(x), y)
            + lambda * (1/M) * sum_{j=1}^{M}
                ReLU( mean_{x in C} PSU_j(x) - mean_{x in P} PSU_j(x) )

    descriptive form
        loss = cross_entropy
               + weight * mean_over_probes(hinge_j)
               hinge_j = relu(mean_clean_psu_j - mean_poisoned_psu_j)

M is the number of probes, j indexes them and PSU_j is the fractional PSU of
`psu_for_batch` under probe j's own (position, operator, rate). Every other
symbol is shared with the single-probe hinge above.

The mean over probes, not the sum, is deliberate: it keeps `weight`'s meaning
fixed regardless of how many probes are trained against, so a 1-probe run
under this path is bit-for-bit the single-probe hinge (`evasion_penalty`) and
adding a 4th probe cannot inflate the penalty's scale by counting alone. A sum
would couple `--evade-weight` to `len(probes)`, so retuning it would be needed
every time a probe is added or removed.

Cost: `_probe_confidences` runs the unperturbed forward once and reuses it
across every probe (passed in as `logits`), so only the `passes` stochastic
passes are paid per probe. M probes at `passes` = k cost 1 + M*k forwards per
step, against 1 + k for a single probe, so a 3-probe run at k=3 costs about
2.5 times a 1-probe run's forward count (1 + 9 against 1 + 3), which is why the
smoke test and the job generator both budget wall clock separately from the
single-probe adaptive-attacker jobs.
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


def _probe_confidences(
    model: nn.Module,
    images: torch.Tensor,
    probe: dict,
    passes: int,
    logits: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The 2 confidences phi_PSU is built from, both differentiable w.r.t. the model.

    Returns (base, dropped, logits), each carrying grad history. base is
    P_c(x; theta) and dropped is the mean of P_c(x; p, theta'_i) over `passes`
    probe samples, both shaped (batch,). logits is the unperturbed forward pass,
    shaped (batch, num_classes), handed back so the caller reuses it for
    cross-entropy. Every retained forward is a full ViT-B/16 activation graph,
    and computing the clean pass twice is enough on its own to run a batch of
    64 out of memory on a 40 GB card.

    Shared by the fractional PSU (`psu_for_batch`, the defender's headline
    statistic) and the absolute PSU (`absolute_psu_for_batch`, the paper's own
    form), which differ only in whether (base - dropped) is normalised by base.

    The model is switched to eval mode for this computation so the attacker
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

    return base, dropped, logits


def psu_for_batch(
    model: nn.Module,
    images: torch.Tensor,
    probe: dict,
    passes: int,
    logits: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sample fractional PSU under the probe, differentiable w.r.t. the model.

    Returns (psu, logits), both carrying grad history: psu is (batch,) and logits
    is (batch, num_classes). This is the defender's headline statistic
    (`defences.scores.psu_ratio_from_cache`), used by the psu_gap_hinge objective.
    """
    base, dropped, logits = _probe_confidences(model, images, probe, passes, logits)
    base_clamped = base.clamp_min(BASE_PROBABILITY_FLOOR)

    psu = (base - dropped) / base_clamped  # (batch,)
    return psu, logits


def absolute_psu_for_batch(
    model: nn.Module,
    images: torch.Tensor,
    probe: dict,
    passes: int,
    logits: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sample absolute PSU under the probe, differentiable w.r.t. the model.

    Returns (psu, logits), both carrying grad history: psu is (batch,) and logits
    is (batch, num_classes).

        original form
            phi_PSU(x) = P_c(x; theta) - (1/k) sum_{i=1}^{k} P_c(x; p, theta'_i)
        descriptive form
            psu_absolute = prob_no_probe(c) - mean_over_passes(prob_with_probe(c))

    This is PSBD's own form of the statistic (sec/4_method.tex), unnormalised by
    the base confidence, and is what the paper's adaptive-attacker loss
    (`psu_mean` objective) is defined over. See `psu_for_batch` for the
    fractional form this repository otherwise reports.
    """
    base, dropped, logits = _probe_confidences(model, images, probe, passes, logits)
    psu = base - dropped  # (batch,)
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


def multi_probe_psu_for_batch(
    model: nn.Module,
    images: torch.Tensor,
    probes: list[dict],
    passes: int,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Fractional PSU under each of `probes`, sharing 1 unperturbed forward pass.

    Returns (psu_per_probe, logits). psu_per_probe has 1 entry per probe, each
    shaped (batch,) and carrying grad history. logits is the shared unperturbed
    forward, shaped (batch, num_classes), computed once on the first probe and
    handed to every later call so the caller pays the base forward once rather
    than once per probe (see the module docstring's cost accounting).
    """
    logits = None
    psu_per_probe = []
    for probe in probes:
        psu, logits = psu_for_batch(model, images, probe, passes, logits)
        psu_per_probe.append(psu)
    return psu_per_probe, logits


def multi_probe_evasion_penalty(
    psu_per_probe: list[torch.Tensor], is_poisoned: torch.Tensor
) -> torch.Tensor:
    """The mean of `evasion_penalty` over every probe, see the module docstring's formula.

    Returns a 0-dim tensor with grad history. Reduces to `evasion_penalty`'s own
    value when `psu_per_probe` has 1 entry, which is what keeps a 1-probe call
    through this path identical to the plain single-probe hinge.
    """
    hinges = torch.stack(
        [evasion_penalty(psu, is_poisoned) for psu in psu_per_probe]
    )  # (M,)
    return hinges.mean()


# The evasion objectives evasive_update dispatches on, read off
# probe.get("objective", "psu_gap_hinge"). Exported so cli.train_backdoor's
# --evade-objective choices cannot drift from what this module actually handles.
# "hinge" and "psbd_paper" are the pre-rename names, kept as accepted choices
# (translated by normalize_evasion_objective) so an already-queued job script
# still runs, and every new script should name the canonical pair instead.
EVASION_OBJECTIVES = ("psu_gap_hinge", "psu_mean", "hinge", "psbd_paper")

# Canonical objective names state what the loss computes, not whose loss it
# is: "psu_gap_hinge" was "hinge", "psu_mean" was "psbd_paper".
EVASION_OBJECTIVE_ALIASES: dict[str, str] = {
    "hinge": "psu_gap_hinge",
    "psbd_paper": "psu_mean",
}


def normalize_evasion_objective(objective: str) -> str:
    """The canonical objective name, translating a deprecated alias unchanged otherwise."""
    canonical = EVASION_OBJECTIVE_ALIASES.get(objective, objective)
    return canonical


def paper_adaptive_penalty(psu_absolute: torch.Tensor) -> torch.Tensor:
    """L_ada from the PSBD paper's adaptive-attacker appendix, see the module docstring.

        original form
            L_ada = phi_PSU(x), x in D^c union D^b
        descriptive form
            loss_ada = mean_over_batch(psu_absolute)

    D^c union D^b is the whole training set, so unlike `evasion_penalty` this
    takes the mean over every sample in the batch and never looks at
    is_poisoned: it pushes PSU down for benign and poisoned samples alike.
    """
    penalty = psu_absolute.mean()
    return penalty


def evasive_update(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    is_poisoned: torch.Tensor,
    criterion,
    optimizer,
    probe: dict | list[dict],
    weight: float,
    passes: int,
) -> tuple[torch.Tensor, dict]:
    """A single optimizer step on cross-entropy plus the chosen evasion objective.

    `probe` is a single probe dict (unchanged behaviour) or a list of probe
    dicts, which routes through the multi-probe hinge (module docstring's 3rd
    formula) regardless of how many probes the list holds. A 1-entry list is
    numerically identical to passing that entry as a plain dict. Every probe in
    a list is read as `objective="psu_gap_hinge"`, since the multi-probe
    formula is only defined for the hinge: a list with any other objective
    raises.

    A single probe's `probe.get("objective", "psu_gap_hinge")` selects the loss
    (normalize_evasion_objective translates the deprecated "hinge" /
    "psbd_paper" spellings first):

    - "psu_gap_hinge" (default, unchanged behaviour): fractional PSU, additive
      penalty, loss = cross_entropy + weight * evasion_penalty(psu, is_poisoned).
    - "psu_mean": absolute PSU, convex combination as the paper defines it,
      loss = (1 - weight) * cross_entropy + weight * paper_adaptive_penalty(psu),
      with weight read as alpha.
    """
    optimizer.zero_grad(set_to_none=True)
    probes = probe if isinstance(probe, list) else [probe]
    objectives = {
        normalize_evasion_objective(p.get("objective", "psu_gap_hinge")) for p in probes
    }
    if len(probes) > 1 and objectives != {"psu_gap_hinge"}:
        raise ValueError(
            "multi-probe evasion only supports the psu_gap_hinge objective, got "
            f"{objectives}"
        )
    objective = next(iter(objectives))

    if objective == "psu_gap_hinge":
        psu_per_probe, logits = multi_probe_psu_for_batch(model, images, probes, passes)
        penalty = multi_probe_evasion_penalty(psu_per_probe, is_poisoned)
        loss = criterion(logits, labels) + weight * penalty
        psu = psu_per_probe[0]
    elif objective == "psu_mean":
        psu, logits = absolute_psu_for_batch(model, images, probes[0], passes)
        penalty = paper_adaptive_penalty(psu)
        loss = (1.0 - weight) * criterion(logits, labels) + weight * penalty
    else:
        raise ValueError(
            f"unknown evasion objective {objective!r}, expected one of "
            f"{EVASION_OBJECTIVES}"
        )

    loss.backward()
    optimizer.step()

    poisoned = is_poisoned.bool()  # (batch,)
    # Both group means are reported beside the gap, whichever objective is
    # active. The cheapest way to close the gap is to drag clean shift down to
    # meet poisoned rather than raise poisoned, which changes the whole model
    # rather than hiding a backdoor, and the gap alone cannot tell the 2 apart.
    # Under multi-probe hinge these read the first listed probe only. The
    # per-probe penalty averaged into `penalty` is the quantity multi-probe
    # runs should track, not this single probe's own PSU split.
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
