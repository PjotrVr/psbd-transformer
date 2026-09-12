"""Training ViT-B/16 or Swin-S on a poisoned dataset, with a plain or SAM optimizer.

This produces the checkpoints the sweep consumes, in the same BackdoorBench format.
It matters for 2 reasons. Swin needs locally trained models because BackdoorBench
ships no Swin backdoored checkpoints. And SAM only helps if the poisoned model is
trained with it, so the low-poison SAM experiment lives here.

The poisoned training loader is built by the training entrypoint from the attack
registry. This module takes any (train_loader, val_loader) whose training set
already carries the trigger and the correct labels, and stays agnostic to which
attack produced it.
"""

import json
import os
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from evaluation.metrics import clean_accuracy
from attacks.evasion import calibrate_probe_rate, train_one_epoch_evasive
from models.backbones import build_resnet18, build_swin, build_vit
from utils.provenance import current_git_commit
from .sam import SAM


def build_model(
    architecture: str, num_classes: int, model_dropout: float = 0.0
) -> nn.Module:
    """The backbone for an architecture, with optional training-time dropout.

    PSBD requires a model trained without dropout, so model_dropout is 0.0 for
    every checkpoint in the project. The argument exists to test that requirement,
    not to change the default.
    """
    if architecture == "vit":
        vit = build_vit(num_classes, dropout=model_dropout)
        return vit

    if architecture == "swin":
        if model_dropout:
            # swin_s takes its own dropout arguments and ships with stochastic
            # depth already active, so a rate here would not mean the same thing
            # it means for ViT. Refusing is better than quietly measuring
            # something else.
            raise ValueError("training-time dropout is implemented for ViT only")
        swin = build_swin(num_classes)
        return swin

    if architecture == "resnet18":
        if model_dropout:
            raise ValueError("training-time dropout is implemented for ViT only")
        resnet = build_resnet18(num_classes)
        return resnet

    raise ValueError(f"Unknown architecture: {architecture}")


def build_optimizer(
    model: nn.Module,
    use_sam: bool,
    learning_rate: float,
    weight_decay: float,
    rho: float,
) -> torch.optim.Optimizer:
    """Adam by default, or Adam wrapped in SAM when use_sam is set.

    Adam is the base in both cases, so the only difference between a plain run and a
    SAM run is the sharpness-aware 2-step, which keeps the comparison clean.
    """
    if use_sam:
        sam_optimizer = SAM(
            model.parameters(),
            torch.optim.Adam,
            rho=rho,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        return sam_optimizer

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    return optimizer


LEARNING_RATE_SCHEDULES = ("constant", "cosine")


def build_scheduler(
    optimizer: torch.optim.Optimizer, schedule: str, epochs: int
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """A per-epoch learning-rate schedule, None for the constant rate every panel run used.

    The cosine option exists for the GTSRB reruns: 13 runs on the constant rate
    collapsed to a single class in their last epochs after 3 to 4 epochs at the
    loss floor (docs/runs/2026-09-11-diverged-gtsrb-runs.md), and annealing to 0
    by the final epoch removes the step size that drove them off the minimum.
    SAM wraps the base optimizer, so the schedule attaches to the base.
    """
    if schedule not in LEARNING_RATE_SCHEDULES:
        raise ValueError(
            f"unknown schedule {schedule!r}, known: {LEARNING_RATE_SCHEDULES}"
        )
    if schedule == "constant":
        return None
    stepped = getattr(optimizer, "base_optimizer", optimizer)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(stepped, T_max=epochs)
    return scheduler


def clip_gradients(model: nn.Module, clip_grad_norm: float | None) -> None:
    """Clip the gradient norm in place when a bound is set, a no-op otherwise."""
    if clip_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)


def plain_update(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    criterion,
    optimizer,
    clip_grad_norm: float | None = None,
) -> torch.Tensor:
    """A plain forward, backward and step, returning the batch loss."""
    optimizer.zero_grad()
    loss = criterion(model(images), labels)  # 0-dim
    loss.backward()
    clip_gradients(model, clip_grad_norm)
    optimizer.step()
    return loss


def sam_update(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    criterion,
    optimizer,
    clip_grad_norm: float | None = None,
) -> torch.Tensor:
    """A SAM update, returning the loss at the original weights.

    SAM does not implement step(), so the 2 passes are the caller's job, in this
    order: backward, first_step to reach the local worst-case weights, a second
    backward there, then second_step to restore the weights and let the base
    optimizer update them. Skipping the second backward silently degrades this to
    a plain Adam update at twice the cost. training.sam has the full contract.
    """
    # ViT and Swin use LayerNorm rather than BatchNorm, so the 2 forward passes
    # carry no running-statistics hazard that SAM has with BatchNorm models.
    loss = criterion(model(images), labels)  # 0-dim
    loss.backward()
    clip_gradients(model, clip_grad_norm)
    optimizer.first_step(zero_grad=True)

    criterion(model(images), labels).backward()
    clip_gradients(model, clip_grad_norm)
    optimizer.second_step(zero_grad=True)
    return loss


class IndexedTrainingSet(Dataset):
    """A training set wrapper that also reports each sample's dataset index.

    Mirrors attacks.evasion.FlaggedPoisonedSet, but carries the plain index
    rather than a poisoned/clean flag: --record-sample-loss needs every
    sample's index to scatter its per-epoch loss into place, not just the
    poisoned ones.
    """

    def __init__(self, inner: Dataset):
        self.inner = inner

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        image, label = self.inner[index]
        return image, label, index


def locate_poison_indices(dataset) -> set[int]:
    """The poison index set from a (possibly wrapped) training dataset, or empty.

    Walks the .inner / .base_dataset chain the loader's Augmented/Flagged/Indexed
    wrappers use, since poison_indices only lives on the innermost
    PoisonedTrainingSet or CoverPoisonedTrainingSet.
    """
    current = dataset
    while current is not None:
        poison_indices = getattr(current, "poison_indices", None)
        if poison_indices is not None:
            return poison_indices
        current = getattr(current, "inner", None) or getattr(
            current, "base_dataset", None
        )
    return set()


def save_sample_loss_record(
    checkpoint_dir: str, sample_loss_history: torch.Tensor, poison_indices: set[int]
) -> None:
    """Write the per epoch per sample loss trajectory as <checkpoint_dir>/sample_loss.npz.

    sample_loss_history is (epochs_so_far, num_samples). Called after every
    epoch so a run interrupted mid training still leaves a usable partial
    trajectory. poison_indices travels alongside it, so the record is
    self-contained and needs no other file to know which columns are poisoned.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, "sample_loss.npz")
    np.savez(
        path,
        sample_loss=sample_loss_history.numpy(),
        poison_indices=np.array(sorted(poison_indices), dtype=np.int64),
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    optimizer,
    device: torch.device,
    use_sam: bool,
    clip_grad_norm: float | None = None,
    sample_loss_row: torch.Tensor | None = None,
) -> float:
    """An epoch of ordinary training, returning the mean batch loss.

    sample_loss_row, when given, is a (num_training_samples,) tensor this epoch
    fills in place. The loader is then expected to yield (image, label, index)
    triples (IndexedTrainingSet) rather than plain (image, label) pairs. Just
    before each batch's update, the per-sample cross entropy is computed with
    reduction="none" from the weights that batch is about to train on and
    scattered into the row by index. Left None (the default), the loader keeps
    its ordinary 2-tuple contract and nothing about a plain run changes.
    """
    model.train()
    per_sample_criterion = (
        nn.CrossEntropyLoss(reduction="none") if sample_loss_row is not None else None
    )

    running_loss = 0.0
    for batch in loader:
        if sample_loss_row is not None:
            images, labels, indices = batch
        else:
            images, labels = batch
        images, labels = (
            images.to(device),
            labels.to(device).long(),
        )  # (batch, C, H, W), (batch,)

        if sample_loss_row is not None:
            with torch.no_grad():
                per_sample_losses = per_sample_criterion(
                    model(images), labels
                )  # (batch,)
            sample_loss_row[indices] = per_sample_losses.cpu()

        update = sam_update if use_sam else plain_update
        loss = update(model, images, labels, criterion, optimizer, clip_grad_norm)
        running_loss += loss.item()

    mean_loss = running_loss / max(len(loader), 1)
    return mean_loss


@dataclass(frozen=True)
class CheckpointMetadata:
    """Training provenance for a checkpoint, written alongside it as args.json.

    Every training entrypoint builds this the same way so the key set never
    drifts between them. as_dict gives the sidecar's exact layout, which
    data.splits.read_checkpoint_metadata and the coverage ledger read back.
    """

    dataset: str
    attack: str
    label_mode: str | None
    target_label: int
    poison_rate: float
    cover_rate: float
    realized_poison_rate: float | None
    architecture: str
    use_sam: bool
    rho: float
    epochs: int
    seed: int
    max_samples: int | None
    clean_accuracy: float | None
    asr: float | None
    started_at: str
    ended_at: str
    # None for every normally trained checkpoint. Present and non-null only for
    # an adaptive-attacker run, so the 2 can never be confused when a detection
    # number is read back off this folder.
    evasion: dict | None = None
    # Training-time dropout. 0.0 for every checkpoint trained before this
    # existed, which is what PSBD's protocol requires.
    model_dropout: float = 0.0
    learning_rate_schedule: str = "constant"
    clip_grad_norm: float | None = None
    best_validation_accuracy: float | None = None
    final_validation_accuracy: float | None = None

    def as_dict(self) -> dict:
        """The args.json layout, the same keys in the same order on every entrypoint."""
        payload = {
            "dataset": self.dataset,
            "attack": self.attack,
            "label_mode": self.label_mode,
            "target_label": self.target_label,
            "poison_rate": self.poison_rate,
            # The rate actually achieved. choose_poison_indices caps the count at
            # the eligible pool, and a clean-label attack is eligible only on the
            # target class, so 1%, 5% and 10% can all resolve to the same poisoned
            # set. Any poison-rate trend has to be read against this, never
            # against the request.
            "realized_poison_rate": self.realized_poison_rate,
            "cover_rate": self.cover_rate,
            "architecture": self.architecture,
            "optimizer": "sam" if self.use_sam else "adam",
            "rho": self.rho if self.use_sam else None,
            "epochs": self.epochs,
            "seed": self.seed,
            "max_samples": self.max_samples,
            "git_commit": current_git_commit(),
            "clean_accuracy": self.clean_accuracy,
            "asr": self.asr,
            "trained_started_at": self.started_at,
            "trained_ended_at": self.ended_at,
            "evasion": self.evasion,
            "model_dropout": self.model_dropout,
            "lr_schedule": self.learning_rate_schedule,
            "clip_grad_norm": self.clip_grad_norm,
            # The validation trajectory's endpoints, so a collapsed run is visible
            # in the sidecar without reading its log.
            "best_validation_accuracy": self.best_validation_accuracy,
            "final_validation_accuracy": self.final_validation_accuracy,
        }
        return payload


def resolve_checkpoint_path(path: str) -> str:
    """The .pt path, whether given the file or the folder that should hold it.

    A job passing checkpoints/<name> without the filename would otherwise write the
    weights to a file called <name> and drop args.json into checkpoints/ itself,
    where the next run overwrites it.
    """
    return path if path.endswith(".pt") else os.path.join(path, "attack_result.pt")


def save_checkpoint(
    model: nn.Module, num_classes: int, path: str, metadata: dict | None = None
) -> None:
    """Save the weights in the format load_checkpoint reads, plus an args.json sidecar.

    metadata is the training provenance. It sits beside the .pt rather than inside
    it so it can be read without loading the weights.
    """
    path = resolve_checkpoint_path(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"model": model.state_dict(), "num_classes": num_classes}, path)

    if metadata:
        args_path = os.path.join(os.path.dirname(path), "args.json")
        with open(args_path, "w") as handle:
            json.dump(metadata, handle, indent=2)


# A run whose final validation accuracy falls below this share of its own best
# collapsed rather than converged. The 13 GTSRB collapses of September 2026 all
# fell from 0.99 to under 0.1, so the bar sits far from any real fluctuation.
DIVERGENCE_FRACTION = 0.5


@dataclass(frozen=True)
class TrainingTrajectory:
    """The validation accuracy after every epoch, in epoch order."""

    validation_accuracies: tuple[float, ...]

    @property
    def best(self) -> float:
        best = max(self.validation_accuracies)
        return best

    @property
    def final(self) -> float:
        final = self.validation_accuracies[-1]
        return final


def check_not_diverged(trajectory: TrainingTrajectory) -> None:
    """Refuse a model whose validation accuracy collapsed after its best epoch.

    Raises before any checkpoint is written, so a wrecked model never lands on
    disk with a plausible ASR beside it, which is what let 13 GTSRB runs enter
    the panel unnoticed (docs/runs/2026-09-11-diverged-gtsrb-runs.md).
    """
    if trajectory.final < DIVERGENCE_FRACTION * trajectory.best:
        raise RuntimeError(
            f"training diverged: validation accuracy ended at {trajectory.final:.4f} "
            f"after a best of {trajectory.best:.4f}. No checkpoint was written. Rerun "
            "with --lr-schedule cosine or --clip-grad-norm 1.0."
        )


def train_classifier(
    architecture: str,
    num_classes: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    use_sam: bool,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    rho: float = 0.1,
    use_bfloat16: bool = True,
    evasion: dict | None = None,
    model_dropout: float = 0.0,
    on_epoch_end: Callable[[nn.Module, int, float], None] | None = None,
    learning_rate_schedule: str = "constant",
    clip_grad_norm: float | None = None,
    record_sample_loss: bool = False,
    checkpoint_dir: str | None = None,
) -> tuple[nn.Module, TrainingTrajectory]:
    """A freshly trained model and its validation trajectory, printed per epoch.

    evasion, when set, swaps in the adaptive attacker's epoch (attacks.evasion). It
    is the attacker's knob, and it is recorded in the checkpoint metadata so the
    run can never be mistaken for an ordinary training run. A run whose
    validation accuracy collapses raises instead of returning.

    record_sample_loss, off by default, additionally scatters every training
    sample's cross entropy into a per-epoch row (train_one_epoch's
    sample_loss_row) and writes the stacked (epoch, sample) history plus the
    poison index set to <checkpoint_dir>/sample_loss.npz after every epoch
    (save_sample_loss_record). It requires train_loader.dataset to yield
    (image, label, index) triples (IndexedTrainingSet) and is not supported
    together with the evasive path.
    """
    model = build_model(architecture, num_classes, model_dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, use_sam, learning_rate, weight_decay, rho)
    scheduler = build_scheduler(optimizer, learning_rate_schedule, epochs)
    validation_accuracies: list[float] = []

    if record_sample_loss and evasion:
        raise ValueError(
            "record_sample_loss is not supported together with the evasive path"
        )
    if record_sample_loss and checkpoint_dir is None:
        raise ValueError("record_sample_loss requires checkpoint_dir")
    num_training_samples = len(train_loader.dataset)
    poison_indices = (
        locate_poison_indices(train_loader.dataset) if record_sample_loss else set()
    )
    sample_loss_history: list[torch.Tensor] = []

    # The rate this evasion run's probe(s) train against, 1 entry per epoch,
    # mutated onto the caller's `evasion` dict so cli.train_backdoor can drop
    # it straight into the checkpoint's args.json without a second return
    # value threading through every other train_classifier caller.
    recalibrate_every = evasion.get("recalibrate_every", 0) if evasion else 0
    rate_history: list = []

    for epoch in range(1, epochs + 1):
        if evasion:
            # A warm-up of `recalibrate_every` epochs keeps the initial
            # calibration, then the probe rate is recalibrated on the
            # CURRENT model every `recalibrate_every` epochs after that: the
            # whole reason to recalibrate at all is that the initial
            # calibration was measured on random init weights, whose shift
            # curve has nothing to do with the trained model's.
            if (
                recalibrate_every > 0
                and epoch > recalibrate_every
                and (epoch - recalibrate_every - 1) % recalibrate_every == 0
            ):
                probes = evasion["probe"]
                probes = probes if isinstance(probes, list) else [probes]
                for probe in probes:
                    probe["rate"] = calibrate_probe_rate(
                        model,
                        val_loader,
                        probe,
                        device,
                        target_sigma=evasion.get("calibration_target", 0.6),
                    )
            current_probes = evasion["probe"]
            rate_history.append(
                [p["rate"] for p in current_probes]
                if isinstance(current_probes, list)
                else current_probes["rate"]
            )
            average_loss, stats = train_one_epoch_evasive(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                evasion["probe"],
                evasion["weight"],
                evasion["passes"],
            )
            extra = (
                f" psu_clean={stats['psu_clean']:.4f}"
                f" psu_poisoned={stats['psu_poisoned']:.4f}"
                f" penalty={stats['penalty']:.4f}"
            )
        else:
            sample_loss_row = (
                torch.full((num_training_samples,), float("nan"))
                if record_sample_loss
                else None
            )
            average_loss = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                use_sam,
                clip_grad_norm,
                sample_loss_row,
            )
            if record_sample_loss:
                sample_loss_history.append(sample_loss_row)
                save_sample_loss_record(
                    checkpoint_dir, torch.stack(sample_loss_history), poison_indices
                )
            extra = ""
        if scheduler is not None:
            scheduler.step()

        validation_accuracy = clean_accuracy(model, val_loader, device, use_bfloat16)
        validation_accuracies.append(validation_accuracy)
        print(
            f"epoch {epoch}: loss={average_loss:.4f} "
            f"val_acc={validation_accuracy:.4f}{extra}"
        )
        # A caller can snapshot the trajectory without the loop learning about
        # checkpoint paths. Every write stays on the caller's side.
        if on_epoch_end is not None:
            on_epoch_end(model, epoch, validation_accuracy)

    if evasion:
        evasion["rate_history"] = rate_history

    trajectory = TrainingTrajectory(tuple(validation_accuracies))
    check_not_diverged(trajectory)
    return model, trajectory
