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

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from evaluation.metrics import clean_accuracy
from attacks.evasion import train_one_epoch_evasive
from models.backbones import build_swin, build_vit
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
        return build_vit(num_classes, dropout=model_dropout)

    if architecture == "swin":
        if model_dropout:
            # swin_s takes its own dropout arguments and ships with stochastic
            # depth already active, so a rate here would not mean the same thing
            # it means for ViT. Refusing is better than quietly measuring
            # something else.
            raise ValueError("training-time dropout is implemented for ViT only")
        return build_swin(num_classes)

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
        return SAM(
            model.parameters(),
            torch.optim.Adam,
            rho=rho,
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    return optimizer


def plain_update(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    criterion,
    optimizer,
) -> torch.Tensor:
    """A plain forward, backward and step, returning the batch loss."""
    optimizer.zero_grad()
    loss = criterion(model(images), labels)
    loss.backward()
    optimizer.step()
    return loss


def sam_update(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    criterion,
    optimizer,
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
    loss = criterion(model(images), labels)
    loss.backward()
    optimizer.first_step(zero_grad=True)

    criterion(model(images), labels).backward()
    optimizer.second_step(zero_grad=True)
    return loss


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    optimizer,
    device: torch.device,
    use_sam: bool,
) -> float:
    """An epoch of ordinary training, returning the mean batch loss."""
    model.train()

    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device).long()
        update = sam_update if use_sam else plain_update
        running_loss += update(model, images, labels, criterion, optimizer).item()

    mean_loss = running_loss / max(len(loader), 1)
    return mean_loss


def checkpoint_metadata(
    dataset: str,
    attack: str,
    label_mode: str | None,
    target_label: int,
    poison_rate: float,
    cover_rate: float,
    realized_poison_rate: float | None,
    architecture: str,
    use_sam: bool,
    rho: float,
    epochs: int,
    seed: int,
    max_samples: int | None,
    clean_accuracy: float | None,
    asr: float | None,
    started_at: str,
    ended_at: str,
    evasion: dict | None = None,
    model_dropout: float = 0.0,
) -> dict:
    """Training provenance for a checkpoint, written alongside it as args.json.

    Every training entrypoint builds this the same way so the key set never drifts
    between them.
    """
    metadata = {
        "dataset": dataset,
        "attack": attack,
        "label_mode": label_mode,
        "target_label": target_label,
        "poison_rate": poison_rate,
        # The rate actually achieved. choose_poison_indices caps the count at the
        # eligible pool, and a clean-label attack is eligible only on the target
        # class, so 1%, 5% and 10% can all resolve to the same poisoned set. Any
        # poison-rate trend has to be read against this, never against the request.
        "realized_poison_rate": realized_poison_rate,
        "cover_rate": cover_rate,
        "architecture": architecture,
        "optimizer": "sam" if use_sam else "adam",
        "rho": rho if use_sam else None,
        "epochs": epochs,
        "seed": seed,
        "max_samples": max_samples,
        "git_commit": current_git_commit(),
        "clean_accuracy": clean_accuracy,
        "asr": asr,
        "trained_started_at": started_at,
        "trained_ended_at": ended_at,
        # None for every normally trained checkpoint. Present and non-null only for
        # an adaptive-attacker run, so the 2 can never be confused when a detection
        # number is read back off this folder.
        "evasion": evasion,
        # Training-time dropout. 0.0 for every checkpoint trained before this
        # existed, which is what PSBD's protocol requires.
        "model_dropout": model_dropout,
    }
    return metadata


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
) -> nn.Module:
    """A freshly trained model, with validation accuracy printed after each epoch.

    evasion, when set, swaps in the adaptive attacker's epoch (attacks.evasion). It
    is the attacker's knob, and it is recorded in the checkpoint metadata so the
    run can never be mistaken for an ordinary training run.
    """
    model = build_model(architecture, num_classes, model_dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, use_sam, learning_rate, weight_decay, rho)

    for epoch in range(1, epochs + 1):
        if evasion:
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
            average_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, device, use_sam
            )
            extra = ""

        validation_accuracy = clean_accuracy(model, val_loader, device, use_bfloat16)
        print(
            f"epoch {epoch}: loss={average_loss:.4f} "
            f"val_acc={validation_accuracy:.4f}{extra}"
        )
        # A caller can snapshot the trajectory without the loop learning about
        # checkpoint paths. Every write stays on the caller's side.
        if on_epoch_end is not None:
            on_epoch_end(model, epoch, validation_accuracy)

    return model
