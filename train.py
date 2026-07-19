"""Training ViT-B/16 or Swin-S on a poisoned dataset, with a plain or SAM optimizer.

This produces the checkpoints the sweep consumes, in the same BackdoorBench
format. It matters for two reasons. Swin needs locally trained models because
BackdoorBench ships no Swin backdoored checkpoints. And SAM only helps if the
poisoned model is trained with it, so the low-poison SAM experiment lives here.

The poisoned training loader is built in train_backdoor.py from the attack
library. This module takes any (train_loader, val_loader) whose training set
already carries the trigger and the correct labels, and stays agnostic to which
attack produced it.
"""

import json
import os
import subprocess
from datetime import datetime, timezone

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from defences.detection import clean_accuracy
from models import build_swin, build_vit
from sam import SAM


def build_model(architecture: str, num_classes: int) -> nn.Module:
    if architecture == "vit":
        return build_vit(num_classes)
    if architecture == "swin":
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

    Adam is the base in both cases so the only difference between the vanilla and
    the SAM run is the sharpness-aware two-step, which keeps the comparison clean.
    """
    if use_sam:
        return SAM(
            model.parameters(),
            torch.optim.Adam,
            rho=rho,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    return torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )


def _plain_update(model, images, labels, criterion, optimizer) -> torch.Tensor:
    optimizer.zero_grad()
    loss = criterion(model(images), labels)
    loss.backward()
    optimizer.step()
    return loss


def _sam_update(model, images, labels, criterion, optimizer) -> torch.Tensor:
    # First pass finds the local worst-case weights, second pass updates from there.
    # ViT and Swin use LayerNorm rather than BatchNorm, so the two forward passes
    # carry no running-statistics hazard that SAM has with BatchNorm models.
    loss = criterion(model(images), labels)
    loss.backward()
    optimizer.first_step(zero_grad=True)

    criterion(model(images), labels).backward()
    optimizer.second_step(zero_grad=True)
    return loss


def train_one_epoch(model, loader, criterion, optimizer, device, use_sam) -> float:
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device).long()
        update = _sam_update if use_sam else _plain_update
        running_loss += update(model, images, labels, criterion, optimizer).item()
    return running_loss / max(len(loader), 1)


def current_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def checkpoint_metadata(
    dataset: str,
    attack: str,
    label_mode: str | None,
    target_label: int,
    poison_rate: float,
    cover_rate: float,
    architecture: str,
    use_sam: bool,
    rho: float,
    epochs: int,
    seed: int,
    clean_accuracy: float | None,
    asr: float | None,
    started_at: str,
    ended_at: str,
) -> dict:
    """Training provenance for a checkpoint, written alongside it as args.json.

    Both train_backdoor.py and train_benign.py build this the same way so the
    key set never drifts between the two entrypoints.
    """
    return {
        "dataset": dataset,
        "attack": attack,
        "label_mode": label_mode,
        "target_label": target_label,
        "poison_rate": poison_rate,
        "cover_rate": cover_rate,
        "architecture": architecture,
        "optimizer": "sam" if use_sam else "adam",
        "rho": rho if use_sam else None,
        "epochs": epochs,
        "seed": seed,
        "git_commit": current_git_commit(),
        "clean_accuracy": clean_accuracy,
        "asr": asr,
        "trained_started_at": started_at,
        "trained_ended_at": ended_at,
    }


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_checkpoint(
    model: nn.Module, num_classes: int, path: str, metadata: dict | None = None
) -> None:
    """Save in the attack_result.pt format the loader reads.

    metadata is training provenance (which attack produced the model, dataset,
    target label, poison rate, seed, ...), written as an args.json sidecar next
    to the checkpoint rather than merged into the .pt, so it can be read without
    loading the model weights.
    """
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
) -> nn.Module:
    """Train a fresh model and report validation accuracy each epoch."""
    model = build_model(architecture, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, use_sam, learning_rate, weight_decay, rho)

    for epoch in range(1, epochs + 1):
        average_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, use_sam
        )
        validation_accuracy = clean_accuracy(model, val_loader, device, use_bfloat16)
        print(
            f"epoch {epoch}: loss={average_loss:.4f} val_acc={validation_accuracy:.4f}"
        )

    return model
