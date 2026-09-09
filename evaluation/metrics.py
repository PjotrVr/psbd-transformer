"""Model behaviour metrics and the atomic evaluation of one checkpoint.

Two concerns live here, folded together because one is only ever used through the
other: the counting primitives that turn a loader into accuracy numbers, and the
3 layers of evaluation built on top of them.

Counting primitives. Accuracy on a loader is a single pass and a comparison, and
what it means depends entirely on which loader it is handed. On a clean loader it
is clean accuracy. On an AttackSuccessSet, whose labels are the attack's intended
labels, the same number is the attack success rate. Both names exist so a call
site says which question it asked.

Evaluation layers, thinnest to widest:
  evaluate_benign / evaluate_attack   model in, metrics out, no filesystem at
                                      all, so a training script can call these
                                      directly on the model it just trained, no
                                      save-then-reload needed.
  evaluate_checkpoint                 a checkpoint path in, metrics out. Loads
                                      the model and its args.json sidecar, then
                                      delegates to one of the 2 above.

There is deliberately no third, directory-walking layer here. Looping over the
whole checkpoints/ tree is orchestration at a higher altitude than "one checkpoint
path in, one metrics dict out", so it belongs to an entrypoint, not to this
module. Its absence is by design.

Every directory this module touches is a parameter with a plain default, not a
bare module-level constant read inside a function body, so any of these are safe
to call against a different tree (a test fixture, a scratch export) just by
passing a different argument.

The detection helpers at the end (threshold, TPR/FPR, AUROC) are the standalone
form of the same rule defences.decision.detection_report applies inside a sweep. They
are kept because a caller holding 2 score tensors and 1 quantile should not have
to build a report dict to ask 1 question. The 2 quantile implementations are NOT
interchangeable: this one uses torch.quantile and defences.decision uses numpy's, and
the 2 differ in the last bits on some inputs.
"""

import json
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from attacks import build_attack, default_config
from data.registry import DATASET_REGISTRY
from .loaders import build_clean_loader, build_poisoned_loader
from defences.inference import forward_probs
from models.backbones import load_checkpoint
from analysis.stealth import cached_stealth_metrics


@torch.inference_mode()
def prediction_accuracy(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_bfloat16: bool,
) -> float:
    """Fraction of loader samples whose argmax matches the loader label.

    What this measures is decided by the loader, not by this function. See
    clean_accuracy and attack_success_rate, which are the 2 questions it answers.
    """
    model.eval()

    correct = 0
    total = 0
    for images, labels in loader:
        labels = labels.to(device).long()  # (batch,)
        predictions = forward_probs(model, images, device, use_bfloat16).argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    accuracy = correct / total if total > 0 else 0.0
    return accuracy


@torch.inference_mode()
def attack_success_rate(
    model: nn.Module,
    backdoor_loader: DataLoader,
    device: torch.device,
    use_bfloat16: bool,
    success_labels: tuple[int, ...] | None = None,
) -> float:
    """Backdoor loader carries the trigger label, so accuracy on it is the ASR.

    The loader must be built over attacks.poisoning.AttackSuccessSet, which selects
    samples by eval-time eligibility and labels them by the eval-time intended
    label. Handing this a training-time poisoned set would measure a different
    quantity under the same name, which for a clean-label attack is the exact
    opposite population.

    success_labels widens what counts as a success from one class to a set, which
    a multi-target clean-label attack needs: its trigger is planted on several
    classes at once and predicts the set rather than any member of it, so
    demanding a particular member would understate the attack by roughly its size.
    Note this also raises the chance baseline from 1/K to |set|/K, which is why
    the target set is meant to stay small.
    """
    if not success_labels or len(success_labels) == 1:
        return prediction_accuracy(model, backdoor_loader, device, use_bfloat16)

    targets = torch.tensor(sorted(success_labels), device=device)
    model.eval()
    hit = 0
    total = 0
    for images, labels in backdoor_loader:
        predictions = forward_probs(model, images, device, use_bfloat16).argmax(dim=1)
        hit += torch.isin(predictions, targets).sum().item()
        total += labels.size(0)
    return hit / total if total > 0 else 0.0


def clean_accuracy(
    model: nn.Module,
    clean_loader: DataLoader,
    device: torch.device,
    use_bfloat16: bool,
) -> float:
    """Accuracy on the untriggered clean counterparts."""
    accuracy = prediction_accuracy(model, clean_loader, device, use_bfloat16)
    return accuracy


@torch.inference_mode()
def class_correct_and_total(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    use_bfloat16: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One pass over loader, returning per-class correct and total counts, both (num_classes,).

    The shared computation clean_accuracy_by_class needs, exposed publicly so a
    caller that wants both the pooled and the per-class accuracy on the same
    loader (see evaluate_benign) can compute this once instead of running the pass
    twice.
    """
    model.eval()

    correct = torch.zeros(num_classes)
    total = torch.zeros(num_classes)
    for images, labels in loader:
        labels = labels.to(device).long()  # (batch,)
        predictions = forward_probs(model, images, device, use_bfloat16).argmax(dim=1)
        for label in range(num_classes):
            mask = labels == label
            total[label] += mask.sum().item()
            correct[label] += (predictions[mask] == label).sum().item()

    return correct, total


def accuracy_by_class_from_counts(
    correct: torch.Tensor, total: torch.Tensor
) -> dict[int, float]:
    """Per-class accuracy, 0.0 for a class the loader never served."""
    by_class = {
        label: (correct[label] / total[label]).item() if total[label] > 0 else 0.0
        for label in range(len(total))
    }
    return by_class


def pooled_accuracy_from_counts(correct: torch.Tensor, total: torch.Tensor) -> float:
    """Overall accuracy from per-class counts."""
    # Count-weighted (micro) average, not a mean of per-class accuracies, so an
    # imbalanced test set (GTSRB) still gets the true pooled accuracy.
    pooled = (correct.sum() / total.sum()).item() if total.sum() > 0 else 0.0
    return pooled


def clean_accuracy_by_class(
    model: nn.Module,
    clean_loader: DataLoader,
    device: torch.device,
    num_classes: int,
    use_bfloat16: bool,
) -> dict[int, float]:
    """Per-class accuracy on the untriggered clean counterparts.

    A benign model's overall accuracy can hide a class the model never learned, so
    this is reported alongside the pooled clean_accuracy rather than instead of it.
    """
    correct, total = class_correct_and_total(
        model, clean_loader, device, num_classes, use_bfloat16
    )

    by_class = accuracy_by_class_from_counts(correct, total)
    return by_class


def evaluate_benign(
    model: nn.Module,
    dataset_name: str,
    device: torch.device,
    raw_data_dir: str = "raw_data",
    batch_size: int = 64,
    max_samples: int | None = None,
    seed: int = 0,
) -> dict:
    """Clean accuracy, total and per class, for a model with no attack of its own."""
    loader = build_clean_loader(
        dataset_name, raw_data_dir, batch_size, max_samples=max_samples, seed=seed
    )
    num_classes = DATASET_REGISTRY[dataset_name].num_classes

    correct, total = class_correct_and_total(
        model, loader, device, num_classes, use_bfloat16=True
    )
    metrics = {
        "clean_accuracy": pooled_accuracy_from_counts(correct, total),
        "clean_accuracy_by_class": accuracy_by_class_from_counts(correct, total),
    }
    return metrics


def evaluate_attack(
    model: nn.Module,
    dataset_name: str,
    attack_name: str,
    config,
    target_label: int,
    device: torch.device,
    raw_data_dir: str = "raw_data",
    batch_size: int = 64,
    max_samples: int | None = None,
    seed: int = 0,
) -> dict:
    """Attack success rate and clean accuracy, for a model probed with one attack.

    config is the attack's own config dataclass, for example
    BadNetConfig(label_mode="all_to_all") from attacks.badnet, or whatever
    attacks.default_config(attack_name) returns. Applies the trigger to every
    eligible test image, never a poison_rate sample of it: poison_rate only
    controls how many training images get poisoned, eval always asks about the
    whole eligible test set.
    """
    image_size = DATASET_REGISTRY[dataset_name].image_size
    attack = build_attack(attack_name, config, image_size, target_label)

    clean_loader = build_clean_loader(
        dataset_name, raw_data_dir, batch_size, max_samples=max_samples, seed=seed
    )
    poisoned_loader = build_poisoned_loader(
        dataset_name,
        attack,
        raw_data_dir,
        batch_size,
        max_samples=max_samples,
        seed=seed,
    )

    metrics = {
        "asr": attack_success_rate(model, poisoned_loader, device, use_bfloat16=True),
        "clean_accuracy": clean_accuracy(
            model, clean_loader, device, use_bfloat16=True
        ),
    }
    return metrics


def read_args_json(checkpoint_dir: str) -> dict:
    """The training-provenance sidecar written next to a checkpoint."""
    with open(os.path.join(checkpoint_dir, "args.json")) as handle:
        return json.load(handle)


def evaluate_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    raw_data_dir: str = "raw_data",
    batch_size: int = 64,
) -> dict:
    """Load one checkpoint and its args.json, evaluate it as benign or under its own attack."""
    args = read_args_json(os.path.dirname(checkpoint_path))
    model = load_checkpoint(args["architecture"], checkpoint_path, device)
    folder_name = os.path.basename(os.path.dirname(checkpoint_path))

    if args["attack"] == "benign":
        benign_metrics = evaluate_benign(
            model, args["dataset"], device, raw_data_dir, batch_size
        )
        return {
            "folder_name": folder_name,
            "architecture": args["architecture"],
            "dataset": args["dataset"],
            **benign_metrics,
        }

    config = default_config(args["attack"])
    attack = build_attack(
        args["attack"],
        config,
        DATASET_REGISTRY[args["dataset"]].image_size,
        args["target_label"],
    )
    metrics = evaluate_attack(
        model,
        args["dataset"],
        args["attack"],
        config,
        args["target_label"],
        device,
        raw_data_dir,
        batch_size,
    )
    stealth = cached_stealth_metrics(
        args["dataset"], args["attack"], attack, raw_data_dir, device
    )
    report = {
        "folder_name": folder_name,
        "architecture": args["architecture"],
        "dataset": args["dataset"],
        "attack": args["attack"],
        "label_mode": args["label_mode"],
        "poison_rate": args["poison_rate"],
        "target_label": args["target_label"],
        **metrics,
        "stealth": stealth,
    }
    return report


def save_metrics(output_path: str, metrics: dict) -> None:
    """Write one metrics dict as JSON, creating its directory."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as handle:
        json.dump(metrics, handle, indent=2)


def threshold_from_validation(
    validation_scores: torch.Tensor, quantile: float
) -> float:
    """The detection threshold is a low quantile of clean validation PSU.

    Setting the threshold this way needs no backdoor knowledge and reads as the
    tolerable false-positive rate on clean data.
    """
    threshold = float(torch.quantile(validation_scores.float(), quantile).item())
    return threshold


def detection_rates(
    clean_scores: torch.Tensor,
    backdoor_scores: torch.Tensor,
    threshold: float,
) -> tuple[float, float]:
    """Return (tpr, fpr) at the threshold, flagging PSU below it as poisoned."""
    tpr = (backdoor_scores < threshold).float().mean().item()
    fpr = (clean_scores < threshold).float().mean().item()
    return float(tpr), float(fpr)


def auroc(clean_scores: torch.Tensor, backdoor_scores: torch.Tensor) -> float:
    """Threshold-free separability. PSU is negated because lower means poisoned."""
    scores = np.concatenate([-clean_scores.numpy(), -backdoor_scores.numpy()])
    labels = np.concatenate(
        [np.zeros(len(clean_scores)), np.ones(len(backdoor_scores))]
    )

    area = float(roc_auc_score(labels, scores))
    return area
