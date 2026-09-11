"""Model behaviour metrics, and the evaluation of a single checkpoint.

Accuracy on a loader is a single pass and a comparison, and what it means depends
on the loader. On a clean loader it is clean accuracy. On an AttackSuccessSet,
whose labels are the attack's intended labels, the same number is the attack
success rate. Both names exist so a call site says which question it asked.

3 evaluation layers sit on top, thinnest first. evaluate_benign and
evaluate_attack take a model and return metrics with no filesystem involved, so a
training script can call them on the model it just trained. evaluate_checkpoint
takes a checkpoint path, loads the model and its args.json and delegates to
either. Walking the whole checkpoints/ tree is cli.evaluate's job, not this
module's. Every directory is a parameter with a plain default, so any of these
runs against a test fixture by passing a different argument.

The detection helpers at the end (threshold, TPR and FPR, AUROC) are the
standalone form of the rule defences.decision.detection_report applies inside a
sweep, for a caller holding 2 score tensors and a quantile. The 2 quantile
implementations are not interchangeable: this one uses torch.quantile, decision
uses numpy's, and they differ in the last bits on some inputs.
"""

import json
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from attacks import build_attack, default_config
from attacks.poisoning import clean_label_target_set
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
        probs = forward_probs(
            model, images, device, use_bfloat16
        )  # (batch, num_classes)
        predictions = probs.argmax(dim=1)  # (batch,)
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
    """Accuracy on the backdoor loader, which is the ASR since its labels are the intended ones.

    The loader must be built over attacks.poisoning.AttackSuccessSet, which selects
    samples by eval-time eligibility and labels them by the eval-time intended
    label. A training-time poisoned set would measure a different quantity under
    the same name, for a clean-label attack the exact opposite population.

    success_labels widens a success from a single class to a set, which a
    multi-target clean-label attack needs: its trigger predicts the set rather than
    any member, so demanding a particular member would understate the attack by
    roughly the set's size. It also raises the chance baseline from 1/K to |set|/K,
    which is why the target set stays small.
    """
    if not success_labels or len(success_labels) == 1:
        accuracy = prediction_accuracy(model, backdoor_loader, device, use_bfloat16)
        return accuracy

    targets = torch.tensor(sorted(success_labels), device=device)  # (num_targets,)
    model.eval()
    hit = 0
    total = 0
    for images, labels in backdoor_loader:
        probs = forward_probs(
            model, images, device, use_bfloat16
        )  # (batch, num_classes)
        predictions = probs.argmax(dim=1)  # (batch,)
        hit += torch.isin(predictions, targets).sum().item()
        total += labels.size(0)

    rate = hit / total if total > 0 else 0.0
    return rate


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
    """Per-class correct and total counts from a single pass, both (num_classes,).

    Exposed so a caller that wants both the pooled and the per-class accuracy on
    the same loader (evaluate_benign) runs the pass once.
    """
    model.eval()

    correct = torch.zeros(num_classes)  # (num_classes,)
    total = torch.zeros(num_classes)  # (num_classes,)
    for images, labels in loader:
        labels = labels.to(device).long()  # (batch,)
        probs = forward_probs(
            model, images, device, use_bfloat16
        )  # (batch, num_classes)
        predictions = probs.argmax(dim=1)  # (batch,)
        for label in range(num_classes):
            mask = labels == label  # (batch,) bool
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
    """Attack success rate and clean accuracy for a model under a single attack.

    config is the attack's own config dataclass, for example
    BadNetConfig(label_mode="all_to_all") or whatever default_config(attack_name)
    returns. The trigger goes on every eligible test image, never a poison_rate
    sample of it.
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

    # A multi-target clean-label attack succeeds by landing anywhere in its target
    # set, so the ASR reader is told the set rather than a single class. This is
    # the definition every clean_label_multi args.json on disk was scored with.
    success_labels = (
        clean_label_target_set(attack.target_label, attack.num_targets)
        if attack.label_mode == "clean_label_multi"
        else None
    )

    metrics = {
        "asr": attack_success_rate(
            model,
            poisoned_loader,
            device,
            use_bfloat16=True,
            success_labels=success_labels,
        ),
        "clean_accuracy": clean_accuracy(
            model, clean_loader, device, use_bfloat16=True
        ),
    }
    return metrics


def read_args_json(checkpoint_dir: str) -> dict:
    """The training-provenance sidecar written next to a checkpoint."""
    with open(os.path.join(checkpoint_dir, "args.json")) as handle:
        args = json.load(handle)
    return args


def evaluate_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    raw_data_dir: str = "raw_data",
    batch_size: int = 64,
) -> dict:
    """Metrics for a checkpoint, as benign or under its own attack, read from its args.json."""
    args = read_args_json(os.path.dirname(checkpoint_path))
    model = load_checkpoint(args["architecture"], checkpoint_path, device)
    folder_name = os.path.basename(os.path.dirname(checkpoint_path))

    if args["attack"] == "benign":
        benign_metrics = evaluate_benign(
            model, args["dataset"], device, raw_data_dir, batch_size
        )
        report = {
            "folder_name": folder_name,
            "architecture": args["architecture"],
            "dataset": args["dataset"],
            **benign_metrics,
        }
        return report

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
    """Write a metrics dict as JSON, creating its directory."""
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
    """(tpr, fpr) at the threshold, flagging PSU below it as poisoned."""
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


def confusion_matrix(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    normalise: bool = False,
) -> torch.Tensor:
    """Counts of (true class, predicted class), (num_classes, num_classes) float32.

    Row i is the true class and column j the prediction, as
    visual_utils.plot_confusion_matrix (lines 627 to 632) fills it. With
    normalise the rows are divided by their totals plus 1e-24, upstream's guard
    against a class with no rows, so an empty row stays 0 rather than NaN.
    """
    if predictions.shape != labels.shape:
        raise ValueError(
            f"predictions {tuple(predictions.shape)} and labels {tuple(labels.shape)} "
            "must be the same length"
        )

    flat = labels.long() * num_classes + predictions.long()  # (N,)
    counts = torch.bincount(flat, minlength=num_classes * num_classes).float()
    matrix = counts.view(num_classes, num_classes)  # (num_classes, num_classes)
    if not normalise:
        return matrix

    normalised = matrix / (matrix.sum(dim=1, keepdim=True) + 1e-24)
    return normalised


def defense_effectiveness_rate(
    acc_bd: float, acc_def: float, asr_bd: float, asr_def: float
) -> float:
    """DER of a defence against the backdoored model it started from, in [0, 1].

    BackdoorBench's utils/metric.py ships 2 versions that disagree in sign.
    defense_effectiveness_rate (line 64) ADDS the clean-accuracy drop, so a
    defence that destroys clean accuracy scores higher, while
    defense_effectiveness_rate_simplied (line 94) subtracts it, which is the
    version visual_metric.py calls and the paper's definition. The simplified
    variant is implemented, quoted here exactly:

        return (max(0, asr_bd - asr_defense) - max(0, acc_bd - acc_defnese) + 1) / 2

    symbol table
        asr_bd, asr_def    attack success rate before and after the defence
        acc_bd, acc_def    clean accuracy before and after the defence
    """
    rate = (max(0.0, asr_bd - asr_def) - max(0.0, acc_bd - acc_def) + 1.0) / 2.0
    return rate


def robust_improvement_rate(
    acc_bd: float, acc_def: float, ra_bd: float, ra_def: float
) -> float:
    """RIR of a defence against the backdoored model it started from, in [0, 1].

    The same 2 versions exist for RIR. robust_improvement_rate (utils/metric.py
    line 80) adds the clean-accuracy drop, robust_improvement_rate_simplied
    (line 98) subtracts it and is the version visual_metric.py calls. The
    simplified variant is implemented, quoted here exactly:

        return (max(0, -ra_bd + ra_defense) - max(0, acc_bd - acc_defnese) + 1) / 2

    symbol table
        ra_bd, ra_def      robust accuracy before and after the defence, the
                           accuracy on triggered images against their true label
        acc_bd, acc_def    clean accuracy before and after the defence
    """
    rate = (max(0.0, ra_def - ra_bd) - max(0.0, acc_bd - acc_def) + 1.0) / 2.0
    return rate
