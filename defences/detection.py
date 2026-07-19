"""Threshold selection, detection metrics, and model behavior metrics.

PSBD flags a sample as poisoned when its PSU falls below a threshold set from
the clean validation quantile. Detection quality is reported as TPR, FPR, and a
threshold-free AUROC.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from .inference import forward_probs


def threshold_from_validation(validation_scores: torch.Tensor, quantile: float) -> float:
    """The detection threshold is a low quantile of clean validation PSU.

    Setting the threshold this way needs no backdoor knowledge and reads as the
    tolerable false-positive rate on clean data.
    """
    return float(torch.quantile(validation_scores.float(), quantile).item())


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
    labels = np.concatenate([np.zeros(len(clean_scores)), np.ones(len(backdoor_scores))])
    return float(roc_auc_score(labels, scores))


@torch.inference_mode()
def _prediction_accuracy(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_bfloat16: bool,
) -> float:
    """Fraction of loader samples whose argmax matches the loader label."""
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        labels = labels.to(device).long()
        predictions = forward_probs(model, images, device, use_bfloat16).argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
    return correct / total if total > 0 else 0.0


def attack_success_rate(
    model: nn.Module,
    backdoor_loader: DataLoader,
    device: torch.device,
    use_bfloat16: bool,
) -> float:
    """Backdoor loader carries the trigger label, so accuracy on it is the ASR."""
    return _prediction_accuracy(model, backdoor_loader, device, use_bfloat16)


def clean_accuracy(
    model: nn.Module,
    clean_loader: DataLoader,
    device: torch.device,
    use_bfloat16: bool,
) -> float:
    """Accuracy on the untriggered clean counterparts."""
    return _prediction_accuracy(model, clean_loader, device, use_bfloat16)


@torch.inference_mode()
def class_correct_and_total(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    use_bfloat16: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One forward pass over loader, per-class correct and total prediction counts.

    The shared computation clean_accuracy_by_class needs, exposed publicly so a
    caller that wants both the pooled and per-class accuracy on the same loader
    (see metrics.py's evaluate_benign) can compute this once instead of running
    the pass twice.
    """
    model.eval()
    correct = torch.zeros(num_classes)
    total = torch.zeros(num_classes)
    for images, labels in loader:
        labels = labels.to(device).long()
        predictions = forward_probs(model, images, device, use_bfloat16).argmax(dim=1)
        for label in range(num_classes):
            mask = labels == label
            total[label] += mask.sum().item()
            correct[label] += (predictions[mask] == label).sum().item()
    return correct, total


def accuracy_by_class_from_counts(correct: torch.Tensor, total: torch.Tensor) -> dict[int, float]:
    return {
        label: (correct[label] / total[label]).item() if total[label] > 0 else 0.0
        for label in range(len(total))
    }


def pooled_accuracy_from_counts(correct: torch.Tensor, total: torch.Tensor) -> float:
    # Count-weighted (micro) average, not a mean of per-class accuracies, so an
    # imbalanced test set (GTSRB) still gets the true pooled accuracy.
    return (correct.sum() / total.sum()).item() if total.sum() > 0 else 0.0


def clean_accuracy_by_class(
    model: nn.Module,
    clean_loader: DataLoader,
    device: torch.device,
    num_classes: int,
    use_bfloat16: bool,
) -> dict[int, float]:
    """Per-class accuracy on the untriggered clean counterparts.

    A benign model's overall accuracy can hide a class the model never learned,
    so this is reported alongside the pooled clean_accuracy rather than instead
    of it.
    """
    correct, total = class_correct_and_total(model, clean_loader, device, num_classes, use_bfloat16)
    return accuracy_by_class_from_counts(correct, total)
