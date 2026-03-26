from __future__ import annotations

import json
import os

import torch
import torch.nn.functional as F


def predict_labels(
    model,
    data: torch.Tensor,
    batch_size: int = 256,
    device: str | None = None,
) -> torch.Tensor:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()

    predictions = []
    with torch.no_grad():
        total = int(data.shape[0])
        for start in range(0, total, int(batch_size)):
            end = min(start + int(batch_size), total)
            batch = data[start:end].to(device)
            logits = model(batch)
            batch_predictions = torch.argmax(logits, dim=1)
            predictions.append(batch_predictions.cpu())

    return torch.cat(predictions, dim=0)


def evaluate_accuracy_and_loss(
    model,
    data: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int = 256,
    device: str | None = None,
) -> dict:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = int(labels.numel())

    with torch.no_grad():
        for start in range(0, total_count, int(batch_size)):
            end = min(start + int(batch_size), total_count)
            batch_data = data[start:end].to(device)
            batch_labels = labels[start:end].to(device)

            logits = model(batch_data)
            loss = F.cross_entropy(logits, batch_labels, reduction="sum")
            batch_predictions = torch.argmax(logits, dim=1)

            total_loss += float(loss.item())
            total_correct += int((batch_predictions == batch_labels).sum().item())

    if total_count == 0:
        return {"loss": 0.0, "accuracy": 0.0}

    return {
        "loss": float(total_loss / total_count),
        "accuracy": float(total_correct / total_count),
    }


def compute_targeted_asr(
    clean_labels: torch.Tensor,
    backdoor_labels: torch.Tensor,
    backdoor_predictions: torch.Tensor,
) -> dict:
    if clean_labels.shape != backdoor_labels.shape:
        raise ValueError("clean_labels and backdoor_labels must have identical shape")
    if backdoor_labels.shape != backdoor_predictions.shape:
        raise ValueError(
            "backdoor_labels and backdoor_predictions must have identical shape"
        )

    changed_mask = backdoor_labels != clean_labels
    unchanged_mask = ~changed_mask

    changed_count = int(changed_mask.sum().item())
    unchanged_count = int(unchanged_mask.sum().item())

    if changed_count == 0:
        targeted_asr = 0.0
    else:
        targeted_success = (
            backdoor_predictions[changed_mask] == backdoor_labels[changed_mask]
        ).float()
        targeted_asr = float(targeted_success.mean().item())

    return {
        "targeted_asr": targeted_asr,
        "changed_label_count": changed_count,
        "unchanged_label_count": unchanged_count,
    }


def evaluate_backdoor_pair(
    model,
    clean_data: torch.Tensor,
    clean_labels: torch.Tensor,
    backdoor_data: torch.Tensor,
    backdoor_labels: torch.Tensor,
    batch_size: int = 256,
    device: str | None = None,
) -> dict:
    clean_metrics = evaluate_accuracy_and_loss(
        model=model,
        data=clean_data,
        labels=clean_labels,
        batch_size=batch_size,
        device=device,
    )

    backdoor_metrics = evaluate_accuracy_and_loss(
        model=model,
        data=backdoor_data,
        labels=backdoor_labels,
        batch_size=batch_size,
        device=device,
    )

    backdoor_predictions = predict_labels(
        model=model,
        data=backdoor_data,
        batch_size=batch_size,
        device=device,
    )

    targeted = compute_targeted_asr(
        clean_labels=clean_labels.cpu(),
        backdoor_labels=backdoor_labels.cpu(),
        backdoor_predictions=backdoor_predictions.cpu(),
    )

    result = {
        "clean_accuracy": clean_metrics["accuracy"],
        "clean_loss": clean_metrics["loss"],
        "asr": backdoor_metrics["accuracy"],
        "backdoor_loss": backdoor_metrics["loss"],
        "targeted_asr": targeted["targeted_asr"],
        "changed_label_count": targeted["changed_label_count"],
        "unchanged_label_count": targeted["unchanged_label_count"],
    }
    return result


def save_metrics_json(metrics: dict, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
