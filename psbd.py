from __future__ import annotations

import glob
import json
import os

import torch
import torch.nn.functional as F

from benign_train_resnet import LightningResNetV2
from defenses import DefenseBase
from utils import get_timestamp, load_tensor

DATASET_NAME = "cifar10"
ATTACK_NAME = "badnet"
POISON_RATE = 0.1
TARGET_LABEL = 0

DROP_RATE = 0.3
MC_SAMPLES = 20
BATCH_SIZE = 256
THRESHOLD_CLEAN_FPR = 0.05


def resolve_poison_data_dir() -> str:
    data_dir = os.path.join(
        "preprocessed_data",
        f"{DATASET_NAME}_{ATTACK_NAME}_poison_rate={POISON_RATE}_target={TARGET_LABEL}",
    )
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Poison data directory not found: {data_dir}")
    return data_dir


def resolve_checkpoint_path() -> str:
    env_path = os.environ.get("PSBD_MODEL_CKPT", "").strip()
    if env_path:
        if not os.path.isfile(env_path):
            raise FileNotFoundError(f"PSBD_MODEL_CKPT does not exist: {env_path}")
        return env_path

    all_checkpoints = glob.glob(os.path.join("runs", "**", "*.ckpt"), recursive=True)
    if len(all_checkpoints) == 0:
        raise FileNotFoundError(
            "No .ckpt file found under runs/. Set PSBD_MODEL_CKPT to a checkpoint path."
        )

    all_checkpoints.sort(key=os.path.getmtime, reverse=True)
    return all_checkpoints[0]


def forward_with_psbd_dropout(model, x: torch.Tensor, drop_rate: float) -> torch.Tensor:
    if not all(
        hasattr(model, name)
        for name in ["conv1", "layer1", "layer2", "layer3", "layer4", "bn", "fc"]
    ):
        raise ValueError(
            "PSBD dropout forward expects the ResNetV2-style model structure"
        )

    out = model.conv1(x)
    out = F.dropout2d(out, p=drop_rate, training=True)

    out = model.layer1(out)
    out = F.dropout2d(out, p=drop_rate, training=True)

    out = model.layer2(out)
    out = F.dropout2d(out, p=drop_rate, training=True)

    out = model.layer3(out)
    out = F.dropout2d(out, p=drop_rate, training=True)

    out = model.layer4(out)
    out = F.dropout2d(out, p=drop_rate, training=True)

    out = F.relu(model.bn(out))
    out = F.avg_pool2d(out, 4)
    out = out.view(out.size(0), -1)
    out = F.dropout(out, p=drop_rate, training=True)
    out = model.fc(out)
    return out


def compute_prediction_shift_scores(
    model,
    data: torch.Tensor,
    target_label: int,
    batch_size: int,
    mc_samples: int,
    drop_rate: float,
    device: str,
) -> dict:
    model = model.to(device)
    model.eval()

    all_target_shift_scores = []
    all_total_variation_scores = []

    total_count = int(data.shape[0])
    with torch.no_grad():
        for start in range(0, total_count, batch_size):
            end = min(start + batch_size, total_count)
            batch = data[start:end].to(device)

            base_logits = model(batch)
            base_probs = torch.softmax(base_logits, dim=1)

            mc_probs = []
            for _ in range(mc_samples):
                dropout_logits = forward_with_psbd_dropout(
                    model, batch, drop_rate=drop_rate
                )
                dropout_probs = torch.softmax(dropout_logits, dim=1)
                mc_probs.append(dropout_probs)

            mean_dropout_probs = torch.stack(mc_probs, dim=0).mean(dim=0)

            target_shift = (
                mean_dropout_probs[:, target_label] - base_probs[:, target_label]
            )
            total_variation = torch.abs(mean_dropout_probs - base_probs).sum(dim=1)

            all_target_shift_scores.append(target_shift.detach().cpu())
            all_total_variation_scores.append(total_variation.detach().cpu())

    target_shift_scores = torch.cat(all_target_shift_scores, dim=0)
    total_variation_scores = torch.cat(all_total_variation_scores, dim=0)
    combined_scores = target_shift_scores + total_variation_scores

    return {
        "target_shift": target_shift_scores,
        "total_variation": total_variation_scores,
        "combined": combined_scores,
    }


def quantile_threshold(scores: torch.Tensor, clean_fpr: float) -> float:
    q = max(0.0, min(1.0, 1.0 - float(clean_fpr)))
    threshold = torch.quantile(scores, q=q)
    return float(threshold.item())


class PSBDDefense(DefenseBase):
    def __init__(
        self,
        model,
        target_label: int,
        drop_rate: float = 0.3,
        mc_samples: int = 20,
        batch_size: int = 256,
        clean_fpr: float = 0.05,
        device: str | None = None,
    ):
        self.model = model
        self.target_label = int(target_label)
        self.drop_rate = float(drop_rate)
        self.mc_samples = int(mc_samples)
        self.batch_size = int(batch_size)
        self.clean_fpr = float(clean_fpr)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = None

    def fit(self, calibration_clean_data: torch.Tensor) -> float:
        score_pack = compute_prediction_shift_scores(
            model=self.model,
            data=calibration_clean_data,
            target_label=self.target_label,
            batch_size=self.batch_size,
            mc_samples=self.mc_samples,
            drop_rate=self.drop_rate,
            device=self.device,
        )
        self.threshold = quantile_threshold(
            score_pack["combined"], clean_fpr=self.clean_fpr
        )
        return self.threshold

    def predict(self, data: torch.Tensor) -> dict:
        if self.threshold is None:
            raise RuntimeError("Call fit() before predict().")

        score_pack = compute_prediction_shift_scores(
            model=self.model,
            data=data,
            target_label=self.target_label,
            batch_size=self.batch_size,
            mc_samples=self.mc_samples,
            drop_rate=self.drop_rate,
            device=self.device,
        )

        combined_scores = score_pack["combined"]
        decisions = combined_scores > float(self.threshold)
        return {
            "scores": combined_scores,
            "decisions": decisions,
            "score_pack": score_pack,
        }


def bool_mean(mask: torch.Tensor) -> float:
    if int(mask.numel()) == 0:
        return 0.0
    return float(mask.float().mean().item())


def evaluate_psbd(
    defense: PSBDDefense,
    clean_data: torch.Tensor,
    clean_labels: torch.Tensor,
    backdoor_data: torch.Tensor,
    backdoor_labels: torch.Tensor,
) -> dict:
    clean_pred = defense.predict(clean_data)
    backdoor_pred = defense.predict(backdoor_data)

    clean_flags = clean_pred["decisions"]
    backdoor_flags = backdoor_pred["decisions"]

    changed_mask = backdoor_labels != clean_labels
    unchanged_mask = ~changed_mask

    targeted_flags = (
        backdoor_flags[changed_mask]
        if int(changed_mask.sum().item()) > 0
        else torch.tensor([])
    )

    return {
        "clean_fpr": bool_mean(clean_flags),
        "backdoor_detection_rate": bool_mean(backdoor_flags),
        "targeted_detection_rate": bool_mean(targeted_flags),
        "changed_label_count": int(changed_mask.sum().item()),
        "unchanged_label_count": int(unchanged_mask.sum().item()),
        "clean_score_mean": float(clean_pred["scores"].mean().item()),
        "backdoor_score_mean": float(backdoor_pred["scores"].mean().item()),
    }


def main():
    data_dir = resolve_poison_data_dir()
    checkpoint_path = resolve_checkpoint_path()

    clean_val_data = load_tensor(data_dir, "clean_val_data.pt")
    clean_val_labels = load_tensor(data_dir, "clean_val_labels.pt")
    backdoor_val_data = load_tensor(data_dir, "backdoor_val_data.pt")
    backdoor_val_labels = load_tensor(data_dir, "backdoor_val_labels.pt")

    clean_test_data = load_tensor(data_dir, "clean_test_data.pt")
    clean_test_labels = load_tensor(data_dir, "clean_test_labels.pt")
    backdoor_test_data = load_tensor(data_dir, "backdoor_test_data.pt")
    backdoor_test_labels = load_tensor(data_dir, "backdoor_test_labels.pt")

    if os.path.exists(os.path.join(data_dir, "val_heldout_data.pt")):
        calibration_clean_data = load_tensor(data_dir, "val_heldout_data.pt")
    else:
        calibration_clean_data = clean_val_data

    lightning_model = LightningResNetV2.load_from_checkpoint(checkpoint_path)
    base_model = lightning_model.model

    defense = PSBDDefense(
        model=base_model,
        target_label=TARGET_LABEL,
        drop_rate=DROP_RATE,
        mc_samples=MC_SAMPLES,
        batch_size=BATCH_SIZE,
        clean_fpr=THRESHOLD_CLEAN_FPR,
    )

    threshold = defense.fit(calibration_clean_data)

    val_metrics = evaluate_psbd(
        defense=defense,
        clean_data=clean_val_data,
        clean_labels=clean_val_labels,
        backdoor_data=backdoor_val_data,
        backdoor_labels=backdoor_val_labels,
    )
    test_metrics = evaluate_psbd(
        defense=defense,
        clean_data=clean_test_data,
        clean_labels=clean_test_labels,
        backdoor_data=backdoor_test_data,
        backdoor_labels=backdoor_test_labels,
    )

    output = {
        "dataset": DATASET_NAME,
        "attack": ATTACK_NAME,
        "poison_rate": POISON_RATE,
        "target_label": TARGET_LABEL,
        "checkpoint_path": checkpoint_path,
        "drop_rate": DROP_RATE,
        "mc_samples": MC_SAMPLES,
        "batch_size": BATCH_SIZE,
        "threshold_clean_fpr": THRESHOLD_CLEAN_FPR,
        "threshold": threshold,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    run_dir = os.path.join(
        "runs", f"psbd_{DATASET_NAME}_{ATTACK_NAME}_{get_timestamp()}"
    )
    os.makedirs(run_dir, exist_ok=True)
    output_path = os.path.join(run_dir, "psbd_metrics.json")

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)

    print("PSBD results saved to:", output_path)
    print("Validation metrics:", val_metrics)
    print("Test metrics:", test_metrics)


if __name__ == "__main__":
    main()
