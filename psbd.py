from __future__ import annotations

import argparse
import json
import os

import torch
import torch.nn.functional as F
from torch import nn

from torch.utils.data import DataLoader

from models import vgg13, vgg16, vgg19
from models.resnet_v2 import PreActBlock, ResNetV2PSBD
from utils import get_timestamp, load_tensor, tensor_loader


class VGGPSBD(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.psbd_dropout_rate = 0.0
        self.use_inference_dropout = False

    def set_inference_dropout(
        self, enabled: bool, dropout_rate: float | None = None
    ) -> None:
        self.use_inference_dropout = bool(enabled)
        if dropout_rate is not None:
            self.psbd_dropout_rate = float(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.backbone(x)
        if (
            (not self.training)
            and self.use_inference_dropout
            and self.psbd_dropout_rate > 0.0
        ):
            logits = F.dropout(logits, p=self.psbd_dropout_rate, training=True)
        return logits


def clean_checkpoint_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            cleaned[key[len("model.") :]] = value
        else:
            cleaned[key] = value
    return cleaned


def load_model_from_checkpoint(
    checkpoint_path: str,
    model_name: str,
    num_classes: int,
    in_channels: int = 3,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> nn.Module:
    model_name = model_name.lower().strip()

    if model_name == "resnet18":
        return ResNetV2PSBD.from_checkpoint(
            checkpoint_path=checkpoint_path,
            block=PreActBlock,
            num_blocks=[2, 2, 2, 2],
            num_classes=num_classes,
            in_channels=in_channels,
            map_location=map_location,
            strict=strict,
        )

    vgg_factories = {
        "vgg13": vgg13,
        "vgg16": vgg16,
        "vgg19": vgg19,
    }
    if model_name not in vgg_factories:
        raise ValueError(
            "Unsupported model_name. Expected one of: resnet18, vgg13, vgg16, vgg19"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False,
    )
    state = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    state = clean_checkpoint_state_dict(state)

    backbone = vgg_factories[model_name](
        num_classes=num_classes, in_channels=in_channels
    )
    backbone.load_state_dict(state, strict=strict)
    return VGGPSBD(backbone)


def set_psbd_dropout_mode(model: nn.Module, enabled: bool, dropout_rate: float):
    dropout_modules = (
        nn.Dropout,
        nn.Dropout1d,
        nn.Dropout2d,
        nn.Dropout3d,
        nn.AlphaDropout,
        nn.FeatureAlphaDropout,
    )

    if hasattr(model, "set_inference_dropout"):
        previous_enabled = bool(getattr(model, "use_inference_dropout", False))
        previous_rate = float(getattr(model, "psbd_dropout_rate", 0.0))

        if enabled:
            model.set_inference_dropout(True, dropout_rate=float(dropout_rate))
        else:
            model.set_inference_dropout(False)

        def restore_custom():
            model.set_inference_dropout(previous_enabled, dropout_rate=previous_rate)

        return restore_custom

    states = []
    for module in model.modules():
        if isinstance(module, dropout_modules):
            previous_p = float(module.p) if hasattr(module, "p") else None
            states.append((module, bool(module.training), previous_p))
            module.train(enabled)
            if enabled and previous_p is not None:
                module.p = float(dropout_rate)

    def restore_generic():
        for module, was_training, previous_p in states:
            module.train(was_training)
            if previous_p is not None:
                module.p = float(previous_p)

    return restore_generic


@torch.no_grad()
def compute_psbd_scores(
    model: nn.Module,
    loader: DataLoader,
    target_label: int,
    mc_samples: int = 10,
    dropout_rate: float = 0.5,
    device: str | torch.device = "cpu",
) -> dict:
    model = model.to(device)
    model.eval()

    all_target_shift = []
    all_total_variation = []

    for batch_x, _ in loader:
        batch_x = batch_x.to(device)

        restore_base = set_psbd_dropout_mode(
            model, enabled=False, dropout_rate=float(dropout_rate)
        )
        base_probs = torch.softmax(model(batch_x), dim=1)
        restore_base()

        restore_mc = set_psbd_dropout_mode(
            model, enabled=True, dropout_rate=float(dropout_rate)
        )
        mc_prob_list = []
        for _ in range(int(mc_samples)):
            mc_prob_list.append(torch.softmax(model(batch_x), dim=1))
        restore_mc()

        mean_mc_probs = torch.stack(mc_prob_list, dim=0).mean(dim=0)

        target_shift = (
            mean_mc_probs[:, int(target_label)] - base_probs[:, int(target_label)]
        )
        total_variation = torch.abs(mean_mc_probs - base_probs).sum(dim=1)

        all_target_shift.append(target_shift.detach().cpu())
        all_total_variation.append(total_variation.detach().cpu())

    target_shift_scores = torch.cat(all_target_shift, dim=0)
    total_variation_scores = torch.cat(all_total_variation, dim=0)
    combined_scores = target_shift_scores + total_variation_scores

    return {
        "target_shift": target_shift_scores,
        "total_variation": total_variation_scores,
        "combined": combined_scores,
    }


def quantile_threshold(clean_scores: torch.Tensor, target_fpr: float) -> float:
    q = max(0.0, min(1.0, float(target_fpr)))
    return float(torch.quantile(clean_scores, q=q).item())


def detection_rate(scores: torch.Tensor, threshold: float) -> float:
    return float((scores < float(threshold)).float().mean().item())


def evaluate_score_pair(
    clean_scores: torch.Tensor,
    backdoor_scores: torch.Tensor,
    threshold: float,
) -> dict:
    return {
        "threshold": float(threshold),
        "clean_fpr": detection_rate(clean_scores, threshold),
        "backdoor_tpr": detection_rate(backdoor_scores, threshold),
    }


@torch.no_grad()
def evaluate_psbd(
    model: nn.Module,
    target_label: int,
    clean_threshold_loader: DataLoader,
    clean_eval_loader: DataLoader,
    backdoor_eval_loader: DataLoader,
    target_fprs: tuple[float, ...] = (0.01, 0.05, 0.1, 0.25),
    mc_samples: int = 10,
    dropout_rate: float = 0.5,
    device: str | torch.device = "cpu",
    score_key: str = "combined",
) -> dict:
    clean_threshold_scores = compute_psbd_scores(
        model=model,
        loader=clean_threshold_loader,
        target_label=target_label,
        mc_samples=mc_samples,
        dropout_rate=dropout_rate,
        device=device,
    )[score_key]

    clean_eval_scores = compute_psbd_scores(
        model=model,
        loader=clean_eval_loader,
        target_label=target_label,
        mc_samples=mc_samples,
        dropout_rate=dropout_rate,
        device=device,
    )[score_key]

    backdoor_eval_scores = compute_psbd_scores(
        model=model,
        loader=backdoor_eval_loader,
        target_label=target_label,
        mc_samples=mc_samples,
        dropout_rate=dropout_rate,
        device=device,
    )[score_key]

    threshold_metrics = {}
    for fpr in target_fprs:
        fpr = float(fpr)
        thr = quantile_threshold(clean_threshold_scores, target_fpr=fpr)
        threshold_metrics[fpr] = evaluate_score_pair(
            clean_scores=clean_eval_scores,
            backdoor_scores=backdoor_eval_scores,
            threshold=thr,
        )

    return {
        "dropout_rate": float(dropout_rate),
        "mc_samples": int(mc_samples),
        "target_label": int(target_label),
        "score_key": score_key,
        "clean_threshold_mean": float(clean_threshold_scores.mean().item()),
        "clean_eval_mean": float(clean_eval_scores.mean().item()),
        "backdoor_eval_mean": float(backdoor_eval_scores.mean().item()),
        "threshold_metrics": threshold_metrics,
    }


@torch.no_grad()
def run_psbd_sweep(
    model: nn.Module,
    target_label: int,
    clean_threshold_loader: DataLoader,
    clean_eval_loader: DataLoader,
    backdoor_eval_loader: DataLoader,
    dropout_rates: tuple[float, ...] = (0.1, 0.2, 0.4, 0.6, 0.8),
    target_fprs: tuple[float, ...] = (0.01, 0.05, 0.1, 0.25),
    mc_samples: int = 10,
    device: str | torch.device = "cpu",
    score_key: str = "combined",
) -> dict:
    by_dropout_rate = {}
    for dropout_rate in dropout_rates:
        dropout_rate = float(dropout_rate)
        by_dropout_rate[dropout_rate] = evaluate_psbd(
            model=model,
            target_label=target_label,
            clean_threshold_loader=clean_threshold_loader,
            clean_eval_loader=clean_eval_loader,
            backdoor_eval_loader=backdoor_eval_loader,
            target_fprs=target_fprs,
            mc_samples=mc_samples,
            dropout_rate=dropout_rate,
            device=device,
            score_key=score_key,
        )

    return {
        "dropout_rates": [float(dropout_rate) for dropout_rate in dropout_rates],
        "target_fprs": [float(fpr) for fpr in target_fprs],
        "mc_samples": int(mc_samples),
        "score_key": score_key,
        "results": by_dropout_rate,
    }


def find_best_dropout_rate(
    sweep_result: dict,
    selection_fpr: float = 0.1,
) -> float:
    selected = float(selection_fpr)
    best_dropout_rate = None
    best_tpr = -1.0

    for dropout_rate, drop_result in sweep_result["results"].items():
        tpr = drop_result["threshold_metrics"][selected]["backdoor_tpr"]
        if tpr > best_tpr:
            best_tpr = float(tpr)
            best_dropout_rate = float(dropout_rate)

    if best_dropout_rate is None:
        raise RuntimeError("Could not select a dropout rate from sweep_result.")
    return best_dropout_rate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PSBD evaluation")
    parser.add_argument("--checkpoint-path", required=True, type=str)
    parser.add_argument(
        "--model-name",
        default="resnet18",
        choices=["resnet18", "vgg13", "vgg16", "vgg19"],
        type=str,
    )
    parser.add_argument("--dataset-name", default="cifar10", type=str)
    parser.add_argument("--attack-name", default="badnet", type=str)
    parser.add_argument("--poison-rate", default=0.1, type=float)
    parser.add_argument("--target-label", default=0, type=int)
    parser.add_argument("--data-dir", default=None, type=str)
    parser.add_argument("--num-classes", default=10, type=int)
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--mc-samples", default=10, type=int)
    parser.add_argument("--selection-fpr", default=0.1, type=float)
    parser.add_argument(
        "--dropout-rates",
        nargs="+",
        default=[0.1, 0.2, 0.4, 0.6, 0.8],
        type=float,
    )
    parser.add_argument(
        "--target-fprs",
        nargs="+",
        default=[0.01, 0.05, 0.1, 0.25],
        type=float,
    )
    parser.add_argument("--device", default="auto", type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device == "auto":
        device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    data_dir = args.data_dir
    if data_dir is None:
        data_dir = os.path.join(
            "preprocessed_data",
            f"{args.dataset_name}_{args.attack_name}_poison_rate={args.poison_rate}_target={args.target_label}",
        )

    clean_val_data = load_tensor(data_dir, "clean_val_data.pt")
    clean_val_labels = load_tensor(data_dir, "clean_val_labels.pt")
    backdoor_val_data = load_tensor(data_dir, "backdoor_val_data.pt")
    backdoor_val_labels = load_tensor(data_dir, "backdoor_val_labels.pt")
    clean_test_data = load_tensor(data_dir, "clean_test_data.pt")
    clean_test_labels = load_tensor(data_dir, "clean_test_labels.pt")
    backdoor_test_data = load_tensor(data_dir, "backdoor_test_data.pt")
    backdoor_test_labels = load_tensor(data_dir, "backdoor_test_labels.pt")

    heldout_path = os.path.join(data_dir, "val_heldout_data.pt")
    heldout_labels_path = os.path.join(data_dir, "val_heldout_labels.pt")
    if os.path.exists(heldout_path) and os.path.exists(heldout_labels_path):
        heldout_data = load_tensor(data_dir, "val_heldout_data.pt")
        heldout_labels = load_tensor(data_dir, "val_heldout_labels.pt")
    else:
        heldout_data = clean_val_data
        heldout_labels = clean_val_labels

    clean_val_loader = tensor_loader(
        clean_val_data,
        clean_val_labels,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    backdoor_val_loader = tensor_loader(
        backdoor_val_data,
        backdoor_val_labels,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    clean_test_loader = tensor_loader(
        clean_test_data,
        clean_test_labels,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    backdoor_test_loader = tensor_loader(
        backdoor_test_data,
        backdoor_test_labels,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    heldout_loader = tensor_loader(
        heldout_data,
        heldout_labels,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = load_model_from_checkpoint(
        checkpoint_path=args.checkpoint_path,
        model_name=args.model_name,
        num_classes=args.num_classes,
        map_location=device,
    )

    val_sweep = run_psbd_sweep(
        model=model,
        target_label=args.target_label,
        clean_threshold_loader=heldout_loader,
        clean_eval_loader=clean_val_loader,
        backdoor_eval_loader=backdoor_val_loader,
        dropout_rates=tuple(float(rate) for rate in args.dropout_rates),
        target_fprs=tuple(float(fpr) for fpr in args.target_fprs),
        mc_samples=args.mc_samples,
        device=device,
    )

    selected_dropout_rate = find_best_dropout_rate(
        val_sweep,
        selection_fpr=args.selection_fpr,
    )

    test_eval = evaluate_psbd(
        model=model,
        target_label=args.target_label,
        clean_threshold_loader=heldout_loader,
        clean_eval_loader=clean_test_loader,
        backdoor_eval_loader=backdoor_test_loader,
        target_fprs=tuple(float(fpr) for fpr in args.target_fprs),
        mc_samples=args.mc_samples,
        dropout_rate=selected_dropout_rate,
        device=device,
    )

    result = {
        "checkpoint_path": args.checkpoint_path,
        "model_name": args.model_name,
        "data_dir": data_dir,
        "target_label": int(args.target_label),
        "mc_samples": int(args.mc_samples),
        "selection_fpr": float(args.selection_fpr),
        "dropout_rates": [float(rate) for rate in args.dropout_rates],
        "target_fprs": [float(fpr) for fpr in args.target_fprs],
        "selected_dropout_rate": float(selected_dropout_rate),
        "val_sweep": val_sweep,
        "test_eval": test_eval,
    }

    run_dir = os.path.join(
        "runs", f"psbd_{args.dataset_name}_{args.attack_name}_{get_timestamp()}"
    )
    os.makedirs(run_dir, exist_ok=True)
    output_path = os.path.join(run_dir, "psbd_metrics.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print("PSBD results saved to:", output_path)


if __name__ == "__main__":
    main()
