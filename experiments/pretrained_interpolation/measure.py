"""A perturbation paradigm outside the activation-noise family: move the WEIGHTS.

Every operator this project has tried perturbs ACTIVATIONS with unstructured,
data-independent noise at a chosen position: dropout, channel/token/head masking,
DropPath, Gaussian, Rademacher, gain scaling. H28 shows PSBD, STRIP, SCALE-UP and
IBD-PSC are one family in which the operator only chooses which Jacobian is probed, and
the measured position variance is 1.43x the operator variance, so the operator axis is
close to exhausted.

This probes a different axis entirely. The victim was finetuned from public ImageNet
weights, so the defender can compute the finetuning delta with no poison label at all:

    tau = theta_finetuned - theta_pretrained
    theta(a) = theta_pretrained + a * tau,     a in [0, 1]

a = 1 is the victim; a = 0 is the public pretrained backbone. Sweeping a walks the model
back along the one weight-space direction the defender gets for free.

    original form
        collapse(x) = min { a : argmax f_a(x) = argmax f_1(x) for all a' >= a }
    descriptive form
        the smallest fraction of the finetuning you can keep and still get the same answer

A priori sign, fixed by the mechanism before any backdoor data is read. A backdoor is
written ENTIRELY during finetuning: the pretrained model has never seen the trigger, so
the trigger-to-target path lives wholly inside tau and is destroyed as a shrinks. Clean
class knowledge does not, because the pretrained backbone still produces useful features
and only the head has to interpret them. So a triggered input should lose its answer at a
HIGHER a than a clean one, i.e. a LARGER collapse value. The reported score is
-collapse, so LOW means poisoned, the convention used everywhere here (H15).

The competing argument, stated so it is not hidden: a backdoor is a high-margin shortcut
and might instead be MORE robust to weight shrinkage than delicate clean features, which
would invert the sign. That is a refutation of the mechanism above, not a licence to flip
the sign, and the benign control decides whether either effect is real at all.

Only the backbone is interpolated. The classification head has no pretrained counterpart
(build_vit replaces it with a fresh layer), so interpolating it would mix in a random
readout and measure that instead.

    PYTHONPATH=. python experiments/pretrained_interpolation/measure.py \
        --checkpoint-folder vit_gtsrb_badnet_a2o_0_1 vit_gtsrb_benign
"""

import argparse
import copy
import json
import os

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from defences.checkpoint_eval import (
    PSBD_SPLIT_SEED,
    build_psbd_loaders_from_checkpoint,
    read_checkpoint_metadata,
)
from defences.psbd_metrics import pair_clean_to_backdoor
from models import build_swin, build_vit, load_checkpoint
from utils.config import DATASET_REGISTRY

# 1.0 first so the reference prediction is the victim's own, then walked back.
ALPHAS = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0)
QUANTILES = (0.01, 0.05, 0.10, 0.25)
HEAD_MARKERS = ("heads.head", "head.weight", "head.bias")


def pretrained_state(architecture: str, num_classes: int) -> dict:
    """The public initialization the victim was finetuned from."""
    builder = build_vit if architecture == "vit" else build_swin
    return builder(num_classes).state_dict()


def interpolated_state(finetuned: dict, pretrained: dict, alpha: float) -> dict:
    """theta_pretrained + alpha * (theta_finetuned - theta_pretrained), backbone only.

    The head is held at its finetuned value because the pretrained network has no head
    for this task; build_vit replaces it with a fresh layer, so interpolating it would
    blend in a random readout and the probe would measure that rather than the backbone.
    Integer buffers are left alone: they are counters, not parameters, and averaging them
    is meaningless.
    """
    blended = {}
    for key, value in finetuned.items():
        reference = pretrained.get(key)
        skip = (
            reference is None
            or reference.shape != value.shape
            or not value.dtype.is_floating_point
            or any(marker in key for marker in HEAD_MARKERS)
        )
        blended[key] = value if skip else reference + alpha * (value - reference)
    return blended


@torch.inference_mode()
def predictions_at(
    model: torch.nn.Module, loader, device: torch.device
) -> torch.Tensor:
    model.eval()
    out = [model(images.to(device)).argmax(dim=1).cpu() for images, _ in loader]
    return torch.cat(out) if out else torch.empty(0, dtype=torch.long)


def collapse_alpha(per_alpha: torch.Tensor, alphas: tuple) -> torch.Tensor:
    """Smallest alpha whose prediction, and every larger one's, matches alpha = 1.

    per_alpha is (len(alphas), N) of argmax labels, with alphas DESCENDING from 1.0.
    Walking down the list, a running AND stays true only while every alpha at least this
    large has agreed, so the count of leading agreements indexes the collapse point.
    """
    agrees = per_alpha == per_alpha[0].view(1, -1)
    kept = torch.cumprod(agrees.long(), dim=0)
    index = kept.sum(dim=0).clamp(max=len(alphas) - 1)
    table = torch.tensor(alphas, dtype=torch.float32)
    return table[index]


def detection(validation, clean, backdoor) -> dict:
    labels = np.concatenate([np.zeros(len(clean)), np.ones(len(backdoor))])
    scores = np.concatenate([-clean.numpy(), -backdoor.numpy()])
    report = {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
    }
    for quantile in QUANTILES:
        threshold = float(np.quantile(validation.numpy(), quantile))
        report[f"q{quantile:.2f}"] = {
            "tpr": float((backdoor < threshold).float().mean()),
            "fpr": float((clean < threshold).float().mean()),
        }
    return report


def analyse(folder: str, args) -> tuple[dict, dict]:
    checkpoint_path = os.path.join(args.checkpoints_dir, folder, "attack_result.pt")
    metadata = read_checkpoint_metadata(checkpoint_path)
    probe = "badnet_a2o" if metadata["attack"] == "benign" else None
    loaders, manifest = build_psbd_loaders_from_checkpoint(
        checkpoint_path,
        seed=PSBD_SPLIT_SEED,
        raw_data_dir=args.raw_data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        probe_attack=probe,
        probe_target_label=0 if probe else None,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(metadata["architecture"], checkpoint_path, device)
    finetuned = copy.deepcopy(model.state_dict())
    spec = DATASET_REGISTRY[metadata["dataset"]]
    pretrained = {
        key: value.to(device)
        for key, value in pretrained_state(
            metadata["architecture"], spec.num_classes
        ).items()
    }

    per_split = {split: [] for split in loaders}
    for alpha in ALPHAS:
        model.load_state_dict(interpolated_state(finetuned, pretrained, alpha))
        for split, loader in loaders.items():
            per_split[split].append(predictions_at(model, loader, device))
    model.load_state_dict(finetuned)

    scores = {
        split: -collapse_alpha(torch.stack(rows), ALPHAS)
        for split, rows in per_split.items()
    }
    report = detection(
        scores["validation"],
        pair_clean_to_backdoor(scores["clean"], manifest),
        scores["backdoor"],
    )
    report.update(
        {
            "folder_name": folder,
            "dataset": metadata["dataset"],
            "attack": metadata["attack"],
            "label_mode": metadata.get("label_mode"),
            "poison_rate": metadata.get("poison_rate"),
            "alphas": list(ALPHAS),
            "mean_collapse": {
                split: float(-value.mean()) for split, value in scores.items()
            },
        }
    )
    return report, scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-folder", nargs="+", required=True)
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--raw-data-dir", default="raw_data")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    for folder in args.checkpoint_folder:
        try:
            report, scores = analyse(folder, args)
        except Exception as error:  # noqa: BLE001
            print(f"[FAILED] {folder}: {type(error).__name__}: {error}")
            continue
        out_dir = os.path.join(args.results_dir, folder)
        os.makedirs(out_dir, exist_ok=True)
        torch.save(scores, os.path.join(out_dir, "pretrained_interp_scores.pt"))
        with open(
            os.path.join(out_dir, "pretrained_interpolation.json"), "w"
        ) as handle:
            json.dump(report, handle, indent=2)
        print(
            f"[ok] {folder}  auroc={report['auroc']:.3f} auprc={report['auprc']:.3f} "
            f"tpr@1%={report['q0.01']['tpr']:.3f} "
            f"collapse clean={report['mean_collapse']['clean']:.3f} "
            f"bd={report['mean_collapse']['backdoor']:.3f}"
        )


if __name__ == "__main__":
    main()
