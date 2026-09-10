"""Disable the weights that matter for THIS prediction, and see whether it survives.

Every operator in this project perturbs ACTIVATIONS with unstructured, data-independent
noise: dropout, masks, DropPath, Gaussian, gain scaling. H28 shows they are one family in
which the operator only chooses a Jacobian. This probes a different object entirely, on 2
axes at once. The perturbation is applied to WEIGHTS rather than activations, and it is
DIRECTED by the sample's own gradient rather than drawn at random.

The mechanism comes from this project's own strongest result. H16 established that the
backdoor is a RANK-1 DIRECTION in activation space: post-LayerNorm rank-1 removal takes
ASR from 1.00 to 0.00 on every checkpoint, while ablating 300 of 768 coordinates does
nothing. The classification head is what reads that direction. So a triggered input's
answer should rest on FEW head weights, the ones aligned with the direction, while a
clean answer rests on evidence spread across many.

    per-sample saliency, for the head W (num_classes x d) and its input a:
        the gradient of the predicted logit z_c w.r.t. W is the outer product
            dz_c / dW = e_c a^T
        so the standard gradient-times-weight attribution is
            S = |W . (e_c a^T)| , i.e. row c is |W_c . a| and every other row is 0

    probe: zero the top-k entries of row c by saliency, recompute the logits, ask whether
    argmax still equals the unperturbed answer.

    statistic: survival(x) = the fraction of removal levels k at which the answer holds.

A priori sign, fixed before any backdoor data was read: the backdoor answer is carried by
a single direction, so it should die at a SMALLER k, giving LOWER survival. Low means
poisoned, the convention used everywhere here (H15).

RESULT: that prediction is REFUTED. On vit_gtsrb_badnet_a2o_0_1 the AUROC is 0.267, i.e.
inverted, with backdoor survival 0.742 against clean 0.639 and a benign control at 0.499.
The backdoor answer is MORE robust to targeted head-weight removal, not less.

The lesson is about H16 rather than about this probe. A rank-1 direction being sufficient
to REMOVE the backdoor when deleted from the activation stream does not mean the
prediction RESTS on few head weights: the head reads that direction through many
coordinates, so ablating the largest ones individually leaves enough of the projection
intact. The observed direction is in fact the field-standard one, that the backdoor path
is the robust path, which is exactly PSBD's own premise. Testing THAT is a separate
pre-registered experiment; flipping the sign here would be the move H15 retired, and the
flipped value (0.733) is far below PSBD's 0.999 on the same cell in any case.

Cost. For a linear layer the per-sample gradient factorizes, so no per-sample backward
pass is needed and the masked logits are an einsum. The whole probe is 1 forward pass
through the backbone plus arithmetic, which is cheaper than every method it is compared
against.

    PYTHONPATH=. python experiments/gradient_weight_ablation/measure.py \\
        --checkpoint-folder vit_gtsrb_badnet_a2o_0_1 vit_gtsrb_benign
"""

import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from data.splits import (
    PSBD_SPLIT_SEED,
    build_psbd_loaders_from_checkpoint,
    read_checkpoint_metadata,
)
from defences.decision import pair_clean_to_backdoor
from models.backbones import load_checkpoint

# Fractions of the predicted row's weights to disable, hardest last.
REMOVAL_FRACTIONS = (0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70)
QUANTILES = (0.01, 0.05, 0.10, 0.25)


def vision_transformer(model: torch.nn.Module) -> torch.nn.Module:
    return model[1] if isinstance(model, torch.nn.Sequential) else model


@torch.inference_mode()
def survival_scores(
    model: torch.nn.Module, loader, device: torch.device
) -> torch.Tensor:
    """Per-sample fraction of removal levels the original answer survives, shape (N,)."""
    model.eval()
    vit = vision_transformer(model)
    head = vit.heads.head
    weight = head.weight  # (num_classes, d)
    bias = head.bias
    dimension = weight.shape[1]

    scores = []
    for images, _ in loader:
        images = images.to(device)
        # The head's input: everything up to and including the final LayerNorm.
        features = _embed(model, images)  # (batch, d), the head's own input
        logits = features @ weight.T + bias
        predicted = logits.argmax(dim=1)  # (batch,)

        # Saliency of the predicted row only: |W_c * a|, elementwise.
        rows = weight[predicted]  # (batch, d)
        saliency = (rows * features).abs()  # (batch, d)
        order = saliency.argsort(dim=1, descending=True)

        survived = torch.zeros(images.size(0), device=device)
        for fraction in REMOVAL_FRACTIONS:
            k = max(1, int(fraction * dimension))
            mask = torch.ones_like(features)
            mask.scatter_(1, order[:, :k], 0.0)
            # Only the predicted class's row is ablated, so the competing rows keep
            # their full evidence and the question is whether c still wins.
            ablated = logits.clone()
            ablated.scatter_(
                1,
                predicted.view(-1, 1),
                ((features * mask) * rows).sum(dim=1, keepdim=True)
                + bias[predicted].view(-1, 1),
            )
            survived += (ablated.argmax(dim=1) == predicted).float()
        scores.append((survived / len(REMOVAL_FRACTIONS)).cpu())
    return torch.cat(scores) if scores else torch.empty(0)


def _embed(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    """The CLS feature the head consumes, i.e. the network up to the final LayerNorm.

    The checkpoint is nn.Sequential(Resize(224), VisionTransformer), so the resize has to
    run first; the ViT asserts on a 32-pixel input otherwise.
    """
    vit = vision_transformer(model)
    if isinstance(model, torch.nn.Sequential):
        images = model[0](images)
    x = vit._process_input(images)
    batch_class_token = vit.class_token.expand(x.shape[0], -1, -1)
    x = torch.cat([batch_class_token, x], dim=1)
    return vit.encoder(x)[:, 0]


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


def analyse(folder: str, args) -> dict:
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

    scores = {
        split: survival_scores(model, loader, device)
        for split, loader in loaders.items()
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
            "poison_rate": metadata.get("poison_rate"),
            "removal_fractions": list(REMOVAL_FRACTIONS),
            "mean_survival": {
                split: float(value.mean()) for split, value in scores.items()
            },
        }
    )
    return report


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
            report = analyse(folder, args)
        except Exception as error:  # noqa: BLE001
            print(f"[FAILED] {folder}: {type(error).__name__}: {error}")
            continue
        out_dir = os.path.join(args.results_dir, folder)
        os.makedirs(out_dir, exist_ok=True)
        with open(
            os.path.join(out_dir, "gradient_weight_ablation.json"), "w"
        ) as handle:
            json.dump(report, handle, indent=2)
        print(
            f"[ok] {folder}  auroc={report['auroc']:.3f} auprc={report['auprc']:.3f} "
            f"tpr@1%={report['q0.01']['tpr']:.3f} "
            f"survival clean={report['mean_survival']['clean']:.3f} "
            f"bd={report['mean_survival']['backdoor']:.3f}"
        )


if __name__ == "__main__":
    main()
