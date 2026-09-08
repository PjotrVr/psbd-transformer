"""Decision leverage: how much decision a sample buys per unit of representational novelty.

Every detector this project has tried perturbs something and measures how far the
prediction moves. H28 shows PSBD, STRIP, SCALE-UP and IBD-PSC are one family in which the
operator only selects which Jacobian is contracted. This has no perturbation at all. One
deterministic forward pass, no Monte Carlo, no Jacobian, no backward pass.

    SPACE. h(x) in R^768 is the CLS row of encoder.ln's output, so the head is exactly
    linear there, z(x) = W h(x) + b. That is the same post-LayerNorm space in which H16
    measured a rank-1 ablation taking ASR from 1.00 to 0.00, so the space is not chosen,
    it is where the head reads.

    DECISION FUNCTIONAL. With chat the model's own argmax and chat2 the runner-up,
        a(x) = W[chat] - W[chat2]
    so a(x)^T h + const is exactly the top-2 logit margin.

    CLEAN REFERENCE, fitted on the 2000 held-out clean images PSBD already uses for its
    threshold, with the model's OWN argmax as the pseudo-label so no ground truth is
    consumed: a global mean mu, a Ledoit-Wolf shrunk covariance Sigma, and per-class means
    shrunk toward mu (a class can hold ~20 images on CIFAR-100, so an unshrunk class mean
    is mostly noise).

    STATISTIC.
        delta(x) = h(x) - mu[chat(x)]
        DLR(x) = a^T delta / ( sqrt(a^T Sigma a) * sqrt(delta^T Sigma^-1 delta) )

    which is a cosine between Sigma^{-1/2} delta and Sigma^{+1/2} a, so it lies in
    [-1, 1] and is invariant to any rescaling of h, W, Sigma or the margin. Read as a
    ratio of 2 quantities this project has measured separately: the numerator is the
    top-2 margin in clean standard deviations along the decision direction (the
    deterministic margin, which scored 0.375 on its own), the denominator is the sample's
    Mahalanobis novelty against its own predicted class. DLR is neither; it is their
    ratio, and it is degree-0 homogeneous where every member of the H28 family is degree-1
    in the perturbation scale.

SIGN, AND THE FACT THAT THE FIRST ONE WAS WRONG.

The a priori prediction was HIGH DLR means POISONED: a backdoor must be carrier-independent
and must override the carrier's true class, so gradient descent should drive it onto the
direction buying the most decision per unit of displacement, which is the argmax of exactly
this ratio.

That is REFUTED. On vit_gtsrb_adaptive_blend_0_05, the hardest cell in the panel, the AUROC
under that sign is 0.082, with clean DLR 0.037 against backdoor -0.029, and a benign
control at 0.502. Backdoored samples have LOWER leverage, not higher.

The re-derivation, which is a different mechanism rather than a flipped tail. The
backdoor's contribution to h is a FIXED, carrier-independent vector u (H16's rank-1
direction). The decision functional a(x) = W[chat] - W[chat2] is NOT fixed: the runner-up
class differs from sample to sample, so a(x) points somewhere different for every input. A
single fixed u cannot be aligned with a whole family of a(x), so the trigger's displacement
is large but mostly ORTHOGONAL to the sample-specific margin direction. It wins by being
big and blunt, not by being efficient. A clean sample's offset from its own class mean has
no such constraint: it is built from the same redundant correlated cues that separate that
class from its neighbours, so it is naturally aligned with a(x).

This predicts LOW DLR means poisoned, and it predicts more than a sign: the effect should
be STRONGER where the runner-up class varies more, i.e. on datasets with more classes, and
it should hold for every carrier-independent trigger regardless of attack family.

Because the sign was changed after seeing one cell, H15 forbids reporting the flipped
number on that cell as a result. The flipped sign is PRE-REGISTERED here and evaluated on
held-out cells that were not looked at when it was formulated. Only the held-out numbers
count.

Clean and validation scores are 2-fold cross-fitted, so the covariance is never fitted on
the samples it scores; otherwise the clean tail is deflated in-sample and the realized FPR
is silently inflated.

    PYTHONPATH=. python experiments/decision_leverage/measure.py \
        --checkpoint-folder vit_gtsrb_adaptive_blend_0_05 vit_gtsrb_benign
"""

import argparse
import json
import os

import numpy as np
import torch
from sklearn.covariance import LedoitWolf
from sklearn.metrics import average_precision_score, roc_auc_score

from defences.checkpoint_eval import (
    PSBD_SPLIT_SEED,
    build_psbd_loaders_from_checkpoint,
    read_checkpoint_metadata,
)
from defences.psbd_metrics import pair_clean_to_backdoor
from models import load_checkpoint

QUANTILES = (0.01, 0.05, 0.10, 0.25)
# Per-class means are shrunk toward the global mean by this pseudo-count. On CIFAR-100 a
# class holds about 20 of the 2000 held-out images, so an unshrunk class mean is noise.
CLASS_SHRINKAGE = 10.0


def vision_transformer(model: torch.nn.Module) -> torch.nn.Module:
    return model[1] if isinstance(model, torch.nn.Sequential) else model


@torch.inference_mode()
def cls_features(
    model: torch.nn.Module, loader, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """(features, predicted class) where features are the head's own linear input."""
    model.eval()
    vit = vision_transformer(model)
    rows, predictions = [], []
    for images, _ in loader:
        images = images.to(device)
        if isinstance(model, torch.nn.Sequential):
            images = model[0](images)
        patches = vit._process_input(images)
        token = vit.class_token.expand(patches.shape[0], -1, -1)
        hidden = vit.encoder(torch.cat([token, patches], dim=1))[:, 0]
        rows.append(hidden.cpu())
        predictions.append(vit.heads(hidden).argmax(dim=1).cpu())
    if not rows:
        return torch.empty(0), torch.empty(0, dtype=torch.long)
    return torch.cat(rows).double(), torch.cat(predictions)


def fit_reference(features: torch.Tensor, predicted: torch.Tensor, num_classes: int):
    """Global mean, Ledoit-Wolf covariance and shrunk per-class means, from clean data."""
    array = features.numpy()
    mean = array.mean(axis=0)
    covariance = LedoitWolf(assume_centered=False).fit(array)
    precision = covariance.get_precision()
    class_means = np.tile(mean, (num_classes, 1))
    for label in np.unique(predicted.numpy()):
        members = array[predicted.numpy() == label]
        count = len(members)
        class_means[label] = (count * members.mean(axis=0) + CLASS_SHRINKAGE * mean) / (
            count + CLASS_SHRINKAGE
        )
    return class_means, covariance.covariance_, precision


def leverage(
    features: torch.Tensor,
    predicted: torch.Tensor,
    weight: np.ndarray,
    logits: np.ndarray,
    class_means: np.ndarray,
    covariance: np.ndarray,
    precision: np.ndarray,
) -> torch.Tensor:
    """DLR per sample, shape (N,). Higher means more decision bought per unit novelty."""
    array = features.numpy()
    order = np.argsort(-logits, axis=1)
    top, runner_up = order[:, 0], order[:, 1]
    direction = weight[top] - weight[runner_up]  # (N, d)
    offset = array - class_means[predicted.numpy()]  # (N, d)

    numerator = np.einsum("nd,nd->n", direction, offset)
    direction_norm = np.sqrt(np.einsum("nd,de,ne->n", direction, covariance, direction))
    offset_norm = np.sqrt(np.einsum("nd,de,ne->n", offset, precision, offset))
    denominator = np.maximum(direction_norm * offset_norm, 1e-12)
    return torch.from_numpy(numerator / denominator).float()


def cross_fitted(features, predicted, weight, logits, num_classes) -> torch.Tensor:
    """Score every row from a reference fitted on the other half of the same split."""
    count = len(features)
    halves = [np.arange(0, count // 2), np.arange(count // 2, count)]
    scores = torch.zeros(count)
    for fit_index, score_index in (halves, halves[::-1]):
        class_means, covariance, precision = fit_reference(
            features[fit_index], predicted[fit_index], num_classes
        )
        scores[score_index] = leverage(
            features[score_index],
            predicted[score_index],
            weight,
            logits[score_index],
            class_means,
            covariance,
            precision,
        )
    return scores


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
    vit = vision_transformer(model)
    weight = vit.heads.head.weight.detach().cpu().double().numpy()
    bias = vit.heads.head.bias.detach().cpu().double().numpy()
    num_classes = weight.shape[0]

    features, predicted = {}, {}
    for split, loader in loaders.items():
        features[split], predicted[split] = cls_features(model, loader, device)

    logits = {split: features[split].numpy() @ weight.T + bias for split in features}
    # The reference for the analysis splits comes from the whole validation split; the
    # validation split itself is cross-fitted so its own tail is not fitted in-sample.
    class_means, covariance, precision = fit_reference(
        features["validation"], predicted["validation"], num_classes
    )
    scores = {
        "validation": cross_fitted(
            features["validation"],
            predicted["validation"],
            weight,
            logits["validation"],
            num_classes,
        )
    }
    for split in ("clean", "backdoor"):
        scores[split] = leverage(
            features[split],
            predicted[split],
            weight,
            logits[split],
            class_means,
            covariance,
            precision,
        )

    # LOW DLR means poisoned (see the sign note in the module docstring), which is
    # already the repo-wide convention, so the score is used as it stands.
    signed = scores
    report = detection(
        signed["validation"],
        pair_clean_to_backdoor(signed["clean"], manifest),
        signed["backdoor"],
    )
    report.update(
        {
            "folder_name": folder,
            "dataset": metadata["dataset"],
            "attack": metadata["attack"],
            "poison_rate": metadata.get("poison_rate"),
            "mean_dlr": {split: float(value.mean()) for split, value in scores.items()},
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
        with open(os.path.join(out_dir, "decision_leverage.json"), "w") as handle:
            json.dump(report, handle, indent=2)
        print(
            f"[ok] {folder}  auroc={report['auroc']:.3f} auprc={report['auprc']:.3f} "
            f"tpr@1%={report['q0.01']['tpr']:.3f} "
            f"dlr clean={report['mean_dlr']['clean']:.3f} "
            f"bd={report['mean_dlr']['backdoor']:.3f}"
        )


if __name__ == "__main__":
    main()
