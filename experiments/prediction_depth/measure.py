"""Prediction depth as a backdoor detector, and whether it survives its own controls.

A backdoor is a shortcut. PSBD detects it by perturbing the network and watching how
far the prediction moves; every method in that family (H28) measures a decision margin
along the perturbation's Jacobian. This measures something else entirely: not how
robust the prediction is, but HOW EARLY IN DEPTH it is already decided.

    original form (Baldock et al., prediction depth, adapted to a logit lens)
        depth(x) = min { l : argmax g(h_j(x)) = argmax f(x) for all j >= l }
        g(h) = head(LN_final(h[CLS]))

    descriptive form
        prediction_depth = the first block after which the logit-lens readout never
                           again disagrees with the network's own final answer

A priori sign, fixed by the mechanism BEFORE any backdoor data is looked at: a trigger
is a strong, simple, high-salience feature, so it should be resolved earlier than a
natural class, giving LOW depth. Low means poisoned, which is the repo-wide one-sided
convention (H15), so the score is the depth itself and no tail is chosen after the
fact. If the effect comes out inverted, that is a refutation and not a licence to flip
the sign.

Why this is not a fifth "where the backdoor sits" attempt (H18, H22, H35, H40 all
REFUTED): those localised the backdoor to a set of units or heads and scored a sample
by that set's behaviour, which needs the backdoor's location to be both stable and
findable. This reads only the network's own output head, at every depth, and never
names a unit, a head or a direction. The quantity is a property of the sample's
trajectory, not of the model's parts.

Why the LayerNorm hazard is handled: reading intermediate activations through a
statistic that is sensitive to their scale is failure mode 3 in the ledger, and it
once produced a clean, monotone, benign-controlled and entirely false result. The
logit lens applies the network's own FINAL LayerNorm to every intermediate activation
before the head, so every depth is read at the scale the head was trained for, and the
statistic is an argmax, which is invariant to any positive rescaling anyway.

    PYTHONPATH=. python experiments/prediction_depth/measure.py --checkpoint-folder X
"""

import argparse
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
from models import load_checkpoint

QUANTILES = (0.01, 0.05, 0.10, 0.25)


def vision_transformer(model: torch.nn.Module) -> torch.nn.Module:
    """The ViT inside the Resize wrapper the checkpoints are saved with."""
    return model[1] if isinstance(model, torch.nn.Sequential) else model


@torch.inference_mode()
def logit_lens_statistics(
    model: torch.nn.Module, loader, device: torch.device
) -> tuple[dict[str, torch.Tensor], int]:
    """Per-sample logit-lens statistics, CLS-only and over all patch tokens.

    One forward pass per sample. The readout reuses the network's own ln and heads,
    so nothing is fitted and nothing is calibrated against the split being scored.

    The CLS token is what the classifier actually reads, but it is an aggregate, and
    a patch trigger occupies 1 or 2 of 196 patches. H32 already showed BadNet's
    largest per-token direction norm sits at the trigger's own position, so a
    statistic that only ever looks at CLS discards the localization the architecture
    hands over for free. Both are computed here so the comparison is measured rather
    than assumed.

    Returned statistics, each (N,), each with its direction fixed a priori so that
    LOW means poisoned and no tail is chosen after the fact (H15):

      depth_cls          first block after which the CLS lens never again disagrees
                         with the final answer. A trigger is a strong simple feature,
                         so it should resolve early.
      depth_token_min    the same, minimised over patch tokens: the earliest depth at
                         which ANY single patch has already locked onto the final
                         answer and never leaves it. A localized trigger should give
                         one token that commits almost immediately.
      token_agreement    fraction of patch tokens whose final-layer lens prediction
                         equals the final answer. A trigger drives the answer from a
                         couple of patches while the rest of the image still votes for
                         its true class, so agreement is LOWER on a triggered image,
                         which is already the low-means-poisoned convention. Not
                         negated: an earlier version negated it and so scored against
                         its own stated prediction.
      depth_soft         the mean over layers of the lens probability of the final
                         answer, negated. Same mechanism as depth_cls but continuous.
                         depth_cls takes 12 integer values, so a clean-quantile threshold
                         lands on a tie and the strict < test drops every sample sitting
                         exactly on it. On vit_gtsrb_badnet_a2o_0_1 that is the whole
                         backdoor population: mean depth is exactly 5.000 against clean
                         7.292, AUROC 0.994, and TPR at 1% FPR reads 0.000 because the
                         threshold IS 5.0. At 5% FPR the same score reads TPR 1.000 at an
                         achieved FPR of 0.011. The detector was never the problem; the
                         integer tie was.
      token_max_prob     the largest final-class probability any single patch token
                         assigns. On a clean image the winning class is supported by
                         many patches and at least one is nearly saturated; a trigger
                         wins through CLS aggregation without any single patch being
                         confident, so this too is LOWER on a triggered image.
    """
    model.eval()
    vit = vision_transformer(model)
    blocks = vit.encoder.layers
    captured: list[torch.Tensor] = []

    def capture(_module, _inputs, output):
        captured.append(output)

    handles = [block.register_forward_hook(capture) for block in blocks]
    collected: dict[str, list[torch.Tensor]] = {
        name: []
        for name in (
            "depth_cls",
            "depth_soft",
            "depth_token_min",
            "token_agreement",
            "token_max_prob",
        )
    }
    try:
        for images, _ in loader:
            captured.clear()
            final = model(images.to(device)).argmax(dim=1)  # (batch,)

            def settle_depth(agrees: torch.Tensor) -> torch.Tensor:
                """First layer from which agreement never breaks again, along dim 0.

                Scanning from the top: a reversed cumulative product stays 1 only
                while every deeper layer has also agreed.
                """
                kept = torch.flip(
                    torch.cumprod(torch.flip(agrees.long(), dims=[0]), dim=0), dims=[0]
                )
                return kept.shape[0] - kept.sum(dim=0)

            cls_logits = torch.stack(
                [vit.heads(vit.encoder.ln(h[:, 0])) for h in captured]
            )  # (layers, batch, classes)
            collected["depth_cls"].append(
                settle_depth(cls_logits.argmax(dim=-1) == final.view(1, -1)).cpu()
            )
            lens_probability = torch.softmax(cls_logits, dim=-1).gather(
                2, final.view(1, -1, 1).expand(cls_logits.shape[0], -1, 1)
            )
            collected["depth_soft"].append(
                -lens_probability.squeeze(2).mean(dim=0).cpu()
            )

            # (layers, batch, tokens): the lens prediction of every patch at every depth
            token_layers = torch.stack(
                [vit.heads(vit.encoder.ln(h[:, 1:])).argmax(dim=-1) for h in captured]
            )
            token_agrees = token_layers == final.view(1, -1, 1)
            # settle_depth reduces dim 0, so tokens ride along in the trailing axis
            per_token_depth = settle_depth(token_agrees)  # (batch, tokens)
            collected["depth_token_min"].append(per_token_depth.min(dim=1).values.cpu())
            collected["token_agreement"].append(
                token_agrees[-1].float().mean(dim=1).cpu()
            )

            last = torch.softmax(vit.heads(vit.encoder.ln(captured[-1][:, 1:])), dim=-1)
            chosen = last.gather(2, final.view(-1, 1, 1).expand(-1, last.shape[1], 1))
            collected["token_max_prob"].append(
                chosen.squeeze(2).max(dim=1).values.cpu()
            )
    finally:
        for handle in handles:
            handle.remove()

    if not collected["depth_cls"]:
        return {name: torch.empty(0) for name in collected}, len(blocks)
    return (
        {name: torch.cat(parts).float() for name, parts in collected.items()},
        len(blocks),
    )


def deviation_from_clean(
    validation: torch.Tensor, values: torch.Tensor
) -> torch.Tensor:
    """How far a sample sits from the clean-validation distribution, either way.

        deviation(x) = -2 * | rank_val(score(x)) - 0.5 |

    Negated so LOW means poisoned, matching the shared convention.

    The raw per-token statistics have a sign that depends on the trigger's spatial
    extent, and the direction is a property of the ATTACK rather than of the
    detector. A local patch trigger drives the answer from 1 or 2 patches while the
    other 194 keep voting their own class, so agreement FALLS (cifar10 badnet: clean
    0.429, backdoor 0.064). A global blend paints every patch, so every patch votes
    the target and agreement RISES (cifar100 adaptive_blend: clean 0.019, backdoor
    0.222). Both are far from clean; they are far in opposite directions.

    This is not the two-sided rule H15 retired. That rule picked which tail to flag
    by reading the AUROC, which needs the poison labels the detector exists to
    predict. This ranks against the clean validation split, which the threat model
    already grants the defender, and applies one fixed rule to every checkpoint: far
    from clean in either direction is suspicious. The threshold is still a quantile
    of the validation deviation, so the false-positive budget is set exactly as
    before and no poison label is consulted at any point.
    """
    reference = validation.sort().values
    positions = torch.searchsorted(reference, values.contiguous())
    ranks = positions.float() / max(len(reference), 1)
    return -2.0 * (ranks - 0.5).abs()


def detection(validation, clean, backdoor) -> dict:
    """One-sided AUROC, AUPRC and TPR at each quantile. Low score means poisoned."""
    labels = np.concatenate([np.zeros(len(clean)), np.ones(len(backdoor))])
    scores = np.concatenate([-clean.cpu().numpy(), -backdoor.cpu().numpy()])
    report = {
        "auroc": float(roc_auc_score(labels, scores)),
        # AUPRC is reported because at 1% poisoning the positive class is rare and
        # AUROC is measurably optimistic about the tail that a defender operates in.
        "auprc": float(average_precision_score(labels, scores)),
        "n_clean": int(len(clean)),
        "n_backdoor": int(len(backdoor)),
    }
    for quantile in QUANTILES:
        threshold = float(np.quantile(validation.cpu().numpy(), quantile))
        report[f"q{quantile:.2f}"] = {
            "threshold": threshold,
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

    per_split = {}
    for split, loader in loaders.items():
        per_split[split], n_layers = logit_lens_statistics(model, loader, device)

    # A deviation variant of every raw statistic, computed before scoring so both
    # the raw and the reference-relative form are reported side by side.
    for name in list(per_split["validation"]):
        for split in per_split:
            per_split[split][f"{name}_dev"] = deviation_from_clean(
                per_split["validation"][name], per_split[split][name]
            )

    statistics = {}
    for name in per_split["validation"]:
        paired_clean = pair_clean_to_backdoor(per_split["clean"][name], manifest)
        statistics[name] = detection(
            per_split["validation"][name], paired_clean, per_split["backdoor"][name]
        )
        statistics[name]["mean"] = {
            split: float(values[name].mean()) for split, values in per_split.items()
        }

    report = {
        "folder_name": folder,
        "dataset": metadata["dataset"],
        "attack": metadata["attack"],
        "label_mode": metadata.get("label_mode"),
        "poison_rate": metadata.get("poison_rate"),
        "n_layers": int(n_layers),
        "statistics": statistics,
        "per_sample": per_split,
    }
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
        # The per-sample tensors, not just the aggregate. cli/baselines.py computes
        # exactly these and throws them away, which is why every later question about
        # fusion or recalibration costs another GPU pass. Keeping them makes any new
        # scoring rule a CPU rescore of a file already on disk.
        torch.save(
            report.pop("per_sample"),
            os.path.join(out_dir, "prediction_depth_scores.pt"),
        )
        with open(os.path.join(out_dir, "prediction_depth.json"), "w") as handle:
            json.dump(report, handle, indent=2)
        print(f"[ok] {folder}")
        for name, block in report["statistics"].items():
            print(
                f"      {name:16s} auroc={block['auroc']:.3f} "
                f"auprc={block['auprc']:.3f} tpr@1%={block['q0.01']['tpr']:.3f} "
                f"clean={block['mean']['clean']:.3f} bd={block['mean']['backdoor']:.3f}"
            )


if __name__ == "__main__":
    main()
