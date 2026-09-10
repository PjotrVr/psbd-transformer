"""Is a masking-based detector reading the backdoor, or reading which tokens it happened to hit?

PSBD masks tokens at random and scores how far the prediction moves. If a few tokens carry
outsized influence, whether a draw happened to hit one of them could dominate that score, and
the detector would be measuring token identity rather than backdoor presence.

The sign is predicted BEFORE measuring, because guessing it afterwards is not a prediction.
Kang et al. (ICLR 2025, arXiv:2503.03321) find that masking visual sink tokens costs little
while masking the same number of ORDINARY tokens costs much more: sink attention is a
recyclable budget. So a draw that hits a high-norm token should produce a SMALLER prediction
shift, not a larger one, and the contamination is downward variance on clean images.

Sun et al. (arXiv:2402.17762) separately show the difference between zeroing an outlier, which
is catastrophic, and replacing it with its mean, which is free. Both are measured here, since
if the effect vanishes under mean substitution then "substitute, do not zero" is a one-line
mitigation.

Reported per layer, over many random masks per image:

    shift_hit / shift_miss   mean prediction shift for draws that did and did not cover a
                             top-norm token, on clean and on triggered images separately
    effect                   hit minus miss. Negative confirms the predicted sign.
    auroc_all / auroc_miss   detection AUROC over all draws, and over only the draws that hit
                             nothing. If the second collapses, the detector was reading
                             outlier tokens; if it holds, it was not.

    PYTHONPATH=. python experiments/residual_stream_mechanism/sink_hit_confound.py \
        --checkpoint-folder vit_gtsrb_badnet_a2o_0_1
"""

import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from data.splits import (
    PSBD_SPLIT_SEED,
    build_psbd_loaders_from_checkpoint,
    read_checkpoint_metadata,
)
from models.backbones import load_checkpoint, network_core


@torch.inference_mode()
def run_split(model, core, loader, device, args):
    """Per-draw prediction shift, and whether the draw covered a top-norm token."""
    layer = args.layer - 1
    block = core.encoder.layers[layer]
    captured = {}
    watch = block.register_forward_hook(
        lambda _m, _i, out: captured.__setitem__("x", out.detach())
    )
    shifts, hits, labels = [], [], []
    seen = 0
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    for images, _ in loader:
        images = images.to(device)
        baseline = model(images)
        probability = baseline.softmax(dim=-1)
        predicted = probability.argmax(dim=-1)
        tracked = probability.gather(1, predicted[:, None]).squeeze(1)
        norms = captured["x"][:, 1:, :].norm(dim=-1)
        outliers = norms.topk(args.top_k, dim=-1).indices  # (B, top_k)

        for _ in range(args.draws):
            keep = (
                torch.rand(images.shape[0], 196, generator=generator).to(device)
                >= args.rate
            ).float()
            hit = (keep.gather(1, outliers) == 0).any(dim=1)

            def mask_hook(_m, _i, out, keep=keep):
                patches = out[:, 1:, :]
                if args.substitute == "mean":
                    filler = patches.mean(dim=1, keepdim=True)
                    patches = patches * keep.unsqueeze(-1) + filler * (
                        1 - keep
                    ).unsqueeze(-1)
                else:
                    patches = patches * keep.unsqueeze(-1)
                return torch.cat([out[:, :1, :], patches], dim=1)

            handle = block.register_forward_hook(mask_hook)
            perturbed = model(images).softmax(dim=-1)
            handle.remove()
            moved = perturbed.gather(1, predicted[:, None]).squeeze(1)
            shifts.append((tracked - moved).cpu())
            hits.append(hit.cpu())
        seen += images.shape[0]
        if seen >= args.limit:
            break
    watch.remove()
    return torch.cat(shifts), torch.cat(hits)


def analyse(folder, args) -> dict:
    path = os.path.join(args.checkpoints_dir, folder, "attack_result.pt")
    metadata = read_checkpoint_metadata(path)
    probe = "badnet_a2o" if metadata["attack"] == "benign" else None
    loaders, _ = build_psbd_loaders_from_checkpoint(
        path,
        seed=PSBD_SPLIT_SEED,
        raw_data_dir=args.raw_data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        probe_attack=probe,
        probe_target_label=0 if probe else None,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(metadata["architecture"], path, device).eval()
    core = network_core(model)

    out = {}
    for split in ("clean", "backdoor"):
        shift, hit = run_split(model, core, loaders[split], device, args)
        out[split] = (shift, hit)
    clean_shift, clean_hit = out["clean"]
    bd_shift, bd_hit = out["backdoor"]

    def auroc(mask_clean, mask_bd):
        a, b = clean_shift[mask_clean], bd_shift[mask_bd]
        length = min(len(a), len(b))
        if length < 10:
            return float("nan")
        labels = np.concatenate([np.zeros(length), np.ones(length)])
        # low shift is the poisoned evidence, so negate
        scores = np.concatenate([-a[:length].numpy(), -b[:length].numpy()])
        return float(roc_auc_score(labels, scores))

    return {
        "folder_name": folder,
        "attack": metadata["attack"],
        "layer": args.layer,
        "rate": args.rate,
        "top_k": args.top_k,
        "substitute": args.substitute,
        "draws": args.draws,
        "hit_rate_clean": float(clean_hit.float().mean()),
        "shift_hit_clean": float(clean_shift[clean_hit].mean()),
        "shift_miss_clean": float(clean_shift[~clean_hit].mean()),
        "effect_clean": float(
            clean_shift[clean_hit].mean() - clean_shift[~clean_hit].mean()
        ),
        "shift_hit_backdoor": float(bd_shift[bd_hit].mean()),
        "shift_miss_backdoor": float(bd_shift[~bd_hit].mean()),
        "effect_backdoor": float(bd_shift[bd_hit].mean() - bd_shift[~bd_hit].mean()),
        "auroc_all": auroc(torch.ones_like(clean_hit), torch.ones_like(bd_hit)),
        "auroc_miss_only": auroc(~clean_hit, ~bd_hit),
        "auroc_hit_only": auroc(clean_hit, bd_hit),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-folder", nargs="+", required=True)
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--raw-data-dir", default="raw_data")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--rate", type=float, default=0.15)
    parser.add_argument("--draws", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--substitute", choices=("zero", "mean"), default="zero")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for folder in args.checkpoint_folder:
        try:
            report = analyse(folder, args)
        except Exception as error:  # noqa: BLE001
            print(f"[FAILED] {folder}: {type(error).__name__}: {error}", flush=True)
            continue
        out_dir = os.path.join(args.results_dir, folder)
        os.makedirs(out_dir, exist_ok=True)
        name = f"sink_hit_confound_{args.substitute}.json"
        with open(os.path.join(out_dir, name), "w") as handle:
            json.dump(report, handle, indent=2)
        print(
            f"\n[ok] {folder} ({report['attack']})  layer {args.layer}, {args.substitute} masking"
        )
        print(
            f"     draws covering a top-{args.top_k} norm token: {report['hit_rate_clean']:.1%}"
        )
        print(f"{'':14s} {'hit':>9s} {'miss':>9s} {'effect':>9s}")
        for split in ("clean", "backdoor"):
            print(
                f"     {split:9s} {report[f'shift_hit_{split}']:9.4f} "
                f"{report[f'shift_miss_{split}']:9.4f} {report[f'effect_{split}']:+9.4f}"
            )
        print(
            f"     AUROC all draws {report['auroc_all']:.3f} | "
            f"misses only {report['auroc_miss_only']:.3f} | hits only {report['auroc_hit_only']:.3f}"
        )


if __name__ == "__main__":
    main()
