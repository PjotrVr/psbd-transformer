"""Rank-average PSU across placements, instead of betting on one placement.

No single placement wins on every attack: the patch trigger wants
before_attention_norm, wanet and lc want post_residual. A defender cannot know
which attack they face, so picking one placement is a bet. Averaging per-sample
ranks across placements is the defender-legal alternative.

Each placement contributes at ITS OWN sigma-selected rate, so placements are
combined at matched disturbance rather than at a shared p, which would just
measure which placement perturbs hardest.

Every split is ranked against the CLEAN VALIDATION distribution of the same
placement, which is the one distribution a defender holds. An earlier version
ranked jointly over torch.cat([clean, backdoor]), so the backdoor split helped
set the scale it was then scored on and the ensemble AUROC was transductive.
"""

import json
import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from defences.psbd_cache import (
    baseline_path,
    dropout_pass_path,
    load_baseline,
    load_dropout_pass_probs,
    read_split_manifest,
)
from defences.psbd_metrics import pair_clean_to_backdoor, psu_ratio_from_cache, to_rank

SURFACE = json.load(open("scratch/surface.json"))
TARGET = 0.6
MEMBERS = (
    "before_attention_norm",
    "before_mlp_residual",
    "pre_residual_blocks_5_8",
    "post_residual",
)


def chosen_rate(folder, placement, target=TARGET):
    sub = [
        r
        for r in SURFACE[folder]
        if r["placement"] == placement
        and r["sigma"] is not None
        and r["sigma"] >= target
    ]
    return min(sub, key=lambda r: r["rate"])["rate"] if sub else None


def scores(psbd_dir, placement, rate, split):
    probs, labels, _ = load_baseline(baseline_path(psbd_dir, split))
    per_pass, _ = load_dropout_pass_probs(
        dropout_pass_path(psbd_dir, placement, rate, split)
    )
    return psu_ratio_from_cache(probs, labels, per_pass)


def auroc(clean, backdoor):
    labels = np.concatenate([np.zeros(len(clean)), np.ones(len(backdoor))])
    return float(roc_auc_score(labels, np.concatenate([-clean, -backdoor])))


def main():
    results_dir = "results"
    print(
        f"rank-average over {len(MEMBERS)} placements, each at its own sigma>={TARGET} rate"
    )
    print("one-sided AUROC\n")
    head = (
        f"{'checkpoint':34} "
        + " ".join(f"{m[:14]:>14}" for m in MEMBERS)
        + f" {'ENSEMBLE':>9}"
    )
    print(head)
    print("-" * len(head))
    gains = []
    scored = []
    for folder in sorted(SURFACE):
        if "sam_rho" in folder:
            continue
        psbd_dir = os.path.join(results_dir, folder, "psbd")
        manifest = read_split_manifest(psbd_dir)
        clean_ranks, backdoor_ranks, singles = [], [], []
        for placement in MEMBERS:
            rate = chosen_rate(folder, placement)
            if rate is None:
                singles.append(None)
                continue
            try:
                clean = pair_clean_to_backdoor(
                    scores(psbd_dir, placement, rate, "clean"), manifest
                )
                backdoor = scores(psbd_dir, placement, rate, "backdoor")
                validation = scores(psbd_dir, placement, rate, "validation")
            except Exception:
                singles.append(None)
                continue
            singles.append(auroc(clean.float().numpy(), backdoor.float().numpy()))
            # Clean validation is the shared fixed reference: it puts placements on
            # one scale without either scored split contributing to that scale.
            clean_ranks.append(to_rank(clean, validation))
            backdoor_ranks.append(to_rank(backdoor, validation))
        if len(clean_ranks) < 2:
            continue
        ens = auroc(
            torch.stack(clean_ranks).mean(0).numpy(),
            torch.stack(backdoor_ranks).mean(0).numpy(),
        )
        cells = " ".join(
            f"{s:>14.3f}" if s is not None else f"{'--':>14}" for s in singles
        )
        note = ""
        valid = [s for s in singles if s is not None]
        if "benign" not in folder and "badnet_a2a" not in folder:
            gains.append(ens - max(valid))
            scored.append((ens, singles[0], max(valid)))
        elif "benign" in folder:
            note = "  <- control"
        print(f"{folder:34} {cells} {ens:>9.3f}{note}")
    print("-" * len(head))
    print(
        f"mean over {len(scored)} backdoored non-a2a checkpoints:"
        f"  ensemble {sum(e for e, _, _ in scored) / len(scored):.3f}"
        f"  {MEMBERS[0][:14]} {sum(m for _, m, _ in scored) / len(scored):.3f}"
        f"  best single {sum(b for _, _, b in scored) / len(scored):.3f}"
    )
    print(
        f"ensemble minus BEST single placement, mean over {len(gains)}: {sum(gains) / len(gains):+.3f}"
        f"   (best single is an oracle a defender cannot pick)"
    )


if __name__ == "__main__":
    main()
