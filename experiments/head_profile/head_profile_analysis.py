"""Read the 144-head leave-one-out profiles and test what separates.

Each sample has a 144-long sensitivity vector: how far its confidence in its own
unablated predicted class falls when each head is deleted in turn.

    s_u(x) = P_c(x) - P_c(x with head u deleted)

The primary statistic is the MEAN sensitivity, whose direction is PSBD's own and
is fixed here from theory before looking at anything: a backdoored prediction is
carried by a robust shortcut, so it should lose less confidence when any single
head is removed, giving a LOW mean. Every other statistic below is exploratory,
carries a direction declared once in DIRECTIONS, and is applied identically to
every checkpoint. No direction is ever chosen per checkpoint, and a score below
0.5 is reported as a failure of that statistic rather than re-signed.

The benign control is the check that matters: a clean model probed with the same
trigger must sit near 0.5 on every statistic. If concentration separates there,
it is reading prediction confidence rather than a backdoor, which is the confound
H12 had to rule out for PSU itself.
"""

import os
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from defences.cache import baseline_path, load_baseline, read_split_manifest
from defences.decision import pair_clean_to_backdoor

# "low" means a low value indicates poisoned, which is PSBD's direction.
DIRECTIONS = {
    "mean": "low",
    "max": "low",
    "min": "low",
    "std": "low",
    "gini": "low",
    "top5_mass": "low",
}


def load_profile(psbd_dir, split):
    path = os.path.join(psbd_dir, "head_profile", f"{split}.pt")
    return torch.load(path)["profile"].float() if os.path.exists(path) else None


def sensitivity(psbd_dir, split):
    """(N, 144) confidence drop per head, from the cached no-ablation baseline."""
    profile = load_profile(psbd_dir, split)
    if profile is None:
        return None
    probs, labels, _ = load_baseline(baseline_path(psbd_dir, split))
    tracked = probs.gather(1, labels.unsqueeze(1)).squeeze(1).float()
    return (tracked.unsqueeze(0) - profile).transpose(0, 1)


def gini(profile):
    x = torch.sort(profile.clamp_min(0), dim=1).values
    n = x.shape[1]
    index = torch.arange(1, n + 1, dtype=x.dtype)
    total = x.sum(dim=1).clamp_min(1e-9)
    return (2 * (index * x).sum(dim=1) / (n * total)) - (n + 1) / n


def statistics(profile):
    positive = profile.clamp_min(0)
    return {
        "mean": profile.mean(dim=1),
        "max": profile.max(dim=1).values,
        "min": profile.min(dim=1).values,
        "std": profile.std(dim=1),
        "gini": gini(profile),
        "top5_mass": positive.topk(5, dim=1).values.sum(dim=1)
        / positive.sum(dim=1).clamp_min(1e-9),
    }


def auroc(clean, backdoor, direction):
    labels = np.concatenate([np.zeros(len(clean)), np.ones(len(backdoor))])
    scores = np.concatenate([clean.numpy(), backdoor.numpy()])
    if direction == "low":
        scores = -scores
    return float(roc_auc_score(labels, scores))


def main():
    results_dir = "results"
    names = list(DIRECTIONS)
    head = f"{'checkpoint':34} " + " ".join(f"{n:>10}" for n in names)
    print("144-head leave-one-out sensitivity, one-sided AUROC")
    print(
        "direction fixed a priori per statistic; below 0.5 is a failure, never flipped\n"
    )
    print(head)
    print("-" * len(head))

    for folder in sys.argv[1:]:
        psbd_dir = os.path.join(results_dir, folder, "psbd")
        clean_raw = sensitivity(psbd_dir, "clean")
        backdoor_raw = sensitivity(psbd_dir, "backdoor")
        if clean_raw is None or backdoor_raw is None:
            print(f"{folder:34}  (no profile yet)")
            continue
        manifest = read_split_manifest(psbd_dir)
        clean_raw = pair_clean_to_backdoor(clean_raw, manifest)

        clean, backdoor = statistics(clean_raw), statistics(backdoor_raw)
        cells = " ".join(
            f"{auroc(clean[n], backdoor[n], DIRECTIONS[n]):>10.3f}" for n in names
        )
        mark = "   <- control, must sit near 0.5" if "benign" in folder else ""
        print(f"{folder:34} {cells}{mark}")

    print("\nPer-head detail for the first checkpoint with a profile")
    for folder in sys.argv[1:]:
        psbd_dir = os.path.join(results_dir, folder, "psbd")
        raw = sensitivity(psbd_dir, "backdoor")
        if raw is None:
            continue
        per_head = raw.mean(dim=0)
        order = torch.argsort(per_head, descending=True)[:8]
        print(f"  {folder}: most damaging heads (block, head, mean drop)")
        for index in order:
            block, head = divmod(int(index), 12)
            print(f"    block {block + 1:>2} head {head:>2}   {per_head[index]:.4f}")
        break


if __name__ == "__main__":
    main()
