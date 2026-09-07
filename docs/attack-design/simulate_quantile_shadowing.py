"""Simulate A1 on cached PSU, with no training and no GPU.

Two halves of the attack are applied to the cached scores of a real checkpoint:

  sharpening  a fraction of the clean population, selected by an input-side
              predicate the defender's validation images share, is driven to
              near-certainty, which sends its PSU to approximately 0
  centring    the poisoned population is parked at the CLEAN MEDIAN rather than
              at the clean mean, which is where a hinge on means would put it

The question is whether the pair (AUROC barely moves, TPR at a deployable FPR
collapses) is reachable. If it is not reachable in simulation it is not worth a
GPU run.
"""

import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, ".")
from psbd.cache import dropout_pass_path
from psbd.decision import pair_clean_to_backdoor
from psbd.scores import psu_ratio_from_cache

FOLDER = sys.argv[1] if len(sys.argv) > 1 else "vit_cifar100_badnet_a2o_0_01"
CONFIG = sys.argv[2] if len(sys.argv) > 2 else "before_attention_norm_token_mask"
RATE = float(sys.argv[3]) if len(sys.argv) > 3 else 0.4
SHARE = float(sys.argv[4]) if len(sys.argv) > 4 else 0.30


def load(psbd_dir, split):
    d = torch.load(
        os.path.join(psbd_dir, f"baseline_{split}.pt"),
        map_location="cpu",
        weights_only=False,
    )
    passes = torch.load(
        dropout_pass_path(psbd_dir, CONFIG, RATE, split),
        map_location="cpu",
        weights_only=False,
    )
    psu = psu_ratio_from_cache(
        d["probs"].float(), d["labels"], passes["per_pass_probs"]
    )
    return d["probs"].float(), psu


def rates(validation, clean, backdoor, q):
    t = float(np.quantile(validation.numpy(), q))
    return (
        float((backdoor < t).float().mean()),
        float((clean < t).float().mean()),
    )


def auroc(clean, backdoor):
    s = np.concatenate([-clean.numpy(), -backdoor.numpy()])
    y = np.concatenate([np.zeros(len(clean)), np.ones(len(backdoor))])
    return float(roc_auc_score(y, s))


def sharpen(psu, probs, share, generator, floor):
    """Send the top-confidence `share` of a split's PSU to approximately 0.

    Selecting by baseline confidence stands in for a deterministic input-space
    predicate. What matters for the simulation is only that the SAME fraction of
    validation and clean-test samples is affected, which is what an input-side
    predicate guarantees and a random training subset would not.
    """
    keep = psu.clone()
    n = int(share * len(psu))
    if n == 0:
        return keep
    chosen = probs.max(dim=1).values.argsort(descending=True)[:n]
    keep[chosen] = floor + torch.rand(n, generator=generator) * 0.02
    return keep


def main():
    psbd_dir = os.path.join("results", FOLDER, "psbd")
    manifest = json.load(open(os.path.join(psbd_dir, "split_manifest.json")))
    val_probs, val_psu = load(psbd_dir, "validation")
    clean_probs, clean_psu_all = load(psbd_dir, "clean")
    _, bd_psu = load(psbd_dir, "backdoor")
    clean_psu = pair_clean_to_backdoor(clean_psu_all, manifest)
    clean_probs = pair_clean_to_backdoor(clean_probs, manifest)

    generator = torch.Generator().manual_seed(0)
    rows = []

    def row(name, v, c, b):
        t25, f25 = rates(v, c, b, 0.25)
        t05, f05 = rates(v, c, b, 0.05)
        rows.append((name, auroc(c, b), t25, f25, t05, f05))

    row("baseline", val_psu, clean_psu, bd_psu)

    centred = (
        torch.full_like(bd_psu, float(clean_psu.median()))
        + torch.randn(len(bd_psu), generator=generator) * float(clean_psu.std()) * 0.5
    )
    row("centring only", val_psu, clean_psu, centred)

    # One floor shared by both splits, because the sharpened samples are the same
    # behaviour of the same model and must not sit at 2 different levels.
    floor = min(float(val_psu.min()), float(clean_psu.min()))
    for share in (0.02, 0.05, 0.08, 0.12, 0.20, 0.30):
        v2 = sharpen(val_psu, val_probs, share, generator, floor)
        c2 = sharpen(clean_psu, clean_probs, share, generator, floor)
        row(f"sharpen {share:.2f}", v2, c2, bd_psu)
        row(f"sharpen {share:.2f} + centre", v2, c2, centred)

    print(f"{FOLDER}  {CONFIG}  rate={RATE}  sharpened share={SHARE}")
    print(
        f"{'variant':18s} {'AUROC':>7s} {'TPR@25':>7s} {'FPR@25':>7s} {'TPR@05':>7s} {'FPR@05':>7s}"
    )
    for name, a, t25, f25, t05, f05 in rows:
        print(f"{name:18s} {a:7.3f} {t25:7.3f} {f25:7.3f} {t05:7.3f} {f05:7.3f}")


if __name__ == "__main__":
    main()
