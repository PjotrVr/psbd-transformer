"""Compute (folder, placement, rate) -> (sigma, one-sided AUROC) once, cache to JSON.

Everything downstream is a slice of this table. sigma is clean-validation only,
so any rule built on it is defender-legal; AUROC is the outcome, never an input.
"""

import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

from defences.psbd_cache import (
    baseline_path,
    dropout_pass_path,
    load_baseline,
    load_dropout_pass_probs,
    read_split_manifest,
)
from defences.psbd_metrics import (
    complete_rates,
    pair_clean_to_backdoor,
    psu_from_cache,
    psu_ratio_from_cache,
    shift_ratio,
)

OUT = "scratch/surface.json"


def one_sided_auroc(clean, backdoor):
    labels = np.concatenate([np.zeros(len(clean)), np.ones(len(backdoor))])
    raw = np.concatenate([-clean.float().numpy(), -backdoor.float().numpy()])
    return float(roc_auc_score(labels, raw))


def row(psbd_dir, manifest, placement, rate):
    out = {"placement": placement, "rate": rate}
    per_split = {}
    for split in ("validation", "clean", "backdoor"):
        probs, labels, _ = load_baseline(baseline_path(psbd_dir, split))
        per_pass, argmax = load_dropout_pass_probs(
            dropout_pass_path(psbd_dir, placement, rate, split)
        )
        per_split[split] = {
            "absolute": psu_from_cache(probs, labels, per_pass),
            "fractional": psu_ratio_from_cache(probs, labels, per_pass),
            "sigma": shift_ratio(labels, argmax),
        }
    out["sigma"] = per_split["validation"]["sigma"]
    out["sigma_backdoor"] = per_split["backdoor"]["sigma"]
    for kind in ("absolute", "fractional"):
        clean = pair_clean_to_backdoor(per_split["clean"][kind], manifest)
        out[kind] = one_sided_auroc(clean, per_split["backdoor"][kind])
    return out


def main():
    results_dir = "results"
    table = {}
    if os.path.exists(OUT):
        table = json.load(open(OUT))
    for folder in sys.argv[1:]:
        psbd_dir = os.path.join(results_dir, folder, "psbd")
        if not os.path.isdir(psbd_dir):
            continue
        manifest = read_split_manifest(psbd_dir)
        rows = []
        for placement in sorted(os.listdir(psbd_dir)):
            if not os.path.isdir(os.path.join(psbd_dir, placement)):
                continue
            for rate in complete_rates(psbd_dir, placement):
                try:
                    rows.append(row(psbd_dir, manifest, placement, rate))
                except Exception as error:
                    print(f"  skip {folder}/{placement}/{rate}: {error}")
        table[folder] = rows
        print(f"{folder}: {len(rows)} cells")
    with open(OUT, "w") as handle:
        json.dump(table, handle)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
