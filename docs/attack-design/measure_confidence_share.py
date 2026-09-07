"""How much of PSU is explained by the softmax vector alone, by probe position.

CPU only, reads the cached sweep tensors. Tests the claim that the closed form
tr(grad^2_z p_c) = 2 p_c (||p||^2 - p_c) makes a head-adjacent probe a function
of p alone, so an attacker who matches p defeats it, while an input-side probe
keeps a residual the attacker never constrained.
"""

import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, ".")
from psbd.cache import dropout_pass_path
from psbd.decision import (
    complete_rates,
    pair_clean_to_backdoor,
    select_rate_at_matched_shift,
)
from psbd.scores import psu_ratio_from_cache, shift_ratio

RESULTS = "results"


def load_split(psbd_dir, split):
    d = torch.load(
        os.path.join(psbd_dir, f"baseline_{split}.pt"),
        map_location="cpu",
        weights_only=False,
    )
    return d["probs"].float(), d["labels"]


def features(probs):
    """(p_c, entropy, l2sq, closed_form) per sample."""
    p = probs.clamp_min(1e-12)
    pc = p.max(dim=1).values
    ent = -(p * p.log()).sum(dim=1)
    l2sq = (p * p).sum(dim=1)
    closed = pc * (pc - l2sq)  # phi at the logit layer, up to sigma^2
    return torch.stack([pc, ent, l2sq, closed], dim=1)


def r2(y, X):
    """R^2 of an ordinary least squares fit of y on [1, X, X^2]."""
    A = torch.cat([torch.ones(len(X), 1), X, X**2], dim=1).numpy()
    yv = y.numpy()
    beta, *_ = np.linalg.lstsq(A, yv, rcond=None)
    pred = A @ beta
    ss_res = float(((yv - pred) ** 2).sum())
    ss_tot = float(((yv - yv.mean()) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12), beta


def auroc(clean, backdoor):
    s = np.concatenate([-clean.numpy(), -backdoor.numpy()])
    y = np.concatenate([np.zeros(len(clean)), np.ones(len(backdoor))])
    return float(roc_auc_score(y, s))


def analyse(folder, position_config, target_sigma=0.6):
    psbd_dir = os.path.join(RESULTS, folder, "psbd")
    manifest = json.load(open(os.path.join(psbd_dir, "split_manifest.json")))
    rates = complete_rates(psbd_dir, position_config)
    if not rates:
        return None

    shift_by_rate = {}
    per_rate = {}
    for rate in rates:
        got = {}
        for split in ("validation", "clean", "backdoor"):
            f = dropout_pass_path(psbd_dir, position_config, rate, split)
            got[split] = torch.load(f, map_location="cpu", weights_only=False)
        per_rate[rate] = got
        vp, vl = load_split(psbd_dir, "validation")
        shift_by_rate[rate] = shift_ratio(vl, got["validation"]["per_pass_argmax"])

    rate = select_rate_at_matched_shift(shift_by_rate, target_sigma)
    if rate is None:
        return None

    out = {}
    probs, labels = {}, {}
    for split in ("validation", "clean", "backdoor"):
        probs[split], labels[split] = load_split(psbd_dir, split)

    psu = {
        split: psu_ratio_from_cache(
            probs[split], labels[split], per_rate[rate][split]["per_pass_probs"]
        )
        for split in ("validation", "clean", "backdoor")
    }
    # Across-pass spread of the tracked probability, free from the same cache.
    spread = {}
    for split in ("validation", "clean", "backdoor"):
        pp = per_rate[rate][split]["per_pass_probs"].float()
        spread[split] = pp.std(dim=0)

    clean_psu = pair_clean_to_backdoor(psu["clean"], manifest)
    clean_probs = pair_clean_to_backdoor(probs["clean"], manifest)
    clean_spread = pair_clean_to_backdoor(spread["clean"], manifest)

    fx_val = features(probs["validation"])
    fx_clean = features(clean_probs)
    fx_bd = features(probs["backdoor"])

    share, beta = r2(psu["validation"], fx_val)

    def predict(fx):
        A = torch.cat([torch.ones(len(fx), 1), fx, fx**2], dim=1).numpy()
        return torch.tensor(A @ beta, dtype=torch.float32)

    resid_clean = clean_psu - predict(fx_clean)
    resid_bd = psu["backdoor"] - predict(fx_bd)

    out["rate"] = rate
    out["sigma"] = shift_by_rate[rate]
    out["auroc_psu"] = auroc(clean_psu, psu["backdoor"])
    out["auroc_closedform"] = auroc(fx_clean[:, 3], fx_bd[:, 3])
    out["auroc_confidence"] = auroc(-fx_clean[:, 0], -fx_bd[:, 0])
    out["auroc_residual"] = auroc(resid_clean, resid_bd)
    out["auroc_spread"] = auroc(-clean_spread, -spread["backdoor"])
    out["r2_psu_on_p"] = share
    for q in (0.05, 0.25):
        t = float(np.quantile(psu["validation"].numpy(), q))
        out[f"tpr{q}"] = float((psu["backdoor"] < t).float().mean())
        out[f"fpr{q}"] = float((clean_psu < t).float().mean())
        tr = float(np.quantile((psu["validation"] - predict(fx_val)).numpy(), q))
        out[f"rtpr{q}"] = float((resid_bd < tr).float().mean())
    return out


if __name__ == "__main__":
    folder = sys.argv[1]
    positions = sys.argv[2:]
    print(f"{folder}")
    print(
        f"{'position':38s} {'rate':>5s} {'sig':>5s} {'PSU':>6s} {'closed':>6s} {'conf':>6s} {'resid':>6s} {'spread':>6s} {'R2':>6s} {'T05':>5s} {'rT05':>5s} {'T25':>5s} {'rT25':>5s}"
    )
    for pos in positions:
        try:
            r = analyse(folder, pos)
        except Exception as exc:
            print(f"{pos:38s} FAILED {exc}")
            continue
        if r is None:
            print(f"{pos:38s} no usable rate")
            continue
        print(
            f"{pos:38s} {r['rate']:5.2f} {r['sigma']:5.2f} {r['auroc_psu']:6.3f} "
            f"{r['auroc_closedform']:6.3f} {r['auroc_confidence']:6.3f} "
            f"{r['auroc_residual']:6.3f} {r['auroc_spread']:6.3f} {r['r2_psu_on_p']:6.3f} "
            f"{r['tpr0.05']:5.2f} {r['rtpr0.05']:5.2f} {r['tpr0.25']:5.2f} {r['rtpr0.25']:5.2f}"
        )
