"""Is the PSBD paper's operating point better than this project's comparison midpoint?

Two different quantities have been used interchangeably in this repo and they are not the
same thing:

  ADAPTIVE_SHIFT_TARGET = 0.8   the PSBD paper's own rule. Yang et al. select the dropout
                                rate "where the sigma of clean validation data approach to a
                                high value (0.8 in our experiments)". This is the OPERATING
                                POINT a deployed PSBD runs at.
  SHIFT_MATCH_TARGETS   = (0.2, 0.4, 0.6, 0.8)   this project's ladder for COMPARING
                                placements at equal effective strength. 0.6 is its midpoint.

A comparison device and an operating point answer different questions, and a placement that
cannot be driven to sigma 0.8 is read at whatever it managed, which quietly turns a matched
comparison into an unmatched one. This script measures both and says which is defensible.

    PYTHONPATH=. python scripts/vit_shift_target_compare.py
"""

import argparse
import json
import statistics

HARD = ("wanet", "tact", "bpp", "adaptive_blend", "lc", "sig")
TOLERANCE = 0.15


def load(path):
    with open(path) as handle:
        return json.load(handle)


def clearing(report):
    return [c for c in report["cells"].values() if c.get("asr_class") == "clears"]


def summarise(cells, placement):
    rows = [c["singles"][placement] for c in cells if placement in c["singles"]]
    if not rows:
        return None
    hard = [
        c["singles"][placement]
        for c in cells
        if placement in c["singles"] and c["attack"] in HARD
    ]
    return {
        "n": len(rows),
        "auroc": statistics.mean([r["auroc"] for r in rows]),
        "hard": statistics.mean([r["auroc"] for r in hard]) if hard else float("nan"),
        "t10": statistics.mean([r["fpr0.10"]["tpr"] for r in rows]),
        "t20": statistics.mean([r["fpr0.20"]["tpr"] for r in rows]),
        "reach": statistics.mean([r["reaches_target"] for r in rows]),
        "sigma": statistics.mean([r["achieved_shift"] for r in rows]),
        "inverted": sum(1 for r in rows if r["auroc"] < 0.5),
        "floor": min(r["auroc"] for r in rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis", default="configs/psbd_basis.json")
    parser.add_argument(
        "--targets",
        nargs="+",
        default=[
            "0.2:results/coverage/probe_fusion_s0.2.json",
            "0.4:results/coverage/probe_fusion_s0.4.json",
            "0.6:results/coverage/probe_fusion.json",
            "0.8:results/coverage/probe_fusion_s0.8.json",
        ],
        help="target:path pairs",
    )
    args = parser.parse_args()

    basis = [e["id"] for e in load(args.basis)["basis"]]
    data = {}
    for pair in args.targets:
        target, path = pair.split(":", 1)
        try:
            data[target] = clearing(load(path))
        except FileNotFoundError:
            continue
    targets = sorted(data, key=float)

    print(
        f"cells clearing the ASR bar: "
        + ", ".join(f"sigma={t}: {len(data[t])}" for t in targets)
    )
    print()
    print("REACH: fraction of cells where the placement actually attained the target")
    print(f"{'placement':46s} " + "".join(f"{'s=' + t:>9s}" for t in targets))
    for placement in basis:
        row = ""
        for target in targets:
            summary = summarise(data[target], placement)
            row += f"{summary['reach']:8.0%} " if summary else "      -- "
        print(f"{placement:46s} {row}")

    print()
    print("MEAN AUROC over all clearing cells")
    print(
        f"{'placement':46s} " + "".join(f"{'s=' + t:>9s}" for t in targets) + "   best"
    )
    for placement in basis:
        values, row = {}, ""
        for target in targets:
            summary = summarise(data[target], placement)
            if summary:
                values[target] = summary["auroc"]
                row += f"{summary['auroc']:9.3f}"
            else:
                row += "       --"
        best = max(values, key=lambda t: values[t]) if values else "--"
        print(f"{placement:46s} {row}   s={best}")

    print()
    print("PANEL-LEVEL, averaged over the 18 basis placements")
    print(
        f"{'target':>8s} {'AUROC':>8s} {'hard':>8s} {'TPR@10':>8s} {'TPR@20':>8s} "
        f"{'reach':>8s} {'inverted':>9s} {'floor':>7s}"
    )
    for target in targets:
        rows = [summarise(data[target], p) for p in basis]
        rows = [r for r in rows if r]
        print(
            f"{target:>8s} {statistics.mean([r['auroc'] for r in rows]):8.3f} "
            f"{statistics.mean([r['hard'] for r in rows]):8.3f} "
            f"{statistics.mean([r['t10'] for r in rows]):8.3f} "
            f"{statistics.mean([r['t20'] for r in rows]):8.3f} "
            f"{statistics.mean([r['reach'] for r in rows]):8.0%} "
            f"{sum(r['inverted'] for r in rows):9d} "
            f"{min(r['floor'] for r in rows):7.3f}"
        )

    print()
    print("RANKING STABILITY: does the winner change with the target?")
    for target in targets:
        scored = {p: summarise(data[target], p) for p in basis}
        scored = {p: s for p, s in scored.items() if s}
        order = sorted(scored, key=lambda p: -scored[p]["auroc"])
        hard_order = sorted(scored, key=lambda p: -scored[p]["hard"])
        print(f"  sigma={target}  overall: {order[0]:44s} hard: {hard_order[0]}")


if __name__ == "__main__":
    main()
