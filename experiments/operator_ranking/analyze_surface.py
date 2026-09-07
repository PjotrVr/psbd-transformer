"""Search the defender-legal configuration space for one that survives 1% poisoning.

A configuration is (placement, score kind, sigma target). The rate is then chosen
by the paper's rule shape: smallest rate whose clean-validation sigma reaches the
target. Nothing here reads a poison label, so every configuration reported is one
a defender could actually deploy.

Scored by mean one-sided AUROC per poison rate, with the benign control and the
count of below-chance checkpoints reported alongside, because a configuration
that gains mean AUROC by inverting somewhere is not an improvement.
"""

import json
from collections import defaultdict

TARGETS = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
TABLE = json.load(open("scratch/surface.json"))


def poison_tag(folder):
    for tag, label in (("_0_01", "1%"), ("_0_05", "5%"), ("_0_1", "10%")):
        if folder.endswith(tag):
            return label
    return "benign" if "benign" in folder else None


def select(rows, target):
    hit = [r for r in rows if r["sigma"] is not None and r["sigma"] >= target]
    return min(hit, key=lambda r: r["rate"]) if hit else None


def evaluate(placement, kind, target):
    """Returns per-poison-rate AUROC lists plus the benign control."""
    by_rate = defaultdict(list)
    benign = None
    for folder, rows in TABLE.items():
        tag = poison_tag(folder)
        if tag is None:
            continue
        sub = [r for r in rows if r["placement"] == placement]
        if not sub:
            continue
        chosen = select(sub, target)
        if chosen is None:
            continue
        if tag == "benign":
            benign = chosen[kind]
        # all-to-all is a known separate failure (H5); keep it out of the mean
        # so it cannot mask or manufacture a low-poison-rate effect.
        elif "badnet_a2a" not in folder:
            by_rate[tag].append((folder, chosen[kind]))
    return by_rate, benign


def main():
    placements = sorted({r["placement"] for rows in TABLE.values() for r in rows})
    results = []
    for placement in placements:
        for kind in ("absolute", "fractional"):
            for target in TARGETS:
                by_rate, benign = evaluate(placement, kind, target)
                if "1%" not in by_rate or len(by_rate["1%"]) < 4:
                    continue
                mean = {
                    k: sum(v for _, v in vals) / len(vals)
                    for k, vals in by_rate.items()
                }
                allv = [v for vals in by_rate.values() for _, v in vals]
                results.append(
                    {
                        "placement": placement,
                        "kind": kind,
                        "target": target,
                        "m1": mean.get("1%"),
                        "m5": mean.get("5%"),
                        "m10": mean.get("10%"),
                        "overall": sum(allv) / len(allv),
                        "below": sum(1 for v in allv if v < 0.5),
                        "n": len(allv),
                        "benign": benign,
                    }
                )

    print("Ranked by mean AUROC at 1% poisoning (all-to-all excluded, one-sided)\n")
    head = f"{'placement':28} {'score':11} {'sig':>5} {'1%':>6} {'5%':>6} {'10%':>6} {'all':>6} {'<0.5':>6} {'benign':>7}"
    print(head)
    print("-" * len(head))
    for r in sorted(results, key=lambda r: -(r["m1"] or 0))[:22]:
        print(
            f"{r['placement']:28} {r['kind']:11} {r['target']:>5.2f} "
            f"{r['m1']:>6.3f} {r['m5'] or 0:>6.3f} {r['m10'] or 0:>6.3f} "
            f"{r['overall']:>6.3f} {r['below']:>3}/{r['n']:<2} "
            f"{r['benign'] if r['benign'] is not None else float('nan'):>7.3f}"
        )

    print("\n\nThe published configuration, for reference")
    print(head)
    print("-" * len(head))
    for r in results:
        if (
            r["placement"] == "post_residual"
            and r["kind"] == "absolute"
            and r["target"] == 0.8
        ):
            print(
                f"{r['placement']:28} {r['kind']:11} {r['target']:>5.2f} "
                f"{r['m1']:>6.3f} {r['m5'] or 0:>6.3f} {r['m10'] or 0:>6.3f} "
                f"{r['overall']:>6.3f} {r['below']:>3}/{r['n']:<2} "
                f"{r['benign'] if r['benign'] is not None else float('nan'):>7.3f}"
            )

    print("\n\nPer-checkpoint detail for the best 1% configuration")
    best = max(results, key=lambda r: r["m1"])
    by_rate, benign = evaluate(best["placement"], best["kind"], best["target"])
    print(f"  {best['placement']} / {best['kind']} / sigma>={best['target']}")
    for tag in ("1%", "5%", "10%"):
        for folder, value in sorted(by_rate.get(tag, [])):
            print(f"    {tag:>4}  {folder:34} {value:.3f}")
    print(f"    benign control: {benign:.3f}")


if __name__ == "__main__":
    main()
