"""Published PSBD configuration against the best defender-legal one, per checkpoint.

Both are one-sided. Neither reads a poison label to pick its dropout rate. The
only thing fitted on results is the CHOICE of placement and sigma target, which
is why this is a derivation on CIFAR-10 and not yet a claim.
"""

import json

TABLE = json.load(open("scratch/surface.json"))

PAPER = ("post_residual", "absolute", 0.8)
CANDIDATE = ("before_attention_norm", "fractional", 0.7)


def select(rows, placement, target):
    sub = [r for r in rows if r["placement"] == placement and r["sigma"] is not None]
    hit = [r for r in sub if r["sigma"] >= target]
    return min(hit, key=lambda r: r["rate"]) if hit else None


def run(folder, config):
    placement, kind, target = config
    chosen = select(TABLE[folder], placement, target)
    return (chosen[kind], chosen["rate"]) if chosen else (None, None)


def main():
    print(f"paper     = {PAPER[0]} / {PAPER[1]} PSU / sigma>={PAPER[2]}")
    print(f"candidate = {CANDIDATE[0]} / {CANDIDATE[1]} PSU / sigma>={CANDIDATE[2]}")
    print("\nOne-sided AUROC. Nothing is flipped; a number below 0.5 is a failure.\n")
    head = f"{'checkpoint':34} {'paper':>7} {'p':>5} {'cand':>7} {'p':>5} {'delta':>7}"
    print(head)
    print("-" * len(head))
    deltas = []
    for folder in sorted(TABLE):
        if "sam_rho" in folder:
            continue
        a, ap = run(folder, PAPER)
        b, bp = run(folder, CANDIDATE)
        if a is None or b is None:
            continue
        mark = ""
        if "benign" in folder:
            mark = "   <- control, both must sit near 0.5"
        elif "badnet_a2a" in folder:
            mark = "   <- all-to-all, known separate failure (H5)"
        else:
            deltas.append(b - a)
        print(f"{folder:34} {a:>7.3f} {ap:>5g} {b:>7.3f} {bp:>5g} {b - a:>+7.3f}{mark}")
    print("-" * len(head))
    print(
        f"mean delta over {len(deltas)} backdoored (a2a excluded): "
        f"{sum(deltas) / len(deltas):+.3f}   wins {sum(1 for d in deltas if d > 0)}/{len(deltas)}"
    )


if __name__ == "__main__":
    main()
