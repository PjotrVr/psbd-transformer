"""Refuse to average over checkpoints that were not all swept the same way.

Three conclusions in this project inverted when they were rebuilt on a common set of
checkpoints, and all 3 failed the same way: a mean was taken over whatever cells
happened to exist, so the groups being compared were measured on different problems.

  the placement ranking     n ranged from 15 to 39 across placements. Balanced to a
                            13x15 panel, the winner changed (H19).
  the findings headline     ranked on 1 slice, oracle AUROC at 10% poisoning. Across
                            the other 5 slices a different placement wins.
  the SAM comparison        n = 225 against n = 16. Matched so only rho differed, the
                            sign of the effect flipped (H19).

The fix is mechanical, so it belongs in a tool rather than in a resolution to be
careful. `balanced_panel` finds the largest complete block of (group x unit) cells
and refuses to report anything outside it; `compare` reports the naive answer next to
the balanced one and flags when they disagree.

A "unit" is whatever identifies the same problem: normally (architecture, dataset,
attack, poison rate). A "group" is what is being compared: a placement, an optimizer,
a rate rule.

Example
    PYTHONPATH=. python scripts/balanced_panels/audit.py
    PYTHONPATH=. python scripts/balanced_panels/audit.py --group rho --metric oracle
"""

import argparse
import glob
import itertools
import json
import os
import statistics

MIN_UNITS = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--group", default="placement", choices=("placement", "rho"))
    parser.add_argument(
        "--metric", default="adaptive", choices=("adaptive", "oracle", "gap")
    )
    parser.add_argument("--architecture", default=None)
    parser.add_argument("--min-units", type=int, default=MIN_UNITS)
    parser.add_argument("--include-a2a", action="store_true")
    return parser.parse_args()


def load_cells(results_dir: str, group: str, architecture: str | None) -> dict:
    """{group_value: {unit: (adaptive, oracle)}} over every psbd_metrics.json."""
    cells: dict = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*", "psbd_metrics.json"))):
        with open(path) as handle:
            report = json.load(handle)
        if report.get("attack") in (None, "benign"):
            continue
        folder = os.path.basename(os.path.dirname(path))
        arch = folder.split("_")[0]
        if architecture and arch != architecture:
            continue
        rho = float(report.get("rho") or 0.0)
        for placement, info in (report.get("placements") or {}).items():
            adaptive, oracle = (info or {}).get("adaptive"), (info or {}).get("oracle")
            if not (adaptive and oracle):
                continue
            if group == "placement":
                key = placement
                unit = (
                    arch,
                    report["dataset"],
                    report["attack"],
                    report["poison_rate"],
                    rho,
                )
            else:
                key = rho
                unit = (
                    arch,
                    report["dataset"],
                    report["attack"],
                    report["poison_rate"],
                    placement,
                )
            cells.setdefault(key, {})[unit] = (adaptive["auroc"], oracle["auroc"])
    return cells


def balanced_panel(cells: dict, min_units: int) -> tuple[list, list]:
    """The largest complete (group x unit) block, preferring more groups.

    Ties are broken toward more groups rather than more units, because a comparison
    that drops a group answers a different question, while dropping units only costs
    precision.
    """
    groups = sorted(cells, key=str)
    for size in range(len(groups), 1, -1):
        best = None
        for combo in itertools.combinations(groups, size):
            shared = set.intersection(*(set(cells[g]) for g in combo))
            if len(shared) >= min_units and (
                best is None or len(shared) > len(best[1])
            ):
                best = (list(combo), sorted(shared))
        if best:
            return best
    return [], []


def score(pair: tuple[float, float], metric: str) -> float:
    adaptive, oracle = pair
    return {"adaptive": adaptive, "oracle": oracle, "gap": oracle - adaptive}[metric]


def main() -> None:
    args = parse_args()
    cells = load_cells(args.results_dir, args.group, args.architecture)
    if not args.include_a2a:
        cells = {
            g: {u: v for u, v in units.items() if "badnet_a2a" not in u}
            for g, units in cells.items()
        }
        cells = {g: u for g, u in cells.items() if u}
    if not cells:
        print("no cells found")
        return

    naive = {
        g: statistics.mean(score(v, args.metric) for v in u.values())
        for g, u in cells.items()
    }
    combo, shared = balanced_panel(cells, args.min_units)

    print(
        f"Comparing by {args.group}, metric {args.metric}"
        f"{'' if not args.architecture else ', ' + args.architecture} "
        f"{'(a2a excluded)' if not args.include_a2a else ''}\n"
    )
    print(f"{'group':30} {'naive':>8} {'n':>5} | {'balanced':>9} {'n':>5}")
    print("-" * 64)
    balanced = {
        g: statistics.mean(score(cells[g][u], args.metric) for u in shared)
        for g in combo
    }
    for g in sorted(naive, key=lambda x: -naive[x]):
        cell = (
            f"{balanced[g]:>9.3f} {len(shared):>5}"
            if g in balanced
            else f"{'dropped':>9} {'':>5}"
        )
        print(f"{str(g):30} {naive[g]:>8.3f} {len(cells[g]):>5} | {cell}")

    if not balanced:
        print("\nNo complete block exists: nothing here can be compared honestly.")
        return

    naive_winner = max(balanced, key=lambda g: naive[g])
    balanced_winner = max(balanced, key=balanced.get)
    spread_n = max(len(u) for u in cells.values()) / max(
        min(len(u) for u in cells.values()), 1
    )
    print(f"\npanel: {len(combo)} groups x {len(shared)} units, every cell present")
    print(f"worst coverage imbalance in the naive table: {spread_n:.1f}x")
    if naive_winner != balanced_winner:
        print(
            f"\n  DISAGREES: naive says {naive_winner}, balanced says {balanced_winner}"
        )
    else:
        print(f"\n  agrees: {balanced_winner} wins either way")


if __name__ == "__main__":
    main()
