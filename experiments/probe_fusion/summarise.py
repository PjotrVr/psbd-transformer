"""Read measure.py's output under the pre-registered selection protocol.

Kept separate from measure.py because the protocol, not the measurement, is what decides
whether a fusion result means anything. Selecting the best of 20 combinations on the cells
the headline is read from manufactures a winner out of noise, so one combination is chosen
on CIFAR-10 and GTSRB and reported on CIFAR-100 and Tiny, which never enter the selection.

The comparison is against a single FIXED probe, not against each combination's own best
member. A defender cannot know which member is best without labels, so "beats its best
member" is an oracle question; "beats the one config I would have deployed anyway" is the
deployable one.

    PYTHONPATH=. python experiments/probe_fusion/summarise.py
"""

import argparse
import collections
import json
import random
import statistics

BOOTSTRAP_DRAWS = 5000


def load(path: str) -> dict:
    with open(path) as handle:
        return json.load(handle)


def implanted_cells(report: dict) -> dict:
    """Only cells whose attack actually implanted.

    A detection number on a backdoor that was never planted measures nothing, and averaging
    it in flatters or punishes the detector at random.
    """
    return {
        name: cell
        for name, cell in report["cells"].items()
        if cell.get("asr_class") == "clears"
    }


def deltas(
    cells: dict, combination: str, reference: str, datasets: set, rule: str
) -> list:
    """Fused AUROC minus the fixed probe's, on cells carrying both."""
    return [
        cell["combinations"][combination][rule]["auroc"]
        - cell["singles"][reference]["auroc"]
        for cell in cells.values()
        if cell.get("dataset") in datasets
        and combination in cell["combinations"]
        and reference in cell["singles"]
    ]


def bootstrap_interval(values: list, seed: int = 0) -> tuple:
    if not values:
        return (float("nan"), float("nan"))
    random.seed(seed)
    draws = sorted(
        statistics.mean(random.choices(values, k=len(values)))
        for _ in range(BOOTSTRAP_DRAWS)
    )
    lower = int(0.025 * BOOTSTRAP_DRAWS)
    upper = int(0.975 * BOOTSTRAP_DRAWS) - 1
    return draws[lower], draws[upper]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fusion", default="results/coverage/probe_fusion.json")
    parser.add_argument("--declaration", default="configs/psbd_basis.json")
    parser.add_argument("--reference", default="before_attention_norm_token_mask")
    parser.add_argument("--rule", default="min_rank", choices=("min_rank", "mean_rank"))
    parser.add_argument("--min-cells", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = load(args.fusion)
    declaration = load(args.declaration)
    protocol = declaration["selection_protocol"]
    select_on = set(protocol["select_on_datasets"])
    report_on = set(protocol["report_on_datasets"])

    cells = implanted_cells(report)
    names = {name for cell in cells.values() for name in cell["combinations"]}

    selection = {
        name: deltas(cells, name, args.reference, select_on, args.rule)
        for name in names
    }
    selection = {
        name: values
        for name, values in selection.items()
        if len(values) >= args.min_cells
    }
    if not selection:
        print("not enough coverage on the selection split yet")
        return

    print(f"cells clearing the ASR bar: {len(cells)}   rule: {args.rule}")
    print(f"reference probe: {args.reference}\n")
    print(f"SELECTION split {sorted(select_on)}")
    ranked = sorted(selection, key=lambda name: -statistics.mean(selection[name]))
    for name in ranked[:6]:
        print(
            f"   {name:26s} n={len(selection[name]):3d}  {statistics.mean(selection[name]):+.4f}"
        )
    winner = ranked[0]

    held = deltas(cells, winner, args.reference, report_on, args.rule)
    lower, upper = bootstrap_interval(held)
    print(f"\nHELD-OUT split {sorted(report_on)}   combination: {winner}")
    print(f"   n           {len(held)}")
    print(f"   mean delta  {statistics.mean(held):+.4f}")
    print(f"   95% CI      [{lower:+.4f}, {upper:+.4f}]")
    print(f"   wins        {sum(1 for value in held if value > 0)}/{len(held)}")
    print(
        f"   verdict     {'HOLDS' if lower > 0 else 'NOT SUPPORTED, the interval spans zero'}"
    )

    per_attack = collections.defaultdict(list)
    for cell in cells.values():
        if (
            cell.get("dataset") in report_on
            and winner in cell["combinations"]
            and args.reference in cell["singles"]
        ):
            per_attack[cell["attack"]].append(
                cell["combinations"][winner][args.rule]["auroc"]
                - cell["singles"][args.reference]["auroc"]
            )
    print("\n   held-out per attack:")
    for attack, values in sorted(
        per_attack.items(), key=lambda item: -statistics.mean(item[1])
    ):
        print(f"     {attack:16s} n={len(values):2d}  {statistics.mean(values):+.4f}")

    controls = [
        name
        for name in names
        if any(
            entry["id"] == name and entry["family"] == "NEG_control"
            for entry in declaration["combinations"]["pairs"]
        )
    ]
    print(
        "\n   negative controls on the held-out split (should be worse than the winner):"
    )
    for name in controls:
        values = deltas(cells, name, args.reference, report_on, args.rule)
        if values:
            print(f"     {name:26s} n={len(values):2d}  {statistics.mean(values):+.4f}")


if __name__ == "__main__":
    main()
