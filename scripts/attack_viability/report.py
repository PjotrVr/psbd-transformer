"""Report measured ASR per (attack, dataset, poison rate, optimizer), from disk.

Reads the metrics.json every trained checkpoint already carries. No GPU, no model
loading, runs in under a second on the login node.

Exists because a detection sweep on a model whose attack never took measures
nothing, and that failure is invisible: the jobs succeed and write well-formed
output. This is the gate that catches it, and pbs/generate_psbd_jobs.py applies
the same rule so the sweep grid can never drift from what is viable.
"""

import argparse
import collections
import json
import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--architecture", default="vit", choices=("vit", "swin"))
    parser.add_argument("--dataset", nargs="*", default=["cifar10", "cifar100"])
    parser.add_argument(
        "--min-asr",
        type=float,
        default=0.8,
        help="viability line; the PSBD paper's own failed-case threshold for TPR",
    )
    return parser.parse_args()


def load_metrics(checkpoints_dir: str) -> list[dict]:
    """Every checkpoint's measured metrics, tagged with its folder name."""
    rows = []
    for folder in sorted(os.listdir(checkpoints_dir)):
        path = os.path.join(checkpoints_dir, folder, "metrics.json")
        if not os.path.exists(path):
            continue
        with open(path) as handle:
            metrics = json.load(handle)
        metrics["folder"] = folder
        rows.append(metrics)
    return rows


def is_sam(folder: str) -> bool:
    return "sam_rho" in folder


def viability_table(rows: list[dict], architecture: str, datasets: list[str]) -> dict:
    """(attack, dataset) -> {poison_rate: asr}, Adam runs only.

    Adam is the unmarked baseline, so the viability question is asked there; the
    SAM columns are checked separately because an attack that only survives under
    one optimizer is not a stable grid member.
    """
    table: dict = collections.defaultdict(dict)
    for row in rows:
        folder = row["folder"]
        if not folder.startswith(f"{architecture}_") or is_sam(folder):
            continue
        if row.get("dataset") not in datasets or row.get("attack") in (None, "benign"):
            continue
        table[(row["attack"], row["dataset"])][row["poison_rate"]] = row["asr"]
    return table


def sam_holds(rows: list[dict], architecture: str, dataset: str, attack: str,
              poison_rate: float, min_asr: float) -> bool:
    """Whether the attack stays viable across every SAM rho at this poison rate."""
    matching = [
        row
        for row in rows
        if is_sam(row["folder"])
        and row["folder"].startswith(f"{architecture}_{dataset}_")
        and row.get("attack") == attack
        and row.get("poison_rate") == poison_rate
    ]
    return bool(matching) and all(row["asr"] >= min_asr for row in matching)


def main() -> None:
    args = parse_args()
    rows = load_metrics(args.checkpoints_dir)
    table = viability_table(rows, args.architecture, args.dataset)

    rates = sorted({rate for column in table.values() for rate in column})
    header = f"{'attack':16} {'dataset':9} " + " ".join(f"{r:>7}" for r in rates)
    print(f"ASR by poison rate ({args.architecture}, Adam), gate at {args.min_asr}\n")
    print(header + "   verdict")
    print("-" * (len(header) + 30))

    viable = []
    for (attack, dataset) in sorted(table):
        column = table[(attack, dataset)]
        swept = [rate for rate in rates if rate in column and rate >= 0.01]
        passes = [rate for rate in swept if column[rate] >= args.min_asr]
        cells = " ".join(
            f"{column[rate]:>7.3f}" if rate in column else f"{'--':>7}" for rate in rates
        )
        if len(passes) == len(swept) and swept:
            sam_ok = all(
                sam_holds(rows, args.architecture, dataset, attack, rate, args.min_asr)
                for rate in passes
            )
            verdict = "VIABLE" if sam_ok else "viable (adam only, SAM drops below gate)"
            if sam_ok:
                viable.append((attack, dataset))
        else:
            failed = [f"{rate:g}" for rate in swept if rate not in passes]
            verdict = f"excluded, fails at {', '.join(failed)}"
        print(f"{attack:16} {dataset:9} {cells}   {verdict}")

    print(f"\nviable everywhere including all SAM rhos: {len(viable)}")
    for attack, dataset in viable:
        print(f"  {dataset:9} {attack}")


if __name__ == "__main__":
    main()
