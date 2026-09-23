"""The reproduction of Karayalcin et al.'s data-free weight detector, from its record.

The detector scores each class's head row against the early layers' output
projections and flags a model when 1 class stands out by more than a Z of 3,
naming that class the target. We ran it over their grid of layer counts and
thresholds. A model is a hit when some grid cell flags it and names its true
target, and its best Z is the largest Z among those cells. On a benign model
any flag is a false alarm, so its reading is the share of the grid that flags.

The appendix used to state these counts by hand and got 3 of them wrong: it said
4 benign models where the record holds 3, all 5 WaNet models where it holds 4,
and that BPP is named only on CIFAR-10.

    PYTHONPATH=. python scripts/paper/app_weight_detector.py \\
        --results-dir /path/to/results --paper-dir paper
"""

import collections
import os
import sys

sys.path.insert(0, os.getcwd())

from scripts.paper._common import (  # noqa: E402
    attack_label,
    build_parser,
    dataset_label,
    experiment_artifact,
    fmt,
    load_json,
    write_macros,
    write_table,
)

GENERATOR = "scripts/paper/app_weight_detector.py"
SLUG = "weight_detector"
RECORD = "summary.json"


def summarize(checkpoint: dict) -> dict:
    """1 model's reading: whether it was named, its best Z and its flag share."""
    cells = checkpoint["cells"]
    hits = [cell for cell in cells if cell["hit"]]
    summary = {
        "hit": bool(hits),
        "best_z": max((cell["z"] for cell in hits), default=None),
        "flag_share": sum(cell["flagged"] for cell in cells) / len(cells),
    }
    return summary


def main() -> None:
    args = build_parser(__doc__).parse_args()
    path = experiment_artifact(args.results_dir, SLUG, RECORD)
    record = load_json(path)
    if record is None:
        raise SystemExit(f"{path} is missing")

    rows = []
    by_attack = collections.defaultdict(list)
    benign_shares = []
    for folder, checkpoint in sorted(record["checkpoints"].items()):
        reading = summarize(checkpoint)
        if checkpoint["is_benign"]:
            benign_shares.append(reading["flag_share"])
            attack = "benign"
        else:
            attack = checkpoint["attack"]
            by_attack[attack].append((checkpoint, reading))
        rows.append(
            [
                dataset_label(checkpoint["dataset"]),
                "benign" if checkpoint["is_benign"] else attack_label(attack),
                "--"
                if checkpoint["is_benign"]
                else f"{checkpoint['poison_rate'] * 100:g}",
                "--"
                if checkpoint["is_benign"]
                else ("yes" if reading["hit"] else "no"),
                fmt(reading["best_z"], places=1),
                fmt(reading["flag_share"], places=2),
            ]
        )

    write_table(
        path=os.path.join(args.paper_dir, "tables", "weight_detector.tex"),
        generator=GENERATOR,
        inputs=[path],
        caption=(
            "The data-free weight detector of Karayalcin et al. over their grid on our "
            "models. Named is whether some grid cell flags the model and names its "
            "true target, best Z the largest Z among those cells and flag share the "
            "share of the grid that flags, which on a benign model is the false-alarm "
            "rate."
        ),
        label="tab:weight-detector",
        header=["dataset", "attack", "rate %", "named", "best Z", "flag share"],
        rows=rows,
        align="llrlrr",
    )

    backdoored = sum(len(readings) for readings in by_attack.values())
    macros = {
        "weight_detector_backdoored": (
            str(backdoored),
            "backdoored models in the weight-detector reproduction",
        ),
        "weight_detector_benign": (
            str(len(benign_shares)),
            "benign models in the weight-detector reproduction",
        ),
        "weight_detector_benign_flag_min": (
            fmt(min(benign_shares) * 100, places=0),
            "smallest share of the grid, in percent, that flags a benign model",
        ),
        "weight_detector_benign_flag_max": (
            fmt(max(benign_shares) * 100, places=0),
            "largest share of the grid, in percent, that flags a benign model",
        ),
    }
    for attack, readings in sorted(by_attack.items()):
        named = [reading for _, reading in readings if reading["hit"]]
        macros[f"weight_detector_{attack}_models"] = (
            str(len(readings)),
            f"{attack} models in the weight-detector reproduction",
        )
        macros[f"weight_detector_{attack}_named"] = (
            str(len(named)),
            f"{attack} models whose true target the weight detector names",
        )
        if named:
            zs = [reading["best_z"] for reading in named]
            macros[f"weight_detector_{attack}_best_z_min"] = (
                fmt(min(zs), places=0),
                f"smallest best Z over the {attack} models the detector names",
            )
            macros[f"weight_detector_{attack}_best_z_max"] = (
                fmt(max(zs), places=0),
                f"largest best Z over the {attack} models the detector names",
            )
    write_macros(
        os.path.join(args.paper_dir, "tables", "weight_detector.macros.json"),
        GENERATOR,
        [path],
        macros,
    )
    print(f"weight detector: {backdoored} backdoored, {len(benign_shares)} benign")


if __name__ == "__main__":
    main()
