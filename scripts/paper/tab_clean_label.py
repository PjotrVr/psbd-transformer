"""The clean-label difficulties, read from training sidecars: caps, SIG, LC, new datasets.

4 tables from checkpoints/<folder>/args.json. The GTSRB SIG runs at target class
1 with 5 seeds, the Label-Consistent epsilon pilot with its adversarial bases, the
SVHN and EuroSAT SIG carriers with their blend controls, and the multi-target
SIG probe on the 2 primary datasets. A 5th number is the count of GTSRB runs
that diverged in their last epochs, read as a clean accuracy below half the
benign reference, since those runs are excluded from every mean.

    PYTHONPATH=. python scripts/paper/tab_clean_label.py \\
        --results-dir /path/to/results --paper-dir paper --checkpoints-dir checkpoints
"""

import collections
import glob
import os
import sys

sys.path.insert(0, os.getcwd())

from scripts.paper._common import (  # noqa: E402
    build_parser_with_checkpoints,
    fmt,
    load_args_json,
    load_declaration,
    mean_or_none,
    std_or_none,
    write_macros,
    write_table,
)

GENERATOR = "scripts/paper/tab_clean_label.py"
DIVERGED_FRACTION_OF_BENIGN = 0.5
CLEAN_LABEL_DATASETS = ("cifar10", "cifar100", "gtsrb", "tiny")


def sidecars(checkpoints_dir: str, pattern: str) -> dict[str, dict]:
    """folder name to args.json for every folder matching the glob pattern."""
    found = {}
    for path in sorted(glob.glob(os.path.join(checkpoints_dir, pattern, "args.json"))):
        folder = os.path.basename(os.path.dirname(path))
        sidecar = load_args_json(checkpoints_dir, folder)
        if sidecar is not None:
            found[folder] = sidecar
    return found


def benign_accuracy(checkpoints_dir: str, dataset: str) -> float | None:
    sidecar = load_args_json(checkpoints_dir, f"vit_{dataset}_benign") or {}
    accuracy = sidecar.get("clean_accuracy")
    return accuracy


def usable(sidecar: dict, benign: float | None) -> bool:
    """A run whose clean accuracy is not collapsed against its benign reference."""
    accuracy = sidecar.get("clean_accuracy")
    if accuracy is None or benign is None:
        return False
    ok = accuracy >= DIVERGED_FRACTION_OF_BENIGN * benign
    return ok


def clears(sidecar: dict, benign: float | None, asr_bar: float, drop_bar: float) -> bool:
    asr = sidecar.get("asr")
    accuracy = sidecar.get("clean_accuracy")
    if asr is None or accuracy is None or benign is None:
        return False
    passes = asr >= asr_bar and (accuracy - benign) >= drop_bar
    return passes


def cell_rows(
    groups: dict[tuple, list[dict]],
    benign_of: dict[str, float | None],
    asr_bar: float,
    drop_bar: float,
) -> list[list[str]]:
    """1 row per (dataset, attack, rate, target): seed means over usable runs."""
    rows = []
    for key in sorted(groups):
        dataset, attack, rate, target = key
        runs = groups[key]
        benign = benign_of[dataset]
        good = [run for run in runs if usable(run, benign)]
        asrs = [run["asr"] for run in good if run.get("asr") is not None]
        accuracies = [run["clean_accuracy"] for run in good]
        rows.append(
            [
                dataset,
                attack,
                f"{rate:g}",
                str(target),
                f"{len(good)}/{len(runs)}",
                fmt(mean_or_none(asrs)),
                fmt(std_or_none(asrs)),
                fmt(min(asrs)) if asrs else "--",
                fmt(max(asrs)) if asrs else "--",
                fmt(mean_or_none(accuracies)),
                str(sum(1 for run in good if clears(run, benign, asr_bar, drop_bar))),
            ]
        )
    return rows


CELL_HEADER = [
    "dataset",
    "attack",
    "rate",
    "target",
    "usable/trained",
    "mean ASR",
    "sd",
    "min ASR",
    "max ASR",
    "mean CA",
    "seeds clearing",
]


def group_by_cell(runs: dict[str, dict]) -> dict[tuple, list[dict]]:
    groups: dict[tuple, list[dict]] = collections.defaultdict(list)
    for sidecar in runs.values():
        key = (
            sidecar["dataset"],
            sidecar["attack"],
            float(sidecar["poison_rate"]),
            int(sidecar.get("target_label") or 0),
        )
        groups[key].append(sidecar)
    return groups


def epsilon_over_255(sidecar: dict) -> int | None:
    overrides = sidecar.get("attack_config_overrides") or {}
    epsilon = overrides.get("adversarial_epsilon")
    if epsilon is None:
        return None
    value = round(float(epsilon) * 255)
    return value


def lc_pilot_rows(
    runs: dict[str, dict], benign_of: dict[str, float | None], asr_bar: float, drop_bar: float
) -> tuple[list[list[str]], dict[str, int | None]]:
    """1 row per pilot run, and the epsilon the rule picks per dataset."""
    rows = []
    chosen: dict[str, int | None] = {}
    by_dataset: dict[str, list[tuple[int, dict]]] = collections.defaultdict(list)
    for folder, sidecar in sorted(runs.items()):
        epsilon = epsilon_over_255(sidecar)
        if epsilon is None:
            continue
        by_dataset[sidecar["dataset"]].append((epsilon, sidecar))
    for dataset in sorted(by_dataset):
        benign = benign_of[dataset]
        passing = []
        for epsilon, sidecar in sorted(by_dataset[dataset]):
            ok = usable(sidecar, benign)
            passes = ok and clears(sidecar, benign, asr_bar, drop_bar)
            if passes:
                passing.append(epsilon)
            rows.append(
                [
                    dataset,
                    f"{float(sidecar['poison_rate']):g}",
                    str(epsilon),
                    fmt(sidecar.get("asr")),
                    fmt(sidecar.get("clean_accuracy")),
                    fmt((sidecar.get("clean_accuracy") or 0) - (benign or 0), signed=True),
                    "yes" if passes else ("diverged" if not ok else "no"),
                ]
            )
        chosen[dataset] = min(passing) if passing else None
    return rows, chosen


def diverged_gtsrb(checkpoints_dir: str, benign: float | None) -> tuple[int, int]:
    """Diverged and total ViT GTSRB folders with a recorded clean accuracy."""
    total = 0
    diverged = 0
    for folder, sidecar in sidecars(checkpoints_dir, "vit_gtsrb_*").items():
        if sidecar.get("clean_accuracy") is None or "benign" in folder:
            continue
        total += 1
        if not usable(sidecar, benign):
            diverged += 1
    return diverged, total


def main() -> None:
    args = build_parser_with_checkpoints(__doc__).parse_args()
    declaration = load_declaration(args.declaration)
    asr_bar = declaration["asr_bar"]
    drop_bar = declaration["clean_accuracy_drop_bar"]
    benign_of = {
        dataset: benign_accuracy(args.checkpoints_dir, dataset)
        for dataset in (*CLEAN_LABEL_DATASETS, "svhn", "eurosat")
    }
    inputs = [f"{args.checkpoints_dir}/<folder>/args.json", args.declaration]

    # SIG on GTSRB at target class 1, every rate and seed the full stage trained.
    sig_tl1 = sidecars(args.checkpoints_dir, "vit_gtsrb_sig_*_tl1*")
    sig_rows = cell_rows(group_by_cell(sig_tl1), benign_of, asr_bar, drop_bar)
    write_table(
        path=os.path.join(args.paper_dir, "tables", "clean_label_sig_gtsrb.tex"),
        generator=GENERATOR,
        inputs=inputs,
        caption=(
            "SIG on GTSRB at target class 1, where the clean-label cap no longer binds. "
            "Means run over usable seeds, a run being usable when its clean accuracy "
            "is at least half the benign reference, so the diverged runs are counted "
            "in the trained column and excluded from every mean."
        ),
        label="tab:clean-label-sig-gtsrb",
        header=CELL_HEADER,
        rows=sig_rows,
        align="lllrrrrrrrr",
    )

    # The Label-Consistent epsilon pilot, 1 run per (dataset, epsilon) at the cap rate.
    lc_pilot = sidecars(args.checkpoints_dir, "vit_*_lc_*_adv*_pilot")
    lc_rows, chosen = lc_pilot_rows(lc_pilot, benign_of, asr_bar, drop_bar)
    write_table(
        path=os.path.join(args.paper_dir, "tables", "clean_label_lc_epsilon.tex"),
        generator=GENERATOR,
        inputs=inputs,
        caption=(
            "The Label-Consistent epsilon pilot with Turner's adversarial bases, "
            "seed 0 at each dataset's highest reachable clean-label rate. A run "
            "passes when it clears the attack success bar inside the clean-accuracy "
            "drop bar, and the chosen epsilon is the smallest passing one."
        ),
        label="tab:clean-label-lc-epsilon",
        header=["dataset", "rate", "epsilon/255", "ASR", "CA", "dCA", "passes"],
        rows=lc_rows,
        align="lrrrrrl",
    )

    # SVHN and EuroSAT, SIG at every panel rate with the blend control, 3 seeds.
    new_datasets = {
        **sidecars(args.checkpoints_dir, "vit_svhn_*"),
        **sidecars(args.checkpoints_dir, "vit_eurosat_*"),
    }
    new_datasets = {
        folder: sidecar for folder, sidecar in new_datasets.items() if "benign" not in folder
    }
    new_rows = cell_rows(group_by_cell(new_datasets), benign_of, asr_bar, drop_bar)
    write_table(
        path=os.path.join(args.paper_dir, "tables", "clean_label_new_datasets.tex"),
        generator=GENERATOR,
        inputs=inputs,
        caption=(
            "SVHN and EuroSAT as clean-label carriers: SIG at every panel rate with "
            "a dirty-label blend control, 3 training seeds per cell, seed means with "
            "the number of seeds clearing both bars."
        ),
        label="tab:clean-label-new-datasets",
        header=CELL_HEADER,
        rows=new_rows,
        align="lllrrrrrrrr",
    )

    # Multi-target SIG on the primary datasets, target sets of 2 and 3 classes.
    multitarget = sidecars(args.checkpoints_dir, "vit_*_sig_0_01_m*")
    multi_groups: dict[tuple, list[dict]] = collections.defaultdict(list)
    for folder, sidecar in multitarget.items():
        targets = (sidecar.get("attack_config_overrides") or {}).get("num_targets", 1)
        multi_groups[(sidecar["dataset"], int(targets))].append(sidecar)
    for dataset in ("cifar100", "tiny"):
        single = load_args_json(args.checkpoints_dir, f"vit_{dataset}_sig_0_01")
        if single is not None:
            multi_groups[(dataset, 1)].append(single)
    multi_rows = []
    for (dataset, targets), runs in sorted(multi_groups.items()):
        asrs = [run["asr"] for run in runs if run.get("asr") is not None]
        multi_rows.append(
            [
                dataset,
                str(targets),
                str(len(runs)),
                fmt(mean_or_none([float(run.get("realized_poison_rate") or 0) for run in runs])),
                fmt(mean_or_none(asrs)),
                fmt(min(asrs)) if asrs else "--",
                fmt(max(asrs)) if asrs else "--",
                fmt(mean_or_none([run["clean_accuracy"] for run in runs])),
                str(sum(1 for run in runs if (run.get("asr") or 0) >= asr_bar)),
            ]
        )
    write_table(
        path=os.path.join(args.paper_dir, "tables", "clean_label_multitarget.tex"),
        generator=GENERATOR,
        inputs=inputs,
        caption=(
            "Multi-target clean-label SIG on the primary datasets at the lowest panel "
            "rate: the target set is the first m classes, the recorded attack success "
            "counts a prediction landing anywhere in the set, and the single-target "
            "row is the panel's own cell."
        ),
        label="tab:clean-label-multitarget",
        header=["dataset", "targets m", "runs", "realized rate", "mean ASR", "min", "max", "mean CA", "runs clearing"],
        rows=multi_rows,
        align="lrrrrrrrr",
    )

    diverged, total = diverged_gtsrb(args.checkpoints_dir, benign_of["gtsrb"])
    sig_5_percent = [
        run for run in sig_tl1.values() if float(run["poison_rate"]) == 0.05 and usable(run, benign_of["gtsrb"])
    ]
    sig_targets = {int(run.get("target_label") or 0) for run in sig_tl1.values()}
    lc_gtsrb_pilot = [run for run in lc_pilot.values() if run["dataset"] == "gtsrb"]
    macros = {
        "sig_gtsrb_target_class": (str(min(sig_targets)) if sig_targets else "--", "the target class of the GTSRB SIG runs with the target tag"),
        "sig_gtsrb_runs": (str(len(sig_tl1)), "GTSRB SIG runs at the tagged target class"),
        "sig_gtsrb_rates": (str(len({float(run["poison_rate"]) for run in sig_tl1.values()})), "rates among the GTSRB SIG runs at the tagged target class"),
        "sig_gtsrb_seeds": (str(len({int(run.get("seed") or 0) for run in sig_tl1.values()})), "seeds among the GTSRB SIG runs at the tagged target class"),
        "lc_epsilon_budgets": (str(len({epsilon_over_255(run) for run in lc_pilot.values()})), "epsilon budgets in the Label-Consistent pilot"),
        "lc_gtsrb_pilot_runs": (str(len(lc_gtsrb_pilot)), "Label-Consistent pilot runs on GTSRB"),
        "lc_gtsrb_pilot_diverged": (str(sum(1 for run in lc_gtsrb_pilot if not usable(run, benign_of["gtsrb"]))), "Label-Consistent pilot runs on GTSRB that diverged"),
        "multitarget_set_sizes": (" and ".join(str(size) for size in sorted({int(row[1]) for row in multi_rows if int(row[1]) > 1})), "the multi-target set sizes probed"),
        "multitarget_seeds": (str(max((int(row[2]) for row in multi_rows if int(row[1]) > 1), default=0)), "seeds per multi-target cell"),
        "new_dataset_seeds": (str(max((int(row[4].split('/')[1]) for row in new_rows), default=0)), "seeds per SVHN and EuroSAT cell"),
        "gtsrb_diverged_runs": (str(diverged), "ViT GTSRB training runs whose clean accuracy collapsed below half the benign reference"),
        "gtsrb_runs_scored": (str(total), "ViT GTSRB training runs with a recorded clean accuracy"),
        "sig_gtsrb_tl_one_five_percent_mean_asr": (
            fmt(mean_or_none([run["asr"] for run in sig_5_percent])),
            "mean ASR of usable SIG GTSRB runs at target class 1 and the middle panel rate",
        ),
        "sig_gtsrb_tl_one_five_percent_seeds_clearing": (
            str(sum(1 for run in sig_5_percent if clears(run, benign_of["gtsrb"], asr_bar, drop_bar))),
            "usable SIG GTSRB seeds at target class 1 and the middle rate that clear both bars",
        ),
        "sig_gtsrb_tl_one_five_percent_seeds": (str(len(sig_5_percent)), "usable SIG GTSRB seeds at target class 1 and the middle rate"),
        "lc_epsilon_chosen": (
            ", ".join(f"{dataset} {epsilon if epsilon is not None else 'none'}" for dataset, epsilon in sorted(chosen.items())),
            "the Label-Consistent epsilon over 255 the pilot rule picks per dataset, none where no run passes",
        ),
        "lc_datasets_with_epsilon": (
            str(sum(1 for epsilon in chosen.values() if epsilon is not None)),
            "datasets where some Label-Consistent epsilon passes both bars",
        ),
        "new_dataset_cells_clearing_on_mean": (
            str(sum(1 for row in new_rows if row[5] != "--" and float(row[5]) >= asr_bar)),
            "SVHN and EuroSAT cells whose seed-mean ASR clears the bar",
        ),
        "new_dataset_cells": (str(len(new_rows)), "SVHN and EuroSAT cells trained"),
        "multitarget_runs_clearing": (
            str(sum(int(row[-1]) for row in multi_rows if int(row[1]) > 1)),
            "multi-target SIG runs whose set-wide attack success clears the bar",
        ),
        "multitarget_runs": (str(sum(int(row[2]) for row in multi_rows if int(row[1]) > 1)), "multi-target SIG runs trained"),
    }
    write_macros(
        os.path.join(args.paper_dir, "tables", "clean_label.macros.json"), GENERATOR, inputs, macros
    )
    print(
        f"clean label: sig tl1 {len(sig_tl1)} runs, lc pilot {len(lc_pilot)}, new {len(new_datasets)}, "
        f"multitarget {len(multitarget)}, diverged {diverged}/{total}, epsilon {chosen}"
    )


if __name__ == "__main__":
    main()
