"""The numbers the attack appendix states: each attack's settings and how well it implanted.

Settings come from attacks.default_config, except the cover rate, which the
training command overrides per run and which is read from the checkpoints'
args.json sidecars instead, because those record what each model was actually
trained with. Implant quality is the range of attack success over the panel's
cells on the 4 main datasets, excluding diverged runs, whose attack success
means nothing. The clean-label rate caps are counted off the training labels
themselves, since a clean-label attack can poison only its target class.

    PYTHONPATH=. python scripts/paper/app_attacks.py \\
        --results-dir /path/to/results --paper-dir paper --raw-data-dir raw_data
"""

import collections
import os
import sys

from torchvision.transforms import v2 as transforms_v2

sys.path.insert(0, os.getcwd())

from attacks import default_config  # noqa: E402
from attacks.adversarial import DEFAULT_PGD_STEPS, STEP_SIZE_FACTOR  # noqa: E402
from data.loading import extract_labels, load_clean_datasets  # noqa: E402
from scripts.paper._common import (  # noqa: E402
    build_parser_with_checkpoints,
    fmt,
    load_args_json,
    load_coverage,
    macro_name,
    word_list,
    write_macros,
)

GENERATOR = "scripts/paper/app_attacks.py"
MAIN_DATASETS = ("cifar10", "cifar100", "gtsrb", "tiny")
LOW_PANEL_RATE = 0.01
# The target classes the clean-label runs use, GTSRB at both because class 0 is
# too small to reach a usable rate and the runs moved to class 1.
CLEAN_LABEL_TARGETS = (
    ("cifar10", 0),
    ("cifar100", 0),
    ("tiny", 0),
    ("gtsrb", 0),
    ("gtsrb", 1),
)
# Config fields the appendix quotes, per attack, with the words for each.
SETTINGS = {
    "blend": {"alpha": "blend ratio"},
    "sig": {
        "amplitude": "signal amplitude in 0-to-1 pixel units",
        "frequency": "cycles across the width",
    },
    "wanet": {"control_grid_size": "control grid side", "strength": "warp strength"},
    "lf": {"strength": "pattern strength", "cutoff": "low-pass radius"},
    "bpp": {"bit_depth": "bits per channel"},
    "adaptive_blend": {
        "alpha": "blend ratio",
        "cells": "grid side",
        "train_cell_fraction": "share of cells revealed in training",
    },
    "tact": {"patch_size": "trigger side"},
}


def setting_macros() -> dict[str, tuple[str, str]]:
    """Every quoted setting, read from the attack's own default config."""
    macros = {}
    for attack, fields in SETTINGS.items():
        config = default_config(attack)
        for field, words in fields.items():
            value = getattr(config, field)
            text = f"{value:g}" if isinstance(value, float) else str(value)
            macros[f"attack_{attack}_{field}"] = (text, f"{attack} {words}")
    return macros


def implant_macros(coverage: dict) -> dict[str, tuple[str, str]]:
    """Per attack, the lowest and highest attack success over its main-dataset cells."""
    by_attack = collections.defaultdict(list)
    for cell in coverage["cells"]:
        if cell["dataset"] not in MAIN_DATASETS or cell.get("diverged"):
            continue
        if cell.get("asr") is None:
            continue
        by_attack[cell["attack"]].append(cell["asr"])
    macros = {}
    for attack, values in sorted(by_attack.items()):
        macros[f"attack_{attack}_asr_min"] = (
            fmt(min(values)),
            f"lowest attack success of {attack} over its {len(values)} main-dataset cells",
        )
        macros[f"attack_{attack}_asr_max"] = (
            fmt(max(values)),
            f"highest attack success of {attack} over its {len(values)} main-dataset cells",
        )
    return macros


def failure_macros(coverage: dict) -> dict[str, tuple[str, str]]:
    """Per attack, where it fails: its range at the lowest rate and on datasets it never clears."""
    by_attack = collections.defaultdict(list)
    for cell in coverage["cells"]:
        usable = cell["dataset"] in MAIN_DATASETS and not cell.get("diverged")
        if usable and cell.get("asr") is not None:
            by_attack[cell["attack"]].append(cell)
    macros = {}
    for attack, cells in sorted(by_attack.items()):
        low_rate = [
            cell["asr"] for cell in cells if cell["poison_rate"] == LOW_PANEL_RATE
        ]
        if low_rate:
            macros[f"attack_{attack}_asr_low_rate_min"] = (
                fmt(min(low_rate)),
                f"lowest attack success of {attack} at the lowest panel rate",
            )
            macros[f"attack_{attack}_asr_low_rate_max"] = (
                fmt(max(low_rate)),
                f"highest attack success of {attack} at the lowest panel rate",
            )
        clearing = {cell["dataset"] for cell in cells if cell["asr_class"] == "clears"}
        never = [cell for cell in cells if cell["dataset"] not in clearing]
        if not never:
            continue
        datasets = sorted({cell["dataset"] for cell in never})
        macros[f"attack_{attack}_asr_never_clearing_min"] = (
            fmt(min(cell["asr"] for cell in never)),
            f"lowest attack success of {attack} on {word_list(datasets)}, "
            "the datasets where no rate clears",
        )
        macros[f"attack_{attack}_asr_never_clearing_max"] = (
            fmt(max(cell["asr"] for cell in never)),
            f"highest attack success of {attack} on {word_list(datasets)}, "
            "the datasets where no rate clears",
        )
    return macros


def rate_cap_macros(raw_data_dir: str) -> dict[str, tuple[str, str]]:
    """Each clean-label ceiling, the target class's share of the training set in percent."""
    transform = transforms_v2.Compose([transforms_v2.ToImage()])
    labels_of = {}
    macros = {}
    for dataset, target in CLEAN_LABEL_TARGETS:
        if dataset not in labels_of:
            train, _ = load_clean_datasets(dataset, transform, raw_data_dir)
            labels_of[dataset] = extract_labels(train)
        labels = labels_of[dataset]
        share = labels.count(target) / len(labels)
        macros[f"clean_label_cap_{dataset}_class_{target}"] = (
            f"{round(100 * share, 2):g}\\%",
            f"clean-label rate ceiling on {dataset} at target class {target}, "
            f"{labels.count(target)} of {len(labels)} training images",
        )
    return macros


def cover_macros(checkpoints_dir: str, coverage: dict) -> dict[str, tuple[str, str]]:
    """Each attack's cover rate as the models were trained, as a ratio or a constant.

    A ratio is reported when every cell's cover rate is the same multiple of its
    poison rate, a constant when every cell used the same cover rate, and
    nothing when neither holds, so the prose cannot state a rule the runs broke.
    """
    by_attack = collections.defaultdict(set)
    for cell in coverage["cells"]:
        sidecar = load_args_json(checkpoints_dir, cell["folder_name"]) or {}
        cover = sidecar.get("cover_rate")
        rate = sidecar.get("poison_rate")
        if cover is None or not rate:
            continue
        by_attack[cell["attack"]].add((rate, cover))
    macros = {}
    for attack, pairs in sorted(by_attack.items()):
        ratios = {round(cover / rate, 6) for rate, cover in pairs}
        covers = {cover for _, cover in pairs}
        if len(ratios) == 1 and next(iter(ratios)) > 0:
            macros[f"attack_{attack}_cover_ratio"] = (
                f"{next(iter(ratios)):g}",
                f"{attack} cover rate as a multiple of its poison rate, the same on every cell",
            )
        elif len(covers) == 1:
            macros[f"attack_{attack}_cover_rate"] = (
                f"{next(iter(covers)):g}",
                f"{attack} cover rate, the same on every cell whatever the poison rate",
            )
    return macros


def main() -> None:
    parser = build_parser_with_checkpoints(__doc__)
    parser.add_argument("--raw-data-dir", default="raw_data")
    args = parser.parse_args()
    coverage = load_coverage(args.results_dir)
    macros = {
        "attack_lc_pgd_steps": (
            str(DEFAULT_PGD_STEPS),
            "PGD steps behind the Label-Consistent bases",
        ),
        "attack_lc_pgd_step_factor": (
            f"{STEP_SIZE_FACTOR:g}",
            "PGD step size as this multiple of epsilon over the step count",
        ),
    }
    macros.update(setting_macros())
    macros.update(implant_macros(coverage))
    macros.update(failure_macros(coverage))
    macros.update(rate_cap_macros(args.raw_data_dir))
    macros.update(cover_macros(args.checkpoints_dir, coverage))
    write_macros(
        os.path.join(args.paper_dir, "tables", "attacks.macros.json"),
        GENERATOR,
        [
            "attacks.default_config",
            f"{args.results_dir}/coverage/coverage.json",
            f"{args.checkpoints_dir}/<folder>/args.json",
            f"{args.raw_data_dir}/<dataset> training labels",
        ],
        macros,
    )
    print(f"attacks: {len(macros)} macros")
    for key, (value, _) in sorted(macros.items()):
        print(f"  {macro_name(key):40s} {value}")


if __name__ == "__main__":
    main()
