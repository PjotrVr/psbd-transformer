"""The confidence null: does the perturbation earn anything over the unperturbed softmax?

PSU subtracts a perturbed confidence from an unperturbed one, so the objection is
that it measures the unperturbed confidence and the passes are decoration. The
record at the results root scores PSU and max-softmax confidence on the same
cached tensors per checkpoint. This keeps the ViT non-SAM rows and reports both,
and the share of rows where PSU wins.

    PYTHONPATH=. python scripts/paper/mech_confidence_null.py \\
        --results-dir /path/to/results --paper-dir paper
"""

import os
import sys

sys.path.insert(0, os.getcwd())

from scripts.paper._common import (  # noqa: E402
    build_parser,
    dataset_label,
    fmt,
    is_panel_folder,
    load_json,
    mean_or_none,
    write_macros,
    write_table,
)

GENERATOR = "scripts/paper/mech_confidence_null.py"
RECORD = "psu_vs_confidence.json"


def main() -> None:
    args = build_parser(__doc__).parse_args()
    path = os.path.join(args.results_dir, RECORD)
    record = load_json(path)
    if record is None:
        raise SystemExit(f"{path} is missing")
    rows = {
        folder: values
        for folder, values in record["rows"].items()
        if folder.startswith("vit_")
        and is_panel_folder(folder)
        and "benign" not in folder
    }
    benign = {
        folder: values
        for folder, values in record["rows"].items()
        if folder.startswith("vit_") and "benign" in folder and is_panel_folder(folder)
    }

    by_dataset: dict[str, list[dict]] = {}
    for folder, values in rows.items():
        by_dataset.setdefault(folder.split("_")[1], []).append(values)
    table_rows = []
    for dataset, values in sorted(by_dataset.items()):
        table_rows.append(
            [
                dataset_label(dataset),
                str(len(values)),
                fmt(mean_or_none([value["psu"] for value in values])),
                fmt(mean_or_none([value["confidence"] for value in values])),
                fmt(
                    mean_or_none(
                        [value["psu"] - value["confidence"] for value in values]
                    ),
                    signed=True,
                ),
                str(sum(1 for value in values if value["psu"] > value["confidence"])),
            ]
        )
    all_values = list(rows.values())
    table_rows.append(
        [
            "all",
            str(len(all_values)),
            fmt(mean_or_none([value["psu"] for value in all_values])),
            fmt(mean_or_none([value["confidence"] for value in all_values])),
            fmt(
                mean_or_none(
                    [value["psu"] - value["confidence"] for value in all_values]
                ),
                signed=True,
            ),
            str(sum(1 for value in all_values if value["psu"] > value["confidence"])),
        ]
    )
    write_table(
        path=os.path.join(args.paper_dir, "tables", "confidence_null.tex"),
        generator=GENERATOR,
        inputs=[path],
        caption=(
            f"PSU against the confidence null at the `{record['placement']}` placement: "
            "mean AUROC of PSU and of the unperturbed max-softmax confidence on the "
            "same cached tensors, over every ViT backdoored checkpoint in the record "
            "outside the SAM, evasion and seed sets, and the count of checkpoints where PSU wins."
        ),
        label="tab:confidence-null",
        header=[
            "dataset",
            "n",
            "PSU AUROC",
            "confidence AUROC",
            "PSU minus confidence",
            "PSU wins",
        ],
        rows=table_rows,
        align="lrrrrr",
    )
    macros = {
        "confidence_null_checkpoints": (
            str(len(all_values)),
            "ViT backdoored checkpoints in the confidence-null record",
        ),
        "confidence_null_psu_auroc": (
            fmt(mean_or_none([value["psu"] for value in all_values])),
            "mean PSU AUROC in the confidence-null record",
        ),
        "confidence_null_auroc": (
            fmt(mean_or_none([value["confidence"] for value in all_values])),
            "mean max-softmax confidence AUROC in the confidence-null record",
        ),
        "confidence_null_psu_wins": (
            str(sum(1 for value in all_values if value["psu"] > value["confidence"])),
            "checkpoints where PSU beats the confidence null",
        ),
        "confidence_null_benign_psu": (
            fmt(mean_or_none([value["psu"] for value in benign.values()])),
            "mean PSU AUROC on the benign references in the confidence-null record",
        ),
    }
    write_macros(
        os.path.join(args.paper_dir, "tables", "confidence_null.macros.json"),
        GENERATOR,
        [path],
        macros,
    )
    print(
        f"confidence null: n={len(all_values)}, psu {macros['confidence_null_psu_auroc'][0]} vs confidence {macros['confidence_null_auroc'][0]}"
    )


if __name__ == "__main__":
    main()
