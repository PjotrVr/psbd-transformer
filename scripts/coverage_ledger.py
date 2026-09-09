"""What has been tested, what has not, and which of it is trustworthy.

Every cross-attack placement ranking this project reported before this script existed was
computed over whichever cells happened to have that placement cached. An audit of the
108-cell ViT all-to-one panel found 0 placements present in every cell, and a bimodal split
where badnet/blend/lf carried a median of 50 configs and wanet/sig/adaptive_blend/bpp/tact
carried 1 or 0. Ranking placements over unequal cell sets ranks the cells, not the
placements, so the ledger has to exist before any table built on it can be read.

The declaration in configs/psbd_basis.json says what SHOULD exist. This script says what
DOES, and writes the difference in a form the job generator consumes directly, so "what we
have not tested" and "what to submit next" are one artifact and cannot drift apart.

Read-only. No GPU, no model loading, safe to run against a live results tree while jobs
are writing into it.

    PYTHONPATH=. python scripts/coverage_ledger.py
"""

import argparse
import collections
import json
import os
from datetime import datetime, timezone

from defences.psbd_metrics import complete_rates

DECLARATION_PATH = "configs/psbd_basis.json"
# Coverage is counted per rate, not per placement. A dir holding two rates cannot be read at
# matched shift ratio, so counting it as present is how a coverage table lies; and because
# already_complete is all-or-nothing per placement, asking for a superset ladder would
# recompute every finished rate. Tracking rates lets the generator request only what is
# missing, which is what made extending four ladders affordable.


def load_declaration(path: str) -> dict:
    with open(path) as handle:
        return json.load(handle)


def is_panel_folder(folder: str, metadata: dict, panel: dict) -> bool:
    """Whether a checkpoint belongs to the panel the declaration describes."""
    if any(token in folder for token in panel["exclude_folder_tokens"]):
        return False
    # Where an attack exists in two variants, only the canonical one is a panel
    # cell. Label-Consistent has both the patch-only runs predating the
    # adversarial bases and the faithful ones carrying them; both clear the ASR
    # bar on CIFAR-10, so without this they would compete for the same slot.
    required = panel.get("canonical_variants", {}).get(metadata.get("attack"))
    if required is not None and required not in folder:
        return False
    if metadata.get("architecture") != panel["architecture"]:
        return False
    if metadata.get("label_mode") not in panel["label_modes"]:
        return False
    return metadata.get("poison_rate") in panel["poison_rates"]


def read_metadata(checkpoints_dir: str, folder: str) -> dict | None:
    path = os.path.join(checkpoints_dir, folder, "args.json")
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)


def malformed_checkpoints(checkpoints_dir: str) -> list[str]:
    """Plain files sitting directly in checkpoints/, which are wrecked runs.

    A generator passing `--output checkpoints/<name>` instead of
    `checkpoints/<name>/attack_result.pt` used to make save_checkpoint write the
    weights as a file named `<name>` and drop its args.json into checkpoints/
    itself. `resolve_checkpoint_path` stops that happening now, but the reason it
    went unnoticed for 4 weeks is that this enumeration skips non-directories, so
    a destroyed run and an unrun one looked identical from here. Report them.
    """
    return sorted(
        entry
        for entry in os.listdir(checkpoints_dir)
        if os.path.isfile(os.path.join(checkpoints_dir, entry))
    )


def panel_cells(checkpoints_dir: str, panel: dict) -> list[dict]:
    """One row per checkpoint the declaration's panel rule selects."""
    cells = []
    for folder in sorted(os.listdir(checkpoints_dir)):
        metadata = read_metadata(checkpoints_dir, folder)
        if metadata is None or not is_panel_folder(folder, metadata, panel):
            continue
        cells.append(
            {
                "folder_name": folder,
                "dataset": metadata.get("dataset"),
                "attack": metadata.get("attack"),
                "label_mode": metadata.get("label_mode"),
                "poison_rate": metadata.get("poison_rate"),
                # The requested rate is a request. A clean-label attack is
                # eligible only on the target class, so it saturates: on Tiny all
                # 4 requested rates resolve to the same 500 images. Carrying both
                # here gives vit_config_tables, vit_top3_tables and
                # vit_config_inventory the applied rate rather than the asked one.
                "realized_poison_rate": metadata.get("realized_poison_rate"),
                "poison_rate_capped": metadata.get("poison_rate_capped"),
                "n_poisoned": metadata.get("n_poisoned"),
                "target_label": metadata.get("target_label"),
                "asr": metadata.get("asr"),
                "asr_source": metadata.get("asr_source"),
                "clean_accuracy": metadata.get("clean_accuracy"),
                "git_commit": metadata.get("git_commit"),
            }
        )
    return cells


def clean_accuracy_of(
    checkpoints_dir: str, results_dir: str, folder: str
) -> float | None:
    """Clean accuracy from either provenance sidecar.

    The benign references predate the args.json convention on two datasets and carry their
    number in results/<folder>/metrics.json instead, so both are read.
    """
    metadata = read_metadata(checkpoints_dir, folder) or {}
    if metadata.get("clean_accuracy") is not None:
        return metadata["clean_accuracy"]
    path = os.path.join(results_dir, folder, "metrics.json")
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle).get("clean_accuracy")


def benign_reference_accuracy(
    checkpoints_dir: str, results_dir: str, reference: dict
) -> dict:
    """Per-dataset benign clean accuracy, the denominator for every dCA in the tables."""
    return {
        dataset: clean_accuracy_of(checkpoints_dir, results_dir, folder)
        for dataset, folder in reference.items()
        if not dataset.startswith("_")
    }


def read_run_sidecar(psbd_dir: str, placement: str) -> dict:
    """Provenance the sweep wrote for this placement, empty when it predates the sidecar."""
    path = os.path.join(psbd_dir, f"run_{placement}.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as handle:
            return json.load(handle)
    except json.JSONDecodeError:
        return {}


def newest_baseline_mtime(psbd_dir: str) -> float | None:
    stamps = [
        os.path.getmtime(os.path.join(psbd_dir, name))
        for name in (
            "baseline_validation.pt",
            "baseline_clean.pt",
            "baseline_backdoor.pt",
        )
        if os.path.exists(os.path.join(psbd_dir, name))
    ]
    return max(stamps) if stamps else None


def stale_baseline(checkpoints_dir: str, folder: str, psbd_dir: str) -> bool:
    """Whether a cached baseline predates the checkpoint it claims to describe.

    load_or_build_baseline reuses any baseline whose row count matches, and a row count
    always matches after a retrain, so a baseline left behind by a previous model turns PSU
    into old-model confidence minus new-model perturbed passes. That defect was found in
    this tree by comparing these two timestamps, which is why it is a standing column and
    not a one-off script.
    """
    baseline = newest_baseline_mtime(psbd_dir)
    weights = os.path.join(checkpoints_dir, folder, "attack_result.pt")
    if baseline is None or not os.path.exists(weights):
        return False
    return baseline < os.path.getmtime(weights)


def placement_rows(checkpoints_dir: str, results_dir: str, folder: str) -> list[dict]:
    """Every cached placement for one cell, with its provenance and rate coverage."""
    psbd_dir = os.path.join(results_dir, folder, "psbd")
    if not os.path.isdir(psbd_dir):
        return []
    baseline_is_stale = stale_baseline(checkpoints_dir, folder, psbd_dir)
    rows = []
    for placement in sorted(os.listdir(psbd_dir)):
        if not os.path.isdir(os.path.join(psbd_dir, placement)):
            continue
        rates = complete_rates(psbd_dir, placement)
        if not rates:
            continue
        sidecar = read_run_sidecar(psbd_dir, placement)
        rows.append(
            {
                "folder_name": folder,
                "placement": placement,
                "position_config": sidecar.get("position_config"),
                "perturbation": sidecar.get("perturbation"),
                "block_range": sidecar.get("block_range"),
                "mask_seed": sidecar.get("mask_seed"),
                "forward_passes": sidecar.get("forward_passes"),
                "git_commit": sidecar.get("git_commit"),
                "complete_rates": rates,
                "n_complete_rates": len(rates),
                "has_provenance": bool(sidecar),
                "stale_baseline": baseline_is_stale,
                "modified_at": datetime.fromtimestamp(
                    os.path.getmtime(os.path.join(psbd_dir, placement)), timezone.utc
                ).isoformat(timespec="seconds"),
            }
        )
    return rows


def cached_rates(rows: list[dict], mask_seed: int) -> dict:
    """Per placement, the rates already on disk at the declaration's seed.

    A stale-baseline dir contributes nothing, so the gap list re-queues it rather than a
    later table quietly reporting a PSU built from a previous model's confidence.
    """
    return {
        row["placement"]: set(row["complete_rates"])
        for row in rows
        if not row["stale_baseline"] and row["mask_seed"] in (None, mask_seed)
    }


def gaps_for_cell(
    cell: dict, cached: dict, basis: list[dict], panel: dict
) -> list[dict]:
    """The basis rates this cell is missing, shaped as sweep arguments."""
    gaps = []
    for entry in basis:
        have = cached.get(entry["id"], set())
        missing = [rate for rate in entry["rates"] if rate not in have]
        if not missing:
            continue
        gaps.append(
            {
                "folder_name": cell["folder_name"],
                "placement": entry["id"],
                "position_config": entry["position_config"],
                "perturbation": entry["perturbation"],
                "block_range": entry["block_range"],
                "mask_seed": panel["mask_seed"],
                "forward_passes": panel["forward_passes"],
                "missing_rates": missing,
                "declared_rates": entry["rates"],
            }
        )
    return gaps


def classify_by_asr(cell: dict, asr_bar: float) -> str:
    if cell["asr"] is None:
        return "unmeasured"
    return "clears" if cell["asr"] >= asr_bar else "below_bar"


def stale_splits(coverage_dir: str) -> set:
    """Cells whose cached split no longer matches what the code builds.

    Produced by scripts/verify_splits.py, which is slow enough to be a separate pass. An
    empty set when it has never run, so the ledger degrades to mtime checking rather than
    claiming an integrity it did not verify.
    """
    path = os.path.join(coverage_dir, "split_integrity.json")
    if not os.path.exists(path):
        return set()
    with open(path) as handle:
        return {
            folder
            for folder, row in json.load(handle).items()
            if row.get("status") == "STALE"
        }


def build_ledger(args, declaration: dict) -> dict:
    panel = declaration["panel"]
    basis = declaration["basis"]
    cells = panel_cells(args.checkpoints_dir, panel)
    stale = stale_splits(args.out_dir)
    benign = benign_reference_accuracy(
        args.checkpoints_dir, args.results_dir, declaration["benign_reference"]
    )

    rows, gaps = [], []
    for cell in cells:
        cell_rows = placement_rows(
            args.checkpoints_dir, args.results_dir, cell["folder_name"]
        )
        cached = cached_rates(cell_rows, panel["mask_seed"])
        cell["n_configs_cached"] = len(cell_rows)
        cell["n_basis_covered"] = sum(
            1
            for entry in basis
            if not set(entry["rates"]) - cached.get(entry["id"], set())
        )
        cell["n_basis_partial"] = sum(
            1
            for entry in basis
            if cached.get(entry["id"]) and set(entry["rates"]) - cached[entry["id"]]
        )
        cell["asr_class"] = classify_by_asr(cell, declaration["asr_bar"])
        cell["stale_split"] = cell["folder_name"] in stale
        reference = benign.get(cell["dataset"])
        cell["clean_accuracy_benign"] = reference
        cell["clean_accuracy_drop"] = (
            None
            if reference is None or cell["clean_accuracy"] is None
            else cell["clean_accuracy"] - reference
        )
        rows.extend(cell_rows)
        gaps.extend(gaps_for_cell(cell, cached, basis, panel))

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "declaration": args.declaration,
        "asr_bar": declaration["asr_bar"],
        "clean_accuracy_drop_bar": declaration["clean_accuracy_drop_bar"],
        "basis_size": len(basis),
        "benign_reference_accuracy": benign,
        "cells": cells,
        "placements": rows,
        "gaps": gaps,
    }


def resolve_one_per_attack(cells: list[dict]) -> dict:
    """One cell per attack, preferring the one that implanted.

    Two checkpoints can share (dataset, attack, poison_rate) and differ only by a
    folder tag: GTSRB clean-label at target class 0 against target class 1, for
    instance, where only the second reaches the requested rate. Keying a dict by
    attack alone let sort order decide, silently. Preferring the cell that clears
    the ASR bar makes the choice explicit, and an ambiguous pair raises rather
    than picking one.
    """
    grouped: dict[str, list[dict]] = collections.defaultdict(list)
    for cell in cells:
        grouped[cell["attack"]].append(cell)
    resolved = {}
    for attack, candidates in grouped.items():
        if len(candidates) == 1:
            resolved[attack] = candidates[0]
            continue
        clearing = [cell for cell in candidates if cell.get("asr_class") == "clears"]
        if len(clearing) == 1:
            resolved[attack] = clearing[0]
            continue
        raise ValueError(
            f"{len(candidates)} cells share ({candidates[0]['dataset']}, {attack}, "
            f"{candidates[0]['poison_rate']}) and "
            f"{len(clearing)} of them clear the ASR bar, so the panel slot is "
            f"ambiguous: {sorted(cell['folder_name'] for cell in candidates)}. "
            "Give the superseded variant a folder token in exclude_folder_tokens, "
            "or name the keeper in canonical_variants."
        )
    return resolved


def pivot_configs(cells: list[dict]) -> str:
    """Basis coverage as dataset rows by attack columns, one table per poison rate."""
    attacks = sorted({cell["attack"] for cell in cells})
    datasets = sorted({cell["dataset"] for cell in cells})
    lines = []
    for rate in sorted({cell["poison_rate"] for cell in cells}):
        lines.append(f"\n### poison rate {rate:.0%}\n")
        lines.append("| dataset | " + " | ".join(attacks) + " |")
        lines.append("|---|" + "---|" * len(attacks))
        for dataset in datasets:
            cellsat = resolve_one_per_attack(
                [
                    cell
                    for cell in cells
                    if cell["dataset"] == dataset and cell["poison_rate"] == rate
                ]
            )
            values = []
            for attack in attacks:
                cell = cellsat.get(attack)
                if cell is None:
                    values.append("--")
                else:
                    values.append(f"{cell['n_basis_covered']}")
            lines.append(f"| {dataset} | " + " | ".join(values) + " |")
    return "\n".join(lines)


def pivot_presence(rows: list[dict], cells: list[dict], basis: list[dict]) -> str:
    """How many cells carry each basis placement, the number the audit turned on."""
    total = len(cells)
    counts = collections.Counter(row["placement"] for row in rows)
    lines = [
        "| basis placement | family | cells | coverage |",
        "|---|---|---:|---:|",
    ]
    for entry in basis:
        seen = counts.get(entry["id"], 0)
        lines.append(
            f"| `{entry['id']}` | {entry['family']} | {seen}/{total} | {seen / total:.0%} |"
        )
    return "\n".join(lines)


def render_markdown(ledger: dict, declaration: dict) -> str:
    cells = ledger["cells"]
    by_class = collections.Counter(cell["asr_class"] for cell in cells)
    complete = sum(
        1 for cell in cells if cell["n_basis_covered"] == ledger["basis_size"]
    )
    stale = sum(1 for row in ledger["placements"] if row["stale_baseline"])
    unprovenanced = sum(1 for row in ledger["placements"] if not row["has_provenance"])

    parts = [
        "# PSBD-ViT coverage ledger",
        "",
        "Generated by `scripts/coverage_ledger.py` from `configs/psbd_basis.json`.",
        "Do not edit by hand.",
        "",
        f"- generated: {ledger['generated_at']}",
        f"- panel cells: **{len(cells)}**",
        f"- basis size: **{ledger['basis_size']}** placements per cell",
        f"- cells at full basis coverage: **{complete}/{len(cells)}**",
        f"- missing (cell, placement) slots: **{len(ledger['gaps'])}**, carrying "
        f"**{sum(len(gap['missing_rates']) for gap in ledger['gaps'])}** rate-units",
        f"- ASR bar {ledger['asr_bar']}: "
        f"{by_class['clears']} clear, {by_class['below_bar']} below, "
        f"{by_class['unmeasured']} never measured",
        f"- integrity: {stale} placements on a stale baseline, "
        f"{unprovenanced} without a run sidecar, "
        f"{sum(1 for cell in cells if cell.get('stale_split'))} cells on a stale split",
        "",
        "## Benign reference clean accuracy",
        "",
        "| dataset | benign ViT clean accuracy |",
        "|---|---:|",
    ]
    for dataset, value in sorted(ledger["benign_reference_accuracy"].items()):
        parts.append(f"| {dataset} | {'--' if value is None else f'{value:.3f}'} |")

    parts += [
        "",
        "## Basis placements covered, per cell",
        "",
        f"Each entry is out of {ledger['basis_size']}. A cell below that number cannot be",
        "compared against the rest of its row on equal footing.",
        pivot_configs(cells),
        "",
        "## Which basis placements exist, panel-wide",
        "",
        pivot_presence(ledger["placements"], cells, declaration["basis"]),
        "",
        "## Attack strength",
        "",
        f"A cell is a valid attack at ASR >= {ledger['asr_bar']} and "
        f"dCA >= {ledger['clean_accuracy_drop_bar']}.",
        "",
        "| cell | ASR | CA | dCA | basis | verdict |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for cell in sorted(
        cells, key=lambda c: (c["attack"], c["dataset"], c["poison_rate"] or 0)
    ):
        asr = "--" if cell["asr"] is None else f"{cell['asr']:.3f}"
        accuracy = (
            "--" if cell["clean_accuracy"] is None else f"{cell['clean_accuracy']:.3f}"
        )
        drop = (
            "--"
            if cell["clean_accuracy_drop"] is None
            else f"{cell['clean_accuracy_drop']:+.3f}"
        )
        verdict = cell["asr_class"]
        if verdict == "clears" and cell["clean_accuracy_drop"] is not None:
            if cell["clean_accuracy_drop"] < ledger["clean_accuracy_drop_bar"]:
                verdict = "clears ASR, FAILS dCA"
        parts.append(
            f"| `{cell['folder_name']}` | {asr} | {accuracy} | {drop} | "
            f"{cell['n_basis_covered']}/{ledger['basis_size']} | {verdict} |"
        )
    return "\n".join(parts) + "\n"


def write_artifacts(out_dir: str, ledger: dict, declaration: dict) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "coverage.json"), "w") as handle:
        json.dump(ledger, handle, indent=2)
    with open(os.path.join(out_dir, "gaps.json"), "w") as handle:
        json.dump(
            {
                "generated_at": ledger["generated_at"],
                "asr_bar": ledger["asr_bar"],
                "count": len(ledger["gaps"]),
                "gaps": ledger["gaps"],
            },
            handle,
            indent=2,
        )
    with open(os.path.join(out_dir, "COVERAGE.md"), "w") as handle:
        handle.write(render_markdown(ledger, declaration))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declaration", default=DECLARATION_PATH)
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out-dir", default="results/coverage")
    parser.add_argument(
        "--architecture",
        default=None,
        help="override the declaration's panel architecture, for a Swin ledger",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    declaration = load_declaration(args.declaration)
    if args.architecture:
        # The basis and the benign references are declared for ViT; a Swin ledger reuses the
        # panel RULE (label modes, rates, excluded tokens) and retargets the architecture and
        # the per-dataset benign reference, so dCA still compares like with like.
        declaration = {
            **declaration,
            "panel": {**declaration["panel"], "architecture": args.architecture},
            "benign_reference": {
                key: value.replace("vit_", f"{args.architecture}_", 1)
                for key, value in declaration["benign_reference"].items()
                if not key.startswith("_")
            },
        }
    ledger = build_ledger(args, declaration)
    write_artifacts(args.out_dir, ledger, declaration)

    cells = ledger["cells"]
    complete = sum(
        1 for cell in cells if cell["n_basis_covered"] == ledger["basis_size"]
    )
    by_class = collections.Counter(cell["asr_class"] for cell in cells)
    print(f"[ok] {args.out_dir}/")
    print(f"     panel cells        {len(cells)}")
    print(f"     basis size         {ledger['basis_size']}")
    print(f"     full coverage      {complete}/{len(cells)}")
    print(f"     missing slots      {len(ledger['gaps'])}")
    print(
        f"     missing rate-units {sum(len(g['missing_rates']) for g in ledger['gaps'])}"
    )
    print(
        f"     ASR bar {ledger['asr_bar']}       {by_class['clears']} clear, "
        f"{by_class['below_bar']} below, {by_class['unmeasured']} unmeasured"
    )

    # Loud, and last, so it is the line left on screen. A wrecked run is otherwise
    # indistinguishable from one that was never launched.
    wrecked = malformed_checkpoints(args.checkpoints_dir)
    if wrecked:
        print(
            f"\n[ERROR] {len(wrecked)} plain files in {args.checkpoints_dir}/ where a "
            "folder was expected. These are training runs whose weights were written "
            "to the wrong path and which no tool here can read:"
        )
        for entry in wrecked:
            print(f"          {entry}")
        print("        Recover with scratch/recover_orphaned_checkpoints.py")


if __name__ == "__main__":
    main()
