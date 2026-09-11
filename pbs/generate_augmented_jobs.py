"""Train the 8 checkpoints a reviewer will ask for, with standard augmentation on.

Every model in the project is fine-tuned with no augmentation beyond
normalization, following the PSBD paper's recipe. A reviewer will call the
models non-standard and ask whether the detector still holds on a model
trained the usual way. This regenerates 8 of the panel's cells with
--augment standard on top of their existing recipe, a CIFAR-100 run and a
Tiny ImageNet run per hard attack (BadNet A2O, Blend, BPP, WaNet) at 5%
poisoning, then sweeps each new checkpoint at the 2 headline placements and
analyzes it, so the comparison against the un-augmented cell is ready as soon
as the job finishes.

Every training argument except --augment and --output is read back from the
existing checkpoints/<folder>/args.json, the same way pbs/generate_seed_jobs.py
rebuilds a checkpoint's recipe, so the augmented run cannot silently diverge
from the run it is compared against. The 2 sweeps read their rate ladders
straight from configs/psbd_basis.json's before_attention_norm_token_mask
(recommended) and post_residual (published) entries, so a ladder edit there is
picked up here without a second copy to keep in sync.

    PYTHONPATH=. python pbs/generate_augmented_jobs.py --dry-run
    PYTHONPATH=. python pbs/generate_augmented_jobs.py
    bash pbs/vit_augmented/submit_all.sh
"""

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from pbs.generate_seed_jobs import MEDIAN_MINUTES, TEMPLATE, pack  # noqa: E402

BASE = REPO

ATTACKS = ("badnet_a2o", "blend", "bpp", "wanet")
DATASETS = ("cifar100", "tiny")
POISON_RATE = 0.05
POISON_RATE_TAG = "0_05"

RECOMMENDED_ID = "before_attention_norm_token_mask"
PUBLISHED_ID = "post_residual"

TRAIN = """python -m cli.train_backdoor \\
    --dataset {dataset} \\
    --attack {attack} \\
    --poison-rate {poison_rate} \\
    --target-label {target_label} \\
    --architecture {architecture} \\
    --epochs {epochs} \\
    --seed {seed} \\
    --augment standard \\
    --output checkpoints/{folder}/attack_result.pt
"""

SWEEP = """python -m cli.sweep \\
    --checkpoint-folder {folder} \\
    --position {position} \\
    --operator {operator} \\
    --rates {rates} \\
    --forward-passes 3 \\
    --skip-existing
"""

ANALYZE = """python -m cli.analyze --checkpoint-folder {folder}
"""


def base_folder_name(dataset: str, attack: str) -> str:
    """The existing, un-augmented folder this run's recipe is read back from."""
    name = f"vit_{dataset}_{attack}_{POISON_RATE_TAG}"
    return name


def augmented_folder_name(dataset: str, attack: str) -> str:
    """The folder the augmented run is written to, tagged so it never overwrites
    the un-augmented checkpoint it is compared against."""
    name = f"{base_folder_name(dataset, attack)}_aug"
    return name


def load_placement_rates(declaration_path: str) -> dict[str, dict]:
    """The 2 headline placements' (position, operator, rates), read from the basis
    declaration so the ladders here can never drift from configs/psbd_basis.json."""
    with open(declaration_path) as handle:
        declared = json.load(handle)

    by_id = {entry["id"]: entry for entry in declared["basis"]}
    placements = {}
    for placement_id in (RECOMMENDED_ID, PUBLISHED_ID):
        entry = by_id[placement_id]
        placements[placement_id] = {
            "position": entry["position"],
            "operator": entry["operator"],
            "rates": entry["rates"],
        }
    return placements


def discover_runs(checkpoints_dir: str) -> list[dict]:
    """The args.json metadata for each of the 8 base checkpoints, tagged with its
    dataset, attack and the folder names the augmented run reads from and writes to."""
    runs = []
    for dataset in DATASETS:
        for attack in ATTACKS:
            base_folder = base_folder_name(dataset, attack)
            args_path = os.path.join(checkpoints_dir, base_folder, "args.json")
            with open(args_path) as handle:
                metadata = json.load(handle)
            metadata["dataset"] = dataset
            metadata["attack"] = attack
            metadata["base_folder"] = base_folder
            metadata["folder"] = augmented_folder_name(dataset, attack)
            runs.append(metadata)
    return runs


def train_command(metadata: dict) -> str:
    """The training command that reruns metadata's recipe with --augment standard.

    seed falls back to 0, the project's unmarked default, because the existing
    checkpoints carry seed: null in their args.json (docs/runs, the provenance
    gap noted for every checkpoint trained before seed was recorded).
    """
    command = TRAIN.format(
        dataset=metadata["dataset"],
        attack=metadata["attack"],
        poison_rate=metadata["poison_rate"],
        target_label=metadata["target_label"],
        architecture=metadata["architecture"],
        epochs=metadata["epochs"],
        seed=metadata.get("seed") or 0,
        folder=metadata["folder"],
    )
    return command


def sweep_and_analyze_commands(folder: str, placements: dict[str, dict]) -> str:
    """The 2 headline sweeps for folder, followed by cli.analyze."""
    lines = []
    for placement in placements.values():
        lines.append(
            SWEEP.format(
                folder=folder,
                position=placement["position"],
                operator=placement["operator"],
                rates=" ".join(str(rate) for rate in placement["rates"]),
            )
        )
    lines.append(ANALYZE.format(folder=folder))
    return "\n".join(lines)


def run_commands(metadata: dict, placements: dict[str, dict]) -> str:
    """The full sequence for 1 run: train, then sweep and analyze the new checkpoint."""
    commands = (
        train_command(metadata)
        + "\n"
        + sweep_and_analyze_commands(metadata["folder"], placements)
    )
    return commands


def write_jobs(
    jobs: list[list[dict]], placements: dict[str, dict], out_dir: str, hours: float
) -> list[str]:
    """1 augmented_NNN.pbs per packed job, plus submit_all.sh, using the same PBS
    header and venv activation as pbs/generate_seed_jobs.py."""
    os.makedirs(out_dir, exist_ok=True)
    log_dir = os.path.join(BASE, "logs", "vit_augmented")
    os.makedirs(log_dir, exist_ok=True)

    written = []
    for index, job in enumerate(jobs, start=1):
        commands = "\n".join(run_commands(metadata, placements) for metadata in job)
        script = TEMPLATE.format(
            base=BASE,
            index=index,
            walltime=f"{int(hours):02d}:00:00",
            commands=commands,
        ).replace("psbd_seed", "vit_augmented")
        path = os.path.join(out_dir, f"augmented_{index:03d}.pbs")
        with open(path, "w") as handle:
            handle.write(script)
        written.append(path)

    submit_path = os.path.join(out_dir, "submit_all.sh")
    with open(submit_path, "w") as handle:
        handle.write("#!/bin/bash\n")
        for path in written:
            handle.write(f"qsub {os.path.join(BASE, path)}\n")
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--declaration", default=os.path.join(BASE, "configs", "psbd_basis.json")
    )
    parser.add_argument("--checkpoints-dir", default=os.path.join(BASE, "checkpoints"))
    parser.add_argument("--out-dir", default=os.path.join(BASE, "pbs", "vit_augmented"))
    parser.add_argument("--hours", type=float, default=12.0)
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    return arguments


def main() -> None:
    args = parse_args()
    placements = load_placement_rates(args.declaration)
    runs = discover_runs(args.checkpoints_dir)

    # The packing budget is training time only, at the same per-dataset minutes
    # pbs/generate_seed_jobs.py measured (84 for CIFAR-100, 167 for Tiny
    # ImageNet), leaving headroom in the 12-hour walltime for the 2 sweeps and
    # the analyze call each run also carries.
    jobs = pack([(metadata, 0) for metadata in runs], args.hours * 60 * 0.9)
    jobs = [[metadata for metadata, _ in job] for job in jobs]

    total_train_minutes = sum(
        MEDIAN_MINUTES.get((m["architecture"], m["dataset"]), 300) for m in runs
    )
    rate_units = sum(len(p["rates"]) for p in placements.values())

    print(f"runs               {len(runs)}")
    for metadata in runs:
        print(f"  {metadata['folder']}")
    print(f"placements/run     {len(placements)} ({rate_units} rate-units)")
    print(f"jobs at {args.hours}h      {len(jobs)}")
    print(f"estimated training GPU hours  {total_train_minutes / 60:.1f}")

    if args.dry_run:
        print("\ndry run, no files written")
        return

    written = write_jobs(jobs, placements, args.out_dir, args.hours)
    print(f"\nwrote {len(written)} job files to {args.out_dir}")
    print(f"submit with        bash {os.path.join(args.out_dir, 'submit_all.sh')}")


if __name__ == "__main__":
    main()
