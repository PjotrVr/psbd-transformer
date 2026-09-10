"""Jobs for the 2 datasets added to answer the clean-label rate question.

A clean-label attack poisons only its target class, so the highest rate it can
reach is |target class| / |train set|. For a balanced K-class dataset that is
exactly 1/K, which caps CIFAR-100 at 1% and Tiny at 0.5%. SVHN and EuroSAT are
the panel's 2 datasets that clear 10%, and this generator trains them.

Three stages, run in order:

    PYTHONPATH=. python pbs/generate_newdataset_jobs.py --stage benign
    PYTHONPATH=. python pbs/generate_newdataset_jobs.py --stage probe
    PYTHONPATH=. python pbs/generate_newdataset_jobs.py --stage clean_label

`benign` trains the clean reference each dataset needs for its clean-accuracy
drop, and which Label-Consistent later needs as the surrogate its adversarial
bases are generated against. `probe` runs SIG across every reachable rate plus a
single Blend control, which is the fastest answer to whether SIG implants at all
here. `clean_label` adds Label-Consistent once the benign checkpoints exist.

Every stage takes --seeds. Seed 0 keeps the bare folder name and a replicate
carries _seed_N, the tag the rest of the panel uses.

    PYTHONPATH=. python pbs/generate_newdataset_jobs.py --stage probe --seeds 1 2 --batch vit_newdata_probe_seeds
"""

import argparse
import collections
import os

import torchvision.transforms.v2 as transforms_v2

from data.loading import extract_labels, load_clean_datasets

PROJECT_ROOT = "/lustre/home/pstika/projects/PSBD-ViT"

# Scaled from the measured CIFAR-10 median of 167 minutes at 15 epochs, by
# training set size. SVHN is 73257 images and EuroSAT 21600.
TRAIN_MINUTES = {"svhn": 245, "eurosat": 75}

DATASETS = ("svhn", "eurosat")
CANDIDATE_RATES = (0.01, 0.05, 0.10)
CONTROL_RATE = 0.10

JOB_TEMPLATE = """#!/bin/bash
#PBS -q gpu
#PBS -l select=1:ngpus=1:ncpus=8:mem=64gb
#PBS -l walltime={walltime}
#PBS -N {name}
#PBS -o {root}/logs/{batch}/{name}.log
#PBS -j oe

export http_proxy="http://10.150.1.1:3128"
export https_proxy="http://10.150.1.1:3128"

echo "Job ID:  $PBS_JOBID"; echo "Node: $(hostname)"; echo "Started: $(date)"
cd {root}
source .venv/bin/activate

{body}
echo "Finished: $(date)"
exit 0
"""

BENIGN_CALL = """echo "=== benign {dataset} ==="
python -m cli.train_benign \\
    --datasets {dataset} \\
    --architecture vit \\
    --epochs 15 \\
    --seed 0
"""

TRAIN_CALL = """echo "=== {folder} ==="
python -m cli.train_backdoor \\
    --dataset {dataset} \\
    --attack {attack} \\
    --poison-rate {rate} \\
    --target-label {target_label} \\
    --architecture vit \\
    --epochs 15 \\
    --seed {seed} \\
    --output checkpoints/{folder}/attack_result.pt
"""


def rate_tag(rate):
    return f"{rate:g}".replace(".", "_")


def largest_class(dataset, raw_data_dir):
    """The class an attacker would pick, and how much of the training set it is.

    An attacker chooses their own target, so the sensible choice is the class with
    the most images, which is what sets the clean-label ceiling.
    """
    transform = transforms_v2.Compose([transforms_v2.ToTensor()])
    train, _ = load_clean_datasets(dataset, transform, raw_data_dir)
    counts = collections.Counter(extract_labels(train))
    label, size = counts.most_common(1)[0]
    return label, size, sum(counts.values())


def reachable_rates(pool, total):
    return [rate for rate in CANDIDATE_RATES if round(rate * total) <= pool]


def folder_name(dataset, attack, rate, target_label, seed=0):
    name = f"vit_{dataset}_{attack}_{rate_tag(rate)}"
    if target_label != 0:
        name += f"_tl{target_label}"
    if seed:
        name += f"_seed_{seed}"
    return name


def runs_for_stage(stage, datasets, raw_data_dir, seeds=(0,)):
    """Every training call this stage should emit, with its predicted cost."""
    runs = []
    for dataset in datasets:
        target_label, pool, total = largest_class(dataset, raw_data_dir)
        rates = reachable_rates(pool, total)
        print(
            f"  {dataset}: class {target_label} holds {pool} of {total} "
            f"({pool / total:.2%}), reachable {[f'{r:.0%}' for r in rates]}"
        )
        minutes = TRAIN_MINUTES[dataset]

        if stage == "benign":
            runs.append((minutes, BENIGN_CALL.format(dataset=dataset)))
            continue

        attacks = ["sig"] if stage == "probe" else ["lc"]
        for seed in seeds:
            for attack in attacks:
                for rate in rates:
                    runs.append(
                        (
                            minutes,
                            TRAIN_CALL.format(
                                folder=folder_name(
                                    dataset, attack, rate, target_label, seed
                                ),
                                dataset=dataset,
                                attack=attack,
                                rate=rate,
                                target_label=target_label,
                                seed=seed,
                            ),
                        )
                    )

            # One dirty-label control, so the new dataset can be compared against
            # the existing panel rather than only against itself.
            if stage == "probe":
                runs.append(
                    (
                        minutes,
                        TRAIN_CALL.format(
                            folder=folder_name(dataset, "blend", CONTROL_RATE, 0, seed),
                            dataset=dataset,
                            attack="blend",
                            rate=CONTROL_RATE,
                            target_label=0,
                            seed=seed,
                        ),
                    )
                )
    return runs


def pack(runs, budget):
    """Greedy bin-pack by predicted minutes, never splitting a run."""
    bundles, current, spent = [], [], 0
    for minutes, body in runs:
        if current and spent + minutes > budget:
            bundles.append((current, spent))
            current, spent = [], 0
        current.append(body)
        spent += minutes
    if current:
        bundles.append((current, spent))
    return bundles


def write_jobs(batch, bundles):
    job_dir = os.path.join("pbs", batch)
    os.makedirs(job_dir, exist_ok=True)
    os.makedirs(os.path.join("logs", batch), exist_ok=True)
    submit_lines = []
    for index, (bodies, spent) in enumerate(bundles, start=1):
        name = f"{batch}_{index}"
        # A job killed at the wall loses every run it had not yet written, so the
        # request carries 2x headroom over the prediction.
        hours = min(48, max(6, int(spent / 60 * 2.0) + 2))
        with open(os.path.join(job_dir, f"{name}.pbs"), "w") as handle:
            handle.write(
                JOB_TEMPLATE.format(
                    walltime=f"{hours}:00:00",
                    name=name,
                    root=PROJECT_ROOT,
                    batch=batch,
                    body="\n".join(bodies),
                )
            )
        submit_lines.append(f"qsub {PROJECT_ROOT}/{job_dir}/{name}.pbs")
    submit = os.path.join(job_dir, "submit_all.sh")
    with open(submit, "w") as handle:
        handle.write("#!/bin/bash\n" + "\n".join(submit_lines) + "\n")
    os.chmod(submit, 0o755)
    total = sum(spent for _, spent in bundles)
    print(f"\n[ok] pbs/{batch}/  {len(bundles)} jobs, {total} predicted minutes")
    print(f"submit with: bash pbs/{batch}/submit_all.sh")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", required=True, choices=("benign", "probe", "clean_label")
    )
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--minutes-per-job", type=int, default=400)
    parser.add_argument("--raw-data-dir", default="raw_data")
    parser.add_argument("--batch", default=None)
    args = parser.parse_args()

    print(f"reachable rates for stage {args.stage}:")
    runs = runs_for_stage(args.stage, args.datasets, args.raw_data_dir, args.seeds)
    print(f"\n  {len(runs)} runs")
    write_jobs(
        args.batch or f"vit_newdata_{args.stage}",
        pack(runs, args.minutes_per_job),
    )


if __name__ == "__main__":
    main()
