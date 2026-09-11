"""Training jobs for the clean-label fixes: adversarial Label-Consistent, and GTSRB at a target class that can actually be poisoned.

Clean-label attacks may only poison images that already carry the target label, so
the highest reachable poison rate is |target class| / |train set|. This generator
computes that from the data and emits ONLY reachable rates, which is the whole
point: 188 of the 315 clean-label folders on disk were trained at a nominal rate
the dataset silently clamped, and three of Tiny's four "rates" are the same run.

Three stages, in order:

    python pbs/generate_cleanlabel_jobs.py --stage bases
    python pbs/generate_cleanlabel_jobs.py --stage pilot
    python pbs/generate_cleanlabel_jobs.py --stage full --epsilon cifar10=16 gtsrb=8

`bases` builds the adversarially perturbed base images Label-Consistent needs.
`pilot` trains one seed per (dataset, epsilon) at each dataset's highest reachable
rate, so the smallest epsilon clearing ASR 0.85 without costing more than 0.05
clean accuracy can be picked per dataset. `full` then runs that choice at seeds
0 to 4.

Basis sweeps are NOT emitted here. Run pbs/generate_basis_jobs.py once these land;
it picks up any cell that clears the ASR bar, and pairing the two would spend a
sweep on a training run that failed.
"""

import argparse
import os

import torchvision.transforms.v2 as transforms_v2

import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from data.loading import extract_labels, load_clean_datasets  # noqa: E402

PROJECT_ROOT = "/lustre/home/pstika/projects/PSBD-ViT"

# Median observed training minutes at 15 epochs, from args.json timestamps.
TRAIN_MINUTES = {
    "vit": {"gtsrb": 90, "cifar10": 167, "cifar100": 167, "tiny": 332},
    "swin": {"gtsrb": 70, "cifar10": 129, "cifar100": 129, "tiny": 258},
}
# Generating bases is one PGD pass over the target class, far cheaper than training.
BASES_MINUTES = {"gtsrb": 25, "cifar10": 60, "cifar100": 15, "tiny": 20}

CANDIDATE_RATES = (0.005, 0.01, 0.05, 0.1)
EPSILONS_OVER_255 = (8, 16, 32)
DATASETS = ("cifar10", "cifar100", "gtsrb", "tiny")

# GTSRB class 0 holds 150 of 26,640 images, so clean-label there caps at 0.56% and
# every rate above it silently collapses to the same run. Classes 1 and 2 hold
# 1,500, which lifts the cap to 5.63%. Nothing in the threat model makes an
# attacker pick the rarest class. CIFAR-100 and Tiny are class-uniform, so no
# choice of target helps them and they stay at class 0.
CLEAN_LABEL_TARGETS = {"gtsrb": 1}

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

TRAIN_CALL = """echo "=== {folder} ==="
python -m cli.train_backdoor \\
    --dataset {dataset} \\
    --attack {attack} \\
    --poison-rate {rate} \\
    --target-label {target_label} \\
    --architecture {architecture} \\
    --epochs 15 \\
    --seed {seed} \\{overrides}
    --output checkpoints/{folder}/attack_result.pt
"""

BASES_CALL = """echo "=== bases {dataset} target {target_label} ==="
python -m cli.lc_bases \\
    --dataset {dataset} \\
    --target-label {target_label} \\
    --epsilon {epsilons} \\
    --steps {steps}
"""


def rate_tag(rate: float) -> str:
    return f"{rate:g}".replace(".", "_")


def target_pool(dataset: str, target_label: int, raw_data_dir: str) -> tuple[int, int]:
    """(images carrying the target label, training set size)."""
    transform = transforms_v2.Compose([transforms_v2.ToTensor()])
    train, _ = load_clean_datasets(dataset, transform, raw_data_dir)
    labels = extract_labels(train)
    return sum(1 for label in labels if int(label) == target_label), len(labels)


def reachable_rates(pool: int, total: int) -> list[float]:
    """The candidate rates this target class has enough images to deliver."""
    return [rate for rate in CANDIDATE_RATES if round(rate * total) <= pool]


def bases_directory(dataset: str, target_label: int, epsilon_over_255: int) -> str:
    return f"results/lc_adversarial/{dataset}_tl{target_label}_eps{epsilon_over_255}"


def folder_name(
    architecture: str,
    dataset: str,
    attack: str,
    rate: float,
    target_label: int,
    adversarial: bool,
    seed: int,
    epsilon_tag: str = "",
) -> str:
    """Canonical template plus the two tags this work introduces.

    `_tl{n}` marks a non-default target class and `_adv` marks Label-Consistent
    carrying its adversarial bases; `_adv` is also what psbd_basis.json names as
    the canonical LC variant, so the older patch-only runs keep their folders
    without competing for the same panel slot. Seed 0 takes no suffix, matching
    the existing seed-replicate convention.
    """
    name = f"{architecture}_{dataset}_{attack}_{rate_tag(rate)}"
    if target_label != 0:
        name += f"_tl{target_label}"
    if adversarial:
        # The pilot trains one folder per epsilon, so the tag has to distinguish
        # them or the three runs overwrite each other. `_pilot` is an excluded
        # panel token, which keeps these diagnostic runs out of the headline;
        # the chosen epsilon reaches the full runs through adversarial_dir in
        # args.json, so `_adv` alone stays traceable there.
        name += f"_adv{epsilon_tag}"
    if seed != 0:
        name += f"_seed_{seed}"
    return name


def training_runs(args) -> list[dict]:
    """Every (dataset, attack, rate, seed) this stage should train, reachable only."""
    runs = [
        run
        for run in _all_training_runs(args)
        # The dirty-label controls belong to whichever clean-label attack runs.
        if run["attack"] not in ("lc", "sig") or run["attack"] in args.attacks
    ]
    return runs


def _all_training_runs(args) -> list[dict]:
    runs = []
    for dataset in args.datasets:
        target_label = CLEAN_LABEL_TARGETS.get(dataset, 0)
        pool, total = target_pool(dataset, target_label, args.raw_data_dir)
        rates = reachable_rates(pool, total)
        print(
            f"  {dataset}: class {target_label} holds {pool} of {total} "
            f"({pool / total:.2%}), reachable rates "
            f"{[f'{rate:.1%}' for rate in rates] or 'none'}"
        )
        if args.stage == "pilot":
            # One rate per dataset: the strongest the data allows, which is where an
            # epsilon that cannot implant at all will fail most visibly.
            selected = [
                (rate, epsilon, 0)
                for rate in rates[-1:]
                for epsilon in EPSILONS_OVER_255
            ]
        else:
            selected = [
                (rate, args.epsilon_by_dataset.get(dataset, args.default_epsilon), seed)
                for rate in rates
                for seed in args.seeds
            ]
        for rate, epsilon, seed in selected:
            runs.append(
                {
                    "epsilon_tag": f"{epsilon}_pilot" if args.stage == "pilot" else "",
                    "dataset": dataset,
                    "attack": "lc",
                    "rate": rate,
                    "target_label": target_label,
                    "seed": seed,
                    "epsilon": epsilon,
                    "adversarial": True,
                }
            )
        # SIG gains nothing from adversarial bases; on GTSRB it only needed a target
        # class with enough images, so it rides along wherever that switch applies.
        if args.stage == "full" and dataset in CLEAN_LABEL_TARGETS:
            runs.extend(
                {
                    "dataset": dataset,
                    "attack": "sig",
                    "rate": rate,
                    "target_label": target_label,
                    "seed": seed,
                    "epsilon": None,
                    "adversarial": False,
                }
                for rate in rates
                for seed in args.seeds
            )
    if args.stage == "full" and args.controls:
        # A dirty-label pair at the same non-default target class. If detection
        # AUROC moves with the target class, the clean-label switch is a confound
        # rather than a fix, and this is what shows it either way.
        runs.extend(
            {
                "dataset": dataset,
                "attack": attack,
                "rate": 0.05,
                "target_label": target,
                "seed": 0,
                "epsilon": None,
                "adversarial": False,
            }
            for dataset, target in CLEAN_LABEL_TARGETS.items()
            if dataset in args.datasets
            for attack in ("badnet_a2o", "blend")
        )
    return runs


def train_body(run: dict, architecture: str) -> str:
    overrides = ""
    if run["adversarial"]:
        directory = bases_directory(run["dataset"], run["target_label"], run["epsilon"])
        # One flag carrying both keys. Repeating the flag also works now that it
        # accumulates, but a single flag cannot regress if that ever changes back.
        overrides = (
            f"\n    --attack-override adversarial_dir={directory}"
            f" adversarial_epsilon={run['epsilon'] / 255:.6f} \\"
        )
    return TRAIN_CALL.format(
        folder=folder_name(
            architecture,
            run["dataset"],
            run["attack"],
            run["rate"],
            run["target_label"],
            run["adversarial"],
            run["seed"],
            run.get("epsilon_tag", ""),
        ),
        dataset=run["dataset"],
        attack=run["attack"],
        rate=run["rate"],
        target_label=run["target_label"],
        architecture=architecture,
        seed=run["seed"],
        overrides=overrides,
    )


def pack(units: list[tuple[int, str]], budget: int) -> list[list[str]]:
    """Greedy bin-pack by predicted minutes, never splitting a unit."""
    bundles, current, spent = [], [], 0
    for minutes, body in units:
        if current and spent + minutes > budget:
            bundles.append(current)
            current, spent = [], 0
        current.append(body)
        spent += minutes
    if current:
        bundles.append(current)
    return bundles


def write_jobs(batch: str, bundles: list[list[str]], minutes: list[int]) -> None:
    job_dir = os.path.join("pbs", batch)
    os.makedirs(job_dir, exist_ok=True)
    os.makedirs(os.path.join("logs", batch), exist_ok=True)
    submit_lines = []
    for index, (bundle, spent) in enumerate(zip(bundles, minutes), start=1):
        name = f"{batch}_{index}"
        # A job killed at the wall loses every run it had not yet written, so the
        # request carries 2x headroom over the predicted time.
        hours = min(48, max(6, int(spent / 60 * 2.0) + 2))
        path = os.path.join(job_dir, f"{name}.pbs")
        with open(path, "w") as handle:
            handle.write(
                JOB_TEMPLATE.format(
                    walltime=f"{hours}:00:00",
                    name=name,
                    root=PROJECT_ROOT,
                    batch=batch,
                    body="\n".join(bundle),
                )
            )
        submit_lines.append(f"qsub {PROJECT_ROOT}/{path}")
    submit = os.path.join(job_dir, "submit_all.sh")
    with open(submit, "w") as handle:
        handle.write("#!/bin/bash\n" + "\n".join(submit_lines) + "\n")
    os.chmod(submit, 0o755)
    print(f"\n[ok] pbs/{batch}/  {len(bundles)} jobs, {sum(minutes)} predicted minutes")
    print(f"submit with: bash pbs/{batch}/submit_all.sh")


def parse_epsilon_map(pairs: list[str] | None) -> dict:
    """`--epsilon cifar10=16 gtsrb=8`, in units of 1/255."""
    mapping = {}
    for pair in pairs or []:
        dataset, _, value = pair.partition("=")
        mapping[dataset] = int(value)
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("bases", "pilot", "full"))
    parser.add_argument("--architecture", default="vit", choices=("vit", "swin"))
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--attacks",
        nargs="+",
        default=["lc", "sig"],
        choices=("lc", "sig"),
        help="which clean-label attacks the full stage trains. SIG needs no "
        "adversarial bases, so it can run before the pilot has chosen an epsilon.",
    )
    parser.add_argument("--epsilon", nargs="+", help="per dataset, e.g. cifar10=16")
    parser.add_argument("--default-epsilon", type=int, default=16)
    parser.add_argument("--pgd-steps", type=int, default=100)
    parser.add_argument("--minutes-per-job", type=int, default=400)
    parser.add_argument("--raw-data-dir", default="raw_data")
    parser.add_argument("--batch", default=None)
    parser.add_argument(
        "--no-controls",
        dest="controls",
        action="store_false",
        help="skip the dirty-label runs at the switched target class",
    )
    args = parser.parse_args()
    args.epsilon_by_dataset = parse_epsilon_map(args.epsilon)
    batch = args.batch or f"{args.architecture}_cleanlabel_{args.stage}"

    if args.stage == "bases":
        units = [
            (
                BASES_MINUTES[dataset] * len(EPSILONS_OVER_255),
                BASES_CALL.format(
                    dataset=dataset,
                    target_label=CLEAN_LABEL_TARGETS.get(dataset, 0),
                    epsilons=" ".join(f"{e / 255:.6f}" for e in EPSILONS_OVER_255),
                    steps=args.pgd_steps,
                ),
            )
            for dataset in args.datasets
        ]
    else:
        print(f"reachable rates for stage {args.stage}:")
        runs = training_runs(args)
        units = [
            (
                TRAIN_MINUTES[args.architecture][run["dataset"]],
                train_body(run, args.architecture),
            )
            for run in runs
        ]
        print(f"\n  {len(runs)} training runs")

    bundles = pack(units, args.minutes_per_job)
    spent = []
    index = 0
    for bundle in bundles:
        spent.append(sum(minutes for minutes, _ in units[index : index + len(bundle)]))
        index += len(bundle)
    write_jobs(batch, bundles, spent)


if __name__ == "__main__":
    main()
