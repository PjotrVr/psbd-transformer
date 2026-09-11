"""3 checks a reviewer would ask for, on the 65-cell panel `scripts/paper/tab_headline.py` reads.

Every number below comes from the same stage-1 caches (`results/<folder>/psbd/`)
and the same scoring functions every other table in the repo reads, imported
rather than reimplemented: `defences.cache` for the raw tensors,
`defences.scores.psu_ratio_from_cache` for the fractional PSU (the canon
headline statistic), `defences.decision` for the rate rule, the threshold and
the detection report, `scripts.paper.tab_headline.measure_cell` and
`common_coverage` for the panel's model selection and
`experiments.probe_union.measure`/`pair_search` for the min-rank union.

Check 1, the size of the defender's clean set (`check1_clean_set_size`). The
canonical pipeline sizes the clean validation set at 2000 images
(`data.splits.PSBD_HELDOUT_SIZE`) and reads both the adaptive rate rule and the
detection threshold off it. This redraws random subsets of 100, 200, 500 and
1000 of those 2000 images, 5 draws per size at a fixed seed, then reruns both
steps on the subset: the per-rate shift ratio (`defences.scores.shift_ratio`,
recomputed from the cached per-pass argmax rather than read off the stored
2000-image figure, since the rate rule the check is testing IS the function of
subset size), `defences.decision.select_rate_adaptively` at the smaller
validation set, and the threshold and detection report
(`defences.decision.detection_report`) at the rate that rule picks. The clean
and backdoor analysis pool is untouched at every subset size, exactly as the
check asks: only the defender's own clean set shrinks.

Check 2, the union chosen on held-out models (`check2_union_generalisation`).
`configs/psbd_basis.json`'s `selection_protocol` names the same split the
paper's placement selection uses: CIFAR-10 and GTSRB models choose, CIFAR-100
and Tiny ImageNet models report. This applies that split a level up, to the
probe union `experiments/probe_union/pair_search.py` builds. Candidate
placements are every basis entry present on all 65 models
(`experiments.probe_union.measure.basis_ids_present_on_all_models`). Every
pair is scored by the min-rank union
(`experiments.probe_union.pair_search.search_pairs`, itself built on
`defences.decision.multi_probe_auroc`/`multi_probe_detection`) on the
selection models only, ranked by mean TPR at the 10% clean-validation
quantile. The winning pair is then read, for the first time, on the held-out
models, beside PSBD-TM alone on the same held-out models, so the reported gain
carries no selection bias.

Check 3, the mask seed (`check3_mask_seed`). Every PSBD-TM number in the paper
comes from 1 draw of the token-masking sequence
(`cli.sweep`'s `PSBD_MASK_SEED = 0`). This reruns 10 models, spread over the 4
main datasets and the attack roster, at PSBD-TM's already-selected rate with
`--mask-seed 1` and `--mask-seed 2` on the login GPU, writing into
`results/_experiments/reviewer_checks/mask_seeds/` so the canonical caches
under `results/<folder>/psbd/` are never touched. `cli.analyze` reads that
separate tree the same way it reads any other, and the check compares AUROC
and TPR at 10% across the 3 seeds per model.

    PYTHONPATH=. .venv/bin/python experiments/reviewer_checks/measure.py
    PYTHONPATH=. .venv/bin/python experiments/reviewer_checks/measure.py --skip-check3
"""

import argparse
import json
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from data.splits import PSBD_HELDOUT_SIZE  # noqa: E402
from defences.cache import (  # noqa: E402
    baseline_path,
    dropout_pass_path,
    load_baseline,
    load_dropout_pass_probs,
    read_split_manifest,
)
from defences.decision import (  # noqa: E402
    ADAPTIVE_SHIFT_TARGET,
    PUBLISHED_PLACEMENT,
    RECOMMENDED_PLACEMENT,
    complete_rates,
    detection_report,
    pair_clean_to_backdoor,
    select_rate_adaptively,
)
from defences.scores import psu_ratio_from_cache, shift_ratio  # noqa: E402
from experiments._paths import experiment_result_path, experiment_results_dir  # noqa: E402
from experiments.probe_union.measure import (  # noqa: E402
    ATTN_BRANCH_TM,
    basis_ids_present_on_all_models,
    paired_gain,
    select_models,
)
from experiments.probe_union.pair_search import (  # noqa: E402
    preload_placement,
    search_pairs,
    set_summary,
    union_set,
)
from scripts.paper._common import (  # noqa: E402
    load_coverage,
    load_declaration,
    load_psbd_metrics,
    mean_or_none,
    std_or_none,
)
from scripts.paper.tab_headline import clearing_cells, common_coverage, measure_cell  # noqa: E402
from utils.provenance import current_git_commit, utc_timestamp  # noqa: E402

SLUG = "reviewer_checks"

CHECK1_PLACEMENTS = {"psbd_tm": RECOMMENDED_PLACEMENT, "psbd_rd": PUBLISHED_PLACEMENT}
CHECK1_SUBSET_SIZES = (100, 200, 500, 1000)
CHECK1_DRAWS = 5
CHECK1_SEED = 0
CHECK1_QUANTILES = (0.10, 0.20)

CHECK2_REFERENCE = RECOMMENDED_PLACEMENT
CHECK2_TARGET_FPRS = (0.10, 0.20)
CHECK2_PRIMARY_FPR = 0.10
CHECK2_BOOTSTRAP_RESAMPLES = 2000
CHECK2_BOOTSTRAP_SEED = 0
# The deployment candidate, PSBD-TM plus token masking on the attention branch
# output, sits on 66 of the 69 panel models, so the strict "on every model"
# candidate rule excludes it. This relaxed rule follows up on that: a
# placement qualifies once it covers at least this many of the 69, and a
# pair's own coverage (union_set's model_has_probes filter) still decides
# which models that particular pair is read on.
CHECK2B_MIN_COVERAGE = 60

CHECK3_POSITION = "before_attention_norm"
CHECK3_OPERATOR = "token_mask"
CHECK3_MASK_SEEDS = (1, 2)
CHECK3_QUANTILE_KEY = "q0.10"
# 10 models spread over the 4 main datasets and the attack roster. The 2
# models the task names explicitly (cifar10 wanet at 10% and cifar100 badnets
# at 1%) are first, the other 8 fill out the remaining datasets and attacks.
CHECK3_MODELS = (
    "vit_cifar10_wanet_0_1",
    "vit_cifar100_badnet_a2o_0_01",
    "vit_cifar10_sig_0_1",
    "vit_cifar10_badnet_a2o_0_1",
    "vit_cifar100_bpp_0_05",
    "vit_cifar100_tact_0_01",
    "vit_gtsrb_bpp_0_05",
    "vit_gtsrb_wanet_0_1",
    "vit_tiny_blend_0_1",
    "vit_tiny_badnet_a2o_0_01",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--declaration", default="configs/psbd_basis.json")
    parser.add_argument(
        "--python-bin",
        default=os.path.join(REPO_ROOT, ".venv", "bin", "python"),
        help="interpreter cli.sweep and cli.analyze subprocesses run under",
    )
    parser.add_argument(
        "--output",
        default=experiment_result_path(SLUG, "reviewer_checks.json"),
    )
    parser.add_argument(
        "--skip-check3",
        action="store_true",
        help="skip the GPU mask-seed rerun, e.g. when it already ran and only checks 1 and 2 changed",
    )
    parser.add_argument(
        "--check2-relaxed-only",
        action="store_true",
        help=(
            "only run check 2's relaxed-coverage follow-up and merge it into the "
            "existing --output file, without rerunning checks 1 and 3"
        ),
    )
    return parser.parse_args()


def selected_cells(results_dir: str) -> list[dict]:
    """The panel `tab_headline.py` reads: clearing cells with both headline placements cached.

    Reuses `measure_cell` and `common_coverage` rather than re-deriving the
    selection, so this script's model set can never drift from the table it is
    reviewing.
    """
    coverage = load_coverage(results_dir)
    cells = clearing_cells(coverage)
    for cell in cells:
        cell["values"] = measure_cell(results_dir, cell["folder_name"])
    covered = common_coverage(cells)
    return covered


def subset_index_table(
    seed: int, heldout_size: int
) -> dict[tuple[int, int], np.ndarray]:
    """Every (size, draw) subset of validation row-indices, drawn once, fixed for every model.

    A single generator advances across every size and draw in turn, so the
    same seed always yields the same sequence of subsets regardless of which
    model or placement reads them, and 2 draws never coincide by accident.
    """
    generator = np.random.default_rng(seed)
    table = {}
    for size in CHECK1_SUBSET_SIZES:
        for draw in range(CHECK1_DRAWS):
            table[(size, draw)] = generator.choice(
                heldout_size, size=size, replace=False
            )
    return table


def check1_model_placement(
    results_dir: str, folder: str, placement: str, index_table: dict
) -> list[dict]:
    """Every (size, draw) row for 1 (model, placement): selected rate, AUROC and TPR.

    Validation's per-pass argmax and probs are loaded once per rate, needed at
    every subset draw to recompute the shift ratio. Clean and backdoor are
    loaded lazily, only for whichever rate a draw's rule actually selects,
    since the analysis pool never changes across subset sizes.
    """
    psbd_dir = os.path.join(results_dir, folder, "psbd")
    manifest = read_split_manifest(psbd_dir)
    baseline_probs_val, baseline_labels_val, _ = load_baseline(
        baseline_path(psbd_dir, "validation")
    )
    baseline_probs_clean, baseline_labels_clean, _ = load_baseline(
        baseline_path(psbd_dir, "clean")
    )
    baseline_probs_bd, baseline_labels_bd, _ = load_baseline(
        baseline_path(psbd_dir, "backdoor")
    )

    rates = complete_rates(psbd_dir, placement)
    per_rate_validation = {
        rate: load_dropout_pass_probs(
            dropout_pass_path(psbd_dir, placement, rate, "validation")
        )
        for rate in rates
    }
    # (clean_psu_paired, backdoor_psu) at whichever rate gets selected, filled lazily.
    analysis_psu_by_rate: dict[float, tuple[torch.Tensor, torch.Tensor]] = {}

    def analysis_psu_at(rate: float) -> tuple[torch.Tensor, torch.Tensor]:
        if rate not in analysis_psu_by_rate:
            clean_probs, _ = load_dropout_pass_probs(
                dropout_pass_path(psbd_dir, placement, rate, "clean")
            )
            backdoor_probs, _ = load_dropout_pass_probs(
                dropout_pass_path(psbd_dir, placement, rate, "backdoor")
            )
            clean_psu = psu_ratio_from_cache(
                baseline_probs_clean, baseline_labels_clean, clean_probs
            )
            backdoor_psu = psu_ratio_from_cache(
                baseline_probs_bd, baseline_labels_bd, backdoor_probs
            )
            analysis_psu_by_rate[rate] = (
                pair_clean_to_backdoor(clean_psu, manifest),
                backdoor_psu,
            )
        return analysis_psu_by_rate[rate]

    rows = []
    for size in CHECK1_SUBSET_SIZES:
        for draw in range(CHECK1_DRAWS):
            subset = torch.as_tensor(
                index_table[(size, draw)], dtype=torch.long
            )  # (size,)

            shift_by_rate = {}
            for rate in rates:
                _, per_pass_argmax = per_rate_validation[rate]
                if per_pass_argmax.numel() == 0:
                    continue
                shift_by_rate[rate] = shift_ratio(
                    baseline_labels_val[subset], per_pass_argmax[:, subset]
                )
            selected_rate = select_rate_adaptively(shift_by_rate, ADAPTIVE_SHIFT_TARGET)

            row = {"size": size, "draw": draw, "selected_rate": selected_rate}
            if selected_rate is None:
                rows.append(row)
                continue

            per_pass_probs, _ = per_rate_validation[selected_rate]
            validation_psu = psu_ratio_from_cache(
                baseline_probs_val[subset],
                baseline_labels_val[subset],
                per_pass_probs[:, subset],
            )  # (size,)
            clean_psu_paired, backdoor_psu = analysis_psu_at(selected_rate)

            for quantile in CHECK1_QUANTILES:
                report = detection_report(
                    validation_psu, clean_psu_paired, backdoor_psu, quantile
                )
                row[f"auroc_q{quantile:.2f}"] = report["auroc"]
                row[f"tpr_q{quantile:.2f}"] = report["tpr"]
            rows.append(row)
    return rows


def check1_model_summary(rows: list[dict], size: int) -> dict | None:
    """1 model's mean and draw-to-draw spread at 1 subset size, None if every draw failed."""
    at_size = [
        row for row in rows if row["size"] == size and row["selected_rate"] is not None
    ]
    if not at_size:
        return None
    summary = {
        "n_draws": len(at_size),
        "selected_rate_mean": mean_or_none([row["selected_rate"] for row in at_size]),
    }
    for quantile in CHECK1_QUANTILES:
        key = f"q{quantile:.2f}"
        summary[f"auroc_{key}_mean"] = mean_or_none(
            [row[f"auroc_{key}"] for row in at_size]
        )
        summary[f"auroc_{key}_draw_std"] = std_or_none(
            [row[f"auroc_{key}"] for row in at_size]
        )
        summary[f"tpr_{key}_mean"] = mean_or_none(
            [row[f"tpr_{key}"] for row in at_size]
        )
        summary[f"tpr_{key}_draw_std"] = std_or_none(
            [row[f"tpr_{key}"] for row in at_size]
        )
    return summary


def check1_size_aggregate(per_model: dict[str, dict | None], size: int) -> dict:
    """1 subset size's aggregate over models: mean of the per-model means, plus the worst model."""
    covered = {
        folder: summary for folder, summary in per_model.items() if summary is not None
    }
    aggregate = {"size": size, "n_models": len(covered)}
    if not covered:
        return aggregate

    aggregate["selected_rate_mean"] = mean_or_none(
        [s["selected_rate_mean"] for s in covered.values()]
    )
    for quantile in CHECK1_QUANTILES:
        key = f"q{quantile:.2f}"
        aggregate[f"auroc_{key}_mean"] = mean_or_none(
            [s[f"auroc_{key}_mean"] for s in covered.values()]
        )
        aggregate[f"auroc_{key}_draw_std_mean"] = mean_or_none(
            [
                s[f"auroc_{key}_draw_std"]
                for s in covered.values()
                if s[f"auroc_{key}_draw_std"] is not None
            ]
        )
        aggregate[f"tpr_{key}_mean"] = mean_or_none(
            [s[f"tpr_{key}_mean"] for s in covered.values()]
        )

    worst_folder = min(covered, key=lambda folder: covered[folder]["auroc_q0.10_mean"])
    aggregate["worst_model"] = {
        "folder": worst_folder,
        "auroc_q0.10_mean": covered[worst_folder]["auroc_q0.10_mean"],
        "tpr_q0.10_mean": covered[worst_folder]["tpr_q0.10_mean"],
    }
    return aggregate


def run_check1(results_dir: str, cells: list[dict]) -> dict:
    """Check 1's full report: every (placement, size) aggregate, over every panel model."""
    index_table = subset_index_table(CHECK1_SEED, PSBD_HELDOUT_SIZE)
    report = {
        "heldout_size": PSBD_HELDOUT_SIZE,
        "subset_sizes": list(CHECK1_SUBSET_SIZES),
        "draws": CHECK1_DRAWS,
        "seed": CHECK1_SEED,
    }

    for name, placement in CHECK1_PLACEMENTS.items():
        per_model_rows = {
            cell["folder_name"]: check1_model_placement(
                results_dir, cell["folder_name"], placement, index_table
            )
            for cell in cells
        }
        per_model_summary = {
            size: {
                folder: check1_model_summary(rows, size)
                for folder, rows in per_model_rows.items()
            }
            for size in CHECK1_SUBSET_SIZES
        }
        report[name] = {
            "placement": placement,
            "n_models": len(cells),
            "by_size": [
                check1_size_aggregate(per_model_summary[size], size)
                for size in CHECK1_SUBSET_SIZES
            ],
        }
    return report


def run_check2(
    results_dir: str, declaration_path: str, resamples: int, seed: int
) -> dict:
    """Check 2's full report: the best-on-selection pair, read fresh on the held-out models."""
    declaration = load_declaration(declaration_path)
    protocol = declaration["selection_protocol"]
    select_datasets = set(protocol["select_on_datasets"])
    heldout_datasets = set(protocol["report_on_datasets"])

    models = select_models(results_dir)
    candidates = basis_ids_present_on_all_models(models, declaration_path)
    selection_models = [
        model for model in models if model["dataset"] in select_datasets
    ]
    heldout_models = [model for model in models if model["dataset"] in heldout_datasets]

    manifest_cache: dict = {}
    psu_cache: dict = {}
    for placement in set(candidates) | {CHECK2_REFERENCE}:
        preload_placement(results_dir, models, placement, manifest_cache, psu_cache)

    reference_rows = union_set(
        results_dir, selection_models, (CHECK2_REFERENCE,), manifest_cache, psu_cache
    )
    pairs = search_pairs(
        results_dir,
        selection_models,
        candidates,
        reference_rows,
        manifest_cache,
        psu_cache,
        resamples,
        seed,
    )
    best_pair = tuple(pairs[0]["summary"]["placements"])

    heldout_reference_rows = union_set(
        results_dir, heldout_models, (CHECK2_REFERENCE,), manifest_cache, psu_cache
    )
    heldout_pair_rows = union_set(
        results_dir, heldout_models, best_pair, manifest_cache, psu_cache
    )
    heldout_gain = paired_gain(
        heldout_reference_rows, heldout_pair_rows, resamples, seed
    )

    report = {
        "select_on_datasets": sorted(select_datasets),
        "report_on_datasets": sorted(heldout_datasets),
        "n_selection_models": len(selection_models),
        "n_heldout_models": len(heldout_models),
        "n_candidate_placements": len(candidates),
        "candidate_placements": sorted(candidates),
        "best_pair_on_selection": {
            "placements": list(best_pair),
            "selection_summary": pairs[0]["summary"],
        },
        "held_out": {
            "psbd_tm_alone": set_summary(heldout_reference_rows, (CHECK2_REFERENCE,)),
            "best_pair": set_summary(heldout_pair_rows, best_pair),
            "pair_gain_over_psbd_tm": heldout_gain,
        },
    }
    return report


def placements_present_on_at_least(
    models: list[dict], declaration_path: str, min_count: int
) -> list[str]:
    """Every basis placement id whose `adaptive_rate` is set on at least `min_count` of `models`.

    The strict rule (`basis_ids_present_on_all_models`) requires every model,
    which drops a placement like the attention branch output token mask that
    covers all but a few of the panel. Each surviving candidate pair still
    reads its own model coverage through `union_set`'s `model_has_probes`
    filter, so relaxing this rule only widens which pairs are considered, not
    which models a given pair is scored on.
    """
    with open(declaration_path) as handle:
        basis_ids = [entry["id"] for entry in json.load(handle)["basis"]]

    present = []
    for placement in basis_ids:
        covering = sum(
            1
            for model in models
            if (model["report"]["placements"].get(placement) or {}).get("adaptive_rate")
            is not None
        )
        if covering >= min_count:
            present.append(placement)
    return present


def run_check2_relaxed_coverage(
    results_dir: str,
    declaration_path: str,
    resamples: int,
    seed: int,
    min_coverage: int,
) -> dict:
    """Check 2 again with the coverage bar relaxed, so the deployment pair can enter the search.

    Same selection protocol as `run_check2`: the pair is chosen on the
    CIFAR-10 and GTSRB selection models by mean TPR at 10%, then read fresh on
    the CIFAR-100 and Tiny ImageNet held-out models. The 1 addition is a
    direct held-out reading of PSBD-TM plus the attention branch output token
    mask, reported whether or not the search actually picks that pair.
    """
    declaration = load_declaration(declaration_path)
    protocol = declaration["selection_protocol"]
    select_datasets = set(protocol["select_on_datasets"])
    heldout_datasets = set(protocol["report_on_datasets"])

    models = select_models(results_dir)
    candidates = placements_present_on_at_least(models, declaration_path, min_coverage)
    selection_models = [
        model for model in models if model["dataset"] in select_datasets
    ]
    heldout_models = [model for model in models if model["dataset"] in heldout_datasets]

    manifest_cache: dict = {}
    psu_cache: dict = {}
    for placement in set(candidates) | {CHECK2_REFERENCE, ATTN_BRANCH_TM}:
        preload_placement(results_dir, models, placement, manifest_cache, psu_cache)

    reference_rows = union_set(
        results_dir, selection_models, (CHECK2_REFERENCE,), manifest_cache, psu_cache
    )
    pairs = search_pairs(
        results_dir,
        selection_models,
        candidates,
        reference_rows,
        manifest_cache,
        psu_cache,
        resamples,
        seed,
    )
    best_pair = tuple(pairs[0]["summary"]["placements"])

    heldout_reference_rows = union_set(
        results_dir, heldout_models, (CHECK2_REFERENCE,), manifest_cache, psu_cache
    )
    heldout_pair_rows = union_set(
        results_dir, heldout_models, best_pair, manifest_cache, psu_cache
    )
    heldout_gain = paired_gain(
        heldout_reference_rows, heldout_pair_rows, resamples, seed
    )

    # Reported directly regardless of whether the search actually picked this
    # pair, since it is the pair a deployer would reach for.
    branch_pair = tuple(sorted((CHECK2_REFERENCE, ATTN_BRANCH_TM)))
    heldout_branch_rows = union_set(
        results_dir, heldout_models, branch_pair, manifest_cache, psu_cache
    )
    heldout_branch_gain = paired_gain(
        heldout_reference_rows, heldout_branch_rows, resamples, seed
    )

    report = {
        "min_coverage": min_coverage,
        "select_on_datasets": sorted(select_datasets),
        "report_on_datasets": sorted(heldout_datasets),
        "n_selection_models": len(selection_models),
        "n_heldout_models": len(heldout_models),
        "n_candidate_placements": len(candidates),
        "candidate_placements": sorted(candidates),
        "best_pair_on_selection": {
            "placements": list(best_pair),
            "selection_summary": pairs[0]["summary"],
        },
        "held_out": {
            "psbd_tm_alone": set_summary(heldout_reference_rows, (CHECK2_REFERENCE,)),
            "best_pair": set_summary(heldout_pair_rows, best_pair),
            "pair_gain_over_psbd_tm": heldout_gain,
        },
        "held_out_attention_branch_pair": {
            "placements": list(branch_pair),
            "summary": set_summary(heldout_branch_rows, branch_pair),
            "gain_over_psbd_tm": heldout_branch_gain,
        },
    }
    return report


def run_sweep_subprocess(
    python_bin: str,
    folder: str,
    rate: float,
    mask_seed: int,
    results_dir: str,
    checkpoints_dir: str,
) -> None:
    """1 GPU sweep at 1 already-selected rate, writing under the reviewer-checks results tree."""
    command = [
        python_bin,
        "-m",
        "cli.sweep",
        "--checkpoint-folder",
        folder,
        "--position",
        CHECK3_POSITION,
        "--operator",
        CHECK3_OPERATOR,
        "--rates",
        str(rate),
        "--mask-seed",
        str(mask_seed),
        "--results-dir",
        results_dir,
        "--checkpoints-dir",
        checkpoints_dir,
    ]
    subprocess.run(command, check=True, cwd=REPO_ROOT)


def run_analyze_subprocess(
    python_bin: str, folder: str, results_dir: str, checkpoints_dir: str
) -> None:
    """CPU stage 2 over the reviewer-checks results tree, writing its own psbd_metrics.json."""
    command = [
        python_bin,
        "-m",
        "cli.analyze",
        "--checkpoint-folder",
        folder,
        "--results-dir",
        results_dir,
        "--checkpoints-dir",
        checkpoints_dir,
    ]
    subprocess.run(command, check=True, cwd=REPO_ROOT)


def rate_row_detection(
    placement_block: dict, rate: float, quantile_key: str
) -> dict | None:
    """The detection_psu_ratio block of the 1 rate row matching `rate`, None if absent."""
    for row in placement_block.get("rates", []):
        if row.get("rate") == rate:
            return row["detection_psu_ratio"][quantile_key]
    return None


def check3_model(
    python_bin: str,
    folder: str,
    results_dir_main: str,
    mask_seed_results_dir: str,
    checkpoints_dir: str,
) -> dict:
    """1 model's AUROC and TPR at 10% under mask seeds 0 (canonical), 1 and 2."""
    canonical_report = load_psbd_metrics(results_dir_main, folder)
    tm_block = canonical_report["placements"][RECOMMENDED_PLACEMENT]
    rate = tm_block["adaptive_rate"]
    seed0 = rate_row_detection(tm_block, rate, CHECK3_QUANTILE_KEY)

    for mask_seed in CHECK3_MASK_SEEDS:
        run_sweep_subprocess(
            python_bin, folder, rate, mask_seed, mask_seed_results_dir, checkpoints_dir
        )
    run_analyze_subprocess(python_bin, folder, mask_seed_results_dir, checkpoints_dir)

    reseeded_report = load_psbd_metrics(mask_seed_results_dir, folder)
    by_seed = {"0": {"auroc": seed0["auroc"], "tpr": seed0["tpr"]}}
    for mask_seed in CHECK3_MASK_SEEDS:
        placement_name = f"{RECOMMENDED_PLACEMENT}_seed{mask_seed}"
        block = reseeded_report["placements"][placement_name]
        detection = rate_row_detection(block, rate, CHECK3_QUANTILE_KEY)
        by_seed[str(mask_seed)] = {"auroc": detection["auroc"], "tpr": detection["tpr"]}

    row = {
        "folder": folder,
        "rate": rate,
        "by_seed": by_seed,
        "auroc_std": std_or_none([entry["auroc"] for entry in by_seed.values()]),
        "tpr_std": std_or_none([entry["tpr"] for entry in by_seed.values()]),
    }
    return row


def run_check3(python_bin: str, results_dir_main: str, checkpoints_dir: str) -> dict:
    """Check 3's full report: PSBD-TM at 3 mask seeds, on 10 models over the 4 main datasets."""
    mask_seed_results_dir = os.path.join(
        experiment_results_dir(SLUG, results_dir_main), "mask_seeds"
    )
    rows = [
        check3_model(
            python_bin, folder, results_dir_main, mask_seed_results_dir, checkpoints_dir
        )
        for folder in CHECK3_MODELS
    ]
    report = {
        "position": CHECK3_POSITION,
        "operator": CHECK3_OPERATOR,
        "mask_seeds": [0] + list(CHECK3_MASK_SEEDS),
        "results_dir": mask_seed_results_dir,
        "models": rows,
        "auroc_std_mean": mean_or_none(
            [row["auroc_std"] for row in rows if row["auroc_std"] is not None]
        ),
        "tpr_std_mean": mean_or_none(
            [row["tpr_std"] for row in rows if row["tpr_std"] is not None]
        ),
    }
    return report


def merge_check2_relaxed(
    output_path: str, results_dir: str, declaration_path: str
) -> None:
    """Load the existing report, add check 2's relaxed-coverage follow-up, write it back.

    Checks 1 and 3 are left exactly as they were, since this follow-up only
    concerns which placements check 2's candidate search considers.
    """
    with open(output_path) as handle:
        report = json.load(handle)

    relaxed = run_check2_relaxed_coverage(
        results_dir,
        declaration_path,
        CHECK2_BOOTSTRAP_RESAMPLES,
        CHECK2_BOOTSTRAP_SEED,
        CHECK2B_MIN_COVERAGE,
    )
    report["check2_union_generalisation"]["relaxed_coverage"] = relaxed
    with open(output_path, "w") as handle:
        json.dump(report, handle, indent=2)
    print(
        f"check 2 relaxed-coverage follow-up done, best pair "
        f"{relaxed['best_pair_on_selection']['placements']}, merged into {output_path}"
    )


def main() -> None:
    args = parse_args()

    if args.check2_relaxed_only:
        merge_check2_relaxed(args.output, args.results_dir, args.declaration)
        return

    cells = selected_cells(args.results_dir)
    print(f"{len(cells)} panel models (tab_headline.py's common_coverage selection)")

    check1 = run_check1(args.results_dir, cells)
    print("check 1 (clean set size) done")

    check2 = run_check2(
        args.results_dir,
        args.declaration,
        CHECK2_BOOTSTRAP_RESAMPLES,
        CHECK2_BOOTSTRAP_SEED,
    )
    print(
        f"check 2 (union generalisation) done, best pair {check2['best_pair_on_selection']['placements']}"
    )

    if args.skip_check3:
        check3 = None
        print("check 3 (mask seed) skipped by --skip-check3")
    else:
        check3 = run_check3(args.python_bin, args.results_dir, args.checkpoints_dir)
        print("check 3 (mask seed) done")

    report = {
        "generated_at": utc_timestamp(),
        "git_commit": current_git_commit(),
        "n_panel_models": len(cells),
        "check1_clean_set_size": check1,
        "check2_union_generalisation": check2,
        "check3_mask_seed": check3,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(report, handle, indent=2)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
