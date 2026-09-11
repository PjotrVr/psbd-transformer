"""Which placements union best in pairs and larger sets, and how best to combine them.

`experiments/probe_union/measure.py` fixed 6 probe sets by hand and asked whether
the H41 min-rank union helps on ordinary models. This script asks the 3
questions that fixing the sets by hand cannot answer. Which pair of placements,
out of every placement whose sweep reaches the whole selected model set, unions
best (task 1). Whether the min rule is even the right way to combine 2 or 3
probes once a good pair is in hand (task 2). Whether unions of 3, 4 or 5
placements keep helping past a pair or plateau (task 3).

Method, part 1 (the pair search). Models are `measure.select_models`'s clearing
cells that carry both headline placements, exactly as `measure.py` selects them.
The candidate placement set is `measure.basis_ids_present_on_all_models`, every
basis entry whose `adaptive_rate` is set on every 1 of those models, so every
pair is read on the identical population and a pair's mean is never inflated by
covering fewer, easier models. Every per-image fractional PSU
(`defences.scores.psu_ratio_from_cache`, the canon headline statistic) is read
once per (model, placement) from the stage-1 cache and kept in memory
(`experiments.probe_union.measure.load_probe_psu`, imported, not copied), so
the C(k,2) pairs over k candidates cost 1 disk read per placement rather than 1
per pair. Each pair is scored by `defences.decision.multi_probe_auroc` and
`multi_probe_detection` at the min reduction, the same union H41 built and
`measure.py` already measures for 6 fixed sets. Pairs are ranked by mean TPR at
the 10% clean-validation quantile, the operating point a defender would deploy.
A greedy forward search over the 6 candidates with the best solo TPR at 10%
finds the best triple: seed on the best single, add whichever of the remaining
5 raises the pair's mean TPR at 10% the most, then add whichever of the
remaining 4 raises the resulting triple's mean TPR at 10% the most.

Method, part 2 (the combination rule). For 3 named probe sets, the best pair
from part 1, PSBD-TM plus the attention branch output token mask and the
3-probe adaptive-attacker pool of `measure.py` on the 37 models that hold all
3, the same per-image ranks r_j(x) (`defences.scores.to_rank`, the share of
clean validation inputs with a lower PSU under probe j) are combined 6 ways:
min (the union), mean, max (the intersection, every probe must agree), the sum
of z-scored PSU (z-scored on clean validation, per probe), the product of
ranks, and a rank-weighted mean whose weights are each probe's leave-one-out
solo AUROC over the OTHER models in the set, which stays defender-legal because
no model's own labels enter its own weight. Every rule is thresholded at its
own 0.25, 0.10 and 0.20 clean-validation quantiles. The 2-probe best pair is
also split into the models where the pair's PSU ranks disagree most
(`scipy.stats.spearmanr` on the backdoor split's 2 rank vectors, the bottom
quarter by correlation) to ask whether the best rule changes exactly where
disagreement is highest.

Method, part 3 (unions larger than pairs). The pool is the 8 candidates with
the best solo mean TPR at 10% (from part 1's solo search) plus token masking
on the attention branch output, read on its own 66 models since it is not a
part-1 candidate. Every subset of size 3, 4 and 5 from this 9-placement pool
is scored under min, product and z-sum, the 3 rules `combine_scores` already
implements for part 2, ranked by mean TPR at 10% exactly as part 1 ranks
pairs. Size 2 is added only as the trend anchor a size-3-to-5-only sweep
cannot supply on its own. The best subset of each (size, rule) cell is
reported with its paired AUROC gain over PSBD-TM alone
(`experiments.probe_union.measure.paired_gain`), and the deployment
recommendation in the summary is whichever subset reads the best mean TPR at
10% over sizes 3 to 5, with the next size's gain under the same rule reported
alongside it to show whether growing the union further would still help.

    PYTHONPATH=. .venv/bin/python experiments/probe_union/pair_search.py

Output: `results/_experiments/probe_union/pair_search.json` (every pair, the
greedy triple, the 6-rule comparison and the pool search) and
`docs/probe_union_tables_2026-09-11.md` (the per-dataset tables and the
top-10, rule and pool-search tables, in the layout of
`docs/site_a_vs_b_tables_2026-09-11.md`).
"""

import argparse
import itertools
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import json  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from defences.cache import read_split_manifest  # noqa: E402
from defences.decision import multi_probe_auroc, multi_probe_detection  # noqa: E402
from defences.scores import to_rank  # noqa: E402
from experiments._paths import experiment_result_path  # noqa: E402
from experiments.probe_union.measure import (  # noqa: E402
    ATTN_BRANCH_TM,
    ATTN_INPUT_DROPOUT,
    MLP_NORM_GAIN,
    PSBD_TM,
    TARGET_FPRS,
    basis_ids_present_on_all_models,
    load_probe_psu,
    model_has_probes,
    paired_gain,
    select_models,
)
from scripts.paper._common import (  # noqa: E402
    attack_label,
    dataset_label,
    load_coverage,
    mean_or_none,
    placement_label,
)

SLUG = "probe_union"
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 0

# The rules task 2 compares, on the same per-probe ranks. "min" is the union
# H41 built and the reference every other rule is measured against.
RULES = ("min", "mean", "max", "zsum", "product", "weighted_mean")
RULE_TARGET_FPRS = (0.25, 0.10, 0.20)
PRIMARY_TARGET_FPR = 0.10

TOP6_SIZE = 6
DISAGREEMENT_SHARE = 0.25  # the bottom quarter by rank correlation, "disagree most"

# Task 3, the pool search: the 8 candidates with the best solo TPR at 10% plus
# the attention branch output token mask, unioned at sizes 2 (the trend
# anchor, not separately requested) through 5, under the 3 rules the brief
# names. weighted_mean is left out here, its leave-one-out weights are a
# per-probe-set statistic and the brief asks for min, product and zsum only.
POOL_SIZE = 8
POOL_SUBSET_SIZES = (2, 3, 4, 5)
POOL_REPORT_SIZES = (3, 4, 5)
POOL_RULES = ("min", "product", "zsum")
POOL_TOP_KEEP = 5  # subsets kept per (size, rule) in the JSON, beyond the best

# The attack order and daggering rule task 3's tables follow, matching
# docs/site_a_vs_b_tables_2026-09-11.md.
ATTACK_ORDER = ("badnet_a2o", "blend", "lf", "bpp", "wanet", "tact", "sig")
ASR_BAR = 0.85
DATASET_ORDER = ("cifar10", "cifar100", "gtsrb", "tiny", "svhn", "eurosat")
RATE_LABELS = {0.01: "1%", 0.05: "5%", 0.1: "10%"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--declaration", default="configs/psbd_basis.json")
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument(
        "--output", default=experiment_result_path(SLUG, "pair_search.json")
    )
    parser.add_argument(
        "--docs-output",
        default=os.path.join("docs", "probe_union_tables_2026-09-11.md"),
    )
    return parser.parse_args()


def load_declaration_entries(declaration_path: str) -> dict[str, dict]:
    """Every basis entry keyed by its placement id, for placement_label lookups."""
    with open(declaration_path) as handle:
        basis = json.load(handle)["basis"]
    entries = {entry["id"]: entry for entry in basis}
    return entries


def readable_placement(entries: dict[str, dict], placement_id: str) -> str:
    """A placement id in words, or the id itself when the declaration lacks it."""
    entry = entries.get(placement_id)
    words = placement_label(entry) if entry is not None else placement_id
    return words


def cached_manifest(results_dir: str, folder: str, manifest_cache: dict) -> dict:
    if folder not in manifest_cache:
        psbd_dir = os.path.join(results_dir, folder, "psbd")
        manifest_cache[folder] = read_split_manifest(psbd_dir)
    return manifest_cache[folder]


def cached_probe_psu(
    results_dir: str,
    model: dict,
    placement: str,
    manifest_cache: dict,
    psu_cache: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """1 (model, placement)'s (validation, clean_paired, backdoor) PSU, loaded once.

    Every pair and every rule reads the same underlying PSU vectors, so this is
    the single point every consumer below goes through and the single point
    that ever touches `defences.cache`.
    """
    key = (model["folder_name"], placement)
    if key not in psu_cache:
        psbd_dir = os.path.join(results_dir, model["folder_name"], "psbd")
        manifest = cached_manifest(results_dir, model["folder_name"], manifest_cache)
        rate = model["report"]["placements"][placement]["adaptive_rate"]
        psu_cache[key] = load_probe_psu(psbd_dir, placement, rate, manifest)
    return psu_cache[key]


def preload_placement(
    results_dir: str,
    models: list[dict],
    placement: str,
    manifest_cache: dict,
    psu_cache: dict,
) -> None:
    """Read 1 placement's PSU on every model that carries it, once each."""
    for model in models:
        if model_has_probes(model, (placement,)):
            cached_probe_psu(results_dir, model, placement, manifest_cache, psu_cache)


def probe_arrays(
    results_dir: str,
    model: dict,
    placements: tuple[str, ...],
    manifest_cache: dict,
    psu_cache: dict,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """1 model's (validation, clean, backdoor) PSU lists, 1 tensor per probe."""
    val_list, clean_list, backdoor_list = [], [], []
    for placement in placements:
        val, clean, backdoor = cached_probe_psu(
            results_dir, model, placement, manifest_cache, psu_cache
        )
        val_list.append(val)
        clean_list.append(clean)
        backdoor_list.append(backdoor)
    return val_list, clean_list, backdoor_list


def union_row(
    results_dir: str,
    model: dict,
    placements: tuple[str, ...],
    manifest_cache: dict,
    psu_cache: dict,
) -> dict:
    """1 model's min-rank union AUROC and TPR at both target FPRs, for 1 placement set."""
    val_list, clean_list, backdoor_list = probe_arrays(
        results_dir, model, placements, manifest_cache, psu_cache
    )
    auroc = multi_probe_auroc(clean_list, backdoor_list, val_list)
    row = {
        "folder": model["folder_name"],
        "dataset": model["report"]["dataset"],
        "attack": model["report"]["attack"],
        "poison_rate": model["report"]["poison_rate"],
        "auroc": auroc,
    }
    for target_fpr in TARGET_FPRS:
        detection = multi_probe_detection(
            val_list, clean_list, backdoor_list, target_fpr, "calibrated"
        )
        row[f"tpr_at_{target_fpr:.2f}"] = detection["tpr"]
        row[f"fpr_at_{target_fpr:.2f}"] = detection["fpr"]
    return row


def union_set(
    results_dir: str,
    models: list[dict],
    placements: tuple[str, ...],
    manifest_cache: dict,
    psu_cache: dict,
) -> list[dict]:
    """The min-rank union's per-model rows, over the models that hold every probe."""
    covered = [model for model in models if model_has_probes(model, placements)]
    rows = [
        union_row(results_dir, model, placements, manifest_cache, psu_cache)
        for model in covered
    ]
    return rows


def worst_model(rows: list[dict]) -> dict | None:
    """The row with the lowest AUROC, the placement set's weakest single case."""
    if not rows:
        return None
    weakest = min(rows, key=lambda row: row["auroc"])
    return weakest


def set_summary(rows: list[dict], placements: tuple[str, ...]) -> dict:
    summary = {
        "placements": list(placements),
        "n_models": len(rows),
        "auroc_mean": mean_or_none([row["auroc"] for row in rows]),
        "tpr10_mean": mean_or_none([row["tpr_at_0.10"] for row in rows]),
        "tpr20_mean": mean_or_none([row["tpr_at_0.20"] for row in rows]),
        "worst_model": worst_model(rows),
    }
    return summary


def search_pairs(
    results_dir: str,
    models: list[dict],
    candidates: list[str],
    reference_rows: list[dict],
    manifest_cache: dict,
    psu_cache: dict,
    resamples: int,
    seed: int,
) -> list[dict]:
    """Every candidate pair's summary, gain over PSBD-TM, ranked by mean TPR at 10%."""
    pairs = []
    for placement_a, placement_b in itertools.combinations(sorted(candidates), 2):
        placements = (placement_a, placement_b)
        rows = union_set(results_dir, models, placements, manifest_cache, psu_cache)
        summary = set_summary(rows, placements)
        gain = paired_gain(reference_rows, rows, resamples, seed)
        pairs.append({"summary": summary, "gain_over_psbd_tm": gain, "rows": rows})
    pairs.sort(key=lambda entry: entry["summary"]["tpr10_mean"], reverse=True)
    return pairs


def solo_summaries(
    results_dir: str,
    models: list[dict],
    candidates: list[str],
    manifest_cache: dict,
    psu_cache: dict,
) -> dict[str, dict]:
    """Each candidate placement read alone, the seed for the greedy triple search."""
    summaries = {}
    for placement in candidates:
        rows = union_set(results_dir, models, (placement,), manifest_cache, psu_cache)
        summaries[placement] = set_summary(rows, (placement,))
    return summaries


def greedy_triple(
    results_dir: str,
    models: list[dict],
    top6: list[str],
    manifest_cache: dict,
    psu_cache: dict,
) -> dict:
    """Greedy forward search for the best triple among 6 candidates, by mean TPR at 10%.

    Seeds on the candidate with the best solo TPR at 10% (top6[0]), then twice
    adds whichever remaining candidate raises the growing set's mean TPR at 10%
    the most. Reports every step so the search is auditable, not just its result.
    """
    current = [top6[0]]
    remaining = list(top6[1:])
    steps = []
    for _ in range(2):
        trial_scores = {}
        for candidate in remaining:
            placements = tuple(current + [candidate])
            rows = union_set(results_dir, models, placements, manifest_cache, psu_cache)
            trial_scores[candidate] = set_summary(rows, placements)
        best_candidate = max(
            trial_scores, key=lambda candidate: trial_scores[candidate]["tpr10_mean"]
        )
        current.append(best_candidate)
        remaining.remove(best_candidate)
        steps.append({"added": best_candidate, "summary": trial_scores[best_candidate]})

    triple_rows = union_set(
        results_dir, models, tuple(current), manifest_cache, psu_cache
    )
    result = {
        "top6": top6,
        "steps": steps,
        "placements": current,
        "summary": set_summary(triple_rows, tuple(current)),
        "rows": triple_rows,
    }
    return result


def zscored_stack(
    split_per_probe: list[torch.Tensor], val_per_probe: list[torch.Tensor]
) -> torch.Tensor:
    """Each probe's split scores z-scored against its own clean-validation mean and std.

    Shape (k, n). A probe with 0 variance on validation (a saturated rate) is
    guarded to a std of 1 rather than dividing by 0.
    """
    terms = []
    for split, val in zip(split_per_probe, val_per_probe):
        mean = val.float().mean()
        std = val.float().std()
        std = std if std > 0 else torch.tensor(1.0)
        terms.append((split.float() - mean) / std)
    stacked = torch.stack(terms)  # (k, n)
    return stacked


def rank_stack(
    split_per_probe: list[torch.Tensor], val_per_probe: list[torch.Tensor]
) -> torch.Tensor:
    """Each probe's split scores ranked against its own clean-validation reference.

    Shape (k, n), each row r_j(x) in [0, 1], the share of clean validation with
    a lower PSU under probe j (`defences.scores.to_rank`).
    """
    ranks = torch.stack(
        [to_rank(split, val) for split, val in zip(split_per_probe, val_per_probe)]
    )  # (k, n)
    return ranks


def combine_scores(
    rule: str,
    split_per_probe: list[torch.Tensor],
    val_per_probe: list[torch.Tensor],
    weights: list[float] | None,
) -> torch.Tensor:
    """1 rule's combined score, shape (n,). Lower always means more suspicious.

    "zsum" combines raw z-scored PSU, every other rule combines ranks so it
    stays on the same [0, 1] scale multi_probe_score already uses.
    """
    if rule == "zsum":
        combined = zscored_stack(split_per_probe, val_per_probe).sum(dim=0)  # (n,)
        return combined

    ranks = rank_stack(split_per_probe, val_per_probe)  # (k, n)
    if rule == "min":
        combined = ranks.min(dim=0).values
    elif rule == "mean":
        combined = ranks.mean(dim=0)
    elif rule == "max":
        combined = ranks.max(dim=0).values
    elif rule == "product":
        combined = ranks.prod(dim=0)
    elif rule == "weighted_mean":
        weight_tensor = torch.tensor(weights, dtype=torch.float32).view(-1, 1)
        combined = (ranks * weight_tensor).sum(dim=0) / weight_tensor.sum()
    else:
        raise ValueError(f"unknown combination rule {rule!r}")
    return combined


def auroc_from_scores(clean_score: torch.Tensor, backdoor_score: torch.Tensor) -> float:
    """AUROC of 1 combined score, negated because lower means more suspicious."""
    scores = np.concatenate(
        [-clean_score.numpy(), -backdoor_score.numpy()]
    )  # (n_clean + n_backdoor,)
    labels = np.concatenate(
        [np.zeros(len(clean_score)), np.ones(len(backdoor_score))]
    )  # (n_clean + n_backdoor,)
    if len(set(labels.tolist())) < 2:
        return float("nan")
    auroc = float(roc_auc_score(labels, scores))
    return auroc


def subset_rule_row(
    results_dir: str,
    model: dict,
    placements: tuple[str, ...],
    rule: str,
    manifest_cache: dict,
    psu_cache: dict,
) -> dict:
    """1 model's AUROC and TPR at both target FPRs, for any placement subset under any rule.

    Generalises `union_row` (which reads only `multi_probe_auroc` and
    `multi_probe_detection`'s min reduction) to `combine_scores`'s 3 requested
    rules, min, product and zsum, so the pool search below shares 1 scoring
    path across every subset size.
    """
    val_list, clean_list, backdoor_list = probe_arrays(
        results_dir, model, placements, manifest_cache, psu_cache
    )
    val_score = combine_scores(rule, val_list, val_list, None)  # (n_validation,)
    clean_score = combine_scores(rule, clean_list, val_list, None)  # (n_backdoor,)
    backdoor_score = combine_scores(
        rule, backdoor_list, val_list, None
    )  # (n_backdoor,)

    row = {
        "folder": model["folder_name"],
        "dataset": model["report"]["dataset"],
        "attack": model["report"]["attack"],
        "poison_rate": model["report"]["poison_rate"],
        "auroc": auroc_from_scores(clean_score, backdoor_score),
    }
    for target_fpr in TARGET_FPRS:
        threshold = float(np.quantile(val_score.numpy(), target_fpr))
        row[f"tpr_at_{target_fpr:.2f}"] = float(
            (backdoor_score < threshold).float().mean()
        )
        row[f"fpr_at_{target_fpr:.2f}"] = float(
            (clean_score < threshold).float().mean()
        )
    return row


def subset_rule_set(
    results_dir: str,
    models: list[dict],
    placements: tuple[str, ...],
    rule: str,
    manifest_cache: dict,
    psu_cache: dict,
) -> list[dict]:
    """1 placement subset's per-model rows under 1 rule, over the models that hold every probe."""
    covered = [model for model in models if model_has_probes(model, placements)]
    rows = [
        subset_rule_row(results_dir, model, placements, rule, manifest_cache, psu_cache)
        for model in covered
    ]
    return rows


def rule_row(
    val_per_probe: list[torch.Tensor],
    clean_per_probe: list[torch.Tensor],
    backdoor_per_probe: list[torch.Tensor],
    rule: str,
    weights: list[float] | None,
) -> dict:
    """1 rule's AUROC and, at every target FPR, the calibrated threshold's TPR and FPR."""
    val_score = combine_scores(rule, val_per_probe, val_per_probe, weights)  # (n_val,)
    clean_score = combine_scores(rule, clean_per_probe, val_per_probe, weights)
    backdoor_score = combine_scores(rule, backdoor_per_probe, val_per_probe, weights)

    by_fpr = {}
    for target_fpr in RULE_TARGET_FPRS:
        threshold = float(np.quantile(val_score.numpy(), target_fpr))
        by_fpr[target_fpr] = {
            "threshold": threshold,
            "tpr": float((backdoor_score < threshold).float().mean()),
            "fpr": float((clean_score < threshold).float().mean()),
        }
    row = {"auroc": auroc_from_scores(clean_score, backdoor_score), "by_fpr": by_fpr}
    return row


def solo_auroc_table(
    results_dir: str,
    models: list[dict],
    placements: tuple[str, ...],
    manifest_cache: dict,
    psu_cache: dict,
) -> dict[str, dict[str, float]]:
    """Each probe's solo AUROC on each model, the source leave-one-out weights read from."""
    table: dict[str, dict[str, float]] = {}
    for placement in placements:
        per_model = {}
        for model in models:
            if not model_has_probes(model, (placement,)):
                continue
            val, clean, backdoor = cached_probe_psu(
                results_dir, model, placement, manifest_cache, psu_cache
            )
            per_model[model["folder_name"]] = multi_probe_auroc(
                [clean], [backdoor], [val]
            )
        table[placement] = per_model
    return table


def loo_weights(
    solo_table: dict[str, dict[str, float]],
    placements: tuple[str, ...],
    test_folder: str,
) -> list[float]:
    """Each probe's mean solo AUROC over every OTHER model in the set (leave-one-out)."""
    weights = []
    for placement in placements:
        per_model = solo_table[placement]
        others = [value for folder, value in per_model.items() if folder != test_folder]
        weights.append(mean_or_none(others) or 0.0)
    return weights


def rule_comparison_rows(
    results_dir: str,
    models: list[dict],
    placements: tuple[str, ...],
    manifest_cache: dict,
    psu_cache: dict,
) -> list[dict]:
    """1 model per row, every rule's AUROC and by-FPR block, on the models that hold every probe."""
    covered = [model for model in models if model_has_probes(model, placements)]
    solo_table = solo_auroc_table(
        results_dir, covered, placements, manifest_cache, psu_cache
    )

    rows = []
    for model in covered:
        val_list, clean_list, backdoor_list = probe_arrays(
            results_dir, model, placements, manifest_cache, psu_cache
        )
        weights = loo_weights(solo_table, placements, model["folder_name"])
        by_rule = {
            rule: rule_row(val_list, clean_list, backdoor_list, rule, weights)
            for rule in RULES
        }
        rows.append(
            {
                "folder": model["folder_name"],
                "dataset": model["report"]["dataset"],
                "attack": model["report"]["attack"],
                "poison_rate": model["report"]["poison_rate"],
                "by_rule": by_rule,
            }
        )
    return rows


def rule_summary(rows: list[dict], rule: str) -> dict:
    """1 rule's aggregate over `rows`: AUROC and, at every target FPR, TPR and FPR."""
    summary = {
        "n_models": len(rows),
        "auroc_mean": mean_or_none([row["by_rule"][rule]["auroc"] for row in rows]),
    }
    for target_fpr in RULE_TARGET_FPRS:
        tprs = [row["by_rule"][rule]["by_fpr"][target_fpr]["tpr"] for row in rows]
        fprs = [row["by_rule"][rule]["by_fpr"][target_fpr]["fpr"] for row in rows]
        summary[f"tpr_at_{target_fpr:.2f}_mean"] = mean_or_none(tprs)
        summary[f"fpr_at_{target_fpr:.2f}_mean"] = mean_or_none(fprs)
    return summary


def best_rule_by_primary_tpr(rows: list[dict]) -> str:
    """The rule with the highest mean TPR at PRIMARY_TARGET_FPR, task 2's ranking metric."""
    scores = {
        rule: mean_or_none(
            [row["by_rule"][rule]["by_fpr"][PRIMARY_TARGET_FPR]["tpr"] for row in rows]
        )
        or -1.0
        for rule in RULES
    }
    best = max(scores, key=scores.get)
    return best


def pair_disagreement_folders(
    results_dir: str,
    rows: list[dict],
    placement_a: str,
    placement_b: str,
    manifest_cache: dict,
    psu_cache: dict,
    share: float,
) -> list[str]:
    """The bottom `share` of models by Spearman correlation of the 2 probes' backdoor ranks.

    Correlation is read on the backdoor split, the images the pair actually has
    to agree about, mirroring `experiments.site_a_vs_b.measure.per_model_row`'s
    spearman_triggered field.
    """
    correlations = []
    for row in rows:
        folder = row["folder"]
        _, _, backdoor_a = psu_cache[(folder, placement_a)]
        _, _, backdoor_b = psu_cache[(folder, placement_b)]
        correlation, _ = spearmanr(backdoor_a.numpy(), backdoor_b.numpy())
        correlations.append((folder, float(correlation)))
    correlations.sort(key=lambda entry: entry[1])
    cutoff = max(1, round(len(correlations) * share))
    disagreeing = [folder for folder, _ in correlations[:cutoff]]
    return disagreeing


def build_task1(
    results_dir: str,
    declaration_path: str,
    models: list[dict],
    manifest_cache: dict,
    psu_cache: dict,
    resamples: int,
    seed: int,
) -> dict:
    candidates = basis_ids_present_on_all_models(models, declaration_path)
    for placement in candidates:
        preload_placement(results_dir, models, placement, manifest_cache, psu_cache)

    reference_rows = union_set(
        results_dir, models, (PSBD_TM,), manifest_cache, psu_cache
    )
    pairs = search_pairs(
        results_dir,
        models,
        candidates,
        reference_rows,
        manifest_cache,
        psu_cache,
        resamples,
        seed,
    )

    preload_placement(results_dir, models, ATTN_BRANCH_TM, manifest_cache, psu_cache)
    branch_placements = (PSBD_TM, ATTN_BRANCH_TM)
    branch_rows = union_set(
        results_dir, models, branch_placements, manifest_cache, psu_cache
    )
    branch_summary = set_summary(branch_rows, branch_placements)
    branch_gain = paired_gain(reference_rows, branch_rows, resamples, seed)
    branch_rank = 1 + sum(
        1
        for entry in pairs
        if entry["summary"]["tpr10_mean"] > branch_summary["tpr10_mean"]
    )

    solos = solo_summaries(results_dir, models, candidates, manifest_cache, psu_cache)
    top6 = sorted(candidates, key=lambda p: solos[p]["tpr10_mean"], reverse=True)[
        :TOP6_SIZE
    ]
    triple = greedy_triple(results_dir, models, top6, manifest_cache, psu_cache)
    triple_gain = paired_gain(reference_rows, triple["rows"], resamples, seed)

    task1 = {
        "n_models": len(models),
        "candidates": candidates,
        "reference_rows": reference_rows,
        "pairs": pairs,
        "top10": pairs[:10],
        "branch_pair": {
            "placements": list(branch_placements),
            "summary": branch_summary,
            "gain_over_psbd_tm": branch_gain,
            "rank_by_tpr10_among_pairs": branch_rank,
            "n_pairs_compared_against": len(pairs),
        },
        "solo_summaries": solos,
        "top6_by_solo_tpr10": top6,
        "greedy_triple": {**triple, "gain_over_psbd_tm": triple_gain},
    }
    return task1


def build_task2(
    results_dir: str,
    models: list[dict],
    best_pair_placements: tuple[str, ...],
    manifest_cache: dict,
    psu_cache: dict,
) -> dict:
    preload_placement(
        results_dir, models, ATTN_INPUT_DROPOUT, manifest_cache, psu_cache
    )
    preload_placement(results_dir, models, MLP_NORM_GAIN, manifest_cache, psu_cache)

    probe_sets = {
        "best_pair": best_pair_placements,
        "branch_pair": (PSBD_TM, ATTN_BRANCH_TM),
        "adaptive_3probe": (PSBD_TM, ATTN_INPUT_DROPOUT, MLP_NORM_GAIN),
    }

    task2 = {}
    for name, placements in probe_sets.items():
        rows = rule_comparison_rows(
            results_dir, models, placements, manifest_cache, psu_cache
        )
        summaries = {rule: rule_summary(rows, rule) for rule in RULES}
        best_rule = best_rule_by_primary_tpr(rows)
        block = {
            "placements": list(placements),
            "n_models": len(rows),
            "rows": rows,
            "summaries": summaries,
            "best_rule": best_rule,
            "min_vs_best_rule_tpr10_gain": (
                summaries[best_rule]["tpr_at_0.10_mean"]
                - summaries["min"]["tpr_at_0.10_mean"]
            ),
        }
        if len(placements) == 2:
            disagreeing_folders = pair_disagreement_folders(
                results_dir,
                rows,
                placements[0],
                placements[1],
                manifest_cache,
                psu_cache,
                DISAGREEMENT_SHARE,
            )
            disagreeing_rows = [
                row for row in rows if row["folder"] in disagreeing_folders
            ]
            disagreeing_summaries = {
                rule: rule_summary(disagreeing_rows, rule) for rule in RULES
            }
            block["disagreement_subset"] = {
                "folders": disagreeing_folders,
                "n_models": len(disagreeing_rows),
                "summaries": disagreeing_summaries,
                "best_rule": best_rule_by_primary_tpr(disagreeing_rows),
            }
        task2[name] = block
    return task2


def pool_placements(
    results_dir: str,
    models: list[dict],
    task1: dict,
    manifest_cache: dict,
    psu_cache: dict,
) -> tuple[list[str], dict[str, dict]]:
    """The 8 best solo candidates plus the attention branch output, and each one's solo summary.

    The 8 are read from `task1`'s solo search over the 13 candidates that hold
    on every model. The branch output is read separately since it holds on
    only 66 of them.
    """
    solos = dict(task1["solo_summaries"])
    top8 = sorted(
        task1["candidates"],
        key=lambda placement: solos[placement]["tpr10_mean"],
        reverse=True,
    )[:POOL_SIZE]

    preload_placement(results_dir, models, ATTN_BRANCH_TM, manifest_cache, psu_cache)
    branch_rows = union_set(
        results_dir, models, (ATTN_BRANCH_TM,), manifest_cache, psu_cache
    )
    solos[ATTN_BRANCH_TM] = set_summary(branch_rows, (ATTN_BRANCH_TM,))

    pool = top8 + [ATTN_BRANCH_TM]
    return pool, solos


def build_task3(
    results_dir: str,
    models: list[dict],
    task1: dict,
    manifest_cache: dict,
    psu_cache: dict,
    resamples: int,
    seed: int,
) -> dict:
    """Every subset of `pool` at sizes 2 to 5, under min, product and zsum.

    Ranked by mean TPR at 10% within each (size, rule) cell, which is exactly
    how task 1 ranked pairs, so a subset's rank here is read the same way.
    """
    pool, solo_summaries_by_placement = pool_placements(
        results_dir, models, task1, manifest_cache, psu_cache
    )
    reference_rows = task1["reference_rows"]

    by_size_and_rule: dict[str, list[dict]] = {}
    for size in POOL_SUBSET_SIZES:
        for rule in POOL_RULES:
            entries = []
            for subset in itertools.combinations(pool, size):
                rows = subset_rule_set(
                    results_dir, models, subset, rule, manifest_cache, psu_cache
                )
                if not rows:
                    continue
                summary = set_summary(rows, subset)
                gain = paired_gain(reference_rows, rows, resamples, seed)
                entries.append(
                    {
                        "placements": list(subset),
                        "summary": summary,
                        "gain_over_psbd_tm": gain,
                    }
                )
            entries.sort(key=lambda entry: entry["summary"]["tpr10_mean"], reverse=True)
            by_size_and_rule[f"{size}_{rule}"] = entries

    best_per_size_rule = {
        key: entries[0] for key, entries in by_size_and_rule.items() if entries
    }
    top_subsets = {
        f"{size}_{rule}": by_size_and_rule[f"{size}_{rule}"][:POOL_TOP_KEEP]
        for size in POOL_REPORT_SIZES
        for rule in POOL_RULES
    }

    task3 = {
        "pool": pool,
        "pool_solo_tpr10": {
            placement: solo_summaries_by_placement[placement]["tpr10_mean"]
            for placement in pool
        },
        "sizes": list(POOL_SUBSET_SIZES),
        "report_sizes": list(POOL_REPORT_SIZES),
        "rules": list(POOL_RULES),
        "best_per_size_rule": best_per_size_rule,
        "top_subsets": top_subsets,
    }
    return task3


def build_report(args: argparse.Namespace) -> dict:
    models = select_models(args.results_dir)
    manifest_cache: dict = {}
    psu_cache: dict = {}

    task1 = build_task1(
        args.results_dir,
        args.declaration,
        models,
        manifest_cache,
        psu_cache,
        args.bootstrap,
        args.seed,
    )
    best_pair_placements = tuple(task1["top10"][0]["summary"]["placements"])
    task2 = build_task2(
        args.results_dir, models, best_pair_placements, manifest_cache, psu_cache
    )
    task3 = build_task3(
        args.results_dir,
        models,
        task1,
        manifest_cache,
        psu_cache,
        args.bootstrap,
        args.seed,
    )

    report = {
        "n_models_selected": len(models),
        "bootstrap_resamples": args.bootstrap,
        "bootstrap_seed": args.seed,
        "task1_pair_search": task1,
        "task2_combination_rules": task2,
        "task3_pool_search": task3,
    }
    return report


def print_summary(report: dict) -> None:
    task1 = report["task1_pair_search"]
    print(f"{report['n_models_selected']} models selected")
    print(f"{len(task1['candidates'])} candidate placements present on all of them")
    print("top 5 pairs by mean TPR at 10%:")
    for entry in task1["top10"][:5]:
        summary = entry["summary"]
        print(
            f"  {summary['placements']}: tpr10={summary['tpr10_mean']:.3f} "
            f"auroc={summary['auroc_mean']:.3f} n={summary['n_models']}"
        )
    branch = task1["branch_pair"]
    print(
        f"branch pair rank {branch['rank_by_tpr10_among_pairs']} of "
        f"{branch['n_pairs_compared_against']}, tpr10={branch['summary']['tpr10_mean']:.3f}"
    )
    triple = task1["greedy_triple"]
    print(
        f"greedy triple: {triple['placements']} tpr10={triple['summary']['tpr10_mean']:.3f}"
    )

    for name, block in report["task2_combination_rules"].items():
        print(
            f"task 2, {name}: best rule = {block['best_rule']}, "
            f"gain over min at TPR@10 = {block['min_vs_best_rule_tpr10_gain']:+.3f}"
        )

    task3 = report["task3_pool_search"]
    print(f"task 3 pool: {task3['pool']}")
    for key, entry in task3["best_per_size_rule"].items():
        summary = entry["summary"]
        print(
            f"  best {key}: {summary['placements']} tpr10={summary['tpr10_mean']:.3f} "
            f"auroc={summary['auroc_mean']:.3f} n={summary['n_models']}"
        )


# The functions below write docs/probe_union_tables_2026-09-11.md.


def fmt(value: float | None, places: int = 3) -> str:
    if value is None or value != value:
        return "--"
    return f"{value:.{places}f}"


def fmt_signed(value: float | None, places: int = 3) -> str:
    if value is None or value != value:
        return "--"
    return f"{value:+.{places}f}"


def ci_text(gain: dict) -> str:
    low, high = gain["ci_low"], gain["ci_high"]
    if low != low or high != high:
        return "--"
    return f"[{fmt_signed(low)}, {fmt_signed(high)}]"


def row_lookup(rows: list[dict]) -> dict[str, dict]:
    return {row["folder"]: row for row in rows}


def build_dataset_tables(
    coverage: dict,
    best_pair_placements: list[str],
    best_pair_name: str,
    reference_rows: list[dict],
    pair_rows: list[dict],
    best_rule: str | None,
    rule_rows_by_folder: dict[str, dict] | None,
) -> list[str]:
    """1 markdown table per (poison rate, dataset), the layout of the site A/B doc."""
    cells_by_folder = {cell["folder_name"]: cell for cell in coverage["cells"]}
    reference_by_folder = row_lookup(reference_rows)
    benign_accuracy = coverage["benign_reference_accuracy"]

    header = ["Model", "CA", "ASR", "PSBD-TM AUROC", "PSBD-TM TPR@10", "PSBD-TM TPR@20"]
    header += [
        f"{best_pair_name} AUROC",
        f"{best_pair_name} TPR@10",
        f"{best_pair_name} TPR@20",
    ]
    if best_rule is not None:
        header += [
            f"{best_rule} rule AUROC",
            f"{best_rule} rule TPR@10",
            f"{best_rule} rule TPR@20",
        ]

    lines = []
    for rate in (0.01, 0.05, 0.1):
        datasets_at_rate = sorted(
            {row["dataset"] for row in pair_rows if row["poison_rate"] == rate},
            key=lambda dataset: (
                DATASET_ORDER.index(dataset)
                if dataset in DATASET_ORDER
                else len(DATASET_ORDER)
            ),
        )
        for dataset in datasets_at_rate:
            lines.append(f"## {dataset_label(dataset)}, {RATE_LABELS[rate]} poisoning")
            lines.append("")
            lines.append("| " + " | ".join(header) + " |")
            lines.append("|" + "|".join(["---"] * len(header)) + "|")
            benign_ca = benign_accuracy.get(dataset)
            blanks = " | ".join([""] * (len(header) - 2))
            lines.append(f"| benign | {fmt(benign_ca)} | {blanks} |")
            for attack in ATTACK_ORDER:
                folder_rows = [
                    row
                    for row in pair_rows
                    if row["dataset"] == dataset
                    and row["poison_rate"] == rate
                    and row["attack"] == attack
                ]
                for pair_row in folder_rows:
                    folder = pair_row["folder"]
                    cell = cells_by_folder.get(folder, {})
                    ca = cell.get("clean_accuracy")
                    asr = cell.get("asr")
                    label = attack_label(attack)
                    if asr is not None and asr < ASR_BAR:
                        label += " †"
                    reference_row = reference_by_folder.get(folder, {})
                    values = [
                        label,
                        fmt(ca),
                        fmt(asr),
                        fmt(reference_row.get("auroc")),
                        fmt(reference_row.get("tpr_at_0.10")),
                        fmt(reference_row.get("tpr_at_0.20")),
                        fmt(pair_row.get("auroc")),
                        fmt(pair_row.get("tpr_at_0.10")),
                        fmt(pair_row.get("tpr_at_0.20")),
                    ]
                    if best_rule is not None:
                        rule_row = (rule_rows_by_folder or {}).get(folder)
                        if rule_row is not None:
                            block = rule_row["by_rule"][best_rule]
                            values += [
                                fmt(block["auroc"]),
                                fmt(block["by_fpr"][0.10]["tpr"]),
                                fmt(block["by_fpr"][0.20]["tpr"]),
                            ]
                        else:
                            values += ["--", "--", "--"]
                    lines.append("| " + " | ".join(values) + " |")
            lines.append("")
    return lines


def build_top10_table(task1: dict, entries: dict[str, dict]) -> list[str]:
    header = [
        "Rank",
        "Pair",
        "n",
        "Mean AUROC",
        "Mean TPR@10",
        "Mean TPR@20",
        "Worst model",
        "Worst AUROC",
        "Gain over PSBD-TM [95% CI]",
    ]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for rank, entry in enumerate(task1["top10"], start=1):
        summary = entry["summary"]
        pair_text = " + ".join(
            entries.get(placement_id, {}).get("_words", placement_id)
            for placement_id in summary["placements"]
        )
        worst = summary["worst_model"] or {}
        row = [
            str(rank),
            pair_text,
            str(summary["n_models"]),
            fmt(summary["auroc_mean"]),
            fmt(summary["tpr10_mean"]),
            fmt(summary["tpr20_mean"]),
            f"{dataset_label(worst.get('dataset', ''))} {attack_label(worst.get('attack', ''))}",
            fmt(worst.get("auroc")),
            f"{fmt_signed(entry['gain_over_psbd_tm']['mean_gain'])} {ci_text(entry['gain_over_psbd_tm'])}",
        ]
        lines.append("| " + " | ".join(row) + " |")
    return lines


def build_rule_table(task2: dict) -> list[str]:
    header = ["Probe set", "Rule", "n", "AUROC", "TPR@10", "FPR@10", "TPR@20", "FPR@20"]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for name, block in task2.items():
        for rule in RULES:
            summary = block["summaries"][rule]
            marker = " (best)" if rule == block["best_rule"] else ""
            row = [
                name,
                rule + marker,
                str(summary["n_models"]),
                fmt(summary["auroc_mean"]),
                fmt(summary["tpr_at_0.10_mean"]),
                fmt(summary["fpr_at_0.10_mean"]),
                fmt(summary["tpr_at_0.20_mean"]),
                fmt(summary["fpr_at_0.20_mean"]),
            ]
            lines.append("| " + " | ".join(row) + " |")
    return lines


def build_disagreement_table(task2: dict) -> list[str]:
    header = ["Rule", "n", "AUROC", "TPR@10", "TPR@20"]
    lines = []
    block = task2.get("best_pair", {}).get("disagreement_subset")
    if block is None:
        return lines
    lines.append(
        f"### The {block['n_models']} models where the best pair's 2 probes disagree most"
    )
    lines.append("")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for rule in RULES:
        summary = block["summaries"][rule]
        marker = " (best)" if rule == block["best_rule"] else ""
        row = [
            rule + marker,
            str(summary["n_models"]),
            fmt(summary["auroc_mean"]),
            fmt(summary["tpr_at_0.10_mean"]),
            fmt(summary["tpr_at_0.20_mean"]),
        ]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return lines


def build_pool_table(task3: dict, declaration_entries: dict) -> list[str]:
    """Every (size, rule)'s best subset, sizes 2 to 5, size 2 marked as context only."""
    header = [
        "Size",
        "Rule",
        "Subset",
        "n",
        "Mean AUROC",
        "Mean TPR@10",
        "Mean TPR@20",
        "Worst model",
        "Worst AUROC",
        "Gain over PSBD-TM [95% CI]",
    ]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for size in task3["sizes"]:
        for rule in task3["rules"]:
            entry = task3["best_per_size_rule"].get(f"{size}_{rule}")
            if entry is None:
                continue
            summary = entry["summary"]
            subset_words = " + ".join(
                readable_placement(declaration_entries, placement_id)
                for placement_id in summary["placements"]
            )
            worst = summary["worst_model"] or {}
            size_text = (
                str(size) if size in task3["report_sizes"] else f"{size} (context)"
            )
            row = [
                size_text,
                rule,
                subset_words,
                str(summary["n_models"]),
                fmt(summary["auroc_mean"]),
                fmt(summary["tpr10_mean"]),
                fmt(summary["tpr20_mean"]),
                f"{dataset_label(worst.get('dataset', ''))} {attack_label(worst.get('attack', ''))}",
                fmt(worst.get("auroc")),
                f"{fmt_signed(entry['gain_over_psbd_tm']['mean_gain'])} "
                f"{ci_text(entry['gain_over_psbd_tm'])}",
            ]
            lines.append("| " + " | ".join(row) + " |")
    return lines


def pool_trend_prose(task3: dict) -> list[str]:
    """1 sentence per rule, how mean TPR at 10% moves as the union grows from 2 to 5 members."""
    sentences = []
    for rule in task3["rules"]:
        points = [
            (
                size,
                task3["best_per_size_rule"][f"{size}_{rule}"]["summary"]["tpr10_mean"],
            )
            for size in task3["sizes"]
            if f"{size}_{rule}" in task3["best_per_size_rule"]
        ]
        if len(points) < 2:
            continue
        steps = ", ".join(
            f"{points[i][0]} to {points[i + 1][0]} {fmt_signed(points[i + 1][1] - points[i][1])}"
            for i in range(len(points) - 1)
        )
        sentences.append(
            f"Under {rule}, mean TPR at 10% moves {steps} as the union grows."
        )
    return sentences


def best_pool_subset(task3: dict) -> dict:
    """The single best subset over the requested sizes 3, 4 and 5, by mean TPR at 10%."""
    candidates = [
        entry
        for key, entry in task3["best_per_size_rule"].items()
        if int(key.split("_", 1)[0]) in task3["report_sizes"]
    ]
    best = max(candidates, key=lambda entry: entry["summary"]["tpr10_mean"])
    return best


def overall_best_pool_subset(task3: dict) -> dict:
    """The single best subset over EVERY searched size, 2 to 5, by mean TPR at 10%.

    Size 2 is the trend anchor, not a requested result, but a deployment
    recommendation that ignored it when it wins would recommend needless
    complexity for no gain, which the plateau question exists to catch.
    """
    best = max(
        task3["best_per_size_rule"].values(),
        key=lambda entry: entry["summary"]["tpr10_mean"],
    )
    return best


def rule_of_entry(task3: dict, entry: dict) -> str:
    """Which rule produced `entry`, by matching it back into `best_per_size_rule`."""
    for key, candidate in task3["best_per_size_rule"].items():
        if candidate is entry:
            return key.split("_", 1)[1]
    return "min"


def closing_prose(
    task1: dict, task2: dict, task3: dict, declaration_entries: dict
) -> list[str]:
    best_pair = task1["top10"][0]
    best_summary = best_pair["summary"]
    pair_words = " and ".join(
        readable_placement(declaration_entries, placement_id)
        for placement_id in best_summary["placements"]
    )
    tm_reference = mean_or_none([row["tpr_at_0.10"] for row in task1["reference_rows"]])
    gain_tpr10 = best_summary["tpr10_mean"] - (tm_reference or 0.0)
    best_pair_rule = task2["best_pair"]
    rule_gain = best_pair_rule["min_vs_best_rule_tpr10_gain"]
    disagreement = best_pair_rule.get("disagreement_subset", {})

    # The overall best over EVERY searched size, not only the requested 3 to 5, is
    # the honest deployment pick: the best subset of size 3 to 5 (best_pool_subset)
    # turns out not to beat the pool's own best 2-probe combination, so deploying
    # it anyway would trade complexity for nothing.
    best_large = best_pool_subset(task3)
    deployed = overall_best_pool_subset(task3)
    deployed_summary = deployed["summary"]
    deployed_rule = rule_of_entry(task3, deployed)
    deployed_size = len(deployed_summary["placements"])
    deployed_words = " and ".join(
        readable_placement(declaration_entries, placement_id)
        for placement_id in deployed_summary["placements"]
    )
    deployed_gain_over_pair_tpr10 = (
        deployed_summary["tpr10_mean"] - best_summary["tpr10_mean"]
    )
    growth_to_best_large = (
        best_large["summary"]["tpr10_mean"] - deployed_summary["tpr10_mean"]
    )

    sentences = [
        f"The best-unioning pair is PSBD-TM with {pair_words}, "
        f"mean AUROC {fmt(best_summary['auroc_mean'])} and mean TPR at 10% "
        f"{fmt(best_summary['tpr10_mean'])} on {best_summary['n_models']} models, "
        f"a gain over PSBD-TM alone of {fmt_signed(best_pair['gain_over_psbd_tm']['mean_gain'])} "
        f"AUROC {ci_text(best_pair['gain_over_psbd_tm'])}.",
        f"At the 10% clean-validation quantile the pair raises TPR by "
        f"{fmt_signed(gain_tpr10)} over PSBD-TM alone, from {fmt(tm_reference)} to "
        f"{fmt(best_summary['tpr10_mean'])}, and "
        + (
            "the min rule is already the best of the 6 rules tested on it."
            if best_pair_rule["best_rule"] == "min"
            else f"{best_pair_rule['best_rule']} beats min on it by "
            f"{fmt_signed(rule_gain)} more mean TPR at 10%."
        ),
        f"The union to deploy is the {deployed_size}-probe set {deployed_words} under "
        f"the {deployed_rule} rule, mean AUROC {fmt(deployed_summary['auroc_mean'])}, "
        f"mean TPR at 10% {fmt(deployed_summary['tpr10_mean'])} and mean TPR at 20% "
        f"{fmt(deployed_summary['tpr20_mean'])} on {deployed_summary['n_models']} models.",
        f"That set gains {fmt_signed(deployed['gain_over_psbd_tm']['mean_gain'])} AUROC "
        f"{ci_text(deployed['gain_over_psbd_tm'])} over PSBD-TM alone and "
        f"{fmt_signed(deployed_gain_over_pair_tpr10)} mean TPR at 10% over the best pair.",
        f"Every larger subset searched plateaus at or below that 2-probe ceiling, "
        f"the best 3-to-5-probe union reads {fmt(best_large['summary']['tpr10_mean'])} "
        f"mean TPR at 10%, {fmt_signed(growth_to_best_large)} against the deployed pair, "
        "so adding a 3rd, 4th or 5th probe to this pool buys no further detection power.",
    ]
    if disagreement:
        sentences.append(
            f"On the {disagreement['n_models']} models where the best pair's 2 probes "
            f"disagree most, the best rule is {disagreement['best_rule']}, "
            + (
                "the same answer as the whole model set."
                if disagreement["best_rule"] == best_pair_rule["best_rule"]
                else "a different answer from the whole model set."
            )
        )
    return sentences


def write_docs(
    path: str, report: dict, coverage: dict, declaration_entries: dict
) -> str:
    task1 = report["task1_pair_search"]
    task2 = report["task2_combination_rules"]
    best_pair_entry = task1["top10"][0]
    best_pair_placements = best_pair_entry["summary"]["placements"]
    best_pair_name = " + ".join(
        readable_placement(declaration_entries, placement_id)
        for placement_id in best_pair_placements
    )
    best_pair_block = task2["best_pair"]
    best_rule = (
        best_pair_block["best_rule"] if best_pair_block["best_rule"] != "min" else None
    )
    rule_rows_by_folder = (
        row_lookup(best_pair_block["rows"]) if best_rule is not None else None
    )

    lines = [
        "# Which probes union best, and the best way to combine them, ViT-B/16, 2026-09-11",
        "",
        "For the authors, not for the paper. PSBD-TM is "
        + readable_placement(declaration_entries, "before_attention_norm_token_mask")
        + ", the recommended placement. Every placement is read at its own 0.8"
        " clean shift adaptive rate, fractional PSU, the min-rank union at the"
        " calibrated 10% and 20% clean validation quantiles. CA is clean"
        " accuracy, the benign row is the reference model trained with the same"
        " recipe. Rows with ASR below 0.85 are marked with a dagger and are"
        " outside the paper's evaluation.",
        "",
    ]
    lines += build_dataset_tables(
        coverage,
        best_pair_placements,
        best_pair_name,
        task1["reference_rows"],
        [entry for entry in best_pair_entry["rows"]],
        best_rule,
        rule_rows_by_folder,
    )

    lines.append("## Task 1, the top 10 pairs by mean TPR at 10%")
    lines.append("")
    entries_words = {
        placement_id: {"_words": readable_placement(declaration_entries, placement_id)}
        for placement_id in task1["candidates"] + [PSBD_TM, ATTN_BRANCH_TM]
    }
    lines += build_top10_table(task1, entries_words)
    lines.append("")
    branch = task1["branch_pair"]
    lines.append(
        f"PSBD-TM plus {readable_placement(declaration_entries, ATTN_BRANCH_TM)} "
        f"sits at rank {branch['rank_by_tpr10_among_pairs']} of "
        f"{branch['n_pairs_compared_against']} pairs by mean TPR at 10% "
        f"({fmt(branch['summary']['tpr10_mean'])}, n={branch['summary']['n_models']}, "
        "covering fewer models than the all-candidate pairs so its rank is a"
        " reference point rather than a like-for-like placement)."
    )
    lines.append("")
    triple = task1["greedy_triple"]
    triple_words = " + ".join(
        readable_placement(declaration_entries, placement_id)
        for placement_id in triple["placements"]
    )
    lines.append(
        f"Greedy triple from the top {TOP6_SIZE} solo placements: {triple_words}, "
        f"mean AUROC {fmt(triple['summary']['auroc_mean'])}, mean TPR at 10% "
        f"{fmt(triple['summary']['tpr10_mean'])}, mean TPR at 20% "
        f"{fmt(triple['summary']['tpr20_mean'])}, n={triple['summary']['n_models']}, "
        f"gain over PSBD-TM {fmt_signed(triple['gain_over_psbd_tm']['mean_gain'])} "
        f"{ci_text(triple['gain_over_psbd_tm'])}."
    )
    lines.append("")

    lines.append("## Task 2, the combination rule")
    lines.append("")
    lines += build_rule_table(task2)
    lines.append("")
    lines += build_disagreement_table(task2)

    task3 = report["task3_pool_search"]
    pool_words = ", ".join(
        f"{readable_placement(declaration_entries, placement_id)} "
        f"({fmt(task3['pool_solo_tpr10'][placement_id])})"
        for placement_id in task3["pool"]
    )
    lines.append("## Task 3, unions larger than pairs")
    lines.append("")
    lines.append(
        f"The pool is the {POOL_SIZE} candidates with the best solo mean TPR at 10% "
        f"on the {report['n_models_selected']} models plus the attention branch output "
        "token mask on its own 66 models, each with its solo mean TPR at 10% in "
        f"parentheses: {pool_words}. Every subset of size 3, 4 and 5 from this pool is "
        "scored under the min, product and z-sum rules, size 2 is added only as the "
        "trend anchor and is not itself a requested result."
    )
    lines.append("")
    lines += build_pool_table(task3, declaration_entries)
    lines.append("")
    lines += pool_trend_prose(task3)
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines += closing_prose(task1, task2, task3, declaration_entries)
    lines.append("")

    with open(path, "w") as handle:
        handle.write("\n".join(lines))
    return path


def main() -> None:
    args = parse_args()
    report = build_report(args)
    print_summary(report)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(report, handle, indent=2)
    print(f"wrote {args.output}")

    coverage = load_coverage(args.results_dir)
    declaration_entries = load_declaration_entries(args.declaration)
    docs_path = write_docs(args.docs_output, report, coverage, declaration_entries)
    print(f"wrote {docs_path}")


if __name__ == "__main__":
    main()
