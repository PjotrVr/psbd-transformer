"""Read the per-checkpoint reports and answer the 4 questions they were built for.

  1. WHICH LAYERS carry each attack, by relative backdoor-direction norm, TAC, and
     the clean-versus-triggered CKA drop.
  2. WHICH DIMENSIONS are the backdoor dimensions at the peak layer, and how few of
     them there are.
  3. DO THEY OVERLAP across attacks, by Jaccard of the top-k TAC sets. A shared set
     would mean one defence could cover several triggers at once.
  4. DOES SAM MOVE THEM, by Jaccard between the Adam checkpoint and the same attack
     trained with SAM at each rho, plus how far the peak layer travels.

Also reports the 2 data-free channels, which are the ones that would be cheap to
deploy: the Lipschitz-versus-TAC rank correlation, and the head-alignment Z rule.

Example
    PYTHONPATH=. python experiments/backdoor_neurons/report.py
"""

import argparse
import glob
import json
import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def load_reports(results_dir: str) -> list[dict]:
    reports = []
    for path in sorted(
        glob.glob(os.path.join(results_dir, "*", "backdoor_neurons.json"))
    ):
        with open(path) as handle:
            reports.append(json.load(handle))
    return reports


def jaccard(a: list[int], b: list[int]) -> float:
    left, right = set(a), set(b)
    return len(left & right) / max(len(left | right), 1)


def rho_label(report: dict) -> str:
    rho = report.get("rho")
    return "adam" if not rho else f"sam {float(rho):g}"


def key(report: dict) -> tuple[str, str]:
    return report["attack"], rho_label(report)


def peak_dims(report: dict, top_k: int) -> list[int]:
    """Top-k TAC dimensions at the layer where the backdoor direction is largest."""
    row = next(r for r in report["layers"] if r["layer"] == report["peak_layer"])
    return row["top_dims"][:top_k]


def print_layer_profile(reports: list[dict]) -> None:
    print("\n1. WHICH LAYERS. Relative backdoor-direction norm per block, Adam only.")
    print("   The direction norm is divided by the mean clean CLS norm at that layer,")
    print("   so growth is real and not just the residual stream getting bigger.\n")
    adam = [r for r in reports if rho_label(r) == "adam"]
    header = f"{'attack':14} " + " ".join(f"{i:>5}" for i in range(1, 13)) + "   peak"
    print(header)
    print("-" * len(header))
    for report in sorted(adam, key=lambda r: r["attack"]):
        values = {row["layer"]: row["rel_direction_norm"] for row in report["layers"]}
        cells = " ".join(f"{values.get(i, 0):>5.2f}" for i in range(1, 13))
        print(f"{report['attack']:14} {cells}   {report['peak_layer']:>4}")

    print("\n   CKA between clean and triggered features (1.00 = trigger invisible):\n")
    print(header)
    print("-" * len(header))
    for report in sorted(adam, key=lambda r: r["attack"]):
        values = {row["layer"]: row["cka"] for row in report["layers"]}
        cells = " ".join(f"{values.get(i, 0):>5.2f}" for i in range(1, 13))
        print(f"{report['attack']:14} {cells}   {report['peak_layer']:>4}")


def print_dimension_profile(reports: list[dict], top_k: int) -> None:
    print("\n\n2. WHICH NEURONS. At each attack's own peak layer, out of 768 dims.\n")
    print(
        f"{'attack':14} {'rho':>8} {'peak':>5} {'outliers':>9} {'tac_max':>9} "
        f"{'tac_mean':>9} {'ratio':>7}"
    )
    print("-" * 66)
    for report in sorted(reports, key=lambda r: (r["attack"], rho_label(r))):
        row = next(r for r in report["layers"] if r["layer"] == report["peak_layer"])
        ratio = row["tac_max"] / max(row["tac_mean"], 1e-9)
        print(
            f"{report['attack']:14} {rho_label(report):>8} {report['peak_layer']:>5} "
            f"{row['n_outlier_dims']:>9} {row['tac_max']:>9.3f} "
            f"{row['tac_mean']:>9.3f} {ratio:>7.1f}"
        )


def print_cross_attack_overlap(reports: list[dict], top_k: int) -> None:
    print(f"\n\n3. DO ATTACKS SHARE NEURONS. Jaccard of top-{top_k} TAC dims at each")
    print("   attack's peak layer, Adam only. 0.00 means fully disjoint.\n")
    adam = sorted(
        [r for r in reports if rho_label(r) == "adam"], key=lambda r: r["attack"]
    )
    names = [r["attack"] for r in adam]
    print(f"{'':14} " + " ".join(f"{n[:9]:>9}" for n in names))
    for report in adam:
        cells = " ".join(
            f"{jaccard(peak_dims(report, top_k), peak_dims(other, top_k)):>9.2f}"
            for other in adam
        )
        print(f"{report['attack']:14} {cells}")


def print_sam_effect(reports: list[dict], top_k: int) -> None:
    print(f"\n\n4. DOES SAM MOVE THEM. Jaccard of top-{top_k} dims against the same")
    print("   attack trained with Adam, and how far the peak layer travels.\n")
    by_key = {key(r): r for r in reports}
    attacks = sorted({r["attack"] for r in reports})
    print(
        f"{'attack':14} {'peak adam':>10} {'peak 0.1':>9} {'peak 0.2':>9} "
        f"{'J(0.1)':>8} {'J(0.2)':>8}"
    )
    print("-" * 62)
    for attack in attacks:
        base = by_key.get((attack, "adam"))
        if base is None:
            continue
        cells = []
        for rho in ("sam 0.1", "sam 0.2"):
            other = by_key.get((attack, rho))
            cells.append(
                (
                    other["peak_layer"],
                    jaccard(peak_dims(base, top_k), peak_dims(other, top_k)),
                )
                if other
                else (0, float("nan"))
            )
        print(
            f"{attack:14} {base['peak_layer']:>10} {cells[0][0]:>9} {cells[1][0]:>9} "
            f"{cells[0][1]:>8.2f} {cells[1][1]:>8.2f}"
        )


def print_data_free(reports: list[dict]) -> None:
    print("\n\n5. THE 2 DATA-FREE CHANNELS, which are the deployable ones.\n")
    print(
        f"{'attack':14} {'rho':>8} {'lips-TAC r':>11} {'Z':>7} {'flag':>6} "
        f"{'target':>7} {'pca_pur':>8} {'umap_pur':>9}"
    )
    print("-" * 74)
    for report in sorted(reports, key=lambda r: (r["attack"], rho_label(r))):
        row = next(r for r in report["layers"] if r["layer"] == report["peak_layer"])
        head = report["head_alignment"]
        umap = report["projections"].get("umap", {}).get("knn_purity")
        print(
            f"{report['attack']:14} {rho_label(report):>8} "
            f"{row.get('lipschitz_tac_spearman', float('nan')):>11.3f} "
            f"{head['z']:>7.2f} {'YES' if head['flagged'] else 'no':>6} "
            f"{'hit' if head['correct'] else 'miss':>7} "
            f"{report['projections']['pca']['knn_purity']:>8.3f} "
            f"{umap if umap is None else f'{umap:>9.3f}'}"
        )


def main() -> None:
    args = parse_args()
    reports = load_reports(args.results_dir)
    if not reports:
        print("no backdoor_neurons.json found")
        return
    print(f"{len(reports)} checkpoints, {reports[0]['samples']} paired samples each.")
    print_layer_profile(reports)
    print_dimension_profile(reports, args.top_k)
    print_cross_attack_overlap(reports, args.top_k)
    print_sam_effect(reports, args.top_k)
    print_data_free(reports)


if __name__ == "__main__":
    main()
