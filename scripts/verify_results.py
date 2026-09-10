"""Dump every AUROC, TPR, FPR and threshold on disk, so 2 states of the results tree can
be diffed against each other.

Reads every tracked results/*/psbd_metrics.json and flattens each placement's oracle and
adaptive report into 1 entry per (folder, placement, mode). It reads disk only, so it
proves nothing about which code produced those numbers: it can show that 2 result trees
agree or disagree, never why. Run it before and after a rerun, a rebase or a metrics
change, and diff the 2 outputs.

    python scripts/verify_results.py > /tmp/before.json
    python scripts/verify_results.py > /tmp/after.json
    diff /tmp/before.json /tmp/after.json
"""

import json
import sys
from pathlib import Path


RESULTS_DIR = Path("results")


def collect_all():
    """Every (folder, placement, mode) entry on disk, flattened into 1 list."""
    summaries = []
    for path in sorted(RESULTS_DIR.glob("*/psbd_metrics.json")):
        with open(path) as f:
            data = json.load(f)

        folder = data.get("folder_name", path.parent.name)
        placements = data.get("placements", {})

        for pname, placement in sorted(placements.items()):
            for mode in ("oracle", "adaptive"):
                report = placement.get(mode)
                if report is None:
                    continue
                summaries.append(
                    {
                        "folder": folder,
                        "placement": pname,
                        "mode": mode,
                        "auroc": round(report["auroc"], 6),
                        "tpr": round(report["tpr"], 6),
                        "fpr": round(report["fpr"], 6),
                        "threshold": round(report["threshold"], 6),
                    }
                )

    return summaries


def main():
    summaries = collect_all()
    json.dump(
        {"n_entries": len(summaries), "entries": summaries},
        sys.stdout,
        indent=2,
        sort_keys=True,
    )
    print()
    print(f"# {len(summaries)} entries from {RESULTS_DIR}", file=sys.stderr)


if __name__ == "__main__":
    main()
