"""Cross-branch result verification.

Reads all tracked psbd_metrics.json files and computes aggregate statistics.
Run on both branches and diff the output to verify they produce identical
numbers.

Usage:
    git checkout exp/psbd-dropout-position
    python scripts/verify_results.py > /tmp/branch_a.json

    git checkout publication-ready
    python scripts/verify_results.py > /tmp/branch_b.json

    diff /tmp/branch_a.json /tmp/branch_b.json
"""

import json
import sys
from pathlib import Path


RESULTS_DIR = Path("results")


def collect_all():
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
