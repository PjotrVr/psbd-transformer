# Multi-probe PSBD defence (H41)

Computes the min-rank combined score across all available perturbation
operators for each evasive checkpoint. The min-rank rule flags a sample when
ANY probe's PSU rank falls below the Bonferroni-corrected threshold.

Answers: can multi-probe PSBD recover detection after the adaptive attacker
collapses a single operator?

Result: multi-probe with 4 operators recovers AUROC from 0.322 (evaded) to
0.951, with 48/56 checkpoints above 0.90. Diminishing returns at k=3.

Runs on CPU using cached PSBD sweep data. No GPU needed.

    python experiments/multi_probe/analyze.py
    python experiments/multi_probe/analyze.py --architecture vit --dataset cifar100

Hypothesis doc: `docs/hypothesis/H41-multi-probe-defence.md`
