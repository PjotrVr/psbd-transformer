# Adaptive attack analysis (H25)

Analyzes evasive checkpoints trained with the hinge penalty from
`adaptive_evasion.py`. For each evasive checkpoint, computes sigma-matched
AUROC at 4 perturbation operators and compares to the non-evasive baseline.

Answers: does evasion against one operator transfer to others?

Result: evasion is probe-specific. The probed operator's AUROC collapses from
0.952 to 0.322, but transfer operators still detect at mean AUROC 0.887.

Runs on CPU using cached PSBD sweep data. No GPU needed.

    python experiments/adaptive_attack/analyze.py
    python experiments/adaptive_attack/analyze.py --architecture vit --dataset cifar100

Hypothesis doc: `docs/hypothesis/H25-adaptive-attacker.md`
