# Adaptive defender diagnostics (H42)

Comprehensive diagnostics for evasive models: per-operator validation PSU
stats, sigma-curve suppression, forensic evasion identification, and
multi-probe recovery.

Answers: can the defender identify which operator the attacker evaded?

Result: REFUTED. Defender-legal identification accuracy is 8.9% (validation
PSU std) and 3.6% (validation PSU IQR), both below the 25% chance level.
This does not matter: multi-probe with all operators (0.951) is within 0.006
of oracle-exclude (0.957). Forensic exclude by val_std actually hurts (0.847).

The protocol: use all operators, do not try to identify the evaded one.

Runs on CPU using cached PSBD sweep data. No GPU needed.

    python scripts/adaptive_defender/analyze.py
    python scripts/adaptive_defender/analyze.py --architecture vit --dataset cifar100

Hypothesis doc: `docs/hypothesis/H42-evasion-identification.md`
Protocol doc: `docs/results/adaptive-defender-protocol.md`
