# Rank averaging across placements (H17, H18, H41)

## Question

No single placement wins on every attack. The patch trigger wants
`before_attention_norm`, while `wanet` and `lc` want `post_residual`. A defender
cannot know which attack they face, so committing to 1 placement is a bet.
Averaging each sample's PSU rank across placements is the defender-legal
alternative, and this measures whether it pays.

Each placement contributes at its own sigma-selected rate, so the members are
combined at matched disturbance rather than at a shared rate. Every split is ranked
against the clean-validation distribution of the same placement, which is the only
distribution a defender holds. An earlier version ranked jointly over
`torch.cat([clean, backdoor])`, which let the backdoor split help set the scale it
was then scored on and made the ensemble AUROC transductive.

## Running it

    PYTHONPATH=. python experiments/detector_ensemble/ensemble.py

CPU only, reading `scratch/surface.json` for the sigma-matched rates and the cached
per-pass probabilities for the ranks.

## Finding

Mean-rank averaging fails. Over 17 backdoored non-all-to-all CIFAR-10 checkpoints
the ensemble reads 0.897 against 0.892 for its best fixed member and 0.947 for the
best single placement per checkpoint, so it loses 0.049 to an oracle a defender
cannot pick. The mechanism is that a mean is not robust to an inverted member: a
placement reading 0.194 on `badnet_a2o` at 1% drags the average down instead of
being ignored.

This is the negative result that shaped H41. The min-rank rule keeps the same
defender-legal setup and replaces the mean with a minimum, which an inverted member
cannot poison, and that version does work.

Hypothesis docs: `docs/hypothesis/H17-low-poison-rate-is-a-placement-artifact.md`,
`docs/hypothesis/H18-sensitivity-profile-over-units.md`,
`docs/hypothesis/H41-multi-probe-defence.md`.
Published discussion: `docs/results/adaptive-defender-protocol.md`.
