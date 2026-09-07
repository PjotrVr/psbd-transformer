# Selecting the detection tail without labels (H15)

## Question

H15 showed both PSBD and STRIP invert on the attacks their stated mechanism does not
cover, and that allowing the sign to flip recovers up to 0.98 AUROC. But
`max(AUROC, 1 - AUROC)` reads the labels, so it is an upper bound rather than a
method. This tests a rule that reads no labels.

The defender holds 2 things: a clean validation split, and a suspicious pool that is
mostly clean with an unknown poisoned fraction. Poisoned samples pile up in 1 tail
of the score distribution, so compare the pool's tails against validation's:

    low_deviation  = quantile(validation, q)     - quantile(pool, q)
    high_deviation = quantile(pool, 1 - q)       - quantile(validation, 1 - q)

Whichever is larger names the tail the anomaly sits in. Both are measured against
the defender's own clean reference, and the pool enters only through its unlabelled
score distribution.

The number that matters is not the AUROC this achieves. It is whether the selected
tail AGREES with the tail the labels would have chosen. High agreement makes the
two-sided gain real and deployable. Low agreement leaves H15 an upper bound.

## Running it

    PYTHONPATH=. python experiments/tail_selection/measure.py --detector strip

CPU only, reading cached detector scores.

## Finding

Agreement with the oracle tail is 24 of 24 on the split as it stands, and tightening
the tail largely rescues the cases where it does not: at q = 0.002 agreement runs
79% to 92%.

The method was retired anyway, and H15 records why. Inversion is a diagnostic
symptom of a broken assumption, never a decision rule. A detector that flags the
opposite tail when its own premise fails has abandoned the premise while keeping the
name. Every downstream result in this project is therefore one-sided: low PSU means
poisoned, always, and an AUROC below 0.5 is reported as the method failing on that
attack rather than recovered into a win. The measurements here stand and are kept as
a recorded negative result.

Hypothesis doc: `docs/hypothesis/H15-one-sided-rules-are-the-common-weakness.md`.
