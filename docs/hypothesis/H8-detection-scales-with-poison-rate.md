# H8 — Detection improves with poison rate

**Status: OPEN**

## Claim

PSBD AUROC increases with the poison rate used at training time, across
{0.01, 0.05, 0.1}.

## Rationale

More poisoned training samples means a more strongly reinforced trigger-to-target
path, which by the neuron-bias story means a more robust backdoor feature that
survives dropout better while clean evidence does not. So the PSU gap should widen.

## Prediction

Monotone increase in AUROC with poison rate, for each attack. Refuted if flat, or
if detection is *worse* at high poison rates.

## Why it is interesting

Mostly as a consistency check on the mechanism: if the story is right, this
follows almost trivially, and a violation would be informative out of proportion
to the effort of measuring it.

There is also a practical angle. The hardest case for a defender is a *low* poison
rate, which is both stealthier and, if this holds, harder to detect. The 1% column
is therefore the one that matters for a deployment claim, and reporting only the
10% number would be flattering the method.

A confound to control: ASR is already near 1.0 for badnet_a2o, blend, bpp and lf
at *every* rate on CIFAR-10, so poison rate here varies the reinforcement of an
already-saturated backdoor rather than whether the backdoor exists. That makes the
prediction weaker than it sounds, and a flat result would be unsurprising rather
than damning.

## Evidence

Pending. All three rates are in the phase-3a grid for all 5 attacks.

## Known interaction, unrelated to this hypothesis but discovered while checking it

Poison rate does **not** mean what the folder name says for clean-label attacks.
`choose_poison_indices` caps the count at the number of eligible samples, and for
clean-label attacks (`sig`, `lc`) only the target class is eligible. On CIFAR-100
that is 500 images, so 0.01, 0.05 and 0.1 all resolve to the same 500 poisoned
samples: three folders, one experiment. `args.json` records the *requested* rate
with no warning. Confirmed by the measured ASR of `vit_cifar100_sig_0_01` (0.250)
and `_0_05` (0.252).

This does not affect this hypothesis, because `sig` and `lc` are excluded from the
grid for unrelated reasons (weak ASR on ViT). It does invalidate any poison-rate
trend anyone draws for clean-label attacks on CIFAR-100, and it should be fixed by
recording a `realized_poison_rate` in `args.json`.

## Subquestions

1. Does 0.005 (already trained, excluded here for weak ASR) extend the trend
   downward for the attacks where it still works?
2. Is the relationship with poison rate, or with ASR? They are decoupled in this
   grid only for the weaker attacks, which are the ones excluded. WaNet, whose ASR
   spans 0.12 to 0.96 across rates, would separate them.
3. Does the *best rate* `p` shift with poison rate? If a stronger backdoor needs
   more dropout to expose, the adaptive rule has to track it.
