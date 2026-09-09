# Trigger imperceptibility across the attack panel

## Question

Detection numbers are only comparable across attacks if the attacks are comparably
visible. An attack whose trigger is obvious to a human is solving an easier problem
than one whose trigger is not, so the panel needs a stealth axis before any ranking
over it means anything.

Measured as PSNR and SSIM between clean and poisoned images at native resolution,
32x32 for CIFAR-10, CIFAR-100 and GTSRB and 64x64 for Tiny ImageNet, before the
224x224 upscale the model sees. Measuring after the upscale would fold the
interpolation into the score.

## Running it

    python experiments/stealth/stealth_metrics.py

200 images per (attack, dataset) pair, drawn from the test split at seed 42, target
label 0, every attack at its default configuration. Poison rate is not a parameter:
the metrics are computed once per (attack, dataset) pair.

That holds for the EVAL trigger, which is what this measures and what a defender
faces at inference. It is no longer true of `apply_trigger`. Two attacks now vary
their training trigger per sample, Adaptive-Blend by planting a subset of its
pattern so the model must generalise over it, and Label-Consistent by substituting
an adversarially perturbed base. Measuring those would make the result depend on
which indices happened to be sampled.

Raw output lands in `experiments/stealth/stealth_metrics_results.json`, which stays out of git
because the published table in `docs/results/stealth-metrics.md` is the record.

## Finding

The published table is in `docs/results/stealth-metrics.md`. The point it settles
for the rest of the project is that the panel spans a wide stealth range, so the
attacks PSBD detects best are not simply the loudest ones.
