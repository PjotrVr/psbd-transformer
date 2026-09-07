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
each attack's `apply_trigger` is deterministic and rate independent, so the trigger
is identical no matter how many training images were selected, and the metrics are
computed once per (attack, dataset) pair.

Raw output lands in `scratch/stealth_metrics_results.json`, which stays out of git
because the published table in `docs/results/stealth-metrics.md` is the record.

## Finding

The published table is in `docs/results/stealth-metrics.md`. The point it settles
for the rest of the project is that the panel spans a wide stealth range, so the
attacks PSBD detects best are not simply the loudest ones.
