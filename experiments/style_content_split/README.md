# Is a backdoor trigger a style feature? No, and it could not have been. (H45)

## The idea

A trigger is a texture and the class evidence is structure, so a style-content
decomposition should put the poison entirely on one side, and normalising the style is a
purification defence. Instrument: Fourier phase carries structure and amplitude carries
texture, so the split is exact, invertible and needs no learned decoder.

    mix(x, d, lam) = IFFT( [(1-lam)|FFT(x)| + lam|FFT(d)|] * exp(i * angle(FFT(x))) )

The suspect's phase is always kept; its amplitude is pulled toward a clean donor's by lam.

Predicted taxonomy: badnet (a 3x3 corner patch) rides in PHASE, blend/sig/lf/bpp (global
patterns) in AMPLITUDE, wanet (an elastic warp) in NEITHER.

## Result: refuted on both halves

**The taxonomy is wrong, and wrong in both directions.** Measured on `vit_gtsrb_*_0_05`,
the deployable operating point being the largest ASR reduction available while clean
accuracy stays within 2 points:

| attack | predicted | ASR at lam=0 | deployable ASR | clean-accuracy cost | gain |
|---|---|---|---|---|---|
| adaptive_blend | amplitude | 0.778 | **0.778** | 0.000 | **none**, and ASR RISES to 0.896 mid-curve |
| wanet | neither | 0.829 | 0.760 | -0.015 | -0.069 |
| **lf** | amplitude | 0.999 | **0.999** | -0.005 | **none** |
| sig | amplitude | 0.015 | - | - | never implanted, uninformative |

`lf` is the low-FREQUENCY attack, the most texture-carried thing in the panel, and it
survives a FULL amplitude replacement at **0.949 ASR** while clean accuracy falls to 0.273.
`adaptive_blend` gets *more* effective as texture is destroyed, because amplitude mixing
damages the clean pathway faster than the backdoor pathway, which is the same
robust-shortcut asymmetry PSBD itself rests on.

## Why it had to fail, which is the part worth keeping

For an additive trigger, write rho(k) = |T(k)|/|X(k)| and Delta(k) for the relative phase.
To first order

    |X_p| - |X| = |T| cos(Delta)          amplitude change
    angle(X_p) - angle(X) = rho sin(Delta) phase change

so for rho << 1 with Delta near-uniform over natural images, the expected share of the
perturbation carried by amplitude tends to **1/2 regardless of the trigger's spatial
structure**. In the opposite limit rho >> 1 the amplitude change is unbounded while the
phase change is bounded by pi, so the share becomes a monotone readout of trigger
**contrast**, not trigger **type**.

A high-contrast 3x3 corner patch and a global sinusoid are therefore forced to the same
place. **Seven of the nine attacks in this panel are additive and so are unattributable by
construction in this basis.** Only multiplicative operators can be attributed: a
multiplicative amplitude filter, a warp (a coordinate operator, hence phase to first
order), and an all-pass phase filter.

## Prior art, which closes it independently

| paper | what it already says |
|---|---|
| **FIBA**, CVPR 2022 | states the amplitude-carries-low-level, phase-carries-semantics premise verbatim and builds a backdoor entirely in amplitude. ASR 99.53, bypasses STRIP, Neural Cleanse anomaly 1.26 against a 2.0 threshold |
| **DUBA**, AAAI 2024 | uses the exact FFT amplitude/phase recombination as a poisoning stage, explicitly to defeat magnitude-based frequency detectors. ASR 99.98, drives a frequency detector to 49.96%, i.e. chance |
| **DFST**, AAAI 2021 | "the trigger is a style", via CycleGAN. GTSRB 0.999 ASR, undetected by NC, ABS and ULP |
| **Zeng et al.**, ICCV 2021 | owns spectral localisation as a detector, and the smooth low-frequency trigger that evades it |
| **Color Backdoor**, CVPR 2023 | direct proof that global style-like triggers are the ones that BEST survive this defence family: 93.92% mean ASR retained |
| **Lite-BD** | already published most of the predicted taxonomy for a radial band-stop: only SIG and LF die, 2 of 10 |
| **FTrojan**, ECCV 2022 | already priced frequency smoothing at 4.33 to 33.91 points of clean accuracy |
| **REFINE**, ICLR 2025 | fills the transformation-defence slot and names the weakness: such defences "must modify all features indiscriminately" |

So "the backdoor is a style feature" is a 2021-2022 attack design premise, not an open
hypothesis. Style normalisation as a defence is unpublished by name and dead in substance.

## What survives

The negative itself, stated as a mechanism: **the Fourier amplitude/phase basis cannot
attribute an additive trigger**, because the split reads contrast rather than type. That is
a short, checkable, falsifiable statement, it explains the published disagreements (FREAK
measures BadNet as amplitude-heavy at 96.60% while the naive taxonomy calls it a patch),
and it predicts exactly which trigger families ARE attributable.

## Running it

    PYTHONPATH=. python experiments/style_content_split/measure.py \
        --checkpoint-folder vit_gtsrb_adaptive_blend_0_05
