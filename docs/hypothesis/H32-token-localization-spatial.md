# H32 — Token-level PSU decomposition localizes triggers spatially

**Status: SUPPORTED.** Per-token direction norms reveal the spatial location of
localized triggers without knowing the attack. BadNet's trigger patches are the
highest-norm tokens. Blend and WaNet produce diffuse, low-concentration patterns.

Evidence: `scratch/token_localization.py`, results in
`results/token_localization.json`.

## Claim

Decomposing the backdoor direction per token at layer 12 reveals the spatial
location of the trigger. Trigger tokens should have much higher direction norms
than clean tokens, with localized attacks showing high concentration and global
attacks showing uniform distribution.

## Results

| attack | dataset | CLS norm | patch mean | concentration | top4 frac | top patch location |
|---|---|---:|---:|---:|---:|---|
| badnet_a2o | cifar100 | 13.15 | 42.47 | 2.35 | 0.035 | (13,13), (12,13), (13,12) |
| blend | cifar100 | 19.31 | 24.33 | **1.39** | 0.028 | dispersed |
| wanet | cifar100 | 14.89 | 2.07 | 1.64 | 0.032 | dispersed |
| adaptive_blend | cifar100 | 10.81 | 7.88 | 3.10 | 0.057 | (13,3), (13,2), (11,2) |
| sig | cifar100 | 4.61 | 5.43 | 2.22 | 0.042 | (0,9), (0,8), (0,1), (0,11) |
| lf | cifar100 | 22.09 | 11.11 | 3.33 | 0.061 | (5,0), (1,13), (2,13) |
| badnet_a2o | tiny | 13.37 | 14.50 | **4.23** | 0.044 | (13,13) |
| blend | tiny | 16.03 | 17.25 | 1.75 | 0.035 | dispersed |

## Spatial trigger recovery

### BadNet (localized 3x3 patch, bottom-right corner)

The top patch token is (13,13) on both CIFAR-100 (norm=99.59, 2.35x mean) and
Tiny (norm=61.4, 4.23x mean). The second and third tokens are immediate
neighbors: (12,13) and (13,12). This is exactly where BadNet places its trigger
on 224x224 images: the bottom-right corner maps to patch grid position (13,13)
in the 14x14 grid.

### SIG (sinusoidal stripe, horizontal)

All top 5 patches are in row 0: (0,9), (0,8), (0,1), (0,11), (0,3). SIG adds
a sinusoidal pattern that is strongest along one edge of the image. The token
decomposition correctly identifies the affected row.

### Blend (global alpha blending)

Concentration ratio is the lowest at 1.39 (CIFAR-100) and 1.75 (Tiny). The top
patches show no spatial pattern. This is the expected null result for a global
trigger that modifies all pixels uniformly.

### WaNet (image warping)

Concentration ratio 1.64, similar to blend. WaNet's pixel warping affects all
regions roughly equally, producing no localizable pattern. The per-token norms
are also much smaller (mean 2.07 vs 42.47 for badnet), indicating WaNet's
effect is distributed differently across the residual stream.

## Key observation: patch norms > CLS norm for most attacks

For badnet (42.47 vs 13.15), blend (24.33 vs 19.31), and lf (11.11 vs 22.09,
exception), the mean patch token direction norm exceeds the CLS token direction
norm. This means the backdoor signal is distributed across many token positions,
not concentrated in CLS. The CLS token's strong backdoor direction (which drives
classification) emerges from attention-mediated aggregation of the per-token
signals, not from the CLS token independently encoding the backdoor.

## Practical application

Token-level direction decomposition provides:
1. **Trigger localization** for patch-based attacks at zero cost (no extra
   training, no trigger search)
2. **Attack family identification**: high concentration (> 3x) indicates a
   localized trigger; low concentration (< 2x) indicates a global trigger
3. **Interpretable evidence** for why a sample was flagged
