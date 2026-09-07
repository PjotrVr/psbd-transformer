# Token-level decomposition of the backdoor direction (H32, H37)

## Question

ViT-B/16 on 224x224 inputs produces 196 patch tokens plus the CLS token, and the
patch tokens map onto a 14x14 spatial grid. Projecting per-token features onto the
backdoor direction therefore has a spatial reading: it says where in the image the
backdoor signal is carried. H32 asks whether that localizes a trigger without
knowing the attack. H37 asks whether the resulting concentration number classifies
the attack family.

## The 2 scripts

| script | what it produces |
| --- | --- |
| `token_localization.py` | per-token direction norms at layer 12, the concentration ratio, and the top patch locations on the 14x14 grid |
| `token_concentration_classifier.py` | thresholds that concentration ratio into localized against global, and tests whether the threshold transfers across datasets and poison rates |

`token_concentration_classifier.py` is CPU only and reads
`results/token_localization.json`, so it has to run after `token_localization.py`.

## Running it

    python experiments/token_structure/token_localization.py
    python experiments/token_structure/token_concentration_classifier.py

## Finding

H32 is supported. Per-token direction norms do recover trigger position. BadNet's
highest-norm tokens are its trigger patches, landing at grid positions (13,13),
(12,13) and (13,12), which is the corner the trigger occupies. `blend` and `wanet`
give diffuse, low-concentration patterns, exactly as a global trigger should.

H37 is refuted. Turning that observation into a classifier gives 62.5% accuracy at
the best threshold, with a negative separation gap of -0.98, meaning the localized
and global ranges overlap rather than separate. `adaptive_blend` reads 3.10 and `lf`
reads 3.33 despite both being global attacks, while `badnet_a2o` on CIFAR-100 reads
2.35 and is missed. The visual finding is real and the assumed correspondence
between trigger locality and direction concentration does not hold.

Hypothesis docs: `docs/hypothesis/H32-token-localization-spatial.md`,
`docs/hypothesis/H37-token-concentration-attack-classifier.md`.
