# H42 -- The evaded operator can be identified without poison labels

**Status: REFUTED.**

Defender-legal identification accuracy is 8.9% (validation PSU std) and 3.6%
(validation PSU IQR), both below the 25% chance level for a 4-operator pool.
The signal is dominated by inherent operator differences, not by the evasion.

The sigma ratio method (needing a baseline model) achieves 52.4%, barely above
chance. With labels (oracle), lowest-AUROC identification is 96.4%.

**This does not matter for the practical protocol.** Multi-probe with all
operators (AUROC 0.951) is within 0.006 of oracle-exclude (0.957), and
forensic exclude by validation std actually hurts (0.847) because it
incorrectly removes the strongest transfer operator. The protocol is: use all
operators, let min-rank handle the rest.

## Claim

Given a model suspected of evasive training against PSBD, the defender can
identify which perturbation operator the attacker evaded using only clean
validation data (no poison labels, no baseline model for comparison). The
evaded operator should show a more compressed validation PSU distribution
(lower std and IQR at sigma-matched rates) because the evasion penalty
specifically optimises PSU to be similar across clean and poisoned populations.

## Why it should work

The evasion penalty matches PSU means between clean and poisoned groups, which
should compress the overall PSU distribution. On clean validation data (100%
clean), this compression should be visible as lower std compared to other
operators at matched sigma.

## Why it failed

The validation PSU std is dominated by inherent differences between operators,
not by the evasion effect. gain_scale (which amplifies rather than removes)
always produces the lowest validation PSU std (0.035 to 0.088) at its matched
sigma, regardless of whether the model is evasive or not. token_mask and
dropout produce stds of 0.19 to 0.33. The evasion compresses token_mask's std
from 0.28 to 0.21, but 0.21 is still higher than gain_scale's 0.04, so the
minimum-std rule always picks gain_scale.

The sigma ratio method (comparing evasive to baseline sigma curves) detects a
real signal: the evaded operator's sigma is 0.66x to 0.80x of baseline at rate
0.3. But this signal overlaps with natural model-to-model variation and only
achieves 52.4% accuracy.

## Identification methods tested (56 checkpoints, ASR > 0.9)

| Method | Accuracy | Notes |
|---|---:|---|
| Lowest single AUROC | 96.4% | Uses labels (oracle, not defender-legal) |
| Lowest sigma ratio at p=0.3 | 52.4% | Needs baseline model |
| Lowest validation PSU std | 8.9% | Defender-legal, fails |
| Lowest validation PSU IQR | 3.6% | Defender-legal, fails |

## What this means for the adaptive defender

Identification is not needed. The multi-probe defence (H41) works without
knowing which operator was evaded:

| Protocol | AUROC |
|---|---:|
| All operators (no identification needed) | 0.951 |
| Oracle exclude (remove the evaded operator) | 0.957 |
| Forensic exclude (remove lowest-std operator) | 0.847 |

Forensic exclude is worse than using all operators because it removes the
wrong operator. The gap between all-operators and oracle-exclude is 0.006,
which is the maximum value of correct identification. The cost of trying and
failing is 0.104 (from 0.951 to 0.847).

Analysis script: `scratch/adaptive_defender_analysis.py`
Full protocol: `docs/results/adaptive-defender-protocol.md`
