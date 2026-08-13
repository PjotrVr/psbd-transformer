# H10 — Aiming dropout at the depth where an attack's direction lives beats spreading it over all blocks

**Status: the premise is SUPPORTED, the predicted direction is REFUTED, and the
inverted rule is the best result in this study.**

Restricting pre-residual dropout to a band of blocks beats applying it to all 12.
But the band to choose is the one **furthest from** where the attack's backdoor
direction is written, not the one on top of it.

## Original prediction

If placement works by disturbing the backdoor path where it is being written, then a
band around an attack's measured onset layer should win: `blend` (onset 5) and `bpp`
(6) should prefer an early band, `badnet_a2o` (9) a late one.

## Evidence

CIFAR-10 ViT, 10% poisoning, `pre_residual` restricted to each band, each at its own
best rate, AUROC at the 25th-percentile threshold.

| attack | onset | blocks 1-4 | blocks 5-8 | blocks 9-12 | all 12 | best | predicted |
|---|---|---|---|---|---|---|---|
| `blend` | 5 | 0.969 | 0.981 | **0.997** | 0.978 | 9-12 | 5-8 |
| `bpp` | 6 | 0.982 | 0.985 | **0.996** | 0.977 | 9-12 | 5-8 |
| `lf` | 8 | 0.925 | 0.988 | **0.998** | 0.947 | 9-12 | 5-8 |
| `badnet_a2o` | 9 | **0.912** | 0.907 | 0.829 | 0.889 | 1-4 | 9-12 |
| `badnet_a2a` | 9 | 0.510 | 0.593 | 0.159 | 0.510 | -- | -- |
| benign control | -- | 0.507 | 0.501 | 0.487 | 0.506 | -- | -- |

**The prediction is wrong for all four working attacks, and wrong in the same
direction each time.** The three attacks whose direction appears *early* are best
caught by perturbing *late*; the one whose direction appears *late* is best caught by
perturbing *early*.

**A band beats all-blocks in every case**, which is the half of the hypothesis that
survives: 0.997 against 0.978 (blend), 0.996 against 0.977 (bpp), 0.998 against 0.947
(lf), 0.912 against 0.889 (badnet_a2o). The benign control stays between 0.487 and
0.507 for every band, so this is not the bands making everything look better.

## The corrected rule, and why it makes more sense than the original

**Perturb where the clean evidence lives and the backdoor does not.**

PSU works by destroying the evidence a *clean* prediction rests on while leaving the
trigger-to-target path intact. So the perturbation should land where clean class
evidence is being assembled and the backdoor is absent:

- `blend`, `bpp`, `lf` write their backdoor direction by layer 5 to 8. By the last
  four blocks it is already established and robust, while clean class evidence is
  still being refined. Perturbing blocks 9-12 therefore removes clean evidence and
  spares the backdoor. Maximum PSU gap.
- `badnet_a2o` only assembles its direction at layers 9 to 12. Perturbing there
  damages the backdoor along with everything else, which is why blocks 9-12 is its
  *worst* band (0.829, below all-blocks). Perturbing blocks 1-4, where the patch
  trigger has not yet been routed into `[CLS]`, spares it.

The original prediction had the mechanism backwards: it aimed the perturbation at the
backdoor, when the point is to aim it at everything else. The onset measurement is
still what selects the band, just with the opposite sign, so the practical recipe
survives intact and is still one cheap measurement away.

## Practical result

`pre_residual` restricted to **blocks 9-12** is the strongest placement found
anywhere in this study for the three distributed-trigger attacks: 0.996 to 0.998,
against 0.947 to 0.978 for the same position applied to all blocks, and against
0.969 for the best all-blocks placement (`before_mlp_residual`).

It is also cheaper: dropout in 4 blocks instead of 12.

## The one anomaly worth chasing

`badnet_a2a` at blocks 9-12 scores **0.159**, far *below* chance rather than near it.
An AUROC that low is a detector working in reverse: backdoor samples are reliably
scoring HIGHER PSU than clean ones. Everything else about `badnet_a2a` sits at chance,
so this is the only signal of any kind found for the all-to-all attack, and its sign
is inverted. Either something is systematically wrong for all-to-all in this band, or
there is real structure that a flipped decision rule would exploit.

## Reproduce

```bash
PYTHONPATH=. python pbs/generate_psbd_jobs.py --phase depth_bands
python psbd_analyze.py --all
```

## Subquestions

1. **Confirm the rule predicts out of sample.** It was derived from four attacks after
   seeing their results. `wanet` at 10% poisoning (ASR 0.96, excluded elsewhere for
   failing at lower rates) is an untouched test case: measure its onset, predict the
   band, then run it.
2. Is 4 blocks the right width, or would 2 blocks, or a single block, do better still?
   `--block-range 12 12` is already supported.
3. Does the strength confound explain any of this? An early band perturbs harder at
   the same rate, yet the early band *loses* for three of four attacks, which is the
   opposite of what the confound would produce. Still worth a matched-sigma check.
4. Why is `badnet_a2a` at blocks 9-12 strongly anti-correlated?
