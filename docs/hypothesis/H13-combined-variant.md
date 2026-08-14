# H13 — The accumulated changes combine into a materially better defence

**Status: SUPPORTED, and confirmed out of sample.** +0.110 mean AUROC on the
derivation set (14/15), and **+0.151 on two held-out attacks (2/2)**.

> **Verified against the coverage confound, and clean.** H13 is a *paired*
> comparison: 2 configurations scored on the same 15 checkpoints, reported as
> per-checkpoint deltas and a win count. That construction is immune to the unequal
> coverage that inverted 4 of 6 comparisons elsewhere in this ledger
> ([H19](H19-placement-ranking-is-rate-selection.md)). Re-run end to end, it
> reproduces exactly: **+0.110, 14/15** on the derivation set and **+0.151, 2/2**
> held out.
>
> **Its placement was challenged and survived.** H13 takes `pre_residual` blocks 5-8
> from [H10](H10-depth-band-placement.md), and
> [H20](H20-input-side-beats-residual-adjacent.md) shows that placement sits in the
> losing family, so swapping it should help. It does not. Re-running the whole
> variant with `--placement`, on the 15 checkpoints all 4 options share:
>
> | placement | delta vs `psbd_paper` | wins | paired vs H13 default |
> |---|---|---|---|
> | `pre_residual_blocks_5_8` (H13) | **+0.110** | 14/15 | -- |
> | `before_attention_norm` | +0.128 | 15/15 | +0.017, **1.1 SE**, better on 7/15 |
> | `before_attention` | +0.081 | 13/15 | -0.029, 1.4 SE |
> | `after_embedding` | +0.055 | 11/15 | -0.055, 2.3 SE |
>
> `before_attention_norm` is nominally ahead but not distinguishable, and on the
> held-out pair the H13 default is clearly better (+0.151 against +0.081).
>
> **A cheaper estimate of this said the opposite and was wrong.** Comparing the same
> placements on the cached `adaptive` block gave +0.055 at 3.2 SE in favour of the
> swap. That block uses absolute PSU, shift target 0.8 and a one-sided rule; H13's
> variant changes all 3. A paired comparison under one configuration does not predict
> another, and the only way to settle it was to run the variant. Recorded because the
> estimate was cheap, confident, and misleading.

> **Generalizes to 3 new datasets.** H13 was derived entirely on CIFAR-10. The sweep
> has since produced 33 backdoored checkpoints on other datasets, which are pure
> out-of-distribution tests since nothing was fitted on them:
>
> | dataset | n | mean delta | wins | worst | best |
> |---|---|---|---|---|---|
> | CIFAR-10 (derivation) | 35 | +0.054 | 25/35 | -0.210 | +0.396 |
> | **CIFAR-100** | 21 | **+0.206** | **21/21** | +0.043 | +0.687 |
> | Tiny ImageNet | 11 | +0.042 | 10/11 | -0.031 | +0.140 |
> | GTSRB | 1 | -0.028 | 0/1 | -- | -- |
> | **all non-CIFAR-10** | **33** | **+0.145** | **31/33** | | |
>
> **Matched on attack strength, the pattern holds and explains itself.** The datasets
> differ in mean ASR (0.955 on CIFAR-100 against 0.749 on CIFAR-10), so restrict to
> checkpoints with ASR >= 0.90 and mean ASR becomes 0.97 everywhere:
>
> | dataset | n | `psbd_paper` | `psbd_vit` | delta | wins |
> |---|---|---|---|---|---|
> | CIFAR-10 | 23 | 0.752 | 0.872 | +0.119 | 22/23 |
> | CIFAR-100 | 18 | 0.731 | **0.938** | **+0.207** | **18/18** |
> | Tiny ImageNet | 11 | **0.929** | 0.972 | +0.043 | 10/11 |
>
> The gain tracks the **headroom**, not the dataset: where the published method is
> weakest (CIFAR-100, 0.731) the variant gains most, and where it is already strong
> (Tiny ImageNet, 0.929) there is little left to win. The baseline does not collapse
> anywhere, so this is the variant improving rather than the comparison flattering it.
> The matched CIFAR-10 figure, +0.119, also lands close to the originally published
> +0.110, which is a useful consistency check on the whole re-run.
>
> CIFAR-100 is a clean sweep, 21 of 21, at nearly double the derivation-set effect.
> GTSRB has 1 checkpoint and decides nothing. Note the CIFAR-10 figure has fallen from
> +0.110 on the original 15 to +0.054 on 35, because the sweep has since added harder
> CIFAR-10 checkpoints (low poison rates, weak attacks); the variant still wins 25 of
> them, and the 2 numbers are not comparable to each other for the coverage reason
> [H19](H19-placement-ranking-is-rate-selection.md) describes.

## Claim

Each earlier hypothesis moved one knob in isolation. Assembled into a single
runnable defence, and scored **without ever looking at a poison label**, they should
beat PSBD as published.

| | `psbd_paper` | `psbd_vit` | from |
|---|---|---|---|
| placement | `post_residual`, all 12 blocks | `pre_residual`, blocks 5-8 | [H10](H10-depth-band-placement.md) |
| score | absolute PSU | fractional PSU | [H12](H12-psu-is-not-just-confidence.md) |
| rate rule | shift ratio >= 0.8 | shift ratio >= 0.7 | [H11](H11-adaptive-rate-overshoots.md) |
| decision | one-sided, flag low | two-sided | [H5](H5-all-to-all-breaks-psbd.md) |
| threshold | 25th percentile | 25th percentile | unchanged |

Both variants pick their dropout rate by the paper's adaptive rule on clean
validation data. Neither is allowed the oracle rate reported elsewhere.

## Evidence

CIFAR-10 ViT, full 10000-image test split, AUROC and TPR/FPR at the 25th-percentile
threshold.

| checkpoint | ASR | paper AUROC | paper TPR | vit AUROC | vit TPR | vit FPR | delta |
|---|---|---|---|---|---|---|---|
| `badnet_a2o` 1% | 1.00 | 0.297 | 0.057 | **0.640** | 0.428 | 0.266 | **+0.344** |
| `badnet_a2a` 5% | 0.93 | 0.476 | 0.255 | **0.739** | 0.577 | 0.255 | **+0.263** |
| `badnet_a2o` 5% | 1.00 | 0.582 | 0.162 | **0.822** | 0.777 | 0.224 | +0.240 |
| `badnet_a2o` 10% | 1.00 | 0.656 | 0.258 | **0.863** | 0.703 | 0.189 | +0.207 |
| `badnet_a2a` 10% | 0.96 | 0.450 | 0.184 | **0.611** | 0.384 | 0.232 | +0.161 |
| `badnet_a2a` 1% | 0.94 | 0.476 | 0.232 | **0.621** | 0.339 | 0.265 | +0.145 |
| `lf` 5% | 1.00 | 0.920 | 0.950 | **0.988** | 0.997 | 0.205 | +0.069 |
| `lf` 1% | 0.98 | 0.904 | 0.915 | **0.966** | 0.964 | 0.201 | +0.062 |
| `lf` 10% | 1.00 | 0.941 | 0.972 | **0.995** | 0.998 | 0.216 | +0.054 |
| `blend` 10% | 1.00 | 0.953 | 0.991 | **0.998** | 1.000 | 0.240 | +0.045 |
| `bpp` 10% | 1.00 | 0.961 | 0.983 | **0.990** | 0.993 | 0.191 | +0.028 |
| `blend` 1% | 1.00 | 0.956 | 0.992 | **0.976** | 0.998 | 0.253 | +0.020 |
| `bpp` 1% | 0.98 | 0.966 | 0.985 | **0.986** | 0.979 | 0.176 | +0.020 |
| `bpp` 5% | 1.00 | 0.990 | 0.998 | **0.997** | 0.998 | 0.168 | +0.007 |
| `blend` 5% | 1.00 | **0.956** | 0.996 | 0.947 | 1.000 | 0.217 | -0.009 |
| | | | | | | | |
| **mean** | | **0.792** | | **0.876** | | | **+0.110** |
| **wins** | | | | | | | **14 / 15** |

**Negative control**, `vit_cifar10_benign` probed with the same trigger:

| | AUROC | TPR | FPR |
|---|---|---|---|
| `psbd_paper` | 0.504 | 0.227 | 0.219 |
| `psbd_vit` | 0.507 | 0.295 | 0.284 |

Both at chance with TPR ≈ FPR, so neither variant manufactures detection.

## Where the gain comes from

Almost entirely from the cases the published method handles worst. On `blend` and
`bpp`, where PSBD already reaches 0.95 to 0.99, the change is +0.007 to +0.045. On
`badnet_a2o` at 1% poisoning it is +0.344, taking the method from **worse than
chance** (0.297) to usable.

Four of the fifteen rows show `psbd_paper` scoring **below 0.5**, i.e. the published
one-sided rule actively anti-correlates on all-to-all and on low-poison-rate patch
triggers. That is what the two-sided rule fixes.

## A bug this comparison caught, worth recording

The first version of the two-sided rule flipped the decision direction without
flipping the threshold. The quantile sets the tolerated clean loss: flagging *below*
the 25th percentile costs 25% of clean data, but flagging *above* that same
percentile costs 75%. The benign control duly reported **TPR 0.813 at AUROC 0.507**,
which is a detector with no signal flagging 81% of everything.

Using the complementary quantile when the direction flips restores TPR 0.295 against
FPR 0.284. AUROC was never affected, being rank-based, which is exactly why AUROC
alone would not have surfaced it.

## Held-out confirmation

`sig` and `lc` at 10% poisoning. Neither was used to choose the band, the shift
target, or anything else; both were excluded from every earlier grid for failing at
lower poison rates.

| checkpoint | ASR | paper AUROC | paper TPR | vit AUROC | vit TPR | delta |
|---|---|---|---|---|---|---|
| `lc` | 0.98 | 0.515 | 0.250 | **0.786** | 0.645 | **+0.271** |
| `sig` | 0.90 | 0.900 | 0.906 | **0.931** | 0.911 | +0.031 |
| **mean** | | 0.708 | | **0.859** | | **+0.151** |

**2 of 2, and the mean gain is larger than on the derivation set** (+0.151 against
+0.110). `lc` is the striking one: the published configuration sits at 0.515, i.e.
chance, and the adapted one reaches 0.786.

So the two fitted choices are not overfitted. The improvement transfers to attacks
that had no influence on them.

## Honest limitations

- **Two of the four changes were fitted on the derivation set.** The blocks 5-8 band
  and the 0.7 shift target were chosen after seeing those results, which is why the
  held-out column above exists and is the one to quote.
- `psbd_paper` is given the *extended* rate grid (down to 0.005), which the published
  method would not have swept. That is deliberately generous to it: on the original
  0.1-to-0.9 grid it would be worse still.
- Single dataset, single architecture, single seed, k=3 passes.
- The evaluation pool is a held-out test split, not the poisoned training set PSBD
  was designed to filter, so none of these numbers are directly comparable to the
  published tables.

## Reproduce

```bash
python psbd_variants.py --held-out vit_cifar10_sig_0_1 vit_cifar10_lc_0_1
```

## Subquestions

1. Ablate the four changes one at a time to see how much each contributes to the
   +0.110. The band and the two-sided rule are the likely carriers.
2. Does the advantage survive on SAM-trained models, and on Swin?
3. What is the clean-accuracy cost of perturbing blocks 5-8 at p=0.8? A defence is
   only usable if that cost is bounded, and it has not been measured.
