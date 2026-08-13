# H13 — The accumulated changes combine into a materially better defence

**Status: SUPPORTED on the derivation set (+0.110 mean AUROC, wins 14/15).
Held-out confirmation is running.**

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

## Honest limitations

- **Two of the four changes are fitted here.** The blocks 5-8 band and the 0.7 shift
  target were both chosen after seeing results on these attacks, so the derivation
  numbers are optimistic. A held-out run on `sig` and `lc` at 10% poisoning is in
  flight and is the number to trust.
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
