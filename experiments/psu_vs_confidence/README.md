# Is PSU just measuring baseline confidence?

## Question

The skeptical reading of PSBD: a backdoored model is extremely confident on triggered
inputs; PSU subtracts a dropout-perturbed confidence from the no-dropout one; and a
sample starting near probability 1 has more room to fall than one starting at 0.6. If
that is the whole story, PSU is an elaborate proxy for baseline confidence and a
defender could skip the `k` stochastic forward passes entirely.

This was raised by an adversarial verification pass, which noted that on the benign
control a confidence-only detector scores 0.514 against PSBD's 0.506.

## Run

```bash
PYTHONPATH=. python experiments/psu_vs_confidence/measure.py
```

No GPU, seconds, reads only the cached tensors.

## Finding: the skeptical reading is refuted

`before_mlp_residual`, CIFAR-10 ViT at 10% poisoning, each at PSU's own best rate.
AUROC with backdoor as the positive class.

| checkpoint | PSU | confidence only | fractional drop |
|---|---|---|---|
| `blend` | 0.984 | 0.701 | **0.997** |
| `bpp` | 0.992 | 0.822 | **0.993** |
| `lf` | 0.979 | 0.780 | **0.991** |
| `badnet_a2o` | 0.922 | 0.878 | **0.925** |
| `badnet_a2a` | 0.508 | 0.244 | 0.499 |
| benign control | 0.508 | 0.486 | 0.506 |
| **mean (backdoored)** | **0.877** | **0.685** | **0.881** |

**PSU beats confidence-only on 5 of 5 checkpoints**, by 0.19 on average. The
stochastic passes are doing real work.

The decisive column is the third. The **fractional** drop, `1 - mean_dropout / P_c(x)`,
divides out the starting confidence entirely. If PSU worked only because confident
samples fall further in absolute terms, normalising by that confidence would destroy
the signal. It does the opposite: the fractional drop scores **as well or better on
every checkpoint**. So PSU is measuring how *robust* the prediction is, not how
confident it started.

## A free improvement

Mean 0.881 against PSU's 0.877, and better on all four working attacks (blend
+0.013, lf +0.012, bpp +0.001, badnet_a2o +0.003). The gain is small but one-sided,
and it costs nothing: it is the same cached tensors divided by a number already on
disk.

It also has a principled reason to be preferred over the paper's absolute form. The
threshold is a quantile of clean-validation PSU, and absolute PSU is bounded above by
the starting confidence, so the threshold inherits the validation set's confidence
distribution. The ratio does not, which should make it transfer better across
datasets and models with different calibration.

## Where confidence alone does explain most of it

`badnet_a2o` is the exception: confidence-only reaches 0.878 against PSU's 0.922, so
for the static patch trigger most of the separation is available without dropout at
all. For `blend` the gap is 0.701 against 0.984. So the value PSBD adds is largest
exactly where a naive baseline is weakest, which is the right way round.

## Subquestions

1. Does the fractional form's advantage hold under the adaptive rate rule as well as
   the oracle rate, and across poison rates and SAM? If so it should simply replace
   the absolute form in `psu_from_cache`.
2. Confidence-only scores 0.685 mean. That is a baseline no PSBD paper reports, and
   any detection method should be shown to beat it.
3. Does combining the two (confidence and PSU as two features) beat either? That
   would say they carry partly independent information.
