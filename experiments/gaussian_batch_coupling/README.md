# Did GaussianNoise's batch-wide std bias the split comparison?

## Question

`GaussianNoise` scaled its noise by `x.std()` reduced over batch, tokens and
channels together, so a sample's perturbation magnitude depended on which other
samples shared its batch. The validation, clean and backdoor splits hold
different image populations, so the 3 could sit at 3 different noise levels while
every comparison between them assumes 1. Every other operator draws per sample.

## Run

```bash
PYTHONPATH=. python experiments/gaussian_batch_coupling/measure.py \
    --checkpoint-folder vit_cifar10_badnet_a2o_0_1 vit_cifar100_badnet_a2o_0_1 vit_cifar100_blend_0_1
```

About 2 minutes on the login-node A100. Writes
`results/gaussian_batch_coupling.json`.

## Finding: position dependent, and it explains a published negative result

Activation std per split, relative to the clean split, 40 batches each:

| checkpoint | position | backdoor vs clean | validation vs clean |
|---|---|---:|---:|
| `vit_cifar10_badnet_a2o_0_1` | `before_mlp` | -0.75% | -0.03% |
| `vit_cifar10_badnet_a2o_0_1` | `before_attention_norm` | **+13.16%** | +0.20% |
| `vit_cifar100_badnet_a2o_0_1` | `before_mlp` | -1.10% | -0.05% |
| `vit_cifar100_badnet_a2o_0_1` | `before_attention_norm` | **+32.30%** | -0.01% |
| `vit_cifar100_blend_0_1` | `before_mlp` | +2.27% | -0.06% |
| `vit_cifar100_blend_0_1` | `before_attention_norm` | **+15.43%** | +0.07% |

**The validation split is clean.** It sits within 0.2% of the clean analysis pool
at every position, so the threshold was always calibrated at the right noise
level. Nothing about the quantile rule is affected.

**`before_mlp` is safe.** Coverage is -1.1% to +2.3%, well inside the 1 to 2%
band where the confound cannot carry a result. The headline gaussian cell
(`gaussian @ before_mlp`, ranked 2nd overall) stands.

**`before_attention_norm` is not.** The backdoor split received 13 to 32% more
noise than the clean split it is compared against. The mechanism is direct:
triggered images carry a larger activation std at the attention input, the old
operator read that std off the batch, so triggered batches were perturbed harder.
More noise on backdoor samples raises backdoor PSU, and since low PSU means
poisoned, that drags AUROC down.

That looked like the shape of the published failure. `docs/results/negative-results.md`
section 7 reports gaussian at `before_attention_norm` on CIFAR-100 badnet 1% at
AUROC 0.168, a "massive inversion", and the checkpoint with the worst coupling in
the table above is CIFAR-100 badnet at +32.3%. The obvious hypothesis was that the
inversion was an artifact of the operator rather than a property of gaussian noise
at that position.

**That hypothesis is refuted.** The corrected operator was re-swept and the same
cells re-scored against their archived counterparts:

| checkpoint | rate | old AUROC | new AUROC | delta |
|---|---:|---:|---:|---:|
| `vit_cifar100_badnet_a2o_0_01` | 0.3 | 0.168 | 0.191 | +0.023 |
| `vit_cifar100_badnet_a2o_0_01` | 0.5 | 0.168 | 0.171 | +0.003 |
| `vit_cifar100_badnet_a2o_0_1` | 0.5 | 0.173 | 0.171 | -0.002 |
| `vit_cifar100_badnet_a2o_0_05` | 0.3 | 0.269 | 0.283 | +0.015 |

Across 50 badnet cells the mean delta is +0.004, the range is -0.043 to +0.067,
and the number of inverted cells is **37 before and 37 after**. Across all 130
cells re-scored so far the mean delta is +0.001.

The reason the large noise difference moves AUROC so little is that AUROC is a
ranking statistic. Raising the noise on every backdoor sample shifts that split's
scores together, which moves a fixed threshold much more than it moves the
ordering. The threshold-dependent numbers are the ones to re-check, not AUROC.

So section 7's explanation stands on its own: gaussian noise at the attention
input really does destroy clean predictions as fast as poisoned ones on a
100-class problem. The fix was still correct to make, since a sample's
perturbation should not depend on which other samples share its batch and every
other operator already drew per sample, but it does not rescue this result.

## Consequence

`gaussian @ before_attention_norm` needs re-sweeping with the per-sample std
before any claim rests on it. `gaussian @ before_mlp` does not, and its cached
numbers can be reported as they are with this measurement cited.
