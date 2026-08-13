# H3 — Post-residual fails because it saturates: it destroys clean and backdoor evidence alike

**Status: SUPPORTED (preliminary, n=64 smoke)**

## Claim

Post-residual dropout does not fail because it is too weak or badly placed. It
fails because it is *too destructive at every usable rate*. Masking the residual
stream once per block means only `(1-p)^12` of coordinates survive the ViT stack.
At p=0.1 that is already 0.28; at p=0.5 it is 0.0002. The model's prediction
becomes noise for clean and backdoor samples alike, so the PSU distributions
collapse onto each other and there is nothing left to threshold.

This is the mechanism behind [H1](H1-pre-beats-post.md). H1 says post-residual
loses; H3 says why, and predicts something H1 does not.

## Prediction

The distinguishing prediction is about the shift ratio, not about AUROC:
**post-residual's clean-validation shift ratio is already saturated at the
smallest swept rate.** If the failure were "too weak", sigma would start near 0
and climb. If it is "too destructive", sigma starts near its ceiling and has
nowhere to go.

Refuted if post-residual shows a low-sigma regime at small p where clean and
backdoor PSU still separate.

## Why it is interesting

It converts a placement result into a structural claim about transformers. A
ResNet BasicBlock has 8 residual adds and a ReLU after each, which re-sparsifies
and bounds the signal. ViT-B/16 has 12 blocks, 24 adds, LayerNorm rather than
ReLU, and a residual stream that CKA shows to be persistent across depth: the
stream *is* the model's working memory. Multiplicatively masking it is not a
perturbation of that memory, it is an erasure. The ConvNet recipe transfers
badly for a reason specific to the architecture, which is a much stronger claim
than "the hyperparameter needs retuning".

It also predicts that the right fix is not "use post-residual with a smaller p" —
there is no small enough p, because the compounding is exponential in depth.

## Evidence

Smoke run, `vit_cifar10_badnet_a2o_0_1`, 64-sample splits. Clean-validation shift
ratio by rate:

| rate | `post_residual` sigma | `pre_residual` sigma |
|---|---|---|
| 0.1 | **0.859** | 0.010 |
| 0.2 | 0.891 | 0.078 |
| 0.3 | 0.885 | 0.417 |
| 0.4 | 0.911 | 0.807 |
| 0.5 | 0.917 | 0.880 |
| 0.9 | 0.948 | 0.922 |

Post-residual enters at 0.86 and never moves: its entire swept range sits above
the shift ratio the PSBD paper picks as its *operating point* (0.8). It has no
low-disturbance regime at all. Pre-residual traverses 0.01 to 0.92 and its best
AUROC (0.903) lands at sigma 0.807, almost exactly the paper's target.

Correspondingly, post-residual's PSU means are clean 0.855 / backdoor 0.848 —
separated by 0.007, i.e. not separated. Both are near the ceiling because
confidence in the original prediction has been destroyed for everything.

## What would change the verdict

- Post-residual showing sigma below 0.5 at some rate. The current grid starts at
  p=0.1; a finer grid (0.01 to 0.09) would test whether a usable regime exists
  below it. **This is worth running and is not yet done.**
- The same experiment on Swin (24 blocks) showing *less* saturation, which would
  contradict the depth-compounding story.

## Reproduce

`results/<folder>/psbd_metrics.json`, key `placements.<name>.rates[i].shift_ratio.validation`.

## Subquestions this opens

1. **Does a sub-0.1 rate rescue post-residual?** If sigma at p=0.02 lands near
   0.4 and AUROC recovers, the story becomes "post-residual needs a rate 10x
   smaller", which is weaker but still useful. Directly testable, cheap.
2. Is the saturation really from depth compounding? Applying post-residual
   dropout at only the last 2 blocks would isolate it.
3. Does the same saturation explain why the PSBD paper needs p in 0.7 to 0.9 on
   ResNet? Their 8 adds compound far less, so their usable range sits much higher.
4. Is there a placement that perturbs the stream *additively* rather than
   multiplicatively (Gaussian noise instead of dropout) and so avoids the
   exponential collapse? The `dropout_factory` argument already supports this and
   nothing has used it.
