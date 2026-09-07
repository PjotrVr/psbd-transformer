# H6 — SAM training makes backdoors more detectable

**Status: REFUTED for prediction-space detection, and the positive control explains
why.** SAM does amplify the backdoor on ViT, exactly as the paper claims, but the
amplification does **not** produce the separability gain their detectors feed on, and
it does produce the clean-variance side effect their method exists to cancel. PSBD has
no way to cancel it.

> **Coverage note, added later.** The naive Adam-versus-SAM aggregate in this project compares n = 225 against n = 16 and is confounded: the SAM checkpoints were swept over a harder set. Matched on architecture, dataset, attack, poison rate and placement, the sign of the difference flips to +0.025 to +0.050 in SAM's favour. See [H19](H19-placement-ranking-is-rate-selection.md). Those matched cells are 13/18 `badnet_a2a`, so they do not settle the question either; this verdict should be re-decided when matched `badnet_a2o`, `blend`, `bpp` and `lf` cells land.

## Why this is a gap and not a reproduction

`papers/reliable_poisoned_sample_detection_.../` claims SAM training amplifies the
backdoor and makes poisoned samples more detectable. Read closely, the claim is
narrower than it sounds:

- It is validated on **five feature-space detectors**: Activation Clustering, Beatrix,
  SCAn, Spectral Signatures, Spectre. All of them cluster penultimate-layer
  representations.
- **PSBD is not among them, and is not cited anywhere in the paper.** Nor is anything
  else based on prediction shift or dropout uncertainty.
- The only **prediction-space** detector they ever measured is STRIP, and it appears
  in three table files that are **commented out** of the submitted paper
  (`tables/cifar0.001.tex`, `0.005`, `0.01`, disabled at `main.tex:100-102`). STRIP's
  mean TPR change under SAM in those files: **-3.4, -3.4, +1.4**. It is the sole
  detector SAM does not help, and it was cut.
- ResNet18 only. No transformer anywhere in the paper. Base optimizer implied SGD,
  and their derivation is a first-order SGD expansion that does not carry to AdamW.

So the open question is not "does SAM help detection" but **"does SAM's benefit
transfer from feature-space detectors to prediction-space ones?"** Nobody has
answered that, their own dropped numbers hint at no, and PSBD is the natural test
case because it is purely prediction-space: it never touches a feature vector.

A negative result here is therefore a finding, not a failed reproduction. It would say
SAM is not a general detection enhancer but a *feature-separability* enhancer, and
that the two families of detector respond to it differently.

## Mechanism that predicts a negative

Their own paper supplies it. Stage 2 of their method exists to undo something SAM
does:

> "to address the increased intra-class variance of clean samples after SAM
> optimization" (Sec. 3.4), and from the introduction, "To prevent SAM from
> increasing feature variance among clean samples, which could cause unstable
> detection".

More clean-feature variance means more unstable clean predictions under perturbation,
which raises clean PSU and collapses the clean-versus-backdoor gap PSU depends on.
Their feature-scaling stage cancels this for feature-space detectors. **PSBD has no
equivalent stage and cannot have one**, because it never looks at features.

That predicts: SAM helps feature-space detection and hurts prediction-shift
detection, through the same mechanism.

## Preliminary evidence, one attack family only

18 SAM checkpoints analysed so far, all `badnet_a2a` (the atypical inverted attack).
Two-sided AUROC, best over rates:

| poison | placement | adam | rho 0.05 | rho 0.1 | rho 0.15 | rho 0.2 |
|---|---|---|---|---|---|---|
| 1% | `post_residual` | **0.938** | 0.627 | 0.724 | 0.666 | 0.737 |
| 5% | `post_residual` | **0.874** | 0.567 | 0.566 | 0.588 | 0.581 |
| 10% | `post_residual` | **0.911** | 0.537 | 0.550 | 0.587 | 0.604 |

Every SAM column is worse than Adam, often much worse. Direction is consistent with
the mechanism above.

**This is not yet evidence for the hypothesis's verdict.** It is one attack, and the
one whose PSU signal is inverted, so it is the least representative case available.
The four normal attacks are in the queue.

## The positive control, run on our own checkpoints with their metrics

CIFAR-10 ViT at 10% poisoning, final block, 500 paired eligible samples, fp32.
`experiments/sam_backdoor_effect/`. SAM minus Adam, averaged over rho in
{0.05, 0.1, 0.15, 0.2}:

| attack | d top-2 TAC | d silhouette | d clean intra-class variance |
|---|---|---|---|
| `badnet_a2o` | **+0.845** | +0.020 | **+0.108** |
| `blend` | **+1.516** | -0.019 | **+0.313** |
| `bpp` | **+0.270** | -0.014 | **+0.177** |
| `lf` | -0.103 | +0.012 | +0.024 |

Three findings, and together they settle it.

**1. SAM does amplify the backdoor, and it scales with rho.** Top-2 TAC, their
"backdoor effect", rises monotonically: `badnet_a2o` 3.25 to 6.07, `blend` 3.30 to
7.57 going from Adam to rho 0.2. So the failure is *not* upstream of PSBD; the
mechanism their paper describes is present in our models.

**2. But separability does not improve, which is what their detectors actually need.**
Silhouette moves by at most ±0.02 and has no trend in rho. Their ResNet18 went 0.19 to
0.32. Ours sits at **0.45 to 0.51 before SAM is applied at all**.

That last number is the explanation. A pretrained ViT already separates clean from
triggered representations more than twice as well as their trained-from-scratch
ResNet18 does *after* SAM. There is almost no headroom left for SAM to add. SAM helps
feature-space detectors when separability is the bottleneck, and on ViT it is not.

**3. The side effect their stage 2 exists to cancel is present and uncancelled.**
Clean intra-class variance rises with rho on every attack (+0.02 to +0.31). Their
Sec. 3.4 introduces feature-scaling specifically "to address the increased intra-class
variance of clean samples after SAM optimization". More clean-feature variance means
more unstable clean predictions under perturbation, which raises clean PSU and closes
the very gap PSU thresholds on.

So SAM hands PSBD the cost without the benefit: an amplified backdoor that PSBD cannot
exploit (it never reads features), plus inflated clean instability that directly
degrades PSU, with no feature-scaling stage available to undo it.

## Still worth running

One feature-space detector (Activation Clustering is cheapest) on these same
checkpoints. If it improves under SAM while PSBD degrades, the detector-family split
is demonstrated within one set of models rather than across two papers.

Their rho sweep says 0.05 to 0.5 all work, so our {0.05, 0.1, 0.15, 0.2} sits inside
their good region and rho choice cannot be blamed for the negative.

## Known divergences from their setup, to state in any writeup

- We do not implement their **feature-scaling stage**, so we are running their "SAM
  without FS" ablation row. That row is much weaker in their own table (79.8 against
  99.8 TPR for Blended/Beatrix). PSBD cannot use their fix, which is the point, but it
  must be said plainly.
- They filter a poisoned **training set** before training; we score held-out inputs
  against an already-trained model.
- SAM-on-AdamW here against SAM-on-SGD there.

## Reproduce

```bash
python psbd_analyze.py --all
python psbd_report.py
```

## Subquestions

1. If SAM hurts PSBD, does it hurt STRIP on the same checkpoints? Both are
   prediction-space, and `defences/baselines.py` already implements STRIP, so this is
   nearly free and would test the family split within our own data.
2. Does the harm scale with rho, or saturate? Their figure shows detector benefit
   saturating by rho 0.15; a mirror-image curve would be strong evidence of one shared
   mechanism acting in opposite directions.
3. Does SAM change the *placement* ranking, or only the level? If blocks 5-8 stays
   best under SAM, the placement result is optimizer-independent and more useful.

---

## Status change, 2026-09-07: DROPPED rather than refuted

This hypothesis is no longer an open question in this project and no claim rests
on it.

The measured effect is +0.009 mean AUROC. Detecting an effect that size against
plausible seed to seed variance requires 10 to 89 training seeds per cell, by
`n >= 8 * (sigma_seed / effect)^2` at 80 percent power. Every checkpoint here is
seed 0, so the variance is unmeasured, and buying enough seeds to conclude that a
0.009 effect is really 0 is not a defensible use of compute.

The decision is therefore to stop measuring it rather than to keep refining the
refutation. SAM checkpoints are 69 percent of the trained set and 75 percent of
all analysis rows, so carrying them also diluted every panel they appeared in,
which is the same unequal-coverage failure mode recorded in the ledger's own
"failure mode" section.

What changed in the code: `scripts/detection_summary.py`, `defence_tables.py` and
`pbs/generate_gaussian_rerun_jobs.py` exclude SAM by default behind an explicit
`--include-sam` flag. The checkpoints and their caches remain on disk. Nothing is
deleted, nothing is claimed.

The 2 observations worth keeping are recorded in `docs/results-report.md` section
4.7, and both are about the backdoor rather than about SAM: SAM does not remove
the backdoor, and it does not resist rank-1 direction removal once the ablation is
done after the final LayerNorm rather than before it.
