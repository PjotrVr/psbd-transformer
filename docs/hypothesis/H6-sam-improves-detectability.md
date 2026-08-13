# H6 — SAM training makes backdoors more detectable

**Status: OPEN.** The sweep is running. But the literature question is now sharp, and
it is the reason this hypothesis is worth testing rather than a reproduction exercise.

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

## What has to be checked before any verdict is written

1. **A positive control on our own checkpoints.** Does SAM actually strengthen the
   backdoor in ViT, as it does in their ResNet18? Measurable with tooling already
   built: backdoor-direction norm and clean-vs-triggered CKA
   (`scripts/backdoor_direction_layers/`) on matched Adam and SAM checkpoints. If SAM
   does *not* amplify the backdoor here, the failure is upstream of PSBD and the
   finding is about transformers or AdamW, not about detector family.
2. **One feature-space detector on the same checkpoints.** If Spectral Signatures or
   Activation Clustering improves under SAM on our ViT models while PSBD degrades,
   the detector-family split is demonstrated directly on one set of models, which is
   a far stronger claim than either half alone.
3. Their rho sweep says 0.05 to 0.5 all work, so our {0.05, 0.1, 0.15, 0.2} is inside
   their good region and rho choice cannot be blamed for a negative.

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
