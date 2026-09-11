# Does SAM training amplify training-set poison detection on ViT (H new)

## Question

Zhang et al. (arXiv 2411.11525, "Reliable poisoned sample detection against
backdoor attacks enhanced by sharpness-aware minimization", local copy under
`papers/reliable_poisoned_sample_detection_against_backdoor_attacks_enhanced_by_sharpness_aware_minimization/`)
claim that training a backdoored model with SAM, rather than vanilla SGD,
widens the gap between poisoned and clean features at the penultimate layer,
which makes off-the-shelf training-set poison detectors more effective. This
project already trains a SAM variant of every checkpoint (`_sam_rho_0_1`) for
an unrelated reason, PSBD's own robustness sweep. This experiment asks whether
the SAM paper's claim, tested on its own terms, reproduces on those checkpoints.

## The paper's setting, exactly

The paper evaluates **training-set poisoned sample detection**: a defender
holds the released poisoned training set itself, trains a model on it, and
must flag which training rows to drop before retraining clean. This is not
PSBD's setting. PSBD probes a finished, frozen model at test time with no
access to the training set at all, and asks whether a single input is
poisoned, not which training rows were.

- **Detectors**: Activation Clustering (AC, Chen et al.), Beatrix (Ma et al.),
  SCAn (Tang et al.), Spectral Signature (SS, Tran et al.), SPECTRE (Hayase et
  al.). Each is a training-set filter, not a test-time input detector, and none
  of them exist in this repo's `detectors/` package, which only holds test-time
  input detectors for PSBD's competitor comparison.
- **Metrics**: TPR, FPR and F1 per detector per attack, each computed against
  the ground-truth poison indices within the target class the detector
  suspects. No fixed removal budget is reported in the main tables. Each
  detector uses its own native decision rule instead (SS flags a multiple of
  the expected poison count, AC flags the smaller of 2 clusters, and so on).
- **Datasets, architectures, poison rate**: CIFAR-10, Tiny ImageNet and GTSRB
  (Tiny and GTSRB pushed to supplementary material for space), on ResNet18,
  VGG19-BN and DenseNet-161 (again, only ResNet18 in the main tables). The
  headline tables use poison rate 5% uniformly. A separate ablation sweeps
  0.1%, 0.5%, 1% and 5% on CIFAR-10 and ResNet18 only, calling anything below
  5% or with a weak trigger (Adaptive-Blend) a "weak backdoor attack".
- **Attacks**: BadNets-A2O, BadNets-A2A, Blended, Label-Consistent, Low-Frequency,
  SSBA, TaCT, Adaptive-Blend, TrojanNN, WaNet. LC only on CIFAR-10, since it can
  only poison the target class.
- **SAM recipe**: `min_theta max_{||eps||_2 <= rho} L(theta + eps)`, plain SAM
  (not the adaptive/ASAM variant), `rho = 0.1`, applied for the entire training
  run in place of the vanilla optimizer, which the paper states is SGD ("Vanilla
  training (SGD)", section 4.3). SAM wraps the whole training, not a phase of it.
  No epoch count, learning rate or schedule is given in the main text or the
  short supplementary section included in the arXiv source (`sec/X_suppl.tex`
  is 16 lines of LaTeX packaging boilerplate, not a hyperparameter appendix).
  Stage-3 of the pipeline additionally reweights features with a PCA projection
  and a covariance-normalized "feature-scaling" step before handing them to the
  base detector, to counter the extra clean-sample variance SAM introduces
  (their Figure 4). This experiment tests SAM alone, without that reweighting,
  because the question is whether SAM by itself moves the needle, which their
  own ablation (`tables/abla.tex`) also isolates: SS TPR on CIFAR-10 BadNets
  rises from 70.8% to 86.0% from SAM alone (no feature-scaling), and 32.9% to
  90.6% on Blended.

## The paper's headline claim, quoted

Abstract: "Extensive experiments on several benchmark datasets show the
reliable detection performance of the proposed method against both weak and
strong backdoor attacks, with significant improvements against various
attacks ($+34.38\%$ TPR on average), over the conventional PSD methods (i.e.,
without SAM enhancement)."

That average is taken over `tables/cifar10.tex` and `tables/gtsrb.tex`
together, all 10 attacks, all 5 detectors (Spectre, SCAn, SS, AC, Beatrix), on
ResNet18 at poison rate 5%. Section 4.2 restates the CIFAR-10 slice of it:
"For CIFAR-10, we improved the True Positive Rate (TPR) by over 25% for four
detection methods." The CIFAR-10 table's own Average row, base PSD next to
"SAM-enhanced PSD" (their name for the full pipeline, base rate first):

| detector | mean delta TPR (points) | mean delta FPR (points) |
| --- | --- | --- |
| Spectre | +29.8 | -1.8 |
| SCAn | +3.2 | -0.0 |
| SS | +24.2 | -1.3 |
| AC | +29.2 | +4.0 |
| Beatrix | +85.5 | -2.3 |

Per-attack TPR before and after is in `tables/cifar10.tex`. For example
BadNets under SS reads 70.8% before, 92.3% after. There is no single shared
FPR or removal budget across the table: each detector keeps its own native
decision rule (SS's top-`k` by score, AC's smaller cluster, and so on), before
and after SAM, exactly as this experiment does.

**This bundles 2 changes, not 1.** "SAM-enhanced PSD" in the paper's own
method section (3.4) is a 3-stage pipeline: train with SAM, then reweight the
resulting features with a PCA projection and a covariance whitening
("feature-scaling", stage 2), then hand the off-the-shelf detector the scaled
features. The `+34.38%` and `+25%` headline numbers are for that whole
pipeline, not for the optimizer swap alone. Their own ablation (`tables/abla.tex`)
isolates the 2 stages and reports SAM alone, no feature-scaling, on CIFAR-10
ResNet18: SS TPR rises 70.8% to 86.0% for BadNets ($+15.2$ points) and 32.9%
to 90.6% for Blended ($+57.7$ points). Beatrix rises 56.6% to 98.4%
($+41.8$ points) for BadNets and 5.0% to 79.8% ($+74.8$ points) for Blended.
That SAM-only ablation, not the headline number, is the fair comparison to
this experiment, since this experiment also isolates SAM without
feature-scaling (see "Our recipe against theirs" below for why).

**No architecture in the paper is a transformer.** Its 3 architectures are
ResNet18, VGG19-BN and DenseNet-161, all convolutional, on CIFAR-10, Tiny
ImageNet and GTSRB. The only place "vision transformer" appears in the paper
at all is 1 sentence in the related-work section citing a different paper
(Chen et al. 2021) for the claim that SAM generalizes across architectures.
The SAM paper itself never trains or evaluates one.

## Our recipe against theirs

`training/sam.py` implements the same 2-pass SAM update
(`first_step` ascends to the worst-case point, `second_step` restores the
original weights and takes the base optimizer's step there) at the same
default `rho = 0.1` (`cli/train_backdoor.py --rho`, default `0.1`), applied
for the entire training run with no phase gating, `adaptive=False` by default,
which all match the paper's own equation and its `rho = 0.1` setting.

Deviations:

1. **Base optimizer**: the paper's vanilla and SAM runs both wrap SGD ("Vanilla
   training (SGD)"). Every checkpoint in this project, SAM included, wraps
   Adam (`training.loop.build_optimizer`), because Adam at a constant learning
   rate is this project's uniform recipe across all 15-epoch runs, for
   comparability across attacks and datasets, not something chosen per attack.
2. **Architecture**: the paper never evaluates a Vision Transformer. Its 3
   architectures are ResNet18, VGG19-BN and DenseNet-161, all convolutional.
   This experiment uses ViT-B/16, the architecture PSBD itself targets.
3. **Datasets and poison rate**: the paper's headline numbers are CIFAR-10 at
   5%. This experiment uses CIFAR-100 (this project's dataset priority) at 1%
   and 5%, closer to the paper's own "weak backdoor attack" ablation range
   (0.1% to 1%) than to its headline rate.
4. **Epoch count, schedule, batch size**: not stated anywhere in the paper's
   available text, so no comparison is possible beyond noting that this
   project trains every checkpoint for a uniform 15 epochs at a constant
   learning rate (`docs/` canon, `.claude/CLAUDE.md`), which a from-scratch
   ResNet18 training-set-detection paper is very unlikely to match.
5. **Stage-3 feature-scaling**: not implemented here, deliberately (see above).
6. **Detectors ported**: only Spectral Signature and Activation Clustering are
   implemented, in project style, directly against the original papers (Tran
   et al., Chen et al.), not against the SAM paper's own code. Beatrix, SCAn
   and SPECTRE are out of scope for this experiment.

## Method

For 6 matched Adam/SAM pairs (`checkpoints/vit_cifar100_<attack>_<rate>` and
`..._sam_rho_0_1`, all trained on CIFAR-100, `badnet_a2o`, `blend` and `wanet`
at 1% and 5% poisoning. LF and BPP have no CIFAR-100 SAM sweep at these exact
rates and are left out rather than substituted):

1. **Rebuild the poisoned training set.** `args.json`'s `seed` (default 0
   where unrecorded, matching the checkpoint naming convention's unmarked
   seed 0) and `max_samples` (`None`, the full 50000-image train set) feed
   into the identical call sequence `cli.train_backdoor` uses:
   `data.loading.load_clean_datasets`, `limit_dataset`,
   `attacks.poisoning.choose_poison_indices` (or `choose_indices_with_cover`
   for WaNet, whose cover rate is twice its poison rate). The rebuilt poison
   index count is asserted against `args.json`'s recorded `n_poisoned` before
   anything downstream runs.
2. **Restrict to the target class.** Both detectors are per-suspected-class
   methods, so the working set is exactly the target class after poisoning:
   every poisoned sample (which lands there under `all_to_one`) plus every
   image whose original label already was the target class. This is at most
   3000 images at these rates on CIFAR-100 (2500 poisoned, roughly 500 native
   clean), always under the 10000-image cap the task set, so no further random
   subsampling of "a share of the others" was needed or applied: the SS and AC
   detectors never see any class but the target one, so extracting features
   for other classes would extract features nothing downstream reads.
3. **Extract features.** The class token after `network.encoder.ln`, the
   classifier's own final LayerNorm (`torchvision`'s `VisionTransformer.Encoder`
   applies it internally, so it is invisible to `analysis.features`' block
   hooks, which stop at the last `EncoderBlock`'s raw output). Batch 256, no
   gradient, `torch.inference_mode()`.
4. **Spectral Signature** (Tran et al.): center the target-class features,
   take the top right-singular vector of the centered matrix, score each
   sample by its squared projection onto that vector, flag the top
   `1.5 * n_poisoned` scores.
5. **Activation Clustering** (Chen et al.): PCA to 10 dimensions, 2-means, flag
   the smaller cluster.
6. **Diagnostics**: the silhouette coefficient of the true poisoned/clean split
   on the same PCA-10 features AC clusters, and the ratio of the top to the
   second singular value of the centered feature matrix (the same SVD SS
   already computes), which is the paper's own kind of separability evidence
   (their Figure 4 variance comparison and Figure 12 t-SNE).

Code: `measure.py`. Run with:

    PYTHONPATH=. .venv/bin/python -m experiments.sam_training_set_detection.measure

Runs on the login node's A100 in under 2 minutes for all 12 checkpoints (about
8 seconds each). Writes 1 JSON per pair to
`results/_experiments/sam_training_set_detection/<dataset>_<attack>_<rate>.json`.

## Result

At the same native decision rule each detector uses in the paper (SS's
top-`1.5 * n_poisoned` by score, AC's smaller-cluster rule), does a gain of
the paper's size (+15 to +58 points, SAM alone, their own ablation, or +25 to
+85 points for the full SAM+feature-scaling pipeline) appear on ViT at 1% and
5% poisoning? No. The largest SS TPR movement from SAM alone across the 6
pairs is +2.0 points (Blend, 1%), and the smallest is -1.0 points (WaNet,
1%), against the paper's own SAM-alone range of +15.2 to +57.7 points on the
same detector. AC's TPR moves by at most +1.0 points in either direction.

| pair | optimizer | poisoned | SS TPR | SS FPR | AC TPR | AC FPR | silhouette | singular ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cifar100 badnet_a2o 1% | adam | 500 | 0.734 | 0.766 | 0.000 | 1.000 | 0.624 | 3.18 |
| cifar100 badnet_a2o 1% | sam | 500 | 0.748 | 0.752 | 0.000 | 1.000 | 0.529 | 2.60 |
| cifar100 badnet_a2o 5% | adam | 2500 | 1.000 | 1.000 | 0.000 | 1.000 | 0.639 | 2.81 |
| cifar100 badnet_a2o 5% | sam | 2500 | 1.000 | 1.000 | 0.000 | 1.000 | 0.570 | 1.87 |
| cifar100 blend 1% | adam | 500 | 0.794 | 0.706 | 0.000 | 1.000 | 0.505 | 2.26 |
| cifar100 blend 1% | sam | 500 | 0.814 | 0.686 | 0.000 | 1.000 | 0.577 | 2.99 |
| cifar100 blend 5% | adam | 2500 | 1.000 | 1.000 | 0.000 | 1.000 | 0.749 | 3.58 |
| cifar100 blend 5% | sam | 2500 | 1.000 | 1.000 | 0.000 | 1.000 | 0.717 | 3.50 |
| cifar100 wanet 1% | adam | 500 | 0.758 | 0.742 | 0.970 | 0.000 | 0.387 | 2.26 |
| cifar100 wanet 1% | sam | 500 | 0.748 | 0.752 | 0.976 | 0.000 | 0.358 | 2.12 |
| cifar100 wanet 5% | adam | 2500 | 1.000 | 1.000 | 0.005 | 0.998 | 0.417 | 1.89 |
| cifar100 wanet 5% | sam | 2500 | 1.000 | 1.000 | 0.015 | 1.000 | 0.412 | 2.11 |

At 5% the target class has 500 native clean images against 2500 poisoned ones
(CIFAR-100 has 100 classes, so roughly 500 images per class to start with),
so Spectral Signature's own removal rule, flag the top `1.5 * n_poisoned`
scores, asks for 3750 flags out of a 3000-image class and degenerates to
flagging everyone (TPR = FPR = 1.0) for both optimizers alike. Those 3 rows
carry no information about SAM either way and are excluded from the averages
below. SS is only informative at 1%.

At 1%, the SAM checkpoint's SS TPR moves by +1.4, +2.0 and -1.0 points across
the 3 attacks (mean +0.8 points), against the paper's own CIFAR-10 ablation
gain of +15.2 points for BadNets and +57.7 points for Blended from SAM alone.
Activation Clustering never once finds the true poisoned cluster for BadNets
or Blend, in either optimizer, at either rate: the smaller of its 2 clusters
is consistently the roughly 500 native clean images, not the poisoned ones,
so AC is inverted (TPR 0, FPR 1) here regardless of SAM. For WaNet, AC is
almost perfect at 1% and almost blind at 5%, again in both optimizers alike,
with SAM shifting AC's TPR by less than 1 point in either regime. Averaged
over all 6 pairs, the silhouette coefficient moves by -0.026 under SAM and the
singular-value ratio by -0.13, both the wrong sign for the paper's own claim
that SAM widens class separation (their reported silhouette gains on
ResNet18/CIFAR-10 were +0.13 for BadNets and +0.26 for SSBA).

## Why our gain is far smaller

Three reasons, in order of how much evidence this experiment actually has for
each.

The clearest reason is a ceiling effect: ViT-B/16's target-class features are
already far more separable under plain Adam than ResNet18's are under plain
SGD. Our Adam-only SS TPR baseline is already 73 to 79% at 1% poisoning for
all 3 attacks, and saturates outright at 5%. The paper's own vanilla ResNet18
SS baseline is comparable for BadNets (70.8%) but far lower for a weaker
trigger like Blend (32.9%). A detector already scoring near its ceiling under
the base optimizer has little room left for SAM to add, while a detector
starting at 33% has 60-odd points of headroom to gain. This alone would
predict exactly the asymmetry seen: Blend, the attack with the largest
CNN-side headroom in the paper, is also the attack with our largest (if
still small) SAM movement, +2.0 points.

The second reason is the confound already flagged above: even the paper's
own SAM-alone ablation, the fair comparison, still describes a CNN trained
with SGD, not a transformer trained with Adam. Everything downstream of that
choice, including how gradient sharpness interacts with the optimizer's own
preconditioning, is untested by the paper and unaddressed by its two-layer
ReLU proposition, which assumes a single hidden layer with no attention, no
LayerNorm and no residual stream at all.

The third reason is that the paper's own proposed mechanism does not show up
here even as a side effect. Its argument for why SAM should help is that
SAM increases the weight norm and TAC of "backdoor neurons" specifically,
which should show up as tighter, more separated clusters, the silhouette and
singular-ratio diagnostics this experiment also measures. Averaged over all 6
pairs, silhouette moves by -0.026 and the singular ratio by -0.13 under SAM,
both the wrong sign, against the paper's own reported +0.13 to +0.26
silhouette gains on ResNet18/CIFAR-10. If SAM were sharpening a
backdoor-relevant direction in ViT's residual stream the way it does in a
convolutional channel, this diagnostic should have moved the same way even
where TPR did not, and it did not.

## Conclusion

SAM's claimed training-set amplification effect does not reproduce on
ViT-B/16 penultimate features at 1% and 5% poisoning on CIFAR-100. Spectral
Signature moves by about 1 point either way at the only rate where its
removal rule is informative, Activation Clustering's success or failure is
set entirely by the attack, not the optimizer, and the paper's own
separability diagnostics, silhouette and singular ratio, drift slightly
negative under SAM on average rather than the positive amplification the
paper reports on ResNet18. The effect the paper documents is architecture and
optimizer specific (SGD-based CNNs), not a property of SAM training in
general, at least not at the poison rates and target-class sizes CIFAR-100
produces. This has no bearing on PSBD's own claims either way: PSBD never
clusters training-set features and never assumes SAM training, so a null
result for SAM-enhanced training-set filtering on ViT says nothing about
whether PSBD's frozen-model, test-time perturbation mechanism works, and the 2
should not be read as the same question answered twice.
