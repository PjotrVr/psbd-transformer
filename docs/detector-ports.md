# Porting the 4 published detectors to ViT

The modules under `detectors/` implement STRIP, SCALE-UP, IBD-PSC and TeCo from
their papers. Each module header lists its deviations from the paper in a line or
2. This document is the full record behind those lines: what the paper says, what
the released code actually does, what this port does and why, and what was checked
numerically. Every deviation is stated rather than absorbed, because a number from
this repository is only comparable to a published number once the differences are
known.

The shared rules are in `detectors/__init__.py`: every score is low for poisoned,
every method that needs clean data gets the same 2000-sample validation split and
every module states its forward-pass cost.

## STRIP (Gao et al., ACSAC 2019)

Paper: "STRIP: A Defence Against Trojan Attacks on Deep Neural Networks",
arXiv:1902.06531. Statistic in Section IV-D, Equations (2) to (4).

1. **Entropy in nats, not bits.** Eq. (2) uses log2 and the port uses the
   natural log. The 2 differ by the constant factor log(2), so no ranking, AUROC
   or quantile position changes. Only the printed threshold value is scaled.
2. **Superimposition in pixel space, then saturated.** The released reference
   calls `cv2.addWeighted(background, 1, overlay, 1, 0)` on uint8 arrays: both
   weights 1, and OpenCV saturating-casts the result at 255. An earlier version of
   this port summed the 2 already-normalized tensors instead. That is a different
   operation: adding in normalized space gives (p1 - m)/s + (p2 - m)/s =
   (p1 + p2 - 2m)/s, the pixel-space sum displaced by a further -m/s per channel,
   which is 2.43 units on CIFAR-10 channel 0, and it also skipped the saturation.
   The loader delivers normalized tensors, so the round trip is denormalize, add,
   saturate, renormalize.
3. **A fixed overlay set.** The overlays are the first N images of the clean
   validation split and are shared by every scored input, rather than resampled per
   input. Every input then faces the same perturbation set, which removes overlay
   choice as a source of per-sample variance in the comparison.
4. **N = 8.** The paper defaults to N = 100 and later reports 10 as sufficient.
   The port uses 8, the value every recorded number in this repository was
   produced with, so ported numbers stay comparable to the earlier ones.

STRIP's score is returned unnegated. Its claim is that a triggered input has low
entropy under superimposition, and low already means poisoned in the shared
convention. Negating it once produced AUROC 0.000, perfect separation with the
sign reversed, which is exactly the failure the `auroc_two_sided` diagnostic
field in `defences.decision.detection_report` surfaces.

## SCALE-UP (Guo et al., ICLR 2023)

Paper: "SCALE-UP: An Efficient Black-box Input-level Backdoor Detection via
Analyzing Scaled Prediction Consistency", arXiv:2302.03251, OpenReview
o0LFPcoFKnr. Statistic in Section 4.2, Equation (2). Data-limited variant in
Section 4.3, Equations (3) and (4). Theorem 1 proves the limiting case for an RBF
kernel regressor at a 50% poisoning rate.

The data-limited variant is worth roughly +0.005 AUROC on average in the paper's
own Tables 1 and 2, which is why data-free is the default here.

1. **Scaling set.** The paper writes S = {3, 5, 7, 9, 11} in Section 4.2, but
   introduces it with "e.g." and never restates it in the experimental settings.
   The authors' released code uses `range(1, 12)`, so S = {1, ..., 11}, which puts
   n = 1 inside the average where it is trivially consistent and lifts the floor of
   SPC from 0 to 1/11. Those are 2 different statistics on 2 different supports.
   The port defaults to the paper's set and offers the code's set as
   `OFFICIAL_CODE_SCALES`, so whichever produced a number is recorded rather than
   inherited.
2. **Per-class statistics on a shared budget.** Eq. (3) is per class and the paper
   budgets 100 benign samples per class. Every method here shares the single
   2000-sample clean validation split, which on CIFAR-100 is about 20 samples per
   class and can be 0 for a class the split happened to miss. A class with fewer
   than `MIN_CLASS_SAMPLES` falls back to the pooled mean and standard deviation,
   which the paper does not specify. Both published third-party ports sidestep this
   by using 1 global mean and standard deviation for every class, which is a
   different method from Eq. (4). A second fallback handles degenerate spread: SPC
   lives on a grid of 1/|S| and takes only |S| + 1 distinct values, so a class
   whose few clean samples all landed on the same grid point has sigma_i exactly 0.
   Left alone that is clamped to `STD_FLOOR` and Eq. (4) returns z-scores of order
   1e5 for that class alone, which does not change AUROC but makes the score scale
   meaningless and any absolute threshold unusable. This was observed on
   vit_cifar100 with the shared budget.
3. **Calibration reference.** Eq. (2) always measures consistency against C(x),
   the model's own prediction. Both third-party ports instead compare against the
   ground-truth validation label when computing Eq. (3), which disagrees with
   Eq. (2) exactly on the samples the model gets wrong. The port uses C(x)
   throughout. The grouping into X_i uses the validation sample's true class, which
   is what Eq. (3) says.
4. **No input noise.** The authors' released code adds 0.02 * U[0, 1) to every
   test sample before scaling. That appears in no equation, has no ablation and is
   not reproduced.
5. **No prediction-correctness masking.** The third-party ports drop every sample
   whose prediction disagrees with its dataset label, which on the poisoned side
   keeps only successfully attacked images. This project already owns that decision
   through `attacks.poisoning.AttackSuccessSet`, so the detector scores every sample
   the loader serves and the eligibility rule stays in 1 place.

`amplify_pixels` multiplies and clips in pixel space, as Section 4.2 states ("we
constrain n * x in [0, 1] during the multiplication process"). Scaling the
normalized tensor directly would amplify the dataset mean as though it were signal
and land the clip, where the method's whole nonlinearity lives, in the wrong
place. Of the 3 published implementations only backdoor-toolbox gets this ordering
right.

## IBD-PSC (Hou et al., ICML 2024)

Paper: "IBD-PSC: Input-level Backdoor Detection via Parameter-oriented Scaling
Consistency", arXiv:2405.09786, PMLR v235 hou24a. Amplification in Section 4.3,
Equation (2). Layer selection in Equation (3) and Algorithm 1. Score in Section
4.4, Equation (4). Defaults from Section 5.1: omega 1.5, n 5, xi 0.6, T 0.9.

**The ViT deviation, which is the substantive one.** The paper scales BatchNorm2d
and only BatchNorm2d. Its released code in BackdoorBox matches, filtering on
`isinstance(module, torch.nn.BatchNorm2d)`. A ViT or a Swin contains no BatchNorm
at all, so `count_BN_layers` returns 0, `sorted_indices` is empty, Algorithm 1
loops over `range(1, 0)` and `start_index` comes back None. IBD-PSC as published
is not runnable on this project's architectures.

The port amplifies `nn.LayerNorm` instead. The substitution is exact at the level
of what Eq. (2) does to a layer's output, because both normalize first and apply
the affine map second:

    omega*gamma * x_hat + omega*beta = omega * (gamma * x_hat + beta)
                                     = omega * (the layer's original output)

What does not carry over is the architectural claim behind the choice. The paper's
L is a count of BatchNorm layers, 1 per convolutional stage, whereas a ViT block
contains 2 LayerNorms sitting on the 2 branch inputs of a residual stream.
Amplifying 1 of those scales a branch, not the stream, so the effect on the logits
is weaker per layer than a BatchNorm scaling is in a ConvNet. Algorithm 1 absorbs
that difference by construction, since it selects k from measured clean error
rather than from a fixed depth, but the resulting k is not comparable to a
published k.

Further deviations:

1. **Layer count in the ensemble.** Eq. (4) sums over i = k .. k+n-1 amplified
   layers. The released code amplifies `sorted_indices[:layer_index+1]`, so
   k+1 .. k+n, 1 more layer at every position than the equation. The port follows
   the equation.
2. **Algorithm 1's range.** The paper loops i = 1 to L. The released code loops
   `range(1, layer_num)`, so it never tests the all-layers configuration and
   returns None when the error rate never crosses xi, which then crashes
   downstream with no message. The port follows the paper, tests i = 1..L and
   falls back to k = L when no i crosses, which is the value Algorithm 1 holds at
   loop exit.
3. **Ensemble clamping.** When k is close to L the window k..k+n-1 runs past L,
   which is undefined. The paper does not address it because its BatchNorm counts
   are large. The port keeps only the members with i <= L, so a late k gives a
   smaller ensemble rather than an invalid one, and records how many members were
   used.
4. **Data budget.** The paper states 100 benign samples. The released demo uses
   2000. The port uses the shared clean validation split, so every method sees the
   same data.
5. **No deep copies.** The released code calls `copy.deepcopy(self.model)` once per
   ensemble member per batch, which is 5 full model copies per batch. The port
   writes the amplified parameters in place and restores them from saved clones,
   which is numerically exact and allocates no second model.

`amplifiable_norm_layers` reverses definition order, as the released code does
with `list(reversed(range(layer_num)))`, so element 0 is the layer nearest the
head. Definition order equals execution order for torchvision's ViT and Swin, so
the reversal really is depth ordering. It would not be for an architecture that
declares its modules out of forward order, which is a latent trap the released
code shares.

## TeCo (Liu et al., CVPR 2023)

Paper: "Detecting Backdoors During the Inference Stage Based on Corruption
Robustness Consistency", arXiv:2303.18191. Score in Section 4.2, Algorithm 1.
Decision rule in Equation (4). There is no numbered equation for the score itself.

Forward-pass cost is K * N + 1 per input, 71 at K = 14 and N = 5 (76 at the
paper's K = 15). Algorithm 1's break would allow an early exit, but the released
code evaluates all K * N corruptions regardless and applies the break to cached
predictions, which gives an identical statistic at the full cost. That is the cost
reported, because it is what a batched implementation pays.

1. **14 corruptions, not 15.** The paper uses the `imagecorruptions` package, which
   is not installed here and cannot be added without a new hard dependency (it
   requires opencv-python, also absent). Every corruption is therefore
   reimplemented in torch against the reference source, and 14 of the 15 reproduce
   it: gaussian_noise, shot_noise, impulse_noise, defocus_blur, glass_blur,
   motion_blur, zoom_blur, snow, fog, brightness, contrast, elastic_transform,
   pixelate, jpeg_compression. Frost is omitted, because it composites 1 of 6
   photographs of frosted glass that ship as binary assets inside the package, and
   there is no way to reproduce those from a formula. The statistic is a standard
   deviation over the corruption types, so dropping 1 of 15 changes the sample it
   is computed over, and a TeCo number from this repository is not numerically
   identical to a published one.
2. **Corruption applied to the pristine image.** Algorithm 1 line 5 applies
   D_k^n to x. Both released implementations instead mutate the image list in
   place and never restore it, so severity 2 lands on the output of severity 1 and
   the second corruption type lands on an image that has already been through all
   5 severities of the first. By the last corruption type each image has
   accumulated 70 sequential operations. That is a cumulative composition rather
   than D_k^n, so the published 0.943 AUROC was produced by a different statistic
   from the one Algorithm 1 defines. The port implements the algorithm, and a
   faithful reproduction is expected to differ from the published number.
3. **Deviation measure.** The released code names the variable `mad`, which reads
   as mean absolute deviation, but the line is `np.std(indexs)`, the population
   standard deviation with ddof=0. The port computes the population standard
   deviation. The paper's own ablation finds mean deviation statistically
   indistinguishable and the coefficient of variation clearly worse, so the
   unnormalized choice is deliberate. The choice rescales every score by a constant
   and cannot move AUROC, but it does move any absolute threshold, including the
   paper's empirical gamma = 1.
4. **Randomness per batch.** The motion blur angle and the snow angle are drawn
   once per batch where the reference draws them once per image. Sharing them
   within a batch removes a per-image nuisance term from a statistic that compares
   corruption types within 1 image, and it is what makes the corruption batchable
   at all. Per-pixel noise is still drawn per image, as in the reference.
5. **Operator substitutions**, each verified numerically against a numpy or scipy
   reference where a dependency was missing. The disk kernel's antialiasing
   Gaussian replaces `cv2.GaussianBlur` and agrees to 5e-4. The pixelate
   downsample uses torch area resampling in place of PIL BOX, which agree exactly
   at integer downscale factors and approximately otherwise. `_clipped_zoom`
   matches `scipy.ndimage.zoom` with `grid_mode=False` (`align_corners=True` in
   torch) to 2e-6. `elastic_transform` samples with torch reflection padding,
   which reflects about the edge pixel where scipy's `mode="reflect"` repeats it,
   a difference confined to border pixels.

Corruption is applied in [0, 1] pixel space at the dataset's native resolution,
before the model wrapper upscales to 224, which is where this project applies
triggers too and what the reference does for CIFAR. Results are requantized to the
8-bit grid after each corruption, because the reference operates on uint8 arrays
throughout and that quantization is part of several of the operators.
