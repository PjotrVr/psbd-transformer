# SentiNet, region transplant against localized universal attacks

SentiNet localizes the region of an input that drives the model's prediction with Grad-CAM, transplants that region onto a set of clean images and asks 2 questions of the model: how many of the clean images now take the input's label and how confident the model stays when the same region is filled with noise instead. A trigger is a small region that answers yes to both, a benign salient region fails at least 1 of them and the decision is a curve fitted over clean inputs in that 2-dimensional plane. This page records what the paper defines, what the 3 released reimplementations do, what the port under `detectors/sentinet.py` runs on ViT and Swin and where they diverge.

## Citation

Chou et al., "SentiNet: Detecting Localized Universal Attacks Against Deep Learning Systems", IEEE S&P Workshops (DLS) 2020, arXiv:1812.00292 (v4 read). Class proposal is Algorithm 1 and mask generation Algorithm 2, both in Section III-A, with the Grad-CAM definition of Selvaraju et al. restated there. The 2 test statistics are Algorithm 3 in Section III-B1 and the decision boundary is Algorithm 4 in Section III-B2. The paper cites no released code, so the 3 reimplementations below are the only executable references.

The reimplementations read are `third_party/Beatrix/defenses/SentiNet/SentiNet.py` at commit `685827e`, `third_party/BackdoorBench/detection_infer/sentinet.py` at commit `f02e353` and `third_party/backdoor-toolbox/other_defenses_tool_box/sentinet.py` at commit `9d4d909`. The Beatrix file is the reference for `fooled`, `avgConf` and the boundary fit, as the plan fixed. The other 2 are read for their departures from it.

## Threat model and data requirement

The adversary mounts a localized universal attack, a contiguous region that hijacks the prediction of any image it is placed on, whether by a trojaned model, a poisoned one or an adversarial patch against a clean one. The defender holds the deployed model white-box, since Grad-CAM differentiates a logit with respect to an intermediate activation. The defender also holds a set $X$ of benign test images "often shipped together with deployed models". The paper uses 100 images for $X$ on every network and about 400 further benign inputs to plot the decision boundary in its Figure 4.

The port draws both from the shared 2000-sample clean validation split. The first 100 images of the split are $X$, the 100 inert noise images are drawn once from the run's seed and the boundary is fitted on the split's own 2000 $(avgConf, fooled)$ points. No label is read anywhere: `fooled` counts against the model's prediction on the input and the clean points need no class. `DATA_REQUIREMENT` therefore records 100 clean images, unlabelled, plus the split for the envelope.

## Mechanism

Grad-CAM and Algorithms 2 to 4 follow in the paper's own notation. The symbol table underneath defines every symbol and the descriptive form after it renames without rederiving. Algorithm 1, the selective-search class proposal, is omitted from the port and from every reimplementation for the reason given in deviation 2 below.

$$
\begin{aligned}
\alpha_c^k &= \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A^k_{ij}} \\
L^c_{Grad\text{-}CAM} &= ReLU\left( \sum_k \alpha_c^k A^k \right)
\end{aligned}
$$

```
Algorithm 2 MaskGeneration
in:  f_m, x, (y, conf) = f_m(x), C the proposed classes
out: M, masks for candidate regions
  mask_y = MaskGradCAM(f_m, x, y)
  M = { (mask_y - MaskGradCAM(f_m, x, y_p), conf_p) : (y_p, conf_p) in C }
  return {mask_y} union M

Algorithm 3 Testing
in:  f_m, x, y the class of x, M the proposed masks, X the benign test images
out: Fooled, AvgConf
  R    = { x * mask : mask in M }
  IP   = InertPattern(M)
  X_R  = Overlay(X, R)
  X_IP = Overlay(X, IP)
  fooled_yR = 0, avg_confIP = 0
  for x_R, x_IP in X_R, X_IP
      (y_R, conf_R), (y_IP, conf_IP) = f(x_R), f(x_IP)
      if y_R == y
          fooled_yR += 1
      avg_confIP += conf_IP
  avg_confIP = avg_confIP / |X|
  return fooled_yR, avg_confIP

Algorithm 4 DecisionBoundary
in:  B, the sampled behaviour of f_m on clean inputs
out: f_curve, d the acceptable distance from f_curve
  f_curve = ApproximateCurve(OutPts(B))
  avg_d = 0
  for (x, y) in B
      if f_curve(x) > y
          avg_d += COBYLA((y, x), f_curve)
  d = avg_d / |B|
  return f_curve, d
```

| Symbol | Meaning |
|---|---|
| $y^c$ | the model's logit for class $c$ |
| $A^k$ | the $k$-th feature map of the layer Grad-CAM reads, indexed by spatial position $(i, j)$ |
| $Z$ | the number of spatial positions in a feature map |
| $\alpha_c^k$ | the neuron importance weight of feature map $k$ for class $c$ |
| $L^c$ | the Grad-CAM heatmap for class $c$, binarized at 15% of its maximum in the paper |
| $f_m$ | the deployed model, returning a label and a confidence |
| $x, y$ | the input under test and the model's prediction on it |
| $C$ | the classes proposed by Algorithm 1, with their confidences |
| $M, R$ | the candidate masks and the regions of $x$ they cut out |
| $X$ | the benign test images the regions are overlaid on |
| $IP$ | the inert pattern, random noise by default, filling the same region |
| $X_R, X_{IP}$ | the overlays of $X$ with the region and with the inert pattern |
| $fooled$ | the count of overlays whose label equals $y$ |
| $avgConf$ | the mean confidence over the inert overlays |
| $B$ | the $(avgConf, fooled)$ points of clean inputs |
| $OutPts$ | the points with the highest $fooled$ in each $avgConf$ interval |
| $f_{curve}$ | the non-linear least-squares curve through those points, a parabola in Figure 4 |
| $d$ | the mean COBYLA distance from the curve over the clean points that lie outside it |

The same pipeline follows with descriptive names in place of the paper's symbols, in the form the port computes on a transformer. The structure is unchanged and only the names differ, with the feature maps replaced by token features and the spatial positions by patch tokens.

$$
\begin{aligned}
\text{token\_weight} &= \text{mean over patch tokens of } \frac{\partial\, \text{predicted logit}}{\partial\, \text{token}} \\
\text{cam}[\text{patch}] &= ReLU\big( \text{token\_weight} \cdot \text{token}[\text{patch}] \big), \text{ scaled per image to } [0, 1] \\
\text{mask} &= \big[ \text{upsampled cam} \ge 0.85 \big] \\
\text{fooled} &= \text{fraction of the clean overlays predicted as the input's label once its region is pasted on them} \\
\text{avg\_conf} &= \text{mean max softmax over the same overlays with uniform noise pasted instead} \\
\text{envelope} &= \text{quadratic through the 2 largest clean fooled values per avg\_conf bin of 0.04} \\
\text{residual} &= \text{fooled} - \text{envelope}(\text{avg\_conf}) \\
\text{sentinet\_score} &= -\text{residual}
\end{aligned}
$$

A trigger is a small region that hijacks whatever it lands on, so pasting it drags the overlays to its label and `fooled` is high, while noise in a small region leaves the overlays' own evidence intact and `avg_conf` stays high. A benign salient region fails 1 of the 2 tests. Too weak to hijack, it leaves `fooled` low. Too large, it is occluded by the noise fill and `avg_conf` drops. Both benign cases sit under the envelope and a triggered input sits above it, which is the top-right corner of the paper's Figure 4.

## What the released code does

The 3 reimplementations agree on the skeleton and disagree on most of the settings. All 3 drop Algorithm 1 and the mask subtraction of Algorithm 2 and take the Grad-CAM map of the predicted class as the only mask. Beatrix and BackdoorBench share 1 `SentiNet` class, so they agree with each other and disagree with backdoor-toolbox.

Beatrix (`SentiNet.py`) runs `pytorch_grad_cam`'s `GradCAM` on `layer4[-1]` of a PreActResNet18 with `target_category=None` for the predicted class (lines 403 to 426) and binarizes the map, which that library already scales to $[0, 1]$, at `MASK_COND` 0.85 (lines 48 and 427). Its main sets `use_truemask = True` (line 561), which replaces the Grad-CAM mask by the attack's true trigger mask on the poisoned side (line 432), so its reported numbers are oracle numbers. `_superimpose` at line 265 composites `background * mask + overlay * (1 - mask)` on uint8 arrays. `_get_entropy` draws `n_sample` = 10 overlays at random from the test set per input (lines 40 and 272) and 10 uniform noise images (line 273). Its inert composite at line 281 is `_superimpose(background, inert_pattern, mask)`, the input's region on a noise background, which is the reverse of Algorithm 3's noise inside the region on a clean image. `fooled` counts against `sentinet_labels`, the model's prediction on the input (lines 459 to 466). `DecisionBoundary` at line 477 bins `avgConf` at `step` 0.04 with half-open bins `(step * i, step * (i + 1)]` (lines 481 to 491), keeps the 2 largest `fooled` per bin (lines 496 to 497), fits `a x^2 + b x + c` with `scipy.optimize.curve_fit` (line 521) and sets `d` as the mean `fmin_cobyla` distance over the boundary points above the curve (lines 537 to 541).

BackdoorBench (`sentinet.py`) reuses that class with 2 changes and no curve. Its overlay set is the whole clean set rather than 10 random draws (`index_overlay = np.arange(len(dataset))`, line 161), with the clean set built as `clean_sample_num / num_classes` images per class (lines 341 to 352). Its inert composite at line 170 is the same reversed one. Masks are `grayscale_cams >= mask_cond` (line 377) and labels are the model's predictions (lines 379 to 393). The decision at line 398 is `avgconf > 0.9` alone, so `fooled` is computed and never read.

backdoor-toolbox (`sentinet.py`) follows Algorithm 3 for both composites, `adv_input[:, mask] = _input[:, mask]` and `inert_input[:, mask] = normalizer(rand)[:, mask]` at lines 94 to 95, on a ResNet at 224 pixels through `layer4` (line 77). Its mask is the top 15% of CAM cells by area (lines 80 to 81), a fixed size rather than a threshold. It holds `N` = 100 clean images for $X$ and 400 further validation images for the curve (lines 29 and 63), bins `avgConf` at 0.02 with 1 maximum per bin placed at the bin centre (lines 115 to 128), fits the quadratic with `sklearn` (lines 132 to 142), sets `d_thr` as the mean COBYLA distance over the clean points below the curve (lines 148 to 159), lifts the curve by a `y_plus` found by a 0.001 line search (lines 161 to 172) and scores every input by the signed perpendicular distance, negative below the curve (lines 325 to 333). A `defense_fpr` then overrides `d_thr` with a quantile of the clean distances (lines 336 to 340).

2 of the toolbox's choices are defects for an inference-time detector. `fooled` counts composites predicted as `_label`, the loader's true label of the input (lines 99 and 228). On the poisoned side it counts them as `poison_label`, the attack's target (line 286), with `c_label + 1` for all-to-all (line 288). None of these is a label a defender has. For `badnet`, `TaCT`, `trojan`, `dynamic` and `adaptive_patch` it also pastes the attack's true trigger region rather than the Grad-CAM mask (lines 250 to 281), computing the trigger mask from `poison_input - _input`, so on those attacks the localization step is bypassed with an oracle.

## What this port does on ViT

1. **CAM at the input of the last block on ViT and the output of the last block on Swin.** The paper reads Grad-CAM at "the model's final pooling layer" of a ConvNet and every reimplementation hooks `layer4` of a ResNet. torchvision's `VisionTransformer` reads `x[:, 0]` after the encoder, so the gradient of any logit with respect to the last block's OUTPUT patch tokens is exactly 0, which `tests/test_capture_primitives.py` pins and `tests/test_detectors_sentinet.py` shows produces an all-zero map. The port hooks the last block's INPUT on ViT, layer 11 of 12 in `captured_layers` numbering, the deepest site whose patch tokens still carry gradient. Swin's average pool reads every token, so its site is the last block's output, layer 24 of 24. `CAM_LAYER_OFFSET = {"vit": -1, "swin": 0}` relative to the block count encodes both. A map with 0 variance on the fixture is the test that catches the site moving. The class token is dropped on ViT and the feature axis plays the paper's feature-map axis, so $Z$ is 196 patches on ViT-B/16 and 49 on Swin-S at 224.
2. **No class proposal and no mask subtraction.** Algorithm 1 segments the input with selective search, classifies each segment and keeps the 2 most confident classes other than the prediction. Algorithm 2 subtracts their Grad-CAM masks from the prediction's mask. Every reimplementation omits both and the port does too, since selective search is 1.9 of the paper's 2.5 seconds per input and no released version of the subtraction exists to match. What this changes is the mask on a benign input with 2 salient objects, which the paper's Figure 3 shows tightening to the suspicious one and which here stays as the raw map thresholds it.
3. **Mask by the scaled map at or above 0.85 rather than the paper's 15% of the maximum or the toolbox's 15% by area.** The paper binarizes "with a threshold of 15% of max intensity" and then subtracts the proposal masks. Without that subtraction a 15% cut covers most of a natural image, which is why the 2 reimplementations that drop Algorithms 1 and 2 tighten the cut to `mask_cond` 0.85 on the map scaled to $[0, 1]$. The port takes that rule as `MASK_THRESHOLD`. The map is min-max scaled per image, bilinearly upsampled to the native image size, scaled once more so the peak is exactly 1 as `pytorch_grad_cam` does after its resize and thresholded. The peak pixel is set explicitly so a constant map yields a 1-pixel region rather than an empty transplant. The toolbox's fixed 15% by area is recorded as `OFFICIAL_TOOLBOX_MASK_FRACTION` and not used, because the 2-dimensional rule needs the mask SIZE to vary: a benign input with a large salient region fools the overlays but noise in that region destroys confidence, a small trigger region gives both and a fixed area removes the axis that separates them and collapses the rule to `fooled` alone.
4. **100 fixed overlays from the shared split and 100 fixed inert noise images.** The paper ships 100 test images with the model and 1 inert pattern per mask. Beatrix redraws 10 overlays per input, BackdoorBench uses its whole clean set and the toolbox its `N` = 100. The port takes the first 100 images of the shared clean validation split through `strip.collect_overlay_batch`, so SentiNet sees the same data budget as every other method here. It draws 100 uniform noise images in $[0, 1]$ once at fit time from the run's seed, 1 per overlay. Both sets are fixed for every scored input, which removes overlay and noise choice as a per-sample source of variance. Compositing follows Algorithm 3 and the toolbox for both composites. Beatrix and BackdoorBench paste the input's region onto the noise instead, which measures how the region alone classifies rather than how the region's absence affects a clean image. The port does not reproduce that.
5. **`fooled` against the prediction.** Algorithm 3 counts $y_R = y$ where $y$ is "the class of $x$". At inference time the only class of $x$ a defender has is the model's prediction. Beatrix and BackdoorBench count against the prediction and the port does the same. The toolbox counts against the loader's true label and, on poisoned inputs, against the attack's target, which is an oracle and which the port does not reproduce. On an input the model misclassifies the 2 readings differ. The prediction is the one the transplanted region actually carries.
6. **Signed vertical residual as the score.** Algorithm 4 sets a scalar $d$ from the mean COBYLA perpendicular distance of the clean points outside the curve and flags a point whose distance exceeds it. BackdoorBench thresholds `avgConf` at 0.9 and reads neither the curve nor `fooled`. The port scores every input by $fooled - f_{curve}(avgConf)$, the signed vertical residual, positive above the envelope. It hands the negation of it to `defences.decision.detection_report`, which sets the threshold at a quantile of the clean validation scores. The vertical residual keeps the sign and the ordering the curve induces without an optimiser per sample. It differs from the perpendicular distance by a factor that depends on the curve's slope at the point, so the ranking among inputs at different `avgConf` can differ from the paper's and the threshold is a quantile rather than the paper's $d$. The toolbox's `defense_fpr` override is the same quantile idea.
7. **Envelope fitted on the shared 2000-sample split, with its own residuals in-sample.** The paper draws its boundary from about 400 benign points and the toolbox from 400 held-out validation images. The port fits `fit_decision_boundary` on the 2000 clean validation points, bins of `BOUNDARY_BIN_WIDTH` 0.04 over `avgConf` with the `BOUNDARY_POINTS_PER_BIN` 2 largest `fooled` per bin as Beatrix does, then a least-squares quadratic through them with `np.polyfit`, the degree falling to 1 with 2 populated bins and 0 with 1. The validation split's residuals are then in-sample by construction of an upper envelope: the 2 points that define each bin's ceiling sit on or near the curve, so the clean residual distribution is pulled toward 0 and the quantile threshold set on it is tighter than a fresh clean split would give. That moves the threshold and the achieved false-positive rate and leaves the AUROC on the paired clean and backdoor splits unchanged, since both are scored against the same fixed curve. `CROSS_FITTED` does not list `sentinet` for that reason. A 2-fold refit inside the split is the remedy if the achieved rate on the paired clean split overshoots the budget.
8. **bf16 autocast for every model query.** The reimplementations run float32. The port runs the CAM forward and backward and the 200 composite forwards under the shared autocast policy. The captured tensor is float32 under bf16 autocast, because the position-embedding add and every residual add promote the bf16 branch output, so the tokens and their gradient are differentiated in float32 and autocast rounds the branches rather than the map. The map is then scaled and thresholded, a rounding of a map rather than an optimisation trajectory, so `PRECISION_POLICY["sentinet"]` is `autocast`.
9. **Compositing at the native resolution.** The paper composites at the network's input resolution and the toolbox at 224. The port composites in $[0, 1]$ pixel space at the dataset's own 32 or 64 pixels after `normalization_buffers` undoes the loader's normalisation, then renormalises before the model's own `Resize` upsamples to 224. The trigger stays at the resolution it was stamped at, the mask is the CAM upsampled to that size and the composite the model sees is the one the attack would produce.

## Hyperparameters

| Symbol | Paper default | This port | Constant name |
|---|---|---|---|
| $\lvert X \rvert$, overlay images | 100 | 100, the first images of the shared split | `DEFAULT_NUM_OVERLAYS`, overridable through `DetectorContext.sentinet_overlays` |
| inert images | 1 pattern per mask, random noise | 100 uniform noise images, 1 per overlay | `DEFAULT_NUM_OVERLAYS` |
| mask threshold | 15% of the maximum, then subtraction | 0.85 on the map scaled to $[0, 1]$ | `MASK_THRESHOLD` |
| mask area, toolbox | none | 0.15, recorded and unused | `OFFICIAL_TOOLBOX_MASK_FRACTION` |
| CAM site | final pooling layer | last block input on ViT, output on Swin | `CAM_LAYER_OFFSET` |
| $avgConf$ interval | unstated | 0.04 | `BOUNDARY_BIN_WIDTH` |
| points per interval | "the highest y-values", count unstated | 2 | `BOUNDARY_POINTS_PER_BIN` |
| curve | non-linear least squares, a parabola in Figure 4 | quadratic, degree falling with fewer than 3 bins | none |
| $d$, acceptable distance | mean COBYLA distance of clean outliers | none, the quantile rule replaces it | none |
| composites per forward | all $\lvert X \rvert$ | 256 | `OVERLAY_CHUNK` |
| map range floor | none | 1e-7 | `CAM_RANGE_FLOOR` |

## Cost

Each input costs 1 forward and 1 backward for the map, then $2 \lvert X \rvert$ forwards for the 2 composite sets. Counting a backward as 1.5 forwards, the shared convention in the registry's cost table, that is $2.5 + 200 = 202.5$ forward-equivalents per input at the default 100 overlays, against 8 for STRIP, 6 for SCALE-UP and IBD-PSC, 71 for TeCo and 251 for CD-L. With the model frozen the backward computes no weight gradients and lands nearer 1 forward, so 2.5 is a ceiling. `FORWARD_PASSES_PER_INPUT["sentinet"]` records $2 + 2 \lvert X \rvert = 202$, the map's forward and backward counted as 2 model queries.

At the measured 2130 images per second for a bfloat16 ViT-B/16 forward on the A100, the plan's estimate is about 37 minutes per GTSRB checkpoint (23208 scored inputs), which with CD-L makes about 80% of the whole detector panel's bill. Fitting adds the same cost over the 2000 validation images, about 3 minutes. The smoke run replaces the estimate with a measured seconds-per-input figure that goes into `pbs/generate_detector_jobs.py`.

## How to run

The smoke runs 1 checkpoint folder at 200 inputs per split and writes outside the results tree. `--allow-missing-psbd-cache` is required there because the smoke tree holds no PSBD split manifest to check against.

```bash
python -m cli.baselines --checkpoint-folder <folder> --detectors sentinet --max-samples 200 \
    --results-dir scratch/detector_smoke/results --allow-missing-psbd-cache
```

The panel runs through the `sentinet` job group of `pbs/generate_detector_jobs.py`, which emits 1 job per checkpoint with `--skip-existing` and a walltime of at least twice the smoke estimate. SentiNet gets its own group because its runtime dwarfs the cheap detectors and a shared job would be sized for the wrong detector.

## Where results land

`results/<folder>/detectors/sentinet_metrics.json` holds the detection report at every quantile plus the provenance record. The raw per-sample scores sit beside it as `sentinet_scores_validation.pt`, `sentinet_scores_clean.pt` and `sentinet_scores_backdoor.pt`, 1 float32 tensor of the split's length each, in the loader's order, so any later threshold or fusion reads them without a rerun. The validation tensor holds the negated in-sample residuals of deviation 7.

## Results

<!-- results:begin -->
<!-- results:end -->

## Known failure modes

A high residual says the input's salient region hijacks clean images while its absence leaves them intact, which is what a patch trigger produces and what the paper was built to catch. A low residual therefore says only that the region the map found does not travel. The map is where the method fails first. A trigger that is spread over the whole image has no compact region for Grad-CAM to isolate. The region it does isolate is the object, which does not travel. Blend, WaNet, SIG and LF cover the whole image, so they are expected failures by design, as the paper's own Section VI concedes for large objects and STRIP and Februus both note. An AUROC above about 0.7 on any of them is a reason to look for a bug before reporting it. The plan's acceptance bar for the smoke is 0.70 on BadNet and an expected 0.35 to 0.70 on Blend.

A flat map is the benign failure. When the class evidence of a clean image is spread over the object, the scaled map can sit above 0.85 over most of the image, the transplant then carries the object and `fooled` rises for a benign reason while the noise fill occludes it and `avg_conf` falls. Whether the envelope absorbs that depends on how many clean inputs behave that way in each `avgConf` bin. A map that is constant scales to all 0 and transplants a single pixel, which the tests pin as the degenerate case rather than an error. The paper's Section V-B also names an adaptive adversary who trains the inert pattern itself as a member of the target class, so that noise in the region keeps the target label and `avg_conf` stays high.

The cost is the deployment failure. At 202 forward-equivalents per input SentiNet is 25 times STRIP and the paper's own latency is 2.5 seconds per input, 1.9 of them selective search, which the port omits. The synthetic fixture in `experiments/preflight/synthetic.py` cannot judge the method at all: its backdoor is added at the logits by boolean indexing on the input, so no token the map reads carries the trigger and the transplanted region is the random ViT's own saliency. The test file prints the fixture's AUROC and asserts nothing about it. `NOT_JUDGEABLE` names `sentinet` with that reason.

## Direction

High is poisoned in the paper's plane, since a triggered input sits above the envelope at the top right of Figure 4. `boundary_residual` keeps that sign. `sentinet_scores` negates once at its boundary so that low means poisoned, PSU's convention. The builder negates the cached validation residuals the same way. The toolbox's signed distance is also positive above the curve, so its `all_d > d_thr` rule agrees with the residual's direction before the negation. A second negation anywhere would produce a well-formed, exactly inverted detector, which is the failure the `direction` field of the report exists to catch.
