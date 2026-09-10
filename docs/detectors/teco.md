# TeCo, test-time corruption robustness consistency

TeCo corrupts each input at every severity of a fixed suite of image corruptions and records, for every corruption type, the lowest severity at which the prediction first stops agreeing with the uncorrupted prediction. The spread of those breaking points across corruption types is the score. An ordinary input's class evidence erodes at roughly the same rate under any of the corruptions, so its breaking points cluster and the spread is small, while a trigger survives some corruption types and not others, blur can destroy a patch while brightness leaves it intact, so a triggered input's breaking points scatter and the spread is large. The method is black-box, needs only predicted labels, and costs $K \times N + 1$ forward passes per input, the most expensive detector in the registry apart from CD-L. This page records what the paper defines, what the released code does, what the port under `detectors/teco.py` runs on ViT and Swin and where they diverge.

## Citation

Liu et al., "Detecting Backdoors During the Inference Stage Based on Corruption Robustness Consistency", CVPR 2023, arXiv:2303.18191. The score is Section 4.2, Algorithm 1, which carries no numbered equation of its own, so lines of the algorithm are cited by number. The decision rule is Equation (4). The paper's own adaptive evaluation, a corruption-matching attack, is Equation 8.

No repository specific to TeCo's own authors is vendored in this project. `third_party/BackdoorBench/detection_infer/teco.py`, read at commit `f02e353`, is the reference implementation this port was checked against. It imports the `imagecorruptions` package at line 64, which is not installed in this project and cannot be added without a new hard dependency, since it in turn requires `opencv-python`, also absent. Its threshold search names its dispersion statistic `mad` at line 340, `mad = np.std(indexs)`, and its corruption loop at lines 235 to 256 assigns `x = images_poison` and then mutates `x[i]` in place at every severity and every corruption type without ever resetting it to the pristine image.

## Threat model and data requirement

The adversary poisons the training data and the defender controls inference, with no knowledge of the trigger, the target class or the poisoning rate. The defender needs only the model's predicted label on an input it supplies, so the method is black-box in the same narrow sense as SCALE-UP.

The score itself needs no clean data at all, since it compares an input's own corrupted predictions against its own uncorrupted prediction. Only the detection threshold needs clean data, and it comes from the same shared clean validation split every other detector in the registry uses, so `DATA_REQUIREMENT["teco"]` records `none`.

## Mechanism

    original form, Algorithm 1
        P_org <- C_theta(x)
        for k = 1..K:
            l <- N + 1
            for n = 1..N:
                if C_theta( D_k^n(x) ) != P_org:
                    l <- n
                    break
            L <- L union {l}
        TeCo(x) = Dev(L)

        Gamma( TeCo(x) ) = 1 if TeCo(x) > gamma else 0              Eq. (4)

| Symbol | Meaning |
|---|---|
| $C_{\theta}$ | the classifier's predicted label |
| $x$ | the input image |
| $P_{org}$ | the prediction on the uncorrupted image |
| $K$ | the number of corruption types |
| $N$ | the number of severities per corruption type |
| $D_k^n$ | corruption type $k$ applied to $x$ at severity $n$ |
| $l$ | the lowest severity of corruption $k$ that moves the prediction away from $P_{org}$, or $N+1$ if it never does |
| $L$ | the set of breaking points, 1 per corruption type |
| $Dev$ | the dispersion statistic over $L$ |
| $\gamma$ | the decision threshold |

The descriptive form renames without rederiving.

$$
\begin{aligned}
\text{reference\_label} &= \text{the label predicted on the uncorrupted image} \\
\text{hardness}[k] &= \text{the lowest severity of corruption } k \text{ at which the prediction stops matching reference\_label,} \\
&\quad \text{or max\_severity} + 1 \text{ when it never stops matching} \\
\text{teco\_score}(\text{image}) &= \text{population standard deviation of hardness over the } K \text{ corruption types}
\end{aligned}
$$

Every corruption in the suite, noise, blur, weather, compression, erodes ordinary class evidence at a broadly similar rate as its severity rises, so a clean input's prediction tends to break at a similar severity under any of them and the spread of breaking points is small. A trigger's survival under corruption is specific to what the trigger is, a low-frequency blend survives blur and breaks under high-frequency noise, a compact patch survives brightness and contrast shifts but breaks under blur or pixelation, so a triggered input's breaking points scatter across the severity range and the spread grows. $Dev$ measures that scatter directly, with no clean reference entering the per-input statistic at all.

## What the released code does

The reference computes the prediction on the uncorrupted poisoned image once, then loops over the corruption suite and, for each type, over severities 1 to 5. At each step it mutates the same underlying image list, `x = images_poison` followed by `x[i] = self.dg(x[i], args)`, and never restores it, so severity 2 of a corruption type is applied to the output of severity 1 rather than to the pristine image, and the first severity of the second corruption type is applied to an image that has already been through all 5 severities of the first. By the last of the 15 corruption types in the loop, each image has accumulated 70 sequential operations rather than 1. The threshold search then computes, for each image, the lowest severity at which each corruption type's prediction first disagrees with the original, collects those breaking points across the 15 types into `indexs`, and takes `mad = np.std(indexs)`, the population standard deviation despite the variable's name. It fits a decision threshold with `sklearn.metrics.roc_curve` against those values and flags an image when its statistic exceeds the chosen cut, matching Eq. (4)'s "greater than gamma" direction.

## What this port does on ViT

1. **14 corruptions, not 15.** The paper uses the `imagecorruptions` package's 15 standard corruptions. `IMAGENET_C_CORRUPTIONS` in `detectors/teco.py` names all 15, and `UNAVAILABLE_CORRUPTIONS` names `frost` as the 1 the port cannot reproduce, since it composites 1 of 6 bundled photographs of frosted glass that ship as binary assets inside the package and there is no formula to regenerate them from. The other 14, `gaussian_noise`, `shot_noise`, `impulse_noise`, `defocus_blur`, `glass_blur`, `motion_blur`, `zoom_blur`, `snow`, `fog`, `brightness`, `contrast`, `elastic_transform`, `pixelate`, `jpeg_compression`, are each reimplemented in torch directly against the reference formulas. `Dev` is a standard deviation over the corruption types, so dropping 1 of 15 changes the sample it is computed over, and a TeCo number from this repository is not numerically identical to a published one.
2. **Corruption applied to the pristine image.** Algorithm 1 line 5 applies $D_k^n$ to $x$, the original input, at every severity. The released code instead mutates the image in place and never restores it, so severity 2 lands on the output of severity 1 and the second corruption type lands on an image that has already been through all 5 severities of the first. `hardness_thresholds` in `detectors/teco.py` denormalizes the batch once, then applies every corruption to that same pristine `pixels` tensor, `corrupted = corrupt(pixels, severity)`, which is a faithful reading of the algorithm rather than the released code's cumulative composition. A faithful reproduction is therefore expected to differ from the published 0.943 AUROC, since that number was produced by a different statistic than the one Algorithm 1 defines.
3. **Deviation measure.** The released code names its dispersion variable `mad`, which reads as mean absolute deviation, but the line computing it is `np.std(indexs)`, the population standard deviation with `ddof=0`. `deviation` in `detectors/teco.py` computes `thresholds.std(dim=1, unbiased=False)`, matching what the code actually runs rather than what its variable name suggests. The paper's own ablation finds mean absolute deviation statistically indistinguishable from this choice and the coefficient of variation clearly worse, so the unnormalized standard deviation is a deliberate choice rather than an accident worth correcting. It rescales every score by a constant and so cannot move AUROC, but it does move any absolute threshold, including the paper's own empirical $\gamma = 1$.
4. **Randomness per batch.** The motion blur angle and the snow angle are each drawn once per batch in `motion_blur` and `snow` rather than once per image, `torch.empty(1).uniform_(...)` outside the loop over images. Sharing them within a batch removes a per-image nuisance term from a statistic that compares corruption types within 1 image, and it is what makes the corruption batchable at all rather than requiring a Python loop over individual images for every severity. Per-pixel noise, `gaussian_noise`, `shot_noise` and `impulse_noise`, is still drawn independently per image, as in the reference.
5. **Operator substitutions**, each verified numerically against a numpy or scipy reference where the reference's own dependency was missing. `_disk_kernel`'s antialiasing Gaussian replaces `cv2.GaussianBlur` and agrees to $5 \times 10^{-4}$. `pixelate`'s downsample uses torch area resampling in place of PIL's BOX filter, which agree exactly at integer downscale factors and approximately otherwise. `_clipped_zoom` matches `scipy.ndimage.zoom` with `grid_mode=False`, `align_corners=True` in torch, to $2 \times 10^{-6}$. `elastic_transform` samples with torch's reflection padding, which reflects about the edge pixel where scipy's `mode="reflect"` repeats it, a difference confined to border pixels.

Corruption is applied in $[0, 1]$ pixel space at the dataset's native resolution, before the model wrapper upscales to 224, which is where this project applies triggers too and what the reference does for CIFAR-scale inputs. `_quantize` rounds the result to the 8-bit grid after every corruption, since the reference operates on uint8 arrays throughout and that quantization is part of several of the operators' own definitions rather than an approximation the port introduces.

## Hyperparameters

| Symbol | Paper default | This port | Constant name |
|---|---|---|---|
| $K$, corruption count | 15 | 14, `frost` dropped | `len(DEFAULT_CORRUPTIONS)` |
| $N$, severities | 5 | 5 | `MAX_SEVERITY` |
| never-flipped sentinel | $N + 1$ | 6 | `NEVER_FLIPPED` |
| $Dev$ | unspecified in the text, population standard deviation in the code | population standard deviation | `deviation` |
| $\gamma$, decision threshold | swept, empirical value near 1 | none, the registry's quantile rule replaces it | none |
| motion blur / snow angle | drawn per image | drawn once per batch | see deviation 4 |
| reduced corruption set for sign checks | not applicable | 4 corruptions, `gaussian_noise`, `defocus_blur`, `brightness`, `contrast` | `experiments.preflight.gate.CHEAP_CORRUPTIONS` |

## Cost

$K \times N + 1$ forward passes per input, 71 at the port's 14 corruptions and 5 severities, against 251 for CD-L and 1 for beatrix or confidence. Algorithm 1's inner break would in principle allow an early exit once a corruption type's prediction first flips, but the released code evaluates every severity of every corruption type regardless and applies the break only when reading back the cached predictions, which gives an identical statistic at the full cost, and that full cost is what a batched implementation on a GPU pays.

Beyond the forward-pass count, several corruptions are CPU-bound rather than GPU-bound, and the registry's forward-count cost understates their wall-clock price. `glass_blur` runs a nested Python loop over every spatial position at every iteration, a per-pixel local shuffle that cannot be vectorized across the image, and `jpeg_compression` round-trips each image through PIL's encoder 1 image at a time on the CPU, since JPEG has no closed-form torch implementation. `pbs/generate_detector_jobs.py` prices this directly, `GROUP_SLOWDOWN["teco"] = 2.0`, doubling the group's estimated wall-clock time relative to its raw forward-pass-equivalent count, where every other group runs at its forward count face value.

## How to run

The smoke runs 1 checkpoint folder at 500 inputs per split and writes outside the results tree. `--allow-missing-psbd-cache` is required there because the smoke tree holds no PSBD split manifest to check against. A cheaper sign check exists separately, through `DetectorContext.teco_corruptions`, which `experiments/preflight/gate.py` sets to 4 corruptions rather than 14 so that verifying the detector's sign does not require paying the full cost.

```bash
python -m cli.baselines --checkpoint-folder <folder> --detectors teco --max-samples 500 \
    --results-dir scratch/detector_smoke/results --allow-missing-psbd-cache
```

The panel runs through its own `teco` job group of `pbs/generate_detector_jobs.py` rather than the `cheap` group, since TeCo's cost dwarfs confidence, STRIP, SCALE-UP, IBD-PSC and Beatrix and a shared job would be sized for the wrong detector.

## Where results land

`results/<folder>/detectors/teco_metrics.json` holds the detection report at every quantile plus the provenance record, whose hyperparameters carry the corruption list actually used, which matters whenever a run was scored under the reduced set rather than the full 14. The raw per-sample scores sit beside it as `teco_scores_validation.pt`, `teco_scores_clean.pt` and `teco_scores_backdoor.pt`, 1 float32 tensor of the split's length each, in the loader's order, so any later threshold or fusion reads them without a rerun.

## Results

<!-- results:begin -->
<!-- results:end -->

## Known failure modes

A low spread says the input's prediction broke at roughly the same severity under every corruption type, which is the clean signature, and it is also what a uniformly thin decision margin produces, since a margin close to the boundary breaks early under any corruption regardless of what that corruption actually is. `python -m experiments.preflight.check_signs` reads TeCo at AUROC 0.9017 on the synthetic fixture, using the reduced 4-corruption set `CHEAP_CORRUPTIONS`, below confidence's or IBD-PSC's 1.0000 on the same unmissable backdoor, consistent with TeCo measuring a dispersion pattern rather than a single clean signal.

The paper's own adaptive section is the strongest evidence of a real weakness, and it is the costliest attack in this project's whole cross-defence comparison. A corruption-matching training term pushes TeCo's AUROC from 0.911 down to 0.576 (Liu et al., Eq. 8), but the reported cost is 40 points of clean accuracy, 0.9153 to 0.5105, and 21 points of attack success rate, 0.9502 to 0.7386. The project's own reading of that trade in `docs/attack-design/cross-defence.md` is that this is a broken model rather than a deployable threat, the only detector in its adaptive-attack comparison whose evasion is a deployment channel the paper's authors already targeted on purpose, since the corruption suite is exactly the transform family a real deployment channel also applies.

A cheaper and more concerning weakness is the all-to-all label mapping, which costs the attacker nothing extra to train. The project's own review reports TeCo's AUROC falling to 0.7749 on an all-to-all attack, taken from the paper's own Table 20, a family-wide limitation it shares with several training-set detectors (`docs/attack-design/cross-defence.md`, Section 5). Since the attack pays no clean-accuracy or ASR cost to achieve this, an all-to-all checkpoint is the harder case for TeCo to clear, not the corruption-matching one.

A 3rd concern is inferred rather than measured in this repository, so it is flagged as a prediction rather than a result. TeCo's mechanism assumes ordinary class evidence erodes at a broadly similar rate under every corruption type, an assumption calibrated against the ResNet family the paper evaluates. Vision transformers are documented elsewhere in the corruption-robustness literature to have markedly uneven robustness profiles across the Hendrycks and Dietterich corruption suite compared to convolutional networks, more robust to some corruption types and less to others, and whether ViT-B/16's or Swin-S's own clean breaking points are as tightly clustered as a ResNet's is not something this project has measured directly. If they are not, the clean population's own spread widens and the separation TeCo relies on narrows for a reason that has nothing to do with a backdoor.

## Direction

Low is poisoned. Eq. (4) flags an input when $TeCo(x) > \gamma$, so the raw statistic is high for poisoned, the opposite of the registry's convention. `teco_scores` negates the raw `deviation` once at the return boundary. A second negation anywhere would produce a well-formed, exactly inverted detector, which is the failure the `auroc_two_sided` diagnostic field in `defences.decision.detection_report` exists to surface.
