# STRIP, STRong Intentional Perturbation

STRIP superimposes each input on a fixed set of clean images and reads the entropy of the model's prediction on the blend. A clean input's class evidence scatters once a second image is laid over it, so its entropy is high, while a trigger keeps dragging the prediction to the target class through the blend, so its entropy stays low. The method is black-box, needs only the model's softmax output and costs 1 forward pass per superimposed copy. This page records what the paper defines, what the released reference does, what the port under `detectors/strip.py` runs on ViT and Swin and where the 2 diverge.

## Citation

Gao et al., "STRIP: A Defence Against Trojan Attacks on Deep Neural Networks", ACSAC 2019, arXiv:1902.06531. The statistic is Section IV-D, Equations (2) to (4). The paper's own adaptive evaluation is Section VI-F.

The released reference is read via the copy bundled in the Beatrix repository, `third_party/Beatrix/defenses/STRIP/STRIP.py`, at commit `685827e`, where `superimpose` at line 54 calls `cv2.addWeighted(background, 1, overlay, 1, 0)`. `third_party/BackdoorBench/detection_infer/strip.py` and `third_party/backdoor-toolbox/cleansers_tool_box/strip.py` were read for comparison and agree on the superimposition rule and the fixed overlay set.

## Threat model and data requirement

The adversary poisons the training data and the defender controls inference, with no knowledge of the trigger, the target class or the poisoning rate. The defender needs only the model's output probabilities on an input it supplies, so the method is black-box. It needs $N$ clean images to superimpose, and the paper draws them from a held-out benign set it assumes the defender already holds for exactly this purpose.

The port draws the $N$ overlays from the shared 2000-sample clean validation split rather than from a separate pool, so STRIP is given the same data budget every other detector in the registry shares and no method gets an advantage from seeing more. `DATA_REQUIREMENT` records `f"{N} clean images, unlabelled"`, since the overlay set carries no label information into the statistic.

## Mechanism

$$
\begin{aligned}
H_n &= - \sum_{i=1}^{M} y_i \log_2 y_i && \text{(2)} \\
H_{sum} &= \sum_{n=1}^{N} H_n && \text{(3)} \\
H &= \frac{1}{N} H_{sum} && \text{(4)}
\end{aligned}
$$

| Symbol | Meaning |
|---|---|
| $y_i$ | softmax probability of class $i$ on a superimposed copy |
| $M$ | the number of classes |
| $H_n$ | entropy of the prediction on the $n$-th superimposed copy |
| $N$ | the number of clean images superimposed on each input |
| $H$ | the mean entropy over the $N$ copies, the STRIP score |

The descriptive form renames without rederiving.

$$
\begin{aligned}
\text{blend\_entropy}_n &= -\sum_c \text{prob}_c \log \text{prob}_c, \ \text{on the } n\text{-th superimposed copy} \\
\text{strip\_score} &= \frac{1}{N} \sum_{n=1}^{N} \text{blend\_entropy}_n
\end{aligned}
$$

A clean input's class evidence lives in the arrangement of its own pixels, and superimposing a second, unrelated image scrambles that arrangement, so the prediction on the blend scatters across classes and its entropy is high. A trigger is built to survive exactly this kind of interference, since a real deployment applies it to inputs the attacker does not control, so a triggered input keeps pulling the prediction toward the target class through the blend and its entropy stays low. The paper flags an input whose $H$ falls below a percentile of the clean entropy distribution, the same quantile-of-clean-validation rule `defences.decision.detection_report` applies here.

## What the released code does

The reference builds the overlay set once, superimposes it onto every scored input at test time and reads the model's softmax on each blend. Superimposition happens in pixel space on uint8 arrays through `cv2.addWeighted(background, 1, overlay, 1, 0)`, which sums 2 images with both weights at 1 and lets OpenCV saturate the result at 255 rather than wrap or renormalize it. That saturating cast is where the method's whole nonlinearity lives, since a bright region overlapping a bright region clips rather than doubles. Entropy is computed in bits, Eq. (2)'s $\log_2$, and averaged over the overlay set as Eq. (4) states. The overlay images are drawn once from the defender's clean pool and reused across every scored input, so the comparison never varies which second image a given input was blended against.

## What this port does on ViT

1. **Entropy in nats, not bits.** Eq. (2) uses $\log_2$ and `blend_entropy` in `detectors/strip.py` uses the natural log. The 2 differ by the constant factor $\log 2$, so no ranking, AUROC or quantile position changes. Only the printed threshold value is scaled, and a reader comparing against a published entropy number should divide the port's value by $\log 2$ first.
2. **Superimposition in pixel space, then saturated.** The port denormalizes to $[0, 1]$, sums the pixels, clamps to $[0, 1]$ and renormalizes, which is `cv2.addWeighted`'s saturating sum read in float space instead of uint8. An earlier version of this port summed the 2 already-normalized tensors directly. That is a different operation, since adding in normalized space gives $(p_1 - m)/s + (p_2 - m)/s = (p_1 + p_2 - 2m)/s$, the pixel-space sum displaced by a further $-m/s$ per channel, 2.43 units on CIFAR-10 channel 0, and it also skipped the saturation entirely. The loaders here deliver normalized tensors, so the round trip the port takes is denormalize, add, saturate, renormalize, matching what the reference does on raw pixels.
3. **A fixed overlay set.** `collect_overlay_batch` draws the first `DEFAULT_NUM_OVERLAYS` images of the clean validation split, in the split's own order, and every scored input is superimposed against that same set. This matches the reference's practice of reusing 1 overlay pool across every scored input rather than resampling it per input, which removes overlay choice as a source of per-sample variance in the comparison.
4. **N = 8.** The paper defaults to $N = 100$ and later reports 10 as sufficient. `DEFAULT_NUM_OVERLAYS` is 8, the value every recorded STRIP number in this repository was produced with, so ported numbers stay comparable to everything already recorded rather than moving to a paper default that was never actually run here.

## Hyperparameters

| Symbol | Paper default | This port | Constant name |
|---|---|---|---|
| $N$, overlay count | 100, later reduced to 10 | 8 | `DEFAULT_NUM_OVERLAYS` |
| entropy base | 2, bits | $e$, nats | none, a constant rescaling, see deviation 1 |
| probability floor | not stated | $10^{-12}$, guards $\log 0$ | `PROBABILITY_FLOOR` |
| detection percentile | a percentile of clean entropy | the registry's quantile rule | none |

## Cost

$N$ forward passes per input, 8 at the port's default, against 1 for confidence, 6 for SCALE-UP and IBD-PSC, 71 for TeCo and 251 for CD-L. There is no fitting step beyond drawing the overlay batch once, a single pass over the first 8 images of the validation loader, so STRIP is absent from `NEEDS_FITTING`.

## How to run

The smoke runs 1 checkpoint folder at 500 inputs per split and writes outside the results tree. `--allow-missing-psbd-cache` is required there because the smoke tree holds no PSBD split manifest to check against.

```bash
python -m cli.baselines --checkpoint-folder <folder> --detectors strip --max-samples 500 \
    --results-dir scratch/detector_smoke/results --allow-missing-psbd-cache
```

The panel runs through the `cheap` job group of `pbs/generate_detector_jobs.py`, which puts STRIP beside confidence, SCALE-UP, IBD-PSC and Beatrix in 1 job per checkpoint with `--skip-existing`, since all of them finish within minutes and a job sized for CD-L or TeCo would idle the GPU on them.

## Where results land

`results/<folder>/detectors/strip_metrics.json` holds the detection report at every quantile plus the provenance record, whose hyperparameters carry the overlay count. The raw per-sample scores sit beside it as `strip_scores_validation.pt`, `strip_scores_clean.pt` and `strip_scores_backdoor.pt`, 1 float32 tensor of the split's length each, in the loader's order, so any later threshold or fusion reads them without a rerun.

## Results

<!-- results:begin -->
<!-- results:end -->

## Known failure modes

A high score says the prediction scattered across classes once a second image was laid over the input, which is the clean signature, so a low score says the prediction kept pointing at 1 class through the blend. That happens for a trigger that is high contrast and spatially compact enough to dominate whatever it is summed with, and it is also what a benign input with a very strong, saturating visual feature produces. `python -m experiments.preflight.check_signs` reads STRIP at AUROC 0.6560 on the synthetic fixture, well above the gate's 0.60 floor but far short of confidence's 1.0000 or IBD-PSC's 1.0000 on the same unmissable-by-construction backdoor, which is consistent with STRIP measuring something more indirect than the other 2.

The paper's own adaptive section is the strongest evidence of a real weakness. Section VI-F reports an entropy-manipulation term that flattens STRIP's separation at a clean-accuracy cost of about 3 points and an ASR cost of essentially 0, which the project's own cross-defence review (`docs/attack-design/cross-defence.md`) calls "essentially free" evasion, the fragile end of the 5-detector comparison it works through. A second, published attack aimed at the same mechanism, Peng et al.'s under-confidence backdoor (LSBA, arXiv:2202.11203), reports its authors' own measurement of STRIP's false acceptance rate at a 1% false rejection rate jumping from near 0 without the rule to 98 to 100% with it, across BadNet, SIG and WaNet on MNIST, GTSRB and CelebA, at a clean-accuracy cost near 0. Both attacks work by keeping the triggered prediction's margin thin rather than by touching the trigger's visual footprint, so neither depends on the specific dataset or architecture this repository trains against, and a low STRIP AUROC on a checkpoint trained against margin suppression is the expected outcome rather than a bug.

STRIP's own threshold is also coupled to the model's general calibration in a way the quantile rule does not remove. `docs/attack-design/A1-operating-point-and-threshold.md` notes that STRIP's rule is itself a percentile of clean entropy, so sharpening a model's confidence lowers clean entropy on the slice near the threshold and moves the same cut point the poisoned scores are compared against, which is a softer version of the same coupling the LSBA attack exploits directly.

## Direction

Low is poisoned and the score is returned unnegated. STRIP's claim is that a triggered input keeps low entropy under superimposition, and low already means poisoned in the shared convention, so `strip_scores` returns the raw entropy. Negating it once produces AUROC 0.000, perfect separation with the sign reversed, which is exactly the failure the `auroc_two_sided` diagnostic field in `defences.decision.detection_report` exists to surface.
