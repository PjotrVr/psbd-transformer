# CD-L, Cognitive Distillation on logits

Cognitive Distillation learns the smallest input mask, 1 input at a time, under which the model still produces the same logits and reads the mask's L1 norm as a backdoor score. The method is white-box and gradient-based, which makes it 30 to 40 times as expensive as the cheap detectors in the registry. It is also the strongest prior training-set detector that PSBD and MSPC compare against. This page records what the paper defines, what the authors released, what the port under `detectors/cd_l.py` runs on ViT and Swin and where the 2 diverge.

## Citation

Huang et al., "Distilling Cognitive Backdoor Patterns within an Image", ICLR 2023, arXiv:2301.10908 (v4 read). The objective is Eq. (1) and the distilled input Eq. (2), both in Section 3.1. The detection rule is Eq. (4) in Section 3.2, the optimiser settings are in Appendix B.3 and the ablations over the 2 regularisation weights are in Appendices B.5 and B.6.

The released code is https://github.com/HanxunH/CognitiveDistillation, read at commit `1d35393` as vendored under `third_party/CognitiveDistillation`. The method is the 63-line class in `detection/cognitive_distillation.py`, the driver is `extract.py` and the score analysis is `analysis/cognitive_distillation.py`. The backdoor-toolbox copy at `third_party/backdoor-toolbox/other_defenses_tool_box/CD.py` was read for comparison and is the source of 2 of the deviations below.

## Threat model and data requirement

The adversary poisons the training data and the defender controls either training or inference, with no knowledge of the trigger, the target class or the poisoning rate. The defender needs white-box access, because the score is the result of 100 Adam steps on an input mask and each step differentiates the logits with respect to the input. The paper uses the method both as a training-set filter and as a test-time input detector. The port covers the test-time role only, which is the role every detector in the registry plays.

The score itself needs no data. The objective compares the model's logits on the distilled input against its logits on the original, so no label, no clean image and no second model enters. Clean data enters only through the threshold. The paper sets $t = \mu - \gamma \sigma$ from the mask norms of 1% of the clean training set with $\gamma = 1$, while the port gives CD-L the same 2000-sample clean validation split and quantile rule as every other detector, so `DATA_REQUIREMENT` records `none`.

## Mechanism

Eq. (1), (2) and (4) follow in the paper's own notation. The symbol table underneath defines every symbol, and the descriptive form after it renames without rederiving.

$$
\begin{aligned}
\arg\min_{\bm{m}} \; & \left\| f_{\theta}(\bm{x}) - f_{\theta}(\bm{x}_{cp}) \right\|_1 + \alpha \left\| \bm{m} \right\|_1 + \beta \, TV(\bm{m}) && \text{(1)} \\
\bm{x}_{cp} &= \bm{x} \odot \bm{m} + (1 - \bm{m}) \odot \delta && \text{(2)} \\
g(\bm{x}) &= \begin{cases} 1 & \text{if } \left\| \bm{m} \right\|_1 \le t \\ 0 & \text{if } \left\| \bm{m} \right\|_1 > t \end{cases} && \text{(4)}
\end{aligned}
$$

| Symbol | Meaning |
|---|---|
| $f_{\theta}$ | the model's logit map (CD-L) or its last convolutional features (CD-F) |
| $\bm{x}$ | the input image in $[0, 1]^{w \times h \times c}$ |
| $\bm{x}_{cp}$ | the distilled cognitive pattern, same shape as $\bm{x}$ |
| $\bm{m}$ | the learnable mask in $[0, 1]^{w \times h}$, shared by the colour channels |
| $\delta$ | a uniform random fill in $[0, 1]^{c}$, 1 value per channel, redrawn every step |
| $\odot$ | elementwise product, broadcast over channels |
| $\alpha$ | weight of the L1 sparsity term |
| $\beta$ | weight of the total variation term |
| $TV$ | total variation of the mask, undefined in the paper, squared differences in the code |
| $t$ | the detection threshold, $g = 1$ meaning backdoor |

The same 3 lines follow with descriptive names in place of the paper's symbols. The structure is unchanged and only the names differ.

$$
\begin{aligned}
\text{objective} &= \big\| \text{logits}(\text{image}) - \text{logits}(\text{distilled}) \big\|_1 + \text{l1\_weight} \cdot \textstyle\sum \text{mask} + \text{tv\_weight} \cdot TV(\text{mask}) \\
\text{distilled} &= \text{image} \cdot \text{mask} + (1 - \text{mask}) \cdot \text{random\_fill} \\
\text{is\_backdoor} &= \big[ \textstyle\sum \text{mask} \le \text{threshold} \big]
\end{aligned}
$$

The L1 term pushes the mask toward 0 at every pixel and only a pixel whose removal changes the logits can hold it up. On a clean input the logits rest on the object, spread over many pixels, so the mask stays large. On a triggered input the backdoor path dominates the logits and the trigger's own pixels are all the model needs to reproduce them, so the mask collapses onto the trigger and its L1 norm is small. The mask is parametrised as $\bm{m} = (\tanh(\theta) + 1) / 2$, the paper's "scaled tanh", so Adam works on an unconstrained $\theta$ and the mask never leaves $[0, 1]$.

## What the released code does

The class initialises the mask parameter to ones, so the effective mask starts at 0.8808 rather than 1. It builds Adam over that single tensor with learning rate 0.1 and betas $(0.1, 0.1)$ and computes the reference logits once through `preprocessor(images)` with the result detached. Each of the 100 steps recomputes the effective mask and draws `torch.rand(b, c, 1, 1)` as the fill. It forms `x_adv = images * mask + (1 - mask) * fill`, runs the model on `x_adv` with no preprocessor and takes the L1 distance between the 2 logit tensors averaged over classes. To that it adds `gamma` times the mask's L1 norm and `beta` times a total variation that sums squared vertical and horizontal differences and divides by the mask's element count, averages over the batch, calls `backward` and steps. With `norm_only=True` the class returns the L1 norm of the final mask per image. That is the quantity the port's `mask_norms` reproduces bit for bit.

The paper's text and the code disagree on 2 of these settings. Appendix B.3 says $\beta = 10$ where the code's default is `beta=1.0`. The code names the L1 weight `gamma`, a letter the paper reserves for the threshold coefficient. The `analysis/cognitive_distillation.py` scorer min-max normalises the norms and returns `1 - score`, so its AUROC input is high for backdoor, while its `predict` flags a norm more than $t$ standard deviations below the clean mean. Both readings agree that a small norm means backdoor.

2 further points in the released driver matter for a port. `extract.py` sets `requires_grad = False` on every model parameter and replaces the training transform by `ToTensor` alone, so the models in that repository consume $[0, 1]$ input directly and the identity preprocessor is what that repository actually runs. Its line 116 then calls `detector(model, images, labels)`, which binds the label tensor into the `preprocessor` slot. The `CD` branch as checked out therefore raises `TypeError: 'Tensor' object is not callable` on the first batch, which I confirmed on a 4-image CPU call. The backdoor-toolbox copy adds 2 more. Its loaders normalise with dataset statistics and the $[0, 1]$ fill is blended into those normalised tensors (line 171). Its `threshold_calculation` uses `mu - self.gamma * std` where `self.gamma` is the L1 weight 0.01 (line 195), so its threshold sits 0.01 standard deviations below the mean rather than the paper's 1.

## What this port does on ViT

1. **TV weight.** The paper's Appendix B.3 sets $\beta = 10$. The released code defaults to 1.0 and the paper's Figure 10 finds detection unchanged across $\beta \in \{1, 5, 10, 50, 100\}$. The port takes 1.0 as `DEFAULT_TV_WEIGHT` because the bit-level cross-check targets the released class and the ablation says the choice does not move the number. A reader comparing against the paper's tables should know the smoothing is 10 times weaker.
2. **Normalisation on both passes.** The paper works in $[0, 1]$ and says nothing about normalisation. The released class puts the reference pass through the preprocessor and the distilled pass through none, which is harmless in a repository whose models consume $[0, 1]$ input. The loaders here serve normalised tensors and the backbones expect them, so the port denormalises with `normalization_buffers`, clamps to $[0, 1]$, forms Eq. (2) in pixel space, renormalises with the same statistics and calls the model. The reference logits come from the served tensor through `forward_logits`. Without this the 2 passes would see the model at 2 different input scales and the logit gap would measure the normalisation rather than the mask. GTSRB's normalisation is the identity, so nothing may assume `std != 1`.
3. **Mask at native resolution.** The paper's mask matches its 32 by 32 inputs. The released code has no resize at all. Every backbone here is `Sequential(Resize((224, 224)), network)`, so the port optimises the mask at the dataset's own 32 or 64 pixels and lets the wrapper upsample the distilled input. At full coverage the L1 term is $\alpha h w$, about 10 at 32 pixels and 41 at 64, in the tens as in every paper setting, whereas a 224 by 224 mask would put it at 502 and swamp the logit term. The number is therefore the paper's statistic at the paper's scale. The trigger stays at the resolution it was stamped at.
4. **Fill drawn on the device.** The reference draws `torch.rand(b, c, 1, 1)` on the CPU and moves it. The port draws on the model's device, which is the same generator on the CPU and the CUDA generator on a GPU, both seeded by `seed_everything`. The bit-level cross-check therefore holds on the CPU. A GPU run reproduces itself but not the CPU fill sequence.
5. **Gradient to the mask only.** The class runs `backward` on the objective and relies on the driver having frozen the model, so a caller that forgets accumulates weight gradients on every step. The port wraps the loop in `frozen_parameters` and takes `torch.autograd.grad(objective, mask_parameter)`, assigning `.grad` by hand before `optimizer.step()`. No model parameter ever holds a gradient and the backward skips the weight-gradient graph, which is about a third of its cost. The test file asserts both the absent gradients and the restored `requires_grad` flags.
6. **Precision.** The released code is float32 end to end. The port runs the forward and the backward under the shared autocast policy, bfloat16 on CUDA when the context asks for it, with the mask parameter, the Adam state and the objective in float32 since `forward_logits` returns float32. Full precision without TF32 is 5 to 8 times slower and would put CD-L at hours per checkpoint. The smoke pair, bfloat16 against float32 on 200 GTSRB BadNet images, decides. Within 0.02 AUROC the `autocast` policy stands, otherwise `PRECISION_POLICY["cd_l"]` flips to `float32` and the estimate in the cost section grows by that factor. Either way 1 value applies to every cell.
7. **Threshold rule.** The paper thresholds at $\mu - \gamma \sigma$ over 1% of the clean training set with $\gamma = 1$. The port hands the raw norms to `defences.decision.detection_report`, which sets the threshold at a quantile of the shared clean validation split, so CD-L is judged at the same false-positive budget as every other detector. AUROC is unaffected. TPR at a fixed quantile is a different operating point from the paper's Table 10.
8. **Tiny ImageNet at 64 pixels.** The paper states $\alpha = 0.01$ for CIFAR-10 and GTSRB at 32 pixels and 0.001 for its ImageNet subset. It has no 64-pixel setting. The port keeps 0.01 on every dataset so that 1 constant covers the panel, which puts Tiny's full-coverage L1 term at 4 times the CIFAR value. A Tiny number from this port uses an $\alpha$ the paper never ran at that resolution.

## Hyperparameters

| Symbol | Paper default | This port | Constant name |
|---|---|---|---|
| $\alpha$, L1 weight (`gamma` in the code) | 0.01 for CD-L on CIFAR-10 and GTSRB, 0.001 on the ImageNet subset | 0.01 on every dataset | `DEFAULT_L1_WEIGHT` |
| $\beta$, TV weight | 10 | 1.0, the released default | `DEFAULT_TV_WEIGHT` |
| learning rate | 0.1 | 0.1 | `DEFAULT_LEARNING_RATE` |
| Adam $\beta_1, \beta_2$ | 0.1, 0.1 | (0.1, 0.1) | `ADAM_BETAS` |
| steps | 100 | 100, overridable through `DetectorContext.cd_l_steps` | `DEFAULT_NUM_STEPS` |
| $p$, mask norm | 1 | 1 | `MASK_NORM` |
| mask parameter at step 0 | unstated, ones in the code | 1.0, effective mask 0.8808 | `MASK_PARAMETER_INIT` |
| mask channels | 1, a 2D mask | 1 | `MASK_CHANNELS` |
| $\gamma$, threshold coefficient | 1 | none, the quantile rule replaces it | none |

The Adam moment decays share the letter $\beta$ with the TV weight in the paper. The table keeps the paper's symbols and disambiguates by name.

## Cost

Each input costs 1 reference forward plus 100 steps of a forward and a backward. Counting a backward as 1.5 forwards, the shared convention in the registry's cost table, that is $1 + 100 \times 2.5 = 251$ forward-equivalents per input, against 8 for STRIP, 6 for SCALE-UP and IBD-PSC and 71 for TeCo. With the model frozen the backward computes no weight gradients and lands nearer 2 forwards, so 2.5 is a ceiling.

At the measured 2130 images per second for a bfloat16 ViT-B/16 forward on the A100, the plan's estimate is about 46 minutes per GTSRB checkpoint (23208 scored inputs), which makes CD-L and SentiNet about 80% of the whole detector panel's bill. The smoke run replaces this estimate with a measured seconds-per-input figure that goes into `pbs/generate_detector_jobs.py`. A float32 policy would multiply the figure by 5 to 8.

## How to run

The smoke runs 1 checkpoint folder at 200 inputs per split and writes outside the results tree. `--allow-missing-psbd-cache` is required there because the smoke tree holds no PSBD split manifest to check against.

```bash
python -m cli.baselines --checkpoint-folder <folder> --detectors cd_l --max-samples 200 \
    --results-dir scratch/detector_smoke/results --allow-missing-psbd-cache
```

The panel runs through the `cd_l` job group of `pbs/generate_detector_jobs.py`, which emits 1 job per checkpoint with `--skip-existing` and a walltime of at least twice the smoke estimate. CD-L gets its own group because its runtime dwarfs the cheap detectors and a shared job would be sized for the wrong detector.

## Where results land

`results/<folder>/detectors/cd_l_metrics.json` holds the detection report at every quantile plus the provenance record. The raw per-sample norms sit beside it as `cd_l_scores_validation.pt`, `cd_l_scores_clean.pt` and `cd_l_scores_backdoor.pt`, 1 float32 tensor of the split's length each, in the loader's order, so any later threshold or fusion reads them without a rerun.

## Results

<!-- results:begin -->
<!-- results:end -->

## Known failure modes

A low score says the model's logits on that input can be reproduced from a small set of its pixels. That is what a patch trigger produces and it is also what any input with 1 dominant, compact feature produces, so a benign image with a small salient object can score low. The paper's own Table 1 reports CD-L at 0.85 AUROC on label-consistent attacks at test time against 0.94 or better elsewhere.

Full-image triggers have no compact region for the mask to collapse onto. Blend, SIG, WaNet and LF cover the whole image. The paper's Eq. (3) analysis shows its distilled masks for such triggers still cover about 40% of the image, so the gap to a clean mask is smaller. PSBD's Table 1 reports CD-L at TPR 0.031 on WaNet and 0.028 on Adaptive-Blend on GTSRB and at 0.462 on Tiny ImageNet BadNet. IBD-PSC's Table A17 reports 0.710 AUROC on WaNet. Blend at transparency under 2% evades it in the paper's own adaptive section.

An input-insensitive model breaks the score from the other side. If no pixel moves the logits, the L1 term wins everywhere and every mask collapses to the same value. 100 steps on a randomly initialised CNN gave a norm under 0.001, identical on every image, so the scores tie and AUROC is undefined. The synthetic fixture in `experiments/preflight/synthetic.py` is the relevant case here: its backdoor is added by boolean indexing, which has no autograd path to the trigger pixels, so CD-L differentiates through the fixture's random ViT alone and reads chance on it, AUROC 0.574 at 30 steps over 192 paired images. The test file takes the direction from a hand-built model with a differentiable trigger gate instead.

The bfloat16 policy is the remaining open risk. Gradient noise in bfloat16 changes an optimisation trajectory where it only rounds a forward pass, and the mask's 100-step path could diverge from the float32 path on a marginal input. The smoke pair exists to bound that.

## Direction

Low is poisoned and the score is returned unnegated. Eq. (4) flags an input whose mask norm falls at or below the threshold and the released analysis flags norms more than $t$ standard deviations below the clean mean. `detection_report` treats a low score as positive evidence, so all 3 agree without a sign change. The released analysis does return `1 - minmax(norm)` when it computes its own AUROC, which is a high-for-backdoor convention local to that script. The port does not reproduce it. A negation here would produce a well-formed, exactly inverted detector, which is the failure the `direction` field of the report exists to catch.
