# IBD-PSC, parameter-oriented scaling consistency

IBD-PSC amplifies the affine parameters of the normalization layers nearest a model's head and reads how often the amplified model still agrees with the unamplified model's prediction. Amplifying inflates every logit, which pushes a benign prediction off its class because the class evidence there is a comparison between similarly sized logits, while a backdoor maps its trigger to the target through a far larger margin and keeps its label and its probability under the same amplification. The method is white-box in the sense that it needs to rewrite the model's own parameters, and it costs $n + 1$ forward passes per input plus a fixed cost of choosing how many layers to amplify. This page records what the paper defines, what the released code does, what the port under `detectors/ibd_psc.py` runs on ViT and Swin, and the substantive architectural deviation the port cannot avoid.

## Citation

Hou et al., "IBD-PSC: Input-level Backdoor Detection via Parameter-oriented Scaling Consistency", ICML 2024, arXiv:2405.09786, PMLR v235 `hou24a`. Amplification is Section 4.3, Equation (2). Layer selection is Equation (3) and Algorithm 1. The score is Section 4.4, Equation (4). Defaults are stated in Section 5.1, $\omega = 1.5$, $n = 5$, $\xi = 0.6$, $T = 0.9$. The paper's own 2 adaptive attack designs are Section 5.4.

The released code is `third_party/BackdoorBox/core/defenses/IBD_PSC.py`, read at commit `af3afd1`. `count_BN_layers` at line 61 filters `isinstance(module1, torch.nn.BatchNorm2d)`, `prob_start` at line 89 loops `for layer_index in range(1, layer_num)` and returns nothing when the loop completes without crossing `xi`, and the ensemble at line 138 amplifies `sorted_indices[:layer_index+1]`. `third_party/backdoor-toolbox/other_defenses_tool_box/IBD_PSC.py` was read for comparison and agrees on the BatchNorm-only filtering.

## Threat model and data requirement

The adversary poisons the training data and the defender controls inference, with no knowledge of the trigger, the target class or the poisoning rate. The defender needs white-box access to the deployed model's normalization layers, since Algorithm 1 and the score both require rewriting a layer's scale and shift and rerunning a forward pass through the modified network.

Layer selection needs a labelled clean set, since Eq. (3) is a top-1 error against ground truth, and the paper states a budget of 100 benign samples. The score itself, Eq. (4), needs none, since it only compares each amplified model's output against the unamplified model's own prediction. The port gives layer selection the shared 2000-sample clean validation split with labels, so `DATA_REQUIREMENT` records the split as labelled even though the per-input score is data-free once the layer count is fixed.

## Mechanism

$$
\begin{aligned}
\hat{F}^{\omega}_k &= FC \circ \hat{f}^{\omega}_L \circ \cdots \circ \hat{f}^{\omega}_{L-k+1} \circ f_{L-k} \circ \cdots \circ f_1 && \text{(2)} \\
&\quad \text{with } \hat{\gamma} = \omega \gamma, \ \hat{\beta} = \omega \beta \\[4pt]
\eta &= \frac{1}{|D_r|} \sum_{(x, y) \in D_r} \mathbb{1}\big( \arg\max( \hat{F}^{\omega}_k(x) ) \neq y \big) && \text{(3)} \\[4pt]
k &= \text{smallest } k \in 1..L \text{ with } \eta > \xi && \text{Algorithm 1} \\[4pt]
PSC(x) &= \frac{1}{n} \sum_{i=k}^{k+n-1} \hat{F}^{\omega}_i(x)_{y'}, \quad y' = \arg\max( F(x) ) && \text{(4)}
\end{aligned}
$$

| Symbol | Meaning |
|---|---|
| $f_j$ | the model's $j$-th layer, unmodified |
| $\hat{f}^{\omega}_j$ | layer $j$ with its normalization affine parameters scaled by $\omega$ |
| $L$ | the number of amplifiable normalization layers |
| $\hat{F}^{\omega}_k$ | the model with the last $k$ normalization layers amplified by $\omega$ |
| $\gamma, \beta$ | a normalization layer's learned scale and shift |
| $D_r$ | the defender's labelled clean reference set |
| $\eta$ | Eq. (3)'s top-1 error of the amplified model on $D_r$ |
| $\xi$ | the error-rate threshold Algorithm 1 stops at |
| $k$ | the smallest amplified-layer count whose error crosses $\xi$ |
| $n$ | the ensemble size, how many further layer counts beyond $k$ are averaged |
| $y'$ | the unamplified model's own predicted label |
| $PSC(x)$ | the mean probability the ensemble assigns to $y'$ |

The descriptive form renames without rederiving.

$$
\begin{aligned}
\text{amplified\_model}(i) &= \text{the model with the last } i \text{ normalization layers' scale and shift both multiplied by } \omega \\
\text{clean\_error}(i) &= \text{top-1 error of amplified\_model}(i) \text{ on the clean validation split} \\
\text{start\_count} &= \text{smallest } i \text{ whose clean\_error exceeds } \xi \\
\text{psc}(\text{image}) &= \text{mean over the ensemble of the probability each amplified model gives the unamplified model's prediction}
\end{aligned}
$$

A normalization layer's affine parameters set the scale of the features that reach the classifier head, and multiplying them by $\omega > 1$ inflates every logit proportionally. A prediction that rests on a narrow margin between 2 or 3 comparably sized logits is easy to flip this way, since inflation amplifies noise in that comparison along with the signal, but a prediction that rests on a logit the model already places far above the rest survives the same inflation because the margin was never close to begin with. A backdoor is engineered to produce exactly the second kind of margin on a triggered input, so PSC stays high for a poisoned input and falls for an ordinary one as $k$ grows past the point where clean predictions start breaking.

## What the released code does

`count_BN_layers` walks the model's modules and counts every `torch.nn.BatchNorm2d`, which on a ConvNet gives 1 count per convolutional stage. `prob_start` runs Algorithm 1 by amplifying the first `layer_index` entries of the reversed layer list, scoring the clean validation set's top-1 error at each step, and returning as soon as the error exceeds `xi`. Its loop is `range(1, layer_num)`, which never reaches `layer_index = layer_num`, so the all-layers configuration is never tested, and the function falls off the end of the loop and returns `None` implicitly when the error never crosses `xi` at any tested count. The ensemble score amplifies `sorted_indices[:layer_index+1]` for `layer_index` running from `start_index` to `start_index + n - 1`, 1 more layer at every position than a literal reading of `sorted_indices[:layer_index]` would give. Amplification and restoration happen by deep-copying the model once per ensemble member per batch, `copy.deepcopy(self.model)`, rather than by writing and restoring parameters on the live model.

## What this port does on ViT

**Amplifying LayerNorm instead of BatchNorm is the substantive deviation.** The paper scales BatchNorm2d and only BatchNorm2d, and the released code matches, filtering on `isinstance(module, torch.nn.BatchNorm2d)`. Neither ViT nor Swin contains a single BatchNorm layer, so `count_BN_layers` returns 0, `sorted_indices` is empty, Algorithm 1's loop runs over `range(1, 0)`, and `start_index` comes back `None`. IBD-PSC as published is not runnable on either architecture this project trains.

The port amplifies `nn.LayerNorm` instead, through `amplifiable_norm_layers` in `detectors/ibd_psc.py`. The substitution is exact at the level of what Eq. (2) does to a single layer's output, since both normalize first and apply the affine map second:

    omega*gamma * x_hat + omega*beta = omega * (gamma * x_hat + beta)
                                     = omega * (the layer's original output)

What does not carry over is the architectural claim behind the choice of BatchNorm. The paper's $L$ counts 1 normalization layer per convolutional stage, whereas a ViT or Swin block holds 2 LayerNorms sitting on the 2 branch inputs of a residual stream, so amplifying 1 of them scales a branch rather than the whole stream and its effect on the logits is weaker per layer than a BatchNorm scaling is in a ConvNet. `amplifiable_norm_layers` finds 25 affine LayerNorm modules on ViT-B/16 and 53 on Swin-S, both far above a typical ConvNet's BatchNorm count. Algorithm 1 absorbs the weaker per-layer effect by construction, since it selects $k$ from measured clean error rather than from a fixed depth, but the $k$ it selects on a ViT is not comparable to a $k$ published against a BatchNorm network.

Further deviations, each smaller than the substitution above:

1. **Layer count in the ensemble.** Eq. (4) sums over $i = k, \ldots, k+n-1$ amplified layers. The released code amplifies `sorted_indices[:layer_index+1]`, 1 more layer at every position than the equation. `psc_scores` follows the equation and sums over exactly `member_counts = range(start_layer_count, start_layer_count + ensemble_size)`, so a port score at a given $k$ and $n$ uses 1 fewer amplified layer at each ensemble position than the released code's score at the same reported $k$ and $n$.
2. **Algorithm 1's range.** The paper loops $i = 1$ to $L$. The released code loops `range(1, layer_num)`, so it never tests the all-layers configuration and returns `None` when the error rate never crosses $\xi$, which crashes the next call downstream with no message. `select_start_layer_count` follows the paper, tests $i = 1 \ldots L$ inclusive and falls back to $k = L$ when no $i$ crosses, the value Algorithm 1 holds at loop exit rather than a value the released code can even represent.
3. **Ensemble clamping.** When $k$ sits close to $L$, the window $k \ldots k+n-1$ runs past $L$, which is undefined in the paper because its BatchNorm counts are large enough that this rarely happens. `psc_scores` keeps only the ensemble members with $i \le L$, so a late $k$ on a shallow network gives a smaller ensemble rather than amplifying a layer index that does not exist, and the record's `hyperparameters` block shows how many members were actually used.
4. **Data budget.** The paper states 100 benign samples for $D_r$. The released demo uses 2000. `select_start_layer_count` uses the shared clean validation split, so IBD-PSC's layer selection sees the same data every other detector in the registry sees.
5. **No deep copies.** The released code calls `copy.deepcopy(self.model)` once per ensemble member per batch, 5 full model copies per batch at the default ensemble size. `amplified_parameters` writes the amplified parameters onto the live model in place and restores them from saved clones of the weight and bias tensors alone on exit, which is numerically exact and allocates no second copy of the model.

`amplifiable_norm_layers` reverses definition order, matching the released code's `list(reversed(range(layer_num)))`, so element 0 is the layer nearest the head. Definition order equals execution order for torchvision's ViT-B/16 and Swin-S, so the reversal really is depth ordering here. That would not hold for an architecture that declares its modules out of forward order, which is a latent assumption the released code shares and the port inherits rather than fixes.

**The calibrated variant.** The smoke of 2026-09-10 (`docs/runs/2026-09-10-detector-smoke.md`) showed the substitution above is not enough on its own. At the paper's $\omega = 1.5$ a 99%-accurate GTSRB ViT keeps its predictions through every amplified LayerNorm, Algorithm 1 never crosses $\xi$, $k$ falls back to $L$, and every input, clean or poisoned, retains its label at probability 1, so the AUROC is 0.506 with 31% of the validation scores tied at the threshold. `ibd_psc_calibrated` runs Algorithm 1 at each $\omega$ in `CALIBRATION_FACTORS`, 1.5, 2, 3, 5 and 8, and scores at the first $\omega$ whose trace crosses $\xi$, recording the chosen $\omega$ and $k$ in the run's `hyperparameters` under `scaling_factor` and `start_layer_count`. A model the paper's setting already breaks is scored exactly as `ibd_psc` scores it, since 1.5 is tried first. Both names run on the panel, the faithful port so the paper's own setting is on record and the calibrated one so the method gets the amplification its mechanism needs on this architecture.

## Hyperparameters

| Symbol | Paper default | This port | Constant name |
|---|---|---|---|
| $\omega$, scaling factor | 1.5 | 1.5 | `DEFAULT_SCALING_FACTOR` |
| $n$, ensemble size | 5 | 5 | `DEFAULT_ENSEMBLE_SIZE` |
| $\xi$, error threshold | 0.6 | 0.6 | `DEFAULT_ERROR_THRESHOLD` |
| $T$, detection threshold | 0.9 | recorded, unused, the registry's quantile rule replaces it | `DEFAULT_DETECTION_THRESHOLD` |
| $L$, amplifiable layer count | count of `BatchNorm2d`, architecture-dependent | count of affine `LayerNorm`, 25 on ViT-B/16, 53 on Swin-S | computed by `amplifiable_norm_layers` |
| clean reference budget | 100 per Section 5.1 | the shared 2000-sample split | none, see deviation 4 |
| `CALIBRATION_FACTORS` | not in the paper | 1.5, 2, 3, 5, 8 | `ibd_psc_calibrated` only, the paper's value first |

## Cost

$n + 1$ forward passes per input, 6 at the default ensemble size, against 1 for confidence, 8 for STRIP, the same 6 for SCALE-UP and 71 for TeCo. Layer selection adds a fixed cost of up to $L$ full passes over the 2000-sample validation split before any input is scored, since Algorithm 1 tests amplified-layer counts 1 by 1 until the error crosses $\xi$ or the loop exhausts $L$, so a checkpoint whose clean error is slow to cross $\xi$ pays close to $L$ passes, 25 on ViT-B/16 or 53 on Swin-S, purely for layer selection. `ibd_psc` is in `NEEDS_FITTING` for exactly this reason, and a deployment pays that cost once per model rather than once per scored input.

## How to run

The smoke runs 1 checkpoint folder at 500 inputs per split and writes outside the results tree. `--allow-missing-psbd-cache` is required there because the smoke tree holds no PSBD split manifest to check against.

```bash
python -m cli.baselines --checkpoint-folder <folder> --detectors ibd_psc --max-samples 500 \
    --results-dir scratch/detector_smoke/results --allow-missing-psbd-cache
```

The panel runs through the `cheap` job group of `pbs/generate_detector_jobs.py`, which puts IBD-PSC beside confidence, STRIP, both SCALE-UP variants and Beatrix in 1 job per checkpoint with `--skip-existing`, since all of them finish within minutes and a job sized for CD-L or TeCo would idle the GPU on them.

## Where results land

`results/<folder>/detectors/ibd_psc_metrics.json` holds the detection report at every quantile plus the provenance record, whose hyperparameters carry the scaling factor, the ensemble size, the error threshold and the selected layer count. The raw per-sample scores sit beside it as `ibd_psc_scores_validation.pt`, `ibd_psc_scores_clean.pt` and `ibd_psc_scores_backdoor.pt`, 1 float32 tensor of the split's length each, in the loader's order, so any later threshold or fusion reads them without a rerun.

## Results

<!-- results:begin -->
<!-- results:end -->

## Known failure modes

A low score says the amplified ensemble stopped agreeing with the unamplified model's own prediction, which is what a thin decision margin produces whether or not it came from a trigger. `python -m experiments.preflight.check_signs` reads IBD-PSC at AUROC 1.0000 on the synthetic fixture, where the backdoor is unmissable by construction, which shows the port's plumbing and sign are correct without saying anything about a real checkpoint's margin structure.

The paper's own adaptive section reports 2 designs that push IBD-PSC toward failure, and both cost the attacker. Design 1 reduces IBD-PSC's worst-case AUROC to 0.819, still well above chance, a partial rather than a full break (Hou et al., Section 5.4). Design 2 defeats the detector more completely but does so by collapsing the model's own benign accuracy to 0.101, which is a broken model rather than a usable attack, the same shape TeCo's own adaptive attack takes when it costs 40 points of clean accuracy to evade. Neither of the paper's own adaptive designs is a free evasion in the way STRIP's or SCALE-UP's are.

The project's own theoretical framework predicts a 3rd failure mode that has not yet been run as a training experiment. `docs/attack-design/A5-low-confidence-backdoor.md` argues that amplifying $\gamma$ and $\beta$ breaks a thin decision margin as easily as it breaks a clean one, since the mechanism the paper relies on, inflating a close comparison until it flips, does not distinguish why the comparison was close. That analysis predicts an AUROC below 0.55 for IBD-PSC against a backdoor deliberately trained at low confidence, and it is a prediction rather than a measurement, carrying the same caveat as the identical claim in `docs/detectors/confidence.md`, that the document's own PSBD conclusion needed a later correction (A21) even though the IBD-PSC row itself was not the part corrected.

The smoke of 2026-09-10 measured the saturation the calibrated variant exists for: on `vit_gtsrb_badnet_a2o_0_05` at $\omega = 1.5$ every backdoor score is exactly $-1$ and the clean scores average $-0.98$ on both the 500-image and the full 2000-image split, so the detector reads 0.481 and 0.506. On `vit_gtsrb_blend_0_05` the same setting reads 0.970, which says the amplification does bite on a model whose clean margins are thinner, and that the failure is a scale mismatch rather than a wrong sign. The table with every number is in `docs/runs/2026-09-10-detector-smoke.md`.

Algorithm 1's selected $k$ is itself worth reading alongside the AUROC. `docs/attack-design/A1-operating-point-and-threshold.md` notes that Algorithm 1 picks $k$ from the clean top-1 error rate crossing $\xi$, so sharpening a model's clean confidence raises the amplification needed to break its predictions, which pushes $k$ upward and, on an architecture with a limited layer budget, can leave fewer than $n$ members available once the ensemble clamp at deviation 3 applies. A $k$ landing at 1 or at $L$ on a given checkpoint is a sign the amplification scale was mismatched to that model rather than evidence about the backdoor.

## Direction

Low is poisoned. The paper's rule is "poisoned if $PSC(x) > T$", so the raw statistic is high for poisoned, the opposite of the registry's convention. `ibd_psc_scores` negates the raw `psc_scores` once at the return boundary. A second negation anywhere would produce a well-formed, exactly inverted detector, which is the failure the `auroc_two_sided` diagnostic field in `defences.decision.detection_report` exists to surface.
