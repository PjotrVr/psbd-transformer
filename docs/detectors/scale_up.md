# SCALE-UP, scaled prediction consistency

SCALE-UP multiplies every pixel of an input by a ladder of integer factors and reads how often the amplified copies keep the same predicted label as the original. Amplifying every pixel toward saturation destroys the class evidence of an ordinary image, so its predicted label tends to move, while a high-contrast trigger survives the amplification and keeps the prediction consistent. The method is black-box, needs only predicted labels and costs $|S| + 1$ forward passes per input. 2 variants are ported in 1 file, data-free (Eq. 2 alone, the paper's headline method) and data-limited (Eqs. 3 and 4, standardized per predicted class against clean statistics). This page records what the paper defines, what the released implementations do, what the port under `detectors/scale_up.py` runs on ViT and Swin and where they diverge.

## Citation

Guo et al., "SCALE-UP: An Efficient Black-box Input-level Backdoor Detection via Analyzing Scaled Prediction Consistency", ICLR 2023, arXiv:2302.03251, OpenReview `o0LFPcoFKnr`. The data-free statistic is Section 4.2, Equation (2). The data-limited variant is Section 4.3, Equations (3) and (4). Theorem 1 proves the limiting case for an RBF kernel regressor at a 50% poisoning rate. The paper's own adaptive evaluation is Eq. 5.

3 implementations were read. The authors' own released code uses `range(1, 12)` for the scaling set, a fact this project's own port record already carried forward from an earlier reading and is not re-derived here since that repository is not vendored in this project. 2 third-party ports were read directly and are vendored: `third_party/BackdoorBox/core/defenses/SCALE_UP.py` at commit `af3afd1`, and `third_party/backdoor-toolbox/other_defenses_tool_box/scale_up.py` at commit `9d4d909`. Both use the paper's own scaling set $\{3, 5, 7, 9, 11\}$ rather than `range(1, 12)`, and both compare the amplified prediction against the clean validation label rather than against $C(x)$ when fitting Eq. (3).

## Threat model and data requirement

The adversary poisons the training data and the defender controls inference, with no knowledge of the trigger, the target class or the poisoning rate. The defender needs only the model's predicted label on an input it supplies, not the full softmax vector, which the paper states explicitly, "we only assume to have the predicted label instead of the predicted probability vector", so the method is black-box in a strictly narrower sense than STRIP or the confidence null.

Data-free needs no clean data at all, since Eq. (2) compares an amplified prediction against the same input's own unamplified prediction. Data-limited standardizes that statistic per class against clean references, which the paper budgets at 100 benign samples per class. The port gives the data-limited variant the shared 2000-sample clean validation split with labels, about 200 samples per class on CIFAR-10, 20 on CIFAR-100, 46 on GTSRB and 10 on Tiny ImageNet, so `DATA_REQUIREMENT["scale_up_data_limited"]` records the split as labelled while `DATA_REQUIREMENT["scale_up"]` stays `none`.

## Mechanism

$$
\begin{aligned}
SPC(x) &= \frac{1}{|S|} \sum_{n \in S} \mathbb{1}\{ C(n \cdot x) = C(x) \} && \text{(2)} \\
\mu_i &= \mathbb{E}_{x \in X_i}[ SPC(x) ] && \text{(3)} \\
\sigma_i &= \sqrt{ \mathbb{E}_{x \in X_i}[ (SPC(x) - \mu_i)^2 ] } && \text{(3)} \\
NSPC(x) &= \frac{ SPC(x) - \mu_{\hat{y}} }{ \sigma_{\hat{y}} }, \quad \hat{y} = C(x) && \text{(4)}
\end{aligned}
$$

| Symbol | Meaning |
|---|---|
| $S$ | the set of integer scaling factors |
| $n$ | 1 scaling factor from $S$ |
| $C(x)$ | the classifier's predicted label on $x$ |
| $X_i$ | the defender's benign samples of true class $i$ |
| $SPC(x)$ | the fraction of amplified copies whose predicted label matches $C(x)$ |
| $\mu_i, \sigma_i$ | the mean and standard deviation of clean $SPC$ within class $i$ |
| $\hat{y}$ | the predicted label of the query, the class $NSPC$ is standardized against |
| $NSPC(x)$ | the standardized SPC, a z-score of $SPC(x)$ against class $\hat{y}$'s clean statistics |

The descriptive form renames without rederiving.

$$
\begin{aligned}
\text{spc}(\text{image}) &= \frac{1}{|S|} \sum_{n \in S} \mathbb{1}\big[\, \text{predicted}(n \cdot \text{image}) = \text{predicted}(\text{image}) \,\big] \\
\text{nspc}(\text{image}) &= \frac{ \text{spc}(\text{image}) - \text{clean\_mean}[\text{predicted class}] }{ \text{clean\_std}[\text{predicted class}] }
\end{aligned}
$$

Multiplying every pixel toward the $[0, 1]$ ceiling and clipping there erases most of an ordinary image's fine structure, since a scaled and clipped image tends toward flat, saturated regions that carry little of the original class evidence. A patch or blend trigger is built to be a strong, high-contrast signal precisely so it survives compression, cropping and other deployment-time transforms, and that same robustness carries it through pixel amplification, so the amplified copies keep agreeing with the unamplified prediction far more often than a clean image's copies do. The paper's decision rule is "backdoor if $SPC(x) > T$", high for poisoned, which the port negates once at the scoring boundary.

## What the released code does

Both released implementations amplify pixels by clipping $n \cdot x$ to $[0, 1]$ and compare the amplified prediction against a reference to build $SPC(x)$, matching Eq. (2). Where they diverge from the paper is in the data-limited variant. `backdoor-toolbox`'s copy denormalizes before multiplying and renormalizes after clipping, `self.normalizer(torch.clip(self.denormalizer(clean_img) * scale, 0.0, 1.0))`, which keeps the multiply and the clip in pixel space as Section 4.2 states. `BackdoorBox`'s copy has that same denormalize and renormalize call present but commented out, and instead multiplies the loader's own tensor directly, `torch.clip(clean_img * scale, 0.0, 1.0)`, which is only correct if the loader already serves $[0, 1]$ pixels rather than normalized ones.

Both fit their per-class Eq. (3) statistics by comparing the amplified prediction against the clean validation set's ground-truth label rather than against $C(x)$, `spc += scale_label == labels` in both `init_spc_norm` and its equivalent, which disagrees with Eq. (2)'s own rule exactly on the samples the model gets wrong. Both also mask scored samples by prediction correctness before reporting a number, `mask = torch.eq(labels, original_pred)` in BackdoorBox and an equivalent `clean_pred_correct_mask` and `poison_attack_success_mask` pairing in backdoor-toolbox, which on the poisoned side keeps only images the trigger actually fooled and on the clean side keeps only images the model already classifies correctly.

## What this port does on ViT

1. **Scaling set.** The paper writes $S = \{3, 5, 7, 9, 11\}$ in Section 4.2, introduced with "e.g." and never restated in the experimental settings. The authors' released code uses `range(1, 12)`, so $S = \{1, \ldots, 11\}$, a different statistic on a different support, since $n = 1$ sits inside the average where it is trivially consistent and lifts the floor of SPC from 0 to $1/11$. `PAPER_SCALES` in `detectors/scale_up.py` defaults to the paper's set, and `OFFICIAL_CODE_SCALES` names the code's set without using it, so whichever set produced a given number is recorded rather than silently inherited.
2. **Per-class statistics on a shared budget.** Eq. (3) is per class, and the paper budgets 100 benign samples per class. Every detector here shares the single 2000-sample clean validation split, about 20 per class on CIFAR-100 and possibly 0 for a class the split missed by chance. `fit_class_spc_statistics` falls back to the pooled mean and standard deviation below `MIN_CLASS_SAMPLES`, a case the paper's own budget never reaches. A second fallback guards a different failure, since SPC lives on a grid of $1/|S|$ and takes only $|S| + 1$ distinct values, so a class whose few clean samples all land on the same grid point has $\sigma_i$ exactly 0. Left alone that would clamp to `STD_FLOOR` and Eq. (4) would return z-scores of order $10^5$ for that class alone, which does not move AUROC but makes the score scale meaningless and any absolute threshold unusable. This was observed on `vit_cifar100` at the shared budget.
3. **Calibration reference.** Eq. (2) always measures consistency against $C(x)$, the model's own prediction. Both released implementations compare against the ground-truth validation label instead when fitting Eq. (3), which disagrees with Eq. (2) exactly on the samples the model gets wrong. `spc_scores` and `standardize_spc` use $C(x)$ throughout, and the grouping into $X_i$ uses the validation sample's true class, which is what Eq. (3) itself specifies for the class membership, distinct from what the released code compares against inside that class.
4. **No input noise.** Both released implementations' broader codebases add $0.02 \cdot U[0, 1)$ to every test sample before scaling in some of their other detectors' preprocessing paths. That noise term appears in no equation of this paper, has no stated ablation here and is not reproduced by `amplify_pixels`.
5. **No prediction-correctness masking.** Both released implementations drop every sample whose prediction disagrees with its dataset label before scoring, which on the poisoned side keeps only successfully attacked images. This project already owns that decision through `attacks.poisoning.AttackSuccessSet`, so `scale_up_scores` scores every sample the loader serves and the eligibility rule stays in exactly 1 place rather than being duplicated inside the detector.
6. **Cross-fitted validation scores.** Neither released implementation faces this problem, because both fit Eq. (3) on a private clean pool and threshold on a separate one. This project's data-limited variant fits Eq. (3) and reads its detection threshold from the same 2000-sample clean validation split, so a validation sample standardized against statistics it helped fit sits closer to its class mean than a fresh sample would, and an in-sample threshold would be too tight, pushing the achieved false positive rate on the paired clean split above the budget the quantile rule targets. `cross_fitted_validation_scores` splits the validation split into `CROSS_FIT_FOLDS` (2) folds by position and standardizes each fold against `fit_class_spc_statistics` fitted on the other fold only, so every validation score is out of fit. `scale_up_data_limited` is in `CROSS_FITTED` for exactly this reason, and the clean and backdoor splits, fitted on the full 2000-sample statistics rather than a held-out fold, are unaffected since they were never part of the fitting set.

## Hyperparameters

| Symbol | Paper default | This port | Constant name |
|---|---|---|---|
| $S$, scaling set | $\{3, 5, 7, 9, 11\}$, introduced with "e.g." | $(3, 5, 7, 9, 11)$ | `PAPER_SCALES` |
| $S$, released code's set | not restated as such | $(1, \ldots, 11)$, recorded, unused | `OFFICIAL_CODE_SCALES` |
| minimum class samples | none, paper budgets 100 per class | 5 | `MIN_CLASS_SAMPLES` |
| $\sigma$ floor | none | $10^{-6}$ | `STD_FLOOR` |
| cross-fit folds | none, the paper never fits and thresholds on the same pool | 2 | `CROSS_FIT_FOLDS` |
| input noise | $0.02 \cdot U[0, 1)$ in parts of the released codebases, no equation here | none | see deviation 4 |
| $T$, detection threshold | a fixed cutoff on $SPC(x)$ or $NSPC(x)$ | none, the registry's quantile rule replaces it | none |

## Cost

$|S| + 1$ forward passes per input, 6 at the paper's default scaling set, for both variants, against 1 for confidence, 8 for STRIP and 71 for TeCo. Data-limited adds a fixed fitting cost, 1 pass over the 2000-sample validation split at the same 6 forwards per input, so `scale_up_data_limited` is in `NEEDS_FITTING` while `scale_up` is not, and a deployment paying that cost pays it once per model rather than once per scored input.

## How to run

The smoke runs 1 checkpoint folder at 500 inputs per split and writes outside the results tree. `--allow-missing-psbd-cache` is required there because the smoke tree holds no PSBD split manifest to check against.

```bash
python -m cli.baselines --checkpoint-folder <folder> --detectors scale_up scale_up_data_limited \
    --max-samples 500 --results-dir scratch/detector_smoke/results --allow-missing-psbd-cache
```

The panel runs through the `cheap` job group of `pbs/generate_detector_jobs.py`, which puts both SCALE-UP variants beside confidence, STRIP, IBD-PSC and Beatrix in 1 job per checkpoint with `--skip-existing`, since all of them finish within minutes and a job sized for CD-L or TeCo would idle the GPU on them.

## Where results land

`results/<folder>/detectors/scale_up_metrics.json` and `results/<folder>/detectors/scale_up_data_limited_metrics.json` each hold that variant's detection report at every quantile plus its provenance record. The raw per-sample scores sit beside each as `<name>_scores_validation.pt`, `<name>_scores_clean.pt` and `<name>_scores_backdoor.pt`, 1 float32 tensor of the split's length each, in the loader's order, so any later threshold or fusion reads them without a rerun. `scale_up_data_limited`'s validation tensor is the cross-fitted, out-of-fit scores rather than an in-sample fit.

## Results

<!-- results:begin -->
<!-- results:end -->

## Known failure modes

A high SPC says the prediction survived pixel amplification, which is what a high-contrast trigger produces and what any input whose class evidence is concentrated in a few extreme-valued pixels also produces, so a naturally saturated or overexposed clean image can score the same way. `SPC(x)` lives on a grid of only $|S| + 1$ values, 6 at the paper's default scaling set, so the statistic is coarse by construction. `experiments/preflight/gate.py` treats this directly, excluding both variants from its synthetic sign check because a barely trained model keeps almost every clean prediction stable under amplification, tying the clean population onto the grid's maximum value, `MAXIMUM_CLEAN_CONCENTRATION = 0.9`. Running `python -m experiments.preflight.check_signs` on the fixture reads `scale_up` at AUROC 0.5117 and `scale_up_data_limited` at 0.0234, both flagged `NOT JUDGED`, and the second number in particular shows how a near-tied grid statistic can read as a strong inversion on the handful of values that do vary once it is standardized, without that reading being evidence about the method itself. A real checkpoint with a well separated clean population under amplification does not have this problem, but a weak or undertrained one can, and the tie share of the clean scores is worth reading alongside the AUROC for exactly that reason.

The paper's own adaptive section is the second failure mode, and it is measured rather than predicted. Eq. 5's scale-resistant regularization term pulls the amplified predictions of a poisoned image away from consistency during training, and the project's own cross-defence review reports this pushes SCALE-UP's AUROC to 0.467, below chance, with the clean-accuracy and ASR cost not reported by the paper's authors (`docs/attack-design/cross-defence.md`, Section 5). Since the attacker pays an unreported price, this is a documented weakness rather than a free evasion in the way STRIP's Section VI-F attack is, but the direction of the effect is the paper's own.

SCALE-UP's threshold is coupled to the model's confidence in the data-limited variant specifically. `docs/attack-design/A1-operating-point-and-threshold.md` notes that Eq. (3)'s standardization is against per-class clean mean and standard deviation, so an attack or a training change that inflates the clean spread inflates $\sigma_i$ and compresses every $NSPC$ score toward 0, which moves the quantile threshold along with the scores it is meant to separate. The data-free variant's fixed cutoff on $SPC(x)$ carries no such coupling, since Eq. (2) alone reads no clean statistic at all.

## Direction

Low is poisoned. The paper's rule is "backdoor if $SPC(x) > T$", so the raw statistic, $SPC(x)$ for the data-free variant and $NSPC(x)$ for the data-limited one, is high for poisoned. `scale_up_scores` negates it once at the return boundary, and `cross_fitted_validation_scores` applies the same negation to the out-of-fit validation scores so the 2 cannot disagree in sign. A second negation anywhere would produce a well-formed, exactly inverted detector, which is the failure the `auroc_two_sided` diagnostic field in `defences.decision.detection_report` exists to surface.
