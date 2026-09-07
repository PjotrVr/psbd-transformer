# SAM-enhanced PSD (arXiv 2411.11525v1, CVPR-2025 submission format, 2024)

Zhang, Zhu, Zhu, Wu (CUHK-Shenzhen). "Reliable Poisoned Sample Detection against Backdoor
Attacks Enhanced by Sharpness Aware Minimization."
Source read: LaTeX at
`papers/reliable_poisoned_sample_detection_against_backdoor_attacks_enhanced_by_sharpness_aware_minimization/`
(`main.tex`, `sec/0_abstract.tex`, `sec/1_intro.tex`, `sec/2_related.tex`, `sec/3_method.tex`,
`sec/4_experiment.tex`, `sec/6_conclusion.tex`, `sec/proof.tex`, `sec/X_suppl.tex`, `tables/*.tex`,
`figs/*.png`). No reference code is released (no GitHub URL anywhere in `sec/` or `main.bib`).
All labels below are `paper` unless stated; there is no code side to compare against.

## Threat model

| Item | Content | Source |
|---|---|---|
| Attack surface | Data poisoning only. Attacker releases a poisoned training set `D_tr` and cannot touch the victim's training process. | paper, Sec. 3.1 "Threat model" |
| Poisoned set construction | `D_tr = (D_cl \ D_sub) ∪ D_poi`, `D_poi = {(g(x, Δ), y_t)}`. Poisoning ratio `p = |D_sub| / |D_cl|`. | paper, Sec. 3.1 |
| Defender position | Defender is the *trainer*. The defender trains the model himself on the untrusted dataset (pre-training defense stage) and then filters poisoned samples out of it. | paper, Sec. 3.1 + Sec. 2 "Backdoor defense" |
| Defender knowledge | Does not know poisoning ratio `p`, trigger `Δ`, or generation function `g`. Has a small clean reference set drawn from the same distribution as `D_cl`. | paper, Sec. 3.1 "Defender's goal" |
| Clean reference set size | 250 samples per class, taken from the *test* set. | paper, Sec. 4.1 "Detection settings" |
| Defender's control | Full control of the optimizer/training algorithm. This is the whole method: the defender swaps vanilla training for SAM. | paper, Sec. 3.3 |
| Defender's goal | Identify `D_poi` inside `D_tr`, i.e. sample-level detection on the *training* set, not on a held-out test set. | paper, Sec. 3.1 |

Key consequence: the object being scored is the **training set**, and the model is trained by the
defender from scratch. This is a pre-training / data-filtering threat model, not a
model-forensics threat model where the defender receives an already-trained third-party model.

## Equations

### 1. Trigger Activation Change (TAC), the "backdoor effect" measure

Original form (Sec. 3.2, Eq. 1; metric imported from CLP):

```
TAC_k^{(l)}(D) = (1 / |D|) * Σ_{x ∈ D} || f_k^{(l)}(x) - f_k^{(l)}(x̃) ||_2
```

Descriptive form:

```
trigger_activation_change(neuron, layer, clean_set) =
    mean over clean images x of
        L2_norm( activation(neuron, layer, x) - activation(neuron, layer, add_trigger(x)) )
```

where `x̃ = g(x, Δ)` is the triggered version of the same image `x`, and the network is written
`f_θ = f^(L) ∘ ... ∘ f^(1)`. Backdoor effect of a model = mean of the **Top-K** TAC values;
`K = 2` in Fig. 2 (`sec/3_method.tex` Fig. `tac_auc` caption). `paper`

### 2. SAM objective

Original form (Sec. 3.3, Eq. 2):

```
min_θ  max_{ε : ||ε||_2 ≤ ρ}  L(θ + ε),
L(θ) = (1 / |D_tr|) Σ_{(x,y) ∈ D_tr} ℓ(f_θ(x); y)
```

Descriptive form:

```
minimize over weights:
    worst_case_loss = max over weight_perturbation with L2_norm <= perturbation_radius of
                          cross_entropy_loss(weights + weight_perturbation)
loss is averaged over the WHOLE poisoned training set (clean part and poisoned part together);
perturbation_radius (rho) > 0 is the weight-perturbation budget
```

Note the loss is over all of `D_tr`; there is no separate poisoned-only loss term and no
clean/poison split in the objective. `paper`

### 3. SAM first-order expansion used in the proof

Original form (`sec/proof.tex`):

```
∇ℓ(θ + ρ ∇ℓ(θ)/||∇ℓ(θ)||_2)
  = ∇ℓ(θ) + ρ ∇²ℓ(θ) ∇ℓ(θ)/||∇ℓ(θ)||_2 + O(ρ²)
  = ∇[ ℓ(θ) + ρ||∇ℓ(θ)||_2 + O(ρ²) ]
```

Descriptive form: to first order, one SAM step equals one gradient step on
`cross_entropy_loss + perturbation_radius * gradient_norm`, i.e. SAM is gradient-norm
regularization. `paper`

### 4. Proposition (the theoretical claim)

Original form (Sec. 3.3, Proposition 3.1):

```
For any activated neuron in a two-layer ReLU network f(θ) = a * σ(Wx) trained with
cross-entropy loss, each SAM update increases the pre-activation ⟨w_j, x̃⟩ relative to SGD,
for a poisoned sample x̃, given

    a_j σ'(⟨w_j, x̃⟩) < - σ(⟨w_j, x̃⟩) / ( (1 - ℓ'(θ)) ||∇f(θ)||²_2 )
```

Descriptive form:

```
for a 2-layer ReLU net (second layer weights `a`, first layer weights `W`),
if  second_layer_weight(j) * relu_derivative(preactivation_j_on_poisoned_input)
      <  - relu(preactivation_j_on_poisoned_input)
         / ( (1 - sigmoid_loss_derivative) * squared_grad_norm_of_f )
then one SAM step raises neuron j's pre-activation on the poisoned input more than one SGD step does.
```

Remark (Sec. 3.3): neurons satisfying the condition have strongly negative `a_j` and are
activated by the poisoned sample; the authors identify these as backdoor neurons and conclude
SAM raises poisoned-sample activation on backdoor neurons. Target label is assumed `y_t = 0`.
`paper`

### 5. Feature scaling (Stage 2 of the method)

Original form (Sec. 3.4, Stage-2):

```
g = φ_{θ_SAM}(x)
g^s = Σ^{-1/2} P g
```

Descriptive form:

```
raw_feature       = penultimate_feature_extractor_of_SAM_model(x)
projection_matrix = PCA basis estimated from the (poisoned) training dataset
covariance_matrix = covariance estimated from the clean reference samples PLUS
                    "potential clean samples dynamically collected from the poisoned dataset"
scaled_feature    = inverse_matrix_square_root(covariance_matrix) @ projection_matrix @ raw_feature
```

Purpose stated: undo the **increased intra-class variance of clean features** that SAM causes
(Fig. 4). Exact PCA rank, how "potential clean samples" are selected, and covariance
regularization are `NOT STATED` (see "Not stated"). `paper`

## Hyperparameters

| Name | Value | Where it came from | paper/code |
|---|---|---|---|
| SAM perturbation radius `rho` (main tables) | 0.1 | Sec. 4.1 "Detection settings", last sentence | paper |
| `rho` sweep (Fig. 10) | 0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50 (read off the x-axis of `figs/fig10_rho.png`); `rho = 0.0` is the vanilla-training baseline | Sec. 4.3 + figure | paper |
| Base optimizer under SAM | `NOT STATED` in text. The proof and Sec. 3.3/Sec. 4 prose contrast SAM against "Vanilla Training (SGD)", so SGD is implied; never written as a setting. BackdoorBench default is SGD. | looked in Sec. 3.3, Sec. 3.4, Sec. 4.1, proof.tex | paper |
| Learning rate | `NOT STATED` | grepped all of `sec/`, `tables/`; only "default settings provided by BackdoorBench" (Sec. 4.1) | paper |
| Epochs | `NOT STATED` | same | paper |
| Batch size | `NOT STATED` | same | paper |
| LR schedule / warmup | `NOT STATED` | same | paper |
| Weight decay, momentum | `NOT STATED` | same | paper |
| Augmentation, resolution, normalization stats | `NOT STATED` (inherits BackdoorBench defaults by reference) | same | paper |
| SAM variant (m-sharpness, adaptive/ASAM, per-batch vs per-GPU) | `NOT STATED`; the equation is plain L2-ball SAM (Foret et al.) | Sec. 3.3 | paper |
| Poisoning ratio (main tables) | 5% for all attacks | Sec. 4.1 "Attack settings" | paper |
| Poisoning ratio sweep (Fig. 11) | 0.1%, 0.5%, 1%, 5% | Sec. 4.2 | paper |
| Attack configs | "default settings provided by BackdoorBench" | Sec. 4.1 | paper |
| BadNets-A2A target rule | `y_t = (y + 1) mod K` | Sec. 4.1 | paper |
| Clean reference set | 250 samples per class, from the **test** set | Sec. 4.1 | paper |
| Feature layer used | Penultimate feature extractor `φ`; the neuron-level analyses use the **last convolutional layer of ResNet18** (512 neurons, confirmed by the x-axis range in `figs/fig3_diff_tac.png`) | Sec. 3.4, Sec. 4.4 | paper |
| PCA rank for `P` | `NOT STATED` | Sec. 3.4 is the only description | paper |
| Seeds / repetitions | `NOT STATED` (no seed count, no error bars anywhere) | all tables and figures | paper |
| Silhouette-coefficient values quoted | BadNets 0.19 (vanilla) -> 0.32 (SAM); SSBA 0.28 -> 0.54 | Sec. 4.4 | paper |
| TAC Top-K | K = 2 | Fig. 2 caption | paper |

## Evaluation protocol

- **Split scored**: the poisoned **training** set `D_tr`. Detection is per training sample; there
  is no held-out detection split. `paper`, Sec. 3.1.
- **Checkpoint**: the single model trained in Stage 1; no statement about which epoch/checkpoint
  is used. `NOT STATED`.
- **Seeds / aggregation**: `NOT STATED`. Tables report one number per (attack, detector, dataset)
  cell. The "Average" row is the mean *delta* (SAM minus base) across the attacks in the column.
- **Metrics reported**: TPR (%), FPR (%), F1 (%). No AUROC in the main tables; AUC appears only in
  the correlation study (Fig. 2) and the intro t-SNE claim. Sec. 4.1 "Evaluation metrics".
- **Metric definitions**: not written out. TPR = fraction of poisoned training samples flagged;
  FPR = fraction of clean training samples flagged; F1 = harmonic mean of precision and recall
  (the sentence defining F1 is commented out in `sec/4_experiment.tex` line 17). Because base rate
  is 5%, F1 is heavily penalized by FPR; several cells show TPR near 100 with F1 near 50 for this
  reason (e.g. AC + SAM on BadNets: TPR 95.4, FPR 13.3, F1 42.5).
- **Threshold**: no global threshold. Each off-the-shelf detector supplies its own decision rule
  (AC K-means split, SS/Spectre robust-statistic cut, SCAn hypothesis test, Beatrix's own
  threshold). Several detectors are visibly run at a **fixed removal budget**: Beatrix rows are
  pinned at FPR = 5.0 across the whole vanilla column, and STRIP rows sit near 10-17% FPR. So
  "successful detection" = flagged by the base detector's own rule, applied to SAM features.
  `paper`; the pinning is inferred from the tables, not stated.
- **ASR convention**: not applicable, ASR is barely used. The only ASR number is "the average
  attack success rate is only 31.8%" at poisoning ratio 0.1% (Sec. 4.2). Whether target-class
  samples are excluded from the ASR set is `NOT STATED`.
- **Correlation study protocol** (Sec. 3.2): CIFAR-10 + ResNet18, poisoning ratios
  {0.5%, 1%, 5%}, Top-2 TAC vs detector AUC. Pearson r = 0.71, R² = 0.51.

## Reported results

Datasets/architectures actually shown in the compiled paper: **CIFAR-10 + ResNet18** (Table 1),
plus one more table whose caption and label disagree (see divergences). Tiny ImageNet, VGG19-BN
and DenseNet-161 are deferred to a supplementary that does not exist in this source or on arXiv.
**No Vision Transformer anywhere.** `paper`, Sec. 4.1.

Attacks (10): BadNets-A2O, BadNets-A2A, Blended, LC, LF, SSBA, TaCT, Adap-Blend, TrojanNN, WaNet.

### Table 1 (`tables/cifar10.tex`), CIFAR-10 / ResNet18 / 5% poisoning, TPR% base / +SAM

| Attack | Spectre | SCAn | SS | AC | Beatrix |
|---|---|---|---|---|---|
| BadNets | 51.1 / 88.4 | 96.0 / 95.2 | 70.8 / 92.3 | 96.8 / 95.4 | 56.6 / 98.8 |
| Blended | 29.9 / 59.7 | 99.2 / 98.7 | 32.9 / 94.6 | 2.3 / 98.8 | 5.0 / 99.8 |
| SSBA | 36.6 / 72.8 | 93.9 / 96.5 | 80.4 / 89.9 | 99.3 / 96.5 | 16.8 / 98.9 |
| LF | 32.0 / 54.1 | 94.1 / 96.1 | 68.2 / 86.5 | 95.6 / 96.1 | 2.4 / 98.8 |
| Adap-Blend | 24.1 / 65.9 | 92.5 / 97.3 | 20.2 / 91.0 | 1.5 / 97.1 | 6.2 / 99.9 |
| LC | 17.0 / 41.7 | 100.0 / 99.9 | 40.5 / 47.8 | 0.0 / 100.0 | 2.2 / 99.9 |
| TaCT | 36.1 / 78.6 | 100.0 / 100.0 | 42.3 / 46.4 | 100.0 / 100.0 | 13.4 / 100.0 |
| TrojanNN | 30.2 / 62.4 | 100.0 / 100.0 | 63.4 / 97.2 | 99.9 / 100.0 | 4.6 / 100.0 |
| WaNet | 66.4 / 97.7 | 66.3 / 90.1 | 71.1 / 86.0 | 85.1 / 90.1 | 1.2 / 95.5 |
| BadNets-A2A | 99.5 / 99.6 | 0.0 / 0.0 | 99.4 / 99.4 | 97.8 / 96.1 | 27.3 / 99.2 |
| **Mean ΔTPR** | **+29.8** | **+3.2** | **+24.2** | **+29.2** | **+85.5** |
| Mean ΔFPR | -1.8 | -0.0 | -1.3 | **+4.0 (worse)** | -2.3 |
| Mean ΔF1 | +25.8 | +1.9 | +22.8 | +8.0 | +69.5 |

Notable per-cell **regressions** with SAM: SCAn/BadNets 96.0 -> 95.2, SCAn/Blended 99.2 -> 98.7,
SCAn/LC 100.0 -> 99.9, AC/BadNets 96.8 -> 95.4 with FPR 0.1 -> 13.3 (F1 97.1 -> 42.5),
AC/SSBA 99.3 -> 96.5 with FPR 3.3 -> 16.2 (F1 76.1 -> 38.3), AC/BadNets-A2A 97.8 -> 96.1.
The abstract's headline "+34.38% TPR on average" does not match any single table average here.

### Table 2 (`tables/gtsrb.tex`, label `tab:gtsrb`, caption says "Tiny"), TPR% base / +SAM

Spectre and SS are only run on BadNets-A2A ("poisoned samples closely match target clean samples,
which breaks the assumptions of Spectre and SS", Sec. 4.2). SCAn cannot handle BadNets-A2A
(needs a single target label).

| Attack | SCAn | AC | Beatrix |
|---|---|---|---|
| BadNets | 97.8 / 91.2 | 96.4 / 91.0 | 25.8 / 99.8 |
| Blended | 88.9 / 99.6 | 0.0 / 99.7 | 46.9 / 100.0 |
| SSBA | 100.0 / 97.3 | 91.0 / 97.4 | 35.7 / 100.0 |
| LF | 91.2 / 85.8 | 0.0 / 87.9 | 32.5 / 99.5 |
| Adap-Blend | 99.8 / 97.6 | 88.3 / 96.8 | 97.9 / 99.9 |
| TrojanNN | 99.9 / 99.9 | 98.3 / 99.9 | 36.8 / 100.0 |
| WaNet | 0.0 / 71.1 | 0.0 / 72.6 | 6.3 / 86.5 |
| BadNets-A2A | 0.0 / 0.0 | 92.1 / 94.2 | 73.7 / 100.0 |
| **Mean ΔTPR** | **+8.1** | **+34.1** | **+53.8** |
| Mean ΔFPR | +1.7 (worse) | -0.2 | -0.0 |

### Ablation (`tables/abla.tex`), CIFAR-10 / ResNet18, TPR / FPR

| Attack | SAM | FS | SS TPR/FPR | Beatrix TPR/FPR |
|---|---|---|---|---|
| BadNets | no | no | 70.8 / 2.4 | 56.6 / 5.0 |
| BadNets | yes | no | 86.0 / 0.6 | 98.4 / 5.0 |
| BadNets | no | yes | 72.8 / 1.2 | 67.0 / 5.0 |
| BadNets | yes | yes | 92.3 / 1.2 | 98.8 / 0.5 |
| Blended | no | no | 32.9 / 4.4 | 5.0 / 5.0 |
| Blended | yes | no | 90.6 / 0.3 | 79.8 / 5.0 |
| Blended | no | yes | 60.4 / 1.9 | 27.1 / 5.0 |
| Blended | yes | yes | 94.6 / 1.1 | 99.8 / 1.5 |

SAM alone carries most of the gain; feature scaling alone gives a smaller but real gain.

### rho sensitivity (Fig. 10, `figs/fig10_rho.png`)

5 attacks (BadNets, Blended, SSBA, LF, TrojanNN), 4 detectors (SCAn, SS, AC, Beatrix), CIFAR-10 /
ResNet18. Read off the plot (no table is given):

- SCAn: flat, TPR 93-100 across the whole range including `rho = 0`.
- SS: TPR ~63-80 at `rho = 0`, jumps to ~85-97 by `rho = 0.05-0.10`, then slowly decays to ~85-92
  at `rho = 0.5`.
- AC: TPR ~0-10 at `rho = 0`, near 95-100 for every `rho >= 0.15`. One dip to ~10-17 at
  `rho = 0.10` for four of five attacks (unexplained).
- Beatrix: TPR 5-57 at `rho = 0`, reaches ~100 by `rho = 0.10-0.15` and stays there.
- FPR stays flat and low for all four across the sweep.
- Text conclusion (Sec. 4.3): "rho proves to be relatively insensitive; a broad range of values can
  be selected with good detection performance." Best region from the plot is roughly
  `rho ∈ [0.10, 0.30]`.

### Poisoning-ratio sweep (Fig. 11, `figs/fig11_poisonratio.png`)

Average TPR over all attacks, CIFAR-10 / ResNet18, base vs +SAM (read off the plot):

| ratio | SCAn base/SAM | SS base/SAM | AC base/SAM | Beatrix base/SAM |
|---|---|---|---|---|
| 0.1% | ~10 / ~11 | ~6 / ~34 | ~10 / ~30 | ~26 / ~52 |
| 0.5% | ~38 / ~70 | ~55 / ~90 | ~31 / ~87 | ~2 / ~88 |
| 1% | ~76 / ~84 | ~40 / ~88 | ~2 / ~93 | ~19 / ~97 |
| 5% | ~85 / ~88 | ~59 / ~83 | ~67 / ~97 | ~14 / ~99 |

At 0.1% poisoning the method still fails (avg TPR below 60%); the paper attributes this to
ASR 31.8%, i.e. "a complete backdoor cannot form".

### Orphan tables (`tables/cifar0.001.tex`, `cifar0.005.tex`, `cifar0.01.tex`) — NOT compiled

These three files are not `\input` by `main.tex` (they are commented out at `main.tex:100-102`).
They add two detectors absent from the compiled paper: **CD (Cognitive Distillation)** and
**STRIP**. Mean ΔTPR from these files:

| file | Spectre | SCAn | SS | AC | Beatrix | CD | STRIP |
|---|---|---|---|---|---|---|---|
| `cifar0.001.tex` | +4.3 | +7.0 | +48.4 | +89.4 | +73.9 | +11.2 | **-3.4** |
| `cifar0.01.tex` | +5.4 | +7.0 | +48.4 | +89.4 | +78.2 | +11.2 | **-3.4** |
| `cifar0.005.tex` | +12.0 | +12.5 | +36.0 | +55.9 | +86.1 | +11.8 | +1.4 |

STRIP is the one **prediction/output-space** detector in the whole paper and it is the only
detector whose average TPR goes **down** or barely moves under SAM, with FPR in the 10-17% band.
These tables were dropped from the compiled version. Treat their numbers as unverified drafts
(their captions are stale, see divergences), but the sign of the STRIP result is consistent
across all three files.

## Paper versus code divergences

No reference implementation is released, so there is no code side. Internal inconsistencies in
the paper source instead:

1. `tables/gtsrb.tex:4` caption says "on **Tiny** and ResNet18" while the label is `tab:gtsrb`
   and Sec. 4.2 discusses it as **GTSRB** (LC is absent from it, consistent with GTSRB/Tiny, and
   Sec. 4.1 says LC is CIFAR-10 only). Dataset identity of Table 2 is ambiguous. `paper`
2. `tables/cifar0.001.tex`, `cifar0.005.tex`, `cifar0.01.tex` all carry the identical caption
   "eta = 0.09, i.e. 500 poisoned samples per class" and the identical label `tab:cifar10`,
   although the filenames imply poisoning ratios 0.1%, 0.5%, 1%. `cifar0.001.tex` and
   `cifar0.01.tex` are near-duplicates differing in only three cells (LC/Spectre 20.8 vs 26.4,
   TaCT/Spectre 18.5 vs 22.6, TrojanNN/Beatrix 57.4 vs 100.0). Stale copies. `paper`
3. Abstract claims "+34.38% TPR on average"; a commented-out earlier abstract claims "+25.54%".
   Neither number equals any table's average (Table 1 column means: +29.8, +3.2, +24.2, +29.2,
   +85.5). The aggregation producing 34.38 is not defined anywhere. `paper`, `sec/0_abstract.tex:13,16`
4. Sec. 4.4 says "As shown in Fig. 13 (assuming the correct figure reference)" — the authors left
   an unresolved cross-reference note in the text (`sec/4_experiment.tex:106`). The weight-norm
   claim spans `fig:weight_norm` and `fig:weightdiff` and it is unclear which supports which
   sentence. `paper`
5. Every "detailed in the supplementary material" pointer (method algorithm box, Tiny ImageNet,
   VGG19-BN, DenseNet-161, 0.1% detailed results, t-SNE) is dangling: `sec/X_suppl.tex` is the
   unmodified CVPR template stub, and `main.tex:98` has the `\input{sec/X_suppl}` commented out.
   The arXiv v1 HTML likewise has no appendix. So the full algorithm, the extra architectures,
   and the extra dataset **do not exist in any public version**. `paper`
6. Sec. 4.1 lists BadNets "in its class-specific (BadNets-A2O) and universal forms (BadNets-A2A)".
   The A2O/A2A naming is inverted relative to normal usage (A2O = all-to-one is the universal
   trigger; A2A = all-to-all is the label-shifting variant). Terminology slip only. `paper`
7. Sec. 4.2 says "For CIFAR-10, we improved the TPR by over 25% for four detection methods";
   Table 1 shows four columns above +24 (Spectre +29.8, SS +24.2, AC +29.2, Beatrix +85.5), so SS
   at +24.2 is under the stated 25 threshold. Minor overclaim. `paper`

## Not stated

Every item below was searched for across all of `sec/*.tex`, `tables/*.tex`, `main.tex`, and the
arXiv v1 HTML:

- Base optimizer under SAM (SGD is implied by "Vanilla Training (SGD)" but never given as a setting),
  learning rate, LR schedule, epochs, batch size, momentum, weight decay, warmup.
- Augmentation, input resolution, normalization statistics.
- SAM variant details: m-sharpness, adaptive vs standard, whether the perturbation is computed
  per micro-batch, whether BatchNorm statistics are frozen in the first pass.
- Number of seeds, variance/error bars, whether tables are single runs.
- PCA rank of the projection matrix `P`, covariance shrinkage/regularization, and the exact rule
  for "potential clean samples dynamically collected from the poisoned dataset" in feature scaling.
- Which layer's features `φ` returns for each architecture (only the neuron-level TAC/weight-norm
  analysis names a layer: the last conv layer of ResNet18).
- The decision threshold for each detector; whether Beatrix/STRIP are run at a fixed FPR budget.
- Formal definitions of TPR, FPR, F1 and the population they are computed over.
- ASR definition and whether target-class samples are excluded from the ASR denominator.
- Any Vision Transformer or attention-based architecture: **absent entirely**. Only ResNet18
  (shown), VGG19-BN and DenseNet-161 (claimed, deferred to a nonexistent supplementary).
- Any layer-depth or layer-wise analysis: **absent**. All neuron-level evidence is the single
  final feature layer.
- Any prediction-space / uncertainty-based detector in the compiled paper: **absent** (STRIP and
  CD appear only in the three uncompiled orphan tables). PSBD is not cited in `main.bib` and not
  evaluated.
- Clean accuracy / ASR of the SAM-trained models versus vanilla (the cost side of the tradeoff)
  except the single 31.8% ASR figure at 0.1% poisoning.
