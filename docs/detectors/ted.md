# TED, topological evolution dynamics

TED stores a bank of clean samples with their activations at every considered layer and, for each input, ranks the bank by distance to the input at each layer and records where the first bank row of the input's predicted class sits. The sequence of ranks over depth is the feature, and an outlier model fitted on the bank's own sequences scores it. The method is white-box and costs 1 forward per input plus a distance matrix per layer. This page records what the paper defines, what the released notebook does, what the port under `detectors/ted.py` runs on ViT and Swin and where the 2 diverge.

## Citation

Mo et al., "Robust Backdoor Detection for Deep Learning via Topological Evolution Dynamics", IEEE S&P 2024, arXiv:2312.02673 (v1 read). The statistic is Section V-B and Algorithm 2, which carries no numbered equation, so lines of the algorithm are cited. The default layer set is stated at the head of Section VI, the layer-type ablation is Section VI-D with Table VIII, the transformer hooks for BERT are in Section VI-E, the reject parameter is in Appendix A and the Z-score alternative in Appendix D.

The released code is https://github.com/tedbackdoordefense/ted, read at commit `fa193a6` as vendored under `third_party/ted`, where the method lives in the single notebook `TED.ipynb`. The cells the port reads are 2 and 4 (defense set size and membership), 9 (hooks), 10 (activation fetch), 14 (rank extraction) and 22 (the outlier detector). The notebook imports `pyod.models.pca.PCA`, which is not installed here, and the port replaces it as recorded below.

## Threat model and data requirement

Section III-B gives the defender white-box access to the deployed model, a small set of clean samples with labels and no knowledge of the attack, its trigger or its target. The adversary of Section III-A controls training and mounts source-specific dynamic triggers, the strongest setting in the paper. TED decides per input at inference time, which is the role every detector in the registry plays.

The clean data enters twice. Labels select the bank, since cell 4 keeps only the samples the model classifies correctly and the paper stores $m$ samples per class. The rank itself reads predicted labels only, both for the bank rows and for the query. The port hands TED the shared 2000-sample clean validation split with its labels, so `DATA_REQUIREMENT` records `the clean validation split, labelled`. The paper's own budget is 20 per class on CIFAR-10 and MNIST, 1000 in total on GTSRB and 200 per label on PubFig and ImageNet-100, and the notebook caps at `DEFENSE_TRAIN_SIZE`, 1000 for CIFAR-10, GTSRB and MNIST and 100 per class otherwise.

## Mechanism

Algorithm 2 follows in the paper's own notation, as a pseudocode block since the paper gives no equation. The symbol table underneath defines every symbol, and the descriptive form after it renames without rederiving.

```
Given a c-class model f with N considered layers, m samples per class,
a metric d, a PCA model PCA(., alpha) with reject parameter alpha and
a test set X_test.

 2   S_1 = S_2 = ... = S_c = {}
 3   for i = 1 to c:
 4       for j = 1 to m:
 5           x = random_sample(X_i)
 6           stack x in S_i
 7           [h_l(x)]_{l=1}^{N} = forward x through f
 8   for i = 1 to |S|:
 9       j = argmax_{k in [1, c]} f(x_i)_k
10       for l = 1 to N:
11           S_sorted = sort_by_distance(d, h_l(.), S, x_i)
12           x_nn     = get_nearest_neighbor(d, h_l(.), S_j - x_i, x_i)
13           K_l^(i)  = get_rank(S_sorted, x_nn)
14       record [K_l^(i)]_{l=1}^{N}
15   M   = PCA({[K_l^(i)]_{l=1}^{N}}_{i=1}^{|S|}, alpha)
16   tau = M.get_detect_threshold(alpha)
17   X_malicious = {}
18   for x in X_test:
19       if M(x) > tau:
20           add x to X_malicious
21   return X_malicious
```

| Symbol | Meaning |
|---|---|
| $f$ | the classifier, $f(x)_k$ its output for class $k$ |
| $c$ | the number of classes |
| $N$ | the number of considered layers |
| $m$ | the stored clean samples per class |
| $X_i$ | the clean samples of class $i$ |
| $S_i$ | the stored bank rows of class $i$, $S$ their union |
| $h_l(x)$ | the representation of $x$ at layer $l$ |
| $d$ | the metric on representations, euclidean throughout the paper |
| $S_j - x_i$ | the bank rows of class $j$ without $x_i$ itself |
| $K_l^{(i)}$ | the rank of the nearest class-$j$ row in the bank sorted by distance to $x_i$ at layer $l$ |
| $\mathbf{M}$ | the PCA outlier model, $\mathbf{M}(x)$ its score |
| $\alpha$ | the reject rate that fixes the threshold $\tau$ |

The same procedure follows with descriptive names in place of the paper's symbols. The structure is unchanged and only the names differ.

```
bank            = clean samples the model classifies correctly, with their
                  features at every considered layer and their predicted labels
for each query:
    predicted   = the label the model gives the query
    for each layer:
        order   = bank sorted by euclidean distance to the query's features
                  at this layer, the query's own row left out if it has 1
        rank    = position in order of the first row whose predicted label
                  equals predicted
    trajectory  = rank over every layer
outlier_model   = fitted on the bank's own trajectories
ted_score       = outlier_model(trajectory), flagged when above a threshold
```

A clean input sits among clean inputs of its predicted class at every depth, so the first same-class row is near the front of the sorted bank at every layer and the ranks stay small and consistent. A poisoned input reaches the target class only through its trigger, so in the early and middle layers its nearest neighbours belong to its source class and the first target-class row sits far back, and the ranks fall only near the end. Section V-A states the rationale in those 2 halves, the deep activations of a triggered input resemble the target class and its shallow activations resemble its source class. The rank at a layer is invariant to any positive rescaling of that layer's activations, so the scale drift along a residual stream that `experiments/prediction_depth` guards against with a logit lens never enters this statistic.

The outlier model is where the port departs from the notebook. Cell 22 fits pyod's `PCA` detector with standardisation on, `weighted=True` and `n_components='mle'`, and pyod's `decision_function`, read at the current master of the pyod repository, computes

$$
\text{score}(x) = \sum_{j=1}^{k} \frac{\lVert z(x) - v_j \rVert_2}{w_j}, \qquad z(x) = \frac{x - \mu}{\sigma}
$$

| Symbol | Meaning |
|---|---|
| $z(x)$ | the trajectory standardised by the fitted `StandardScaler` |
| $\mu, \sigma$ | per-layer mean and standard deviation of the clean trajectories |
| $v_j$ | row $j$ of `components_`, a unit principal axis read as a point in trajectory space |
| $w_j$ | `explained_variance_ratio_[j]`, the weight under `weighted=True` |
| $k$ | `n_components_`, every component when none is deselected |

That quantity sums the distances from a standardised point to each eigenvector treated as a point, weighted by the inverse of its variance ratio, so it is a full-rank variance-weighted distance from the clean centre rather than a reconstruction error onto minor components. The port keeps the standardisation and replaces the sum by the squared Mahalanobis distance under the clean covariance, with a floor on the diagonal so a layer whose clean ranks never vary cannot make the covariance singular.

$$
M(x) = z(x)^{\top} \left( \Sigma + \epsilon I \right)^{+} z(x), \qquad \Sigma = \operatorname{cov}\!\left( z \right)
$$

| Symbol | Meaning |
|---|---|
| $z(x)$ | the standardised trajectory, with $\sigma$ replaced by 1 on a layer whose spread is under `STANDARD_DEVIATION_FLOOR` |
| $\Sigma$ | the covariance of the standardised clean trajectories |
| $\epsilon$ | `COVARIANCE_FLOOR`, $10^{-6}$ |
| $(\cdot)^{+}$ | the pseudo-inverse, taken in float64 |

Both scores grow with the standardised distance from the clean centre, which is the direction line 19 of Algorithm 2 relies on, and nothing more is claimed. The 2 are different functions of the same standardised trajectory, and a number from this port is a Mahalanobis number rather than a pyod number.

## What the released code does

Cell 2 sets `DEFENSE_TRAIN_SIZE` from the dataset and cell 4 builds the bank. It runs the defense loader through the model, keeps the samples whose argmax equals the label, draws `DEFENSE_TRAIN_SIZE` of them at random without replacement when more survive and reloads them with `shuffle=True`. Cell 9 walks `model.modules()` and registers a forward hook on every `Conv2d` whose kernel is not 1 by 1, every `ReLU` and every `Linear`, in module order.

Cell 10, `fetch_activation`, runs the loader twice, once to initialise a container per hook and once to collect. For each batch it stores the argmax prediction, flattens each hooked activation with `.view(batch, -1)` and appends every row to a Python list on the device, then stacks the lists, so the bank is float32 and lives on the device as 1 tensor per hook. Cell 14 defines the rank. `get_dis_sort` takes torchmetrics' `pairwise_euclidean_distance` of 1 item against the whole bank and returns `torch.sort`'s index order. `getDefenseRegion` walks the bank's own rows of 1 predicted class, drops the first sorted entry with `ranking_array[1:]` since that entry is the row itself, reads the predicted labels along the order and records `.index(label)` when the label appears at all. `getLayerRegionDistance` does the same for outside queries without the drop, grouped by the query's predicted label. A query whose predicted label appears on no bank row is skipped in both functions, so it contributes no trajectory and is never scored.

Cell 22 turns the per-layer lists into a `(samples, layers)` array per label, splits benign labels from the 3 temporary labels the notebook assigns to victim-triggered, non-victim-triggered and no-trigger test samples, fits a 2-component sklearn PCA for a scatter plot and then fits `pyod.models.pca.PCA(contamination=0.01, n_components='mle')` on the benign array. It scores the benign array in sample with `decision_function`, scores the unknown array the same way and reports AUC of the victim-triggered label against the rest and a confusion matrix at pyod's contamination threshold. Appendix A of the paper states $\alpha = 5\%$ where the notebook's contamination is 1%.

## What this port does on ViT

1. **Considered layers.** The paper takes all Conv2D outputs by default and adds ReLU and Linear outputs for shallow networks, while its BERT experiment hooks the dense, self-attention and embedding layers. The notebook hooks every non-pointwise Conv2d, every ReLU and every Linear. A ViT block has no Conv2d and no ReLU, and its 4 Linear layers per block include 2 inside attention whose outputs are head-split projections rather than a representation of the input, so that rule has no direct reading. The port reads the residual stream at every block boundary through `analysis.features.captured_layers`, 13 hook points on ViT-B/16 (the input of block 0 and every block output) and 25 on Swin-S, which is every place the representation is rewritten and the transformer analogue of a convolutional block's output. Table VIII of the paper reports similar AUC across layer sets on a deep network, so the count of 13 rather than a Conv2D count is not expected to move the number, but no ViT number in the paper exists to check against.
2. **Outlier model.** The paper fits a PCA outlier model with reject rate $\alpha$ and the notebook uses pyod's `PCA` detector, whose score is the weighted eigenvector-distance sum quoted above. pyod is not installed and is not added as a dependency. The port standardises the trajectories per layer as pyod does and scores the squared Mahalanobis distance under the floored clean covariance in float64. The direction is preserved, since a trajectory far from the clean centre scores high under both. No equivalence is claimed. The fit is exact under a singular covariance, which the test file pins on a constant layer.
3. **Bank membership and size.** The paper stores $m$ samples per class and the notebook keeps every correctly classified sample of its defense loader up to a random cap of 1000. The port keeps every correctly classified sample of the shared 2000-sample split with no cap, about 1900 on a ViT checkpoint, so every detector sees the same budget and no seed-dependent draw enters. Per class that is about 190 on CIFAR-10, 19 on CIFAR-100 and 10 on Tiny ImageNet, with an uneven spread on GTSRB. Ranks therefore run to about 1900 rather than 1000 and the paper's box plots are not comparable in magnitude.
4. **Absent predicted class.** The notebook drops a query whose predicted label appears on no bank row, from the bank's own rows in `getDefenseRegion` and from test queries in `getLayerRegionDistance`. The port gives such a query the rank equal to the bank size at every layer, the position just past the sorted bank, and scores it, since a detector that returns no score for an input has made no decision on it. The result is an extreme outlier score whether or not the input carries a trigger. `classes_without_reference(bank, num_classes)` lists the classes this rule fires on, and a run should record it beside its scores, since Tiny ImageNet will hit it.
5. **Precision and storage.** The notebook holds the bank in float32 on the device and computes each query's distances against the whole bank at once. The port rounds every reduced feature to float16, the bank rows and the queries alike, keeps the bank on the device and takes distances in float32 through `torch.cdist` over chunks of `DISTANCE_CHUNK` bank rows, so the largest transients are the 2 chunks cast to float32 and the `(batch, bank_size)` distance matrix. Rounding the queries as well as the bank makes a bank member scored through its own loader identical to its stored row, given a reproducible forward, so its self-exclusion removes exactly its own distance and the loader path reproduces `leave_one_out_trajectories` to the integer. float16 carries about 3 decimal digits, so a near tie between 2 bank rows at the 4th digit can flip a rank, and `cdist` uses the same matmul expansion of the squared distance that torchmetrics uses in the notebook.
6. **Leave-one-out validation scores.** Line 12 of Algorithm 2 takes the nearest neighbour from $S_j - x_i$ and the notebook realises it with `ranking_array[1:]`, so the bank's own trajectories never count the row itself. The notebook then scores the benign array in sample with `decision_function`. The port fits the trajectory model on the bank's leave-one-out trajectories and scores the validation split through a second pass over its loader with each bank member's own row masked, so the validation scores the threshold is read from are leave-one-out at the rank level for the bank members and ordinary for the misclassified rest. The trajectory model's mean and covariance are still fitted on every bank trajectory including the row being scored, where each row's own weight in that fit is about 13 in 1900, which is recorded rather than removed.
7. **Token reduction.** The notebook flattens every activation with `.view(batch, -1)`, so the port's default `flatten` does the same, 197 tokens by 768 channels on ViT-B/16. `DetectorContext.ted_reduction` overrides it to `cls`, the class token alone, or `mean`, the token average, and `cls` raises on Swin as `analysis.features` does since Swin has no class token. Early class tokens carry little content, since at layer 0 the class token is the same learned embedding plus its position for every input and in the first blocks it has only begun to aggregate the patches, so under `cls` the first ranks are noise. Flatten costs 7.5 GB on ViT-B/16 and about 10 GB on Swin-S for the bank, where `cls` and `mean` cost 38 MB. The synthetic fixture prefers `cls` because its head is linear on the class token at the last layer while its patch tokens are random-init noise.
8. **Threshold rule.** The paper thresholds at the reject rate $\alpha$ and Appendix D offers a 4-sigma Z-score alternative. The port hands the negated outlier score to `defences.decision.detection_report`, which sets the threshold at a quantile of the shared clean validation split, so TED is judged at the same false-positive budget as every other detector. AUROC is unaffected. TPR at a fixed quantile is a different operating point from Table III.

## Hyperparameters

| Symbol | Paper default | This port | Constant name |
|---|---|---|---|
| $N$, considered layers | all Conv2D outputs, Conv2D plus ReLU plus Linear on shallow networks | 13 on ViT-B/16, 25 on Swin-S, every `captured_layers` hook point | none, from `transformer_blocks` |
| $m$, stored samples per class | 20 on CIFAR-10 and MNIST, 1000 in total on GTSRB, 200 on PubFig and ImageNet-100, notebook cap 1000 | every correctly classified sample of the 2000 split, about 1900 in total | `PAPER_DEFENSE_SET_SIZE`, a record only |
| $d$, the metric | euclidean | euclidean in float32 | none |
| token reduction | none stated, the notebook flattens | `flatten`, overridable to `cls` or `mean` | `DEFAULT_REDUCTION` |
| $\alpha$, reject rate | 5%, notebook contamination 1% | none, the quantile rule replaces it | none |
| outlier model | pyod PCA, `n_components='mle'`, standardised, weighted | squared Mahalanobis on standardised trajectories | `COVARIANCE_FLOOR`, `STANDARD_DEVIATION_FLOOR` |
| bank dtype | float32 | float16 | `BANK_DTYPE` |
| distance chunk | the whole bank per query | 256 rows per `cdist` call | `DISTANCE_CHUNK` |
| absent-class rank | the sample is dropped | the bank size | none |

## Cost

Each input costs 1 forward plus 1 distance matrix against the bank at each of the 13 layers, 25 on Swin-S. The fixed cost is 2 passes over the validation split, 1 to collect the bank and 1 to score the split with each bank member's own row excluded, plus the bank's leave-one-out ranks, which are 13 bank-against-bank distance matrices.

At the measured 2130 images per second for a bfloat16 ViT-B/16 forward on the A100, a GTSRB checkpoint's 23208 scored inputs plus 4000 validation forwards take about 13 seconds. A batch of 256 flattened queries against a bank of 1900 costs about 1.9 TFLOP of float32 matmul over the 13 layers, about 0.1 seconds, so the 106 batches add about 10 seconds, and the estimate is about 1 minute per checkpoint. The bank is 13 by 1900 by 151296 halves, 7.5 GB, on ViT-B/16 under `flatten` and about 10 GB on Swin-S, whose 25 hook points sum to 2634240 features per sample, and it stays resident until the detector callable is dropped. Under `cls` or `mean` the bank is 38 MB. The smoke run replaces the estimate with a measured seconds-per-input figure for `pbs/generate_detector_jobs.py`.

## How to run

The smoke runs 1 checkpoint folder at 500 inputs per split and writes outside the results tree. `--allow-missing-psbd-cache` is required there because the smoke tree holds no PSBD split manifest to check against.

```bash
python -m cli.baselines --checkpoint-folder <folder> --detectors ted --max-samples 500 \
    --results-dir scratch/detector_smoke/results --allow-missing-psbd-cache
```

The panel runs through the `cheap` job group of `pbs/generate_detector_jobs.py`, which batches TED with the other detectors that finish in about a minute per checkpoint. A job in that group needs the bank's memory on top of the model, which fits an A100 with room to spare.

## Where results land

`results/<folder>/detectors/ted_metrics.json` holds the detection report at every quantile plus the provenance record. The raw per-sample scores sit beside it as `ted_scores_validation.pt`, `ted_scores_clean.pt` and `ted_scores_backdoor.pt`, 1 float32 tensor of the split's length each, in the loader's order, so any later threshold or fusion reads them without a rerun.

## Results

<!-- results:begin -->
<!-- results:end -->

## Known failure modes

A low score says the input's rank trajectory is far from the clean cloud in the standardised Mahalanobis metric. That is what a triggered input produces when its early-layer neighbours belong to another class, and it is also what an input of a rare predicted class produces, since a class with 1 or 2 bank rows gives coarse ranks and a class with none gives the bank size at every layer. On Tiny ImageNet the shared split holds about 10 samples per class before the misclassified ones are removed, so some classes fall out of the bank entirely and every clean input the model sends to such a class is flagged. `classes_without_reference` counts them and the count belongs beside any Tiny number.

Clean-label attacks are an expected failure by the mechanism itself. Under SIG or LC the triggered input's source class is the target class, so its early-layer neighbours already carry the predicted label and its trajectory has no source-class prefix to stand out with. The paper's evaluation never includes a clean-label attack and this reading is an inference from Section V-A, to be checked on the panel rather than assumed.

The bank's memory is the operational risk. Under `flatten` it is 7.5 GB on ViT-B/16 and about 10 GB on Swin-S for the 2000 split, resident on the device from the fit until the detector callable is released, and a larger validation split or a wider model scales it linearly. `cls` and `mean` reduce it 4000-fold at the cost of reading 1 vector per layer instead of every token, and which reduction the panel should run on is an open question the smoke is meant to settle.

Adaptive attacks against TED exist. Section VI-C of the paper finds 3 adaptive losses ineffective and label substitution effective only above 20% of the target class, and later work under the name TED-LaST (arXiv:2506.10722, read only through the survey notes) reports adaptive attacks that defeat it. The synthetic fixture is a limited judge here. Under its default seed the random-weight model never predicts the fixture's target class on clean noise, so the bank holds no target-class row and every triggered input is scored by the absent-class rule alone, which gives a high AUROC that tests deviation 4 rather than the rank dynamics. The test file reads a second seed whose bank holds target-class rows and prints both numbers, which at the time of writing are AUROC 0.805 under `cls` and 0.566 under `flatten` on the second seed against 0.999 under the absent-class rule on the default seed, so the fixture verdict is read from `cls`.

## Direction

High is poisoned in the paper and the score is negated once. Line 19 of Algorithm 2 flags an input whose outlier score exceeds $\tau$, and pyod assigns larger scores to outliers, so the paper's statistic is high for a triggered input. `outlier_scores` returns that statistic unnegated, so it can be read against the paper's box plots, and `ted_scores` returns its negation, which is the only sign change in the module. `detection_report` treats a low score as positive evidence, so a second negation anywhere would produce a well-formed, exactly inverted detector, which the `direction` field of the report exists to catch.
