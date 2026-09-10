# Beatrix, class-conditional Gram-matrix deviation

Beatrix reads the Gram matrix of a model's intermediate features at 1 layer, raises it to the orders 1 to $P$ and scores an input by how far its Gram entries fall outside a median-and-MAD band fitted on the clean inputs the model assigns to the same class. The method is white-box and costs 1 forward pass per input, which puts it among the cheap detectors in the registry, and it is the baseline TED and TeCo measure themselves against. This page records what the paper defines, what the authors released, what the port under `detectors/beatrix.py` runs on ViT and Swin and where the 3 diverge.

## Citation

Ma et al., "The 'Beatrix' Resurrections: Robust Backdoor Detection via Gram Matrices", NDSS 2023, arXiv:2209.11715 (v3 read). Feature modelling is Section IV-A, Eq. (8) and (9). The deviation measurement is Section IV-B, Eq. (10) to (14), with the threshold determination paragraph at the end of that section. The order bound and the clean budget are fixed in Section V-A under "The order of Gram matrix" and "Clean data for deviation measurement". The offline infected-class test, RMMD in Eq. (15) to (19), is outside the port.

The released code is https://github.com/wanlunsec/Beatrix, read at commit `685827e` as vendored under `third_party/Beatrix`. The method is the class `Feature_Correlations` at `defenses/Beatrix/Beatrix.py:307`, the validation jackknife is `threshold_determine` at line 388 and the driver is `BEAT_detector` at line 409. The BackdoorBench copy at `third_party/BackdoorBench/detection_pretrain/beatrix.py` was read for comparison. Its `G_p` at line 158 omits the line `temp = temp**p` that the authors' file has at line 334, so every order it computes is the first-order Gram under a $1/p$ root rather than Eq. (9), and the port follows the authors' file.

## Threat model and data requirement

The adversary poisons the training data and has no access to the deployed model. The defender is white-box, reads an intermediate feature representation for every input and holds a small clean set, 30 images per class in the paper with 8 shown to be enough. The paper uses the deviation both online, per input at inference, and offline, to name infected classes through a kernel two-sample test. The port covers the online role only, which is the role every detector in the registry plays.

The class-conditional statistics are keyed by the model's predicted label. Section IV-A defines $\mathcal{X}_t$ as clean samples of class $t$ and compares a query against the class its predicted label $\hat{y_t}$ names, and the released driver groups its clean references by the argmax of the model's own prediction at `Beatrix.py:118`, so no ground-truth label enters at any point. The port therefore gives Beatrix the shared 2000-sample clean validation split with its labels ignored, and `DATA_REQUIREMENT` records the split as unlabelled. The paper's budget of 30 per class is a different budget from the shared split's, and deviation 4 below states what the split gives on each dataset.

## Mechanism

Eq. (8) to (14) follow in the paper's own notation. The symbol table underneath defines every symbol, and the descriptive form after it renames without rederiving.

$$
\begin{aligned}
G &= v v^{T} && \text{(8)} \\
G^{p} &= \left( v^{p} \, {v^{p}}^{T} \right)^{1/p} && \text{(9)} \\
s &= \left[ \vec{G^{1}}, \ \vec{G^{2}}, \ \dots, \ \vec{G^{P}} \right] \in \mathbb{R}^{\frac{1}{2} n (n+1) P} \\
\tilde{s}_{j} &= \operatorname{median}\left( \left\{ s_{ij}, \ \forall i \in \{1, 2, \dots, |\mathcal{X}_{t}|\} \right\} \right) && \text{(10)} \\
MAD_{j} &= \operatorname{median}\left( \left\{ \left| s_{ij} - \tilde{s}_{j} \right|, \ \forall i \in \{1, 2, \dots, |\mathcal{X}_{t}|\} \right\} \right) && \text{(11)} \\
\delta_{j} &= \delta(\hat{s}_{j}) && \text{(12)} \\
&= \begin{cases} 0 & \text{if } min \le \hat{s}_{j} \le max \\[4pt] \dfrac{min - \hat{s}_{j}}{min} & \text{if } \hat{s}_{j} \le min \\[4pt] \dfrac{\hat{s}_{j} - max}{max} & \text{if } max \le \hat{s}_{j} \end{cases} && \text{(13)} \\[4pt]
\delta &= \frac{2}{n(n+1)P} \sum_{j=1}^{\frac{1}{2} n (n+1) P} \delta_{j} && \text{(14)}
\end{aligned}
$$

with $min = \tilde{s}_{j} - k \cdot MAD$ and $max = \tilde{s}_{j} + k \cdot MAD$ as the paper writes them under Eq. (13).

| Symbol | Meaning |
|---|---|
| $h$ | the sub-model up to layer $l$ |
| $v = h(x) \in \mathbb{R}^{n \times m}$ | the feature representation of input $x$ at layer $l$, $n$ channels by $m$ spatial positions |
| $G \in \mathbb{R}^{n \times n}$ | the Gram matrix of $v$, channel against channel |
| $v^{p}$ | the elementwise $p$-th power of $v$ |
| $G^{p}$ | the $p$-th order Gram matrix, rooted back by $1/p$ |
| $P$ | the order bound, 4 in the paper's main experiments |
| $\vec{G^{p}}$ | the upper triangle of $G^{p}$ with its diagonal, as a vector of $\frac{1}{2} n (n+1)$ entries |
| $s$ | the concatenated Gramian feature vector of an input |
| $\mathcal{X}_{t}$ | the clean samples of class $t$ |
| $s_{ij}$ | entry $j$ of the feature vector of clean sample $i$ |
| $\tilde{s}_{j}$ | the median of entry $j$ over the clean samples of the class |
| $MAD_{j}$ | the median absolute deviation of entry $j$ over the same samples |
| $\hat{s}$, $\hat{s}_{j}$ | the feature vector of the query and its entry $j$ |
| $k$ | the band scale factor, 10 |
| $min$, $max$ | the band edges of entry $j$, written without a subscript in the paper |
| $\delta_{j}$ | the relative excess of entry $j$ outside its band |
| $\delta$ | the deviation of the query, the mean of $\delta_{j}$ over every entry |

The same lines follow with descriptive names in place of the paper's symbols and with the token matrix in the place of the channel-by-position feature map. The structure is unchanged and only the names differ.

$$
\begin{aligned}
\text{gram}_{p} &= \operatorname{sign}\!\left( (\text{tokens}^{p})^{T} \, \text{tokens}^{p} \right) \left| (\text{tokens}^{p})^{T} \, \text{tokens}^{p} \right|^{1/p} \\
\text{features} &= \left[ \operatorname{triu}(\text{gram}_{1}), \ \dots, \ \operatorname{triu}(\text{gram}_{P}) \right] \\
\text{lower}_{j} &= \text{median}_{j} - \text{band\_width} \cdot \text{mad}_{j} \\
\text{upper}_{j} &= \text{median}_{j} + \text{band\_width} \cdot \text{mad}_{j} \\
\text{deviation} &= \frac{1}{\text{num\_features}} \sum_{j} \left( \frac{\operatorname{relu}(\text{lower}_{j} - \text{features}_{j})}{|\text{lower}_{j}|} + \frac{\operatorname{relu}(\text{features}_{j} - \text{upper}_{j})}{|\text{upper}_{j}|} \right)
\end{aligned}
$$

A Gram entry is the inner product of 2 feature dimensions across every position, so it reads a second moment of the representation, and the higher orders weight the large activations more heavily. A trigger drives the prediction to the target class through activations the clean members of that class never show, so the query's entries fall outside the band the class's clean references span, and the deviation grows with how many entries fall outside and by how much. A clean input of the class lands inside the band on nearly every entry and its deviation stays at or near 0. The band is a median and a MAD rather than a mean and a standard deviation because the defender has 30 samples per class, and the paper's Section IV-B argues that a Gaussian fit on that few is moved by any outlier.

## What the released code does

The driver hooks the input of `layer4` of a PreActResNet-18 through `LayerActivations` at `Beatrix.py:57`, runs the clean and the poisoned test set through the model once each and stores the feature maps together with the argmax of the clean prediction as `ori_label` at line 118. `BEAT_detector._detecting` shuffles the 4 arrays, then for every class takes the clean features whose predicted label is that class, keeps the first `clean_data_perclass = 30` as the defender's set and calls `threshold_determine` on them. That function cuts the 30 into 5 contiguous blocks of `len // 5` by position, fits on the other 4 blocks, scores the held-out block and pools the 5 results into the benign deviation distribution whose 95th and 99th percentiles it prints. Any remainder of an uneven cut is never scored. The detector is then refitted on all 30 and applied to the last 500 clean features of the class and, for the target class only, to the poisoned features.

`Feature_Correlations.G_p` at line 332 raises the feature map to the power $p$, reshapes it to `(N, channels, positions)`, multiplies it by its own transpose into `(N, channels, channels)`, zeroes the strict lower triangle with `triu()`, applies `sign(.) * abs(.) ** (1 / p)` and flattens the whole `channels * channels` matrix, zeros included, into the feature vector. It records `num_feature = channels * channels / 2`. The signed root is the released reading of Eq. (9), whose printed $1/p$ root of a possibly negative entry is undefined at odd $p$. `get_median_mad` takes `median(dim=0)` per entry, which returns the lower of the 2 middle values on an even count, and the MAD as the median of the absolute deviations from it. `minmax_mad` sets the band to `median - mads * 10` and `median + mads * 10`. `get_deviations_` at line 358 sums `relu(min - g) / abs(min + 1e-6)` and `relu(g - max) / abs(max + 1e-6)` over every entry and every order and divides by `num_feature * len(power)`. A second method `get_deviations` without the underscore computes a MAD z-score instead and is never called by the driver.

2 settings in the driver differ from the paper's text. `BEAT_detector.__init__` at line 410 sets `order_list = np.arange(1, 9)`, orders 1 to 8, where Section V-A fixes $P = 4$ for every experiment after the order ablation. The normalisation divides by `channels * channels / 2` rather than the $\frac{1}{2} n (n+1)$ of Eq. (14). The zeroed lower triangle contributes nothing to the sum, since a band fitted on constant zeros has `min = max = 0` and `relu(0 - 0) / 1e-6` is 0, so the difference is the constant factor $\frac{n}{n+1}$ on every score.

## What this port does on ViT

1. **Order bound.** The paper starts at $P = 9$, finds the false positive rate stable from $P \ge 4$ and sets $P = 4$ for the remaining experiments. The released code runs orders 1 to 8. The port runs `PAPER_POWERS = (1, 2, 3, 4)` and keeps `OFFICIAL_CODE_POWERS` as a named, unused constant. Beyond the paper's own choice there is a range reason on a ViT: a Gram entry sums 197 products of 2 $p$-th powers, and block outputs on ViT-B/16 carry a few massive-activation dimensions in the hundreds, so a value of 1000 contributes $10^{24}$ at $p = 4$ and $10^{48}$ at $p = 8$ against the $3.4 \times 10^{38}$ float32 holds. `gram_features` raises on any non-finite entry rather than letting an `inf` band silently pass every query. The number is the paper's main-experiment statistic, and a number from orders 1 to 8 is never produced.
2. **Feature site and Gram axis.** The paper takes $v$ at layer $l$ of a ConvNet as $n$ channels by $m$ positions, and the code hooks the input of `layer4`, 3 quarters of the way through the network. The port reads the residual stream through `analysis.features.captured_layers` at index 9 on ViT-B/16, the output of block 9 of 12, and at index 22 on Swin-S, the output of block 22 of 24, which is the last block of its third stage and the last point before the final patch merge. The captured activation is a token matrix of shape (tokens, dim), (197, 768) on ViT and (196, 384) on Swin after `as_token_sequence` flattens the grid, and the Gram contracts over the tokens so it is (dim, dim). That is the faithful analogue of the paper's channel Gram, whose $m$ spatial positions are the contracted axis, and a Gram over the 197 tokens would instead correlate positions, which the paper never does. The class token stays in, as every spatial position stays in on a ConvNet. The feature vector has $\frac{1}{2} \cdot 768 \cdot 769 = 295296$ entries per order and 1181184 at $P = 4$ on ViT, 73920 and 295680 on Swin. `FEATURE_LAYER_BY_ARCHITECTURE` holds the 2 indices and `DetectorContext.beatrix_layer` overrides them, which the synthetic fixture's 2-block model needs.
3. **Predicted-label grouping.** Section IV-A writes $\mathcal{X}_t$ as clean samples in class $t$ and picks $t$ from the query's predicted label. The released code groups the clean references by `ori_label`, the argmax of the clean prediction, so a reference the model misclassifies joins the band of the class it was put in. The port does the same through `collect_reference_tokens`, which returns the predicted label from the same forward pass the tokens come from and never reads the loader's labels. This is the only label-free reading and the only 1 that makes sense at inference, where a query has no true label. On a model with 95% or better clean accuracy the contamination of a band by misclassified references is small, and on a weak model it widens the bands of the classes the model confuses.
4. **Reference budget and the pooled fallback.** The paper budgets 30 clean images per class and the code takes the first 30 of each predicted class. The port gives Beatrix the shared 2000-sample split, which puts about 200 references per class on CIFAR-10, 20 on CIFAR-100, 46 on GTSRB and 10 on Tiny ImageNet, with the random shuffle and the model's own predictions spreading those counts unevenly. A class with fewer than `MIN_CLASS_SAMPLES = 5` predicted references takes the pooled band over every reference, since a median and a MAD of 4 values bound nothing, and `ClassBands.counts` and `pooled_class_count` record how many classes fell back. On Tiny the bands rest on a third of the paper's budget and some classes will pool, so a Tiny number from this port is a data-limited one, and the optional control in the plan, 30 correct images per class from the clean training set, exists to separate the budget from the method there.
5. **Normalisation.** Eq. (14) divides the sum by $\frac{1}{2} n (n+1) P$. The released code divides by $\frac{1}{2} n^{2} P$ after flattening the full zeroed matrix. The port divides by `gram_entry_count(dim) * len(powers)` as Eq. (14) writes it. Every score from the port is the released code's times $\frac{n}{n+1}$, 768/769 on ViT, a constant that moves no ranking, AUROC or quantile position, and the bit-level test folds it in before comparing.
6. **Jackknife for the validation split.** The paper's threshold determination draws $\frac{1}{T}$ of the clean set as test samples, fits on the rest and repeats $T$ times. The released code fixes $T = 5$ and cuts by position into contiguous blocks, dropping the remainder. The port's `jackknife_deviations` cuts the 2000 by position into 5 contiguous blocks with boundaries at $i \cdot 5 / N$ so every reference is scored exactly once, fits `fit_class_bands` on the other 4 blocks with the same pooled fallback and scores the held-out block, and the registry returns those scores whenever the validation loader is scored. A sample deviates less from a band it helped fit, so in-sample scores would set the threshold too tight and the achieved false positive rate on the paired clean split would exceed the budget. The contiguous cut holds out a random fifth because the shared split is a `randperm` served without reshuffling, and a class-sorted loader would hold out whole classes. Each fold fits on 1600 references, so a class with 5 or 6 members can pool inside a fold while holding its own band in the full fit.
7. **Precision and the reference bank.** The paper and the code are float32 end to end. The port runs the forward under the shared autocast policy, bfloat16 on CUDA, and `captured_token_matrix` casts the captured activation to float16, the dtype of the reference bank, on the reference and the query side alike, so a band and the query it judges are rounded the same way. Every Gram is taken in float32 after that cast, since a bfloat16 or float16 matmul over 197 terms accumulates at 8 or 11 bits of mantissa, the reason `analysis.features` casts before reducing, and `gram_features` refuses a half-precision input rather than promoting it. The float16 rounding is 1 part in 2048 per value against band widths of 10 MAD. An activation past float16's range raises at the cast, naming the layer.
8. **Relative epsilon.** Eq. (13) divides by $min$ and $max$. The released code divides by $|min + 10^{-6}|$ and $|max + 10^{-6}|$, which keeps an entry whose band edge sits at exactly 0 from dividing by 0. The port follows the code through `RELATIVE_EPSILON`, and the bit-level test depends on it.
9. **Threshold rule.** The paper flags an input whose deviation exceeds a percentile of the benign deviations, 95% in its example, which is a false positive budget of 5%. The port hands the negated deviations to `defences.decision.detection_report`, whose quantile of the shared validation split is the same rule at the registry's budgets, 0.25 as the headline and 0.01 and 0.05 alongside. AUROC is unaffected, and the paper's TPR at 1% or 5% FPR is read at the matching quantile rather than at the headline.

## Hyperparameters

| Symbol | Paper default | This port | Constant name |
|---|---|---|---|
| $P$, order bound | 4, after an ablation from 9 | 4, orders (1, 2, 3, 4) | `PAPER_POWERS` |
| orders in the released code | 1 to 8 | recorded, unused | `OFFICIAL_CODE_POWERS` |
| $k$, band scale factor | 10 | 10 | `MAD_BAND` |
| denominator guard | none in Eq. (13), $10^{-6}$ in the code | $10^{-6}$ | `RELATIVE_EPSILON` |
| clean images per class | 30 | the shared split, about 10 to 200 by dataset | `PAPER_CLEAN_PER_CLASS`, recorded only |
| minimum references per class | none stated | 5, pooled band below it | `MIN_CLASS_SAMPLES` |
| $T$, jackknife folds | "$T$ iterations", 5 in the code | 5, contiguous by position | `JACKKNIFE_FOLDS` |
| $l$, feature layer | input of `layer4` | block 9 of 12 on ViT, block 22 of 24 on Swin, overridable through `DetectorContext.beatrix_layer` | `FEATURE_LAYER_BY_ARCHITECTURE` |
| Gram batch | whole split at once | 64 token matrices | `GRAM_BATCH_SIZE` |
| band fit chunk | whole split at once | 1 GiB of float32 columns | `FIT_CHUNK_BYTES` |
| detection percentile | 95% or 99% of benign deviations | the registry's quantile rule | none |

## Cost

Each input costs 1 forward pass plus the Gram algebra, 4 products of a (768, 197) by a (197, 768) matrix per input, about 0.9 GFLOP against the 17.6 GFLOP of a ViT-B/16 forward. The fixed cost is 1 pass over the validation split for the bank and the predicted labels, then the Gram of every reference once for the full bands, once per jackknife fold for the validation scores and about 9 more times over the whole bank whenever a pooled band is needed, since the pooled fit recomputes the Gram per column chunk rather than holding the 9.4 GB matrix of 2000 by 1181184 floats. At the measured 2130 images per second for a bfloat16 forward on the A100 the forwards take about 11 s per GTSRB checkpoint and the Gram algebra seconds, so the plan's estimate is about 1 minute per checkpoint. The smoke run replaces this estimate with a measured seconds-per-input figure that goes into `pbs/generate_detector_jobs.py`.

Memory is the constraint that shaped the module. Gram vectors are never stored for a whole split, since 1 sample is 1181184 floats at $P = 4$ and a split of 2000 would be 9.4 GB. `collect_reference_tokens` keeps the token matrices instead, 2000 by 197 by 768 in float16 on the CPU, 605 MB on ViT and 301 MB on Swin. `fit_class_bands` moves 1 class at a time to the device, takes its Gram in batches of `GRAM_BATCH_SIZE`, fits its median and MAD and keeps only the 2 band vectors, 4.72 MB each in float32 and 9.45 MB per class, 1.89 GB on the device for Tiny's 200 classes. Scoring a batch of 64 holds its Gram vectors, the 2 gathered band matrices and 2 temporaries, about 1.5 GB at the peak. A pooled fit over 2000 references holds a column chunk of up to 1 GiB plus a temporary of the same size for the MAD.

## How to run

The smoke runs 1 checkpoint folder at 500 inputs per split and writes outside the results tree. `--allow-missing-psbd-cache` is required there because the smoke tree holds no PSBD split manifest to check against.

```bash
python -m cli.baselines --checkpoint-folder <folder> --detectors beatrix --max-samples 500 \
    --results-dir scratch/detector_smoke/results --allow-missing-psbd-cache
```

The panel runs through the `cheap` job group of `pbs/generate_detector_jobs.py`, which puts Beatrix beside confidence, STRIP, SCALE-UP, IBD-PSC and TED in 1 job per checkpoint with `--skip-existing`, since all of them finish within minutes and a job sized for CD-L or SentiNet would idle the GPU on them.

## Where results land

`results/<folder>/detectors/beatrix_metrics.json` holds the detection report at every quantile plus the provenance record, whose hyperparameters carry the order bound, the band factor, the layer index, the minimum class count and the fold count. The raw per-sample scores sit beside it as `beatrix_scores_validation.pt`, `beatrix_scores_clean.pt` and `beatrix_scores_backdoor.pt`, 1 float32 tensor of the split's length each, in the loader's order and negated so that low means poisoned, so any later threshold or fusion reads them without a rerun.

## Results

<!-- results:begin -->
<!-- results:end -->

## Known failure modes

A low score says the input's Gram entries at the feature layer fall outside the band of the class the model put it in. That happens when a trigger reroutes the prediction through activations the class's clean references never show, and it also happens to any clean input whose representation is unusual for its predicted class, which on a weak model includes the inputs it misclassifies. The paper's own adaptive attack, a Gram-matching loss at $\lambda = 1$, brings TPR at 1% FPR down to 14% on CIFAR-10, so an attacker who controls training can close the gap the method reads.

The massive-activation dimensions of a ViT residual stream are the first thing to watch. A few dimensions at block 9 carry values in the hundreds while the rest sit near 1, and the Gram rows of those dimensions are the entries at the top of float32's range at $p = 4$. The deviation is relative to the band edge, so their scale alone leaves the score unmoved. They are the entries most sensitive to a change in the token that carries them, so a patch trigger that moves the massive activation of its own tokens registers on them first. A layer index off by 1 or 2 changes which dimensions those are.

Tiny ImageNet's budget is the second. About 10 references per predicted class gives a median and a MAD read off 10 values, and the fold fits of the jackknife rest on 8. Bands that narrow flag clean inputs, and the classes that fall under 5 references take the pooled band, which spans every class and is therefore wide, so a poisoned input predicted into a pooled class deviates less than 1 predicted into a class with its own band. The all-to-one target class holds its own band whenever the model puts at least 5 clean references there, which a target class of 200 does on a clean-accurate model.

Benign checkpoints are scored under a probe attack, and there the trigger patch can shift the predicted class of a clean input without any backdoor. A shifted input is then compared against the band of the class it moved to, where its Gram is foreign for a reason that has nothing to do with poisoning, and an unshifted 1 still carries the patch's tokens in its Gram against its own class band. Both read as detections on a model that has no backdoor, so the benign rows of the panel can sit above the [0.35, 0.65] window the smoke acceptance draws around chance. `attacks.poisoning.AttackSuccessSet` keeps the backdoor split to the inputs the trigger actually fooled, which removes the unshifted case from the scored set but leaves the shifted 1.

A large share of clean scores sits at exactly 0, every entry inside its band, so the score floor is a tie. On the synthetic fixture 57% to 73% of the clean and validation scores are 0, and any quantile above the nonzero share lands the threshold on that tie, where the strict less-than rule flags nothing. `tie_share_at_threshold` in the record surfaces it. The headline quantile of 0.25 lies inside the nonzero tail on the fixture, and whether it does on a real checkpoint is a number the smoke run has to read.

The synthetic fixture in `experiments/preflight/synthetic.py` reads differently from the plan's expectation. Its inner ViT is randomly initialised on random-noise images, its head puts 123 of 128 validation inputs in 1 class, so 9 of the 10 classes fall back to the pooled band, and every triggered input is compared against that pooled band as the target class has no reference. Even so the 6 by 6 checkerboard sits inside the last 8 by 8 patch and rewrites 1 of the 16 patch tokens at the embedding, the random blocks pass that change through, and the Gram over the tokens registers it: at layer 1 with 128 samples the port reads AUROC 0.921 with TPR 0.898 at FPR 0.250, at 192 samples 0.900, and at layer 2 0.785 and 0.887. The reading says that the Gram statistic separates a fixed input patch from noise on random features, which is a property of the patch rather than of a backdoor, so the fixture exercises the plumbing and the direction and says nothing about the method on a trained model.

## Direction

High deviation is poisoned. The paper flags an input whose $\delta$ exceeds a percentile of the benign deviations and the released driver counts a detection when a deviation exceeds the 95th or 99th percentile of the clean ones, so the paper's statistic is high for poisoned, the opposite of the registry's convention. `deviation_scores` negates it once, and both the jackknife scores of the validation split and the scores of every other loader pass through that 1 function, so the 2 cannot disagree in sign. `beatrix_deviations` and `jackknife_deviations` return the unnegated statistic, comparable to a published deviation, and `beatrix_scores` returns the negated 1. A second negation anywhere would produce a well-formed, exactly inverted detector, which is the failure the `direction` field of the report exists to catch.
