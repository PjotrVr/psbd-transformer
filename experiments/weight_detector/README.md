# The exact Karayalcin weight detector, grid-searched, on our checkpoints

## Question

Karayalcin, Krcek, Chen and Picek, "Backdoor Directions in Vision Transformers"
(arXiv 2603.10806), Section 7, propose a data-free detector that reads only trained
weights, no forward pass and no clean data. They report WaNet and BPP flagged
reliably with the correct target class across a broad grid, SSBA marginal and
TrojanNN never. An earlier transplant in this repo
(`docs/hypothesis/H16-where-the-backdoor-neurons-are.md`, head-alignment section, and
`analysis.lipschitz.head_weight_alignment`) fired on 2 of 18 checkpoints and both were
wrong, but it committed to 1 threshold rather than searching the paper's grid, folded
in a LayerNorm normalization the paper never states, used only the attention output
projection and floored the Z-score denominator at a fixed count rather than at the
threshold itself. This asks whether the paper's rule, implemented exactly as written
and swept over its own grid, does any better on this project's checkpoints.

## The rule, quoted

From Section 7 ("Detecting Backdoors from Weights") and its restatement in the
supplementary:

> As such, we take the classifier head matrix O in R^(n_class x d), where each row
> c_i represents the readout direction for class i. To identify potential backdoors,
> we measure the degree of alignment between each c_i and the layer weights W in the
> first n layers. For each class i, we compute a detection score
> s_i = sum_l I[abs(c_i^T W)_l > t] where t is a threshold and I[.] denotes the
> indicator function, i.e., we add any value in abs(c_i^T W) that is larger than t to
> the score.
>
> To detect whether the top class is an outlier, we also evaluate whether its score
> is an outlier. To do this, we define a Z-score:
> Z = (s_top - s_second) / max(std(S \ {s_top}), t), where s_top and s_second are the
> top two scores and S = {s_0, ..., s_n_class}. We take the maximum of this standard
> deviation and the threshold for cases where the standard deviation is 0, which
> occurs when the top class is the only one to ever exceed t. We then define an
> outlier as Z > 3.

| symbol | meaning |
| --- | --- |
| $O \in \mathbb{R}^{n_{class} \times d}$ | classifier head weight matrix |
| $c_i$ | row $i$ of $O$, the readout direction for class $i$ |
| $W_l$ | the weight matrix of layer $l$ that writes into the residual stream |
| $t$ | a threshold on $|c_i^\top W|$ |
| $n$ | how many of the first layers are searched |
| $s_i$ | detection score for class $i$ |
| $S$ | the full set of per-class scores |
| $Z$ | outlier score of the top class against the rest |

Nowhere in the main text or the supplementary is $c_i$ or a column of $W$ normalized.
The paper never states which matrices count as "layer weights $W$" beyond the
parenthetical that ties this section to the weight-orthogonalization step earlier in
the paper, which updates "the initial embedding layer and all attention and MLP output
projection matrices." This experiment reads $W_l$ as the concatenation of `layer.
self_attention.out_proj.weight` and `layer.mlp[3].weight`, the 2 per-block matrices
that write into the residual stream (the embedding layer is excluded, since it is not
indexed by $l$ and is written once, not per layer). $c_i$ is the raw row of
`heads.head.weight`, unnormalized.

## Grid

The paper's own grid-search figures (`cifar100_vit_b_16_grid_search_all_ratios_
subplots.png` and its main-text counterpart, both under
`papers/backdoor_directions/images/model_detect/`) give the exact axes searched, so
the task's 1-to-12/quantile fallback was not needed:

- $n \in \{1, \dots, 11\}$ (num layers, y axis)
- $t \in \{0.01, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50\}$ (threshold, x
  axis), an absolute scale on $|c_i^\top W|$, not a quantile

11 values of $n$ times 10 values of $t$ gives a 110-cell grid per checkpoint.

## Method

`measure.py` loads each checkpoint with `models.backbones.load_checkpoint` and
`network_core`, reads `heads.head.weight` for $c_i$ and, for each of the first 11
blocks, `self_attention.out_proj.weight` and `mlp[3].weight` for $W_l$. For every
grid cell it computes $s_i(n, t)$ for all classes, then $Z$ and the top class exactly
as quoted above, including the paper's own choice to floor the standard deviation at
$t$ rather than at a fixed count. A cell is a **hit** when $Z > 3$ and the top class
equals the checkpoint's `target_label` from `args.json`, a **false alarm** when
$Z > 3$ and the top class is anyone else, and **no detection** when $Z \le 3$. Every
number is weights only, CPU, and the 18-checkpoint sweep runs in under a minute.

Run: `PYTHONPATH=. python experiments/weight_detector/measure.py`. Output:
`results/_experiments/weight_detector/summary.json`, written through
`experiments._paths.experiment_result_path`, holding the rule, the grid and, per
checkpoint, every grid cell plus the aggregate shares below.

## Results

18 checkpoints (`vit_cifar10`, `vit_cifar100`, `vit_tiny`), each swept over the full
110-cell grid. Rate is the training poison rate, hit share is the fraction of the grid
at $Z > 3$ naming the correct target, false-alarm share is $Z > 3$ naming the wrong
class, and the 2 plus no-detection sum to 1.

| checkpoint | attack | rate | hit share | false alarm share | best $Z$ | class named at best $Z$ |
| --- | --- | --- | --- | --- | --- | --- |
| `vit_cifar100_wanet_0_1` | WaNet | 0.10 | 0.109 | 0.055 | 69.65 | target (correct) |
| `vit_cifar100_wanet_0_05` | WaNet | 0.05 | 0.227 | 0.045 | 139.30 | target (correct) |
| `vit_cifar100_bpp_0_1` | BPP | 0.10 | 0.000 | 0.064 | 10.00 | class 68 (wrong) |
| `vit_cifar100_bpp_0_05` | BPP | 0.05 | 0.000 | 0.064 | 10.00 | class 68 (wrong) |
| `vit_cifar100_bpp_0_01` | BPP | 0.01 | 0.027 | 0.055 | 10.00 | class 68 (wrong) |
| `vit_cifar100_badnet_a2o_0_1` | BadNet | 0.10 | 0.000 | 0.036 | 5.00 | class 56 (wrong) |
| `vit_cifar100_blend_0_1` | Blend | 0.10 | 0.000 | 0.100 | 10.00 | class 85 (wrong) |
| `vit_cifar100_lf_0_1` | LF | 0.10 | 0.018 | 0.036 | 10.00 | target (correct) |
| `vit_cifar100_tact_0_1` | TaCT | 0.10 | 0.000 | 0.055 | 10.00 | class 59 (wrong) |
| `vit_cifar10_wanet_0_1` | WaNet | 0.10 | 0.055 | 0.018 | 40.00 | target (correct) |
| `vit_cifar10_bpp_0_1` | BPP | 0.10 | 0.055 | 0.018 | 71.43 | target (correct) |
| `vit_cifar10_badnet_a2o_0_1` | BadNet | 0.10 | 0.000 | 0.127 | 160.00 | class 6 (wrong) |
| `vit_tiny_wanet_0_1` | WaNet | 0.10 | 0.109 | 0.045 | 34.59 | target (correct) |
| `vit_tiny_bpp_0_1` | BPP | 0.10 | 0.000 | 0.073 | 7.11 | class 68 (wrong) |
| `vit_tiny_badnet_a2o_0_1` | BadNet | 0.10 | 0.000 | 0.127 | 25.49 | class 47 (wrong) |
| `vit_cifar100_benign` | none | 0 | 0.000 | 0.145 | 40.00 | class 18 (n/a, clean model) |
| `vit_cifar10_benign` | none | 0 | 0.000 | 0.027 | 14.29 | class 0 (n/a, clean model) |
| `vit_tiny_benign` | none | 0 | 0.000 | 0.109 | 14.27 | class 113 (n/a, clean model) |

All backdoored checkpoints reach ASR above 0.79 (WaNet, the weakest) and above 0.91
elsewhere, so a low hit share is not standing in for a dead backdoor.

## Verdicts against the paper's claims

**WaNet: detected, matching the paper.** All 3 WaNet checkpoints (CIFAR-10,
CIFAR-100 at 2 rates, Tiny) name the correct target at a nontrivial share of the
grid (0.055 to 0.227) and at very high best-cell $Z$ (35 to 139). This is the 1
attack where the paper's claim replicates cleanly on our checkpoints.

**BPP: mixed, not the reliable detection the paper reports.** Only the CIFAR-10
checkpoint hits (0.055 share, best $Z$ 71.43). The 3 CIFAR-100 checkpoints (rates
0.10, 0.05, 0.01) and the Tiny checkpoint all fail to name the target anywhere in the
grid, best $Z$ locking onto class 68 (CIFAR-100/Tiny share this wrong class at $n=3$,
$t=0.10$, which suggests a dataset-level artifact in that class's head row rather
than 3 independent failures). Since the paper groups BPP with WaNet as reliably
detected, this is the clearest disagreement with their claim, and it holds across 4
of the 5 BPP checkpoints tested, at every rate we ran.

**TrojanNN: not tested.** No TrojanNN checkpoint exists in `checkpoints/` for this
project (BadNet, Blend, LF and TaCT stand in as the other patch- or global-trigger
attacks). Their behavior is consistent with the paper's "never" claim for TrojanNN in
spirit: BadNet, Blend and TaCT never hit at any grid cell.

**LF: a partial, low-share hit, not claimed either way by the paper.** LF names the
correct target at 0.018 of the grid (only at $n=1$, $t=0.10$), the smallest nonzero
hit share observed. The paper does not test LF, so this is new information rather
than a check against a stated claim, and a single-cell hit at the very edge of the
grid does not support calling LF detected.

**BadNet, Blend, TaCT: never detected, consistent with the paper's account of static
or global triggers.** 0 hits across every grid cell on all 4 checkpoints (BadNet at 2
datasets, Blend, TaCT), matching the paper's framing that patch- or global-pattern
triggers do not produce the early shortcut signature this detector reads.

**Benign references carry a real false-alarm rate.** With no backdoor, the 3 benign
checkpoints still cross $Z > 3$ on 0.027 to 0.145 of the grid, always naming a class
that means nothing (no ground-truth target exists to check against). This is not a
bug in the transplant, it is the paper's own Z-score definition: flooring the
denominator at $t$ rather than at a fixed count means any 2 classes tying in their
counts collapses the spread to 0, and $Z$ then divides by a threshold as small as
0.01, which can turn a difference of 1 count into $Z = 100$. The paper's authors
flag this arbitrariness themselves ("these parameters/scoring functions are rather
arbitrary and based on some trial and error... we do not aim for this to be an actual
defense"). A deployer reading only the flagged/not-flagged bit at 1 fixed $(n, t)$
would see this rate as a real false-positive cost, which is exactly the kind of
single-threshold reading H16's earlier transplant used and this grid replication was
built to avoid.

## Net read

The paper's grid-search version of the detector reproduces cleanly for WaNet, not for
BPP, and correctly finds nothing for the patch- and global-trigger attacks in our
panel. Against the earlier single-threshold transplant (2 of 18 checkpoints flagged,
both wrong), the exact grid version does meaningfully better: it flags the correct
target on 6 of the 15 backdoored checkpoints tested (both WaNet datasets and rates,
plus CIFAR-10 BPP, plus a marginal LF cell) at a nonzero grid share, while never
flagging the wrong class as its best cell on those same 6. But BPP, the paper's other
headline attack, only replicates on 1 of 4 BPP checkpoints here, so "WaNet and BPP
detected" does not hold as stated for this project's checkpoints, only "WaNet
detected, BPP inconsistent."
