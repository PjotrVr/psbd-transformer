# Session record, 2026-09-08

Head at write time: `372a595`. 28 PBS jobs in flight.

## Defects found and fixed

| # | severity | defect |
|---|---|---|
| 1 | **critical** | A re-sweep removed only `results/<cell>/psbd/<placement>`, leaving `baseline_<split>.pt`. `load_or_build_baseline` reuses any baseline whose row count matches, and it always matches, because the split depends on the dataset and seed rather than on the model. So a retrained checkpoint was scored as the OLD model's confidence minus the NEW model's dropout passes, with the tracked class also from the old model. Caught by mtime: `vit_cifar10_wanet_0_1` retrained 14:33 against a baseline written 2026-08-13. 28 of 36 in-flight cells had not reached their sweep and were cleared by hand; 9 need a re-sweep |
| 2 | high | STRIP superimposed in **normalized** space, giving `(p1 + p2 - 2m)/s` where `cv2.addWeighted` on uint8 gives `(p1 + p2)/s` saturated: a further `-m/s` per channel, 2.43 units on CIFAR-10 channel 0, and no clip. Fixed in all 3 copies |
| 3 | medium | `median_rank_union`, `targeted_head_psbd` and `occlusion_probe` never paired clean to backdoor. `targeted_head_psbd` also took a first-500 slice of a class-ordered test set, covering **10 of 200 classes on Tiny** |
| 4 | medium | 8 pre-fix Gaussian caches survived audit A2, all `before_mlp_gaussian_k20` at **1% poisoning**, i.e. the H24 x H23 x low-rate intersection |
| 5 | low | H17 still led with the table audit A16 withdrew, and never used the word "withdrawn" |

## Measurements

**Monte Carlo passes do not matter.** From the 51 cached k=20 cells: AUROC 0.779 at k=1,
0.800 at k=3, 0.814 at k=20. k=3 to 20 buys +0.014 for 6.7x the compute. And H24's framing
is wrong: the run-to-run spread of AUROC across 6 disjoint triples from the same cache is
**0.0009**, so k=3 is a precise estimate, not a noise floor.

**AUROC hides the failure that matters.** Non-SAM ViT at 1%, ASR >= 0.5, n=19: mean AUROC
0.908 but mean TPR at 1% FPR **0.464**. Six cells have AUROC >= 0.85 with TPR@1%FPR < 0.05,
worst 0.960 / 0.000. Achieved FPR tracks nominal, so it is a genuine tail overlap. AUPRC and
partial AUROC are computed nowhere in the repo.

**PSBD is not a margin detector.** The deterministic logit margin, exactly `z1 - z2` and
never previously tested, scores **0.375** against PSU's **0.891** over 55 all-to-one cells.
An independent adversarial review reached the same conclusion by stratifying on the top-2
margin and finding PSBD's AUROC barely moves. This kills the margin-matching attack family,
including the one proposed at the start of the session.

**Entropy covers all-to-all** (H43). 0.761 against PSBD's 0.411 over 12 cells, benign at
chance, from the deterministic softmax the sweep already caches and discards.

**A label-free sign rule selects between them** (H43). `d` = suspect-pool mean PSU minus
clean-validation mean PSU; use PSU if `d < 0`, else entropy. Threshold 0 is fixed by the
mechanism. Over 64 cells: 0.819 to **0.865**, CI [+0.017, +0.079], **98% of oracle-max**,
hurting **0** cells, and +0.044 to +0.059 at all 4 placements including the published one.

## The strongest result: continuous prediction depth

A logit lens reads every encoder block through the network's OWN final LayerNorm and head,
so a per-sample statistic can ask WHEN the answer was decided rather than how far it moves
under perturbation. That is algebraically independent of the whole PSBD family, which H28
shows all read one magnitude.

`depth_cls` takes 12 integer values, and that alone was hiding the result. A
clean-validation quantile lands on a tie and the strict `<` test drops every sample sitting
exactly on it. On `vit_gtsrb_badnet_a2o_0_1` that was the entire backdoor population: mean
depth **exactly 5.000** against clean 7.292, AUROC 0.994, and TPR at 1% FPR reading
**0.000** because the threshold IS 5.0. The same score gives TPR 1.000 at 5% FPR with an
achieved FPR of 0.011.

`depth_soft`, the mean over layers of the lens probability of the final answer, is the same
mechanism made continuous:

| cell | `depth_cls` | `depth_soft` |
|---|---|---|
| gtsrb badnet_a2o | 0.994, TPR@1% **0.000** | **1.000**, TPR@1% **1.000**, AUPRC 1.000 |
| gtsrb wanet | 0.933, TPR@1% 0.000 | 0.946, TPR@1% **0.839** |
| gtsrb benign | 0.499 | **0.501** |

One forward pass against PSBD's 30, which on that badnet cell reads 0.999 AUROC and 0.981
TPR at 1% FPR. Across the GTSRB dirty-label group `depth_cls` alone means **0.983** against
PSBD's 0.974.

**Novelty warning.** TED (S&P 2024) and TED++ do layer-wise trajectory analysis with
clean-only calibration for backdoor input detection, above 0.95 AUROC, on ResNets. That is
the closest prior art and the number to beat, not a related-work citation. Orion (IJCAI
2023) also uses internal-readout-disagrees-with-final-answer as a per-input poisoned score.

## The per-token direction

A logit lens reading every block through the network's own final LayerNorm and head. On
`vit_cifar10_badnet_a2o_0_1`, the fraction of patch tokens agreeing with the model's answer
gives AUROC **0.982** and TPR@1%FPR **0.287** from **1** forward pass, against PSBD's 0.961
and 0.000 from 30. CLS alone gives 0.826.

**And it does not generalise as stated.** The sign depends on the trigger's spatial extent
AND on the class count:

| cell | raw AUROC | clean -> backdoor agreement |
|---|---|---|
| cifar10 badnet_a2o | 0.982 | 0.429 -> 0.064 |
| cifar10 wanet | 0.062 | 0.195 -> 0.707 |
| cifar10 sig | 0.100 | 0.335 -> 0.539 |
| **gtsrb badnet_a2o** | **0.162** | 0.071 -> 0.033 |

The same attack inverts between CIFAR-10 and GTSRB. With 43 classes a patch token rarely
resolves the true class anyway (chance 1/43), so clean agreement is already near the floor
and the confident trigger patches raise it. The sign router recovers 7 of 9 cells (0.423 raw,
0.577 flipped, **0.731** routed) but fails exactly where `|d|` is small.

**Consequence for methodology: pilot on GTSRB, not CIFAR-10.** CIFAR-10's 10 classes made a
class-count-sensitive statistic look far stronger than it is.

## Two probes outside the activation-noise family

The user asked for a genuinely new perturbation paradigm rather than another operator.

**Gradient-guided head-weight ablation, REFUTED.** For each sample, rank the head's weights
by `|W_c . a|`, the exact gradient-times-weight attribution for the predicted logit, disable
the top-k and ask whether the answer survives. The a priori sign came from H16: the backdoor
is a rank-1 direction, so the answer should rest on few head weights and die early. It does
not. AUROC **0.267**, backdoor survival 0.742 against clean 0.639, benign at 0.499. A
rank-1 direction being sufficient to remove the backdoor when deleted from the activation
stream does not imply the prediction rests on few head weights. Recorded, not flipped.

**Weight interpolation toward the pretrained initialization, in flight.** The victim was
finetuned from public ImageNet weights, so `tau = theta_ft - theta_pre` is computable with
no poison label and `theta(a) = theta_pre + a * tau` walks the model back along it. The
backdoor was written entirely during finetuning, so it should die as `a` shrinks while clean
knowledge, partly inherited, survives.

## The content-dependence law, first data point

`vit_gtsrb_badnet_a2m2_0_1`: **ASR 0.989, CA 0.992**. So m = 2 costs essentially no attack
success, against all-to-all's 0.947 on the same dataset. The ASR side of the law behaves as
predicted; 33 more cells are training.

## Built

- `all_to_m`, the label map `(y + 1) mod m`, verified to reduce **exactly** to all-to-one at
  m = 1 and all-to-all at m = num_classes, with a guard against `m > num_classes` emitting an
  out-of-range label. 34 cells training to turn "PSBD fails on all-to-all" into a curve in
  `log2(m)`.
- `experiments/prediction_depth/`, which unlike `cli/baselines.py` caches its per-sample
  tensors, so a new scoring rule is a CPU rescore rather than another GPU pass.
- `experiments/all_to_all_entropy/` and its router.

## Open

1. The 9 rebuild cells swept against stale baselines need re-sweeping.
2. Whether the content-dependence curve is smooth in `log2(m)`. This is the sharpest
   falsifier available for the regime-router story: if the regime is a continuum, a router
   has to estimate a position on it rather than pick a side.
3. A per-token statistic whose sign is stable across class count and trigger extent.
4. AUPRC and TPR at 1/5/10% FPR belong in `detection_report`, deferred until the queue drains
   because in-flight jobs execute the working tree.
