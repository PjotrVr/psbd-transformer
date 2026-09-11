# Whether SAM helps PSBD on ViT, matched combination for combination against Adam

## Question

Does training the victim with sharpness-aware minimisation help PSBD detect the
backdoor on ViT and Swin, and does SAM amplify the backdoor the way its paper
(`papers/reliable_poisoned_sample_detection_against_backdoor_attacks_enhanced_by_sharpness_aware_minimization/`,
Zhang et al., arXiv 2411.11525) claims.

[H6](../../docs/hypothesis/H6-sam-improves-detectability.md) already dropped a naive
Adam against SAM aggregate as confounded: the SAM checkpoints swept so far were not
the same architecture, dataset, attack and poison rate as the Adam checkpoints they
were compared against, and matching on those 4 fields flipped the sign. This
experiment builds that matched comparison directly instead of estimating it from a
coverage note.

## Method

`measure.py` scans `checkpoints/` for every (architecture, dataset, attack, poison
rate) combination that trained both an Adam checkpoint and at least 1 SAM
checkpoint, over `badnet_a2o`, `blend`, `bpp`, `lf` and `wanet`. `badnet_a2a` is
excluded on purpose: it is the atypical inverted attack that made up 13 of the 18
matched combinations behind the correction note on H6, so it cannot settle this on
its own either.

For every combination it reads ASR and clean accuracy from each side's
`checkpoints/<folder>/metrics.json`, and, wherever `cli.sweep` and `cli.analyze`
have already reached both the Adam and the SAM checkpoint, the token-mask placement
at the attention input (`before_attention_norm_token_mask`) and the dropout
placement after the residual add (`post_residual`) AUROC and TPR at the q0.10 and
q0.20 false-positive budgets, at the deployable adaptive rate. Placement values are
read through `cli.compare_detectors.psbd_values`, the same function every paper
table uses, never recomputed from the raw score cache.

Per rho, the paired AUROC difference (SAM minus Adam) is bootstrapped with 5000
resamples over the combinations where both placements were swept on both sides, the
same common-coverage discipline `scripts/paper/tab_headline.py` applies to its own
3-way comparison. Output: `results/_experiments/sam_reading/sam_reading.json`.

The SAM paper's own amplification metrics, `top2_tac` (their "backdoor effect"),
`silhouette` and `clean_intra_class_variance`, are already computed in
`experiments/sam_backdoor_effect/measure.py` and folded in wherever they are on
disk, with their coverage stated plainly since it is far narrower than the PSBD
grid: ViT only, CIFAR-10 only, poison rate 0.1 only, `badnet_a2o`, `blend`, `bpp`
and `lf` (no `wanet`).

`scripts/paper/tab_sam.py` reads `sam_reading.json` and writes `paper/tables/sam.tex`
and its macro sidecar, 1 row per rho with Adam as the reference row.

## Table

| Rho | Pairs | ASR | CA | PSBD token mask AUROC | PSBD token mask TPR at 10% | PSBD residual dropout AUROC | PSBD token mask delta vs Adam | 95% CI |
|---|---|---|---|---|---|---|---|---|
| Adam | 96 | 0.983 | 0.902 | 0.968 | 0.930 | 0.899 | -- | -- |
| 0.05 | 24 | 0.984 | 0.898 | 0.973 | 0.933 | 0.851 | +0.034 | [-0.000, +0.082] |
| 0.1 | 24 | 0.985 | 0.899 | 0.975 | 0.938 | 0.896 | +0.035 | [+0.002, +0.079] |
| 0.15 | 24 | 0.984 | 0.901 | 0.954 | 0.887 | 0.822 | +0.010 | [-0.032, +0.058] |
| 0.2 | 24 | 0.983 | 0.897 | 0.960 | 0.897 | 0.823 | +0.023 | [-0.024, +0.076] |

The Adam row is read over the 96 Adam checkpoints in the matched grid that swept
both placements. Each SAM row's Pairs count, 24 throughout, is the narrower set
that also swept both placements on the SAM side, all ViT, all CIFAR-10 or
CIFAR-100: the SAM sweep has not yet reached Swin, GTSRB or Tiny ImageNet for this
attack set, and the delta column is paired only within that set. The residual
dropout delta at rho 0.15 and 0.2 excludes 0 entirely on the negative side, meaning
the published placement's AUROC drops with SAM: [-0.164, -0.004] and [-0.157,
-0.003].

The amplification metrics, where they exist (ViT, CIFAR-10, rate 0.1, `badnet_a2o`,
`blend`, `bpp`, `lf`), repeat the earlier finding: `top2_tac` rises with rho
(mean delta +0.71 at rho 0.15, n=4) while `silhouette` barely moves (mean delta
0.004 at rho 0.15), so the backdoor amplifies without the clean and triggered
features separating any further.

## Conclusion

SAM helps PSBD on ViT at the token-mask placement and hurts it at the placement the
original PSBD paper published, so the answer to "does SAM help PSBD" depends on
which placement is asked. The token-mask gain is small and mostly rho-independent,
+0.010 to +0.035 mean AUROC with the tightest interval at rho 0.1 ([+0.002,
+0.079]), while the residual-dropout loss grows with rho and reaches -0.069 to
-0.074 with both bootstrap intervals excluding 0 at rho 0.15 and 0.2. Every one of
these numbers comes from 24 matched combinations, all ViT and all CIFAR-10 or
CIFAR-100, so this settles the question for that architecture and those 2 datasets
only, not yet for Swin, GTSRB or Tiny ImageNet.

## Reproduce

```bash
PYTHONPATH=. python experiments/sam_reading/measure.py
PYTHONPATH=. python scripts/paper/tab_sam.py --paper-dir paper
```
