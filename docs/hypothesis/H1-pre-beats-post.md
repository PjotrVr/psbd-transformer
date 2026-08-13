# H1 — Pre-residual dropout beats post-residual dropout on ViT

**Status: REFUTED as a general claim.** It holds for exactly one trigger family
(static patch) and is reversed for the other three. `pre_residual` is also not the
best placement available: it ranks 4th of 11.

## Claim

Placing PSBD's dropout on each transformer branch just before the residual add
separates clean from backdoor samples better than placing it on the residual stream
just after each add, on ViT-B/16. This was the project's founding observation.

## Evidence

Full grid: CIFAR-10 ViT, 11 placements x 5 attacks x 3 poison rates + benign control,
full 10000-image test split. Best-rate AUROC at the 25th-percentile threshold, 10%
poisoning. Crucially, `post_residual` is now swept over **its own** rate window
(0.005 to 0.09) as well as the standard grid, which it was not in phase 3a.

| placement | blend | bpp | lf | badnet_a2o | badnet_a2a | benign | mean (4 working) |
|---|---|---|---|---|---|---|---|
| `before_mlp_residual` | 0.984 | 0.992 | **0.979** | 0.922 | 0.508 | 0.508 | **0.969** |
| `before_attention_norm` | **0.999** | 0.986 | 0.937 | 0.939 | 0.471 | 0.503 | 0.965 |
| `before_attention` | **0.999** | 0.992 | 0.906 | **0.952** | 0.498 | 0.507 | 0.962 |
| `pre_residual` | 0.978 | 0.977 | 0.947 | 0.889 | 0.510 | 0.506 | 0.948 |
| `before_mlp` | 0.988 | **0.993** | 0.930 | 0.877 | 0.517 | 0.502 | 0.947 |
| `post_residual` | 0.989 | 0.991 | 0.969 | 0.747 | 0.523 | 0.507 | 0.924 |
| `before_mlp_norm` | 0.943 | 0.986 | 0.924 | 0.799 | 0.503 | 0.504 | 0.913 |
| `before_attention_residual` | 0.939 | 0.943 | 0.880 | 0.877 | 0.497 | 0.504 | 0.910 |
| `after_mlp_residual` | 0.977 | 0.985 | 0.940 | 0.682 | 0.517 | 0.506 | 0.896 |
| `after_attention_residual` | 0.961 | 0.975 | 0.903 | 0.654 | 0.514 | 0.508 | 0.873 |
| `after_embedding` | 0.973 | 0.987 | 0.907 | 0.601 | 0.498 | 0.508 | 0.867 |

Head to head, at each placement's own best rate:

| attack | `pre_residual` | `post_residual` | post's best rate | winner |
|---|---|---|---|---|
| `blend` | 0.978 | **0.989** | 0.07 | post |
| `bpp` | 0.977 | **0.991** | 0.07 | post |
| `lf` | 0.947 | **0.969** | 0.07 | post |
| `badnet_a2o` | **0.889** | 0.747 | 0.02 | pre |

**Post-residual wins on three of the four working attacks.** The founding claim
survives only for the static patch trigger, where pre-residual leads by +0.142.

## Why the earlier verdict was wrong

Phase 3a swept `post_residual` only from p=0.1 upward. Its usable window is p = 0.005
to 0.09, so every phase-3a post-residual number was measured *outside its operating
range*. Given its own window, it gains +0.065 (blend), +0.057 (bpp), +0.053 (lf), and
+0.080 (badnet_a2o) over its phase-3a scores.

This is [H9](H9-strength-not-position.md), the adversarial hypothesis, being right.
The apparent pre-versus-post gap was mostly an artifact of comparing one placement
inside its range against another outside its own.

## Two further findings the ranking exposes

**`pre_residual` is worse than half of itself.** It combines
`before_attention_residual` (0.910) and `before_mlp_residual` (0.969), and scores
0.948 — *below* its better component. Adding the attention-branch perturbation
actively dilutes the MLP-branch one. `post_residual` behaves oppositely: its
components score 0.873 and 0.896 and the combination reaches 0.924. So combining
positions is not additive in either direction, and the two-position combos that
frame this whole study were never the right unit of analysis.

**The best placements are on the attention side, and none of them is a residual
position at all.** `before_mlp_residual`, `before_attention_norm` and
`before_attention` take the top three. Two of those perturb *inputs to a sublayer*,
not contributions to the stream.

## Reproduce

```bash
python psbd_analyze.py --all
python psbd_report.py
```

## Subquestions

1. Why does adding `before_attention_residual` to `before_mlp_residual` make it
   worse? A rate mismatch (the combo applies the same p at both) is the obvious
   candidate and is testable with a per-position rate.
2. `before_mlp_residual` alone is the best general placement measured. Is it also
   best under SAM, and on Swin?
3. `badnet_a2o` is the only attack where any placement matters much. Is the whole
   placement question therefore only interesting for hard-to-detect attacks?
