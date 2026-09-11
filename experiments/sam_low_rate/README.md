# Does SAM's PSBD gain on ViT survive at 1% and 5% poisoning

## Question

experiments/sam_reading pooled every poison rate together and found a small
token-mask (PSBD-TM, `before_attention_norm_token_mask`) gain from training the
victim with sharpness-aware minimisation and a loss for the placement the
original PSBD paper published (PSBD-RD, `post_residual`). That pooled number
cannot say whether the gain is real at the poison rates an attacker would
actually use, whether it is separation or an artefact of the adaptive rule
picking a different dropout rate on the SAM side, or whether it holds across
the whole disturbance ladder or only at 1 rung of it. `measure.py` re-slices
the same matched Adam-versus-SAM cells by poison rate to answer those 3
questions, plus a fourth: whether the gain concentrates in the attacks whose
trigger is diffuse (BPP, WaNet) rather than firm and localised (BadNets, Blend,
LF), which is what the SAM paper's own backdoor-neuron-amplification account
would predict.

## Coverage at 1% and 5%

40 matched (dataset, attack, architecture) combinations exist at each of 1%
and 5% poisoning, all 4 datasets (CIFAR-10, CIFAR-100, GTSRB, Tiny ImageNet)
by all 5 attacks (`badnet_a2o`, `blend`, `bpp`, `lf`, `wanet`) by both
architectures (ViT-B/16 and Swin-S), and every one reaches all 4 swept rhos
(0.05, 0.1, 0.15, 0.2). That is the full checkpoint grid. The narrower number
that matters for the table is how many of those 40 also have `cli.sweep` and
`cli.analyze` output for both PSBD-TM and PSBD-RD on both sides: 7 at 1% and
7 to 8 at 5% (`results/_experiments/sam_low_rate/sam_low_rate.json`,
`by_rate.<token>.by_rho.<rho>.n_pairs`), all ViT, all CIFAR-10 or CIFAR-100.
The PSBD sweep has not yet reached Swin, GTSRB or Tiny ImageNet for both
placements at these 2 rates, so this subset is large enough to bootstrap an
interval but not to say anything about architecture or the harder datasets at
low poison rates yet.

## The ASR gate

SAM sometimes breaks implantation rather than changing detectability. Of the
swept-both-placements grid, 1 pair drops below `ASR_CLEARS_THRESHOLD` (0.85):
10% poisoning, CIFAR-100, WaNet, ViT, rho 0.05, where Adam ASR is 0.900 and
SAM ASR is 0.831. Only 1 side needs to fall under the bar for the pair to
drop, and here it is the SAM side. Every comparison in this experiment drops a
pair where either side's ASR falls below 0.85 before computing a delta, so a
"loss" is never read off a pair where SAM quietly failed to plant the
trigger. `paper/tables/sam_low_rate_excluded.tex` lists this 1 dropped pair.
No pair at 1% or 5% poisoning was dropped, because the pairs that reached
both placements at those rates all cleared 0.85 ASR on both sides. The
WaNet-at-1%-poisoning collapse the SAM audit flagged (ASR near 0.03 to 0.04
even under Adam, since WaNet is already weak at 1% before SAM is added) sits
in cells `cli.sweep` has not yet reached for both placements, so it is absent
from `n_pairs` for a coverage reason rather than being silently averaged in.

## Part 1: does the gain survive at 1% and 5%

`paper/tables/sam_low_rate.tex`, 1 row per (poison rate, rho):

| Rate | Rho | Pairs | TM Adam | TM SAM | TM delta | TM 95% CI | RD Adam | RD SAM | RD delta | RD 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| 1% | 0.05 | 7 | 0.961 | 0.960 | -0.001 | [-0.035, +0.027] | 0.906 | 0.829 | -0.077 | [-0.304, +0.072] |
| 1% | 0.1  | 7 | 0.961 | 0.971 | +0.010 | [-0.018, +0.039] | 0.906 | 0.853 | -0.053 | [-0.237, +0.069] |
| 1% | 0.15 | 7 | 0.961 | 0.925 | -0.036 | [-0.125, +0.025] | 0.906 | 0.829 | -0.077 | [-0.226, +0.051] |
| 1% | 0.2  | 7 | 0.961 | 0.909 | -0.052 | [-0.132, +0.024] | 0.906 | 0.891 | -0.015 | [-0.068, +0.052] |
| 5% | 0.05 | 8 | 0.960 | 0.989 | +0.029 | [+0.002, +0.071] | 0.904 | 0.852 | -0.052 | [-0.198, +0.039] |
| 5% | 0.1  | 8 | 0.960 | 0.979 | +0.019 | [-0.024, +0.071] | 0.904 | 0.919 | +0.015 | [-0.046, +0.075] |
| 5% | 0.15 | 7 | 0.983 | 0.957 | -0.026 | [-0.073, +0.014] | 0.907 | 0.745 | -0.162 | [-0.379, -0.002] |
| 5% | 0.2  | 7 | 0.958 | 0.985 | +0.027 | [-0.003, +0.073] | 0.892 | 0.709 | -0.183 | [-0.429, +0.002] |
| 10% | 0.05 | 8 | 0.911 | 0.980 | +0.069 | [-0.010, +0.199] | 0.873 | 0.861 | -0.012 | [-0.063, +0.036] |
| 10% | 0.1  | 9 | 0.904 | 0.974 | +0.070 | [+0.008, +0.173] | 0.871 | 0.908 | +0.038 | [-0.038, +0.132] |
| 10% | 0.15 | 10 | 0.904 | 0.971 | +0.067 | [+0.011, +0.151] | 0.883 | 0.872 | -0.011 | [-0.082, +0.048] |
| 10% | 0.2 | 10 | 0.904 | 0.977 | +0.074 | [+0.007, +0.178] | 0.883 | 0.855 | -0.027 | [-0.088, +0.029] |

At 1% poisoning the token-mask delta is indistinguishable from 0 at every rho:
every interval straddles 0 and the point estimate itself flips sign across
rho (-0.052 to +0.010). At 5% it is positive at 3 of 4 rhos and its interval
excludes 0 at rho 0.05 ([+0.002, +0.071]), but the estimate still swings from
-0.026 to +0.029 across rho. Only at 10% does the gain become both consistent
in sign and mostly bootstrap-significant: +0.065 to +0.074 at every rho, 3 of
4 intervals excluding 0. The residual-dropout placement never gains at 1% or
5%, and its interval at rho 0.15, 5% ([-0.379, -0.002]) is the only 1% or 5%
row where either placement's interval excludes 0 in the losing direction.

## Part 2: is this separation or calibration

The adaptive rule picks the smallest dropout rate whose clean-validation shift
ratio reaches 0.8, so the rate itself can differ between Adam and SAM even at
the same nominal poison rate. It does, systematically: the mean token-mask
rate SAM selects is always higher than Adam's, at every rate and rho (at 10%
poisoning, rho 0.2: Adam 0.550 against SAM 0.690, and at 1% poisoning, rho
0.2: Adam 0.600 against SAM 0.686). SAM-trained ViTs are more resistant to
the same nominal token-mask disturbance, so the rule reaches for more dropout
to hit the same 0.8 shift
target on clean validation data. That alone would explain a spurious "gain" if
AUROC simply rose with rate everywhere, but the rate ladder (part 3) shows
SAM's curve already sits above Adam's at matched shift ratio, so the
calibration shift is real but is not the whole story behind the AUROC delta.
For residual dropout the calibration gap is much smaller (10%, rho 0.2: Adam
0.081 against SAM 0.122) and moves in the same direction, so the placement
that loses AUROC is not explained by SAM needing less disturbance either.

## Part 3: the whole disturbance ladder, at rho 0.1

Binning every swept rate's (clean-validation shift ratio, headline AUROC)
point from the 1% and 5% matched cells at rho 0.1 (`rate_ladder.points` in the
JSON) by shift ratio:

Token mask, mean AUROC by shift-ratio bin (Adam / SAM): 0.0-0.1: 0.734 / 0.828,
0.1-0.2: 0.791 / 0.899, 0.2-0.3: 0.896 / 0.915, 0.5-0.6: 0.932 / 0.982, 0.7-0.8:
0.961 / 0.989, 0.8-0.9: 0.867 / 0.927, 0.9-1.0: 0.905 / 0.924. SAM's curve sits
above Adam's at every bin with more than a handful of points, including the
0.8-0.9 and 0.9-1.0 bins where the adaptive rule (target 0.8) actually lands.
This is a real separation gain, not only a calibration artefact, since it
holds at matched disturbance and not just at matched nominal rate.

Residual dropout tells a different story: 0.0-0.1: 0.608 / 0.783, 0.1-0.2:
0.693 / 0.807, 0.2-0.3: 0.753 / 0.900, but 0.8-0.9: 0.700 / 0.702 and 0.9-1.0:
0.635 / 0.668, both roughly tied. SAM's residual-dropout curve is higher at
low disturbance and about the same as Adam's at the high disturbance the
adaptive rule actually selects, which is consistent with the rho 0.1 row in
the per-rate table showing a near-0 or even positive delta there (+0.015 at
5%, +0.038 at 10%) while rho 0.15 and 0.2 (not read on this ladder) show the
large losses. The residual-dropout loss is concentrated at higher rho, not
visible in this rho-0.1 slice of the ladder.

## Part 4: per attack

Pooling every poison rate and dataset within 1 attack (`by_attack` in the
JSON), the token-mask delta separates cleanly by how firm the attack's
shortcut is:

| Attack | rho 0.05 | rho 0.1 | rho 0.15 | rho 0.2 |
|---|---|---|---|---|
| BadNets | -0.002 [-0.010, +0.006] | +0.006 [+0.003, +0.009] | -0.003 [-0.009, +0.004] | -0.041 [-0.110, +0.001] |
| Blend | +0.006 [-0.016, +0.029] | +0.009 [-0.020, +0.044] | -0.006 [-0.061, +0.048] | +0.032 [-0.001, +0.065] |
| LF | +0.003 [-0.009, +0.016] | -0.013 [-0.043, +0.011] | -0.062 [-0.156, +0.009] | -0.033 [-0.109, +0.012] |
| BPP | +0.039 [-0.026, +0.103] | +0.054 [+0.008, +0.108] | +0.040 [+0.007, +0.085] | +0.041 [-0.029, +0.105] |
| WaNet | n=1 +0.499 | n=2 +0.265 | n=2 +0.232 | n=2 +0.279 |

BadNets, Blend and LF, the firm local or global triggers, all sit within
[-0.06, +0.04] of 0 with intervals that straddle it at every rho except
BadNets rho 0.1 (a barely positive +0.006 to +0.009). BPP, the diffuse
image-quantisation trigger, gains at every rho and its interval excludes 0 at
rho 0.1 and 0.15. WaNet gains the most by far, but only 1 or 2 matched cells
back every rho (GTSRB and Tiny WaNet cells have not reached both placements on
both sides yet), so its size cannot be trusted the way BPP's can. Both
diffuse-trigger attacks move in the same direction and the firm-trigger
attacks do not, which is the pattern the SAM paper's own account (SAM
amplifies backdoor neurons) would predict if the effect is attack-dependent
rather than uniform.

## Answer

SAM's small token-mask AUROC gain on ViT is not a fixed effect of the
optimizer: it is close to 0 and sign-unstable at 1% poisoning (7 matched
cells, deltas from -0.052 to +0.010, every interval crossing 0), only
occasionally significant at 5% (+0.029 [+0.002, +0.071] at rho 0.05, but
-0.026 at rho 0.15), and becomes consistently positive only at 10% (+0.065 to
+0.074, 3 of 4 intervals excluding 0). Splitting the same matched cells at rho
0.1 by clean-validation shift ratio shows SAM's token-mask curve sitting above
Adam's at every disturbance level including the one the adaptive rule selects
(0.8-0.9 bin: 0.867 against 0.927), so the part of the gain that does appear
is genuine separation and not only the rule reaching a different nominal rate,
even though the rule does reach a different rate too (mean selected rate at
10%, rho 0.2: Adam 0.550, SAM 0.690). The residual-dropout placement never
gains at 1% or 5% and its worst interval ([-0.379, -0.002] at rho 0.15, 5%
poisoning) excludes 0 entirely in the losing direction, while the same
placement's rho-0.1 ladder shows its curve roughly tied with Adam's at the
disturbance the rule actually selects, so its loss is concentrated at the
higher rhos this narrower slice does not cover rather than present everywhere.
Per attack, the gain concentrates in BPP and WaNet, the diffuse triggers, and
is near 0 or negative for BadNets, Blend and LF, the firm ones, matching the
SAM paper's own backdoor-neuron-amplification account only for the subset of
attacks whose shortcut is diffuse rather than as a property of SAM training in
general.

## Reproduce

```bash
PYTHONPATH=. python experiments/sam_low_rate/measure.py
PYTHONPATH=. python scripts/paper/tab_sam_low_rate.py --paper-dir paper
```
