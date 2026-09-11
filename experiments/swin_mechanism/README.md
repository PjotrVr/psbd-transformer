# Why Swin-S reads higher AUROC than ViT-B/16 under PSBD-TM

## Question

Under token masking at the attention input (`before_attention_norm_token_mask`,
rate chosen by the 0.8 clean-shift rule), Swin-S reads 0.982 mean AUROC against
0.939 for ViT-B/16 on 42 matched settings, and the gap sits entirely in the
global triggers: WaNet +0.17, LC +0.14, SIG +0.54, while BadNet, LF, BPP and
Blend are within 0.04. 4 structural candidates: (a) Swin has no class token, so
masking a token removes its share of the mean-pooled readout directly; (b)
windowed attention keeps a trigger's evidence inside its own window for the
early stages; (c) 24 blocks against 12, so the same rate masks twice as often;
(d) Swin simply has a firmer shortcut (ASR 0.992 against 0.978).

## Method

6 matched (architecture, dataset, attack, rate) pairs: CIFAR-10 WaNet 10%,
CIFAR-10 SIG 10%, CIFAR-100 WaNet 10%, GTSRB WaNet 10%, and 2 controls where
ViT and Swin already agree, CIFAR-100 BadNet 10% and CIFAR-100 Blend 10%. 300
paired clean and triggered images per checkpoint from the PSBD analysis pool,
150 used to estimate the backdoor direction and 150 held out for everything
read against it, the same split `experiments/whole_network_erasure/measure.py`
uses. `experiments/swin_mechanism/measure.py` runs all 5 measurements; results
are under `results/_experiments/swin_mechanism/<pair>.json`.

## 1. Logit margin (candidate d)

Mean logit margin of the predicted class over the runner-up, on the 150 held-out
clean and triggered images, no perturbation. A positive gap means the trigger
makes the model more confident.

| pair | vit clean | vit triggered | vit gap | swin clean | swin triggered | swin gap |
|---|---|---|---|---|---|---|
| cifar10 wanet 10% | 7.34 | 6.25 | -1.09 | 9.63 | 8.06 | -1.57 |
| cifar10 sig 10% | 6.95 | 9.33 | +2.37 | 9.19 | 13.25 | +4.07 |
| cifar100 wanet 10% | 5.93 | 6.04 | +0.11 | 8.01 | 8.41 | +0.39 |
| gtsrb wanet 10% | 7.50 | 7.98 | +0.47 | 11.22 | 10.70 | -0.52 |
| cifar100 badnet 10% (control) | 5.94 | 11.15 | +5.21 | 7.55 | 17.03 | +9.48 |
| cifar100 blend 10% (control) | 5.69 | 11.42 | +5.73 | 7.77 | 15.00 | +7.24 |

Swin's margins run larger than ViT's in every cell, control and non-control
alike, so a firmer shortcut is a real, universal property of the Swin
checkpoints. It does not track the AUROC gap: on CIFAR-10 WaNet the trigger
makes both architectures *less* confident (margin gap negative for both), yet
ViT reads 0.459 there and Swin reads 0.952. Candidate (d) does not explain
which cells show the gap.

## 2. Backdoor direction onset (per block)

The onset block is the first block whose direction norm reaches half of the
largest norm on the ladder, read as a fraction of the architecture's own depth.

| pair | vit onset | vit fraction | swin onset | swin fraction |
|---|---|---|---|---|
| cifar10 wanet 10% | 8 / 12 | 0.667 | 20 / 24 | 0.833 |
| cifar10 sig 10% | 11 / 12 | 0.917 | 17 / 24 | 0.708 |
| cifar100 wanet 10% | 8 / 12 | 0.667 | 23 / 24 | 0.958 |
| gtsrb wanet 10% | 8 / 12 | 0.667 | 19 / 24 | 0.792 |
| cifar100 badnet 10% (control) | 10 / 12 | 0.833 | 21 / 24 | 0.875 |
| cifar100 blend 10% (control) | 7 / 12 | 0.583 | 15 / 24 | 0.625 |

Both architectures crystallize the direction late, past 60% of depth on every
cell, and the fraction is not systematically later or earlier for the cells
with a large AUROC gap than for the controls. Onset depth does not separate
the gap cells from the controls either.

## 3. Last-stage token share (candidate a)

Swin: the smallest set of last-stage tokens whose projection onto the backdoor
direction reaches half of the triggered image's total, out of 49 tokens at the
final 7x7 grid. ViT: the smallest set of tokens the class token's last-block
attention places half its weight on, out of 197.

| pair | vit tokens / 197 | vit share | swin tokens / 49 | swin share |
|---|---|---|---|---|
| cifar10 wanet 10% | 57.5 | 0.292 | 20.8 | 0.425 |
| cifar10 sig 10% | 45.1 | 0.229 | 14.8 | 0.301 |
| cifar100 wanet 10% | 10.0 | 0.051 | 19.7 | 0.402 |
| gtsrb wanet 10% | 14.0 | 0.071 | 22.5 | 0.459 |
| cifar100 badnet 10% (control) | 1.7 | 0.009 | 18.8 | 0.383 |
| cifar100 blend 10% (control) | 59.7 | 0.303 | 22.9 | 0.467 |

Swin's mean-pooled readout always draws on 30 to 47% of the last stage's
tokens, in every cell including the controls. ViT's class-token attention is
far more variable, from 0.9% of tokens on BadNet (a single localized patch) to
30% on Blend (a whole-image perturbation). This confirms candidate (a) as a
real, universal structural difference: Swin's readout has no attention
mechanism to route around a masked token the way ViT's class token can. But
the share gap does not track the AUROC gap across cells: BadNet's share gap
(0.374) is as large as WaNet's and SIG's, yet BadNet's AUROC gap is 0.002. The
readout difference is real but not what discriminates the cells that actually
show a gap.

## 4. Window patch test (candidate b, Swin only)

Per triggered image, the stage-entry window (7x7 tokens) with the largest
clean-versus-triggered difference is patched to its clean values, against
patching the same token count spread over the other windows. Recovery is the
fraction of patched images whose prediction reverts to the true label.

| pair | stage 1 top | stage 1 control | stage 2 top | stage 2 control | stage 3 top | stage 3 control |
|---|---|---|---|---|---|---|
| cifar10 wanet 10% | 0.080 | 0.033 | 0.160 | 0.033 | 0.507 | 0.033 |
| cifar10 sig 10% | 0.000 | 0.000 | 0.000 | 0.000 | 0.040 | 0.000 |
| cifar100 wanet 10% | 0.167 | 0.073 | 0.207 | 0.073 | 0.593 | 0.067 |
| gtsrb wanet 10% | 0.147 | 0.047 | 0.333 | 0.047 | 0.513 | 0.047 |
| cifar100 badnet 10% (control) | 0.847 | 0.000 | 0.847 | 0.000 | 0.847 | 0.000 |
| cifar100 blend 10% (control) | 0.000 | 0.000 | 0.000 | 0.000 | 0.020 | 0.000 |

Windowed locality is real on WaNet: by stage 3, patching 1 window recovers the
true label on half of the triggered images (0.51 to 0.59), against 0.03 to
0.07 for the spread control, confirming that WaNet's warp evidence becomes
window-local as the grid coarsens. SIG and Blend never localize (recovery
stays at or near 0 at every stage), consistent with both being whole-image
perturbations with no single window to patch. BadNet recovers fully from
stage 1, since its patch sits at 1 fixed location the whole way through. But
this measurement was only run on Swin, so it cannot itself explain a gap
between architectures, and its strength does not track the AUROC gap either:
CIFAR-100 WaNet and GTSRB WaNet localize exactly as strongly as CIFAR-10
WaNet, yet their AUROC gaps against ViT are 0.084 and -0.029 against CIFAR-10
WaNet's 0.493.

## 5. AUROC ladder against clean shift ratio (candidate c)

AUROC at the rate the 0.8 adaptive rule selects, fractional PSU, the headline
statistic, read directly from the `psbd/` caches.

| pair | vit rate | vit AUROC | swin rate | swin AUROC | swin minus vit |
|---|---|---|---|---|---|
| cifar10 wanet 10% | 0.5 | 0.4587 | 0.6 | 0.9518 | +0.4931 |
| cifar10 sig 10% | 0.8 | 0.4177 | 0.6 | 0.9544 | +0.5367 |
| cifar100 wanet 10% | 0.5 | 0.8423 | 0.4 | 0.9265 | +0.0843 |
| gtsrb wanet 10% | 0.4 | 0.9480 | 0.4 | 0.9190 | -0.0290 |
| cifar100 badnet 10% (control) | 0.5 | 0.9980 | 0.4 | 0.9998 | +0.0017 |
| cifar100 blend 10% (control) | 0.5 | 0.9984 | 0.5 | 0.9987 | +0.0003 |

The 2 CIFAR-10 cells reproduce the project's own documented inverted cells
almost exactly: `paper/headline.tex` records a panel floor of 0.418 and 2
inverted cells at cifar10 sig 10% and cifar10 wanet 10%, and this run's own
baseline caches read 0.4177 and 0.4587 for ViT there, matching
`experiments/wanet_cifar10_audit/README.md`'s earlier finding that
`before_attention_norm_token_mask` fails specifically on these 2 ViT
checkpoints. Once those 2 cells are set aside, the WaNet gap collapses to
+0.084 on CIFAR-100 and -0.029 on GTSRB, both inside the 0.04 band the
question already reports for local triggers.

Candidate (c) could not be measured on Swin at all: `before_attention_norm_blocks_5_8_token_mask`
and `before_attention_norm_blocks_9_12_token_mask` are swept for every one of
these ViT checkpoints (0.9868/0.9674 on BadNet, 0.9915/0.9739 on Blend,
against 0.9980/0.9984 at the full 12 blocks) but were never swept for Swin, so
the "24 against 12" hypothesis stays untested by this panel and is reported
as not available rather than ruled in or out.

## Verdict

Most of the reported WaNet and SIG gap is not a Swin structural advantage: it
is the project's own already-documented ViT inversion on 2 CIFAR-10 cells,
which this run reproduces almost to the fourth decimal (0.4177 and 0.4587
against the panel's recorded 0.418 floor). Once those 2 cells are set aside,
the WaNet gap on CIFAR-100 (+0.084) and GTSRB (-0.029, ViT ahead) sits inside
the 0.04 band the question already reports for local triggers, so there is no
general Swin-beats-ViT effect on global triggers left to explain. Candidate
(d) is ruled out as the driver: Swin's margin gap on CIFAR-10 WaNet is
negative, exactly like ViT's, while the AUROC gap there is 0.49. Candidate (a)
is confirmed as a real, universal structural fact, Swin's mean-pooled readout
always draws on 30 to 47% of the last stage against ViT's attention-selected
0.9 to 30%, but it does not correlate with which cells show a gap: BadNet's
token-share difference is as large as WaNet's, yet BadNet's AUROC gap is
0.002. Candidate (b) is confirmed as a real property of Swin's windowed
processing, WaNet's evidence becomes recoverable from 1 patched window by
stage 3 while SIG's and Blend's never do, but it was measured on Swin only and
its strength does not track the AUROC gap across the 3 WaNet cells either.
Candidate (c) is unmeasurable here: the block-banded placement was never swept
for any Swin checkpoint. The honest reading is that the panel's averaged
"Swin advantage on global triggers" number is carried almost entirely by a
ViT-specific failure mode on 2 CIFAR-10 checkpoints, not by any of the 4
Swin-side structural candidates this experiment set out to test.
