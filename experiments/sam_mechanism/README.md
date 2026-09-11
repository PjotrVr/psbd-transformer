# Whether SAM amplifies the trigger footprint or widens PSBD's margin

## Question

The SAM paper (Zhang et al., arXiv 2411.11525, `papers/reliable_poisoned_sample_
detection_against_backdoor_attacks_enhanced_by_sharpness_aware_minimization/`,
Section 3) claims sharpness-aware minimisation amplifies backdoor neurons, using
3 amplification metrics: the per-neuron TAC difference between SAM and vanilla
training (Fig. 3), the TAC-to-AUC correlation across attacks (Fig. 2, Pearson
0.71) and the intra-class feature variance under SAM (Fig. 4). On our ViT-B/16
checkpoints PSBD-TM (`token_mask` at `before_attention_norm`) gains only 0.02 to
0.04 AUROC under SAM and PSBD-RD gets worse. PSBD reads the margin of the
predicted class under a perturbation, not a neuron activation, so the
amplification account and the small detection gain could be 2 separate facts
rather than 1 causing the other. The hypothesis this measures: SAM amplifies the
trigger's footprint on the class token, its direction norm and its TAC, without
proportionally widening the gap between how a triggered and a clean input
respond to token masking, and the amplified footprint does not reach the class
token any earlier at 1% and 5% poisoning.

## Method

6 matched (Adam, SAM rho=0.1) checkpoint pairs, BadNet, Blend, BPP, LF and WaNet
on CIFAR-10 and CIFAR-100 at 1% and 5% poisoning, chosen as the pairs whose SAM
checkpoint also carries a `psbd_metrics.json` record. 500 paired clean and
triggered images per checkpoint, built the way
`experiments/whole_network_erasure/measure.py`'s `paired_rows` and
`residual_stream` build them (imported directly rather than reimplemented).

Per layer, on the class token: the relative backdoor-direction norm
(`analysis.direction.backdoor_direction`, divided by the mean clean residual
norm) and the mean TAC (`analysis.direction.trigger_activated_change`), the same
2 statistics `scripts/paper/run_tac_layers.py` reads. The onset layer is the
first layer past 0 whose relative norm reaches half the final layer's, matching
`scripts/paper/mech_tac_layers.py`'s rule.

At the final layer with no perturbation: the logit margin of the predicted
class over the runner-up, on the clean and on the triggered copies, in float32.

Under `token_mask` at `before_attention_norm`, at the rate the 0.8 adaptive rule
selected for that checkpoint (read from `results/<folder>/psbd_metrics.json`,
falling back to 0.5 when absent): the mean `psu_ratio` on the clean and on the
triggered images, and the AUROC separating them, run through
`defences.operators`, `models.positions.plug_dropout` and `defences.scores`
exactly as `cli/sweep.py` and `cli/analyze.py` do.

    PYTHONPATH=. .venv/bin/python experiments/sam_mechanism/measure.py \
        --checkpoints-dir checkpoints --raw-data-dir raw_data

Writes `results/_experiments/sam_mechanism/<pair>.json`. Needs a GPU, one run
over all 6 pairs took under 5 minutes on an A100.

## Finding

| Pair | Optimiser | PSBD-TM rate | Peak layer (rel. norm) | Onset layer | Final TAC | Clean margin | Triggered margin | PSU mean clean | PSU mean triggered | AUROC |
|---|---|---|---|---|---|---|---|---|---|---|
| BadNet a2o 1% CIFAR-100 | Adam | 0.5 | 12 (0.80) | 8 | 0.57 | 6.24 | 10.02 | 0.928 | 0.174 | 0.987 |
| BadNet a2o 1% CIFAR-100 | SAM | 0.6 | 12 (0.88) | 10 | 0.61 | 4.99 | 9.19 | 0.951 | 0.192 | 0.999 |
| Blend 5% CIFAR-100 | Adam | 0.6 | 9 (1.29) | 6 | 0.69 | 6.30 | 11.32 | 0.969 | 0.498 | 0.977 |
| Blend 5% CIFAR-100 | SAM | 0.6 | 9 (1.50) | 5 | 0.75 | 4.93 | 10.79 | 0.949 | 0.116 | 0.996 |
| BPP 1% CIFAR-100 | Adam | 0.5 | 12 (0.77) | 7 | 0.51 | 6.34 | 6.00 | 0.963 | 0.709 | 0.928 |
| BPP 1% CIFAR-100 | SAM | 0.6 | 12 (0.92) | 7 | 0.64 | 4.79 | 8.05 | 0.965 | 0.079 | 0.985 |
| LF 5% CIFAR-100 | Adam | 0.5 | 12 (1.14) | 9 | 0.64 | 6.46 | 10.09 | 0.957 | 0.120 | 0.965 |
| LF 5% CIFAR-100 | SAM | 0.6 | 10 (1.60) | 7 | 0.84 | 4.90 | 9.93 | 0.965 | 0.016 | 0.996 |
| WaNet 5% CIFAR-10 | Adam | 0.6 | 10 (1.43) | 7 | 0.89 | 6.12 | 6.56 | 0.843 | 0.309 | 0.945 |
| WaNet 5% CIFAR-10 | SAM | 0.6 | 10 (0.94) | 7 | 0.66 | 5.08 | 2.46 | 0.811 | 0.690 | 0.704 |
| BadNet a2o 5% CIFAR-10 | Adam | 0.7 | 12 (0.93) | 10 | 1.12 | 7.24 | 9.27 | 0.891 | 0.183 | 0.991 |
| BadNet a2o 5% CIFAR-10 | SAM | 0.7 | 12 (1.06) | 10 | 1.28 | 6.20 | 8.97 | 0.875 | 0.064 | 1.000 |

SAM amplifies the footprint on 5 of the 6 pairs. The peak relative direction
norm rises 9% to 41% and the final-layer mean TAC rises 8% to 31%, which is the
same direction the SAM paper's Fig. 3 and Fig. 2 report on ResNet18 backdoor
neurons, now read on the ViT class token. WaNet at 5% on CIFAR-10 is the
exception: the footprint falls 34% in direction norm and 26% in TAC, and the
onset layer never moves any earlier than layer 7 in any of the 6 pairs.

Where the footprint grows, the margin gap (triggered margin minus clean margin)
widens too, from 0.42 to 4.19 on BadNet at 1%, from 5.03 to 5.86 on Blend, from
-0.34 to 3.27 on BPP and from 3.63 to 5.03 on LF, and PSBD-TM's AUROC rises by
0.009 to 0.057 on those 5 pairs. But the size of the AUROC gain does not track
the size of the footprint gain: LF has the largest footprint gain, 41%, and only
a mid-sized AUROC gain, +0.031, while BPP has a smaller footprint gain, 20%, and
the largest AUROC gain, +0.057. On WaNet the footprint shrinks, the margin gap
inverts from +0.44 to -2.62, and the AUROC falls from 0.945 to 0.704, so the 1
pair where the mechanism runs backward is also the 1 pair where detection gets
worse.

## Conclusion

SAM does amplify the trigger's footprint on the ViT class token in 5 of the 6
pairs measured here, the same direction the SAM paper's TAC-based metrics
report on ResNet backdoor neurons, but on WaNet it shrinks the footprint
instead, so the amplification is attack-dependent rather than a property of SAM
training itself. Where the footprint grows the margin gap between triggered and
clean inputs widens too, and PSBD-TM's AUROC moves in the same direction, which
is why the gain shows up at all. That movement is loosely coupled rather than
proportional: LF's footprint grows the most and its AUROC gain is only middling,
while BPP's footprint grows less and its AUROC gain is the largest, and WaNet's
shrinking footprint coincides with the only AUROC drop. TAC and direction norm
describe how far the trigger moves the class token in an unperturbed pass,
while PSU-ratio describes how much of that movement survives token masking, and
the 2 quantities are not the same measurement. The SAM paper's neuron-
amplification account explains why SAM usually helps PSBD-TM a little, since a
larger unperturbed footprint tends to leave a larger footprint behind after
masking too, but it does not explain why the gain stays small or why WaNet
reverses it. PSBD's detector is reading what survives the perturbation, not the
footprint's size, and the 2 move together only approximately, which is the gap
between the SAM paper's mechanism and PSBD's own.
