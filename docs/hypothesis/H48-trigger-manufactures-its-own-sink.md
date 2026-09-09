# H48 — The trigger manufactures an attention sink this ViT does not natively have, and does not hijack one

**Status: SUPPORTED on three independent measurements. The candidate defence that follows from
the alternative hypothesis FAILS, as predicted.**

## Context

Large self-supervised ViTs spontaneously produce high-norm "artifact" or register tokens that
attract CLS attention (Darcet et al., ICLR 2024; Sun et al., arXiv:2402.17762). If such a
population exists here it is a confound: a detector that masks tokens at random might be
measuring whether the draw hit an artifact token. If it does not exist but the trigger creates
one, that is a mechanism.

## This model has no native sink structure

| criterion | threshold | measured on the benign checkpoint |
|---|---|---|
| Sun et al., scalars | magnitude > 100 AND >= ~1000x layer median | max ratio **73x**, zero scalars qualify |
| Darcet et al., tokens | bimodal norm distribution, ~2% outliers | unimodal, max **1.4x** median, **0.000%** above 2x |

Layer-12 patch-norm quantiles on the benign model: 29.0 / 33.5 / 37.6 / 39.6 / 41.0. The
registers reported in CLIP and DINOv2 are absent. **To our knowledge supervised ImageNet
ViT-B/16 has not been characterised before; prior work covers CLIP, DINOv2, DeiT-III and MAE.**

That settles the confound. A masking detector cannot be reading an artifact-token population
because there is not one.

## The trigger creates one, and it is not the existing machinery

| measurement | result |
|---|---|
| max/median patch norm, layer 6 | **3.38** triggered against **1.29** clean |
| tokens above 2x median, layer 6 | **0.81%** triggered against **0.00%** clean |
| highest-norm patch token is a trigger token, layers 2 to 10 | **100%**, against 0% by layer 7 on the benign model |
| trigger-token norm rank | **0.97** at layer 2 decaying to **0.43** at layer 12 |

The sink is built early, read by attention at layers 5 to 8, and discarded.

**It is a parallel mechanism, not a hijack.** Three measurements agree:

1. **Register neurons are disjoint.** Following Jiang et al. (arXiv:2506.08010), the neurons
   driving the trigger's tokens share **0 of the top 10** and **0 of the top 50** with those
   driving naturally high-norm tokens; 4 of 200 against 1.09 expected. The clean set peaks at
   layer 11, the trigger's at layer 5.
2. **The role is not replaceable.** Lu et al. (arXiv:2507.16018) show a natural sink hands off
   to a dormant token when masked. Here the concentration is restored at layers 1 to 4
   (takeover 0.85 to 1.01) and **collapses at layers 5 to 12 (0.13 to 0.36)**.
3. **Spare capacity is ignored.** Appending up to 64 test-time register tokens leaves ASR at
   **1.000**, clean accuracy at 0.973 and CLS attention on the trigger at 0.370 against 0.373.

Measurement 3 is a candidate defence and it **fails**, which is the null the mechanism
predicts: there is no sink competition to redirect.

## What the trigger token is made of

Value norms are **ordinary**, 0.97 to 1.26 times other tokens, so it is not a "broadcast" sink
in the sense of Fesser et al. (arXiv:2606.08105). The naturally most-attended token sits
**below** ordinary tokens at mid depth (1.34 against 2.66), which is their no-op signature. The
trigger is a third category: **ordinary payload receiving extraordinary attention**.

## How it differs by trigger family

| class | attack | trigger-token norm rank | peak ratio vs clean | detector AUROC |
|---|---|---:|---|---:|
| local | lc / badnet / tact | **0.92 / 0.85 / 0.77** | 2.81 / 3.38 / 1.78 vs ~1.4 | **1.000** |
| global | lf / blend / sig / bpp | 0.50 (median) | 2.07 / 1.77 / 1.65 / 1.83 vs ~1.3 | 0.898 to 1.000 |
| geometric | **wanet** | 0.500 | **2.09 vs clean 2.05** | **0.756** |
| control | benign | -- | 1.50 vs clean 1.50 | **0.540** |

Only local triggers elevate their own tokens. Global triggers sit at exactly median rank and
build no sink, yet remain detectable through a small global norm shift. WaNet has **no
differential at all**, which is why it is the hard case here as it is for PSBD.

## Open

The detector claim (max over median patch norm, one forward pass, no trigger knowledge) is
measured on single cells and is under audit. Layers 1 and 2 must be excluded: the benign
control reads 0.69 to 0.76 there because the patch is visible in raw pixels before the network
does anything with it. 197 tokens is the low end of the range where sinks have been studied, so
the attention shares may be sequence-length dependent; unmeasured.

`experiments/residual_stream_mechanism/`: `artifact_tokens.py`, `register_neurons.py`,
`sink_anatomy.py`, `test_time_registers.py`.
