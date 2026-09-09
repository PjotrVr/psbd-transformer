# H49 — The backdoor lives in its own patches until layer 7 and in CLS after layer 9, and attention is what moves it

**Status: SUPPORTED, causally. This is the account of why perturbing the attention input works.**

## The decomposition is exact

The residual stream is linear, `h^(l+1) = h^l + a^l + m^l`, so the backdoor direction
`d = mean(backdoor) - mean(clean)` splits exactly into per-sublayer writes,
`d_h^(l+1) = d_h^l + d_a^l + d_m^l`. No approximation is involved.

## Three phases (gtsrb badnet at 10%, n=2388 paired)

**Layers 1 to 4, dormant.** CLS attention to the trigger's 4 tokens is 0.0158 against 0.0157
clean, indistinguishable. The direction is 5 to 24 percent of its final size and nearly
orthogonal to it (cos 0.03 to 0.08).

**Layers 5 to 8, routing.** CLS attention on those 4 of 196 tokens goes 0.18 to **0.65**
(clean 0.008 to 0.002), attention writing peaks (0.088, 0.183, 0.155, 0.117), and the
direction reaches its **largest share of the stream, 0.44**. Attention entropy drops 0.81 at
layer 8.

**Layers 9 to 12, consolidation.** Attention stays hijacked, peaking at **0.931 on 2% of the
tokens** at layer 12 against 0.0003 clean. The MLP amplifies (write 0.084 to 0.315, alignment
to final 0.59 to 0.69). The direction's share of the stream declines 0.41 to 0.27 as the stream
itself grows.

## The MLP writes more; only attention can move anything

The MLP writes **1.64x** more raw norm and contributes **3.36x** more aligned with the final
direction. But the MLP is token-wise: it can amplify what a token already holds and can never
move content between tokens. Only attention can carry a corner patch to the CLS token the
classifier reads. **Routing is the step with no substitute**, and `before_attention_norm` is
the tensor attention reads.

## Causal confirmation

Activation patching with paired clean and triggered images, which is the symmetric
counterfactual Zhang and Nanda (ICLR 2024) recommend over Gaussian corruption and which this
setting provides for free. Normalised recovery of the clean logit difference:

| layer | patch the 4 trigger tokens | patch CLS | patch random, same count |
|---:|---:|---:|---:|
| 1 to 6 | **1.000** | ~0.000 | **0.000** |
| 7 | 0.932 | 0.003 | 0.000 |
| 8 | 0.681 | 0.055 | 0.000 |
| 9 | 0.412 | 0.250 | 0.000 |
| 12 | 0.188 | **0.621** | 0.000 |

Before layer 7 the backdoor is causally located **entirely** in its own patches: patching them
alone fully restores the clean answer. Between 7 and 9 it transfers to CLS, crossing over at 9
to 10. The random-token null is exactly 0.000 at every layer.

## What it predicts, and what holds

- **Depth bands.** Blocks 5 to 8 is where routing happens and where the direction is largest
  relative to the stream, and it is the best band for residual dropout (+0.054, CI [+0.030,
  +0.081]). Blocks 1 to 4 is before routing, nothing to disrupt, and is the worst (-0.062, CI
  [-0.082, -0.042]).
- **Banding is operator-specific.** It helps `pre_residual` and HURTS
  `before_attention_norm_token_mask` (5-8 at -0.062, 9-12 at -0.060). "A band beats the full
  stack" is a property of residual-adjacent dropout, not of depth.
- **Trigger class.** A trigger needing routing is disruptable by token removal; one already in
  every token is not. Against H4's CLS-onset depths (blend 5, bpp 6, lf 8, badnet 9) the
  token-mask advantage over channel masking is **perfectly monotone, r = +0.942** (n=4 attacks),
  stronger than against measured trigger footprint (+0.845).
- **Masking response is a step function.** With mask draws balanced between hitting and missing
  the trigger, backdoor confidence moves **0.0952** when a draw covers the trigger and
  **-0.0000** when it does not, against 0.0076 and 0.0046 on clean.

## Scope and prior art

The onset ordering and the "distributed early, consolidated at CLS mid-to-late" story are
already in the companion paper (`papers/backdoor_directions/`, arXiv:2603.10806) in
residual-stream direction language, on the same architecture. What is added here is the **exact
per-sublayer decomposition**, the token-mixing argument, and the **causal patching**, none of
which that paper does.

Correlations against footprint report n at the ATTACK level (4 to 8), not the cell level;
footprint is constant within an attack.

`experiments/residual_stream_mechanism/`: `residual_decomposition.py`, `cls_routing.py`,
`activation_patching.py`, `sink_hit_confound.py`.
