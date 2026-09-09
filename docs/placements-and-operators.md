# Placements and operators: what we tested, what we found, and how it maps to Swin

A **placement** in this project is two independent choices, and the cache directory name is
just the two concatenated:

    <position>_<operator>          e.g.  before_attention_residual_token_mask

- **position** — *where* in the network the probe is attached
- **operator** — *what* the probe does to the activation there

Both are swept independently. `before_attention_norm_token_mask` is `token_mask` applied at
`before_attention_norm`. A bare name like `pre_residual` means the default operator,
`nn.Dropout`.

A third choice is implicit and is the one most people miss:

- **block range** — *which* transformer blocks get a probe. The default is **every block**.

## How a probe is attached

`plug_dropout` reads a `PositionSpec` table and attaches a **fresh** probe module at each
named submodule boundary, by forward pre-hook or forward hook, then `unplug_dropout` removes
it by handle. It never toggles a dropout the model already contains, because a trained
dropout's inverted-scaling factor was calibrated against the next layer's weights, so
switching it back on at inference would conflate the model's own regularisation with the
probe.

Two positions cannot be a hook at all, because the tensor they perturb is a local variable
that never crosses a module boundary. `after_attention_residual` (the stream between the
attention add and its two consumers) and `attention_heads` (per-head outputs inside
`F.multi_head_attention_forward`) use a removable per-instance forward wrapper instead.

---

# The positions

## The 9 atomic positions, in forward order through one block

These are the placement study proper. Each is swept in isolation, at every block.

| position | ViT module | what it perturbs |
|---|---|---|
| `after_embedding` | `encoder.dropout` (pre) | the patch embeddings, once, before the block stack |
| `before_attention_norm` | `ln_1` (pre) | the attention branch's input, skip untouched |
| `before_attention` | `self_attention` (pre) | the normalized attention input |
| `before_attention_residual` | `dropout` (post) | the attention branch's **output**, just before it is added |
| `after_attention_residual` | the stream (wrapper) | the residual stream **after** `x = x + attn`, so both the `ln_2` branch and the skip see it |
| `before_mlp_norm` | `ln_2` (pre) | the MLP branch's input, skip untouched |
| `before_mlp` | `mlp` (pre) | the normalized MLP input |
| `before_mlp_residual` | `mlp` (post) | the MLP branch's **output**, just before it is added |
| `after_mlp_residual` | the stream (wrapper) | the residual stream after the MLP add |

The distinction that the whole study turns on is **branch versus stream**.
`before_mlp_residual` perturbs only the MLP's contribution; `after_attention_residual`
perturbs the stream itself, so the perturbation propagates through every later block. That
is the PSBD paper's own ConvNet placement.

## 2 structured positions

Kept out of the 9 because the channel axis there indexes a *transformer unit* rather than an
arbitrary feature, so a structured operator masks something meaningful.

| position | module | one "channel" is |
|---|---|---|
| `attention_heads` | inside `self_attention` | one attention head |
| `mlp_neurons` | `mlp.3` (pre), the 3072-dim post-GELU tensor | one MLP hidden neuron |

## 4 ported positions

These exist to host ports of published detectors, not for the placement study.

| position | module | hosts |
|---|---|---|
| `attention_norm_out` | `ln_1` (post) | IBD-PSC's LayerNorm amplification |
| `mlp_norm_out` | `ln_2` (post) | same |
| `final_norm_out` | `encoder.ln` (post) | same, at the readout |
| `input_pixels` | model input (pre) | SCALE-UP's pixel amplification |

Note `attention_norm_out` is a **post**-hook on `ln_1` while `before_attention_norm` is a
**pre**-hook on the same module. They perturb the output and the input of the same layer and
are not the same placement.

## 3 named combinations

| name | positions | meaning |
|---|---|---|
| `pre_residual` | `before_attention_residual` + `before_mlp_residual` | perturb each branch's contribution; the stream itself is never touched |
| `post_residual` | `after_attention_residual` + `after_mlp_residual` | perturb the stream after each add. **The PSBD paper's ConvNet placement** |
| `both_sublayer_inputs` | `before_attention_norm` + `before_mlp_norm` | perturb both branch inputs |

`pre_residual` versus `post_residual` is the comparison this project was founded on. It was
**refuted**: measured at matched shift ratio the gap is +0.002, indistinguishable from noise
(H1, H20).

---

# The operators

| operator | what it does | verdict |
|---|---|---|
| `dropout` | `nn.Dropout`, element-wise, independent per token | the baseline, and still the best on average (0.859 over 773 cells) |
| `token_mask` | zeroes **whole tokens**, all channels. CLS is never masked, since dropping the classifier's only read point destroys the prediction instead of perturbing it | 0.854. Best on patch triggers, worst on warps (H27) |
| `channel_mask` | zeroes whole channel groups, **shared across tokens**, so a neuron is either alive for the sample or not | 0.792. **Refuted** as an improvement on dropout (H26) |
| `head_mask` | zeroes whole attention heads, only meaningful at `attention_heads` | 0.835, but heads are redundant and the backdoor is not in them (H22, H35) |
| `droppath` | zeroes a whole branch output, stochastic depth. The residual-native perturbation | 0.775. **Confirmed to lose**, including the prediction that it would (H21) |
| `gaussian` | additive noise at the activation's own scale, **removes no capacity** | 0.788. The control that refuted the capacity-removal account (H23) |
| `rademacher` | additive ±1 noise, same covariance as gaussian, lower estimator variance | **implemented, never swept.** The one theoretically motivated operator still unmeasured |
| `gain_scale` | amplifies a LayerNorm's output. IBD-PSC's perturbation, exact for LayerNorm since scaling gamma and beta together *is* scaling the output | 0.664, worst. **Deterministic**, so k=1 is exact |
| `scale_up` | amplifies input pixels. SCALE-UP's perturbation | 0.789. **Deterministic** |

`gain_scale` and `scale_up` are deterministic: every Monte Carlo pass returns the same value,
so PSU is exact at k=1 and a k>1 sweep writes k identical rows. Their shift ratio is a
per-sample flip indicator, not a fraction of passes, which matters whenever sigma is compared
across operators.

---

# Do you need every layer, or only some?

**No, and a band beats the full stack.** `pre_residual` swept over all 12 blocks against the
same positions restricted to a 4-block band, on the 44 ViT cells carrying all four:

| block range | mean AUROC |
|---|---|
| blocks 1–4 (early) | 0.811 |
| **all 12 blocks** | **0.881** |
| blocks 9–12 (late) | 0.898 |
| **blocks 5–8 (middle)** | **0.909** |

Early blocks are clearly worse. A middle or late band is **better than perturbing
everything**, which is not obvious: adding probes to the early blocks actively costs
detection. This is consistent with H30's finding that the backdoor direction crystallizes at
layers 8–10, and with H10's band premise.

**But blocks 5–8 must not be deployed**, and this is the important caveat. Ranked within
poison rate it is 1st at 5% and 10% and **8th at 1%** (0.872). A defender may guess the
attack but can never know the poison rate, so a configuration whose ranking depends on it is
not a usable defence. The full-stack default is rate-stable; the band is not.

---

# How this translates to Swin

The position **names are shared** — `POSITION_REGISTRY` keys on architecture and every name
resolves to the analogous Swin module — so every sweep, table and comparison is written once
and runs on both.

| position | ViT module | Swin module |
|---|---|---|
| `after_embedding` | `encoder.dropout` | `features.0` (patch embed), post |
| `before_attention_norm` | `ln_1` | `norm1` |
| `before_attention` | `self_attention` | `attn` |
| `before_attention_residual` | `dropout`, post | **`attn`, post** |
| `before_mlp_norm` | `ln_2` | `norm2` |
| `before_mlp` / `before_mlp_residual` | `mlp` | `mlp` |
| `attention_norm_out` / `mlp_norm_out` | `ln_1` / `ln_2`, post | `norm1` / `norm2`, post |
| `final_norm_out` | `encoder.ln` | `norm` |

**The one genuine asymmetry** is `before_attention_residual`. Swin has a single
`stochastic_depth` instance called **twice per block**
(`x + stochastic_depth(attn(...))`, then `x + stochastic_depth(mlp(...))`), so a hook on it
cannot tell which branch invoked it. We hook `attn` and `mlp` directly instead, which is
unambiguous at the cost of landing just *before* stochastic depth sees the branch output
rather than just after. `stochastic_depth` itself is never touched. Swin also has **24
blocks** to ViT's 12, so its bands are 1–8, 9–16, 17–24.

## What Swin actually measures

45 implanted Swin cells with sweeps:

| configuration | n | AUROC |
|---|---|---|
| `before_attention` (dropout) | 14 | **0.931** |
| `before_attention_norm_token_mask` | 13 | **0.913** |
| `after_embedding` | 14 | 0.896 |
| **`pre_residual_blocks_17_24`** (late band) | 35 | **0.889** |
| `pre_residual` (all 24 blocks) | 42 | 0.847 |
| `pre_residual_blocks_9_16` | 35 | 0.839 |
| `pre_residual_blocks_1_8` (early band) | 35 | 0.847 |

**The depth-band finding replicates on Swin**: the late band (0.889) beats the full stack
(0.847) by the same margin and in the same direction as ViT, and the early band is not
better. Two architectures, different block counts, same conclusion — perturbing every block
is not the right default, and the useful depth is late-middle.

The deployed configuration reads **0.913** on Swin against 0.869 on ViT, so nothing about the
port is ViT-specific in a way that breaks.

**Caveat on all Swin numbers:** coverage is uneven (n = 13 to 42 depending on the
configuration) and every Swin comparison here is unbalanced. This project has already had 4
of 6 comparisons invert when rebalanced, so treat the Swin ordering as indicative, not
settled.
