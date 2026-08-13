# H22 — Attention heads are the transformer's own unit, and masking them is the ViT-native PSBD

**Status: PRE-REGISTERED.** Written before the pilot results land. Jobs
`psbd_pilot_001/002`, submitted 2026-08-14 at commit `b37ceaf`. One early
data point exists and is noted below.

## Mechanism

ViT-B/16 has 12 heads per block over 12 blocks, so 144 heads, each a
64-dimensional slice of the 768-wide concatenation that feeds `out_proj`.
`head_mask` removes whole heads for a sample.

Reaching the head axis needed a forward wrapper rather than a hook.
`nn.MultiheadAttention` runs `F.multi_head_attention_forward`, which reads
`out_proj.weight` directly and never calls `out_proj` as a module, so a forward
hook on `out_proj` never fires and the model output comes back bit-identical.
This was verified the hard way. The wrapper recomputes attention and exposes the
per-head tensor; at rate 0 it reproduces PyTorch to 3.6e-07 on logits of scale
0.79, which is float32 associativity.

## Why it might work

This is the operator the mentor's framing points at: a transformer's natural unit
is the head, not the channel, and PSBD's dropout is aligned with neither.

- Heads are **functionally specialised**, so a backdoor implemented as "attend to
  the trigger patch, write the target direction" plausibly lives in a small number
  of heads. Removing one of them should collapse the shortcut while leaving clean
  predictions, which distribute over many heads, intact.
- 144 units is a good granularity: fine enough to be selective, coarse enough that
  random masking hits a relevant head often (at p = 0.3, roughly 43 heads go).
- It makes the perturbation **interpretable**. A head index is something you can
  point at in a paper, unlike a channel index in a 768-dim residual stream.

## Why it might fail

[H16](H16-where-the-backdoor-neurons-are.md) supplies two specific reasons, and
they are serious:

- The backdoor lives in **residual-stream dimensions**, not demonstrably in heads.
  H16 measured direction norms and TAC per dimension of the stream, and the MLP
  writes to that stream too. If the backdoor is written mostly by MLPs, head
  masking probes the wrong subsystem entirely.
- H16 **refuted the head-alignment localizer**: the Karayalcin Z > 3 rule scored
  0 of 15 true positives and 1 of 3 false positives on these very checkpoints. If
  heads carried the backdoor cleanly, that rule should have worked.

Additionally, random head masking is not head *selection*. Even if a few heads
carry the backdoor, hitting them at random dilutes the signal by 144/k, which is
what [H18](H18-sensitivity-profile-over-units.md)'s per-unit profile is meant to
address and this operator is not.

## Prediction

`head_mask` at matched sigma **beats `dropout` on the patch-trigger attacks**
(`badnet_a2o`, `badnet_a2a`) where attention to a specific spatial location is
most obviously head-mediated, and is **comparable elsewhere**.

An early smoke measurement is consistent but far too small to lean on: on
`vit_cifar10_badnet_a2o_0_01` with only 300 samples and 2 rates, `head_mask`
scored 0.633 against the published configuration's 0.297 on the full split. That
is 1 checkpoint, 300 samples, and no matched-sigma control, so it is a reason to
run the pilot, not a result.

## What would refute it

- `head_mask` within +/-0.02 of `dropout` at matched sigma across the pilot: the
  head is not a privileged unit for this problem, and H16's refutation of the
  head-alignment localizer extends to head perturbation.
- It wins uniformly across all attacks including `blend` and `wanet`: then it is
  not exploiting attention specialisation, it is just another capacity removal,
  and the cheaper `channel_mask` should be preferred.
- Benign control departs from 0.5: the Python attention path differs from the
  fused one in some way the rate-0 check did not catch.

## Reproduce

    python pbs/generate_perturbation_jobs.py --stage pilot --operator head_mask
    python psbd_analyze.py --all

Folder `attention_heads_head_mask`. `scratch/test_perturbations.py` covers the
operator's mechanics and must stay at ALL PASS.
