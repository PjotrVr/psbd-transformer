# H21 — DropPath is the residual-native perturbation and should beat activation noise

**Status: CONFIRMED, including the part that predicted it would lose.**

## Result

`droppath` / `before_attention_residual` scores 0.860 at 10% poisoning (matched
sigma 0.6, mean over 8 backdoored attacks), below `dropout` at the same position
(0.935) and well below the best configuration in the study (0.944).

Both halves of the prediction hold. Its usable window sits an order of magnitude
lower than the mask operators', as the grid anticipated (sigma 0.194 at rate 0.1
against 0.65 at 0.3), and at matched disturbance it is comparable to or worse
than plain dropout rather than better.

**What the failure says.** DropPath is the ViT-native noise and the only operator
here that removes a *computation* rather than a *feature*. That it loses is
evidence that the unit that matters is not the computation. Since
[H16](H16-where-the-backdoor-neurons-are.md)'s correction the sharper statement is
available: the backdoor is a single non-axis-aligned direction in the residual
stream, and a branch either writes to that direction or does not. Removing whole
branches deletes the backdoor's contribution and the clean signal together,
because they share the branches, so the ratio PSU measures does not move.

The follow-up the pre-registration named, restricting droppath to H16's peak
layers with `--block-range`, is still worth one cheap run before the operator is
set aside.

---

## Original pre-registration, retained

## Mechanism

A ViT block computes `x = x + Attn(LN(x))` then `x = x + MLP(LN(x))`. `droppath`
zeroes a whole branch output for a whole sample, so the block becomes the
**identity** for that sample rather than computing a corrupted version of its
function. Every other operator leaves the branch running and damages what it
produces.

This is also the noise ViT is actually built with. Stochastic depth is standard
ViT regularization; Bernoulli dropout on activations is not, and torchvision's
ViT-B/16 ships with attention dropout at 0.0.

## Why it might work

[H3](H3-why-post-residual-fails.md) and [H1](H1-pre-beats-post.md) between them
established that perturbing the residual stream is a different act from
perturbing a branch, and that the stream carries the perturbation forward in a way
that saturates. `droppath` sidesteps the whole issue: it never touches the stream,
it only decides whether a branch contributes.

If the backdoor is written into the residual stream at a specific depth
([H16](H16-where-the-backdoor-neurons-are.md): peak layers 10 to 12), then
removing the *writing* operation at that depth should suppress the backdoor
sharply, while clean predictions, which accumulate evidence across many blocks,
degrade gracefully. That asymmetry is exactly what PSU measures.

## Why it might fail

- **Only 24 units exist** (12 blocks x 2 branches), so the granularity is coarse
  and the disturbance is lumpy. At p = 0.1 an average sample loses 2.4 branches,
  and which 2 dominates the outcome, so between-pass variance is large at k = 3.
- Removing a branch is **not** removing a feature. If the backdoor is one
  direction among 768 written by a block that also writes everything else,
  droppath deletes the backdoor and the clean signal together, and the ratio PSU
  measures does not move.
- Depth is confounded with quantity: at any rate, the sample loses branches spread
  over all 12 blocks, so this operator cannot isolate the depth H16 identifies.
  The `--block-range` mechanism exists and should be used as the follow-up if the
  undirected version shows anything.

## Prediction

`droppath` reaches its usable shift-ratio window at a **much lower rate** than any
other operator (the grid runs 0.01 to 0.7 for this reason), and at matched sigma
it performs **comparably to `dropout`, not better**, because 24 units is too
coarse to separate a 5-to-17-dimension backdoor from the clean signal sharing
those branches.

This is the one operator in the study predicted *not* to win. It is included
because it is the ViT-native choice and its failure would be informative: it would
say the unit that matters is the feature, not the computation.

## What would refute it

- `droppath` beats `channel_mask` and `dropout` at matched sigma on 3 or more
  pilot attacks. Then branch-level removal is the right granularity, the
  prediction above is wrong, and the immediate follow-up is `--block-range`
  restriction to H16's peak layers.
- Its usable window turns out to be at ordinary rates (0.3 to 0.7 reaching sigma
  in the mid range), meaning branch removal is far gentler than expected and the
  grid was mis-specified.

## Reproduce

    python pbs/generate_perturbation_jobs.py --stage pilot --operator droppath
    python psbd_analyze.py --all

Folder `pre_residual_droppath`, which applies droppath at both
`before_attention_residual` and `before_mlp_residual`, the two branch outputs.
