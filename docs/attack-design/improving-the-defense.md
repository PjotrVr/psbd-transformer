# Improving the defense, 5 probes the basis does not contain

Written 2026-09-23, after reading the operator and position registries against the
mechanism the paper argues for. The paper's account is that a backdoor in a ViT is
1 direction in the residual stream, written in the last third of the network and
routed through attention from the trigger's own tokens to the class token.

Every proposal here is a probe, so none of it changes the statistic, the rate rule
or the threshold. They slot into `defenses/operators.py` and
`models/positions.py` as new entries and are swept by `cli.sweep` unchanged.

## The gap

`models.positions.POSITION_REGISTRY` holds 15 positions on ViT and every one of
them is a tensor boundary: the input to a sublayer, the output of a sublayer, the
residual stream, the per-head outputs, the MLP hidden units or the pixels.
`defenses.operators.OPERATORS` holds 9 operators and every one of them acts on a
tensor of activations.

Nothing in the basis perturbs the attention map. The route the mechanism names is
the one object the 18 measured placements never touch. `attention_heads`
perturbs what attention has already produced, which is downstream of the routing
decision, and every other position is either upstream of it or in the residual
stream beside it. If the mechanism is right, the attention weights are where the
backdoor's signature is most concentrated, and the basis has no reading there.

That is the strongest single argument for the probes below, and it is an argument
the paper itself makes without following through.

## Probe 1, masking the attention map

**What.** Mask entries of the pre-softmax attention logits, setting a random
fraction to negative infinity so those key positions cannot be attended to, then
let the softmax renormalize over what remains. This is DropKey (Li et al., CVPR
2023) used as an inference-time probe rather than as a regularizer.

**Why it should work.** A clean prediction draws on many tokens, so removing a
random share of the available keys degrades it. A triggered prediction draws on
the trigger's tokens through a small number of attention edges, and it survives
until one of those edges is the one removed. The 2 populations should separate on
a different axis from token masking, because token masking removes a token's
content while this removes a token's reachability.

**Why it might beat token masking.** Token masking at the attention input removes
a token from every head of that block at once. Masking the attention map removes
an edge per head, which is a finer and better targeted intervention on a route
the mechanism says is narrow.

**Cost.** 1 new position, `attention_map`, which needs the same removable forward
wrapper that `attention_heads` already uses, because the attention weights are a
local variable inside `F.multi_head_attention_forward` and never cross a module
boundary. `models/positions.py` already carries that machinery for 2 positions.

**On a ResNet.** There is no attention map, so this probe is transformer only.
That is a real limitation and it is why probe 2 matters.

## Probe 2, substituting tokens rather than zeroing them

**What.** Replace a random share of tokens with the corresponding token of
another image in the batch, or with the dataset's mean token, instead of setting
them to 0.

**Why it should work.** A zeroed token is off the data manifold. The network
never saw a zero token in training, so part of any prediction shift it causes is
the network reacting to an input it has no calibration for, which is noise with
respect to the question being asked. Substitution keeps every token in
distribution and isolates the effect of removing that token's specific content.

**Why this is the most promising of the 5.** It is a strictly better-posed
version of the operator that already wins. If token masking works because it
removes the trigger's tokens from the routing, substitution removes them just as
well while adding no off-manifold component, so it should separate at least as
cleanly and plausibly better. If it separates worse, that is informative too: it
would mean part of the current headline is the off-manifold artifact rather than
the mechanism, which is a result worth knowing before publication.

**Cost.** 1 new operator, no new position. It needs the batch, which the operator
already receives.

**On a ResNet.** A convolutional feature map has spatial positions rather than
tokens, and substituting a spatial patch from another image is the exact
analogue, so this probe transfers.

## Probe 3, attention temperature

**What.** Divide the pre-softmax attention logits by a temperature above 1,
flattening the attention distribution toward uniform without removing anything.

**Why it should work.** It tests firmness rather than presence. A shortcut that
routes through a few sharp attention edges loses those edges as the distribution
flattens. Clean evidence already spread over many tokens is closer to uniform to
begin with and has less to lose.

**Why it is worth having even if it loses.** It is deterministic, so 1 forward
pass is exact and `DETERMINISTIC_OPERATORS` already handles that case. It is the
cheapest probe in this document by a wide margin, and a union wants a cheap
member.

## Probe 4, contiguous patch masking

**What.** Mask a random patch-aligned rectangle of tokens rather than a uniformly
random subset, which is Cutout geometry applied in token space.

**Why it should work.** A local trigger occupies contiguous tokens. Uniform token
dropping removes a share of the trigger's tokens in proportion to the rate, so it
degrades the shortcut gradually. A contiguous mask either covers the trigger or
misses it, which turns a gradual degradation into a high-variance one. Higher
variance across passes is itself a signal, and the statistic already averages
over k passes.

**Why it might fail.** It should do nothing for a global trigger such as Blend,
and the panel's hard attacks include global ones.

## Probe 5, a rank-1 residual perturbation

**What.** Add or remove a random rank-1 component of the residual stream, rather
than isotropic noise across all 768 coordinates.

**Why it should work.** The mechanism says the backdoor is 1 direction. Isotropic
Gaussian noise spends most of its energy in the 767 directions that do not carry
it, which is why Gaussian trails token masking by 0.183 at the attention input. A
rank-1 perturbation is the shape the mechanism predicts should matter, and
averaging over k random rank-1 draws asks how much of the decision lives in any
single direction.

**Why it is ranked last.** The argument predicts it should be weak, not strong. A
random direction in 768 dimensions is nearly orthogonal to the backdoor
direction, so most draws will do nothing. It is included because it is the
control that the Gaussian result implies: if rank-1 random perturbation also
trails token masking, the reason is dimension rather than structure, and the
paper can say so.

## On the multi-probe union

The union already measured pairs token masking at the attention input with token
masking at the attention branch output, and gains 0.022 with an interval
excluding 0 because the 2 probes fail on different models. Adding the published
residual placement instead gains 0.006 with an interval crossing 0.

What that pattern says is that a union is worth what its members' failures are
decorrelated, not what its members individually score. The 2 token-mask probes
sit 1 sublayer apart and still disagree usefully. A probe that perturbs a
different object entirely should disagree more.

That is the concrete refinement. The natural partner for a token-level probe is
probe 1, which perturbs reachability rather than content, and not a third token
probe at a fourth site. A union of a content probe and a route probe is the
2-member union the mechanism predicts should be strongest, and it has never been
measured because the route probe does not exist.

## Ranking, and what to do first

| rank | probe | new position | new operator | transfers to a ResNet | why this rank |
|---:|---|---|---|---|---|
| 1 | token substitution | no | yes | yes | strictly better posed than the operator that already wins, cheapest to build, and informative whichever way it goes |
| 2 | attention map masking | yes | yes | no | the only probe that touches the route the mechanism names, and the best union partner |
| 3 | attention temperature | yes | yes | no | deterministic and nearly free, so a union can afford it |
| 4 | contiguous patch masking | no | yes | yes | matched to local trigger geometry, predicted to do nothing for global triggers |
| 5 | rank-1 residual | no | yes | yes | the control the Gaussian result implies rather than a candidate winner |

Probes 1 and 4 need no new position and no new hook machinery, so they are the 2
to write first. Probe 1 can be evaluated against the existing token mask on the
same cells at the same rates, which makes it a paired comparison on data the
panel already covers.

## What this does not address

None of these probes changes the statistic, so the margin cancellation derived in
`breaking-psbd-non-adaptive.md` still holds and none of them recovers the factor
the fractional form divides out. An attacker who conditions the trigger on the
input's own class evidence defeats a content probe and a route probe alike,
because the conjunction leaves the true class alive as the runner up in both. The
counter to that attack is a different statistic, entropy or margin fused with
PSU, and not a better probe.
