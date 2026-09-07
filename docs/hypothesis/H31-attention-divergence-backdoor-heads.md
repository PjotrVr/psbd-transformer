# H31 — Attention map divergence reveals consistent backdoor heads

**Status: SUPPORTED.** A small set of heads (layer 5 head 0, layer 6 head 3,
layer 5 head 10) show elevated JS divergence between clean and triggered
attention patterns across ALL tested attacks and datasets. BadNet additionally
recruits late-layer heads (layer 9 head 7, layer 10 head 9) that are specific
to localized triggers.

Evidence: `experiments/attention_heads/attention_divergence.py`, results in
`results/attention_divergence.json`.

## Claim

Backdoored inputs produce distinctive attention patterns in a small number of
"backdoor heads." These heads attend to the trigger region (for localized
attacks) or show globally different attention distributions (for non-localized
attacks).

## Results

### Top 5 heads by JS divergence, per checkpoint

| checkpoint | top heads (layer, head, JS) |
|---|---|
| cifar100 badnet 10% | (10,9, 0.572), (9,7, 0.567), (6,5, 0.429), (5,0, 0.385), (6,3, 0.367) |
| cifar100 blend 10% | (5,0, 0.318), (5,10, 0.310), (6,3, 0.291), (3,5, 0.290), (6,6, 0.275) |
| cifar100 wanet 10% | (5,0, 0.355), (7,1, 0.337), (5,10, 0.329), (4,2, 0.326), (6,3, 0.319) |
| cifar100 a_blend 10% | (7,1, 0.373), (6,3, 0.358), (5,10, 0.347), (5,0, 0.347), (6,6, 0.342) |
| tiny badnet 10% | (9,7, 0.514), (5,0, 0.388), (5,10, 0.375), (6,3, 0.350), (7,2, 0.336) |
| tiny blend 10% | (5,10, 0.324), (5,0, 0.322), (6,3, 0.292), (3,5, 0.286), (5,4, 0.283) |

### Cross-attack consistency

Three heads appear in the top 5 across nearly all checkpoints:

| head | appearances in top 5 | attacks |
|---|---:|---|
| layer 5 head 0 | 6/6 | all |
| layer 6 head 3 | 6/6 | all |
| layer 5 head 10 | 5/6 | all except cifar100 badnet |

These are mid-network heads in the attention crystallization zone (layers 5 to
7), consistent with the H30 finding that the backdoor direction begins forming
around layers 7 to 9. The attention divergence appears one to two layers earlier,
suggesting these heads are part of the circuit that WRITES the backdoor direction
into the residual stream.

### BadNet-specific late heads

BadNet uniquely activates late-layer heads:
- Layer 10 head 9 (JS=0.572, highest of any head on any attack)
- Layer 9 head 7 (JS=0.567 on CIFAR-100, 0.514 on Tiny)

No other attack produces JS > 0.4 at layers 9 to 10. This is consistent with
BadNet's localized patch trigger creating a strong spatial signal that late
layers can attend to specifically.

## Interpretation

1. **Backdoor circuits are partially shared across attacks.** The three
   consistent heads (5:0, 5:10, 6:3) may represent a general "anomaly
   processing" circuit in the pretrained ViT that all attacks co-opt. They
   diverge between clean and triggered inputs because the trigger (any trigger)
   changes the token interactions these heads mediate.

2. **Localized triggers recruit additional specialized heads.** BadNet's patch
   trigger creates a spatially concentrated signal that late-layer heads can
   attend to, producing the highest per-head JS divergence of any attack.
   Global triggers (blend, wanet) distribute their effect more evenly and do not
   recruit these additional heads.

3. **JS divergence magnitude predicts attack detectability.** BadNet (max
   JS=0.572) is easy to detect; blend (max JS=0.318) and wanet (max JS=0.355)
   are harder. This aligns with the detection AUROC ordering from the operator
   comparison results.
