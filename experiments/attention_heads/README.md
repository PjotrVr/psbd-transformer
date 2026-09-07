# Attention divergence and entropy in backdoor heads (H31, H40)

## Question

If a backdoor is implemented by a circuit rather than smeared across the network,
a small number of attention heads should route the trigger, and their attention
maps should change when the trigger appears. H31 looks for those heads. H40 then
asks whether the same heads give a per-sample detection feature, which is the only
version of the finding a defender could deploy.

## The 2 scripts

| script | what it measures |
| --- | --- |
| `attention_divergence.py` | per-head attention matrices for paired clean and triggered inputs, the divergence between them per head, and whether masking only the top heads neutralizes the backdoor |
| `attention_entropy.py` | per-sample attention entropy inside the heads H31 named, scored as a standalone detector |

## Running it

    python experiments/attention_heads/attention_divergence.py
    python experiments/attention_heads/attention_entropy.py

Both extract attention weights through forward hooks and want a GPU. Output goes to
`results/attention_divergence.json` and `results/attention_entropy.json`.

## Finding

H31 is supported. Layer 5 head 0, layer 6 head 3, and layer 5 head 10 show elevated
divergence between clean and triggered attention across every tested attack and
dataset. BadNet additionally recruits layer 9 head 7 and layer 10 head 9, which are
specific to localized triggers and match the late routing that
`backdoor_direction_layers` measured for patch triggers.

H40 is refuted as stated, and its failure is more informative than a flat null.
Entropy in those heads gives AUROC 0.48 to 0.54 for most attacks, indistinguishable
from random. `blend` is the exception and runs strongly INVERTED: its backdoor
samples have HIGHER entropy, reading 0.000 to 0.006 under the stated "lower entropy
means backdoor" convention. The trigger makes attention more diffuse there, not more
focused, so the premise is backwards for global triggers. Consistent with H15, that
is reported as the premise failing rather than recovered by flipping the sign.

Hypothesis docs: `docs/hypothesis/H31-attention-divergence-backdoor-heads.md`,
`docs/hypothesis/H40-attention-entropy-detection.md`.
