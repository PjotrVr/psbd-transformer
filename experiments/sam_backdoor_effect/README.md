# Whether SAM amplifies the backdoor on ViT (H6)

## Question

This is the positive control for H6. PSBD gets worse on SAM-trained models, and
there are 2 very different explanations for that.

Either SAM does amplify the backdoor here, exactly as the SAM paper claims, and the
benefit simply does not reach a prediction-space detector, which is a finding about
detector families and the interesting outcome. Or SAM does not amplify the backdoor
in a ViT at all, so there was never anything for PSBD to miss, and the finding is
about transformers or AdamW instead and says nothing about detector families.

Only a measurement on our own checkpoints separates them, and it uses the SAM
paper's own metrics so the comparison is on their terms.

## The 3 metrics

| metric | definition |
| --- | --- |
| `top2_tac` | mean of the 2 largest per-dimension trigger-activated changes. The paper's "backdoor effect", the quantity they show correlates with detector AUC at r = 0.71 |
| `silhouette` | separation of clean and triggered features at the penultimate layer. They report 0.19 rising to 0.32 for BadNets under SAM |
| `rel_direction` | the norm of the mean paired difference over the mean clean norm, this project's scale-free direction magnitude, included so the result ties back to the rest of the ledger |

All 3 are computed at the final block output in fp32 over eligible paired samples
only, matching the paper's "last convolutional layer" choice.

## Running it

    PYTHONPATH=. python experiments/sam_backdoor_effect/measure.py \
        --attack badnet_a2o blend bpp lf --poison-tag 0_1

Writes `results/sam_backdoor_effect.json`. Needs a GPU.

## Finding

Explanation (a). SAM does amplify the backdoor on ViT as the paper claims, and the
amplification does not produce the separability gain their detectors feed on. It
does produce the clean-variance side effect their method exists to cancel, and PSBD
has no way to cancel it.

The H6 verdict carries a coverage caveat that this control does not remove. The
naive Adam against SAM aggregate compares 225 checkpoints to 16 and is confounded,
because the SAM checkpoints were swept over a harder set. Matched on architecture,
dataset, attack, poison rate and placement, the sign flips to between +0.025 and
+0.050 in SAM's favour, and 13 of those 18 matched cells are `badnet_a2a`. The
verdict should be re-decided once matched `badnet_a2o`, `blend`, `bpp` and `lf`
cells land.

Hypothesis doc: `docs/hypothesis/H6-sam-improves-detectability.md`.
