# H20 — Removing whole tokens separates local triggers from distributed ones

**Status: PRE-REGISTERED.** Written before the pilot results land. Jobs
`psbd_pilot_001/002`, submitted 2026-08-14 at commit `b37ceaf`.

## Mechanism

A ViT sees an image as 196 patch tokens plus CLS. `token_mask` drops whole tokens
for a sample, which is the transformer's version of occluding a region of the
image. CLS is never masked: it is the classifier's only read point, so removing it
destroys the prediction instead of perturbing it, and PSU would then be measuring
a broken forward pass rather than a disturbed one.

This is the only operator in the study whose unit is **spatial**. Every other one
acts on features or on computation.

## Why it might work

Triggers differ in spatial support, and this is the operator that can see the
difference:

- `badnet_a2o` and `badnet_a2a` use a small corner patch, which lands in roughly 4
  of 196 tokens. Masking a fraction p of tokens removes the trigger entirely with
  probability about p^4 per sample, so at high p the trigger is often simply gone.
- `blend`, `wanet`, `adaptive_blend` modify every pixel, so every token carries
  some trigger and no amount of token removal deletes it.

If the backdoor prediction survives token removal far better than the clean
prediction does, PSU separates. For a patch trigger there is also a sharper
possibility: token masking is the one perturbation that can remove the *cause*
rather than degrade the *representation*.

## Why it might fail

- For distributed triggers this operator should do nothing that a channel mask
  would not do better, so at best it works on 2 of the 5 pilot attacks.
- **The direction may invert for patch triggers**, and this is the outcome to
  watch. If masking deletes the trigger outright, the backdoored sample reverts to
  its clean class, which is a *large* prediction shift, i.e. **high** PSU. PSBD
  flags low PSU. So the operator that most directly removes the trigger may score
  below 0.5 by making backdoored samples look maximally uncertain.
  Under this project's one-sided rule that is reported as a failure, not
  re-signed ([H15](H15-one-sided-rules-are-the-common-weakness.md)).
- ViT-B/16 at 224x224 upsamples CIFAR-10 from 32x32, so a 3x3 CIFAR trigger
  becomes roughly 21x21 pixels and may span more tokens than the naive count
  suggests, weakening the locality contrast.

## Prediction

Two-part, and the second is the interesting one:

1. `token_mask` gives its **best relative performance on `badnet_a2o` and
   `badnet_a2a`** and its worst on `blend` and `wanet`, i.e. the opposite ranking
   to every other operator in the study, which favours distributed triggers.
2. On patch triggers, `token_mask` AUROC **falls with rate** and may cross 0.5 at
   high p, as masking shifts from perturbing the trigger to deleting it.

## What would refute it

- No attack-family ordering: `token_mask` ranks attacks the same way `dropout`
  does, so spatial support is not what it is measuring.
- It works equally well on `blend` and `wanet`, which have no spatial locality to
  exploit; then any gain comes from generic capacity removal and `channel_mask`
  is the cleaner way to get it.
- Prediction 2 fails and AUROC *rises* monotonically with rate on patch triggers,
  which would mean trigger deletion is not happening even at p = 0.9.

## Reproduce

    python pbs/generate_perturbation_jobs.py --stage pilot --operator token_mask
    python psbd_analyze.py --all

Positions `after_embedding_token_mask` and `before_attention_norm_token_mask`.
The rate-dependence in prediction 2 is read off the per-rate blocks, not off the
adaptive or matched summary, which would hide the trend.
