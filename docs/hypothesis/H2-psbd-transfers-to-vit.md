# H2 — PSBD works on ViT at all, and gives chance-level results on benign models

**Status: SUPPORTED**

## Claim

At its best placement, PSBD separates clean from triggered inputs on ViT-B/16 well
above chance (AUROC clearly above 0.5), *and* on a benign model with the same
trigger applied it gives roughly 0.5.

Both halves are required. The first alone is not evidence: a probe that reacts to
any trigger-shaped input perturbation would also score high, on any model,
backdoored or not.

## Prediction

- Backdoored, best placement: AUROC well above 0.5 on every attack with ASR above 0.8.
- `vit_cifar10_benign` probed with the BadNet trigger: AUROC near 0.5.

Refuted if the benign control scores high. In that case the pipeline is measuring
an artifact and every other hypothesis in this ledger is void.

## Why it is interesting

It is the load-bearing sanity check, and it very nearly did not exist:
`build_attack("benign", ...)` raised, so the control was unrunnable until
`resolve_probe_attack` was added. A study whose negative control cannot be
executed is a study that cannot be wrong.

Note also that a benign model has no target class, so the trigger is just an
unusual input patch. If PSU flags it, PSU is an out-of-distribution detector
wearing a backdoor-detector costume.

## Evidence

**Both halves hold.**

*Backdoored, best placement, AUROC at the 25th-percentile threshold:* `blend`
0.978 to 0.986, `bpp` 0.964 to 0.995, `lf` 0.887 to 0.947, `badnet_a2o` 0.686 to
0.889. Well above chance on all four.

*Benign control:* `vit_cifar10_benign` probed with the same BadNet trigger scores
**0.506** (post-residual) and **0.503** (pre-residual). Chance, to within noise.

So PSBD on ViT is detecting a learned backdoor, not reacting to the presence of a
trigger-shaped input perturbation. Everything else in the ledger depends on this.

The latent measurement agrees independently: probed with the same trigger, the
benign model's relative backdoor-direction norm peaks at 0.089 and *decays* with
depth, against 1.0 to 2.2 growing monotonically for every backdoored model, and its
clean-vs-triggered CKA at layer 12 is 0.998 against 0.118 to 0.419.

One attack is the exception and it is not a failure of this hypothesis:
`badnet_a2a` scores 0.51 to 0.60 despite ASR 0.94 to 0.96. See
[H5](H5-all-to-all-breaks-psbd.md).

## Reproduce

```bash
python psbd_dropout_sweep.py --checkpoint-folder vit_cifar10_benign \
    --position-config pre_residual --probe-attack badnet_a2o --probe-target-label 0
python psbd_analyze.py --checkpoint-folder vit_cifar10_benign
```

## Subquestions

1. The floor came out at 0.506, so essentially nothing needs subtracting. But it
   was measured with one trigger on one benign model; a patch trigger is a larger
   input perturbation than BPP's quantization, so the floor could differ by trigger
   and each should get its own.
2. Does the benign floor differ by trigger? A patch trigger is a larger input
   perturbation than BPP's quantization, so it should have a higher floor.
3. Would a SAM-trained benign model have a different floor from an Adam one?
