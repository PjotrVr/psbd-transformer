# H2 — PSBD works on ViT at all, and gives chance-level results on benign models

**Status: OPEN**

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

Pending. `vit_cifar10_benign` is in the phase-3a grid, probed with
`--probe-attack badnet_a2o --probe-target-label 0`.

## Reproduce

```bash
python psbd_dropout_sweep.py --checkpoint-folder vit_cifar10_benign \
    --position-config pre_residual --probe-attack badnet_a2o --probe-target-label 0
python psbd_analyze.py --checkpoint-folder vit_cifar10_benign
```

## Subquestions

1. If benign AUROC is not 0.5 but, say, 0.6, how much of the backdoored score is
   trigger-response rather than backdoor-response? The benign number is the floor
   every other number should be read against, not just a pass/fail gate.
2. Does the benign floor differ by trigger? A patch trigger is a larger input
   perturbation than BPP's quantization, so it should have a higher floor.
3. Would a SAM-trained benign model have a different floor from an Adam one?
