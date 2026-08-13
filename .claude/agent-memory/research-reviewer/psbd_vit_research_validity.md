---
name: psbd-vit-research-validity
description: Standing research-validity facts about PSBD-ViT that are easy to miss and affect how results must be interpreted
metadata:
  type: project
---

Facts about PSBD-ViT that change how a result should be read. Re-verify against the code before acting on any of them; they were true as of 2026-08-13.

**Clean-label and source-specific poison rates silently cap.** `poison.choose_poison_indices` and `choose_indices_with_cover` take `min(round(rate * len(dataset)), len(eligible_pool))`. For `sig`/`lc` the eligible pool is only the target class (500 images on CIFAR-100), and for `tact` only the source classes. So on CIFAR-100 the 0.01, 0.05 and 0.1 clean-label checkpoints are all the same realized 1% poisoning, while `args.json`/`metrics.json` report the requested rate. Confirmed empirically: `vit_cifar100_sig_0_01` and `_0_05` have near-identical ASR and clean accuracy.
**Why:** any poison-rate trend line for sig/lc/tact on CIFAR-100 is measuring noise between identical poisonings, and any table citing those rates is wrong.
**How to apply:** when reviewing or writing up rate sweeps, treat clean-label and TaCT rows as a single realized rate unless the cap has been fixed and the models retrained.

**PSBD here is reframed from the paper.** The published method scores the *suspicious training set*; this repo scores held-out *test* images (clean vs triggered) and reports TPR/FPR/AUROC over that. Also `PSBD_HELDOUT_SIZE = 2000` is fixed, where the paper uses 5% of the training set, and training normalizes inputs where the paper explicitly does not.
**Why:** these make the numbers non-comparable to PSBD's published table even when the code is internally correct.
**How to apply:** flag any claim that compares a PSBD-ViT number to the paper's reported number; the reframing has to be stated as a divergence, not assumed equivalent.

**badnet_a2a ASR is around 0.50 at 1% poison on CIFAR-100.** The `AttackSuccessSet` "backdoor" split contains every triggered image regardless of whether the trigger actually flipped it.
**Why:** for all-to-all, roughly half the detector's positive set is not behaviourally backdoored, so a low TPR cannot be read as "the detector failed."
**How to apply:** any all-to-all falsification probe needs the positive set restricted to successfully-flipped samples, or the result reported conditioned on attack success.

See [[user-role]].
