# SAM findings, 2026-09-11

This is the current statement of what this project knows about sharpness aware
minimization (SAM) and PSBD on ViT. It replaces every earlier SAM verdict dated
2026-09-07 through 2026-09-10, all of which rested on an unmatched aggregate
of 18 combinations, 13 of them the atypical `badnet_a2a` attack, that read
+0.009 mean AUROC and treated SAM as negligible. That number is superseded and
should not be cited going forward except as a labeled historical figure.

## What was measured

4 new experiments, run and written on 2026-09-11, replace the old aggregate
with matched Adam against SAM comparisons that hold architecture, dataset,
attack and poison rate fixed on both sides before taking a difference.
`experiments/sam_reading/README.md` builds the matched grid itself and reads
2 placements, the recommended token mask placement at the attention input
(PSBD-TM) and the published residual dropout placement (PSBD-RD), across all
4 swept rhos. `experiments/sam_low_rate/README.md` re-slices that same grid
by poison rate to ask whether the gain survives at the rates an attacker
would actually use. `experiments/sam_mechanism/README.md` asks whether SAM's
claimed amplification of the trigger's footprint on the class token explains
the PSBD-TM gain or is a separate fact. `experiments/sam_training_set_detection/README.md`
tests the SAM paper's own headline claim, a training-set detection gain, on
its own terms rather than on PSBD's. `docs/sam-audit-2026-09-11.md` inventories
every checkpoint and every SAM number on record and states the coverage gaps
directly.

Every reading below carries its source file and the number of matched
checkpoint pairs it rests on. All 4 experiments cover ViT only, and only
CIFAR-10 and CIFAR-100, since the PSBD sweep has not yet reached Swin, GTSRB
or Tiny ImageNet for both placements on the SAM side.

## The test-time reading by rate and attack

`experiments/sam_reading/README.md`, 24 matched checkpoint pairs per rho,
reads a positive PSBD-TM delta at every rho, +0.010 to +0.035 mean AUROC,
with the tightest interval at rho 0.1 excluding 0, [+0.002, +0.079]. The
published PSBD-RD placement loses instead, with intervals excluding 0 in the
losing direction at rho 0.15 ([-0.164, -0.004]) and rho 0.2 ([-0.157,
-0.003]).

Pooling every rho hides how thin that gain is at the poison rates that
matter. `experiments/sam_low_rate/README.md` re-slices the same checkpoints
by rate. At 1 percent poisoning, 7 matched pairs, the PSBD-TM delta is
indistinguishable from 0 at every rho and even flips sign across rho, from
-0.052 to +0.010. At 5 percent, 7 to 8 pairs, the delta is positive at 3 of 4
rhos but its interval excludes 0 only once, at rho 0.05, [+0.002, +0.071].
Only at 10 percent poisoning, 8 to 10 pairs, does the gain become both
consistent in sign and mostly significant, +0.065 to +0.074 across rho, 3 of
4 intervals excluding 0. A single pair is dropped from this grid before any
delta is computed, 10 percent poisoning, CIFAR-100, WaNet, ViT, rho 0.05,
because the SAM side's ASR falls to 0.831, under the 0.85 clearing bar, so a
broken implantation is never read as a detection loss.

Splitting the same 1 and 5 percent checkpoints by attack shows the gain is
not uniform across triggers. BadNets, Blend and LF, the firm local or global
triggers, sit within 0.06 of 0 at every rho with intervals that mostly
straddle it. BPP, a diffuse trigger, gains at every rho with intervals
excluding 0 at rho 0.1 and 0.15. WaNet gains the most by far but rests on
only 1 or 2 matched pairs per rho, so its size cannot yet be trusted the way
BPP's can.

## The mechanism reading

`experiments/sam_mechanism/README.md`, 6 matched checkpoint pairs (BadNet,
Blend, BPP, LF and WaNet on CIFAR-10 and CIFAR-100 at 1 and 5 percent
poisoning), reads the trigger's footprint on the class token directly rather
than PSBD's score. SAM raises the peak relative backdoor direction norm by 9
to 41 percent and the final layer mean TAC by 8 to 31 percent in 5 of the 6
pairs, the same direction the SAM paper's own metrics report on ResNet
backdoor neurons. WaNet at 5 percent on CIFAR-10 is the exception, where the
footprint shrinks by roughly a third and PSBD-TM's AUROC falls from 0.945 to
0.704.

Where the footprint grows, the gap between the triggered and clean logit
margin widens too, and PSBD-TM's AUROC rises by 0.009 to 0.057 on those 5
pairs. The 2 movements are coupled loosely rather than proportionally. LF
carries the largest footprint gain (41 percent) and only a middling AUROC
gain (0.031). BPP's footprint gain is smaller (20 percent) yet its AUROC
gain is the largest of the 5 (0.057). TAC and direction norm describe how far
the trigger moves the class token in an unperturbed pass, while PSBD's own
PSU ratio describes how much of that movement survives token masking, and
this experiment is the first direct evidence that the 2 quantities measure
related but distinct things.

## The training-set reading against the paper's claim

The SAM paper's headline result is a training-set poison filter, not a
test-time detector like PSBD. `experiments/sam_training_set_detection/README.md`
tests that claim on its own terms on 6 matched checkpoint pairs. CIFAR-100
with `badnet_a2o`, `blend` and `wanet` at 1 and 5 percent poisoning supplies
the pairs, using the paper's own 2 simplest detectors, Spectral Signature and
Activation Clustering, at their native decision rules. The paper's own SAM-only ablation
(no feature scaling, the fair comparison since this experiment also omits
feature scaling) reports Spectral Signature TPR rising 15.2 to 57.7 points on
ResNet18 CIFAR-10. On ViT-B/16 CIFAR-100 the same detector's TPR moves by at
most 2.0 points and by as little as -1.0 points across the 6 pairs at the 1
percent rate, the only rate where the detector is informative (at 5 percent
the removal rule asks for more flags than the class holds and both optimizers
degenerate to flagging everyone).

Activation Clustering's TPR moves by at most 1.0 point under SAM in either
direction, and whether it finds the true poisoned cluster at all is set
entirely by the attack, not by the optimizer, in both the Adam and the SAM
checkpoints alike. Averaged over all 6 pairs, the silhouette coefficient of
the true poisoned against clean split moves by -0.026 under SAM and the top
to second singular value ratio moves by -0.13, both the wrong sign against
the paper's own reported silhouette gains of 0.13 to 0.26 on ResNet18
CIFAR-10. The most likely reason is a ceiling effect, ViT-B/16's target class
features are already 73 to 79 percent separable under plain Adam at 1
percent poisoning, well above the paper's own ResNet18 baseline for a weak
trigger like Blend, 32.9 percent, leaving little room for SAM to add.

## What is still running and what would change the conclusion

`docs/sam-audit-2026-09-11.md` ranks the open measurements. The single
largest coverage gap is Swin, 0 of 496 SAM checkpoints there have both
placements swept for the matched comparison, so nothing in this document says
anything about whether the token mask gain holds on that architecture. GTSRB
and Tiny ImageNet are the same gap on the dataset side, 0 of 496 SAM
checkpoints between the 2 datasets have both placements swept, and Tiny
ImageNet is this project's own priority dataset, so that gap should close
before either the CIFAR-10 or CIFAR-100 reading is treated as final. 512
already-trained SAM checkpoints have a `psbd_metrics.json` for a single
placement but not both, so a large share of the needed sweep is cheap CPU
and GPU reuse rather than new training.

2 further checks would most directly change the conclusion above. First, a
filter of the wanet rows already inside `sam_reading`'s 24-combination grid
against their own ASR, to confirm the known SAM WaNet instability at low
poison rates has not quietly inflated or deflated the pooled rho deltas the
way it does in the low-rate grid, where 1 pair was already excluded on this
basis. Second, a second training seed per attack at rho 0.1 on ViT CIFAR-10,
since every SAM checkpoint on disk today is seed 0 and the mechanism reading
above cannot yet be told apart from ordinary seed to seed variance in how
far a trigger's footprint moves under retraining.
