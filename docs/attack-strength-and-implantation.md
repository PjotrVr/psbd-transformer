# Why some attacks do not implant, and which of those are fixable

A detection result on a cell whose attack never implanted is meaningless: there is no
backdoor to detect. The panel's ASR bar of 0.85 exists to keep those cells out, and
until now the cells below it were treated as one undifferentiated pile of failures.

They are not one pile. Sorting them by cause gives **four distinct failure modes, only
two of which are fixable**, and mixing them was hiding both the fixes and the genuine
limits.

## The measurement

ViT, seed 0, base cells only (no SAM, evasion, epoch snapshots, seed replicates, or
strength-sweep variants). ASR at each nominal poison rate; `--` is a cell never trained.

| attack | dataset | 0.5% | 1% | 5% | 10% | verdict |
|---|---|---:|---:|---:|---:|---|
| badnet_a2o | cifar10 | 0.999 | 0.997 | 1.000 | 1.000 | ok |
| badnet_a2o | cifar100 | 0.988 | 1.000 | 1.000 | 1.000 | ok |
| badnet_a2o | gtsrb | -- | 1.000 | 1.000 | 1.000 | ok |
| badnet_a2o | tiny | -- | 0.999 | 1.000 | 1.000 | ok |
| blend | cifar10 | 1.000 | 1.000 | 1.000 | 1.000 | ok |
| blend | cifar100 | 0.999 | 0.989 | 1.000 | 1.000 | ok |
| blend | gtsrb | -- | 1.000 | 1.000 | 1.000 | ok |
| blend | tiny | -- | 0.999 | 0.999 | 1.000 | ok |
| lf | cifar10 | 0.902 | 0.982 | 0.997 | 0.999 | ok |
| lf | cifar100 | 0.836 | 0.942 | 0.953 | 0.995 | ok |
| lf | gtsrb | -- | 0.979 | 0.999 | 0.999 | ok |
| lf | tiny | -- | 0.922 | 0.987 | 0.973 | ok |
| bpp | cifar10 | 0.960 | 0.982 | 0.987 | 0.994 | ok |
| bpp | cifar100 | 0.939 | 0.914 | 0.988 | 0.991 | ok |
| bpp | gtsrb | -- | 0.868 | 0.991 | 0.995 | ok |
| bpp | tiny | -- | 0.971 | 0.985 | 0.996 | ok |
| tact | cifar10 | 0.112 | 0.985 | 0.996 | 1.000 | ok |
| tact | cifar100 | -- | 0.989 | 1.000 | 0.989 | ok |
| tact | gtsrb | -- | **0.000** | 1.000 | 1.000 | ok, one broken run |
| tact | tiny | -- | 0.976 | 0.952 | 0.929 | ok |
| badnet_a2a | cifar10 | 0.838 | 0.941 | 0.933 | 0.958 | ok |
| badnet_a2a | cifar100 | 0.275 | 0.503 | 0.801 | 0.789 | weak |
| badnet_a2a | gtsrb | 0.759 | **0.048** | 0.931 | 0.983 | ok, one broken run |
| badnet_a2a | tiny | 0.189 | 0.380 | 0.722 | 0.734 | weak |
| wanet | cifar10 | 0.025 | 0.112 | 0.961 | 0.890 | ok above 1% |
| wanet | cifar100 | 0.011 | 0.057 | 0.643 | 0.793 | weak |
| wanet | gtsrb | 0.005 | 0.119 | 0.830 | 0.947 | ok above 1% |
| wanet | tiny | 0.006 | 0.178 | 0.922 | 0.968 | ok above 1% |
| adaptive_blend | cifar10 | 0.284 | 0.622 | 0.594 | 0.622 | weak |
| adaptive_blend | cifar100 | -- | 0.536 | 0.579 | 0.604 | weak |
| adaptive_blend | gtsrb | -- | 0.560 | 0.777 | 0.838 | weak |
| adaptive_blend | tiny | -- | 0.554 | 0.783 | 0.505 | weak |
| lc | cifar10 | 0.119 | 0.249 | 0.408 | 0.979 | ok at cap only |
| lc | cifar100 | -- | 0.873 | 0.545 | 0.785 | capped, see below |
| lc | gtsrb | -- | 0.464 | 0.239 | 0.129 | capped |
| lc | tiny | -- | 0.387 | 0.706 | 0.623 | capped |
| sig | cifar10 | 0.129 | 0.340 | 0.599 | 0.901 | ok at cap only |
| sig | cifar100 | -- | 0.250 | 0.249 | 0.178 | capped |
| sig | gtsrb | -- | 0.460 | 0.015 | 0.415 | capped |
| sig | tiny | -- | 0.076 | 0.432 | 0.122 | capped |

## Mode 1: structural. The rate was never applied (sig, lc)

Clean-label attacks may only poison the target class, so the columns above are not what
they say. On CIFAR-100 the 1%, 5% and 10% cells are the **same 500 images**; on GTSRB the
same 150; on Tiny the same 500 at every rate including 0.5%. Their spread is training
noise, not a dose-response.

`poison.choose_poison_indices` clamps silently. This much was established by audit
finding A8 on 2026-09-07; 189 of the 316 clean-label folders are affected. The
reachability table and the fixes are in
[clean-label-rate-caps.md](clean-label-rate-caps.md).

Two things follow. First, **`lc` was also the wrong attack**: Turner's adversarial
pre-perturbation was never implemented, which is why the patch-only variant needs
essentially the whole target class before it implants (CIFAR-10 clears only at 10%, which
*is* 100% of the target class). Second, **GTSRB's target class was a self-imposed
handicap**: class 0 holds 150 images against 1,500 in classes 1 and 2.

Fixable: yes for GTSRB (target class) and for `lc` everywhere (adversarial bases). Not
fixable for `sig` above each dataset's cap; 10% clean-label is impossible anywhere but
CIFAR-10, and that is a property of the datasets.

## Mode 2: trigger too weak. WaNet at 1% (fixable, and partly fixed)

WaNet's implementation is faithful to Nguyen and Tran: `k x k` uniform control offsets,
normalised by mean absolute value, bicubic upsampled, applied as
`identity + strength * field / image_size`. At the default `strength = 0.5` the
displacement is roughly a quarter of a pixel, which is too subtle to learn from few
poisoned samples. That is a dose problem, not a bug, so it responds to dose.

Measured at **1% poisoning**, the rate where WaNet failed everywhere:

| dataset | strength 0.5 (default) | 1.0 | 2.0 | 4.0 |
|---|---:|---:|---:|---:|
| cifar10 | 0.112 | 0.165 | 0.775 | 0.757 |
| cifar100 | 0.057 | 0.167 | 0.569 | 0.792 |
| gtsrb | 0.119 | 0.838 | **0.934** | **0.918** |

**GTSRB clears the bar at strength 2.0**: ASR 0.934 at clean accuracy 0.987 against a
benign reference of 0.991, so the clean-accuracy cost is 0.004 and far inside the 0.05
budget. Strength 4.0 also clears (0.918 at 0.983) but buys nothing, so by the
smallest-sufficient-dose rule **2.0 is the value to adopt for GTSRB**. This is the first
WaNet configuration to reach 0.85 at 1% poisoning on any dataset.

CIFAR-10 and CIFAR-100 climb steeply but have not cleared by strength 4.0. CIFAR-10's
0.775 at 2.0 against 0.757 at 4.0 is flat within the run-to-run spread rather than a real
non-monotonicity. Both need strength 8 or higher tested.

Sweep jobs are still in flight and will fill the missing cells, including 5% and the
BadNet patch-size arm. These results only exist because the sweep's output bug was caught
while it was running; see
[checkpoint-integrity-2026-09-09.md](checkpoint-integrity-2026-09-09.md).

## Mode 3: seed sensitivity (adaptive_blend)

Adaptive-Blend fails at seed 0 on all four datasets (0.505 to 0.838) but clears at seeds
1 and 2. That is not a dose problem and not a structural one; it is a single-run readout
of a quantity with a wide distribution. An asymmetric-trigger correction is committed but
untested at scale.

This is the mode that most argues for seeds. See the next section.

## Mode 4: broken individual runs

Two cells are not weak attacks but failed training runs, and should be retrained rather
than analysed:

- `vit_gtsrb_tact_0_01`: ASR 0.000 with clean accuracy 0.121 against a benign 0.991. The
  model did not train at all.
- `vit_gtsrb_badnet_a2a_0_01`: ASR 0.048 sitting between 0.759 at 0.5% and 0.931 at 5%.

## How wide is the error bar on a single ASR?

Wider than the 0.85 bar assumes. The clean-label cap accidentally produced replicates:
three CIFAR-100 `lc` runs at identical configurations, down to the poisoned index set,
read **0.873 / 0.545 / 0.785**, and three GTSRB `lc` runs read **0.464 / 0.239 / 0.129**.
Spreads of 0.33 and 0.34.

Those runs predate the seeding commit, so they differ only by an unrecorded random
initialisation and shuffle. They are, in effect, free seed replicates, and they say that
a hard threshold applied to one training run is not a reliable classification. Every new
cell in the clean-label work is therefore trained at seeds 0 to 4, and reported as a mean
with a spread rather than as a single number.

## What to do next, in priority order

1. **Land the clean-label fix.** Bases, then the epsilon pilot, then seeds 0-4. Commands
   in [clean-label-rate-caps.md](clean-label-rate-caps.md).
2. **Extend the WaNet sweep to strength 8 and 16** on CIFAR-10 and CIFAR-100 at 1%, and
   accept the smallest strength clearing 0.85 with clean-accuracy drop no worse than
   -0.05. GTSRB is already answered at 4.0.
3. **Retrain the two broken runs** (`vit_gtsrb_tact_0_01`, `vit_gtsrb_badnet_a2a_0_01`).
4. **Test the Adaptive-Blend asymmetry fix** at seeds 0 to 4, since seed sensitivity is
   the whole of its failure mode.
5. **Decide the `sig` policy.** Its GTSRB cells inherit the target-class fix, but
   CIFAR-100 and Tiny are capped at 1% and 0.5% and `sig` does not implant there at any
   reachable rate. Either raise the sinusoid amplitude above Barni's 40/255 and say so, or
   report that SIG does not implant at the maximum achievable clean-label rate on
   many-class datasets. The second is a legitimate finding rather than a gap.
6. **Re-read every table that used a clean-label row**, since those rows were reporting
   rates that were never applied.
