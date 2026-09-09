# Fixing the clean-label attacks: a target class that can be poisoned, and the Label-Consistent step that was missing

**The cap itself is not new here.** `docs/audit-2026-09-07.md` finding A8 established it
on 2026-09-07, `notebooks/04-datasets-and-splits.ipynb` derives it from the library, and
every `args.json` already carries `realized_poison_rate` and `poison_rate_capped` from
that work. What A8 did not say is what to do about it.

This document is the fix, written 2026-09-09: which rates are reachable and which are
permanently out of reach, how GTSRB gets a usable clean-label cell, and why
Label-Consistent underperformed even at the cap (it was missing the adversarial
pre-perturbation that makes the attack work).

## The arithmetic

A clean-label attack may only poison images that already carry the target label,
because it keeps the label and relies on the trigger becoming the easier feature.
So the highest reachable poison rate is

    max_rate = |target class| / |training set|

`poison.choose_poison_indices` clamps to this silently
([poison.py:214](../poison.py#L214)): `count = min(round(rate * n), len(eligible))`.
When the requested count exceeds the pool, `rng.choice(eligible, size=len(eligible),
replace=False)` returns a permutation of the whole pool, so **every rate above the cap
selects the identical index set**. Verified directly, not inferred.

| dataset | classes | class sizes | cap at class 0 | best cap over classes | reachable of 0.5 / 1 / 5 / 10% |
|---|---:|---|---:|---:|---|
| cifar10 | 10 | uniform 5000 | 10.00% | 10.00% | all four |
| cifar100 | 100 | uniform 500 | 1.00% | 1.00% | 0.5, 1 |
| gtsrb | 43 | **150 to 1500** | 0.56% | **5.63%** (class 1 or 2) | 0.5, 1, 5 |
| tiny | 200 | uniform 500 | 0.50% | 0.50% | 0.5 only |

**189 of the 316 clean-label checkpoint folders carry a rate the dataset clamped**
(99 ViT, 90 Swin; A8 counted 180 on 2026-09-07 and the set has grown since, with 188 of
them already carrying A8's backfilled `poison_rate_capped` flag). On Tiny all four nominal rates resolve to the same 500 images, so its
clean-label "rate curve" is four replicates of one point. On CIFAR-100 and GTSRB, two
of the three panel rates collapse onto one.

What this forecloses, permanently and regardless of any code change:

- **10% clean-label is impossible except on CIFAR-10.**
- **5% clean-label is impossible on CIFAR-100 and Tiny.** GTSRB reaches it only at a
  target class larger than class 0.
- **1% clean-label is impossible on Tiny.** Its ceiling is 0.5%.

A consequence worth stating separately: the panel declares rates `[0.01, 0.05, 0.1]`
([configs/psbd_basis.json](../configs/psbd_basis.json)), so **Tiny clean-label cells
cannot be panel cells at all**, and CIFAR-100 contributes only its 1% cell. That is a
property of the datasets, not a gap in the sweep.

## Fix 1: GTSRB's target class was a self-imposed handicap

A8 recorded GTSRB as the worst-overstated dataset (17.8x) and stopped there. The reason it
is worst is also the reason it is the only one that can be fixed.

GTSRB is the only dataset where the choice of target class matters, because it is the
only one that is not class-uniform. Class 0 (speed limit 20) holds 150 of 26,640
images; classes 1 and 2 hold 1,500 each, a 10x larger pool that lifts the cap from
0.56% to 5.63%.

Nothing in the threat model forces an attacker to pick the rarest class; an attacker
picking a target picks one they can actually poison. Clean-label runs on GTSRB
therefore use **target class 1**, tagged `_tl1` in the folder name. Dirty-label GTSRB
cells keep class 0, because the cap does not bind for them and switching would
invalidate the existing GTSRB panel for no gain.

Because that leaves clean-label and dirty-label GTSRB cells at different target
classes, a control set of dirty-label runs (`badnet_a2o`, `blend`) is trained at target
1 as well. If detection AUROC moves with the target class, the switch is a confound
rather than a fix, and the control is what shows it either way.

This does not rescue 10%: 10% of 26,640 is 2,664 and the largest class holds 1,500. The
GTSRB clean-label rows stop at 5%, and that is a fact about the dataset.

The split question is separate and does not change any of this. This repo uses
torchvision's 26,640-image GTSRB archive rather than the literature's 39,209
([gtsrb-training-split-mismatch.md](gtsrb-training-split-mismatch.md)). New cells stay
on the current split so they remain comparable with the 329 GTSRB runs already on disk;
on the larger archive class 1 holds 2,250 for a 5.74% cap, and the reachability row is
unchanged.

## Fix 2: Label-Consistent was missing the step that makes it work

The docstring said so from the beginning. Turner et al. (2019) do not simply stamp a
patch on target-class images: they first perturb each base image adversarially, so its
natural features stop supporting its own label and the trigger becomes the only cue
that reliably does. Without that step the model can still read the class off the
untouched picture and has no reason to prefer the trigger, which is exactly why the
patch-only variant needed nearly the whole target class before it implanted.

This is not a detail the PSBD paper glossed over either. Its appendix records using
precomputed adversarial images throughout: *"On CIFAR-10, we directly use the
adversarial images provided by the original paper; on GTSRB and Tiny ImageNet we use
the adversarial images provided by BackdoorBench"*
(`papers/PSBD/sec/7_appendix.tex:543-546`).

### What was implemented

`psbd/adversarial.py` generates the bases with untargeted L-infinity PGD against a
model trained on clean data, maximising cross-entropy on each image's **true** label:

    x^{t+1} = Proj_{B_eps(x) ∩ [0,1]} ( x^t + alpha * sign( grad_x L(f(x^t), y) ) )

with Madry's step size `alpha = 2.5 * epsilon / steps`, 100 steps, and a random start
inside the ball. `cli/lc_bases.py` is the runner.

Everything happens at the dataset's **native** resolution in 0-to-1 pixel space,
because that is where triggers are applied: `train_backdoor.base_transform` stops at
`ToTensor`, the trigger goes on, and normalisation comes last
([poison.py:236-247](../poison.py#L236-L247)). PGD therefore backpropagates through
normalisation and through the model wrapper's own Resize to 224.

**One surrogate serves every victim.** The bases are generated once per
`(dataset, target label, epsilon)` from `checkpoints/vit_{dataset}_benign` and used to
train both ViT and Swin victims. That is the faithful threat model, since the attacker
publishes a poisoned dataset rather than a model, and it avoids needing Swin benign
checkpoints that exist only for CIFAR-100.

### Train and eval are deliberately asymmetric

`apply_trigger` substitutes the adversarial base and then stamps; `apply_trigger_eval`
stamps only. This is required, not a convenience: at eval time `is_eval_poisonable`
flips clean-label eligibility to *non*-target classes
([poison.py:169-170](../poison.py#L169-L170)), and those have no adversarial base by
construction. `AttackSuccessSet` already selects the eval trigger when one exists
([poison.py:295](../poison.py#L295)); Adaptive-Blend set the precedent.

`stealth.py` was changed to read through the eval trigger for the same reason. It holds
*test* indices and would otherwise miss every entry in a train-indexed cache.

### The defaults are inert, on purpose

`adversarial_dir` defaults to `""` and `adversarial_epsilon` to `0.0`, which reproduces
the patch-only behaviour byte for byte. Every LC checkpoint trained before this change
rebuilds through `default_config("lc")`, so a drift here would silently re-interpret
them. `tests/test_rewrite_library.py` pins the two attack trees bit-for-bit and
`test_lc_default_config_is_the_patch_only_variant` pins the default path.

A poisoned index with no base falls back to the clean image rather than raising,
because analysis code holding test indices reaches the same closure. That fallback must
never fire during training, so `build_training_set` calls `missing_adversarial_bases`
and refuses before training starts. One consequence: `--max-samples` is incompatible
with adversarial LC, because `limit_dataset` makes indices subset-local and the cache is
keyed on raw dataset indices. It fails loudly rather than training the weak variant.

### Evidence the perturbation does its job

Measured on the surrogate, before any victim is trained. If target-class accuracy does
not collapse, the bases carry no attack and no amount of training will produce one.

| dataset | target class | epsilon | steps | surrogate accuracy on the target class |
|---|---:|---|---:|---|
| gtsrb | 1 | 16/255 | 10 | 1.000 -> 0.043 |
| cifar100 | 0 | 16/255 | 5 | 0.984 -> 0.000 |

The saved tensors were checked to stay inside the L-infinity ball (max \|delta\| 0.0627
against a 0.0627 budget) and inside `[0, 1]`.

## Naming and panel registration

Folders follow the canonical template plus two new tags:

    {arch}_{dataset}_{attack}_{rate_tag}[_tl{N}][_adv][_seed_{n}]

`_tl{N}` marks a non-default target class, `_adv` marks Label-Consistent carrying its
adversarial bases. Both were checked against every substring filter in the repo and
collide with none of `sam_rho`, `evade`, `_ep`, `a2a`, `a2m`, `_trig`, `seed_`,
`benign`.

Since the old patch-only LC runs stay on disk and some of them clear the ASR bar
(`vit_cifar10_lc_0_1` reads 0.979), both variants would otherwise compete for the same
panel slot. `configs/psbd_basis.json` now declares

```json
"canonical_variants": {"lc": "_adv"}
```

so only the faithful variant is a panel cell. Two related fixes went in alongside:

- `scripts/coverage_ledger.py`'s `pivot_configs` keyed cells by `(dataset, rate) ->
  {attack: cell}`, so two cells sharing a slot silently overwrote each other by sort
  order. It now prefers the cell that clears the ASR bar and raises on a genuine tie
  rather than picking one.
- Pilot runs that sweep epsilon carry `_pilot`, which is an excluded token, so the
  diagnostic runs never enter the headline.

## What the old numbers were, and what replaces them

The superseded clean-label rows are not deleted; they were real runs and they stay on
disk. They are simply read at their realized rate now, with replicate count, instead of
being printed as a rate curve.

The collapse is also useful. `vit_cifar100_lc_0_01`, `_0_05` and `_0_1` are the same
configuration down to the poisoned index set, and they read ASR 0.873 / 0.545 / 0.785;
the three GTSRB LC cells read 0.464 / 0.239 / 0.129. Those spreads, 0.33 and 0.34, are
free replicates and they are the honest error bar on a single training run. A hard
0.85 ASR bar applied to one run is shaky at that spread, which is why the replacement
cells are trained at seeds 0 to 4 rather than at seed 0 alone.

## Reproducing

```bash
python pbs/generate_cleanlabel_jobs.py --stage bases     # PGD bases, all datasets
python pbs/generate_cleanlabel_jobs.py --stage pilot     # epsilon in {8,16,32}/255, seed 0
python pbs/generate_cleanlabel_jobs.py --stage full --epsilon cifar10=16 gtsrb=16
```

The generator computes each dataset's reachable rates from the data itself and emits
only those, so a clamped rate cannot be requested again by accident. Epsilon is chosen
from the pilot by the rule already used for the WaNet strength sweep: the **smallest**
value reaching ASR >= 0.85 with a clean-accuracy drop no worse than -0.05 against the
dataset's benign reference.

The pilot writes one folder per epsilon (`..._adv16_pilot`), and `_pilot` is an excluded
panel token, so the diagnostic runs never reach the headline. The full stage writes
`_adv` and records the chosen epsilon through `adversarial_dir` in `args.json`.

### One trap worth knowing, if you write the commands by hand

Pass both override keys in a **single** `--attack-override` flag:

```bash
--attack-override adversarial_dir=results/lc_adversarial/gtsrb_tl1_eps16 \
                  adversarial_epsilon=0.062745
```

Repeating the flag used to keep only the last key, because the argument was declared
`nargs="*"` rather than an accumulating action, and losing `adversarial_dir` silently
reverts the attack to its patch-only variant. That is fixed (`action="extend"`, so both
spellings now work) and a config carrying an epsilon with no directory is refused before
training starts, but the single-flag form cannot regress even if that changes again.
