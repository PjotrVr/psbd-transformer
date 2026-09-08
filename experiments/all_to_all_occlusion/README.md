# Can a spatially selective probe detect an all-to-all backdoor? (H5 follow-up)

## Question

PSBD fails on all-to-all because dropout degrades the trigger channel and the natural
image channel together, so there is no asymmetry left to measure (H5,
`docs/all-to-all-inversion.md`). Occlusion is different: it is spatially selective, so
blanking the right patch should kill a localized trigger while the object survives, and
the prediction should fall back from `y + 1` to `y`.

That fallback is the signal, and reading it needs the permutation. The permutation is
**estimated, not assumed**, by Hungarian assignment on aggregate transition counts, and
calibrated against clean validation data.

## Status: written, not yet validated

The probe and its controls exist. The controls have not been run, so **no number from
this directory should be reported**. `run_controls.sh` documents the 2 that decide whether
it is a detector or an artifact:

1. a benign model probed with the all-to-all trigger. If this fires, the probe reacts to
   trigger-shaped pixels rather than to a backdoor, and the whole thing is worthless.
2. an all-to-one checkpoint, which should **not** fire.

## Known defects, to fix before it is run

Found during the 2026-09-08 audit and not yet corrected:

- **Unpaired populations.** `build_eval_loaders_from_attack` balances the clean and
  backdoor sets by class and then `AttackSuccessSet` filters, so on CIFAR-10 with
  `examples_per_class=150` the clean loader serves 1500 rows and the backdoor loader
  1350. Unlike the PSBD path it returns no manifest, so the pairing cannot be recovered,
  and `occlusion_probe.py` consumes the 2 as if they were aligned.
- **Split-half slice uses the wrong midpoint.** `half` is the *backdoor* split's midpoint
  (675) but is applied as a slice index into the *clean* split (1500), so the held-out
  comparison is 675 rows against 825 non-matching ones. `balance_by_class` emits in class
  order, so that tail is a specific set of high-numbered classes rather than a random
  half.
- No results file is written; output goes to stdout only.

## Why it is kept

It carries its own falsification machinery, which is the part worth preserving: an
ablation that removes the permutation readout, a clean-validation null permutation, and a
split-half transfer test. The defects above are in how the splits are built, not in that
design.

## Running it

    PYTHONPATH=. python experiments/all_to_all_occlusion/occlusion_probe.py --help
    bash experiments/all_to_all_occlusion/run_controls.sh
