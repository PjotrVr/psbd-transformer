# The GTSRB training split this project uses is not the one the backdoor literature uses

Found 2026-09-07, while wiring up a reproduction of the original PSBD ResNet-18 results
(`scratch/psbd-upstream/`). Not found by a failing test: every GTSRB run in this repo
completed normally and produced plausible numbers.

## The finding

GTSRB ships two different training archives, and they are not the same size.

| archive | images | used by |
|---|---|---|
| `GTSRB-Training_fixed.zip` | **26,640** | `torchvision.datasets.GTSRB`, and therefore this repo |
| `GTSRB_Final_Training_Images.zip` | **39,209** | BackdoorBench, backdoor-toolbox, PSBD, and the backdoor literature generally |

This repo loads GTSRB through `tv_datasets.GTSRB` ([utils/datasets.py:52](utils/datasets.py#L52),
[psbd/data.py:73](psbd/data.py#L73)). torchvision's train split downloads
`GTSRB-Training_fixed.zip` (md5 `513f3c79a4c5141765e10e952eaa2478`) and reads
`root/gtsrb/GTSRB/Training/`, taking labels from the directory names. That archive is the
2010/2011 online-competition training set: 26,640 images.

PSBD gets GTSRB from BackdoorBench's `gtsrb_download.sh`, which fetches
`GTSRB_Final_Training_Images.zip` and reads `GTSRB/Final_Training/Images/`. The PSBD repo
then hard-codes the size it expects:

```python
# utils/supervisor.py, get_info_dataset
elif args.dataset == 'gtsrb':
    num_train_imgs = 39209
```

Verified on disk. `raw_data/gtsrb/gtsrb/` contains `GTSRB-Training_fixed.zip` extracted to
`GTSRB/Training/`, and the per-class `.ppm` counts sum to exactly 26,640, with each class's
`GT-000XX.csv` row count matching its file count. The data is not corrupt or partially
extracted; it is a complete copy of the smaller archive.

**The test split is fine.** Both sources use `GTSRB_Final_Test_Images.zip` (12,630 images)
and `GTSRB_Final_Test_GT.zip`, and both are already correct in `raw_data/`.

## Blast radius: verified, and it is small

329 checkpoint folders under `checkpoints/` are GTSRB runs. Three checks were run to size the
actual exposure, and all three come back clean.

**No result of ours is compared against a published GTSRB number.** The only cross-paper
comparison in the repo is `docs/results/comparison-to-published.md`, and every table in it is
CIFAR-10. GTSRB appears there once, in a caveat line noting which datasets PSBD and IBD-PSC
report on. Nothing to correct.

**No document states a GTSRB training-set size.** Grepping `docs/`, the paper sources and
`README.md` for either count returns nothing outside this file, so there is no factual claim
to fix.

**Every GTSRB comparison is internal.** All 329 runs used the same 26,640-image split, so
pre-versus-post, input-side-versus-residual-adjacent, operator rankings and the seed
replication are all like-for-like and stand as they are. GTSRB is also a completion dataset
here, not a primary one; the headline claims rest on CIFAR-100 and Tiny.

**The data itself is not defective.** All 43 classes are present, and each class's
`GT-000XX.csv` row count matches its `.ppm` file count exactly. This is a complete, correctly
labelled GTSRB training set, and it is precisely what `torchvision.datasets.GTSRB` gives
anyone who asks for the train split. It is a smaller release, not a broken one.

So the practical consequence is one sentence of hygiene: if a GTSRB number from this repo is
ever placed beside a published GTSRB number, or a paper table lists dataset sizes, say which
split was used. Re-running is optional, not a correction.

## The corrected data

`raw_data/new_gtsrb/` now holds the literature-standard split, laid out so that
`tv_datasets.GTSRB(root="raw_data/new_gtsrb")` reads it directly:

```
raw_data/new_gtsrb/gtsrb/
  GTSRB/Training/000{00..42}/*.ppm     39,209   <- GTSRB_Final_Training_Images.zip, relocated
  GTSRB/Final_Test/Images/*.ppm        12,630
  GT-final_test.csv
```

Note the deliberate relocation: the archive extracts to `GTSRB/Final_Training/Images/`, but
torchvision looks in `GTSRB/Training/`, so the directory is moved to that name. The class
subdirectories and their `GT-000XX.csv` files are preserved, which also makes this tree
readable by PSBD's own `dataset/GTSRB.py`.

`raw_data/` is gitignored, so this note is the tracked record that the directory exists and
what is in it.

## Re-running GTSRB

Point the dataset registry at the new root. In `utils/config.py` the GTSRB `DatasetSpec`
carries the data root; changing it to `raw_data/new_gtsrb` is enough for both
`utils/datasets.py` and `psbd/data.py`, since both go through `tv_datasets.GTSRB` with the
spec's root.

Given the blast radius above, re-running is optional and currently unmotivated: no claim in
the repo depends on the split. The corrected data is staged so the choice stays open.

The case for re-running would be a reviewer asking for GTSRB numbers on the same data as the
prior work, or a decision to add a GTSRB row to a cross-paper table. If that happens, re-run
only the cells that appear in that table rather than all 329.
