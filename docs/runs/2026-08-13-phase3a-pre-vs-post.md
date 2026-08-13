# Run 2026-08-13, phase 3a: pre_residual vs post_residual

Launched from commit `baeaf159250dd694bf3bfe3a97a80f13c0759b0c`.

- **Question**: does pre-residual dropout beat post-residual on ViT-B/16
  (hypotheses H1, H3, H9)?
- **Grid**: CIFAR-10 ViT, 5 attacks (badnet_a2o, blend, bpp, lf, badnet_a2a)
  x 3 poison rates (0.01, 0.05, 0.1) + `vit_cifar10_benign` as the negative
  control, probed with the BadNet trigger. 16 checkpoints x 2 placements = 32 jobs.
- **Gate**: every checkpoint passed the ASR >= 0.8 viability check.
- **Per job**: 9 dropout rates x 3 splits x k=3 passes over the full 10000-image
  CIFAR-10 test split. 15 min walltime, 1 GPU.
- **Job IDs**: 1017932 through 1017963.
- **Output**: `results/<folder>/psbd/` (raw tensors, gitignored) and
  `results/<folder>/psbd_metrics.json` (derived, committed).
