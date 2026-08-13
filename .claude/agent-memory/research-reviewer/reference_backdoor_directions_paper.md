---
name: reference-backdoor-directions-paper
description: Karayalcin et al. "Backdoor Directions in Vision Transformers" LaTeX source lives in papers/backdoor_directions/ and is the spec for analysis/
metadata:
  type: reference
---

The specification the `analysis/` subpackage reproduces is the LaTeX source of
Karayalcin, Krcek, Chen, Picek, "Backdoor Directions in Vision Transformers",
checked into the repo:

- `papers/backdoor_directions/main.tex` — backdoor direction `r^l` (mean paired
  difference), CLS vs all-token activation variants, best-layer selection
  (Eq. `backdoor_direction`, an ASR/RA-difference criterion), weight
  orthogonalization `W_new = W - r r^T W`, and the data-free head detector
  (`s_i`, `Z`, outlier at `Z > 3`) in Section `sec:detecting`.
- `papers/backdoor_directions/sec/sup.tex` — supplementary tables and the
  commented-out original derivation of the detection score.

Useful facts when checking fidelity: the paper's models are **BackdoorBench**
ViT-B/16 downloads (not locally trained), it considers **targeted dirty-label
attacks only**, and its supplementary cosine-similarity tables are indexed L1..L11
for a 12-block model, so the paper never states its layer-index convention
explicitly.

Related: [[project-latent-analysis-paper]].
