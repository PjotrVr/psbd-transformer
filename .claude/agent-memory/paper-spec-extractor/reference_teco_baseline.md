---
name: reference-teco-baseline
description: TeCo (CVPR 2023) spec sources and the compounding-corruption bug in both official reference implementations
metadata:
  type: reference
---

TeCo = "Detecting Backdoors During the Inference Stage Based on Corruption Robustness Consistency" (CVPR 2023, arXiv 2303.18191). Test-time trigger-sample detection baseline; black-box, hard-label, data-free.

Where to read it:
- Paper text with equations renders reliably at `https://ar5iv.labs.arxiv.org/html/2303.18191`. The CVF PDF at openaccess.thecvf.com returns HTTP 403 to WebFetch, and `arxiv.org/pdf/2303.18191` comes back as undecodable binary. ar5iv duplicates math symbols in its HTML ("55 times" means "5 times", "II" means "I") -- do not quote its digits blindly.
- Two official implementations, both real: `CGCL-codes/TeCo` at `BackdoorBench-v1.0-merge/defense/teco/teco.py` (authors' own, BackdoorBench v1 fork) and `SCLBD/BackdoorBench` at `detection_infer/teco.py` (upstream v2 port). They differ in clean-set sampling.
- GitHub raw URLs work with WebFetch; the rendered github.com HTML page does not expose file contents.

**Why:** Anyone reproducing or comparing against TeCo needs the exact statistic, and the single most consequential fact is not in the paper.

**How to apply:** Both reference implementations mutate the image list in place inside the severity loop (`x = images_poison; x[i] = dg(x[i], args)`), so corruptions compound across all 5 severities AND across all 15 corruption types instead of each being applied to the pristine image as Algorithm 1 specifies. Any reimplementation that follows Algorithm 1 literally will not reproduce the published numbers. Flag this whenever TeCo is used as a baseline, and state which convention was implemented. Second divergence: the code's reported "F1" is `sklearn.f1_score(..., average='micro')` on binary labels, which equals accuracy, not the paper's Equation (5) F1.

Related: [[feedback-no-corner-cutting]]
