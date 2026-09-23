---
name: citation-audit
description: paper/references.bib conventions and the 2026-09-23 citation audit, including the 8 uncited entries the user must rule on
metadata:
  type: reference
---

`docs/citation-audit-2026-09-23.md` holds the full inventory of `paper/references.bib`, 56
entries against 48 used keys, with per-entry venue verification. The bib convention is
`@inproceedings` with the venue spelled out plus `note={arXiv:NNNN.NNNNN}` on published work,
`@article` with `journal={arXiv preprint arXiv:...}` only for genuinely unpublished work, and
every acronym or proper noun in a title braced because `plainnat` lowercases titles.

8 entries are uncited and their fate is the user's call, not an agent's: `wu2022backdoorbench`,
`foret2021sam`, `madry2018pgd` and `xiong2020layernorm` each mark a method the prose uses with
no source, and `cohen2019smoothing`, `mitchell2023detectgpt`, `rajabi2023mdtd` and
`wang2025a2x` are residue from withdrawn text.

**Why:** the user asked for no dead citations, no overciting and no missed work, and a reviewer
punishes a missing ViT-specific backdoor defense hardest. The largest gap found was that
`paper/sections/method.tex` carries 0 citations while asserting priority over perturbation-site
choice.

**How to apply:** before adding a citation, check the audit for whether the work is already
there under a surprising key. When verifying a venue, arXiv's API works through the proxy but
DBLP is behind an Anubis challenge and Semantic Scholar rate-limits without a key. See
[[literature-folder-layout]].
