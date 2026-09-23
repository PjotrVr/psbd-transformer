---
name: literature-folder-layout
description: literature/ is a flat gitignored PDF library plus LaTeX-source subdirs; new downloads go to literature/<slug>/<slug>.pdf
metadata:
  type: reference
---

`literature/` holds the paper library. It is gitignored, so nothing there is under version
control and PDFs must be re-fetched rather than recovered from git. Layout as of 2026-09-23:
flat PDFs named `<topic>-<author>-<venue><year>.pdf`, a few LaTeX-source directories such as
`psbd-li-arxiv2024-source/`, `sam-poisoned-detection-zhang-arxiv2024-source/` and
`backdoor-directions-karayalcin-source/`, plus `literature/README.md`, an annotated catalog
with a "For us" note per paper. Newer downloads use `literature/<slug>/<slug>.pdf`.

**Why:** the annotated README is where the project records why a paper matters rather than
just that it exists, and PSBD's LaTeX source is the only way to read its appendix tables.

**How to apply:** grep `literature/README.md` before downloading anything, because STRIP,
SCALE-UP, TeCo, IBD-PSC, Beatrix, Qi's latent separability papers and PSBD itself are already
local. There is no pdftotext on the cluster, so extract text with `.venv/bin/python` and
`pypdfium2`. Network needs the proxy exports from CLAUDE.md.
