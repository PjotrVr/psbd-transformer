---
name: pdf-text-extraction
description: How to read a paper PDF on this cluster: WebFetch saves the binary to the tool-results dir, then extract with pypdfium2 via .venv/bin/python
metadata:
  type: reference
---

WebFetch cannot read a PDF and returns a note saying the content is compressed streams, but it
does save the file and prints the absolute path under the session's `tool-results/` directory.
Feed that path to `.venv/bin/python` with `pypdfium2`, looping
`doc[i].get_textpage().get_text_range()` over the pages and writing the result to the
scratchpad, then grep the text file. `ar5iv.labs.arxiv.org/html/<id>` works for older arXiv
papers that have no native HTML and returns the full body, which is how the Carlini 2019
robustness checklist was read.

**Why:** there is no pdftotext on the cluster and several central sources (the Arp et al.
USENIX PDF, ACM pages behind Cloudflare) are only reachable as PDFs.

**How to apply:** prefer `arxiv.org/html/<id>` first, then ar5iv, then the PDF path above.
`www.acm.org` returns a Cloudflare block for both WebFetch and curl, so cite ACM policy pages
from a mirror such as sigir.org and verify the wording by search. See
[[literature-folder-layout]].
