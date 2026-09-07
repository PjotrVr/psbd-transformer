"""Make the pre-rewrite tree importable, for the equivalence tests only.

The rewrite moved the library into `psbd/`. The old modules live on under `old/`
during the transition, and the equivalence tests import BOTH so they can assert
the 2 produce identical results. Those tests are the evidence that the rewrite
preserved behaviour, so they have to keep running until the old tree is deleted.

The old modules import each other by their original top-level names (`from poison
import Attack`, `from utils.config import DATASET_REGISTRY`). Putting `old/` on
the path makes every one of those resolve unchanged, which is the point: code
scheduled for deletion should not be refactored to satisfy its own removal.

There is no name collision with the new package. The old names are `poison`,
`models`, `attacks` and so on, the new ones are `psbd.poisoning`, `psbd.models`,
`psbd.attacks`, so both are importable at once and a test can compare them
directly.

This file disappears with `old/`.
"""

import os
import sys

OLD_TREE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "old")

if os.path.isdir(OLD_TREE) and OLD_TREE not in sys.path:
    # Appended rather than prepended, so a name that exists in both trees resolves
    # to the live one. Nothing should rely on that, but the failure mode if
    # something did is a stale import winning silently.
    sys.path.append(OLD_TREE)
