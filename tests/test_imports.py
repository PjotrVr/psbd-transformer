"""Every module in every package, plus every root-level entrypoint, still imports.

A reorganization's real failure mode is a broken import nobody happens to
exercise in another test, so this is deliberately exhaustive rather than
relying on the rest of the suite to catch it incidentally. One test case per
module, parametrized, so a break names exactly which one broke.
"""

import importlib

import pytest

MODULES = [
    # Root-level: shared pipeline core and entrypoints, deliberately not
    # folded into any package (see docs/plans/reorganization.md).
    "backdoor_data",
    "metrics",
    "models",
    "poison",
    "sam",
    "train",
    "train_backdoor",
    "train_benign",
    # attacks/: the registry plus its 10 sibling attack modules.
    "attacks",
    "attacks.adaptive_blend",
    "attacks.badnet",
    "attacks.blend",
    "attacks.bpp",
    "attacks.generated",
    "attacks.lc",
    "attacks.lf",
    "attacks.sig",
    "attacks.tact",
    "attacks.wanet",
    # analysis/: latent-space tools.
    "analysis",
    "analysis.analyze_latent",
    "analysis.cka",
    "analysis.direction",
    "analysis.embedding",
    "analysis.features",
    "analysis.lipschitz",
    # defences/: the PSBD detection mechanism.
    "defences",
    "defences.checkpoint_eval",
    "defences.detection",
    "defences.dropout",
    "defences.inference",
    # utils/: config and dataset helpers (models.py deliberately excluded).
    "utils",
    "utils.config",
    "utils.datasets",
    # plotting/: currently an empty scaffold.
    "plotting",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name):
    importlib.import_module(module_name)
