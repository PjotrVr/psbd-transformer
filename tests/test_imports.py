"""Every module in every package, plus every root-level entrypoint, still imports.

A reorganization's real failure mode is a broken import nobody happens to
exercise in another test, so this is deliberately exhaustive rather than
relying on the rest of the suite to catch it incidentally. One test case per
module, parametrized, so a break names exactly which one broke.
"""

import importlib

import pytest

MODULES = [
    "adaptive_evasion",
    "backdoor_data",
    "baseline_detect",
    "defence_tables",
    "detector_comparison",
    "detector_fusion",
    "evaluate",
    "loaders",
    "metrics",
    "models",
    "pbs_grid",
    "poison",
    "psbd_analyze",
    "psbd_dropout_sweep",
    "psbd_head_profile",
    "psbd_operating_points",
    "psbd_report",
    "psbd_variants",
    "sam",
    "stealth",
    "train",
    "train_backdoor",
    "train_benign",
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
    "analysis",
    "analysis.analyze_latent",
    "analysis.cka",
    "analysis.direction",
    "analysis.embedding",
    "analysis.features",
    "analysis.lipschitz",
    "defences",
    "defences.baselines",
    "defences.checkpoint_eval",
    "defences.detection",
    "defences.dropout",
    "defences.inference",
    "defences.perturbations",
    "defences.psbd_cache",
    "defences.psbd_metrics",
    "utils",
    "utils.config",
    "utils.datasets",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name):
    importlib.import_module(module_name)
