"""Configuration and dataset helpers shared across the rest of the repo.

models.py stays at repo root rather than moving here, since it's loaded and
built from nearly every entrypoint and package the same way train_backdoor.py
and train_benign.py are. Import the specific submodule you need explicitly,
for example `from utils.config import DATASET_REGISTRY`. This package
deliberately does not re-export its submodules.
"""
