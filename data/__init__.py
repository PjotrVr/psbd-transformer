"""Datasets: the registry of static facts, loading, splits and BackdoorBench input.

registry holds what is true about a dataset regardless of any experiment (class
count, normalization constants, native resolution). loading turns a name into
tensors. splits carves the test set the way a PSBD sweep needs it. backdoorbench
reads poisoned PNGs published by BackdoorBench, the one path where images arrive
already triggered.

Import the submodule you need. This package re-exports nothing.
"""
