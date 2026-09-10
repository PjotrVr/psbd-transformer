"""The 2 backbones, and where a probe may attach inside each.

backbones builds and loads ViT-B/16 and Swin-S. positions owns the per-architecture
registry naming every point a perturbation can be injected, which is a fact about
the architecture rather than about the defence, and so lives here.

Import the submodule you need. This package re-exports nothing.
"""
