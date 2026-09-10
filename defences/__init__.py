"""PSBD itself: perturb a forward pass, score the shift, decide.

The 4 stages in the order data flows through them. operators owns WHAT a probe does
once attached (models.positions owns where). inference runs the unperturbed and
perturbed passes. scores turns the cached passes into a number per sample. decision
turns numbers into a verdict against a threshold. cache is the on-disk layout the
2 sweep stages hand results through.

Import the submodule you need. This package re-exports nothing.
"""
