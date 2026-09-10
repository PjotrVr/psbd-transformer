"""Latent-space analysis: where a trigger lives in the residual stream and how it looks.

cases loads a checkpoint into paired clean and backdoor features with a single
call. features extracts the per-block residual stream. direction, cka, embedding,
lipschitz and distribution each answer a question about those features. latent
is the worked example cli.analyze_latent runs, and stealth measures how visible a
trigger is in pixel space, which needs no model at all.

Import the submodule you need, for example from analysis.cka import
debiased_linear_cka. This package re-exports nothing, which keeps the module a
symbol lives in visible at the point of use.
"""
