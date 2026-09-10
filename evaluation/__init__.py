"""What a trained model does: clean accuracy, attack success and the tables of both.

metrics evaluates a single checkpoint. loaders builds the clean and poisoned test
loaders it reads. summary loads the aggregated detection table, refusing rows that
are not safe to use.

This is the attack-side question (did the backdoor implant), distinct from
defences/, which asks whether a detector can find it.

Import the submodule you need. This package re-exports nothing.
"""
