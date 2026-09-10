"""The command line entrypoints, and the only place a main() lives.

Every module here is a thin shell over the library packages: an argparse parser,
a short piece of orchestration and a main(). The real work lives in attacks/,
data/, defences/ and the rest, so an entrypoint reads end to end without leaving
the file and anything worth testing is testable without a command line.

Run any of them as a module from the repository root, for example
python -m cli.analyze --all, so the packages resolve without a PYTHONPATH tweak.
The pipeline in order: train_backdoor or train_benign, then sweep (stage 1, GPU),
then analyze (stage 2, CPU), then summary, report and tables.

This package re-exports nothing.
"""
