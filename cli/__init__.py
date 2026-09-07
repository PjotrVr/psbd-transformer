"""Command line entrypoints for PSBD-ViT.

Every module here is a thin shell over psbd/: an argparse parser, a short piece of
orchestration, and a main(). All the real work lives in the library, so an
entrypoint can be read end to end without leaving the file, and any logic worth
testing is testable without a command line.

Run one as a module from the repository root, for example
``python -m cli.analyze --all``, so psbd/ resolves without a PYTHONPATH tweak.

This package deliberately re-exports nothing.
"""
