"""Paper figures from the analysis statistics, 1 module per figure family.

The statistics live in analysis/ and never import matplotlib. This package
turns them into figures with 1 style (style.py) and 1 sidecar convention (every
figure writes a JSON of the numbers it plotted next to the PDF). cli.visualize
runs a tool by name against a checkpoint. Tools register in cheap_tools.py
(seconds to minutes per checkpoint) and expensive_tools.py (the Hessian
spectrum, the loss landscape and feature visualisation), 2 files so 2 authors
never edit the same registry.
"""
