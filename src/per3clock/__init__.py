"""per3clock: shared code for the PER3 genotype × clock-gene timing project (GSE39445).

Modules
-------
config     paths and constants (gene lists, period, seed)
data       data loading and acquisition (Part 1)
qc, eda    preprocessing / QC and exploratory analysis (Part 2)
rhythm     cosinor fitting and rhythmicity scan (Part 3)
circstats  circular statistics for acrophase comparison (Part 3)
interpret  enrichment / network interpretation (Part 4)
ml         circadian-phase prediction (Part 5)
plotting   shared plot style and figure saving
"""

__version__ = "0.1.0"
