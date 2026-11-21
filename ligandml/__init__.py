"""Ligand-based activity prediction, evaluated honestly.

A small research package for building and *critically* evaluating machine
learning models that predict whether a compound is active at a protein
target from its structure alone.

The modules are deliberately thin and independent so that each claim in the
project README traces back to one testable function:

``chembl``
    Fetch, clean and cache bioactivity data from the ChEMBL REST API.
``featurise``
    Molecular representations: RDKit 2D descriptors, Morgan/ECFP
    fingerprints and MACCS keys.
``splits``
    Random, Bemis-Murcko scaffold and publication-year time splits.
``models``
    Regression and classification estimators with sensible defaults.
``evaluate``
    Metrics, including the screening metrics (enrichment factor, BEDROC),
    calibration and bootstrap confidence intervals.
``domain``
    Applicability domain by nearest-neighbour Tanimoto similarity.
``nulls``
    Y-scrambling (label permutation) null models.
"""

from __future__ import annotations

__all__ = [
    "RANDOM_STATE",
    "__version__",
]

__version__ = "0.1.0"

RANDOM_STATE: int = 20261121
"""Global seed. Every stochastic step in the study derives from this value."""
