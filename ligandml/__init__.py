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
    Random, Bemis-Murcko scaffold and publication-year time splits,
    including a class-stratified scaffold split for screening libraries.
``models``
    Regression and classification estimators with sensible defaults.
``evaluate``
    Metrics, including the screening metrics (enrichment factor, BEDROC),
    calibration and bootstrap confidence intervals.
``domain``
    Applicability domain by nearest-neighbour Tanimoto similarity.
``nulls``
    Y-scrambling (label permutation) null models.
``uncertainty``
    Spread on the split-gap estimate: across targets and models, and
    across repeated draws of the same split.
``decoys``
    Property-matched decoy libraries, so the screening metrics have room
    to discriminate instead of sitting on their ceiling - with a distance
    cap and a per-property bias breakdown, because nearest-available
    matching will happily return a decoy from the wrong region and report
    nothing amiss.
``conformal``
    Prediction sets with a proved coverage guarantee, class-conditional
    so the promise holds for the actives and not just on average.
"""

from __future__ import annotations

__all__ = [
    "RANDOM_STATE",
    "__version__",
]

__version__ = "0.1.0"

RANDOM_STATE: int = 20261121
"""Global seed. Every stochastic step in the study derives from this value."""
