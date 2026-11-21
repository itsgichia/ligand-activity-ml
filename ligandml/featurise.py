"""Molecular representations for ligand-based activity prediction.

Three families of 2D representation, all computed with RDKit:

RDKit 2D descriptors
    A curated block of interpretable physicochemical and topological
    descriptors (molecular weight, cLogP, TPSA, ring and rotatable-bond
    counts, fraction of sp3 carbon, the Chi/Kappa connectivity indices,
    charge and E-state summaries).  Dense, low-dimensional, and the only
    representation whose individual columns mean anything to a chemist.
Morgan / ECFP fingerprints
    Circular substructure fingerprints at two radii and two bit sizes.
    ``ecfp4_2048`` (radius 2) is the field's default workhorse;
    ``ecfp6_2048`` (radius 3) sees larger environments; ``ecfp4_1024``
    tests whether halving the bit budget costs anything.
MACCS keys
    166 hand-curated structural keys plus a padding bit.  A fixed,
    human-designed vocabulary, and a useful low-capacity baseline.

Every featuriser returns a dense ``float64`` array with one row per input
molecule and a matching list of column names, so that downstream code can
treat all representations identically.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, MACCSkeys
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

LOGGER = logging.getLogger(__name__)
RDLogger.DisableLog("rdApp.*")

#: Names of the RDKit descriptor functions used for the dense
#: representation.  Chosen to be interpretable, cheap and free of the
#: fragile fragment counts that are mostly zero on drug-like sets.
DESCRIPTOR_NAMES: Final[tuple[str, ...]] = (
    "MolWt",
    "HeavyAtomCount",
    "MolLogP",
    "MolMR",
    "TPSA",
    "LabuteASA",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "RingCount",
    "NumAromaticRings",
    "NumAliphaticRings",
    "NumSaturatedRings",
    "FractionCSP3",
    "NumHeteroatoms",
    "NHOHCount",
    "NOCount",
    "HeavyAtomMolWt",
    "BalabanJ",
    "Chi0v",
    "Chi1v",
    "Chi2v",
    "Chi3v",
    "Chi4v",
    "Kappa1",
    "Kappa2",
    "Kappa3",
    "HallKierAlpha",
    "MaxPartialCharge",
    "MinPartialCharge",
    "MaxAbsPartialCharge",
    "MinAbsPartialCharge",
    "MaxEStateIndex",
    "MinEStateIndex",
)
"""Descriptor functions in the dense block.

Deliberately excludes the few RDKit descriptors that dominate the cost of
featurising tens of thousands of molecules (``BertzCT``, ``qed``) without
adding information the rest of the block does not already carry, and the
sparse fragment counts, which are mostly zero on drug-like sets.
"""


@dataclass(frozen=True, slots=True)
class Representation:
    """A named molecular representation.

    Attributes:
        name: Short slug used in result tables, e.g. ``"ecfp4_2048"``.
        label: Human-readable description.
        kind: ``"descriptors"`` or ``"fingerprint"``.  Dense descriptor
            blocks need feature scaling for linear models and kernels;
            binary fingerprints do not.
        builder: Callable mapping molecules to ``(matrix, feature_names)``.
    """

    name: str
    label: str
    kind: str
    builder: Callable[[Sequence[Chem.Mol]], tuple[np.ndarray, list[str]]]

    def __call__(
        self, mols: Sequence[Chem.Mol]
    ) -> tuple[np.ndarray, list[str]]:
        """Compute the representation for a sequence of molecules."""
        return self.builder(mols)


def mols_from_smiles(smiles: Sequence[str]) -> tuple[list[Chem.Mol], np.ndarray]:
    """Parse SMILES into RDKit molecules, reporting which ones succeeded.

    Args:
        smiles: SMILES strings.

    Returns:
        A tuple of the parsed molecules (failures omitted) and a boolean mask
        over the input marking which entries parsed.
    """
    mols: list[Chem.Mol] = []
    keep = np.zeros(len(smiles), dtype=bool)
    for index, text in enumerate(smiles):
        mol = Chem.MolFromSmiles(text) if isinstance(text, str) else None
        if mol is None:
            LOGGER.warning("unparseable SMILES at row %d: %r", index, text)
            continue
        mols.append(mol)
        keep[index] = True
    return mols, keep


def rdkit_descriptors(
    mols: Sequence[Chem.Mol], names: Sequence[str] = DESCRIPTOR_NAMES
) -> tuple[np.ndarray, list[str]]:
    """Compute a block of interpretable RDKit 2D descriptors.

    Non-finite values (RDKit returns NaN for partial charges on a few exotic
    valences) are replaced with 0.0, which is the neutral value for every
    descriptor in :data:`DESCRIPTOR_NAMES` after scaling.

    Args:
        mols: Parsed RDKit molecules.
        names: Descriptor function names to compute.

    Returns:
        An ``(n_molecules, n_descriptors)`` array and the column names.

    Raises:
        KeyError: If a requested descriptor name is unknown to RDKit.
    """
    registry = dict(Descriptors.descList)
    try:
        functions = [(name, registry[name]) for name in names]
    except KeyError as exc:  # pragma: no cover - guards a typo in the constant
        raise KeyError(f"unknown RDKit descriptor {exc.args[0]!r}") from exc

    matrix = np.zeros((len(mols), len(functions)), dtype=np.float64)
    for row, mol in enumerate(mols):
        for col, (_, function) in enumerate(functions):
            try:
                value = float(function(mol))
            except Exception:  # pragma: no cover - a few descriptors throw
                value = 0.0
            matrix[row, col] = value if np.isfinite(value) else 0.0
    return matrix, [name for name, _ in functions]


def morgan_fingerprints(
    mols: Sequence[Chem.Mol], radius: int = 2, n_bits: int = 2048
) -> tuple[np.ndarray, list[str]]:
    """Compute binary Morgan (ECFP) fingerprints.

    Args:
        mols: Parsed RDKit molecules.
        radius: Morgan radius; 2 gives ECFP4, 3 gives ECFP6.
        n_bits: Length of the folded bit vector.

    Returns:
        An ``(n_molecules, n_bits)`` array of 0.0/1.0 and the bit names.
    """
    generator = GetMorganGenerator(radius=radius, fpSize=n_bits)
    matrix = np.zeros((len(mols), n_bits), dtype=np.float64)
    for row, mol in enumerate(mols):
        for bit in generator.GetFingerprint(mol).GetOnBits():
            matrix[row, bit] = 1.0
    diameter = 2 * radius
    return matrix, [f"ecfp{diameter}_{index}" for index in range(n_bits)]


def maccs_keys(mols: Sequence[Chem.Mol]) -> tuple[np.ndarray, list[str]]:
    """Compute MACCS structural keys.

    RDKit returns 167 bits; bit 0 is unused padding and is kept so that bit
    indices match the published MACCS key numbering.

    Args:
        mols: Parsed RDKit molecules.

    Returns:
        An ``(n_molecules, 167)`` array of 0.0/1.0 and the key names.
    """
    matrix = np.zeros((len(mols), 167), dtype=np.float64)
    for row, mol in enumerate(mols):
        for bit in MACCSkeys.GenMACCSKeys(mol).GetOnBits():
            matrix[row, bit] = 1.0
    return matrix, [f"maccs_{index}" for index in range(167)]


#: The representations compared in the study.
REPRESENTATIONS: Final[tuple[Representation, ...]] = (
    Representation(
        "rdkit2d",
        f"RDKit 2D descriptors ({len(DESCRIPTOR_NAMES)})",
        "descriptors",
        rdkit_descriptors,
    ),
    Representation(
        "ecfp4_1024",
        "Morgan radius 2, 1024 bits (ECFP4)",
        "fingerprint",
        lambda mols: morgan_fingerprints(mols, radius=2, n_bits=1024),
    ),
    Representation(
        "ecfp4_2048",
        "Morgan radius 2, 2048 bits (ECFP4)",
        "fingerprint",
        lambda mols: morgan_fingerprints(mols, radius=2, n_bits=2048),
    ),
    Representation(
        "ecfp6_2048",
        "Morgan radius 3, 2048 bits (ECFP6)",
        "fingerprint",
        lambda mols: morgan_fingerprints(mols, radius=3, n_bits=2048),
    ),
    Representation(
        "maccs",
        "MACCS structural keys (167)",
        "fingerprint",
        maccs_keys,
    ),
)

REPRESENTATIONS_BY_NAME: Final[dict[str, Representation]] = {
    representation.name: representation for representation in REPRESENTATIONS
}
"""Lookup from slug to :class:`Representation`."""

#: The representation used for every analysis that is not itself a
#: representation comparison: the field's default, and the one against which
#: new methods are normally reported.
DEFAULT_REPRESENTATION: Final[str] = "ecfp4_2048"


def get_representation(name: str) -> Representation:
    """Look up a representation by slug.

    Args:
        name: Representation slug, e.g. ``"maccs"``.

    Returns:
        The matching :class:`Representation`.

    Raises:
        KeyError: If the slug is unknown.
    """
    try:
        return REPRESENTATIONS_BY_NAME[name]
    except KeyError as exc:
        known = ", ".join(sorted(REPRESENTATIONS_BY_NAME))
        raise KeyError(f"unknown representation {name!r}; known: {known}") from exc


def featurise_frame(
    frame: pd.DataFrame,
    representation: str | Representation,
    smiles_column: str = "canonical_smiles",
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """Featurise a dataset, dropping rows whose SMILES will not parse.

    Args:
        frame: A cleaned activity table.
        representation: Representation slug or object.
        smiles_column: Column holding the SMILES strings.

    Returns:
        A tuple of the feature matrix, the feature names, and the subset of
        ``frame`` (index reset) that the matrix rows correspond to.

    Raises:
        KeyError: If ``smiles_column`` is absent or the slug is unknown.
    """
    if smiles_column not in frame.columns:
        raise KeyError(f"frame has no column {smiles_column!r}")
    if isinstance(representation, str):
        representation = get_representation(representation)
    mols, keep = mols_from_smiles(frame[smiles_column].tolist())
    kept = frame.loc[keep].reset_index(drop=True)
    matrix, names = representation(mols)
    return matrix, names, kept
