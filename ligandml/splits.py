"""Train/test split strategies, and why the choice of strategy is the study.

A virtual screen is always an extrapolation.  The model is trained on the
compounds a project has already tested and then asked about compounds it has
not, which in practice means *new chemistry* - new scaffolds, bought or
designed after the training data was published.  A random split asks a much
easier question: it scatters members of the same congeneric series across
both sides, so the test set is full of close analogues of training
compounds, and the model can score well by near-neighbour lookup rather than
by having learned any structure-activity relationship that generalises.

This module implements the three splits the study compares:

:func:`random_split`
    Uniformly random. The optimistic baseline, and the split most published
    numbers are quietly reported under.
:func:`scaffold_split`
    Bemis-Murcko scaffold split. Compounds sharing a ring-system scaffold
    are kept together, so no scaffold can straddle the boundary and the
    model is tested on chemotypes it has never seen. Largest scaffold
    groups go to training, which is the standard, deliberately hard
    construction.
:func:`time_split`
    Split on first-publication year from ChEMBL document years. The most
    realistic of the three, because it reproduces the actual temporal
    structure of a project: train on the past, predict the future.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

LOGGER = logging.getLogger(__name__)
RDLogger.DisableLog("rdApp.*")

ACYCLIC_SENTINEL: Final[str] = "<acyclic>"
"""Scaffold key for molecules with no ring system.

Bemis-Murcko reduces an acyclic molecule to the empty string.  Rather than
give every such molecule its own private group - which would quietly let
them behave like a random split - they are pooled under one sentinel group,
which is the conservative choice.
"""

DEFAULT_TEST_SIZE: Final[float] = 0.2
"""Target test fraction for every strategy, so the three are comparable."""

SPLIT_STRATEGIES: Final[tuple[str, ...]] = ("random", "scaffold", "time")
"""The strategies compared in the study, from most to least optimistic."""


@dataclass(slots=True)
class SplitResult:
    """One train/test partition.

    Attributes:
        strategy: ``"random"``, ``"scaffold"`` or ``"time"``.
        train_index: Positional indices of the training rows.
        test_index: Positional indices of the test rows.
        n_dropped: Rows excluded from the split entirely (time splits drop
            compounds with no usable publication year).
        details: Strategy-specific metadata for the results tables, such as
            the scaffold group count or the time-split cutoff year.
    """

    strategy: str
    train_index: np.ndarray
    test_index: np.ndarray
    n_dropped: int = 0
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def n_train(self) -> int:
        """Number of training rows."""
        return int(self.train_index.size)

    @property
    def n_test(self) -> int:
        """Number of test rows."""
        return int(self.test_index.size)

    @property
    def test_fraction(self) -> float:
        """Realised test fraction over the rows the split actually used."""
        total = self.n_train + self.n_test
        return self.n_test / total if total else 0.0

    def validate(self, n_rows: int) -> None:
        """Check the partition is a valid, disjoint subset of ``n_rows``.

        Args:
            n_rows: Size of the dataset the split was built from.

        Raises:
            ValueError: If the two sides overlap, if either is empty, or if
                any index is out of range.
        """
        train = set(self.train_index.tolist())
        test = set(self.test_index.tolist())
        if not train or not test:
            raise ValueError(f"{self.strategy} split left an empty side")
        if train & test:
            raise ValueError(
                f"{self.strategy} split has {len(train & test)} rows on both sides"
            )
        combined = train | test
        if combined and (min(combined) < 0 or max(combined) >= n_rows):
            raise ValueError(f"{self.strategy} split index out of range")
        if len(combined) + self.n_dropped != n_rows:
            raise ValueError(
                f"{self.strategy} split covers {len(combined)} + {self.n_dropped} "
                f"rows, expected {n_rows}"
            )


def murcko_scaffold(smiles: str, include_chirality: bool = False) -> str:
    """Return the Bemis-Murcko scaffold of a molecule, as SMILES.

    The Bemis-Murcko scaffold is the molecule's ring systems plus the linkers
    joining them, with all side chains removed.  Two compounds in the same
    congeneric series almost always share one.

    Args:
        smiles: A SMILES string.
        include_chirality: Keep stereochemistry in the scaffold. Off by
            default, so that stereoisomers group together.

    Returns:
        The scaffold SMILES, or :data:`ACYCLIC_SENTINEL` for a molecule with
        no ring system, or the input string unchanged if it cannot be parsed
        (which keeps unparseable rows in groups of their own rather than
        silently merging them).
    """
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return str(smiles)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol, includeChirality=include_chirality
    )
    return scaffold if scaffold else ACYCLIC_SENTINEL


def scaffold_groups(smiles: Sequence[str]) -> np.ndarray:
    """Map each molecule to its Bemis-Murcko scaffold.

    Args:
        smiles: SMILES strings.

    Returns:
        An array of scaffold SMILES, one per input, in input order.
    """
    cache: dict[str, str] = {}
    out: list[str] = []
    for text in smiles:
        key = str(text)
        if key not in cache:
            cache[key] = murcko_scaffold(key)
        out.append(cache[key])
    return np.asarray(out, dtype=object)


def random_split(
    n_rows: int, test_size: float = DEFAULT_TEST_SIZE, seed: int = 0
) -> SplitResult:
    """Partition rows uniformly at random.

    Args:
        n_rows: Number of rows to split.
        test_size: Target test fraction.
        seed: Seed for the permutation.

    Returns:
        The split.

    Raises:
        ValueError: If ``n_rows`` is too small or ``test_size`` is not in
            the open interval (0, 1).
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")
    if n_rows < 2:
        raise ValueError(f"need at least 2 rows to split, got {n_rows}")
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_rows)
    n_test = max(1, min(n_rows - 1, int(round(test_size * n_rows))))
    return SplitResult(
        strategy="random",
        train_index=np.sort(order[n_test:]),
        test_index=np.sort(order[:n_test]),
        details={"seed": seed},
    )


def scaffold_split(
    smiles: Sequence[str],
    test_size: float = DEFAULT_TEST_SIZE,
    shuffle_groups: bool = False,
    seed: int = 0,
) -> SplitResult:
    """Partition rows by Bemis-Murcko scaffold, keeping each group intact.

    Groups are filled into the training set largest-first until the training
    quota is met; everything left becomes the test set.  This concentrates
    the populous, well-studied chemotypes in training and leaves the long
    tail of singleton scaffolds for testing, which is both the standard
    construction and the hardest honest one.

    Args:
        smiles: SMILES strings, in dataset row order.
        test_size: Target test fraction.
        shuffle_groups: Assign groups in random order instead of
            largest-first.  Useful for checking that a result does not hinge
            on the size ordering.
        seed: Seed used when ``shuffle_groups`` is true, and to break ties
            between equally sized groups.

    Returns:
        The split, with the scaffold group count in ``details``.

    Raises:
        ValueError: If the inputs cannot produce a two-sided split.
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")
    n_rows = len(smiles)
    if n_rows < 2:
        raise ValueError(f"need at least 2 rows to split, got {n_rows}")

    groups = scaffold_groups(smiles)
    members: dict[str, list[int]] = {}
    for index, key in enumerate(groups):
        members.setdefault(str(key), []).append(index)

    rng = np.random.default_rng(seed)
    keys = list(members)
    if shuffle_groups:
        rng.shuffle(keys)
    else:
        # Largest first; hash the key for a deterministic, data-independent
        # tie-break so the split does not depend on dict insertion order.
        keys.sort(key=lambda k: (-len(members[k]), k))

    n_train_target = n_rows - max(1, min(n_rows - 1, int(round(test_size * n_rows))))
    train: list[int] = []
    test: list[int] = []
    for key in keys:
        if len(train) < n_train_target:
            train.extend(members[key])
        else:
            test.extend(members[key])
    if not test and len(members) > 1:
        # The training quota swallowed every group: hold out the smallest one
        # so the caller still gets a two-sided split.
        smallest = min(keys, key=lambda k: (len(members[k]), k))
        test = members[smallest]
        held_out = set(test)
        train = [index for index in train if index not in held_out]

    return SplitResult(
        strategy="scaffold",
        train_index=np.sort(np.asarray(train, dtype=int)),
        test_index=np.sort(np.asarray(test, dtype=int)),
        details={
            "n_scaffolds": len(members),
            "n_singleton_scaffolds": sum(1 for v in members.values() if len(v) == 1),
            "shuffle_groups": shuffle_groups,
            "seed": seed,
        },
    )


def time_split(
    years: Sequence[float], test_size: float = DEFAULT_TEST_SIZE
) -> SplitResult:
    """Partition rows by first-publication year.

    The cutoff year is chosen as the one whose realised test fraction is
    closest to ``test_size``; every compound first published in the cutoff
    year or later goes to the test set.  Because years are coarse, the
    realised fraction is reported in ``details`` and will rarely be exactly
    ``test_size``.  Compounds with no usable year are excluded from the
    split and counted in ``n_dropped``.

    Args:
        years: First-publication year per row; zero, negative or missing
            values mean unknown.
        test_size: Target test fraction.

    Returns:
        The split, with the cutoff year and realised fraction in ``details``.

    Raises:
        ValueError: If fewer than two distinct usable years are present.
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")
    year_array = pd.to_numeric(pd.Series(list(years)), errors="coerce").to_numpy(
        dtype=float
    )
    usable = np.isfinite(year_array) & (year_array > 0)
    n_dropped = int((~usable).sum())
    usable_years = year_array[usable]
    distinct = np.unique(usable_years)
    if distinct.size < 2:
        raise ValueError(
            "a time split needs at least two distinct publication years, "
            f"found {distinct.size}"
        )

    n_usable = usable_years.size
    best_cutoff = float(distinct[-1])
    best_gap = float("inf")
    for candidate in distinct[1:]:
        fraction = float((usable_years >= candidate).sum()) / n_usable
        gap = abs(fraction - test_size)
        if gap < best_gap:
            best_gap, best_cutoff = gap, float(candidate)

    positions = np.flatnonzero(usable)
    is_test = year_array[positions] >= best_cutoff
    train_index = positions[~is_test]
    test_index = positions[is_test]
    return SplitResult(
        strategy="time",
        train_index=train_index,
        test_index=test_index,
        n_dropped=n_dropped,
        details={
            "cutoff_year": int(best_cutoff),
            "train_year_min": (
                int(year_array[train_index].min()) if train_index.size else 0
            ),
            "train_year_max": (
                int(year_array[train_index].max()) if train_index.size else 0
            ),
            "test_year_max": int(year_array[test_index].max()) if test_index.size else 0,
            "n_unknown_year": n_dropped,
        },
    )


def make_split(
    frame: pd.DataFrame,
    strategy: str,
    test_size: float = DEFAULT_TEST_SIZE,
    seed: int = 0,
    smiles_column: str = "canonical_smiles",
    year_column: str = "first_year",
) -> SplitResult:
    """Build a split by strategy name and validate it.

    Args:
        frame: A cleaned activity table.
        strategy: One of :data:`SPLIT_STRATEGIES`.
        test_size: Target test fraction.
        seed: Seed for the strategies that use one.
        smiles_column: Column holding SMILES, for the scaffold split.
        year_column: Column holding first-publication years, for the time
            split.

    Returns:
        A validated :class:`SplitResult`.

    Raises:
        ValueError: On an unknown strategy or an invalid partition.
        KeyError: If a column the strategy needs is absent.
    """
    if strategy == "random":
        result = random_split(len(frame), test_size=test_size, seed=seed)
    elif strategy == "scaffold":
        if smiles_column not in frame.columns:
            raise KeyError(f"frame has no column {smiles_column!r}")
        result = scaffold_split(
            frame[smiles_column].tolist(), test_size=test_size, seed=seed
        )
    elif strategy == "time":
        if year_column not in frame.columns:
            raise KeyError(f"frame has no column {year_column!r}")
        result = time_split(frame[year_column].tolist(), test_size=test_size)
    else:
        raise ValueError(
            f"unknown split strategy {strategy!r}; "
            f"known: {', '.join(SPLIT_STRATEGIES)}"
        )
    result.validate(len(frame))
    return result


def stratified_scaffold_split(
    smiles: Sequence[str],
    labels: np.ndarray,
    test_size: float = DEFAULT_TEST_SIZE,
) -> SplitResult:
    """Scaffold-split each class separately, then pool the halves.

    A plain scaffold split over a screening library fills training
    largest-group-first, and in a decoy library the actives are
    scaffold-clustered while the decoys are mostly singletons. The test
    active rate therefore ends up wherever the group sizes put it, which
    on this study's benchmarks was 5.6% against a 2% library.

    That is not a cosmetic difference. Enrichment factor is bounded by
    ``1 / active_rate`` *of the set it is computed on*, so a test set
    three times as active as the library has a ceiling three times lower,
    and the reported enrichment is no longer comparable to the library
    the benchmark claims to represent.

    Splitting within each class holds the test active rate at the
    library's, while leaving the scaffold separation - the part that makes
    the task hard and honest - intact inside each class.

    Args:
        smiles: SMILES for the whole library.
        labels: Class label per compound; each distinct value is split on
            its own scaffolds.
        test_size: Target test fraction, applied within each class.

    Returns:
        The pooled split, with per-class scaffold counts in ``details``.

    Raises:
        ValueError: If ``labels`` does not align with ``smiles``, or there
            is nothing to split.
    """
    labels = np.asarray(labels)
    if len(labels) != len(smiles):
        raise ValueError(
            f"labels and smiles must align: {len(labels)} vs {len(smiles)}"
        )
    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    details: dict[str, Any] = {"stratified": True}
    for value in np.unique(labels):
        rows = np.flatnonzero(labels == value)
        part = scaffold_split([smiles[i] for i in rows], test_size=test_size)
        train_parts.append(rows[part.train_index])
        test_parts.append(rows[part.test_index])
        details[f"n_scaffolds_class_{value}"] = part.details.get("n_scaffolds")
    if not train_parts:
        raise ValueError("no compounds to split")
    return SplitResult(
        strategy="scaffold",
        train_index=np.sort(np.concatenate(train_parts)),
        test_index=np.sort(np.concatenate(test_parts)),
        details=details,
    )


def scaffold_overlap(
    smiles: Sequence[str], split: SplitResult
) -> dict[str, float]:
    """Measure how much scaffold structure a split leaks across its boundary.

    This is the number that explains the headline result: under a random
    split most test scaffolds are also present in training, so the model is
    largely being asked to recognise chemistry it has already seen.

    Args:
        smiles: SMILES strings, in dataset row order.
        split: The partition to assess.

    Returns:
        A dictionary with the scaffold counts on each side, the number and
        fraction of test *scaffolds* also seen in training, and the fraction
        of test *compounds* whose scaffold was seen in training.
    """
    groups = scaffold_groups(smiles)
    train_scaffolds = set(groups[split.train_index].tolist())
    test_groups = groups[split.test_index]
    test_scaffolds = set(test_groups.tolist())
    shared = test_scaffolds & train_scaffolds
    n_test_compounds_seen = int(
        sum(1 for key in test_groups if key in train_scaffolds)
    )
    return {
        "n_train_scaffolds": float(len(train_scaffolds)),
        "n_test_scaffolds": float(len(test_scaffolds)),
        "n_shared_scaffolds": float(len(shared)),
        "frac_test_scaffolds_seen": (
            len(shared) / len(test_scaffolds) if test_scaffolds else 0.0
        ),
        "frac_test_compounds_seen": (
            n_test_compounds_seen / test_groups.size if test_groups.size else 0.0
        ),
    }
