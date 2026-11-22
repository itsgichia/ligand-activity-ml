"""Property-matched decoys, so the screening metrics mean something.

Every dataset in this study is between 50% and 83% active, because ChEMBL
records the compounds somebody thought worth making and measuring. That
makes the screening section of the study close to vacuous: enrichment
factor cannot exceed ``1 / active_rate``, so a ceiling of 1.21 on DPP-4
leaves no room for a model to demonstrate anything, and a reported "EF =
1.21" is indistinguishable from a model pinned at the maximum by
construction.

A real screen is the other way round. A vendor library is well under 1%
active, the ceiling is in the hundreds, and the question - *how far up the
ranking do the actives come?* - has an informative answer.

This module builds that situation out of the data already fetched. For a
chosen target it keeps a sample of measured actives and surrounds them
with **decoys**: compounds measured against one of the *other* targets in
the study, absent from this target's table, matched to the actives on bulk
physicochemical properties and required to be topologically dissimilar
from every active.

That construction is the DUD-E recipe, and it carries DUD-E's assumption:
a decoy is *presumed* inactive, not *measured* inactive. A compound tested
against D2 and never against hERG is probably not a hERG blocker, but
nobody has checked. Property matching is what stops the model solving the
benchmark by reading molecular weight instead of pharmacology - without
it, a classifier separates actives from decoys on size alone and the
enrichment is an artefact. Topological dissimilarity is what stops a true
analogue of an active being labelled a decoy.

The resulting benchmark is reported *alongside* the measured-inactive one,
never instead of it. The two answer different questions: the measured
split asks whether the model can rank compounds a chemist already thought
worth making, and the decoy screen asks whether it can find the needle in
a library it has no reason to care about.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

#: Bulk properties decoys are matched on. Deliberately the coarse ones a
#: model could otherwise exploit: a classifier that separates actives from
#: decoys on molecular weight has learned the benchmark, not the target.
MATCH_PROPERTIES: Final[tuple[str, ...]] = (
    "MolWt",
    "MolLogP",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "TPSA",
)

#: A candidate this similar to any active is treated as a possible analogue
#: and refused, however well its properties match.
MAX_DECOY_SIMILARITY: Final[float] = 0.35

#: Decoys per active. 49 gives a 2% active rate and a ceiling of 50, which
#: is a plausible hit rate for a focused library; a diversity screen would
#: be an order of magnitude lower still.
DEFAULT_DECOY_RATIO: Final[int] = 49


@dataclass(frozen=True)
class DecoySet:
    """A screening benchmark of sampled actives plus matched decoys.

    Attributes:
        target: Target the actives were measured against.
        smiles: SMILES for every compound, actives first.
        labels: 1 for an active, 0 for a decoy.
        n_actives: Number of actives kept.
        n_decoys: Number of decoys attached.
        requested_ratio: Decoys per active that were asked for.
        property_bias: Largest standardised difference between the active
            and decoy means over :data:`MATCH_PROPERTIES`. Near zero means
            the two groups are indistinguishable on bulk properties.
        max_similarity: Highest Tanimoto similarity any decoy reaches
            against any active.
        pool_exhausted: Whether the candidate pool ran out before the
            requested ratio was met.
    """

    target: str
    smiles: tuple[str, ...]
    labels: np.ndarray
    n_actives: int
    n_decoys: int
    requested_ratio: int
    property_bias: float
    max_similarity: float
    pool_exhausted: bool

    @property
    def active_rate(self) -> float:
        """Fraction of the benchmark that is active."""
        total = self.n_actives + self.n_decoys
        return self.n_actives / total if total else 0.0

    @property
    def enrichment_ceiling_1pct(self) -> float:
        """Largest enrichment factor at 1% this benchmark can produce."""
        rate = self.active_rate
        return min(1.0 / rate, 100.0) if rate > 0 else float("nan")

    def as_row(self) -> dict[str, object]:
        """Return the construction summary as a flat row."""
        return {
            "target": self.target,
            "n_actives": self.n_actives,
            "n_decoys": self.n_decoys,
            "requested_ratio": self.requested_ratio,
            "achieved_ratio": round(self.n_decoys / max(self.n_actives, 1), 2),
            "active_rate": round(self.active_rate, 5),
            "ef_ceiling_1pct": round(self.enrichment_ceiling_1pct, 2),
            "property_bias": round(self.property_bias, 4),
            "max_decoy_similarity": round(self.max_similarity, 4),
            "pool_exhausted": self.pool_exhausted,
        }


def standardised_bias(
    active_properties: np.ndarray, decoy_properties: np.ndarray
) -> float:
    """Largest standardised mean difference across the matched properties.

    This is the number that says whether the matching worked. A benchmark
    whose decoys are systematically lighter or greasier than its actives
    can be solved without any pharmacology, and the enrichment it reports
    is an artefact of the construction.

    Args:
        active_properties: ``(n_actives, n_properties)`` array.
        decoy_properties: ``(n_decoys, n_properties)`` array.

    Returns:
        The largest absolute standardised difference, or 0.0 if either
        group is empty. Below about 0.1 the groups are indistinguishable
        on these properties.
    """
    if active_properties.size == 0 or decoy_properties.size == 0:
        return 0.0
    pooled = np.concatenate([active_properties, decoy_properties], axis=0)
    spread = pooled.std(axis=0, ddof=0)
    spread[spread == 0] = 1.0
    difference = np.abs(
        active_properties.mean(axis=0) - decoy_properties.mean(axis=0)
    )
    return float((difference / spread).max())


def _normalise(properties: np.ndarray) -> np.ndarray:
    """Scale each property to unit variance so distances weigh them evenly."""
    spread = properties.std(axis=0, ddof=0)
    spread[spread == 0] = 1.0
    return (properties - properties.mean(axis=0)) / spread


def select_decoys(
    active_properties: np.ndarray,
    pool_properties: np.ndarray,
    pool_similarity_to_actives: np.ndarray,
    ratio: int = DEFAULT_DECOY_RATIO,
    max_similarity: float = MAX_DECOY_SIMILARITY,
) -> np.ndarray:
    """Choose decoy indices: property-matched, topologically dissimilar.

    Each active claims its nearest unused candidates in standardised
    property space, taken in turn so no single active monopolises the
    close matches. Candidates too similar to *any* active are removed
    before matching rather than penalised during it, because a possible
    analogue is disqualified no matter how well its properties fit.

    Args:
        active_properties: ``(n_actives, n_properties)`` array.
        pool_properties: ``(n_pool, n_properties)`` array in the same
            column order.
        pool_similarity_to_actives: Highest Tanimoto similarity of each
            pool compound against any active, length ``n_pool``.
        ratio: Decoys to select per active.
        max_similarity: Candidates at or above this are refused.

    Returns:
        Indices into the pool, in selection order.

    Raises:
        ValueError: If the shapes disagree or ``ratio`` is not positive.
    """
    if ratio < 1:
        raise ValueError(f"ratio must be at least 1, got {ratio}")
    if active_properties.ndim != 2 or pool_properties.ndim != 2:
        raise ValueError("properties must be two-dimensional")
    if active_properties.shape[1] != pool_properties.shape[1]:
        raise ValueError(
            "actives and pool must share property columns: "
            f"{active_properties.shape[1]} vs {pool_properties.shape[1]}"
        )
    if len(pool_similarity_to_actives) != len(pool_properties):
        raise ValueError("one similarity per pool compound is required")

    eligible = np.flatnonzero(
        np.asarray(pool_similarity_to_actives) < max_similarity
    )
    if eligible.size == 0:
        return np.empty(0, dtype=int)

    combined = np.concatenate([active_properties, pool_properties[eligible]])
    scaled = _normalise(combined)
    scaled_actives = scaled[: len(active_properties)]
    scaled_pool = scaled[len(active_properties) :]

    # Rank every candidate against every active once, then deal them out.
    distances = np.linalg.norm(
        scaled_pool[None, :, :] - scaled_actives[:, None, :], axis=2
    )
    order = np.argsort(distances, axis=1, kind="stable")
    cursors = np.zeros(len(active_properties), dtype=int)
    taken: set[int] = set()
    chosen: list[int] = []
    wanted = ratio * len(active_properties)
    while len(chosen) < wanted:
        progressed = False
        for active in range(len(active_properties)):
            if len(chosen) >= wanted:
                break
            row = order[active]
            while cursors[active] < len(row) and int(row[cursors[active]]) in taken:
                cursors[active] += 1
            if cursors[active] >= len(row):
                continue
            pick = int(row[cursors[active]])
            cursors[active] += 1
            taken.add(pick)
            chosen.append(int(eligible[pick]))
            progressed = True
        if not progressed:
            break
    return np.asarray(chosen, dtype=int)


def build_decoy_set(
    target: str,
    active_smiles: Sequence[str],
    active_properties: np.ndarray,
    pool_smiles: Sequence[str],
    pool_properties: np.ndarray,
    pool_similarity_to_actives: np.ndarray,
    ratio: int = DEFAULT_DECOY_RATIO,
    max_similarity: float = MAX_DECOY_SIMILARITY,
) -> DecoySet:
    """Assemble a screening benchmark and record how well it was matched.

    Args:
        target: Target name, carried through to the summary.
        active_smiles: SMILES of the sampled actives.
        active_properties: Their properties, in :data:`MATCH_PROPERTIES`
            order.
        pool_smiles: SMILES of the candidate decoys.
        pool_properties: Their properties, same column order.
        pool_similarity_to_actives: Highest Tanimoto similarity of each
            candidate against any active.
        ratio: Decoys per active to aim for.
        max_similarity: Similarity at which a candidate is refused.

    Returns:
        The assembled benchmark.

    Raises:
        ValueError: If no actives are supplied.
    """
    if len(active_smiles) == 0:
        raise ValueError("need at least one active to build a decoy set")
    picked = select_decoys(
        active_properties,
        pool_properties,
        pool_similarity_to_actives,
        ratio=ratio,
        max_similarity=max_similarity,
    )
    decoy_smiles = [pool_smiles[i] for i in picked]
    bias = standardised_bias(active_properties, pool_properties[picked])
    worst = (
        float(np.max(np.asarray(pool_similarity_to_actives)[picked]))
        if picked.size
        else 0.0
    )
    labels = np.concatenate(
        [np.ones(len(active_smiles), dtype=int), np.zeros(len(decoy_smiles), dtype=int)]
    )
    return DecoySet(
        target=target,
        smiles=tuple(list(active_smiles) + decoy_smiles),
        labels=labels,
        n_actives=len(active_smiles),
        n_decoys=len(decoy_smiles),
        requested_ratio=ratio,
        property_bias=bias,
        max_similarity=worst,
        pool_exhausted=len(picked) < ratio * len(active_smiles),
    )


def cross_target_pool(
    frames: dict[str, pd.DataFrame],
    target: str,
    smiles_column: str = "canonical_smiles",
) -> pd.DataFrame:
    """Candidate decoys: compounds measured against the *other* targets.

    A compound present in this target's own table has a measurement
    against it and cannot be presumed inactive, so it is excluded however
    weak that measurement is.

    Args:
        frames: Cleaned activity tables by target name.
        target: The target whose benchmark is being built.
        smiles_column: Column holding SMILES.

    Returns:
        One row per distinct candidate, with a ``source_target`` column.

    Raises:
        KeyError: If ``target`` is absent from ``frames``.
    """
    if target not in frames:
        raise KeyError(f"no frame for target {target!r}")
    own = set(frames[target][smiles_column])
    parts: list[pd.DataFrame] = []
    for name, frame in frames.items():
        if name == target:
            continue
        slice_ = frame.loc[~frame[smiles_column].isin(own), [smiles_column]].copy()
        slice_["source_target"] = name
        parts.append(slice_)
    if not parts:
        return pd.DataFrame(columns=[smiles_column, "source_target"])
    pool = pd.concat(parts, ignore_index=True)
    return pool.drop_duplicates(subset=smiles_column).reset_index(drop=True)
