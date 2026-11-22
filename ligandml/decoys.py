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

Two safeguards exist because the first version of this module produced a
benchmark that looked well built and was not. Matching to the *nearest
available* candidate always succeeds: when the pool holds nothing like an
active, it returns the least bad compound and the summary reports a full
set. :data:`MAX_MATCH_DISTANCE` refuses those matches, so an active the
pool cannot disguise is dropped and counted rather than handed a free
hit. And :func:`signed_bias_by_property` breaks the bias down per
property, because the single worst-case number says a set is biased
without saying that the cause is, for instance, three more hydrogen-bond
donors on every active than anything in the pool - a diagnosis that tells
you no amount of re-matching will help and a different pool is needed.
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

#: How far, in standardised property space, a candidate may sit from the
#: active it is matched to. Nearest-available matching will always return
#: *something*; this is what stops that something being a compound from a
#: different region entirely. Two compounds drawn from the same
#: distribution sit about ``sqrt(2 * n_properties)`` apart, so with six
#: properties a cap of 1.0 is a demand for a genuinely close match rather
#: than the best of a bad pool.
MAX_MATCH_DISTANCE: Final[float] = 1.0

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
        n_actives_requested: Actives offered to the matcher, before any
            were dropped for want of a close enough candidate.
        pool_size: Candidates the matcher had to choose from.
        bias_by_property: Signed standardised difference per property,
            which is what names the cause when the matching fails.
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
    n_actives_requested: int = 0
    pool_size: int = 0
    bias_by_property: tuple[tuple[str, float], ...] = ()

    @property
    def n_actives_dropped(self) -> int:
        """Actives discarded because the pool held no acceptable match."""
        return max(self.n_actives_requested - self.n_actives, 0)

    @property
    def pool_fraction_used(self) -> float:
        """Share of the candidate pool consumed by this benchmark.

        A figure near 1 means the matcher had no freedom left: the decoys
        are not a matched sample of the pool, they are the pool.
        """
        return self.n_decoys / self.pool_size if self.pool_size else 0.0

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
            "n_actives_requested": self.n_actives_requested,
            "n_actives_dropped": self.n_actives_dropped,
            "pool_size": self.pool_size,
            "pool_fraction_used": round(self.pool_fraction_used, 4),
            "worst_property": max(
                self.bias_by_property, key=lambda kv: abs(kv[1]), default=("", 0.0)
            )[0],
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


def signed_bias_by_property(
    active_properties: np.ndarray,
    decoy_properties: np.ndarray,
    names: Sequence[str] = MATCH_PROPERTIES,
) -> dict[str, float]:
    """Standardised difference per property, keeping its sign.

    :func:`standardised_bias` collapses the matching to its worst number,
    which says *that* a benchmark is biased but not *how*. The sign and
    the breakdown are what identify the cause: a set whose actives carry
    three more hydrogen-bond donors than anything in the pool has not been
    matched badly, it has been matched against a pool that does not
    contain the right compounds, and no amount of re-matching will fix it.

    Args:
        active_properties: ``(n_actives, n_properties)`` array.
        decoy_properties: ``(n_decoys, n_properties)`` array.
        names: Property names, in column order.

    Returns:
        Property name to signed standardised difference, positive when the
        actives sit higher. Empty if either group is empty.

    Raises:
        ValueError: If ``names`` does not match the column count.
    """
    if active_properties.size == 0 or decoy_properties.size == 0:
        return {}
    if active_properties.shape[1] != len(names):
        raise ValueError(
            f"expected {active_properties.shape[1]} names, got {len(names)}"
        )
    pooled = np.concatenate([active_properties, decoy_properties], axis=0)
    spread = pooled.std(axis=0, ddof=0)
    spread[spread == 0] = 1.0
    difference = active_properties.mean(axis=0) - decoy_properties.mean(axis=0)
    return {
        name: float(value) for name, value in zip(names, difference / spread)
    }


def _normalise(properties: np.ndarray) -> np.ndarray:
    """Scale each property to unit variance so distances weigh them evenly."""
    spread = properties.std(axis=0, ddof=0)
    spread[spread == 0] = 1.0
    return (properties - properties.mean(axis=0)) / spread


@dataclass(frozen=True)
class DecoyMatch:
    """The outcome of matching, with each decoy attributed to its active.

    Attributes:
        indices: Pool indices of the chosen decoys, in selection order.
        owner: The active each chosen decoy was matched to, same length
            as ``indices``.
        counts: Decoys obtained per active, length ``n_actives``. An
            active whose count falls short of the requested ratio had no
            acceptable candidate left within the distance cap.
    """

    indices: np.ndarray
    owner: np.ndarray
    counts: np.ndarray


def match_decoys(
    active_properties: np.ndarray,
    pool_properties: np.ndarray,
    pool_similarity_to_actives: np.ndarray,
    ratio: int = DEFAULT_DECOY_RATIO,
    max_similarity: float = MAX_DECOY_SIMILARITY,
    max_distance: float | None = None,
) -> DecoyMatch:
    """Match decoys to actives and record which active claimed each.

    Each active claims its nearest unused candidates in standardised
    property space, taken in turn so no single active monopolises the
    close matches. Candidates too similar to *any* active are removed
    before matching rather than penalised during it, because a possible
    analogue is disqualified no matter how well its properties fit.

    With ``max_distance`` set, an active stops claiming once its nearest
    remaining candidate is further away than the cap. Nearest-available
    matching otherwise always succeeds - it returns the least bad
    candidate even when the pool holds nothing suitable - and the
    resulting benchmark looks fully populated while being matched on
    nothing. Refusing those matches is what turns a hidden bias into a
    visible shortfall.

    Args:
        active_properties: ``(n_actives, n_properties)`` array.
        pool_properties: ``(n_pool, n_properties)`` array in the same
            column order.
        pool_similarity_to_actives: Highest Tanimoto similarity of each
            pool compound against any active, length ``n_pool``.
        ratio: Decoys to select per active.
        max_similarity: Candidates at or above this are refused.
        max_distance: Furthest acceptable match in standardised property
            space, or ``None`` to accept the nearest whatever its
            distance.

    Returns:
        The match, including the per-active counts.

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

    n_actives = len(active_properties)
    empty = DecoyMatch(
        indices=np.empty(0, dtype=int),
        owner=np.empty(0, dtype=int),
        counts=np.zeros(n_actives, dtype=int),
    )
    eligible = np.flatnonzero(
        np.asarray(pool_similarity_to_actives) < max_similarity
    )
    if eligible.size == 0:
        return empty

    combined = np.concatenate([active_properties, pool_properties[eligible]])
    scaled = _normalise(combined)
    scaled_actives = scaled[:n_actives]
    scaled_pool = scaled[n_actives:]

    # Rank every candidate against every active once, then deal them out.
    distances = np.linalg.norm(
        scaled_pool[None, :, :] - scaled_actives[:, None, :], axis=2
    )
    order = np.argsort(distances, axis=1, kind="stable")
    cursors = np.zeros(n_actives, dtype=int)
    counts = np.zeros(n_actives, dtype=int)
    taken: set[int] = set()
    chosen: list[int] = []
    owners: list[int] = []
    wanted = ratio * n_actives
    while len(chosen) < wanted:
        progressed = False
        for active in range(n_actives):
            if len(chosen) >= wanted or counts[active] >= ratio:
                continue
            row = order[active]
            while cursors[active] < len(row) and int(row[cursors[active]]) in taken:
                cursors[active] += 1
            if cursors[active] >= len(row):
                continue
            pick = int(row[cursors[active]])
            if max_distance is not None and distances[active, pick] > max_distance:
                # The row is sorted, so nothing further along qualifies.
                cursors[active] = len(row)
                continue
            cursors[active] += 1
            taken.add(pick)
            chosen.append(int(eligible[pick]))
            owners.append(active)
            counts[active] += 1
            progressed = True
        if not progressed:
            break
    return DecoyMatch(
        indices=np.asarray(chosen, dtype=int),
        owner=np.asarray(owners, dtype=int),
        counts=counts,
    )


def select_decoys(
    active_properties: np.ndarray,
    pool_properties: np.ndarray,
    pool_similarity_to_actives: np.ndarray,
    ratio: int = DEFAULT_DECOY_RATIO,
    max_similarity: float = MAX_DECOY_SIMILARITY,
    max_distance: float | None = None,
) -> np.ndarray:
    """Choose decoy indices: property-matched, topologically dissimilar.

    Thin wrapper over :func:`match_decoys` for callers that need only the
    chosen indices and not the attribution.

    Args:
        active_properties: ``(n_actives, n_properties)`` array.
        pool_properties: ``(n_pool, n_properties)`` array in the same
            column order.
        pool_similarity_to_actives: Highest Tanimoto similarity of each
            pool compound against any active, length ``n_pool``.
        ratio: Decoys to select per active.
        max_similarity: Candidates at or above this are refused.
        max_distance: Furthest acceptable match, or ``None`` for no cap.

    Returns:
        Indices into the pool, in selection order.

    Raises:
        ValueError: If the shapes disagree or ``ratio`` is not positive.
    """
    return match_decoys(
        active_properties,
        pool_properties,
        pool_similarity_to_actives,
        ratio=ratio,
        max_similarity=max_similarity,
        max_distance=max_distance,
    ).indices


def build_decoy_set(
    target: str,
    active_smiles: Sequence[str],
    active_properties: np.ndarray,
    pool_smiles: Sequence[str],
    pool_properties: np.ndarray,
    pool_similarity_to_actives: np.ndarray,
    ratio: int = DEFAULT_DECOY_RATIO,
    max_similarity: float = MAX_DECOY_SIMILARITY,
    max_distance: float | None = None,
    min_decoys_per_active: int = 1,
    property_names: Sequence[str] | None = None,
) -> DecoySet:
    """Assemble a screening benchmark and record how well it was matched.

    An active that the matcher could not serve - no candidate within
    ``max_distance``, or fewer than ``min_decoys_per_active`` of them - is
    dropped along with the decoys it did claim. Keeping it would mean
    putting a compound into the benchmark that the pool cannot disguise,
    which is the same as handing the model a free hit.

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
        max_distance: Furthest acceptable match in standardised property
            space, or ``None`` to accept the nearest candidate whatever
            its distance.
        min_decoys_per_active: Below this an active is dropped.
        property_names: Names for the per-property bias breakdown.
            Defaults to :data:`MATCH_PROPERTIES` when the column count
            agrees, and to positional labels otherwise.

    Returns:
        The assembled benchmark.

    Raises:
        ValueError: If no actives are supplied.
    """
    if len(active_smiles) == 0:
        raise ValueError("need at least one active to build a decoy set")
    width = active_properties.shape[1] if active_properties.ndim == 2 else 0
    if property_names is None:
        property_names = (
            MATCH_PROPERTIES
            if width == len(MATCH_PROPERTIES)
            else tuple(f"property_{i}" for i in range(width))
        )
    match = match_decoys(
        active_properties,
        pool_properties,
        pool_similarity_to_actives,
        ratio=ratio,
        max_similarity=max_similarity,
        max_distance=max_distance,
    )
    served = match.counts >= max(min_decoys_per_active, 1)
    keep_decoy = served[match.owner] if match.indices.size else np.zeros(0, dtype=bool)
    picked = match.indices[keep_decoy]
    kept_actives = np.flatnonzero(served)

    active_smiles = [active_smiles[i] for i in kept_actives]
    kept_properties = active_properties[kept_actives]
    decoy_smiles = [pool_smiles[i] for i in picked]
    bias = standardised_bias(kept_properties, pool_properties[picked])
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
        n_actives_requested=len(match.counts),
        pool_size=len(pool_smiles),
        bias_by_property=tuple(
            signed_bias_by_property(
                kept_properties, pool_properties[picked], property_names
            ).items()
        ),
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
