"""Uncertainty on the split-gap estimate.

The headline claim of this study is an average: *across every target and
model, a scaffold split costs about 0.08 of ROC AUC*. An average needs a
spread, and the spread that matters depends on what the reader wants to
generalise over.

Two things are worth separating, and they answer different questions:

**Across cells.** The gap is measured once per target and model, so the 20
cells are 20 paired observations of the same quantity. Summarising them
answers *does this hold across chemistry and across model families, or is
it one target dragging the mean?* This is the question the headline
sentence actually makes a claim about, and it costs nothing beyond
arithmetic on results already in hand. Both a parametric interval and the
Wilcoxon signed-rank statistic are reported, because with 20 paired values
and one known outlier the two can disagree, and a reader is entitled to see
that rather than be handed whichever one is prettier.

**Across split draws.** Any one split is a sample. Re-drawing it under a
different seed moves the number, and that movement is the error bar a
benchmark normally quotes (MoleculeNet, Therapeutics Data Commons and most
cheminformatics papers report mean and standard deviation over a handful of
seeds). This costs a refit per seed, so it runs over the default model and
representation rather than the whole grid.

A bootstrap over test compounds, which the screening tables already use,
answers a third and narrower question: how precisely this one test set
pins down this one number. It is the wrong tool for a claim about an
average over targets, which is why it is not the primary estimate here.

The time split has no seed - the cutoff year determines it - so its
counterpart is a sensitivity analysis over the cutoff, exposed through
:func:`time_cutoff_grid`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Final

import numpy as np

#: Two-sided t critical values at 95%, indexed by degrees of freedom. The
#: study has no SciPy dependency and the table is only needed for the small
#: samples this module deals with, so the handful of values are inlined and
#: anything larger falls back to the normal approximation.
_T95: Final[dict[int, float]] = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160,
    14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
    20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}
_Z95: Final[float] = 1.959964


def t_critical_95(df: int) -> float:
    """Two-sided 95% t critical value for ``df`` degrees of freedom.

    Args:
        df: Degrees of freedom; must be at least 1.

    Returns:
        The critical value, falling back to the normal approximation above
        the tabulated range.

    Raises:
        ValueError: If ``df`` is less than 1.
    """
    if df < 1:
        raise ValueError(f"df must be at least 1, got {df}")
    return _T95.get(df, _Z95)


@dataclass(frozen=True)
class GapSummary:
    """The split gap summarised over a set of paired observations.

    Attributes:
        comparison: Which pair of strategies the gap is between.
        metric: The metric the gap is measured in.
        n: Number of paired observations.
        mean: Mean gap. Positive means the honest split scored lower.
        sd: Standard deviation across observations.
        se: Standard error of the mean.
        ci_low: Lower bound of the 95% interval for the mean.
        ci_high: Upper bound of the 95% interval for the mean.
        median: Median gap, which the Wilcoxon statistic speaks to.
        n_positive: Observations where the honest split scored lower.
        n_negative: Observations that contradict the claim.
        wilcoxon_w: Signed-rank statistic, the smaller of the two rank sums.
        wilcoxon_p: Two-sided p-value for the signed-rank test.
        exceptions: Labels of the contradicting observations.
    """

    comparison: str
    metric: str
    n: int
    mean: float
    sd: float
    se: float
    ci_low: float
    ci_high: float
    median: float
    n_positive: int
    n_negative: int
    wilcoxon_w: float
    wilcoxon_p: float
    exceptions: tuple[str, ...] = field(default_factory=tuple)

    def as_row(self) -> dict[str, object]:
        """Return the summary as a flat row for a results table."""
        return {
            "comparison": self.comparison,
            "metric": self.metric,
            "n_cells": self.n,
            "mean_gap": round(self.mean, 4),
            "sd": round(self.sd, 4),
            "se": round(self.se, 4),
            "ci95_low": round(self.ci_low, 4),
            "ci95_high": round(self.ci_high, 4),
            "median_gap": round(self.median, 4),
            "n_cells_positive": self.n_positive,
            "n_cells_negative": self.n_negative,
            "wilcoxon_w": round(self.wilcoxon_w, 1),
            "wilcoxon_p": round(self.wilcoxon_p, 4),
            "exceptions": "; ".join(self.exceptions),
        }


def _rank_with_ties(values: np.ndarray) -> np.ndarray:
    """Rank ascending, averaging the ranks within each group of ties."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def wilcoxon_signed_rank(values: Sequence[float]) -> tuple[float, float]:
    """Two-sided Wilcoxon signed-rank test against a median of zero.

    Zero differences are dropped, which is the standard Wilcoxon handling,
    and the p-value uses the tie-corrected normal approximation with a
    continuity correction. That approximation is poor below about six
    non-zero differences, so this returns a p-value of 1.0 there rather
    than a confident-looking number it cannot support.

    Args:
        values: The paired differences.

    Returns:
        The signed-rank statistic (the smaller of the two rank sums) and
        the two-sided p-value.
    """
    array = np.asarray([v for v in values if v == v], dtype=float)
    non_zero = array[array != 0.0]
    n = len(non_zero)
    if n == 0:
        return 0.0, 1.0
    ranks = _rank_with_ties(np.abs(non_zero))
    w_plus = float(ranks[non_zero > 0].sum())
    w_minus = float(ranks[non_zero < 0].sum())
    statistic = min(w_plus, w_minus)
    if n < 6:
        return statistic, 1.0
    mean = n * (n + 1) / 4.0
    _, counts = np.unique(np.abs(non_zero), return_counts=True)
    tie_term = float(((counts**3) - counts).sum())
    variance = (n * (n + 1) * (2 * n + 1) - 0.5 * tie_term) / 24.0
    if variance <= 0:
        return statistic, 1.0
    z = (abs(statistic - mean) - 0.5) / math.sqrt(variance)
    p = math.erfc(z / math.sqrt(2.0))
    return statistic, min(1.0, max(0.0, p))


def summarise_gaps(
    gaps: Sequence[float],
    comparison: str,
    metric: str,
    labels: Sequence[str] | None = None,
) -> GapSummary:
    """Summarise paired split gaps across cells.

    Args:
        gaps: One gap per cell, positive when the honest split scored
            lower. ``NaN`` entries are dropped.
        comparison: Label for the pair of strategies being compared.
        metric: The metric the gaps are measured in.
        labels: Optional cell labels, used to name the contradicting cells.

    Returns:
        The summary.

    Raises:
        ValueError: If no finite gaps are supplied.
    """
    pairs = [
        (g, labels[i] if labels is not None and i < len(labels) else str(i))
        for i, g in enumerate(gaps)
        if g == g and math.isfinite(g)
    ]
    if not pairs:
        raise ValueError("need at least one finite gap to summarise")
    values = np.asarray([p[0] for p in pairs], dtype=float)
    n = len(values)
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else 0.0
    half = t_critical_95(n - 1) * se if n > 1 else 0.0
    statistic, p = wilcoxon_signed_rank(values)
    return GapSummary(
        comparison=comparison,
        metric=metric,
        n=n,
        mean=mean,
        sd=sd,
        se=se,
        ci_low=mean - half,
        ci_high=mean + half,
        median=float(np.median(values)),
        n_positive=int((values > 0).sum()),
        n_negative=int((values < 0).sum()),
        wilcoxon_w=statistic,
        wilcoxon_p=p,
        exceptions=tuple(label for value, label in pairs if value < 0),
    )


def summarise_repeats(
    scores_by_strategy: dict[str, Sequence[float]],
    reference: str = "random",
) -> list[dict[str, object]]:
    """Summarise repeated split draws of the same target and model.

    Args:
        scores_by_strategy: Metric values per strategy, one per seed. Every
            strategy is expected to carry the same number of seeds.
        reference: The strategy the others are compared against.

    Returns:
        One row per strategy, each carrying the mean and standard deviation
        over seeds and, for the non-reference strategies, the mean gap and
        its standard deviation across seeds.

    Raises:
        KeyError: If ``reference`` is absent.
    """
    if reference not in scores_by_strategy:
        raise KeyError(f"no scores for reference strategy {reference!r}")
    ref = np.asarray(scores_by_strategy[reference], dtype=float)
    rows: list[dict[str, object]] = []
    for strategy, values in scores_by_strategy.items():
        array = np.asarray(values, dtype=float)
        row: dict[str, object] = {
            "strategy": strategy,
            "n_seeds": int(len(array)),
            "mean": round(float(array.mean()), 4),
            "sd": round(float(array.std(ddof=1)) if len(array) > 1 else 0.0, 4),
            "min": round(float(array.min()), 4),
            "max": round(float(array.max()), 4),
        }
        if strategy != reference and len(array) == len(ref):
            gap = ref - array
            row["mean_gap_vs_reference"] = round(float(gap.mean()), 4)
            row["sd_gap_vs_reference"] = round(
                float(gap.std(ddof=1)) if len(gap) > 1 else 0.0, 4
            )
        rows.append(row)
    return rows


def time_cutoff_grid(
    years: Sequence[float], fractions: Sequence[float] = (0.15, 0.2, 0.25)
) -> list[int]:
    """Cutoff years that hold out roughly each of ``fractions``.

    A time split has no seed, so its sensitivity analysis is over where the
    line is drawn rather than over a random draw.

    Args:
        years: First-publication year per compound; non-finite entries are
            ignored.
        fractions: Target test fractions.

    Returns:
        The distinct cutoff years, ascending. A compound is in the test set
        when its year is greater than or equal to the cutoff.

    Raises:
        ValueError: If no finite years are supplied.
    """
    finite = np.asarray(
        [y for y in years if y is not None and y == y and math.isfinite(y)],
        dtype=float,
    )
    if finite.size == 0:
        raise ValueError("need at least one finite year")
    cutoffs = {
        int(math.ceil(float(np.quantile(finite, 1.0 - f)))) for f in fractions
    }
    return sorted(cutoffs)
