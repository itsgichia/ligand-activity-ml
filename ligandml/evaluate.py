"""Metrics for regression, classification and - above all - screening.

Why screening metrics are separate
----------------------------------
ROC AUC asks how well a model orders the *whole* library.  A screening
campaign never buys the whole library: it buys the top of the ranked list.
A model can have a respectable AUC of 0.80 and still put nothing useful in
the first hundred compounds you order, because AUC averages over regions of
the ranking nobody will ever look at.

So the metrics this module treats as primary are the early-recognition ones:

Enrichment factor (EF)
    The hit rate in the top fraction of the ranking, divided by the hit rate
    of the whole set.  EF = 10 at 1% means that screening the top 1% finds
    ten times as many actives per compound tested as random picking.  Its
    ceiling is ``min(1/f, 1/active_rate)``: 1/f because the top slice can
    hold only so many compounds, and 1/active_rate because a perfect model
    can at best raise the hit rate to 1.  On a set that is already mostly
    active the second term dominates and EF has almost no headroom, which
    is a property of the dataset rather than of the model - so
    :func:`classification_metrics` reports the attainable ceiling
    (``ef_ceiling_1pct``) beside every EF.
BEDROC
    A continuous, bounded (0-1) early-recognition score with an exponential
    weight on rank, which fixes EF's two weaknesses: EF is blind to the
    ordering *within* the top slice and jumps discontinuously as compounds
    cross the cut.  The weight parameter alpha is set here from the fraction
    of interest: alpha = ln(5)/f places 80% of the total weight in the top
    f of the list, giving alpha = 160.9 for 1% and 32.2 for 5%.

    One trap worth stating plainly, because it is widely misreported:
    **BEDROC = 0.5 does not mean random ranking.**  Unlike ROC AUC, BEDROC's
    expectation under random ranking depends on alpha and on the active
    ratio, and for a sparse active set at large alpha it is far below 0.5.
    :func:`bedroc_random_expectation` computes it, and
    :func:`classification_metrics` reports it beside every BEDROC value so
    the number can be read against the right baseline.
Calibration
    Whether a predicted probability of 0.8 means 80% of such compounds are
    active.  A screen that triages by a probability threshold, or that feeds
    predictions into an expected-value calculation, needs calibration;
    ranking metrics say nothing about it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    matthews_corrcoef,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)

TOP_FRACTIONS: Final[tuple[float, ...]] = (0.01, 0.05)
"""Ranked-list fractions at which screening metrics are reported."""

#: pChEMBL threshold separating active from inactive.
#:
#: pChEMBL 6.0 is 1 micromolar.  It is the conventional cut for "this
#: compound does something at this target" in public bioactivity data: it is
#: roughly where cell-based assays stop being confounded by non-specific
#: effects, it is the usual floor for a hit worth optimising, and - being a
#: round number in the middle of the ChEMBL potency distribution - it splits
#: these datasets without creating a degenerate minority class.  The choice
#: matters, so :func:`classification_metrics` records it in every result row
#: and the study reports the class balance it produces per target.
ACTIVITY_THRESHOLD: Final[float] = 6.0

N_BOOTSTRAP: Final[int] = 1000
"""Bootstrap resamples used for confidence intervals."""

CALIBRATION_BINS: Final[int] = 10
"""Equal-width probability bins used for the reliability curve and ECE."""


def bedroc_alpha(fraction: float) -> float:
    """Return the BEDROC alpha that puts 80% of the weight in the top fraction.

    Args:
        fraction: Fraction of the ranked list of interest, in (0, 1).

    Returns:
        The alpha parameter, ``ln(5) / fraction``.

    Raises:
        ValueError: If ``fraction`` is not in the open interval (0, 1).
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must be in (0, 1), got {fraction}")
    return math.log(5.0) / fraction


def _ranked_labels(y_true: Sequence[int], scores: Sequence[float]) -> np.ndarray:
    """Order binary labels by descending score, breaking ties pessimistically.

    Ties are broken by placing inactives first, so a model that gives every
    compound the same score scores like random picking rather than being
    flattered by label order.

    Args:
        y_true: Binary labels, 1 for active.
        scores: Ranking scores; higher means more likely active.

    Returns:
        The labels in ranked order.

    Raises:
        ValueError: If the inputs differ in length or are empty.
    """
    labels = np.asarray(y_true, dtype=int).ravel()
    values = np.asarray(scores, dtype=float).ravel()
    if labels.size != values.size:
        raise ValueError(
            f"y_true has {labels.size} entries but scores has {values.size}"
        )
    if labels.size == 0:
        raise ValueError("cannot rank an empty set")
    order = np.lexsort((labels, -values))
    return labels[order]


def enrichment_factor(
    y_true: Sequence[int], scores: Sequence[float], fraction: float
) -> float:
    """Compute the enrichment factor at a fraction of the ranked list.

    ``EF_f = (actives in top k / k) / (total actives / N)`` where
    ``k = max(1, round(f * N))``.

    Args:
        y_true: Binary labels, 1 for active.
        scores: Ranking scores; higher means more likely active.
        fraction: Fraction of the list to screen, in (0, 1].

    Returns:
        The enrichment factor, or ``nan`` if the set contains no actives.

    Raises:
        ValueError: If ``fraction`` is outside (0, 1].
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    ranked = _ranked_labels(y_true, scores)
    n_total = ranked.size
    n_actives = int(ranked.sum())
    if n_actives == 0:
        return float("nan")
    k = max(1, int(round(fraction * n_total)))
    hit_rate_top = float(ranked[:k].sum()) / k
    hit_rate_all = n_actives / n_total
    return hit_rate_top / hit_rate_all


def hit_rate(
    y_true: Sequence[int], scores: Sequence[float], fraction: float
) -> float:
    """Fraction of the top slice of the ranking that is truly active.

    Args:
        y_true: Binary labels, 1 for active.
        scores: Ranking scores; higher means more likely active.
        fraction: Fraction of the list to screen, in (0, 1].

    Returns:
        The hit rate in the top slice.

    Raises:
        ValueError: If ``fraction`` is outside (0, 1].
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    ranked = _ranked_labels(y_true, scores)
    k = max(1, int(round(fraction * ranked.size)))
    return float(ranked[:k].sum()) / k


def recall_at_fraction(
    y_true: Sequence[int], scores: Sequence[float], fraction: float
) -> float:
    """Fraction of all true actives recovered in the top slice.

    This is the number a screening demo should quote: "the top 1% of the
    ranked list contains X% of the actives".

    Args:
        y_true: Binary labels, 1 for active.
        scores: Ranking scores; higher means more likely active.
        fraction: Fraction of the list to screen, in (0, 1].

    Returns:
        Recall within the top slice, or ``nan`` if there are no actives.

    Raises:
        ValueError: If ``fraction`` is outside (0, 1].
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    ranked = _ranked_labels(y_true, scores)
    n_actives = int(ranked.sum())
    if n_actives == 0:
        return float("nan")
    k = max(1, int(round(fraction * ranked.size)))
    return float(ranked[:k].sum()) / n_actives


def bedroc(
    y_true: Sequence[int], scores: Sequence[float], alpha: float = 20.0
) -> float:
    """Compute BEDROC, the Boltzmann-enhanced discrimination of ROC.

    Implements Truchon and Bayly (2007), *J. Chem. Inf. Model.* 47, 488:
    the robust initial enhancement is computed from the exponentially
    weighted ranks of the actives and then min-max normalised against the
    best and worst achievable values at the observed active ratio, so the
    result lies in [0, 1] with 0.5 meaning random ranking.

    Args:
        y_true: Binary labels, 1 for active.
        scores: Ranking scores; higher means more likely active.
        alpha: Exponential weight. Larger values concentrate the weight
            further up the list; see :func:`bedroc_alpha`.

    Note that the random-ranking expectation is *not* 0.5; use
    :func:`bedroc_random_expectation` for the baseline this value should be
    read against.

    Returns:
        BEDROC in [0, 1], or ``nan`` if the set is all-active or all-inactive
        (in which case no ranking metric is defined).

    Raises:
        ValueError: If ``alpha`` is not positive.
    """
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    ranked = _ranked_labels(y_true, scores)
    n_total = ranked.size
    n_actives = int(ranked.sum())
    if n_actives in (0, n_total):
        return float("nan")

    ratio = n_actives / n_total
    active_ranks = np.flatnonzero(ranked == 1) + 1  # 1-based ranks
    sum_exp = float(np.exp(-alpha * active_ranks / n_total).sum())

    # Random-expectation normaliser for the raw sum.
    random_sum = (1.0 - math.exp(-alpha)) / (math.exp(alpha / n_total) - 1.0)
    rie = (sum_exp / n_actives) / (random_sum / n_total)

    rie_max, rie_min = _rie_bounds(ratio, alpha)
    if math.isclose(rie_max, rie_min):
        return float("nan")
    return float((rie - rie_min) / (rie_max - rie_min))


def _rie_bounds(ratio: float, alpha: float) -> tuple[float, float]:
    """Return the best and worst achievable RIE at an active ratio.

    Args:
        ratio: Fraction of the set that is active.
        alpha: BEDROC exponential weight.

    Returns:
        A ``(rie_max, rie_min)`` tuple.
    """
    rie_max = (1.0 - math.exp(-alpha * ratio)) / (ratio * (1.0 - math.exp(-alpha)))
    rie_min = (1.0 - math.exp(alpha * ratio)) / (ratio * (1.0 - math.exp(alpha)))
    return rie_max, rie_min


def bedroc_random_expectation(
    n_actives: int, n_total: int, alpha: float = 20.0
) -> float:
    """Return the BEDROC a random ranking is expected to score.

    A random ranking has robust initial enhancement 1 by construction, so
    its BEDROC is ``(1 - RIE_min) / (RIE_max - RIE_min)``.  That value
    depends on alpha and on the active ratio, which is why a BEDROC must
    always be quoted with its own baseline rather than against 0.5.

    Args:
        n_actives: Number of actives in the set.
        n_total: Size of the set.
        alpha: BEDROC exponential weight.

    Returns:
        The expected BEDROC under random ranking, or ``nan`` if the set is
        all-active or all-inactive.

    Raises:
        ValueError: If ``alpha`` is not positive or ``n_total`` is not.
    """
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    if n_total <= 0:
        raise ValueError(f"n_total must be positive, got {n_total}")
    if n_actives in (0, n_total):
        return float("nan")
    ratio = n_actives / n_total
    rie_max, rie_min = _rie_bounds(ratio, alpha)
    if math.isclose(rie_max, rie_min):
        return float("nan")
    return float((1.0 - rie_min) / (rie_max - rie_min))


def regression_metrics(
    y_true: Sequence[float], y_pred: Sequence[float]
) -> dict[str, float]:
    """Compute regression metrics for pChEMBL prediction.

    Args:
        y_true: Observed pChEMBL values.
        y_pred: Predicted pChEMBL values.

    Returns:
        A dictionary with ``r2``, ``rmse``, ``mae``, ``pearson_r``,
        ``spearman_rho`` and ``n``.

    Raises:
        ValueError: If the inputs differ in length or are empty.
    """
    observed = np.asarray(y_true, dtype=float).ravel()
    predicted = np.asarray(y_pred, dtype=float).ravel()
    if observed.size != predicted.size:
        raise ValueError("y_true and y_pred must have the same length")
    if observed.size == 0:
        raise ValueError("cannot score an empty set")
    residuals = predicted - observed
    metrics = {
        "n": float(observed.size),
        "r2": float(r2_score(observed, predicted)) if observed.size > 1 else float("nan"),
        "rmse": float(np.sqrt(np.mean(residuals**2))),
        "mae": float(mean_absolute_error(observed, predicted)),
    }
    if observed.size > 2 and np.std(observed) > 0 and np.std(predicted) > 0:
        metrics["pearson_r"] = float(stats.pearsonr(observed, predicted).statistic)
        metrics["spearman_rho"] = float(stats.spearmanr(observed, predicted).statistic)
    else:
        metrics["pearson_r"] = float("nan")
        metrics["spearman_rho"] = float("nan")
    return metrics


def classification_metrics(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    fractions: Sequence[float] = TOP_FRACTIONS,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute classification and screening metrics from predicted probabilities.

    Args:
        y_true: Binary labels, 1 for active.
        y_prob: Predicted probability of the active class.
        fractions: Ranked-list fractions for the screening metrics.
        threshold: Probability cut for the hard-label metrics.

    Returns:
        A dictionary of ranking metrics (``roc_auc``, ``pr_auc``),
        hard-label metrics (``balanced_accuracy``, ``mcc``), probabilistic
        metrics (``brier``, ``log_loss``, ``ece``), and, per fraction,
        ``ef_{pct}``, ``bedroc_{pct}``, ``hit_rate_{pct}`` and
        ``recall_{pct}``.

    Raises:
        ValueError: If the inputs differ in length or are empty.
    """
    labels = np.asarray(y_true, dtype=int).ravel()
    probs = np.asarray(y_prob, dtype=float).ravel()
    if labels.size != probs.size:
        raise ValueError("y_true and y_prob must have the same length")
    if labels.size == 0:
        raise ValueError("cannot score an empty set")

    n_actives = int(labels.sum())
    single_class = n_actives in (0, labels.size)
    metrics: dict[str, float] = {
        "n": float(labels.size),
        "n_actives": float(n_actives),
        "active_rate": n_actives / labels.size,
    }
    if single_class:
        metrics.update({"roc_auc": float("nan"), "pr_auc": float("nan")})
    else:
        metrics["roc_auc"] = float(roc_auc_score(labels, probs))
        metrics["pr_auc"] = float(average_precision_score(labels, probs))

    hard = (probs >= threshold).astype(int)
    metrics["balanced_accuracy"] = (
        float(balanced_accuracy_score(labels, hard)) if not single_class else float("nan")
    )
    metrics["mcc"] = (
        float(matthews_corrcoef(labels, hard)) if not single_class else float("nan")
    )
    clipped = np.clip(probs, 1e-9, 1.0 - 1e-9)
    metrics["brier"] = float(brier_score_loss(labels, clipped))
    metrics["log_loss"] = (
        float(log_loss(labels, clipped, labels=[0, 1]))
        if not single_class
        else float("nan")
    )
    metrics["ece"] = expected_calibration_error(labels, clipped)

    for fraction in fractions:
        tag = f"{fraction:.0%}".replace("%", "pct")
        alpha = bedroc_alpha(fraction)
        metrics[f"ef_{tag}"] = enrichment_factor(labels, probs, fraction)
        metrics[f"ef_ceiling_{tag}"] = (
            min(1.0 / fraction, labels.size / n_actives)
            if n_actives
            else float("nan")
        )
        metrics[f"bedroc_{tag}"] = bedroc(labels, probs, alpha=alpha)
        metrics[f"bedroc_{tag}_random"] = bedroc_random_expectation(
            n_actives, labels.size, alpha=alpha
        )
        metrics[f"hit_rate_{tag}"] = hit_rate(labels, probs, fraction)
        metrics[f"recall_{tag}"] = recall_at_fraction(labels, probs, fraction)
    metrics["bedroc_alpha20"] = bedroc(labels, probs, alpha=20.0)
    metrics["bedroc_alpha20_random"] = bedroc_random_expectation(
        n_actives, labels.size, alpha=20.0
    )
    return metrics


@dataclass(slots=True)
class CalibrationCurve:
    """A binned reliability curve.

    Attributes:
        bin_centres: Centre of each probability bin.
        mean_predicted: Mean predicted probability within each occupied bin.
        observed_fraction: Observed active fraction within each occupied bin.
        counts: Number of compounds in each occupied bin.
        ece: Expected calibration error, the count-weighted mean absolute
            gap between predicted and observed.
    """

    bin_centres: np.ndarray
    mean_predicted: np.ndarray
    observed_fraction: np.ndarray
    counts: np.ndarray
    ece: float


def calibration_curve(
    y_true: Sequence[int], y_prob: Sequence[float], n_bins: int = CALIBRATION_BINS
) -> CalibrationCurve:
    """Bin predicted probabilities and compare them with observed frequencies.

    Args:
        y_true: Binary labels, 1 for active.
        y_prob: Predicted probability of the active class.
        n_bins: Number of equal-width bins over [0, 1].

    Returns:
        A :class:`CalibrationCurve` covering only the occupied bins.

    Raises:
        ValueError: If ``n_bins`` is below 2 or the inputs are mismatched.
    """
    if n_bins < 2:
        raise ValueError(f"n_bins must be at least 2, got {n_bins}")
    labels = np.asarray(y_true, dtype=int).ravel()
    probs = np.asarray(y_prob, dtype=float).ravel()
    if labels.size != probs.size:
        raise ValueError("y_true and y_prob must have the same length")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    index = np.clip(np.digitize(probs, edges[1:-1], right=False), 0, n_bins - 1)
    centres, predicted, observed, counts = [], [], [], []
    for bin_id in range(n_bins):
        mask = index == bin_id
        if not mask.any():
            continue
        centres.append(0.5 * (edges[bin_id] + edges[bin_id + 1]))
        predicted.append(float(probs[mask].mean()))
        observed.append(float(labels[mask].mean()))
        counts.append(int(mask.sum()))

    count_array = np.asarray(counts, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    observed_array = np.asarray(observed, dtype=float)
    ece = (
        float(
            np.sum(count_array * np.abs(predicted_array - observed_array))
            / count_array.sum()
        )
        if count_array.size
        else float("nan")
    )
    return CalibrationCurve(
        bin_centres=np.asarray(centres, dtype=float),
        mean_predicted=predicted_array,
        observed_fraction=observed_array,
        counts=np.asarray(counts, dtype=int),
        ece=ece,
    )


def expected_calibration_error(
    y_true: Sequence[int], y_prob: Sequence[float], n_bins: int = CALIBRATION_BINS
) -> float:
    """Return the expected calibration error.

    Args:
        y_true: Binary labels, 1 for active.
        y_prob: Predicted probability of the active class.
        n_bins: Number of equal-width probability bins.

    Returns:
        The count-weighted mean absolute gap between predicted probability
        and observed frequency.
    """
    return calibration_curve(y_true, y_prob, n_bins=n_bins).ece


def bootstrap_ci(
    y_true: Sequence[Any],
    y_score: Sequence[float],
    metric: str = "roc_auc",
    n_resamples: int = N_BOOTSTRAP,
    confidence: float = 0.95,
    seed: int = 0,
    **metric_kwargs: Any,
) -> tuple[float, float, float]:
    """Bootstrap a confidence interval for one metric over the test set.

    Resampling is over test compounds with replacement, which captures the
    sampling error of the test set but *not* the much larger variance that
    comes from choosing a different split - that is why the study reports
    several splits rather than one interval.

    Args:
        y_true: Observed labels or values.
        y_score: Predicted probabilities or values.
        metric: One of ``"roc_auc"``, ``"pr_auc"``, ``"r2"``, ``"rmse"``,
            ``"spearman_rho"``, ``"ef"``, ``"bedroc"``, ``"recall"``.
        n_resamples: Number of bootstrap resamples.
        confidence: Interval coverage, e.g. 0.95.
        seed: Seed for the resampler.
        **metric_kwargs: Extra arguments for the metric, such as
            ``fraction`` for ``"ef"`` and ``"recall"`` or ``alpha`` for
            ``"bedroc"``.

    Returns:
        A ``(point_estimate, lower, upper)`` tuple; the bounds are ``nan``
        if too few resamples were computable.

    Raises:
        ValueError: On an unknown metric name or mismatched inputs.
    """
    functions: dict[str, Any] = {
        "roc_auc": lambda t, s: float(roc_auc_score(t, s)),
        "pr_auc": lambda t, s: float(average_precision_score(t, s)),
        "r2": lambda t, s: float(r2_score(t, s)),
        "rmse": lambda t, s: float(
            np.sqrt(
                np.mean(
                    (np.asarray(s, dtype=float) - np.asarray(t, dtype=float)) ** 2
                )
            )
        ),
        "spearman_rho": lambda t, s: float(stats.spearmanr(t, s).statistic),
        "ef": lambda t, s: enrichment_factor(t, s, **metric_kwargs),
        "bedroc": lambda t, s: bedroc(t, s, **metric_kwargs),
        "recall": lambda t, s: recall_at_fraction(t, s, **metric_kwargs),
    }
    if metric not in functions:
        raise ValueError(
            f"unknown metric {metric!r}; known: {', '.join(sorted(functions))}"
        )
    function = functions[metric]
    truth = np.asarray(y_true)
    scores = np.asarray(y_score, dtype=float)
    if truth.size != scores.size:
        raise ValueError("y_true and y_score must have the same length")

    try:
        point = float(function(truth, scores))
    except (ValueError, IndexError):
        point = float("nan")

    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(n_resamples):
        pick = rng.integers(0, truth.size, truth.size)
        try:
            value = float(function(truth[pick], scores[pick]))
        except (ValueError, IndexError):
            continue
        if math.isfinite(value):
            samples.append(value)
    if len(samples) < max(20, n_resamples // 10):
        return point, float("nan"), float("nan")
    tail = 100.0 * (1.0 - confidence) / 2.0
    lower, upper = np.percentile(samples, [tail, 100.0 - tail])
    return point, float(lower), float(upper)


def binarise(
    pchembl: Sequence[float], threshold: float = ACTIVITY_THRESHOLD
) -> np.ndarray:
    """Label compounds active or inactive at a pChEMBL threshold.

    Args:
        pchembl: pChEMBL values.
        threshold: Values greater than or equal to this are active.

    Returns:
        An integer array of 0/1 labels.
    """
    return (np.asarray(pchembl, dtype=float) >= threshold).astype(int)
