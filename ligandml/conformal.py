"""Conformal prediction: coverage you can hold the model to.

The applicability domain section of this study shows error rising as a
compound sits further from the training set, and the coverage curve shows
what refusing the far ones buys. Both are diagnostics: they describe the
model's behaviour without committing it to anything.

Conformal prediction turns that into a guarantee. Given a calibration set
the model never trained on, and a significance level, it emits a
*prediction set* per compound with a proved property: under exchangeability
the true label falls inside the set at least ``1 - epsilon`` of the time.
Not approximately, not on average over a favourable benchmark - by
construction, for any underlying model, including a badly calibrated one.

That is the form a discovery team can act on. "Rank these by score" invites
the question *how far down do I read?*. "These 312 compounds, 90%
confidence, and for 180 of them the prediction is unambiguous" is a
decision.

Two refinements matter here and both are implemented:

**Mondrian (class-conditional) calibration.** Validity across the whole
set can be bought by being right about the majority class and hopeless on
the minority - fatal when the minority class is the actives. Calibrating
each class against its own quantile makes the guarantee hold *per class*,
which is the version worth having on an imbalanced screen.

**Four outcomes, not two.** A prediction set can hold one label (a
confident call), both (the model declines to separate them) or neither (the
compound resembles no calibration example of either class). Empty sets are
not a failure mode to be hidden - on a screening library full of chemistry
unlike anything in training, an empty set is the model saying so.

The implementation is inductive (split) conformal: fit once, calibrate
once, apply to everything after. Transductive conformal would be tighter
and costs a refit per test compound, which is not affordable at the sizes
here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

#: Significance levels reported by default: 80%, 90% and 95% confidence.
DEFAULT_SIGNIFICANCES: Final[tuple[float, ...]] = (0.2, 0.1, 0.05)


def nonconformity(probabilities: np.ndarray, labels: Sequence[int]) -> np.ndarray:
    """Score how poorly each example fits its own label.

    The standard choice for a probabilistic classifier: one minus the
    probability assigned to the true class. Larger is stranger.

    Args:
        probabilities: ``(n, 2)`` class probabilities, or ``(n,)`` taken as
            the probability of the positive class.
        labels: The true labels, 0 or 1.

    Returns:
        One score per example.

    Raises:
        ValueError: If the shapes disagree or a label is not 0 or 1.
    """
    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim == 1:
        probs = np.column_stack([1.0 - probs, probs])
    if probs.ndim != 2 or probs.shape[1] != 2:
        raise ValueError(f"expected (n, 2) probabilities, got {probs.shape}")
    y = np.asarray(labels, dtype=int)
    if len(y) != len(probs):
        raise ValueError(f"got {len(probs)} probabilities for {len(y)} labels")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("labels must be 0 or 1")
    return 1.0 - probs[np.arange(len(y)), y]


@dataclass(frozen=True)
class ConformalCalibration:
    """Calibration scores, kept per class for Mondrian validity.

    Attributes:
        scores_by_class: Sorted nonconformity scores for each class.
    """

    scores_by_class: dict[int, np.ndarray]

    @property
    def n_calibration(self) -> int:
        """Total calibration examples."""
        return int(sum(len(v) for v in self.scores_by_class.values()))

    def p_values(self, probabilities: np.ndarray) -> np.ndarray:
        """Conformal p-value for each test compound under each label.

        The p-value for label ``c`` is the fraction of class-``c``
        calibration examples at least as strange as this compound would be
        if it were class ``c``. The ``+1`` in numerator and denominator is
        the standard smoothing that keeps the guarantee valid at finite
        calibration sizes rather than merely asymptotically.

        Args:
            probabilities: ``(n, 2)`` class probabilities, or ``(n,)``
                taken as the probability of the positive class.

        Returns:
            An ``(n, 2)`` array of p-values.
        """
        probs = np.asarray(probabilities, dtype=float)
        if probs.ndim == 1:
            probs = np.column_stack([1.0 - probs, probs])
        out = np.zeros((len(probs), 2), dtype=float)
        for label in (0, 1):
            calibration = self.scores_by_class.get(label)
            if calibration is None or len(calibration) == 0:
                # No calibration examples: refuse to claim anything, which
                # shows up as the label never entering a prediction set.
                out[:, label] = 0.0
                continue
            hypothetical = 1.0 - probs[:, label]
            at_least_as_strange = len(calibration) - np.searchsorted(
                calibration, hypothetical, side="left"
            )
            out[:, label] = (at_least_as_strange + 1.0) / (len(calibration) + 1.0)
        return out


def calibrate(
    probabilities: np.ndarray, labels: Sequence[int]
) -> ConformalCalibration:
    """Build a Mondrian calibration from held-out predictions.

    Args:
        probabilities: Predicted probabilities for the calibration set.
        labels: Its true labels.

    Returns:
        The calibration.

    Raises:
        ValueError: If the calibration set is empty.
    """
    y = np.asarray(labels, dtype=int)
    if len(y) == 0:
        raise ValueError("need a non-empty calibration set")
    scores = nonconformity(probabilities, y)
    return ConformalCalibration(
        scores_by_class={
            label: np.sort(scores[y == label]) for label in np.unique(y)
        }
    )


def prediction_sets(p_values: np.ndarray, significance: float) -> np.ndarray:
    """Which labels survive at this significance.

    Args:
        p_values: ``(n, 2)`` conformal p-values.
        significance: Epsilon; the guarantee is coverage of at least
            ``1 - significance``.

    Returns:
        An ``(n, 2)`` boolean array; entry ``[i, c]`` is True when label
        ``c`` is in compound ``i``'s prediction set.

    Raises:
        ValueError: If ``significance`` is outside (0, 1).
    """
    if not 0.0 < significance < 1.0:
        raise ValueError(f"significance must be in (0, 1), got {significance}")
    return np.asarray(p_values, dtype=float) > significance


@dataclass(frozen=True)
class ConformalReport:
    """What a conformal run achieved at one significance level.

    Attributes:
        significance: The epsilon used.
        target_coverage: ``1 - significance``, what was promised.
        coverage: Fraction of compounds whose set held the true label.
        coverage_by_class: The same, per class. This is the number
            Mondrian calibration exists to protect.
        singleton_rate: Fraction with exactly one label, the efficiency
            of the predictor.
        empty_rate: Fraction with no label.
        both_rate: Fraction with both labels.
        singleton_accuracy: Accuracy among the confident calls only.
        n: Number of test compounds.
    """

    significance: float
    target_coverage: float
    coverage: float
    coverage_by_class: dict[int, float]
    singleton_rate: float
    empty_rate: float
    both_rate: float
    singleton_accuracy: float
    n: int

    @property
    def is_valid(self) -> bool:
        """Whether observed coverage met the promise, within sampling noise.

        A conformal predictor is allowed to miss slightly on a finite test
        set; the tolerance here is two standard errors of a binomial at
        the target rate.
        """
        if self.n == 0:
            return False
        se = float(
            np.sqrt(self.target_coverage * (1.0 - self.target_coverage) / self.n)
        )
        return self.coverage >= self.target_coverage - 2.0 * se

    def as_row(self) -> dict[str, object]:
        """Return the report as a flat row for a results table."""
        return {
            "significance": self.significance,
            "target_coverage": round(self.target_coverage, 3),
            "coverage": round(self.coverage, 4),
            "coverage_inactive": round(self.coverage_by_class.get(0, float("nan")), 4),
            "coverage_active": round(self.coverage_by_class.get(1, float("nan")), 4),
            "singleton_rate": round(self.singleton_rate, 4),
            "empty_rate": round(self.empty_rate, 4),
            "both_rate": round(self.both_rate, 4),
            "singleton_accuracy": round(self.singleton_accuracy, 4),
            "n_test": self.n,
            "valid": self.is_valid,
        }


def evaluate_sets(
    sets: np.ndarray, labels: Sequence[int], significance: float
) -> ConformalReport:
    """Score prediction sets for validity and efficiency.

    Validity is the guarantee: did the true label land inside the set as
    often as promised? Efficiency is whether the sets are useful: a
    predictor that always returns both labels is perfectly valid and
    perfectly useless.

    Args:
        sets: ``(n, 2)`` boolean prediction sets.
        labels: True labels.
        significance: The epsilon the sets were built at.

    Returns:
        The report.

    Raises:
        ValueError: If the shapes disagree.
    """
    membership = np.asarray(sets, dtype=bool)
    y = np.asarray(labels, dtype=int)
    if len(membership) != len(y):
        raise ValueError(f"got {len(membership)} sets for {len(y)} labels")
    n = len(y)
    if n == 0:
        raise ValueError("need at least one test compound")

    covered = membership[np.arange(n), y]
    sizes = membership.sum(axis=1)
    singleton = sizes == 1
    singleton_correct = covered & singleton
    return ConformalReport(
        significance=significance,
        target_coverage=1.0 - significance,
        coverage=float(covered.mean()),
        coverage_by_class={
            int(label): float(covered[y == label].mean())
            for label in np.unique(y)
        },
        singleton_rate=float(singleton.mean()),
        empty_rate=float((sizes == 0).mean()),
        both_rate=float((sizes == 2).mean()),
        singleton_accuracy=(
            float(singleton_correct.sum() / singleton.sum())
            if singleton.any()
            else float("nan")
        ),
        n=n,
    )


def run(
    calibration_probabilities: np.ndarray,
    calibration_labels: Sequence[int],
    test_probabilities: np.ndarray,
    test_labels: Sequence[int],
    significances: Sequence[float] = DEFAULT_SIGNIFICANCES,
) -> list[ConformalReport]:
    """Calibrate once, then report validity and efficiency per level.

    Args:
        calibration_probabilities: Predictions on the calibration split.
        calibration_labels: Its true labels.
        test_probabilities: Predictions on the test split.
        test_labels: Its true labels.
        significances: Epsilons to report.

    Returns:
        One report per significance, in the order given.
    """
    calibration = calibrate(calibration_probabilities, calibration_labels)
    p = calibration.p_values(test_probabilities)
    return [
        evaluate_sets(prediction_sets(p, epsilon), test_labels, epsilon)
        for epsilon in significances
    ]
