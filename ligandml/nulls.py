"""Y-scrambling: proving the signal is real before interpreting it.

Any flexible model fitted to a few thousand compounds and a few thousand
features can find *something*, and a single held-out score gives no way to
tell a real structure-activity relationship from a well-dressed artefact.
The standard remedy in QSAR is y-scrambling: permute the labels, refit the
whole pipeline, and score it exactly as the real model was scored.  Repeat
to build a null distribution.

If the real score sits inside that null distribution, the model has learned
nothing beyond the dataset's shape, no matter how respectable the number
looks in isolation.  If the real score sits well outside it, there is
signal, and the permutation test gives a p-value for that claim.

Two details matter for the test to mean anything:

* **Permute the labels, not the features.** Permuting labels destroys the
  structure-activity relationship while keeping the chemical space, the
  class balance, and any peculiarity of the split exactly as they were.
* **Refit inside the loop.** Scrambling after fitting tests nothing. The
  pipeline, including its preprocessing, is refitted on every permutation.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from ligandml.models import fit_predict

LOGGER = logging.getLogger(__name__)

N_PERMUTATIONS: Final[int] = 20
"""Default permutations. Enough to place a score against the null and to
support a one-sided p-value resolution of about 1/(n+1).
"""


@dataclass(slots=True)
class NullResult:
    """Outcome of a y-scrambling experiment.

    Attributes:
        metric: Name of the scored metric.
        observed: Score of the model trained on the real labels.
        null_scores: Scores from each label permutation.
        p_value: One-sided permutation p-value,
            ``(1 + #{null >= observed}) / (1 + n_permutations)``, which is
            the standard conservative estimator and never returns zero.
        z_score: How many null standard deviations the observed score sits
            above the null mean; ``nan`` if the null has no spread.
    """

    metric: str
    observed: float
    null_scores: np.ndarray
    p_value: float
    z_score: float

    @property
    def null_mean(self) -> float:
        """Mean of the null distribution."""
        return float(np.mean(self.null_scores)) if self.null_scores.size else float("nan")

    @property
    def null_std(self) -> float:
        """Standard deviation of the null distribution."""
        return float(np.std(self.null_scores)) if self.null_scores.size else float("nan")

    def to_dict(self) -> dict[str, Any]:
        """Return a flat, JSON- and CSV-friendly summary."""
        return {
            "metric": self.metric,
            "observed": self.observed,
            "null_mean": self.null_mean,
            "null_std": self.null_std,
            "null_min": (
                float(np.min(self.null_scores))
                if self.null_scores.size
                else float("nan")
            ),
            "null_max": (
                float(np.max(self.null_scores))
                if self.null_scores.size
                else float("nan")
            ),
            "n_permutations": int(self.null_scores.size),
            "p_value": self.p_value,
            "z_score": self.z_score,
        }


def y_scramble(
    model: Pipeline,
    x_train: np.ndarray,
    y_train: Sequence[float],
    x_test: np.ndarray,
    y_test: Sequence[float],
    scorer: Callable[[np.ndarray, np.ndarray], float],
    task: str = "classification",
    metric: str = "roc_auc",
    n_permutations: int = N_PERMUTATIONS,
    seed: int = 0,
) -> NullResult:
    """Build a y-scrambled null distribution for one model and split.

    Args:
        model: An unfitted pipeline; it is cloned before every fit, so the
            object passed in is never mutated.
        x_train: Training feature matrix.
        y_train: Training labels or values.
        x_test: Test feature matrix.
        y_test: Test labels or values, left unpermuted throughout.
        scorer: Callable ``(y_test, predictions) -> score``, where a higher
            score is better.
        task: ``"regression"`` or ``"classification"``.
        metric: Name of the metric, recorded in the result.
        n_permutations: Number of label permutations.
        seed: Seed for the permutations.

    Returns:
        A :class:`NullResult`.

    Raises:
        ValueError: If ``n_permutations`` is below 1 or inputs mismatch.
    """
    if n_permutations < 1:
        raise ValueError(f"n_permutations must be at least 1, got {n_permutations}")
    y_train_array = np.asarray(y_train)
    y_test_array = np.asarray(y_test)
    if y_train_array.shape[0] != x_train.shape[0]:
        raise ValueError("x_train and y_train disagree on the number of rows")
    if y_test_array.shape[0] != x_test.shape[0]:
        raise ValueError("x_test and y_test disagree on the number of rows")

    observed_prediction = fit_predict(
        clone(model), x_train, y_train_array, x_test, task
    )
    observed = float(scorer(y_test_array, observed_prediction))

    rng = np.random.default_rng(seed)
    scores: list[float] = []
    for index in range(n_permutations):
        shuffled = rng.permutation(y_train_array)
        try:
            prediction = fit_predict(clone(model), x_train, shuffled, x_test, task)
            score = float(scorer(y_test_array, prediction))
        except ValueError as exc:  # A permutation can leave a degenerate class.
            LOGGER.warning("permutation %d failed to score: %s", index, exc)
            continue
        if np.isfinite(score):
            scores.append(score)

    null = np.asarray(scores, dtype=float)
    n_valid = null.size
    p_value = (
        float(1 + int((null >= observed).sum())) / float(1 + n_valid)
        if n_valid
        else float("nan")
    )
    spread = float(np.std(null)) if n_valid else float("nan")
    z_score = (
        float((observed - float(np.mean(null))) / spread)
        if n_valid and spread > 0
        else float("nan")
    )
    return NullResult(
        metric=metric,
        observed=observed,
        null_scores=null,
        p_value=p_value,
        z_score=z_score,
    )
