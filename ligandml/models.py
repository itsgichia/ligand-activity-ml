"""Estimators for both framings of the activity-prediction problem.

The study runs two framings over the same data:

Regression on pChEMBL
    Predict the potency directly.  Keeps all the information in the label
    and avoids inventing a threshold, but is punished by the heterogeneity
    of public assay data, where the same compound measured in two labs can
    differ by a log unit.
Classification at a stated threshold
    Predict active / inactive at pChEMBL >= 6.0.  Throws away potency
    detail but is robust to that same noise, and matches the decision a
    screening campaign actually makes: order this compound or not.

Four model families, chosen to span capacity rather than to win a
leaderboard: a regularised linear model, a bagged tree ensemble, a boosted
tree ensemble, and a linear support vector machine.  Nothing here is
hyperparameter-tuned per target, deliberately: the comparison of interest
is between *evaluation protocols*, and tuning each cell of the matrix would
confound it with tuning noise.  Defaults are the standard ones.

Every estimator is wrapped in a pipeline that matches the representation:
dense descriptor blocks are imputed and standardised, binary fingerprints
have their constant (all-zero) bits dropped, which costs nothing and makes
the tree ensembles several times faster.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, LinearSVR

TASKS: Final[tuple[str, ...]] = ("regression", "classification")
"""The two framings."""


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """A model family available to the study.

    Attributes:
        name: Slug used in result tables, e.g. ``"random_forest"``.
        label: Human-readable description.
        cost: Rough relative cost, used by ``--fast`` to drop the
            expensive members of the matrix.
    """

    name: str
    label: str
    cost: int = 1


MODELS: Final[tuple[ModelSpec, ...]] = (
    ModelSpec("linear", "Logistic regression (L2) / ridge regression", 1),
    ModelSpec("svm", "Linear SVM, Platt-scaled for probabilities", 2),
    ModelSpec("random_forest", "Random forest, 300 trees", 3),
    ModelSpec("gradient_boosting", "Histogram gradient boosting", 3),
)

MODELS_BY_NAME: Final[dict[str, ModelSpec]] = {m.name: m for m in MODELS}
"""Lookup from slug to :class:`ModelSpec`."""

DEFAULT_MODEL: Final[str] = "random_forest"
"""Model used wherever the study needs one representative estimator."""

N_TREES: Final[int] = 300
"""Trees in the random forest. Past a few hundred, more changes nothing."""

BOOSTING_ITERATIONS: Final[int] = 200
"""Boosting rounds, with early stopping off so every cell of the matrix
fits for exactly the same budget and the comparison stays clean.
"""

BOOSTING_MAX_BINS: Final[int] = 16
"""Histogram bins for the boosted trees.

The default of 255 is for continuous features. These representations are
binary or low-cardinality, so 16 bins lose nothing measurable and cut
fitting time roughly threefold, which is what keeps the full matrix inside
its time budget.
"""

BOOSTING_MAX_FEATURES: Final[float] = 0.3
"""Fraction of features considered per split by the boosted trees."""

SVM_REGULARISATION: Final[float] = 1.0
"""Inverse regularisation strength for the SVM classifier."""

SVR_REGULARISATION: Final[float] = 0.1
"""Inverse regularisation strength for the SVM regressor.

Lower than the classifier's, because on high-dimensional sparse
fingerprints the weaker-regularised regressor neither converges within a
sane iteration budget nor generalises as well.
"""


def _preprocessor(representation_kind: str) -> list[tuple[str, Any]]:
    """Return pipeline steps appropriate to a representation.

    Args:
        representation_kind: ``"descriptors"`` or ``"fingerprint"``.

    Returns:
        A list of ``(name, transformer)`` pipeline steps.

    Raises:
        ValueError: On an unknown representation kind.
    """
    if representation_kind == "descriptors":
        return [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    if representation_kind == "fingerprint":
        # Drop bits that are constant across the training set: no model can
        # use them, and removing them cuts tree-building time sharply.
        return [("drop_constant", VarianceThreshold(threshold=0.0))]
    raise ValueError(f"unknown representation kind {representation_kind!r}")


def _classifier(name: str, seed: int) -> BaseEstimator:
    """Return the classifier for a model slug.

    Args:
        name: Model slug from :data:`MODELS`.
        seed: Random seed.

    Returns:
        An unfitted scikit-learn classifier.

    Raises:
        KeyError: On an unknown model slug.
    """
    if name == "linear":
        return LogisticRegression(
            C=1.0, max_iter=3000, class_weight="balanced", random_state=seed
        )
    if name == "svm":
        # LinearSVC has no predict_proba, so Platt scaling via internal
        # cross-validation supplies one. Note in the calibration results
        # that this model is calibrated by construction and the others are
        # not; that is exactly why it looks better there.
        return CalibratedClassifierCV(
            LinearSVC(
                C=SVM_REGULARISATION, class_weight="balanced", random_state=seed
            ),
            method="sigmoid",
            cv=3,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=N_TREES,
            min_samples_leaf=1,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )
    if name == "gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=BOOSTING_ITERATIONS,
            learning_rate=0.1,
            max_bins=BOOSTING_MAX_BINS,
            max_features=BOOSTING_MAX_FEATURES,
            early_stopping=False,
            random_state=seed,
        )
    raise KeyError(f"unknown model {name!r}; known: {', '.join(MODELS_BY_NAME)}")


def _regressor(name: str, seed: int) -> BaseEstimator:
    """Return the regressor for a model slug.

    Args:
        name: Model slug from :data:`MODELS`.
        seed: Random seed.

    Returns:
        An unfitted scikit-learn regressor.

    Raises:
        KeyError: On an unknown model slug.
    """
    if name == "linear":
        return Ridge(alpha=1.0, random_state=seed)
    if name == "svm":
        return LinearSVR(
            C=SVR_REGULARISATION, max_iter=50_000, random_state=seed
        )
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=N_TREES,
            min_samples_leaf=1,
            max_features="sqrt",
            n_jobs=-1,
            random_state=seed,
        )
    if name == "gradient_boosting":
        return HistGradientBoostingRegressor(
            max_iter=BOOSTING_ITERATIONS,
            learning_rate=0.1,
            max_bins=BOOSTING_MAX_BINS,
            max_features=BOOSTING_MAX_FEATURES,
            early_stopping=False,
            random_state=seed,
        )
    raise KeyError(f"unknown model {name!r}; known: {', '.join(MODELS_BY_NAME)}")


def build_model(
    name: str,
    task: str,
    representation_kind: str = "fingerprint",
    seed: int = 0,
) -> Pipeline:
    """Build a fitted-ready pipeline for one model, task and representation.

    Args:
        name: Model slug from :data:`MODELS`.
        task: ``"regression"`` or ``"classification"``.
        representation_kind: ``"descriptors"`` or ``"fingerprint"``.
        seed: Random seed for the estimator.

    Returns:
        An unfitted :class:`~sklearn.pipeline.Pipeline`.

    Raises:
        ValueError: On an unknown task or representation kind.
        KeyError: On an unknown model slug.
    """
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}, got {task!r}")
    estimator = (
        _regressor(name, seed) if task == "regression" else _classifier(name, seed)
    )
    steps = _preprocessor(representation_kind)
    steps.append(("model", estimator))
    return Pipeline(steps)


def fit_predict(
    model: Pipeline,
    x_train: np.ndarray,
    y_train: Sequence[float],
    x_test: np.ndarray,
    task: str,
) -> np.ndarray:
    """Fit a pipeline and return test-set predictions.

    For classification the returned values are probabilities of the active
    class; for regression they are predicted pChEMBL values.

    Args:
        model: An unfitted pipeline from :func:`build_model`.
        x_train: Training feature matrix.
        y_train: Training labels or values.
        x_test: Test feature matrix.
        task: ``"regression"`` or ``"classification"``.

    Returns:
        Predictions for ``x_test``.

    Raises:
        ValueError: On an unknown task.
    """
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}, got {task!r}")
    model.fit(x_train, np.asarray(y_train))
    if task == "regression":
        return np.asarray(model.predict(x_test), dtype=float)
    probabilities = model.predict_proba(x_test)
    return np.asarray(probabilities[:, 1], dtype=float)


def get_model_spec(name: str) -> ModelSpec:
    """Look up a model by slug.

    Args:
        name: Model slug.

    Returns:
        The matching :class:`ModelSpec`.

    Raises:
        KeyError: If the slug is unknown.
    """
    try:
        return MODELS_BY_NAME[name]
    except KeyError as exc:
        known = ", ".join(sorted(MODELS_BY_NAME))
        raise KeyError(f"unknown model {name!r}; known: {known}") from exc
