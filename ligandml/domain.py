"""Applicability domain: knowing which predictions to ignore.

A fitted model will return a number for any molecule you hand it, including
molecules from chemistry it has never seen.  The number is meaningless
there, but nothing in the model says so.  The applicability domain is the
region where predictions can be trusted, and for a ligand-based model the
cheapest honest definition of that region is similarity to the training set.

This module measures, for each test compound, the Tanimoto similarity to
its nearest training neighbour, and then shows two things:

1. **Error rises as similarity falls.** Binning test compounds by
   nearest-neighbour similarity and plotting the error per bin turns the
   applicability domain from an assertion into a measurement.
2. **There is a coverage/accuracy trade-off.** Refusing to predict below a
   similarity threshold raises accuracy on what is left but shrinks what is
   left.  The curve over thresholds is the thing to report, because the
   right operating point depends on how expensive a false positive is.

Tanimoto is computed from dense binary matrices so that any fingerprint
representation can be used unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

CHUNK_ROWS: Final[int] = 512
"""Test rows per similarity chunk, to bound peak memory."""

SIMILARITY_BINS: Final[tuple[float, ...]] = (
    0.0,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    1.0,
)
"""Bin edges for the error-versus-similarity curve.

Unequal widths on purpose: almost all the interesting behaviour happens
below 0.4, and above 0.7 a compound is effectively an analogue of something
in the training set.
"""

DOMAIN_PERCENTILE: Final[float] = 5.0
"""Percentile of the training set's own neighbour similarities used as the
default in-domain threshold: a test compound is out of domain if it is less
similar to the training set than 95% of training compounds are to each
other.
"""


def _validate_binary(matrix: np.ndarray, name: str) -> np.ndarray:
    """Check a matrix is a non-empty 2D binary array and return it as float32.

    Args:
        matrix: Candidate feature matrix.
        name: Name used in error messages.

    Returns:
        The matrix as ``float32``.

    Raises:
        ValueError: If the matrix is not 2D, is empty, or is not binary.
    """
    array = np.asarray(matrix, dtype=np.float32)
    if array.ndim != 2 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty 2D array, got {array.shape}")
    unique = np.unique(array)
    if unique.size and not np.all(np.isin(unique, (0.0, 1.0))):
        raise ValueError(
            f"{name} must be binary for Tanimoto similarity; "
            "use a fingerprint representation, not a descriptor block"
        )
    return array


def tanimoto_matrix(x_query: np.ndarray, x_reference: np.ndarray) -> np.ndarray:
    """Compute all pairwise Tanimoto similarities between two bit matrices.

    Tanimoto (Jaccard) similarity of two bit vectors is the size of their
    intersection over the size of their union.

    Args:
        x_query: ``(n_query, n_bits)`` binary matrix.
        x_reference: ``(n_reference, n_bits)`` binary matrix.

    Returns:
        An ``(n_query, n_reference)`` array of similarities in [0, 1]; pairs
        where both vectors are all-zero are given similarity 0.

    Raises:
        ValueError: If either matrix is not binary, or their widths differ.
    """
    query = _validate_binary(x_query, "x_query")
    reference = _validate_binary(x_reference, "x_reference")
    if query.shape[1] != reference.shape[1]:
        raise ValueError(
            f"bit width mismatch: {query.shape[1]} vs {reference.shape[1]}"
        )
    query_bits = query.sum(axis=1, keepdims=True)
    reference_bits = reference.sum(axis=1, keepdims=True).T

    out = np.zeros((query.shape[0], reference.shape[0]), dtype=np.float32)
    for start in range(0, query.shape[0], CHUNK_ROWS):
        stop = min(start + CHUNK_ROWS, query.shape[0])
        intersection = query[start:stop] @ reference.T
        union = query_bits[start:stop] + reference_bits - intersection
        with np.errstate(divide="ignore", invalid="ignore"):
            block = np.where(union > 0, intersection / union, 0.0)
        out[start:stop] = block
    return out


def nearest_neighbour_similarity(
    x_query: np.ndarray, x_reference: np.ndarray
) -> np.ndarray:
    """Return each query compound's Tanimoto similarity to its nearest reference.

    Args:
        x_query: ``(n_query, n_bits)`` binary matrix, e.g. the test set.
        x_reference: ``(n_reference, n_bits)`` binary matrix, e.g. the
            training set.

    Returns:
        A ``(n_query,)`` array of maximum similarities.

    Raises:
        ValueError: If either matrix is not binary or their widths differ.
    """
    return tanimoto_matrix(x_query, x_reference).max(axis=1).astype(float)


def self_neighbour_similarity(x_reference: np.ndarray) -> np.ndarray:
    """Nearest-neighbour similarity of a set to itself, excluding self-matches.

    Used to characterise how tightly packed the training set is, which sets
    the scale against which test-set similarities should be read.

    Args:
        x_reference: ``(n, n_bits)`` binary matrix.

    Returns:
        An ``(n,)`` array of each compound's similarity to its nearest
        *other* member.

    Raises:
        ValueError: If the matrix has fewer than two rows or is not binary.
    """
    matrix = tanimoto_matrix(x_reference, x_reference)
    if matrix.shape[0] < 2:
        raise ValueError("need at least 2 rows for self-similarity")
    np.fill_diagonal(matrix, -1.0)
    return matrix.max(axis=1).astype(float)


def domain_threshold(
    x_reference: np.ndarray, percentile: float = DOMAIN_PERCENTILE
) -> float:
    """Derive an in-domain similarity threshold from the training set itself.

    Args:
        x_reference: ``(n, n_bits)`` binary training matrix.
        percentile: Percentile of the training set's own nearest-neighbour
            similarities to use as the threshold.

    Returns:
        The similarity threshold.

    Raises:
        ValueError: If ``percentile`` is outside [0, 100].
    """
    if not 0.0 <= percentile <= 100.0:
        raise ValueError(f"percentile must be in [0, 100], got {percentile}")
    return float(np.percentile(self_neighbour_similarity(x_reference), percentile))


@dataclass(slots=True)
class SimilarityBin:
    """Error statistics for one nearest-neighbour similarity bin.

    Attributes:
        lower: Inclusive lower edge of the bin.
        upper: Exclusive upper edge (inclusive for the last bin).
        n: Compounds falling in the bin.
        mean_similarity: Mean nearest-neighbour similarity in the bin.
        mean_absolute_error: Mean absolute error in the bin.
        rmse: Root-mean-square error in the bin.
    """

    lower: float
    upper: float
    n: int
    mean_similarity: float
    mean_absolute_error: float
    rmse: float


def error_vs_similarity(
    similarity: Sequence[float],
    errors: Sequence[float],
    bins: Sequence[float] = SIMILARITY_BINS,
) -> list[SimilarityBin]:
    """Bin prediction errors by nearest-neighbour similarity.

    Args:
        similarity: Nearest-neighbour Tanimoto similarity per compound.
        errors: Signed or absolute prediction error per compound; the
            absolute value is taken internally.
        bins: Monotonically increasing bin edges.

    Returns:
        One :class:`SimilarityBin` per occupied bin, in increasing order of
        similarity.

    Raises:
        ValueError: If the inputs are mismatched or the edges are not
            increasing.
    """
    sim = np.asarray(similarity, dtype=float).ravel()
    err = np.abs(np.asarray(errors, dtype=float).ravel())
    if sim.size != err.size:
        raise ValueError("similarity and errors must have the same length")
    edges = np.asarray(bins, dtype=float)
    if edges.size < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError("bins must be strictly increasing with at least 2 edges")

    out: list[SimilarityBin] = []
    for index in range(edges.size - 1):
        lower, upper = float(edges[index]), float(edges[index + 1])
        last = index == edges.size - 2
        mask = (sim >= lower) & ((sim <= upper) if last else (sim < upper))
        if not mask.any():
            continue
        out.append(
            SimilarityBin(
                lower=lower,
                upper=upper,
                n=int(mask.sum()),
                mean_similarity=float(sim[mask].mean()),
                mean_absolute_error=float(err[mask].mean()),
                rmse=float(np.sqrt(np.mean(err[mask] ** 2))),
            )
        )
    return out


def coverage_curve(
    similarity: Sequence[float],
    y_true: Sequence[float],
    y_pred: Sequence[float],
    task: str = "regression",
    n_steps: int = 20,
) -> list[dict[str, float]]:
    """Trace the coverage/accuracy trade-off over similarity thresholds.

    At each threshold, predictions for compounds below it are refused and
    the metric is recomputed on what remains.  A model with a working
    applicability domain gets monotonically better as the threshold rises,
    and the curve shows what that costs in coverage.

    Args:
        similarity: Nearest-neighbour similarity per test compound.
        y_true: Observed pChEMBL values (regression) or 0/1 labels
            (classification).
        y_pred: Predicted values or active-class probabilities.
        task: ``"regression"`` or ``"classification"``.
        n_steps: Number of thresholds, taken as quantiles of the observed
            similarity distribution so each step drops a similar number of
            compounds.

    Returns:
        A list of dictionaries with ``threshold``, ``coverage``, ``n`` and
        the task's metrics (``rmse``/``mae`` or
        ``roc_auc``/``balanced_accuracy``).

    Raises:
        ValueError: On mismatched inputs or an unknown task.
    """
    if task not in ("regression", "classification"):
        raise ValueError(f"unknown task {task!r}")
    sim = np.asarray(similarity, dtype=float).ravel()
    truth = np.asarray(y_true, dtype=float).ravel()
    pred = np.asarray(y_pred, dtype=float).ravel()
    if not sim.size == truth.size == pred.size:
        raise ValueError("similarity, y_true and y_pred must have the same length")
    if n_steps < 2:
        raise ValueError(f"n_steps must be at least 2, got {n_steps}")

    quantiles = np.linspace(0.0, 0.9, n_steps)
    thresholds = np.unique(np.quantile(sim, quantiles))
    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        mask = sim >= threshold
        n_kept = int(mask.sum())
        if n_kept < 20:
            continue
        row = {
            "threshold": float(threshold),
            "n": float(n_kept),
            "coverage": n_kept / sim.size,
        }
        if task == "regression":
            residual = pred[mask] - truth[mask]
            row["rmse"] = float(np.sqrt(np.mean(residual**2)))
            row["mae"] = float(np.mean(np.abs(residual)))
        else:
            labels = truth[mask].astype(int)
            if labels.min() == labels.max():
                continue
            row["roc_auc"] = float(roc_auc_score(labels, pred[mask]))
            row["balanced_accuracy"] = float(
                balanced_accuracy_score(labels, (pred[mask] >= 0.5).astype(int))
            )
        rows.append(row)
    return rows


def spearman_trend(x: Sequence[float], y: Sequence[float]) -> float:
    """Rank correlation between two sequences, for monotonicity checks.

    Args:
        x: First sequence.
        y: Second sequence.

    Returns:
        Spearman's rho, or ``nan`` if either sequence is constant or shorter
        than three elements.
    """
    from scipy import stats

    first = np.asarray(x, dtype=float).ravel()
    second = np.asarray(y, dtype=float).ravel()
    if first.size < 3 or np.std(first) == 0 or np.std(second) == 0:
        return float("nan")
    return float(stats.spearmanr(first, second).statistic)
