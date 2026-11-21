"""Tests for the applicability domain.

The claim under test is monotonicity: as nearest-neighbour similarity to
the training set falls, prediction error rises. The synthetic case below
constructs that relationship exactly, so a failure means the measurement
machinery is broken rather than that the chemistry is noisy.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ligandml import domain


def _bits(rows: list[list[int]]) -> np.ndarray:
    """Build a small binary matrix from literal bit rows."""
    return np.asarray(rows, dtype=float)


class TestTanimoto:
    """Tanimoto is intersection over union."""

    def test_identical_vectors_are_one(self) -> None:
        vector = _bits([[1, 1, 0, 1]])
        assert domain.tanimoto_matrix(vector, vector)[0, 0] == pytest.approx(1.0)

    def test_disjoint_vectors_are_zero(self) -> None:
        first = _bits([[1, 1, 0, 0]])
        second = _bits([[0, 0, 1, 1]])
        assert domain.tanimoto_matrix(first, second)[0, 0] == pytest.approx(0.0)

    def test_hand_computed(self) -> None:
        """{1,2,3} vs {2,3,4}: intersection 2, union 4, so Tanimoto 0.5."""
        first = _bits([[1, 1, 1, 0]])
        second = _bits([[0, 1, 1, 1]])
        assert domain.tanimoto_matrix(first, second)[0, 0] == pytest.approx(0.5)

    def test_hand_computed_asymmetric_sizes(self) -> None:
        """{1} vs {1,2,3}: intersection 1, union 3, so Tanimoto 1/3."""
        first = _bits([[1, 0, 0]])
        second = _bits([[1, 1, 1]])
        assert domain.tanimoto_matrix(first, second)[0, 0] == pytest.approx(1 / 3)

    def test_symmetric(self) -> None:
        rng = np.random.default_rng(0)
        left = (rng.random((7, 64)) < 0.2).astype(float)
        right = (rng.random((5, 64)) < 0.2).astype(float)
        forward = domain.tanimoto_matrix(left, right)
        backward = domain.tanimoto_matrix(right, left)
        assert np.allclose(forward, backward.T)

    def test_all_zero_pair_is_zero_not_nan(self) -> None:
        zeros = _bits([[0, 0, 0]])
        value = domain.tanimoto_matrix(zeros, zeros)[0, 0]
        assert value == 0.0 and not math.isnan(value)

    def test_bounded(self) -> None:
        rng = np.random.default_rng(1)
        matrix = (rng.random((40, 128)) < 0.1).astype(float)
        values = domain.tanimoto_matrix(matrix, matrix)
        assert values.min() >= 0.0 and values.max() <= 1.0

    def test_chunking_is_transparent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The chunked implementation must match an unchunked one exactly."""
        rng = np.random.default_rng(2)
        query = (rng.random((100, 64)) < 0.2).astype(float)
        reference = (rng.random((30, 64)) < 0.2).astype(float)
        full = domain.tanimoto_matrix(query, reference)
        monkeypatch.setattr(domain, "CHUNK_ROWS", 7)
        chunked = domain.tanimoto_matrix(query, reference)
        assert np.allclose(full, chunked)

    def test_non_binary_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be binary"):
            domain.tanimoto_matrix(_bits([[0.5, 1.0]]), _bits([[1.0, 1.0]]))

    def test_width_mismatch_rejected(self) -> None:
        with pytest.raises(ValueError, match="bit width mismatch"):
            domain.tanimoto_matrix(_bits([[1, 0]]), _bits([[1, 0, 1]]))

    def test_empty_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            domain.tanimoto_matrix(np.zeros((0, 4)), _bits([[1, 0, 0, 0]]))


class TestNearestNeighbourSimilarity:
    """Each query's similarity to its closest reference compound."""

    def test_exact_duplicate_scores_one(self) -> None:
        reference = _bits([[1, 1, 0, 0], [0, 0, 1, 1]])
        query = _bits([[1, 1, 0, 0]])
        assert domain.nearest_neighbour_similarity(query, reference)[0] == pytest.approx(
            1.0
        )

    def test_takes_the_maximum(self) -> None:
        reference = _bits([[1, 0, 0, 0], [1, 1, 1, 0]])
        query = _bits([[1, 1, 0, 0]])
        # vs row 0: 1/2 = 0.5; vs row 1: 2/3 = 0.667. Maximum is 0.667.
        assert domain.nearest_neighbour_similarity(query, reference)[0] == pytest.approx(
            2 / 3
        )

    def test_shape(self) -> None:
        rng = np.random.default_rng(3)
        query = (rng.random((17, 32)) < 0.2).astype(float)
        reference = (rng.random((9, 32)) < 0.2).astype(float)
        assert domain.nearest_neighbour_similarity(query, reference).shape == (17,)

    def test_self_similarity_excludes_self(self) -> None:
        matrix = _bits([[1, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 1]])
        values = domain.self_neighbour_similarity(matrix)
        assert values[0] == pytest.approx(0.5)  # vs row 1, not vs itself
        assert values.max() < 1.0

    def test_self_similarity_needs_two_rows(self) -> None:
        with pytest.raises(ValueError, match="at least 2 rows"):
            domain.self_neighbour_similarity(_bits([[1, 0]]))


class TestDomainThreshold:
    """The threshold derived from the training set's own density."""

    def test_in_range(self) -> None:
        rng = np.random.default_rng(4)
        matrix = (rng.random((80, 64)) < 0.2).astype(float)
        threshold = domain.domain_threshold(matrix)
        assert 0.0 <= threshold <= 1.0

    def test_tighter_set_gives_higher_threshold(self) -> None:
        """A training set of near-duplicates must demand more similarity than
        a diverse one before a prediction is trusted.
        """
        rng = np.random.default_rng(5)
        base = (rng.random((1, 128)) < 0.2).astype(float)
        tight = np.repeat(base, 60, axis=0)
        flip = rng.random((60, 128)) < 0.02
        tight = np.abs(tight - flip.astype(float))
        diverse = (rng.random((60, 128)) < 0.2).astype(float)
        assert domain.domain_threshold(tight) > domain.domain_threshold(diverse)

    @pytest.mark.parametrize("percentile", [-1.0, 101.0])
    def test_bad_percentile(self, percentile: float) -> None:
        rng = np.random.default_rng(6)
        with pytest.raises(ValueError, match="percentile"):
            domain.domain_threshold(
                (rng.random((10, 16)) < 0.2).astype(float), percentile=percentile
            )


class TestErrorVsSimilarity:
    """Binning errors by similarity."""

    def test_synthetic_monotonicity(self) -> None:
        """Build errors that are exactly linear in (1 - similarity): the
        binned mean absolute error must then fall monotonically as
        similarity rises.
        """
        rng = np.random.default_rng(7)
        similarity = rng.uniform(0.05, 0.95, 5000)
        errors = 2.0 * (1.0 - similarity)
        bins = domain.error_vs_similarity(similarity, errors)
        maes = [b.mean_absolute_error for b in bins]
        assert len(maes) >= 5
        assert all(
            earlier > later for earlier, later in zip(maes, maes[1:], strict=False)
        ), maes

    def test_noisy_synthetic_monotonicity(self) -> None:
        """The trend must survive noise of the same scale as the signal."""
        rng = np.random.default_rng(8)
        similarity = rng.uniform(0.05, 0.95, 20_000)
        errors = 2.0 * (1.0 - similarity) + rng.normal(0.0, 0.5, 20_000)
        bins = domain.error_vs_similarity(similarity, errors)
        rho = domain.spearman_trend(
            [b.mean_similarity for b in bins],
            [b.mean_absolute_error for b in bins],
        )
        assert rho < -0.9

    def test_flat_errors_give_flat_bins(self) -> None:
        """A control: if error does not depend on similarity, the curve must
        be flat, so the test above cannot be passing by construction.
        """
        rng = np.random.default_rng(9)
        similarity = rng.uniform(0.05, 0.95, 20_000)
        errors = np.full(20_000, 1.0)
        bins = domain.error_vs_similarity(similarity, errors)
        maes = np.array([b.mean_absolute_error for b in bins])
        assert np.allclose(maes, 1.0)

    def test_counts_sum_to_n(self) -> None:
        rng = np.random.default_rng(10)
        similarity = rng.uniform(0.0, 1.0, 500)
        bins = domain.error_vs_similarity(similarity, rng.normal(size=500))
        assert sum(b.n for b in bins) == 500

    def test_bins_are_ordered_and_contiguous(self) -> None:
        rng = np.random.default_rng(11)
        bins = domain.error_vs_similarity(
            rng.uniform(0.0, 1.0, 2000), rng.normal(size=2000)
        )
        for earlier, later in zip(bins, bins[1:], strict=False):
            assert earlier.upper <= later.lower
            assert earlier.mean_similarity < later.mean_similarity

    def test_absolute_value_taken(self) -> None:
        bins = domain.error_vs_similarity([0.5, 0.5], [-2.0, 2.0], bins=[0.0, 1.0])
        assert bins[0].mean_absolute_error == pytest.approx(2.0)

    def test_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            domain.error_vs_similarity([0.5], [1.0, 2.0])

    def test_bad_bins(self) -> None:
        with pytest.raises(ValueError, match="strictly increasing"):
            domain.error_vs_similarity([0.5], [1.0], bins=[1.0, 0.0])


class TestCoverageCurve:
    """The coverage/accuracy trade-off."""

    def test_accuracy_improves_as_coverage_shrinks(self) -> None:
        """Synthetic case: error is inversely proportional to similarity, so
        refusing the least similar compounds must lower the RMSE.
        """
        rng = np.random.default_rng(12)
        n = 4000
        similarity = rng.uniform(0.05, 0.95, n)
        truth = rng.normal(6.0, 1.0, n)
        predicted = truth + rng.normal(0.0, 1.0, n) * (1.0 - similarity) * 2.0
        rows = domain.coverage_curve(similarity, truth, predicted, task="regression")
        assert len(rows) > 5
        assert rows[0]["coverage"] == pytest.approx(1.0)
        assert rows[-1]["rmse"] < rows[0]["rmse"]
        rho = domain.spearman_trend(
            [r["coverage"] for r in rows], [r["rmse"] for r in rows]
        )
        assert rho > 0.9

    def test_coverage_is_monotone_decreasing(self) -> None:
        rng = np.random.default_rng(13)
        n = 2000
        similarity = rng.uniform(0.0, 1.0, n)
        rows = domain.coverage_curve(
            similarity, rng.normal(size=n), rng.normal(size=n), task="regression"
        )
        coverages = [r["coverage"] for r in rows]
        assert all(
            earlier >= later
            for earlier, later in zip(coverages, coverages[1:], strict=False)
        )

    def test_classification_task(self) -> None:
        rng = np.random.default_rng(14)
        n = 3000
        similarity = rng.uniform(0.05, 0.95, n)
        labels = rng.integers(0, 2, n)
        noise = rng.random(n) * (1.0 - similarity) * 2.0
        probs = np.clip(labels * 0.6 + 0.2 + noise - 0.3, 0.0, 1.0)
        rows = domain.coverage_curve(
            similarity, labels, probs, task="classification"
        )
        assert rows
        assert all("roc_auc" in r for r in rows)
        assert rows[-1]["roc_auc"] > rows[0]["roc_auc"]

    def test_unknown_task(self) -> None:
        with pytest.raises(ValueError, match="unknown task"):
            domain.coverage_curve([0.5] * 30, [1.0] * 30, [1.0] * 30, task="ranking")

    def test_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            domain.coverage_curve([0.5], [1.0, 2.0], [1.0, 2.0])


class TestSpearmanTrend:
    """The small helper the monotonicity checks lean on."""

    def test_perfect_increase(self) -> None:
        assert domain.spearman_trend([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(
            1.0
        )

    def test_perfect_decrease(self) -> None:
        assert domain.spearman_trend([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(
            -1.0
        )

    def test_constant_is_nan(self) -> None:
        assert math.isnan(domain.spearman_trend([1, 1, 1], [1, 2, 3]))

    def test_too_short_is_nan(self) -> None:
        assert math.isnan(domain.spearman_trend([1, 2], [1, 2]))
