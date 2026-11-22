"""Tests for inductive conformal prediction.

The whole value of conformal prediction is that the coverage guarantee is
*proved*, not hoped for. A bug here would produce a confident-looking
number with nothing behind it, so the central test fits no model at all:
it feeds the machinery random, uninformative scores and checks that
coverage still lands on target, which is the property that distinguishes
conformal prediction from ordinary thresholding.
"""

from __future__ import annotations

import numpy as np
import pytest

from ligandml import conformal


class TestNonconformity:
    """One minus the probability given to the true class."""

    def test_hand_computed(self) -> None:
        probs = np.array([[0.9, 0.1], [0.3, 0.7]])
        scores = conformal.nonconformity(probs, [0, 1])
        assert scores == pytest.approx([0.1, 0.3])

    def test_wrong_label_scores_high(self) -> None:
        scores = conformal.nonconformity(np.array([[0.9, 0.1]]), [1])
        assert scores == pytest.approx([0.9])

    def test_accepts_positive_class_vector(self) -> None:
        flat = conformal.nonconformity(np.array([0.1, 0.7]), [0, 1])
        matrix = conformal.nonconformity(
            np.array([[0.9, 0.1], [0.3, 0.7]]), [0, 1]
        )
        assert flat == pytest.approx(matrix)

    def test_rejects_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="probabilities for"):
            conformal.nonconformity(np.array([[0.9, 0.1]]), [0, 1])

    def test_rejects_non_binary_labels(self) -> None:
        with pytest.raises(ValueError, match="must be 0 or 1"):
            conformal.nonconformity(np.array([[0.9, 0.1]]), [2])


class TestPValues:
    """p = (number at least as strange + 1) / (calibration size + 1)."""

    def test_hand_computed_p_value(self) -> None:
        """Calibration class 1 scores 0.1, 0.2, 0.3, 0.4.

        A test compound with p(active) = 0.75 has nonconformity 0.25 under
        the active hypothesis. Two calibration scores (0.3, 0.4) are at
        least that strange, so p = (2 + 1) / (4 + 1) = 0.6.
        """
        calibration = conformal.ConformalCalibration(
            {1: np.array([0.1, 0.2, 0.3, 0.4]), 0: np.array([0.5])}
        )
        p = calibration.p_values(np.array([[0.25, 0.75]]))
        assert p[0, 1] == pytest.approx(0.6)

    def test_p_value_never_zero(self) -> None:
        """The +1 smoothing keeps the guarantee valid at finite sizes."""
        calibration = conformal.ConformalCalibration({1: np.array([0.0]), 0: np.array([0.0])})
        p = calibration.p_values(np.array([[0.0, 1.0]]))
        assert (p > 0).all()

    def test_class_with_no_calibration_is_never_predicted(self) -> None:
        calibration = conformal.ConformalCalibration({0: np.array([0.1, 0.2])})
        p = calibration.p_values(np.array([[0.5, 0.5]]))
        assert p[0, 1] == 0.0

    def test_calibrate_splits_scores_by_class(self) -> None:
        probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8]])
        calibration = conformal.calibrate(probs, [0, 0, 1])
        assert len(calibration.scores_by_class[0]) == 2
        assert len(calibration.scores_by_class[1]) == 1
        assert calibration.n_calibration == 3

    def test_calibrate_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            conformal.calibrate(np.empty((0, 2)), [])


class TestPredictionSets:
    """A label is in the set when its p-value survives the threshold."""

    def test_threshold(self) -> None:
        sets = conformal.prediction_sets(np.array([[0.3, 0.02]]), 0.05)
        assert list(sets[0]) == [True, False]

    def test_both_and_empty_are_possible(self) -> None:
        sets = conformal.prediction_sets(np.array([[0.9, 0.9], [0.01, 0.01]]), 0.1)
        assert list(sets[0]) == [True, True]
        assert list(sets[1]) == [False, False]

    def test_rejects_significance_outside_the_unit_interval(self) -> None:
        with pytest.raises(ValueError, match="must be in"):
            conformal.prediction_sets(np.array([[0.5, 0.5]]), 1.0)


class TestEvaluateSets:
    """Validity and efficiency, counted by hand."""

    def test_hand_counted_report(self) -> None:
        """Four compounds: a correct singleton, a wrong singleton, a both
        and an empty. Coverage counts the first and the third."""
        sets = np.array(
            [[False, True], [False, True], [True, True], [False, False]]
        )
        report = conformal.evaluate_sets(sets, [1, 0, 1, 0], significance=0.1)
        assert report.n == 4
        assert report.coverage == pytest.approx(0.5)
        assert report.singleton_rate == pytest.approx(0.5)
        assert report.both_rate == pytest.approx(0.25)
        assert report.empty_rate == pytest.approx(0.25)
        # One of the two singletons was right.
        assert report.singleton_accuracy == pytest.approx(0.5)

    def test_coverage_is_reported_per_class(self) -> None:
        sets = np.array([[True, False], [True, False]])
        report = conformal.evaluate_sets(sets, [0, 1], significance=0.1)
        assert report.coverage_by_class[0] == pytest.approx(1.0)
        assert report.coverage_by_class[1] == pytest.approx(0.0)

    def test_singleton_accuracy_is_nan_without_singletons(self) -> None:
        report = conformal.evaluate_sets(
            np.array([[True, True]]), [1], significance=0.1
        )
        assert np.isnan(report.singleton_accuracy)

    def test_rejects_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="sets for"):
            conformal.evaluate_sets(np.array([[True, False]]), [0, 1], 0.1)

    def test_rejects_empty_test_set(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            conformal.evaluate_sets(np.empty((0, 2), dtype=bool), [], 0.1)


class TestGuarantee:
    """The property that makes conformal prediction worth using."""

    @pytest.mark.parametrize("significance", [0.05, 0.1, 0.2])
    def test_coverage_holds_for_a_useless_model(self, significance: float) -> None:
        """A model with no signal must still be valid.

        This is the point of the method: validity comes from the
        calibration step, not from the model being any good. A predictor
        that cannot tell the classes apart pays for it in *efficiency* -
        it returns both labels constantly - but coverage still lands on
        target. If this test fails the guarantee is not implemented.
        """
        rng = np.random.default_rng(0)
        n = 4000
        labels = rng.integers(0, 2, size=n)
        noise = rng.uniform(0.2, 0.8, size=n)  # unrelated to the labels
        probs = np.column_stack([1.0 - noise, noise])
        half = n // 2
        reports = conformal.run(
            probs[:half], labels[:half], probs[half:], labels[half:],
            significances=[significance],
        )
        assert reports[0].coverage >= (1.0 - significance) - 0.02
        assert reports[0].is_valid

    def test_coverage_holds_per_class_under_imbalance(self) -> None:
        """Mondrian calibration protects the minority class.

        With 5% actives, an overall-valid predictor can still be hopeless
        on the actives. Class-conditional calibration is what stops that,
        and the actives are the class a screen cares about.
        """
        rng = np.random.default_rng(1)
        n = 6000
        labels = (rng.uniform(size=n) < 0.05).astype(int)
        score = rng.uniform(0.1, 0.9, size=n)
        probs = np.column_stack([1.0 - score, score])
        half = n // 2
        reports = conformal.run(
            probs[:half], labels[:half], probs[half:], labels[half:],
            significances=[0.1],
        )
        assert reports[0].coverage_by_class[1] >= 0.85
        assert reports[0].coverage_by_class[0] >= 0.85

    def test_a_useful_model_buys_efficiency_not_validity(self) -> None:
        """A good model returns singletons; a bad one returns both labels.

        Both are valid. The difference between them is efficiency, which
        is why the report carries it beside coverage.
        """
        rng = np.random.default_rng(2)
        n = 4000
        labels = rng.integers(0, 2, size=n)
        confident = np.where(
            labels == 1,
            rng.uniform(0.85, 0.99, size=n),
            rng.uniform(0.01, 0.15, size=n),
        )
        probs = np.column_stack([1.0 - confident, confident])
        half = n // 2
        good = conformal.run(
            probs[:half], labels[:half], probs[half:], labels[half:],
            significances=[0.1],
        )[0]
        useless = rng.uniform(0.2, 0.8, size=n)
        flat = np.column_stack([1.0 - useless, useless])
        bad = conformal.run(
            flat[:half], labels[:half], flat[half:], labels[half:],
            significances=[0.1],
        )[0]
        assert good.is_valid and bad.is_valid
        assert good.singleton_rate > 0.9
        assert bad.both_rate > good.both_rate

    def test_report_row_is_flat(self) -> None:
        report = conformal.evaluate_sets(
            np.array([[False, True]]), [1], significance=0.1
        )
        row = report.as_row()
        assert row["coverage"] == 1.0
        assert all(not isinstance(v, (list, dict, np.ndarray)) for v in row.values())
