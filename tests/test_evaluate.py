"""Tests for the metrics, pinned to hand-computed answers.

Enrichment factor and BEDROC are easy to implement subtly wrongly - an
off-by-one in the rank, a ceiling computed from the wrong denominator - and
a wrong screening metric would overstate the only result a screening
project cares about. Every metric here is checked against a value worked
out on paper, not against a reference implementation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ligandml import evaluate


class TestEnrichmentFactor:
    """EF_f = (hit rate in the top k) / (hit rate overall), k = round(f*N)."""

    def test_hand_computed_perfect_ranking(self) -> None:
        """20 compounds, 5 actives, top 25% = 5 compounds, all 5 active.

        Hit rate in the top slice is 5/5 = 1.0; overall is 5/20 = 0.25;
        so EF = 1.0 / 0.25 = 4.0, which is also the ceiling 1/f.
        """
        labels = [1] * 5 + [0] * 15
        scores = list(range(20, 0, -1))
        assert evaluate.enrichment_factor(labels, scores, 0.25) == pytest.approx(4.0)

    def test_hand_computed_partial_ranking(self) -> None:
        """20 compounds, 5 actives, top 25% = 5 compounds, 3 of them active.

        Hit rate in the top slice is 3/5 = 0.6; overall is 0.25;
        so EF = 0.6 / 0.25 = 2.4.
        """
        labels = [1, 1, 0, 1, 0, 0, 0, 0, 1, 1] + [0] * 10
        scores = list(range(20, 0, -1))
        assert evaluate.enrichment_factor(labels, scores, 0.25) == pytest.approx(2.4)

    def test_hand_computed_one_percent(self) -> None:
        """100 compounds, 10 actives, top 1% = 1 compound, and it is active.

        EF = (1/1) / (10/100) = 10.0, which is also the ceiling 1/f divided
        by ten because only one of the ten actives can occupy that slot.
        """
        labels = [1] + [0] * 90 + [1] * 9
        scores = list(range(100, 0, -1))
        assert sum(labels) == 10
        assert evaluate.enrichment_factor(labels, scores, 0.01) == pytest.approx(10.0)

    def test_random_ranking_is_about_one(self) -> None:
        rng = np.random.default_rng(0)
        labels = rng.integers(0, 2, 4000)
        scores = rng.random(4000)
        assert evaluate.enrichment_factor(labels, scores, 0.1) == pytest.approx(
            1.0, abs=0.25
        )

    def test_worst_ranking_is_zero(self) -> None:
        labels = [0] * 90 + [1] * 10
        scores = list(range(100, 0, -1))
        assert evaluate.enrichment_factor(labels, scores, 0.1) == 0.0

    def test_ceiling_is_inverse_fraction(self) -> None:
        """EF cannot exceed 1/f, whatever the model does."""
        labels = [1] * 50 + [0] * 50
        scores = list(range(100, 0, -1))
        assert evaluate.enrichment_factor(labels, scores, 0.01) == pytest.approx(2.0)
        assert evaluate.enrichment_factor(labels, scores, 0.5) == pytest.approx(2.0)

    def test_reported_ceiling_accounts_for_the_active_rate(self) -> None:
        """On a mostly-active set the ceiling is 1/active_rate, not 1/f, so a
        small EF there is a property of the data and not of the model.
        """
        dense = evaluate.classification_metrics(
            [1] * 78 + [0] * 22, list(np.linspace(1.0, 0.0, 100))
        )
        assert dense["ef_ceiling_1pct"] == pytest.approx(100 / 78, rel=1e-6)
        assert dense["ef_1pct"] <= dense["ef_ceiling_1pct"] + 1e-9

        sparse = evaluate.classification_metrics(
            [1] * 2 + [0] * 998, list(np.linspace(1.0, 0.0, 1000))
        )
        assert sparse["ef_ceiling_1pct"] == pytest.approx(100.0)

    def test_full_list_is_exactly_one(self) -> None:
        labels = [1] * 7 + [0] * 13
        scores = list(range(20, 0, -1))
        assert evaluate.enrichment_factor(labels, scores, 1.0) == pytest.approx(1.0)

    def test_ties_broken_pessimistically(self) -> None:
        """A model that scores everything identically must not be flattered."""
        labels = [1] * 10 + [0] * 90
        scores = [0.5] * 100
        assert evaluate.enrichment_factor(labels, scores, 0.1) == 0.0

    def test_no_actives_is_nan(self) -> None:
        assert math.isnan(evaluate.enrichment_factor([0] * 10, range(10), 0.1))

    @pytest.mark.parametrize("fraction", [0.0, -0.1, 1.1])
    def test_bad_fraction(self, fraction: float) -> None:
        with pytest.raises(ValueError, match="fraction"):
            evaluate.enrichment_factor([1, 0], [1.0, 0.0], fraction)

    def test_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="entries"):
            evaluate.enrichment_factor([1, 0, 1], [1.0, 0.0], 0.5)


class TestHitRateAndRecall:
    """The two numbers a screening demo should quote."""

    def test_hit_rate_hand_computed(self) -> None:
        """Top 25% of 20 compounds is 5; 3 are active, so the hit rate is 0.6."""
        labels = [1, 1, 0, 1, 0] + [0] * 10 + [1, 1, 0, 0, 0]
        scores = list(range(20, 0, -1))
        assert evaluate.hit_rate(labels, scores, 0.25) == pytest.approx(0.6)

    def test_recall_hand_computed(self) -> None:
        """Of 5 actives, 3 sit in the top 25%, so recall is 0.6."""
        labels = [1, 1, 0, 1, 0] + [0] * 10 + [1, 1, 0, 0, 0]
        scores = list(range(20, 0, -1))
        assert evaluate.recall_at_fraction(labels, scores, 0.25) == pytest.approx(0.6)

    def test_recall_at_full_list_is_one(self) -> None:
        labels = [1, 0, 1, 0, 1]
        scores = [0.9, 0.1, 0.8, 0.2, 0.7]
        assert evaluate.recall_at_fraction(labels, scores, 1.0) == pytest.approx(1.0)

    def test_recall_relates_to_enrichment(self) -> None:
        """EF_f = recall_f / f exactly, when k = f*N is an integer."""
        rng = np.random.default_rng(3)
        labels = rng.integers(0, 2, 1000)
        scores = rng.random(1000)
        for fraction in (0.01, 0.05, 0.1):
            ef = evaluate.enrichment_factor(labels, scores, fraction)
            recall = evaluate.recall_at_fraction(labels, scores, fraction)
            assert ef == pytest.approx(recall / fraction, rel=1e-9)


class TestBedroc:
    """BEDROC is bounded in [0, 1], with 0.5 for a random ranking."""

    def test_perfect_ranking_is_one(self) -> None:
        labels = [1] * 20 + [0] * 180
        scores = list(range(200, 0, -1))
        assert evaluate.bedroc(labels, scores, alpha=20.0) == pytest.approx(
            1.0, abs=1e-6
        )

    def test_worst_ranking_is_zero(self) -> None:
        labels = [0] * 180 + [1] * 20
        scores = list(range(200, 0, -1))
        assert evaluate.bedroc(labels, scores, alpha=20.0) == pytest.approx(
            0.0, abs=1e-6
        )

    def test_random_ranking_matches_its_own_expectation(self) -> None:
        """BEDROC's random baseline is NOT 0.5: it depends on alpha and the
        active ratio. A random ranking must land on the value
        :func:`bedroc_random_expectation` predicts, not on 0.5.
        """
        rng = np.random.default_rng(1)
        labels = np.zeros(5000, dtype=int)
        labels[rng.choice(5000, 500, replace=False)] = 1
        scores = rng.random(5000)
        expected = evaluate.bedroc_random_expectation(500, 5000, alpha=20.0)
        assert expected == pytest.approx(0.1157, abs=0.001)
        assert expected < 0.3, "the 0.5-is-random intuition does not transfer"
        assert evaluate.bedroc(labels, scores, alpha=20.0) == pytest.approx(
            expected, abs=0.03
        )

    def test_bounded(self) -> None:
        rng = np.random.default_rng(2)
        for _ in range(20):
            labels = rng.integers(0, 2, 300)
            if labels.min() == labels.max():
                continue
            scores = rng.random(300)
            for alpha in (20.0, 32.2, 160.9):
                value = evaluate.bedroc(labels, scores, alpha=alpha)
                assert 0.0 <= value <= 1.0

    def test_random_expectation_rises_with_the_active_ratio(self) -> None:
        """A denser active set is easier to hit by chance, so the baseline
        a reported BEDROC must beat is higher.
        """
        baselines = [
            evaluate.bedroc_random_expectation(actives, 1000, alpha=20.0)
            for actives in (10, 50, 200, 500)
        ]
        assert all(
            earlier < later
            for earlier, later in zip(baselines, baselines[1:], strict=False)
        ), baselines

    def test_random_expectation_edge_cases(self) -> None:
        assert math.isnan(evaluate.bedroc_random_expectation(0, 100))
        assert math.isnan(evaluate.bedroc_random_expectation(100, 100))
        with pytest.raises(ValueError, match="alpha"):
            evaluate.bedroc_random_expectation(10, 100, alpha=-1.0)
        with pytest.raises(ValueError, match="n_total"):
            evaluate.bedroc_random_expectation(10, 0)

    def test_perfect_and_worst_bracket_the_random_baseline(self) -> None:
        labels = [1] * 50 + [0] * 450
        best = list(range(500, 0, -1))
        baseline = evaluate.bedroc_random_expectation(50, 500, alpha=20.0)
        assert evaluate.bedroc(labels, best, alpha=20.0) > baseline
        assert evaluate.bedroc(labels[::-1], best, alpha=20.0) < baseline

    def test_rewards_early_recognition(self) -> None:
        """Two rankings with equal ROC AUC but different early behaviour must
        get different BEDROC scores, which is the whole point of the metric.
        """
        early = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0] * 10
        late = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1] * 10
        scores = list(range(100, 0, -1))
        assert evaluate.bedroc(early, scores, alpha=20.0) > evaluate.bedroc(
            late, scores, alpha=20.0
        )

    def test_single_class_is_nan(self) -> None:
        assert math.isnan(evaluate.bedroc([1] * 10, range(10)))
        assert math.isnan(evaluate.bedroc([0] * 10, range(10)))

    def test_bad_alpha(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            evaluate.bedroc([1, 0], [1.0, 0.0], alpha=0.0)


class TestBedrocAlpha:
    """alpha = ln(5)/f places 80% of the weight in the top fraction f."""

    @pytest.mark.parametrize(
        ("fraction", "expected"), [(0.01, 160.94), (0.05, 32.19), (0.08, 20.12)]
    )
    def test_known_alphas(self, fraction: float, expected: float) -> None:
        assert evaluate.bedroc_alpha(fraction) == pytest.approx(expected, abs=0.01)

    def test_weight_mass_is_eighty_percent(self) -> None:
        """Verify the derivation: 1 - exp(-alpha*f) should be 0.8."""
        for fraction in (0.01, 0.05, 0.1):
            alpha = evaluate.bedroc_alpha(fraction)
            assert 1.0 - math.exp(-alpha * fraction) == pytest.approx(0.8)

    @pytest.mark.parametrize("fraction", [0.0, 1.0, -1.0])
    def test_bad_fraction(self, fraction: float) -> None:
        with pytest.raises(ValueError, match="fraction"):
            evaluate.bedroc_alpha(fraction)


class TestRegressionMetrics:
    """Regression metrics against closed-form answers."""

    def test_perfect_prediction(self) -> None:
        truth = [5.0, 6.0, 7.0, 8.0]
        metrics = evaluate.regression_metrics(truth, truth)
        assert metrics["r2"] == pytest.approx(1.0)
        assert metrics["rmse"] == pytest.approx(0.0)
        assert metrics["mae"] == pytest.approx(0.0)
        assert metrics["spearman_rho"] == pytest.approx(1.0)

    def test_mean_prediction_is_zero_r2(self) -> None:
        truth = np.array([4.0, 5.0, 6.0, 7.0, 8.0])
        metrics = evaluate.regression_metrics(truth, np.full(5, truth.mean()))
        assert metrics["r2"] == pytest.approx(0.0)

    def test_hand_computed_rmse_and_mae(self) -> None:
        """Errors of +1, -1, +2, -2: MAE 1.5, RMSE sqrt(10/4) = 1.5811."""
        truth = [0.0, 0.0, 0.0, 0.0]
        predicted = [1.0, -1.0, 2.0, -2.0]
        metrics = evaluate.regression_metrics(truth, predicted)
        assert metrics["mae"] == pytest.approx(1.5)
        assert metrics["rmse"] == pytest.approx(math.sqrt(2.5))

    def test_worse_than_mean_is_negative_r2(self) -> None:
        metrics = evaluate.regression_metrics([1.0, 2.0, 3.0], [3.0, 2.0, 1.0])
        assert metrics["r2"] < 0.0

    def test_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            evaluate.regression_metrics([1.0, 2.0], [1.0])

    def test_empty(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            evaluate.regression_metrics([], [])


class TestClassificationMetrics:
    """Classification and screening metrics bundled together."""

    def test_perfect_classifier(self) -> None:
        labels = [1] * 10 + [0] * 90
        probs = [0.99] * 10 + [0.01] * 90
        metrics = evaluate.classification_metrics(labels, probs)
        assert metrics["roc_auc"] == pytest.approx(1.0)
        assert metrics["pr_auc"] == pytest.approx(1.0)
        assert metrics["balanced_accuracy"] == pytest.approx(1.0)
        assert metrics["mcc"] == pytest.approx(1.0)
        assert metrics["ef_1pct"] == pytest.approx(10.0)
        assert metrics["recall_5pct"] == pytest.approx(0.5)

    def test_active_rate_recorded(self) -> None:
        metrics = evaluate.classification_metrics([1] * 25 + [0] * 75, [0.5] * 100)
        assert metrics["active_rate"] == pytest.approx(0.25)
        assert metrics["n_actives"] == 25.0
        assert metrics["n"] == 100.0

    def test_single_class_metrics_are_nan(self) -> None:
        metrics = evaluate.classification_metrics([1] * 10, [0.6] * 10)
        assert math.isnan(metrics["roc_auc"])
        assert math.isnan(metrics["mcc"])

    def test_all_expected_keys(self) -> None:
        rng = np.random.default_rng(5)
        metrics = evaluate.classification_metrics(
            rng.integers(0, 2, 200), rng.random(200)
        )
        for key in (
            "roc_auc", "pr_auc", "balanced_accuracy", "mcc", "brier",
            "log_loss", "ece", "ef_1pct", "ef_5pct", "bedroc_1pct",
            "bedroc_5pct", "hit_rate_1pct", "recall_5pct",
        ):
            assert key in metrics, key

    def test_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            evaluate.classification_metrics([1, 0], [0.5])


class TestCalibration:
    """A calibrated model's predicted probabilities match observed frequency."""

    def test_perfectly_calibrated_has_tiny_ece(self) -> None:
        rng = np.random.default_rng(7)
        probs = rng.random(40_000)
        labels = (rng.random(40_000) < probs).astype(int)
        curve = evaluate.calibration_curve(labels, probs, n_bins=10)
        assert curve.ece < 0.02
        assert np.allclose(curve.mean_predicted, curve.observed_fraction, atol=0.03)

    def test_systematically_overconfident_has_large_ece(self) -> None:
        labels = [1] * 500 + [0] * 500
        probs = [0.99] * 500 + [0.9] * 500
        assert evaluate.expected_calibration_error(labels, probs) > 0.4

    def test_counts_sum_to_n(self) -> None:
        rng = np.random.default_rng(8)
        probs = rng.random(1000)
        labels = rng.integers(0, 2, 1000)
        curve = evaluate.calibration_curve(labels, probs, n_bins=10)
        assert int(curve.counts.sum()) == 1000

    def test_bins_are_ordered(self) -> None:
        rng = np.random.default_rng(9)
        curve = evaluate.calibration_curve(
            rng.integers(0, 2, 1000), rng.random(1000), n_bins=10
        )
        assert np.all(np.diff(curve.bin_centres) > 0)

    def test_too_few_bins(self) -> None:
        with pytest.raises(ValueError, match="n_bins"):
            evaluate.calibration_curve([1, 0], [0.9, 0.1], n_bins=1)


class TestBootstrapCI:
    """Bootstrap intervals over the test set."""

    def test_interval_brackets_the_point_estimate(self) -> None:
        rng = np.random.default_rng(11)
        labels = rng.integers(0, 2, 500)
        scores = labels * 0.5 + rng.random(500) * 0.5
        point, lower, upper = evaluate.bootstrap_ci(
            labels, scores, metric="roc_auc", n_resamples=200, seed=0
        )
        assert lower <= point <= upper
        assert 0.0 < lower < upper <= 1.0

    def test_wider_for_smaller_test_sets(self) -> None:
        """Interval width must track test-set size, which is the only
        uncertainty a bootstrap over compounds can capture.
        """
        widths = []
        for size in (80, 2000):
            rng = np.random.default_rng(12)
            labels = rng.integers(0, 2, size)
            # Overlapping score distributions, so the AUC is not degenerate.
            scores = labels * 0.3 + rng.random(size) * 0.7
            point, lower, upper = evaluate.bootstrap_ci(
                labels, scores, metric="roc_auc", n_resamples=400, seed=1
            )
            assert 0.5 < point < 0.95, point
            widths.append(upper - lower)
        assert widths[0] > 2 * widths[1], widths

    def test_deterministic_given_a_seed(self) -> None:
        rng = np.random.default_rng(13)
        labels = rng.integers(0, 2, 300)
        scores = rng.random(300)
        first = evaluate.bootstrap_ci(
            labels, scores, metric="roc_auc", n_resamples=100, seed=4
        )
        second = evaluate.bootstrap_ci(
            labels, scores, metric="roc_auc", n_resamples=100, seed=4
        )
        assert first == second

    def test_screening_metric_interval(self) -> None:
        rng = np.random.default_rng(14)
        labels = rng.integers(0, 2, 1000)
        scores = labels * 0.4 + rng.random(1000) * 0.6
        point, lower, upper = evaluate.bootstrap_ci(
            labels, scores, metric="ef", n_resamples=200, seed=2, fraction=0.05
        )
        assert lower <= point <= upper

    def test_unknown_metric(self) -> None:
        with pytest.raises(ValueError, match="unknown metric"):
            evaluate.bootstrap_ci([1, 0], [0.9, 0.1], metric="nonsense")


class TestBinarise:
    """The activity threshold."""

    def test_threshold_is_inclusive(self) -> None:
        labels = evaluate.binarise([5.99, 6.0, 6.01], threshold=6.0)
        assert labels.tolist() == [0, 1, 1]

    def test_default_threshold_is_one_micromolar(self) -> None:
        assert evaluate.ACTIVITY_THRESHOLD == 6.0
        from ligandml.chembl import nm_from_pchembl

        assert nm_from_pchembl(evaluate.ACTIVITY_THRESHOLD) == pytest.approx(1000.0)

    def test_returns_integers(self) -> None:
        labels = evaluate.binarise([4.0, 8.0])
        assert labels.dtype.kind == "i"
