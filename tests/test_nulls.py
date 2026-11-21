"""Tests for the y-scrambling null model.

Two things must hold for the permutation test to mean anything: on data
with real signal the observed score must sit far above the null, and on
data with no signal it must sit inside it. A null model that always says
"significant" is worse than none at all, so both directions are tested.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from sklearn.metrics import r2_score, roc_auc_score

from ligandml import models, nulls


@pytest.fixture
def signal_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Binary features where the label depends on bit 0, plus noise bits."""
    rng = np.random.default_rng(0)
    features = (rng.random((600, 40)) < 0.3).astype(float)
    labels = features[:, 0].astype(int)
    flip = rng.random(600) < 0.1  # 10% label noise, so the task is not trivial
    labels = np.where(flip, 1 - labels, labels)
    return features[:400], labels[:400], features[400:], labels[400:]


@pytest.fixture
def noise_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Binary features with labels that are independent of them.

    The test set is deliberately large (1600 rows). With a small test set a
    model fitted to pure noise can reach ROC AUC 0.58 by chance, which would
    make the control assertion below a coin toss rather than a check.
    """
    rng = np.random.default_rng(1)
    features = (rng.random((2000, 40)) < 0.3).astype(float)
    labels = rng.integers(0, 2, 2000)
    return features[:400], labels[:400], features[400:], labels[400:]


def _auc(truth: np.ndarray, prediction: np.ndarray) -> float:
    """ROC AUC as a scorer, higher being better."""
    return float(roc_auc_score(truth, prediction))


class TestYScramble:
    """The permutation test itself."""

    def test_real_signal_beats_the_null(
        self, signal_data: tuple[np.ndarray, ...]
    ) -> None:
        x_train, y_train, x_test, y_test = signal_data
        model = models.build_model("linear", "classification", "fingerprint", seed=0)
        result = nulls.y_scramble(
            model, x_train, y_train, x_test, y_test, _auc,
            task="classification", metric="roc_auc", n_permutations=12, seed=0,
        )
        assert result.observed > 0.8
        assert result.observed > result.null_scores.max()
        assert result.p_value < 0.1
        assert result.z_score > 3.0

    def test_null_is_centred_on_chance(
        self, signal_data: tuple[np.ndarray, ...]
    ) -> None:
        """Scrambled labels must score like a coin toss, ROC AUC 0.5."""
        x_train, y_train, x_test, y_test = signal_data
        model = models.build_model("linear", "classification", "fingerprint", seed=0)
        result = nulls.y_scramble(
            model, x_train, y_train, x_test, y_test, _auc,
            task="classification", metric="roc_auc", n_permutations=20, seed=1,
        )
        assert result.null_mean == pytest.approx(0.5, abs=0.08)

    def test_no_signal_is_not_significant(
        self, noise_data: tuple[np.ndarray, ...]
    ) -> None:
        """The control that matters: on label-independent features the test
        must fail to reject, or it would call everything significant.
        """
        x_train, y_train, x_test, y_test = noise_data
        model = models.build_model("linear", "classification", "fingerprint", seed=0)
        result = nulls.y_scramble(
            model, x_train, y_train, x_test, y_test, _auc,
            task="classification", metric="roc_auc", n_permutations=20, seed=2,
        )
        assert result.observed < 0.56, "features should carry no signal"
        assert result.p_value > 0.05
        assert abs(result.z_score) < 3.0

    def test_regression_framing(self) -> None:
        rng = np.random.default_rng(3)
        features = (rng.random((500, 30)) < 0.3).astype(float)
        values = 2.0 * features[:, 0] + rng.normal(0.0, 0.3, 500)
        model = models.build_model("linear", "regression", "fingerprint", seed=0)
        result = nulls.y_scramble(
            model, features[:350], values[:350], features[350:], values[350:],
            lambda t, p: float(r2_score(t, p)),
            task="regression", metric="r2", n_permutations=10, seed=0,
        )
        assert result.observed > 0.5
        assert result.null_mean < 0.1

    def test_p_value_is_never_zero(
        self, signal_data: tuple[np.ndarray, ...]
    ) -> None:
        """The (1 + k) / (1 + n) estimator is conservative by design."""
        x_train, y_train, x_test, y_test = signal_data
        model = models.build_model("linear", "classification", "fingerprint", seed=0)
        result = nulls.y_scramble(
            model, x_train, y_train, x_test, y_test, _auc,
            task="classification", n_permutations=9, seed=0,
        )
        assert result.p_value >= 1.0 / 10.0
        assert result.p_value > 0.0

    def test_permutation_count_respected(
        self, signal_data: tuple[np.ndarray, ...]
    ) -> None:
        x_train, y_train, x_test, y_test = signal_data
        model = models.build_model("linear", "classification", "fingerprint", seed=0)
        result = nulls.y_scramble(
            model, x_train, y_train, x_test, y_test, _auc,
            task="classification", n_permutations=7, seed=0,
        )
        assert result.null_scores.size == 7

    def test_deterministic_given_a_seed(
        self, signal_data: tuple[np.ndarray, ...]
    ) -> None:
        x_train, y_train, x_test, y_test = signal_data
        model = models.build_model("linear", "classification", "fingerprint", seed=0)
        kwargs = dict(task="classification", n_permutations=6, seed=42)
        first = nulls.y_scramble(
            model, x_train, y_train, x_test, y_test, _auc, **kwargs
        )
        second = nulls.y_scramble(
            model, x_train, y_train, x_test, y_test, _auc, **kwargs
        )
        assert np.allclose(first.null_scores, second.null_scores)
        assert first.observed == second.observed

    def test_input_model_not_mutated(
        self, signal_data: tuple[np.ndarray, ...]
    ) -> None:
        """The pipeline is cloned per fit, so the caller's object stays
        unfitted and reusable.
        """
        from sklearn.exceptions import NotFittedError

        x_train, y_train, x_test, y_test = signal_data
        model = models.build_model("linear", "classification", "fingerprint", seed=0)
        nulls.y_scramble(
            model, x_train, y_train, x_test, y_test, _auc,
            task="classification", n_permutations=3, seed=0,
        )
        with pytest.raises(NotFittedError):
            model.predict(x_test)

    def test_test_labels_are_not_permuted(
        self, signal_data: tuple[np.ndarray, ...]
    ) -> None:
        """Permuting the test labels too would destroy the comparison; the
        observed score must match a plain fit exactly.
        """
        x_train, y_train, x_test, y_test = signal_data
        model = models.build_model("linear", "classification", "fingerprint", seed=0)
        direct = models.fit_predict(
            models.build_model("linear", "classification", "fingerprint", seed=0),
            x_train, y_train, x_test, "classification",
        )
        result = nulls.y_scramble(
            model, x_train, y_train, x_test, y_test, _auc,
            task="classification", n_permutations=2, seed=0,
        )
        assert result.observed == pytest.approx(_auc(y_test, direct))

    def test_summary_dictionary(self, signal_data: tuple[np.ndarray, ...]) -> None:
        x_train, y_train, x_test, y_test = signal_data
        model = models.build_model("linear", "classification", "fingerprint", seed=0)
        result = nulls.y_scramble(
            model, x_train, y_train, x_test, y_test, _auc,
            task="classification", metric="roc_auc", n_permutations=5, seed=0,
        )
        summary = result.to_dict()
        for key in (
            "metric", "observed", "null_mean", "null_std", "null_min",
            "null_max", "n_permutations", "p_value", "z_score",
        ):
            assert key in summary
        assert summary["n_permutations"] == 5
        assert not math.isnan(summary["observed"])

    def test_bad_permutation_count(
        self, signal_data: tuple[np.ndarray, ...]
    ) -> None:
        x_train, y_train, x_test, y_test = signal_data
        model = models.build_model("linear", "classification", "fingerprint", seed=0)
        with pytest.raises(ValueError, match="n_permutations"):
            nulls.y_scramble(
                model, x_train, y_train, x_test, y_test, _auc, n_permutations=0
            )

    def test_mismatched_rows(self, signal_data: tuple[np.ndarray, ...]) -> None:
        x_train, y_train, x_test, y_test = signal_data
        model = models.build_model("linear", "classification", "fingerprint", seed=0)
        with pytest.raises(ValueError, match="disagree"):
            nulls.y_scramble(
                model, x_train, y_train[:-1], x_test, y_test, _auc,
                task="classification", n_permutations=2,
            )
