"""Tests for the estimator factory."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from ligandml import models


@pytest.fixture
def learnable() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Binary features with a label that depends on the first two bits."""
    rng = np.random.default_rng(0)
    features = (rng.random((400, 50)) < 0.3).astype(float)
    values = 3.0 * features[:, 0] - 2.0 * features[:, 1] + rng.normal(0, 0.2, 400)
    labels = (values > values.mean()).astype(int)
    return features, values, labels


class TestBuildModel:
    """Pipeline construction."""

    @pytest.mark.parametrize("name", [m.name for m in models.MODELS])
    @pytest.mark.parametrize("task", models.TASKS)
    @pytest.mark.parametrize("kind", ["descriptors", "fingerprint"])
    def test_every_combination_builds(
        self, name: str, task: str, kind: str
    ) -> None:
        pipeline = models.build_model(name, task, kind, seed=0)
        assert isinstance(pipeline, Pipeline)
        assert pipeline.steps[-1][0] == "model"

    def test_descriptors_are_scaled(self) -> None:
        pipeline = models.build_model("linear", "regression", "descriptors")
        step_names = [name for name, _ in pipeline.steps]
        assert "scale" in step_names
        assert "impute" in step_names

    def test_fingerprints_are_not_scaled(self) -> None:
        pipeline = models.build_model("linear", "regression", "fingerprint")
        step_names = [name for name, _ in pipeline.steps]
        assert "scale" not in step_names
        assert "drop_constant" in step_names

    def test_unknown_model(self) -> None:
        with pytest.raises(KeyError, match="unknown model"):
            models.build_model("deep_learning", "regression")

    def test_unknown_task(self) -> None:
        with pytest.raises(ValueError, match="task must be"):
            models.build_model("linear", "ranking")

    def test_unknown_representation_kind(self) -> None:
        with pytest.raises(ValueError, match="unknown representation kind"):
            models.build_model("linear", "regression", "graph")

    def test_registry_is_consistent(self) -> None:
        assert len(models.MODELS_BY_NAME) == len(models.MODELS)
        assert models.DEFAULT_MODEL in models.MODELS_BY_NAME
        assert models.get_model_spec("linear").name == "linear"

    def test_unknown_spec_lookup(self) -> None:
        with pytest.raises(KeyError, match="unknown model"):
            models.get_model_spec("nope")


class TestFitPredict:
    """Fitting and prediction, including the probability contract."""

    @pytest.mark.parametrize("name", [m.name for m in models.MODELS])
    def test_classification_returns_probabilities(
        self, name: str, learnable: tuple[np.ndarray, ...]
    ) -> None:
        features, _, labels = learnable
        prediction = models.fit_predict(
            models.build_model(name, "classification", "fingerprint", seed=0),
            features[:300], labels[:300], features[300:], "classification",
        )
        assert prediction.shape == (100,)
        assert prediction.min() >= 0.0 and prediction.max() <= 1.0

    @pytest.mark.parametrize("name", [m.name for m in models.MODELS])
    def test_classification_learns_the_signal(
        self, name: str, learnable: tuple[np.ndarray, ...]
    ) -> None:
        from sklearn.metrics import roc_auc_score

        features, _, labels = learnable
        prediction = models.fit_predict(
            models.build_model(name, "classification", "fingerprint", seed=0),
            features[:300], labels[:300], features[300:], "classification",
        )
        assert roc_auc_score(labels[300:], prediction) > 0.75

    @pytest.mark.parametrize("name", [m.name for m in models.MODELS])
    def test_regression_learns_the_signal(
        self, name: str, learnable: tuple[np.ndarray, ...]
    ) -> None:
        from sklearn.metrics import r2_score

        features, values, _ = learnable
        prediction = models.fit_predict(
            models.build_model(name, "regression", "fingerprint", seed=0),
            features[:300], values[:300], features[300:], "regression",
        )
        assert prediction.shape == (100,)
        assert r2_score(values[300:], prediction) > 0.5

    def test_unknown_task(self, learnable: tuple[np.ndarray, ...]) -> None:
        features, values, _ = learnable
        with pytest.raises(ValueError, match="task must be"):
            models.fit_predict(
                models.build_model("linear", "regression"),
                features, values, features, "clustering",
            )

    def test_descriptor_pipeline_handles_nan(self) -> None:
        """A descriptor matrix with a missing value must still fit, because
        the imputer sits in front of the estimator.
        """
        rng = np.random.default_rng(1)
        features = rng.normal(size=(200, 10))
        features[3, 4] = np.nan
        values = features[:, 0] * 2.0
        prediction = models.fit_predict(
            models.build_model("linear", "regression", "descriptors", seed=0),
            features[:150], values[:150], features[150:], "regression",
        )
        assert np.isfinite(prediction).all()


class TestDeterminism:
    """Same seed, same answer."""

    @pytest.mark.parametrize("name", [m.name for m in models.MODELS])
    @pytest.mark.parametrize("task", models.TASKS)
    def test_repeatable(
        self, name: str, task: str, learnable: tuple[np.ndarray, ...]
    ) -> None:
        features, values, labels = learnable
        target = values if task == "regression" else labels
        first = models.fit_predict(
            models.build_model(name, task, "fingerprint", seed=7),
            features[:300], target[:300], features[300:], task,
        )
        second = models.fit_predict(
            models.build_model(name, task, "fingerprint", seed=7),
            features[:300], target[:300], features[300:], task,
        )
        assert np.allclose(first, second)

    def test_different_seeds_can_differ(
        self, learnable: tuple[np.ndarray, ...]
    ) -> None:
        features, values, _ = learnable
        first = models.fit_predict(
            models.build_model("random_forest", "regression", "fingerprint", seed=1),
            features[:300], values[:300], features[300:], "regression",
        )
        second = models.fit_predict(
            models.build_model("random_forest", "regression", "fingerprint", seed=2),
            features[:300], values[:300], features[300:], "regression",
        )
        assert not np.allclose(first, second)
