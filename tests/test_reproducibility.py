"""Reproducibility tests: the same inputs and seed must give the same numbers.

Every claim in the README is a number produced by this code. If any stage
is silently stochastic, those numbers drift and the README becomes
decoration. These tests pin determinism at each stage - splits, features,
fits, metrics, nulls - and check that the committed result tables are
internally consistent with each other.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ligandml import RANDOM_STATE, domain, evaluate, featurise, models, splits
from tests.conftest import DATA_DIR, REPO_ROOT

RESULTS_DIR = REPO_ROOT / "results"


@pytest.fixture
def small_dataset() -> pd.DataFrame:
    """A deterministic 60-compound dataset drawn from a cached target.

    The cached tables are sorted by descending potency, so taking the first
    60 rows would give an all-active subset and silently skip the
    classification checks. Rows are therefore taken at an even stride
    across the whole potency range, which keeps the subset deterministic
    while guaranteeing both classes are present.

    Falls back to a synthetic frame when no cache is present, so the
    determinism checks still run on a fresh clone.
    """
    from ligandml.chembl import TARGETS, dataset_paths, load_target

    for spec in TARGETS:
        if dataset_paths(spec.name, DATA_DIR)[0].exists():
            frame = load_target(spec.name, DATA_DIR)
            stride = max(1, len(frame) // 60)
            return frame.iloc[::stride].head(60).reset_index(drop=True)
    rng = np.random.default_rng(0)
    bases = ["c1ccccc1", "c1ccncc1", "c1ccc2ccccc2c1", "C1CCCCC1", "c1ccsc1"]
    tails = ["C", "CC", "CCC", "CCCC", "O", "CO", "N", "CN", "F", "Cl", "Br", "C(=O)O"]
    smiles = [f"{tail}{base}" for base in bases for tail in tails]
    return pd.DataFrame(
        {
            "canonical_smiles": smiles[:60],
            "pchembl_value": rng.uniform(4.0, 9.0, min(60, len(smiles))),
            "first_year": rng.integers(1995, 2024, min(60, len(smiles))),
        }
    )


class TestGlobalSeed:
    """One seed governs the whole study."""

    def test_seed_is_an_integer_constant(self) -> None:
        assert isinstance(RANDOM_STATE, int)
        assert RANDOM_STATE > 0


class TestFeatureDeterminism:
    """Featurisation has no hidden randomness."""

    @pytest.mark.parametrize("name", [r.name for r in featurise.REPRESENTATIONS])
    def test_repeated_featurisation_is_identical(
        self, small_dataset: pd.DataFrame, name: str
    ) -> None:
        first, names_a, kept_a = featurise.featurise_frame(small_dataset, name)
        second, names_b, kept_b = featurise.featurise_frame(small_dataset, name)
        assert np.array_equal(first, second)
        assert names_a == names_b
        assert len(kept_a) == len(kept_b)

    def test_row_order_does_not_change_per_row_features(
        self, small_dataset: pd.DataFrame
    ) -> None:
        """Featurising a reversed frame must give the reversed matrix, so no
        feature depends on its neighbours.
        """
        forward, _, _ = featurise.featurise_frame(small_dataset, "maccs")
        reversed_frame = small_dataset.iloc[::-1].reset_index(drop=True)
        backward, _, _ = featurise.featurise_frame(reversed_frame, "maccs")
        assert np.allclose(forward, backward[::-1])


class TestSplitDeterminism:
    """Splits are reproducible, and two of them are seed-independent."""

    @pytest.mark.parametrize("strategy", splits.SPLIT_STRATEGIES)
    def test_repeated_split_is_identical(
        self, small_dataset: pd.DataFrame, strategy: str
    ) -> None:
        first = splits.make_split(small_dataset, strategy, seed=RANDOM_STATE)
        second = splits.make_split(small_dataset, strategy, seed=RANDOM_STATE)
        assert np.array_equal(first.train_index, second.train_index)
        assert np.array_equal(first.test_index, second.test_index)

    def test_scaffold_split_ignores_the_seed(
        self, small_dataset: pd.DataFrame
    ) -> None:
        """The default scaffold split is deterministic by construction, so a
        reported scaffold-split number cannot be a lucky seed.
        """
        first = splits.make_split(small_dataset, "scaffold", seed=1)
        second = splits.make_split(small_dataset, "scaffold", seed=99)
        assert np.array_equal(first.test_index, second.test_index)

    def test_time_split_ignores_the_seed(self, small_dataset: pd.DataFrame) -> None:
        first = splits.make_split(small_dataset, "time", seed=1)
        second = splits.make_split(small_dataset, "time", seed=99)
        assert np.array_equal(first.test_index, second.test_index)

    def test_scaffold_split_ignores_row_order(
        self, small_dataset: pd.DataFrame
    ) -> None:
        """Shuffling the rows must move the same *compounds* across the
        boundary, so the split depends on chemistry and not on file order.
        """
        split = splits.make_split(small_dataset, "scaffold", seed=RANDOM_STATE)
        held_out = set(
            small_dataset["canonical_smiles"].iloc[split.test_index].tolist()
        )
        shuffled = small_dataset.sample(
            frac=1.0, random_state=7
        ).reset_index(drop=True)
        shuffled_split = splits.make_split(shuffled, "scaffold", seed=RANDOM_STATE)
        shuffled_held_out = set(
            shuffled["canonical_smiles"].iloc[shuffled_split.test_index].tolist()
        )
        assert held_out == shuffled_held_out


class TestPipelineDeterminism:
    """End to end: the same seed gives the same metrics."""

    @pytest.mark.parametrize("task", models.TASKS)
    def test_full_cell_is_reproducible(
        self, small_dataset: pd.DataFrame, task: str
    ) -> None:
        matrix, _, kept = featurise.featurise_frame(small_dataset, "maccs")
        split = splits.make_split(kept, "scaffold", seed=RANDOM_STATE)
        target = (
            evaluate.binarise(kept["pchembl_value"])
            if task == "classification"
            else kept["pchembl_value"].to_numpy()
        )
        if task == "classification" and len(set(target.tolist())) < 2:
            pytest.skip("degenerate class balance in this subset")

        def run() -> np.ndarray:
            return models.fit_predict(
                models.build_model("random_forest", task, "fingerprint",
                                   seed=RANDOM_STATE),
                matrix[split.train_index],
                target[split.train_index],
                matrix[split.test_index],
                task,
            )

        assert np.allclose(run(), run())

    def test_metrics_are_pure_functions(self) -> None:
        rng = np.random.default_rng(RANDOM_STATE)
        labels = rng.integers(0, 2, 400)
        probs = rng.random(400)
        first = evaluate.classification_metrics(labels, probs)
        second = evaluate.classification_metrics(labels, probs)
        assert first == second

    def test_similarity_is_deterministic(
        self, small_dataset: pd.DataFrame
    ) -> None:
        matrix, _, _ = featurise.featurise_frame(small_dataset, "maccs")
        first = domain.nearest_neighbour_similarity(matrix[30:], matrix[:30])
        second = domain.nearest_neighbour_similarity(matrix[30:], matrix[:30])
        assert np.array_equal(first, second)


class TestCommittedResults:
    """The committed tables must be self-consistent, or the README is wrong.

    Skipped on a fresh clone, where ``results/`` has not been generated.
    """

    @pytest.fixture
    def comparison(self) -> pd.DataFrame:
        path = RESULTS_DIR / "split_comparison.csv"
        if not path.exists():
            pytest.skip("no committed results; run scripts/run_study.py")
        return pd.read_csv(path)

    def test_summary_matches_the_tables(self, comparison: pd.DataFrame) -> None:
        path = RESULTS_DIR / "run_summary.json"
        if not path.exists():
            pytest.skip("no committed run summary")
        summary = json.loads(path.read_text(encoding="utf-8"))
        assert summary["seed"] == RANDOM_STATE
        assert set(summary["splits"]) == set(comparison["split"].unique())
        assert set(summary["models"]) == set(comparison["model"].unique())
        assert {t["name"] for t in summary["targets"]} == set(
            comparison["target"].unique()
        )

    def test_every_cell_is_present(self, comparison: pd.DataFrame) -> None:
        """The matrix must be complete: no silently dropped cells."""
        expected = (
            comparison["target"].nunique()
            * comparison["model"].nunique()
            * comparison["split"].nunique()
            * comparison["task"].nunique()
        )
        assert len(comparison) == expected

    def test_scaffold_splits_leaked_nothing(self, comparison: pd.DataFrame) -> None:
        scaffold = comparison[comparison["split"] == "scaffold"]
        assert not scaffold.empty
        assert (scaffold["n_shared_scaffolds"] == 0).all()
        assert (scaffold["frac_test_compounds_leaked"] == 0.0).all()

    def test_random_splits_leaked_a_lot(self, comparison: pd.DataFrame) -> None:
        """The premise of the study, read back off the committed table."""
        random_rows = comparison[comparison["split"] == "random"]
        assert not random_rows.empty
        assert (random_rows["frac_test_compounds_leaked"] > 0.2).all()

    def test_classification_metrics_are_in_range(
        self, comparison: pd.DataFrame
    ) -> None:
        rows = comparison[comparison["task"] == "classification"]
        assert rows["roc_auc"].between(0.0, 1.0).all()
        assert rows["pr_auc"].between(0.0, 1.0).all()
        assert rows["bedroc_1pct"].between(0.0, 1.0).all()
        assert rows["ece"].between(0.0, 1.0).all()
        assert (rows["ef_1pct"] <= 100.0 + 1e-9).all()
        assert (rows["ef_1pct"] >= 0.0).all()

    def test_test_fractions_are_near_the_target(
        self, comparison: pd.DataFrame
    ) -> None:
        realised = comparison["n_test"] / (
            comparison["n_train"] + comparison["n_test"]
        )
        assert realised.between(0.1, 0.35).all()

    def test_gap_table_agrees_with_the_comparison(
        self, comparison: pd.DataFrame
    ) -> None:
        path = RESULTS_DIR / "split_gap.csv"
        if not path.exists():
            pytest.skip("no committed gap table")
        gap = pd.read_csv(path)
        classification = gap[gap["task"] == "classification"]
        for row in classification.itertuples():
            match = comparison[
                (comparison["target"] == row.target)
                & (comparison["model"] == row.model)
                & (comparison["task"] == "classification")
                & (comparison["split"] == "random")
            ]
            assert float(match["roc_auc"].iat[0]) == pytest.approx(row.random)
            if pd.notna(row.scaffold):
                assert row.drop_vs_random_scaffold == pytest.approx(
                    row.random - row.scaffold, abs=1e-4
                )

    def test_dataset_summary_clears_the_threshold(self) -> None:
        path = RESULTS_DIR / "dataset_summary.csv"
        if not path.exists():
            pytest.skip("no committed dataset summary")
        from ligandml.chembl import MIN_COMPOUNDS

        summary = pd.read_csv(path)
        assert len(summary) >= 3
        assert (summary["n_compounds"] >= MIN_COMPOUNDS).all()
        assert summary["active_rate"].between(0.02, 0.98).all()

    def test_attrition_is_monotone(self) -> None:
        path = RESULTS_DIR / "attrition.csv"
        if not path.exists():
            pytest.skip("no committed attrition table")
        attrition = pd.read_csv(path)
        for _, group in attrition.groupby("target"):
            remaining = group["remaining"].tolist()
            assert all(
                later <= earlier
                for earlier, later in zip(remaining, remaining[1:], strict=False)
            )

    def test_nulls_show_real_signal(self) -> None:
        path = RESULTS_DIR / "y_scrambling.csv"
        if not path.exists():
            pytest.skip("no committed null results")
        nulls_frame = pd.read_csv(path)
        assert (nulls_frame["observed"] > nulls_frame["null_max"]).all()
        assert (nulls_frame["null_mean"] < 0.65).all()

    def test_figures_exist_for_every_result_table(self) -> None:
        figures = REPO_ROOT / "figures"
        if not figures.exists():
            pytest.skip("no committed figures")
        for name in (
            "split_strategy_performance.png",
            "split_strategy_enrichment.png",
            "split_gap.png",
            "split_leakage.png",
            "representation_comparison.png",
            "error_vs_similarity.png",
            "coverage_tradeoff.png",
            "enrichment_curves.png",
            "calibration.png",
            "y_scrambling.png",
            "chemical_space_pca.png",
        ):
            assert (figures / name).exists(), name
            assert (figures / name).stat().st_size > 5_000, name


def test_repo_layout() -> None:
    """The package exposes exactly the modules the README documents."""
    package = REPO_ROOT / "ligandml"
    for module in (
        "chembl.py", "featurise.py", "splits.py", "models.py",
        "evaluate.py", "domain.py", "nulls.py", "plots.py",
    ):
        assert (package / module).exists(), module
    assert (REPO_ROOT / "scripts" / "run_study.py").exists()
    assert (REPO_ROOT / "README.md").exists()
    assert (REPO_ROOT / "LICENSE").exists()


def test_license_is_mit_and_attributed() -> None:
    text = (Path(REPO_ROOT) / "LICENSE").read_text(encoding="utf-8")
    assert "MIT License" in text
    assert "Gichia Muiruri" in text
    assert "2026" in text
