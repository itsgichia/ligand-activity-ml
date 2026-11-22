"""Tests for the split strategies.

The central claim of the project is that scaffold and time splits are
harder than random splits *because they do not leak analogues*. If the
scaffold split quietly let a scaffold group straddle the boundary, the
headline result would be an artefact. The integrity tests here are
therefore the load-bearing ones in the suite.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ligandml import splits


@pytest.fixture
def frame(congeneric_smiles: list[str]) -> pd.DataFrame:
    """Twelve compounds in three scaffold families, with years and potencies."""
    return pd.DataFrame(
        {
            "canonical_smiles": congeneric_smiles,
            "pchembl_value": np.linspace(5.0, 8.0, len(congeneric_smiles)),
            "first_year": [2000, 2000, 2001, 2002, 2005, 2006, 2007, 2008,
                           2015, 2016, 2017, 2018],
        }
    )


class TestMurckoScaffold:
    """Bemis-Murcko reduction to ring systems and linkers."""

    @pytest.mark.parametrize(
        ("smiles", "expected"),
        [
            ("Cc1ccccc1", "c1ccccc1"),
            ("CCCCc1ccccc1", "c1ccccc1"),
            ("OCc1ccccc1", "c1ccccc1"),
            ("Cc1ccncc1", "c1ccncc1"),
        ],
    )
    def test_side_chains_removed(self, smiles: str, expected: str) -> None:
        assert splits.murcko_scaffold(smiles) == expected

    def test_congeneric_series_shares_one_scaffold(
        self, congeneric_smiles: list[str]
    ) -> None:
        groups = splits.scaffold_groups(congeneric_smiles)
        assert len(set(groups.tolist())) == 3
        assert len(set(groups[:4].tolist())) == 1
        assert len(set(groups[4:8].tolist())) == 1
        assert len(set(groups[8:].tolist())) == 1

    def test_acyclic_molecules_pooled(self) -> None:
        groups = splits.scaffold_groups(["CCCCCC", "CCCCO", "CC(C)CC"])
        assert set(groups.tolist()) == {splits.ACYCLIC_SENTINEL}

    def test_unparseable_kept_separate(self) -> None:
        groups = splits.scaffold_groups(["not-a-smiles", "also-not"])
        assert len(set(groups.tolist())) == 2

    def test_stereoisomers_group_together(self) -> None:
        groups = splits.scaffold_groups(
            ["C[C@H](O)c1ccccc1", "C[C@@H](O)c1ccccc1"]
        )
        assert len(set(groups.tolist())) == 1

    def test_linker_retained(self) -> None:
        """A biphenyl-ether keeps both rings and the linking oxygen."""
        scaffold = splits.murcko_scaffold("CC(=O)Nc1ccc(Oc2ccccc2)cc1")
        assert scaffold.count("1") + scaffold.count("2") >= 4
        assert "O" in scaffold


class TestScaffoldSplitIntegrity:
    """No scaffold group may appear on both sides. This is non-negotiable."""

    def test_no_scaffold_straddles(self, frame: pd.DataFrame) -> None:
        split = splits.make_split(frame, "scaffold", test_size=0.33, seed=0)
        groups = splits.scaffold_groups(frame["canonical_smiles"].tolist())
        train = set(groups[split.train_index].tolist())
        test = set(groups[split.test_index].tolist())
        assert not train & test

    @pytest.mark.parametrize("seed", range(8))
    @pytest.mark.parametrize("test_size", [0.1, 0.2, 0.3, 0.5])
    def test_no_scaffold_straddles_over_many_settings(
        self, frame: pd.DataFrame, seed: int, test_size: float
    ) -> None:
        split = splits.scaffold_split(
            frame["canonical_smiles"].tolist(),
            test_size=test_size,
            shuffle_groups=True,
            seed=seed,
        )
        groups = splits.scaffold_groups(frame["canonical_smiles"].tolist())
        assert not set(groups[split.train_index].tolist()) & set(
            groups[split.test_index].tolist()
        )

    def test_overlap_report_is_zero(self, frame: pd.DataFrame) -> None:
        split = splits.make_split(frame, "scaffold", test_size=0.33, seed=0)
        overlap = splits.scaffold_overlap(
            frame["canonical_smiles"].tolist(), split
        )
        assert overlap["n_shared_scaffolds"] == 0.0
        assert overlap["frac_test_scaffolds_seen"] == 0.0
        assert overlap["frac_test_compounds_seen"] == 0.0

    def test_cached_data_has_no_scaffold_leak(
        self, cached_targets: list[str]
    ) -> None:
        """The integrity guarantee must also hold on the real datasets, which
        are far messier than the fixture.
        """
        if not cached_targets:
            pytest.skip("no cached datasets")
        from ligandml.chembl import load_target
        from tests.conftest import DATA_DIR

        for name in cached_targets[:2]:
            data = load_target(name, DATA_DIR)
            split = splits.make_split(data, "scaffold", seed=0)
            overlap = splits.scaffold_overlap(
                data["canonical_smiles"].tolist(), split
            )
            assert overlap["n_shared_scaffolds"] == 0.0

    def test_largest_groups_go_to_training(self, frame: pd.DataFrame) -> None:
        """The deterministic construction must put populous chemotypes in
        training and leave the long tail for testing.
        """
        smiles = frame["canonical_smiles"].tolist() + ["Cc1ccc2ccccc2c1"] * 0
        extra = ["c1ccc2c(c1)oc1ccccc12"]  # One singleton scaffold.
        split = splits.scaffold_split(smiles + extra, test_size=0.1)
        assert split.test_index.tolist() == [len(smiles)]


class TestRandomSplitLeaks:
    """The counterpart: a random split is expected to leak, and does."""

    def test_random_split_shares_scaffolds(self, frame: pd.DataFrame) -> None:
        split = splits.make_split(frame, "random", test_size=0.33, seed=3)
        overlap = splits.scaffold_overlap(
            frame["canonical_smiles"].tolist(), split
        )
        assert overlap["n_shared_scaffolds"] > 0

    def test_sizes_and_disjointness(self) -> None:
        split = splits.random_split(100, test_size=0.2, seed=7)
        assert split.n_train == 80
        assert split.n_test == 20
        split.validate(100)

    def test_seed_is_deterministic(self) -> None:
        first = splits.random_split(500, test_size=0.2, seed=11)
        second = splits.random_split(500, test_size=0.2, seed=11)
        third = splits.random_split(500, test_size=0.2, seed=12)
        assert np.array_equal(first.test_index, second.test_index)
        assert not np.array_equal(first.test_index, third.test_index)

    @pytest.mark.parametrize("test_size", [0.0, 1.0, -0.1, 1.5])
    def test_bad_test_size(self, test_size: float) -> None:
        with pytest.raises(ValueError, match="test_size"):
            splits.random_split(100, test_size=test_size)

    def test_too_few_rows(self) -> None:
        with pytest.raises(ValueError, match="at least 2 rows"):
            splits.random_split(1)


class TestTimeSplit:
    """Train on the past, test on the future."""

    def test_no_future_leaks_into_training(self, frame: pd.DataFrame) -> None:
        split = splits.make_split(frame, "time", test_size=0.33)
        years = frame["first_year"].to_numpy()
        assert years[split.train_index].max() < years[split.test_index].min()

    def test_cutoff_recorded(self, frame: pd.DataFrame) -> None:
        split = splits.make_split(frame, "time", test_size=0.33)
        cutoff = split.details["cutoff_year"]
        years = frame["first_year"].to_numpy()
        assert (years[split.test_index] >= cutoff).all()
        assert (years[split.train_index] < cutoff).all()

    def test_unknown_years_excluded_and_counted(self) -> None:
        frame = pd.DataFrame({"first_year": [2000, 2001, 0, 2010, 2011, 0]})
        split = splits.time_split(frame["first_year"].tolist(), test_size=0.4)
        assert split.n_dropped == 2
        assert split.n_train + split.n_test == 4
        split.validate(len(frame))

    def test_needs_two_distinct_years(self) -> None:
        with pytest.raises(ValueError, match="two distinct publication years"):
            splits.time_split([2001] * 10)

    def test_fraction_is_close_to_target(self) -> None:
        years = list(range(1990, 2020)) * 10
        split = splits.time_split(years, test_size=0.2)
        assert abs(split.test_fraction - 0.2) < 0.06


class TestSplitResult:
    """Validation of the partition object itself."""

    def test_overlap_rejected(self) -> None:
        bad = splits.SplitResult(
            "random", np.array([0, 1, 2]), np.array([2, 3])
        )
        with pytest.raises(ValueError, match="both sides"):
            bad.validate(4)

    def test_empty_side_rejected(self) -> None:
        bad = splits.SplitResult("random", np.array([0, 1]), np.array([]))
        with pytest.raises(ValueError, match="empty side"):
            bad.validate(2)

    def test_incomplete_coverage_rejected(self) -> None:
        bad = splits.SplitResult("random", np.array([0]), np.array([1]))
        with pytest.raises(ValueError, match="covers"):
            bad.validate(5)

    def test_out_of_range_rejected(self) -> None:
        bad = splits.SplitResult("random", np.array([0]), np.array([9]))
        with pytest.raises(ValueError):
            bad.validate(3)


class TestMakeSplit:
    """The dispatcher."""

    @pytest.mark.parametrize("strategy", splits.SPLIT_STRATEGIES)
    def test_every_strategy_validates(
        self, frame: pd.DataFrame, strategy: str
    ) -> None:
        split = splits.make_split(frame, strategy, test_size=0.33, seed=0)
        assert split.strategy == strategy
        assert split.n_train > 0 and split.n_test > 0

    def test_unknown_strategy(self, frame: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="unknown split strategy"):
            splits.make_split(frame, "magic")

    def test_missing_column(self) -> None:
        with pytest.raises(KeyError):
            splits.make_split(pd.DataFrame({"x": [1, 2, 3]}), "scaffold")


def test_leakage_ordering_on_real_data(cached_targets: list[str]) -> None:
    """The project's premise, checked on real data: a random split leaks
    far more scaffold structure than a scaffold or time split.
    """
    if not cached_targets:
        pytest.skip("no cached datasets")
    from ligandml.chembl import load_target
    from tests.conftest import DATA_DIR

    data = load_target(cached_targets[0], DATA_DIR)
    smiles = data["canonical_smiles"].tolist()
    leak = {
        strategy: splits.scaffold_overlap(
            smiles, splits.make_split(data, strategy, seed=0)
        )["frac_test_compounds_seen"]
        for strategy in splits.SPLIT_STRATEGIES
    }
    assert leak["random"] > leak["time"]
    assert leak["scaffold"] == 0.0


class TestStratifiedScaffoldSplit:
    """Scaffold separation within each class, class balance across the split.

    A screening library is mostly decoys, and enrichment factor is bounded
    by the active rate of the set it is computed on. A split that moves
    that rate moves the ceiling the result is judged against, so these
    tests check the balance is held and the scaffold separation survives.
    """

    def _library(self, n_actives: int = 20, n_decoys: int = 200):
        """Carbocycles as actives, N-heterocycles as decoys.

        Ring size is what makes each Murcko scaffold distinct here;
        substituent chains would all collapse to the same scaffold.
        """
        actives = ["C1" + "C" * n + "1" for n in range(2, 2 + n_actives)]
        decoys = ["C1" + "C" * n + "N1" for n in range(2, 2 + n_decoys)]
        labels = np.array([1] * n_actives + [0] * n_decoys)
        return actives + decoys, labels

    def test_test_set_keeps_the_library_active_rate(self) -> None:
        smiles, labels = self._library()
        split = splits.stratified_scaffold_split(smiles, labels, test_size=0.2)
        assert labels[split.test_index].mean() == pytest.approx(
            labels.mean(), abs=0.02
        )

    def test_a_plain_split_does_not(self) -> None:
        """The failure this function exists to fix."""
        smiles, labels = self._library()
        plain = splits.scaffold_split(smiles, test_size=0.2)
        stratified = splits.stratified_scaffold_split(smiles, labels, test_size=0.2)
        plain_error = abs(labels[plain.test_index].mean() - labels.mean())
        strat_error = abs(labels[stratified.test_index].mean() - labels.mean())
        assert strat_error < plain_error

    def test_split_is_a_partition(self) -> None:
        smiles, labels = self._library()
        split = splits.stratified_scaffold_split(smiles, labels, test_size=0.2)
        split.validate(len(smiles))
        combined = np.concatenate([split.train_index, split.test_index])
        assert sorted(combined.tolist()) == list(range(len(smiles)))

    def test_no_scaffold_crosses_the_boundary(self) -> None:
        """Splitting per class must not reintroduce scaffold leakage."""
        smiles, labels = self._library()
        split = splits.stratified_scaffold_split(smiles, labels, test_size=0.2)
        groups = splits.scaffold_groups(smiles)
        assert not set(groups[split.train_index]) & set(groups[split.test_index])

    def test_records_scaffold_counts_per_class(self) -> None:
        smiles, labels = self._library()
        details = splits.stratified_scaffold_split(
            smiles, labels, test_size=0.2
        ).details
        assert details["stratified"] is True
        assert details["n_scaffolds_class_1"] == 20
        assert details["n_scaffolds_class_0"] == 200

    def test_honours_the_test_size_within_each_class(self) -> None:
        smiles, labels = self._library()
        split = splits.stratified_scaffold_split(smiles, labels, test_size=0.3)
        for value in (0, 1):
            rows = np.flatnonzero(labels == value)
            in_test = np.isin(split.test_index, rows).sum()
            assert in_test / rows.size == pytest.approx(0.3, abs=0.1)

    def test_single_class_behaves_like_a_scaffold_split(self) -> None:
        smiles, _ = self._library()
        labels = np.ones(len(smiles), dtype=int)
        split = splits.stratified_scaffold_split(smiles, labels, test_size=0.2)
        assert split.n_test > 0
        assert split.n_train > 0

    def test_rejects_misaligned_labels(self) -> None:
        with pytest.raises(ValueError, match="must align"):
            splits.stratified_scaffold_split(["C1CC1", "C1CCC1"], np.array([1]))
