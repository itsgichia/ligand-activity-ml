"""Tests for the property-matched decoy construction.

A decoy benchmark is only worth reporting if the decoys are genuinely
hard: matched on the bulk properties a model could otherwise exploit, and
far enough from the actives that none of them is a mislabelled analogue.
Both of those are what these tests check, because a badly built decoy set
produces a flattering enrichment factor that looks exactly like a real
one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ligandml import decoys


class TestStandardisedBias:
    """The number that says whether the matching worked."""

    def test_identical_groups_have_no_bias(self) -> None:
        block = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        assert decoys.standardised_bias(block, block) == pytest.approx(0.0)

    def test_offset_groups_are_biased(self) -> None:
        actives = np.array([[1.0], [2.0], [3.0]])
        far = np.array([[11.0], [12.0], [13.0]])
        assert decoys.standardised_bias(actives, far) > 1.0

    def test_empty_group_is_zero(self) -> None:
        assert decoys.standardised_bias(np.empty((0, 2)), np.ones((3, 2))) == 0.0

    def test_constant_property_does_not_divide_by_zero(self) -> None:
        actives = np.array([[5.0], [5.0]])
        pool = np.array([[5.0], [5.0]])
        assert decoys.standardised_bias(actives, pool) == pytest.approx(0.0)


class TestSelectDecoys:
    """Matching on properties, refusing anything that looks like an analogue."""

    def test_picks_the_nearest_candidates_in_property_space(self) -> None:
        actives = np.array([[0.0, 0.0]])
        pool = np.array([[0.1, 0.1], [5.0, 5.0], [0.2, 0.2]])
        picked = decoys.select_decoys(
            actives, pool, np.zeros(3), ratio=2, max_similarity=0.35
        )
        assert set(picked.tolist()) == {0, 2}

    def test_refuses_candidates_too_similar_to_an_active(self) -> None:
        """A close analogue is disqualified however well its properties fit."""
        actives = np.array([[0.0]])
        pool = np.array([[0.01], [9.0]])
        similarity = np.array([0.9, 0.0])  # the near one is an analogue
        picked = decoys.select_decoys(
            actives, pool, similarity, ratio=1, max_similarity=0.35
        )
        assert picked.tolist() == [1]

    def test_no_candidate_is_used_twice(self) -> None:
        actives = np.array([[0.0], [0.05]])
        pool = np.array([[0.0], [0.1], [0.2], [0.3]])
        picked = decoys.select_decoys(
            actives, pool, np.zeros(4), ratio=2, max_similarity=0.35
        )
        assert len(picked) == len(set(picked.tolist())) == 4

    def test_stops_when_the_pool_runs_out(self) -> None:
        actives = np.array([[0.0]])
        pool = np.array([[1.0], [2.0]])
        picked = decoys.select_decoys(
            actives, pool, np.zeros(2), ratio=10, max_similarity=0.35
        )
        assert len(picked) == 2

    def test_all_candidates_refused_returns_empty(self) -> None:
        picked = decoys.select_decoys(
            np.array([[0.0]]), np.array([[1.0]]), np.array([0.99]), ratio=1
        )
        assert picked.size == 0

    def test_actives_take_turns_rather_than_one_monopolising(self) -> None:
        """Dealing round-robin stops the first active claiming every match."""
        actives = np.array([[0.0], [100.0]])
        pool = np.array([[0.1], [0.2], [100.1], [100.2]])
        picked = decoys.select_decoys(
            actives, pool, np.zeros(4), ratio=1, max_similarity=0.35
        )
        # One near each active, not two near the first.
        assert sorted(picked.tolist()) == [0, 2]

    def test_rejects_bad_ratio(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            decoys.select_decoys(np.array([[0.0]]), np.array([[1.0]]), np.zeros(1), 0)

    def test_rejects_mismatched_property_columns(self) -> None:
        with pytest.raises(ValueError, match="share property columns"):
            decoys.select_decoys(
                np.array([[0.0, 1.0]]), np.array([[1.0]]), np.zeros(1)
            )

    def test_rejects_mismatched_similarity_length(self) -> None:
        with pytest.raises(ValueError, match="one similarity per"):
            decoys.select_decoys(
                np.array([[0.0]]), np.array([[1.0], [2.0]]), np.zeros(1)
            )


class TestBuildDecoySet:
    """The assembled benchmark and its construction summary."""

    def _inputs(self, n_pool: int = 40):
        rng = np.random.default_rng(0)
        actives = rng.normal(300.0, 30.0, size=(4, 2))
        pool = rng.normal(300.0, 30.0, size=(n_pool, 2))
        return (
            [f"A{i}" for i in range(4)],
            actives,
            [f"D{i}" for i in range(n_pool)],
            pool,
            np.zeros(n_pool),
        )

    def test_labels_and_counts(self) -> None:
        a_smiles, a_props, p_smiles, p_props, sim = self._inputs()
        built = decoys.build_decoy_set(
            "herg", a_smiles, a_props, p_smiles, p_props, sim, ratio=5
        )
        assert built.n_actives == 4
        assert built.n_decoys == 20
        assert built.labels.sum() == 4
        assert len(built.smiles) == 24
        assert list(built.labels[:4]) == [1, 1, 1, 1]

    def test_active_rate_and_ceiling(self) -> None:
        a_smiles, a_props, p_smiles, p_props, sim = self._inputs()
        built = decoys.build_decoy_set(
            "herg", a_smiles, a_props, p_smiles, p_props, sim, ratio=9
        )
        # 4 actives, 36 decoys -> 10% active, ceiling 1/0.1 = 10.
        assert built.active_rate == pytest.approx(0.1)
        assert built.enrichment_ceiling_1pct == pytest.approx(10.0)

    def test_matching_keeps_the_bias_small(self) -> None:
        """Drawn from one distribution, the two groups should be alike."""
        a_smiles, a_props, p_smiles, p_props, sim = self._inputs(n_pool=200)
        built = decoys.build_decoy_set(
            "herg", a_smiles, a_props, p_smiles, p_props, sim, ratio=10
        )
        assert built.property_bias < 0.5

    def test_records_an_exhausted_pool(self) -> None:
        a_smiles, a_props, p_smiles, p_props, sim = self._inputs(n_pool=5)
        built = decoys.build_decoy_set(
            "herg", a_smiles, a_props, p_smiles, p_props, sim, ratio=10
        )
        assert built.pool_exhausted is True
        assert built.n_decoys == 5

    def test_ceiling_is_capped_at_the_top_one_percent(self) -> None:
        """At 1% screening depth no model can exceed an EF of 100."""
        a_smiles, a_props, p_smiles, p_props, sim = self._inputs(n_pool=1000)
        built = decoys.build_decoy_set(
            "herg", a_smiles, a_props, p_smiles, p_props, sim, ratio=250
        )
        assert built.enrichment_ceiling_1pct == pytest.approx(100.0)

    def test_requires_actives(self) -> None:
        with pytest.raises(ValueError, match="at least one active"):
            decoys.build_decoy_set(
                "herg", [], np.empty((0, 2)), ["D"], np.ones((1, 2)), np.zeros(1)
            )

    def test_as_row_is_flat(self) -> None:
        a_smiles, a_props, p_smiles, p_props, sim = self._inputs()
        row = decoys.build_decoy_set(
            "herg", a_smiles, a_props, p_smiles, p_props, sim, ratio=5
        ).as_row()
        assert row["target"] == "herg"
        assert row["achieved_ratio"] == 5.0
        assert all(not isinstance(v, (list, dict, np.ndarray)) for v in row.values())


class TestCrossTargetPool:
    """Candidates come from the other targets, never from this one."""

    def _frames(self) -> dict[str, pd.DataFrame]:
        return {
            "herg": pd.DataFrame({"canonical_smiles": ["CC", "CCC"]}),
            "d2": pd.DataFrame({"canonical_smiles": ["CCC", "CCCC"]}),
            "ache": pd.DataFrame({"canonical_smiles": ["CCCCC"]}),
        }

    def test_excludes_the_targets_own_compounds(self) -> None:
        pool = decoys.cross_target_pool(self._frames(), "herg")
        assert set(pool["canonical_smiles"]) == {"CCCC", "CCCCC"}

    def test_deduplicates_across_sources(self) -> None:
        frames = self._frames()
        frames["adrb2"] = pd.DataFrame({"canonical_smiles": ["CCCC"]})
        pool = decoys.cross_target_pool(frames, "herg")
        assert list(pool["canonical_smiles"]).count("CCCC") == 1

    def test_records_where_each_candidate_came_from(self) -> None:
        pool = decoys.cross_target_pool(self._frames(), "herg")
        assert set(pool["source_target"]) <= {"d2", "ache"}

    def test_single_target_gives_an_empty_pool(self) -> None:
        pool = decoys.cross_target_pool(
            {"herg": pd.DataFrame({"canonical_smiles": ["CC"]})}, "herg"
        )
        assert pool.empty

    def test_unknown_target(self) -> None:
        with pytest.raises(KeyError, match="no frame for target"):
            decoys.cross_target_pool(self._frames(), "nope")


class TestSignedBiasByProperty:
    """The breakdown that names *which* property the matching failed on."""

    def test_sign_follows_the_actives(self) -> None:
        actives = np.array([[10.0], [10.0]])
        decoys_ = np.array([[1.0], [1.0]])
        bias = decoys.signed_bias_by_property(actives, decoys_, ["MolWt"])
        assert bias["MolWt"] > 0
        assert decoys.signed_bias_by_property(decoys_, actives, ["MolWt"])["MolWt"] < 0

    def test_names_every_property(self) -> None:
        block = np.array([[1.0, 2.0, 3.0]])
        bias = decoys.signed_bias_by_property(block, block, ["a", "b", "c"])
        assert set(bias) == {"a", "b", "c"}

    def test_empty_group_gives_nothing(self) -> None:
        assert decoys.signed_bias_by_property(np.empty((0, 1)), np.ones((2, 1))) == {}

    def test_rejects_wrong_name_count(self) -> None:
        with pytest.raises(ValueError, match="expected 2 names"):
            decoys.signed_bias_by_property(np.ones((2, 2)), np.ones((2, 2)), ["only"])

    def test_isolates_the_offending_property(self) -> None:
        """One property far apart, one identical: only the first shows."""
        actives = np.array([[0.0, 5.0], [1.0, 5.0]])
        decoys_ = np.array([[9.0, 5.0], [10.0, 5.0]])
        bias = decoys.signed_bias_by_property(actives, decoys_, ["far", "same"])
        assert abs(bias["far"]) > 1.0
        assert bias["same"] == pytest.approx(0.0)


class TestDistanceCap:
    """Refusing a match the pool cannot actually supply."""

    def test_distant_candidate_is_refused(self) -> None:
        actives = np.array([[0.0], [0.0]])
        pool = np.array([[0.0], [50.0]])
        match = decoys.match_decoys(
            actives, pool, np.zeros(2), ratio=1, max_distance=0.5
        )
        assert match.indices.tolist() == [0]
        assert match.counts.tolist() == [1, 0]

    def test_without_a_cap_the_distant_one_is_taken(self) -> None:
        actives = np.array([[0.0], [0.0]])
        pool = np.array([[0.0], [50.0]])
        match = decoys.match_decoys(actives, pool, np.zeros(2), ratio=1)
        assert sorted(match.indices.tolist()) == [0, 1]

    def test_owner_attributes_each_decoy(self) -> None:
        actives = np.array([[0.0], [100.0]])
        pool = np.array([[0.1], [100.1]])
        match = decoys.match_decoys(actives, pool, np.zeros(2), ratio=1)
        assert len(match.owner) == len(match.indices)
        assert sorted(match.owner.tolist()) == [0, 1]

    def test_counts_never_exceed_the_ratio(self) -> None:
        actives = np.array([[0.0], [0.1]])
        pool = np.array([[0.0], [0.1], [0.2], [0.3], [0.4]])
        match = decoys.match_decoys(actives, pool, np.zeros(5), ratio=2)
        assert match.counts.tolist() == [2, 2]

    def test_select_decoys_still_returns_bare_indices(self) -> None:
        picked = decoys.select_decoys(
            np.array([[0.0]]), np.array([[0.1], [0.2]]), np.zeros(2), ratio=2
        )
        assert isinstance(picked, np.ndarray)
        assert picked.size == 2


class TestDroppingUnmatchableActives:
    """An active the pool cannot disguise is removed, not given a free pass."""

    def _build(self, max_distance=None):
        # Two actives in one region, one far outlier the pool cannot match.
        active_props = np.array([[0.0], [0.2], [500.0]])
        pool_props = np.array([[0.0], [0.1], [0.2], [0.3]])
        return decoys.build_decoy_set(
            "herg",
            ["A0", "A1", "OUTLIER"],
            active_props,
            ["D0", "D1", "D2", "D3"],
            pool_props,
            np.zeros(4),
            ratio=1,
            max_distance=max_distance,
        )

    def test_outlier_is_dropped_under_a_cap(self) -> None:
        built = self._build(max_distance=0.5)
        assert built.n_actives == 2
        assert built.n_actives_requested == 3
        assert built.n_actives_dropped == 1
        assert "OUTLIER" not in built.smiles

    def test_outlier_survives_without_a_cap(self) -> None:
        built = self._build()
        assert built.n_actives == 3
        assert built.n_actives_dropped == 0

    def test_dropping_takes_the_decoys_with_it(self) -> None:
        """A dropped active must not leave its decoys behind."""
        built = self._build(max_distance=0.5)
        assert built.n_decoys == 2
        assert len(built.smiles) == built.n_actives + built.n_decoys
        assert len(built.labels) == len(built.smiles)

    def test_labels_still_line_up_after_dropping(self) -> None:
        built = self._build(max_distance=0.5)
        assert built.labels[: built.n_actives].sum() == built.n_actives
        assert built.labels[built.n_actives :].sum() == 0

    def test_min_decoys_per_active_drops_the_underserved(self) -> None:
        built = decoys.build_decoy_set(
            "herg",
            ["A0", "A1"],
            np.array([[0.0], [0.2]]),
            ["D0"],
            np.array([[0.1]]),
            np.zeros(1),
            ratio=1,
            min_decoys_per_active=1,
        )
        # Only one candidate exists, so only one active can be served.
        assert built.n_actives == 1
        assert built.n_actives_dropped == 1

    def test_dropping_improves_the_reported_bias(self) -> None:
        """The cap exists to make this number honest, so check it moves."""
        assert self._build(max_distance=0.5).property_bias < self._build().property_bias


class TestPoolDiagnostics:
    """Reporting how much room the matcher actually had."""

    def test_pool_fraction_used(self) -> None:
        built = decoys.build_decoy_set(
            "herg",
            ["A"],
            np.array([[0.0]]),
            [f"D{i}" for i in range(10)],
            np.arange(10.0).reshape(-1, 1),
            np.zeros(10),
            ratio=5,
        )
        assert built.pool_size == 10
        assert built.pool_fraction_used == pytest.approx(0.5)

    def test_empty_pool_fraction_is_zero(self) -> None:
        built = decoys.build_decoy_set(
            "herg", ["A"], np.array([[0.0]]), [], np.empty((0, 1)), np.zeros(0)
        )
        assert built.pool_fraction_used == 0.0

    def test_row_names_the_worst_property(self) -> None:
        actives = np.array([[0.0, 0.0], [0.0, 1.0]])
        pool = np.array([[9.0, 0.0], [9.5, 1.0]])
        row = decoys.build_decoy_set(
            "herg",
            ["A0", "A1"],
            actives,
            ["D0", "D1"],
            pool,
            np.zeros(2),
            ratio=1,
            property_names=["heavy", "light"],
        ).as_row()
        assert row["worst_property"] == "heavy"
        assert row["pool_size"] == 2
