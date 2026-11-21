"""Tests for the ChEMBL fetch, cleaning and caching pipeline.

The cleaning stage is where a silent bug would do the most damage: a
censored value treated as exact, or a pChEMBL conversion off by a factor of
1000, would corrupt every downstream number while leaving the pipeline
apparently healthy. These tests pin each rule against hand-computed
answers.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from ligandml import chembl


class TestPchemblConversion:
    """pChEMBL is -log10 of the molar potency; nM is 1e-9 M."""

    @pytest.mark.parametrize(
        ("value_nm", "expected"),
        [
            (1.0, 9.0),  # 1 nM
            (10.0, 8.0),
            (1000.0, 6.0),  # 1 uM
            (1_000_000.0, 3.0),  # 1 mM
            (0.1, 10.0),
        ],
    )
    def test_known_values(self, value_nm: float, expected: float) -> None:
        assert chembl.pchembl_from_nm(value_nm) == pytest.approx(expected)

    @pytest.mark.parametrize("value_nm", [0.0, -1.0, -1e-9])
    def test_non_positive_rejected(self, value_nm: float) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            chembl.pchembl_from_nm(value_nm)

    @pytest.mark.parametrize("value_nm", [0.5, 7.3, 42.0, 9999.0])
    def test_round_trip(self, value_nm: float) -> None:
        back = chembl.nm_from_pchembl(chembl.pchembl_from_nm(value_nm))
        assert back == pytest.approx(value_nm)

    def test_matches_chembl_convention(self) -> None:
        """A 50 nM IC50 is pChEMBL 7.3, the value ChEMBL itself reports."""
        assert chembl.pchembl_from_nm(50.0) == pytest.approx(7.3010, abs=1e-4)


class TestStandardiseSmiles:
    """Structure standardisation: desalt, canonicalise, size-filter."""

    def test_salt_reduces_to_parent(self) -> None:
        parent = chembl.standardise_smiles("Cc1ccccc1")
        salt = chembl.standardise_smiles("Cc1ccccc1.Cl")
        assert parent == salt != None  # noqa: E711 - identity with a value

    def test_sodium_salt_reduces_to_parent(self) -> None:
        acid = chembl.standardise_smiles("CC(=O)Oc1ccccc1C(=O)O")
        sodium = chembl.standardise_smiles("CC(=O)Oc1ccccc1C(=O)[O-].[Na+]")
        assert acid == sodium

    def test_canonical_form_is_stable(self) -> None:
        """Two SMILES of one molecule standardise to the same string."""
        assert chembl.standardise_smiles("c1ccccc1C") == chembl.standardise_smiles(
            "Cc1ccccc1"
        )

    @pytest.mark.parametrize(
        "smiles", ["", None, "not-smiles", "C(C(C", "CCO", "C", "[Na+]"]
    )
    def test_rejected(self, smiles: Any) -> None:
        assert chembl.standardise_smiles(smiles) is None

    def test_oversized_rejected(self) -> None:
        huge = "C" * 200
        assert chembl.standardise_smiles(huge) is None

    def test_drug_sized_accepted(self) -> None:
        assert chembl.standardise_smiles("CC(=O)Oc1ccccc1C(=O)O") is not None


class TestCleanActivities:
    """The ordered cleaning policy, against a hand-built batch."""

    def test_expected_survivors(self, raw_activities: list[dict[str, Any]]) -> None:
        """Only the aspirin and toluene structures survive."""
        frame, _ = chembl.clean_activities(raw_activities)
        assert len(frame) == 2
        assert set(frame["molecule_chembl_id"]) == {"CHEMBL_A", "CHEMBL_B"}

    def test_censored_rows_dropped(self, raw_activities: list[dict[str, Any]]) -> None:
        frame, attrition = chembl.clean_activities(raw_activities)
        assert "CHEMBL_C" not in set(frame["molecule_chembl_id"])
        step = next(s for s in attrition if "censored" in s.step)
        assert step.removed == 1

    @pytest.mark.parametrize("relation", [">", "<", ">=", "<=", "~", ""])
    def test_every_censoring_relation_dropped(
        self, raw_activities: list[dict[str, Any]], relation: str
    ) -> None:
        record = dict(raw_activities[2])
        record["standard_relation"] = relation
        frame, _ = chembl.clean_activities([record])
        assert frame.empty

    def test_flagged_rows_dropped(self, raw_activities: list[dict[str, Any]]) -> None:
        frame, attrition = chembl.clean_activities(raw_activities)
        assert not {"CHEMBL_D", "CHEMBL_E"} & set(frame["molecule_chembl_id"])
        step = next(s for s in attrition if "duplicate flag" in s.step)
        assert step.removed == 2

    def test_out_of_range_dropped(self, raw_activities: list[dict[str, Any]]) -> None:
        frame, attrition = chembl.clean_activities(raw_activities)
        assert "CHEMBL_G" not in set(frame["molecule_chembl_id"])
        step = next(s for s in attrition if "outside" in s.step)
        assert step.removed == 1

    def test_unparseable_and_tiny_dropped(
        self, raw_activities: list[dict[str, Any]]
    ) -> None:
        frame, attrition = chembl.clean_activities(raw_activities)
        assert not {"CHEMBL_H", "CHEMBL_I"} & set(frame["molecule_chembl_id"])
        step = next(s for s in attrition if "parse failure" in s.step)
        assert step.removed == 2

    def test_missing_dropped(self, raw_activities: list[dict[str, Any]]) -> None:
        frame, attrition = chembl.clean_activities(raw_activities)
        assert "CHEMBL_F" not in set(frame["molecule_chembl_id"])
        step = next(s for s in attrition if "missing SMILES" in s.step)
        assert step.removed == 1

    def test_dedupe_takes_median(self, raw_activities: list[dict[str, Any]]) -> None:
        """100 nM (pChEMBL 7) and 10 nM (pChEMBL 8) merge to 7.5."""
        frame, _ = chembl.clean_activities(raw_activities)
        row = frame.set_index("molecule_chembl_id").loc["CHEMBL_A"]
        assert row["pchembl_value"] == pytest.approx(7.5)
        assert row["n_measurements"] == 2
        assert row["n_assays"] == 2

    def test_dedupe_merges_salt_with_parent(
        self, raw_activities: list[dict[str, Any]]
    ) -> None:
        """Toluene and toluene hydrochloride are one compound, not two."""
        frame, _ = chembl.clean_activities(raw_activities)
        row = frame.set_index("molecule_chembl_id").loc["CHEMBL_B"]
        assert row["n_measurements"] == 2
        assert row["pchembl_value"] == pytest.approx(6.0)

    def test_merged_nm_agrees_with_merged_pchembl(
        self, raw_activities: list[dict[str, Any]]
    ) -> None:
        frame, _ = chembl.clean_activities(raw_activities)
        for _, row in frame.iterrows():
            assert row["standard_value_nm"] == pytest.approx(
                chembl.nm_from_pchembl(row["pchembl_value"])
            )

    def test_first_year_is_the_earliest(
        self, raw_activities: list[dict[str, Any]]
    ) -> None:
        """Aspirin was measured in 2010 and 2004; first_year must be 2004."""
        frame, _ = chembl.clean_activities(raw_activities)
        row = frame.set_index("molecule_chembl_id").loc["CHEMBL_A"]
        assert row["first_year"] == 2004

    def test_dedupe_is_order_independent(
        self, raw_activities: list[dict[str, Any]]
    ) -> None:
        """Shuffling the input must not change the cleaned table."""
        forward, _ = chembl.clean_activities(raw_activities)
        backward, _ = chembl.clean_activities(list(reversed(raw_activities)))
        columns = ["canonical_smiles", "pchembl_value", "n_measurements"]
        pd.testing.assert_frame_equal(
            forward.sort_values("canonical_smiles")[columns].reset_index(drop=True),
            backward.sort_values("canonical_smiles")[columns].reset_index(drop=True),
        )

    def test_dedupe_is_idempotent(self, raw_activities: list[dict[str, Any]]) -> None:
        """Cleaning twice changes nothing: duplicating the batch must not
        duplicate compounds or shift the medians.
        """
        once, _ = chembl.clean_activities(raw_activities)
        twice, _ = chembl.clean_activities(raw_activities + raw_activities)
        assert len(once) == len(twice)
        assert once["pchembl_value"].tolist() == pytest.approx(
            twice["pchembl_value"].tolist()
        )

    def test_attrition_table_is_internally_consistent(
        self, raw_activities: list[dict[str, Any]]
    ) -> None:
        """Each step's remaining count must equal the previous one minus its
        own removals, so the table in the README cannot drift from reality.
        """
        _, attrition = chembl.clean_activities(raw_activities)
        filters = attrition[: attrition.index(
            next(s for s in attrition if s.step == "Exact measurements retained")
        )]
        for previous, current in zip(filters, filters[1:], strict=False):
            assert current.remaining == previous.remaining - current.removed
        assert filters[0].remaining == len(raw_activities)

    def test_empty_input(self) -> None:
        frame, attrition = chembl.clean_activities([])
        assert frame.empty
        assert list(frame.columns) == list(chembl.OUTPUT_COLUMNS)
        assert attrition[-1].remaining == 0

    def test_output_columns(self, raw_activities: list[dict[str, Any]]) -> None:
        frame, _ = chembl.clean_activities(raw_activities)
        assert list(frame.columns) == list(chembl.OUTPUT_COLUMNS)

    def test_pchembl_recomputed_not_trusted(self) -> None:
        """ChEMBL's own pchembl_value is ignored in favour of recomputation,
        so one wrong upstream value cannot poison the dataset.
        """
        record = {
            "molecule_chembl_id": "CHEMBL_X",
            "canonical_smiles": "Cc1ccccc1",
            "standard_type": "IC50",
            "standard_relation": "=",
            "standard_value": "1000.0",
            "standard_units": "nM",
            "pchembl_value": "3.0",  # Deliberately wrong.
            "assay_chembl_id": "A",
            "document_chembl_id": "D",
            "document_year": 2000,
            "potential_duplicate": 0,
            "data_validity_comment": None,
            "target_chembl_id": "T",
        }
        frame, _ = chembl.clean_activities([record])
        assert frame["pchembl_value"].iat[0] == pytest.approx(6.0)


class TestTargetSpecs:
    """The declared study targets."""

    def test_slugs_and_ids_unique(self) -> None:
        assert len({t.name for t in chembl.TARGETS}) == len(chembl.TARGETS)
        assert len({t.chembl_id for t in chembl.TARGETS}) == len(chembl.TARGETS)

    def test_count_is_in_the_promised_range(self) -> None:
        assert 3 <= len(chembl.TARGETS) <= 5

    def test_every_target_has_a_rationale(self) -> None:
        for spec in chembl.TARGETS:
            assert spec.rationale.strip()
            assert spec.chembl_id.startswith("CHEMBL")


class TestCache:
    """Round-tripping a cleaned dataset through the cache."""

    def test_write_and_read(
        self, tmp_path: Path, raw_activities: list[dict[str, Any]]
    ) -> None:
        frame, attrition = chembl.clean_activities(raw_activities)
        csv_path, meta_path = chembl.dataset_paths("demo", tmp_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(csv_path, index=False)
        report = chembl._build_report(
            chembl.TargetSpec("CHEMBL_T", "demo", "Demo target", "testing"),
            frame,
            attrition,
            n_raw=len(raw_activities),
            n_exact=4,
            n_documents=1,
        )
        meta_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

        loaded = chembl.load_target("demo", tmp_path)
        assert len(loaded) == len(frame)
        meta = chembl.load_meta("demo", tmp_path)
        assert meta["target_chembl_id"] == "CHEMBL_T"
        assert meta["n_compounds"] == len(frame)
        assert meta["met_threshold"] is False  # Two compounds, threshold 800.

    def test_missing_cache_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="no cached dataset"):
            chembl.load_target("absent", tmp_path)

    def test_below_threshold_target_is_dropped(
        self, tmp_path: Path, raw_activities: list[dict[str, Any]]
    ) -> None:
        """A target that falls short must be excluded, not quietly used."""
        frame, attrition = chembl.clean_activities(raw_activities)
        spec = chembl.TargetSpec("CHEMBL_T", "demo", "Demo target", "testing")
        csv_path, meta_path = chembl.dataset_paths("demo", tmp_path)
        frame.to_csv(csv_path, index=False)
        report = chembl._build_report(spec, frame, attrition, 11, 4, 1)
        meta_path.write_text(json.dumps(report.to_dict()), encoding="utf-8")
        assert chembl.available_targets(tmp_path, specs=[spec]) == []


class TestCachedDatasets:
    """Checks against the committed cache, skipped on a fresh clone."""

    def test_cache_matches_metadata(self, cached_targets: list[str]) -> None:
        if not cached_targets:
            pytest.skip("no cached datasets; run python -m ligandml.chembl --fetch-all")
        from tests.conftest import DATA_DIR

        for name in cached_targets:
            frame = chembl.load_target(name, DATA_DIR)
            meta = chembl.load_meta(name, DATA_DIR)
            assert len(frame) == meta["n_compounds"]
            assert list(frame.columns) == list(chembl.OUTPUT_COLUMNS)

    def test_cache_has_no_censored_or_duplicate_rows(
        self, cached_targets: list[str]
    ) -> None:
        if not cached_targets:
            pytest.skip("no cached datasets")
        from tests.conftest import DATA_DIR

        for name in cached_targets:
            frame = chembl.load_target(name, DATA_DIR)
            assert frame["canonical_smiles"].is_unique
            assert frame["pchembl_value"].between(1.0, 14.0).all()
            assert (frame["n_measurements"] >= 1).all()

    def test_cache_pchembl_matches_nm(self, cached_targets: list[str]) -> None:
        if not cached_targets:
            pytest.skip("no cached datasets")
        from tests.conftest import DATA_DIR

        for name in cached_targets:
            frame = chembl.load_target(name, DATA_DIR)
            recomputed = frame["standard_value_nm"].map(chembl.pchembl_from_nm)
            assert (recomputed - frame["pchembl_value"]).abs().max() < 1e-6

    def test_usable_targets_clear_the_threshold(
        self, cached_targets: list[str]
    ) -> None:
        if not cached_targets:
            pytest.skip("no cached datasets")
        from tests.conftest import DATA_DIR

        usable = chembl.available_targets(DATA_DIR)
        assert len(usable) >= 3, "the study needs at least 3 usable targets"
        for spec in usable:
            frame = chembl.load_target(spec.name, DATA_DIR)
            assert len(frame) >= chembl.MIN_COMPOUNDS

    def test_years_are_plausible(self, cached_targets: list[str]) -> None:
        if not cached_targets:
            pytest.skip("no cached datasets")
        from tests.conftest import DATA_DIR

        for name in cached_targets:
            frame = chembl.load_target(name, DATA_DIR)
            years = frame.loc[frame["first_year"] > 0, "first_year"]
            assert years.between(1960, 2030).all()
            assert len(years) / len(frame) > 0.9, "most compounds need a year"


def test_pchembl_monotonic_in_potency() -> None:
    """More potent (smaller nM) must mean a larger pChEMBL."""
    values = [0.1, 1.0, 10.0, 100.0, 1000.0]
    pchembls = [chembl.pchembl_from_nm(v) for v in values]
    assert all(
        earlier > later for earlier, later in zip(pchembls, pchembls[1:], strict=False)
    )
    assert math.isclose(pchembls[0] - pchembls[-1], 4.0)
