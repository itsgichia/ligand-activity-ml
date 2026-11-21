"""Tests for the molecular representations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem

from ligandml import featurise


@pytest.fixture
def mols() -> list[Chem.Mol]:
    """A handful of small, well-known drug-like molecules."""
    smiles = [
        "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
        "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",  # caffeine
        "CC(C)Cc1ccc(C(C)C(=O)O)cc1",  # ibuprofen
        "Cc1ccccc1",  # toluene
    ]
    return [Chem.MolFromSmiles(s) for s in smiles]


class TestMolsFromSmiles:
    """Parsing, with failures reported rather than hidden."""

    def test_good_smiles_all_kept(self) -> None:
        parsed, keep = featurise.mols_from_smiles(["Cc1ccccc1", "CCO"])
        assert len(parsed) == 2
        assert keep.tolist() == [True, True]

    def test_bad_smiles_masked_out(self) -> None:
        parsed, keep = featurise.mols_from_smiles(["Cc1ccccc1", "nope", "CCO"])
        assert len(parsed) == 2
        assert keep.tolist() == [True, False, True]

    def test_non_string_masked_out(self) -> None:
        parsed, keep = featurise.mols_from_smiles([None, 3.5, "CCO"])
        assert len(parsed) == 1
        assert keep.tolist() == [False, False, True]


class TestRdkitDescriptors:
    """The dense, interpretable block."""

    def test_shape_and_names(self, mols: list[Chem.Mol]) -> None:
        matrix, names = featurise.rdkit_descriptors(mols)
        assert matrix.shape == (len(mols), len(featurise.DESCRIPTOR_NAMES))
        assert names == list(featurise.DESCRIPTOR_NAMES)

    def test_all_finite(self, mols: list[Chem.Mol]) -> None:
        matrix, _ = featurise.rdkit_descriptors(mols)
        assert np.isfinite(matrix).all()

    def test_known_values(self, mols: list[Chem.Mol]) -> None:
        """Aspirin: MW 180.16, 13 heavy atoms, 2 rotatable bonds, 1 ring."""
        matrix, names = featurise.rdkit_descriptors(mols[:1])
        lookup = dict(zip(names, matrix[0], strict=True))
        assert lookup["MolWt"] == pytest.approx(180.16, abs=0.01)
        assert lookup["HeavyAtomCount"] == 13
        assert lookup["RingCount"] == 1
        assert lookup["NumAromaticRings"] == 1

    def test_caffeine_has_no_rotatable_bonds(self, mols: list[Chem.Mol]) -> None:
        matrix, names = featurise.rdkit_descriptors([mols[1]])
        lookup = dict(zip(names, matrix[0], strict=True))
        assert lookup["NumRotatableBonds"] == 0
        assert lookup["RingCount"] == 2

    def test_deterministic(self, mols: list[Chem.Mol]) -> None:
        first, _ = featurise.rdkit_descriptors(mols)
        second, _ = featurise.rdkit_descriptors(mols)
        assert np.array_equal(first, second)

    def test_unknown_descriptor_rejected(self, mols: list[Chem.Mol]) -> None:
        with pytest.raises(KeyError, match="unknown RDKit descriptor"):
            featurise.rdkit_descriptors(mols, names=["NotADescriptor"])

    def test_excludes_the_expensive_ones(self) -> None:
        """BertzCT and qed are deliberately out; see the module docstring."""
        assert "BertzCT" not in featurise.DESCRIPTOR_NAMES
        assert "qed" not in featurise.DESCRIPTOR_NAMES


class TestMorganFingerprints:
    """Circular substructure fingerprints."""

    @pytest.mark.parametrize(("radius", "n_bits"), [(2, 1024), (2, 2048), (3, 2048)])
    def test_shape(self, mols: list[Chem.Mol], radius: int, n_bits: int) -> None:
        matrix, names = featurise.morgan_fingerprints(
            mols, radius=radius, n_bits=n_bits
        )
        assert matrix.shape == (len(mols), n_bits)
        assert len(names) == n_bits

    def test_binary(self, mols: list[Chem.Mol]) -> None:
        matrix, _ = featurise.morgan_fingerprints(mols)
        assert set(np.unique(matrix).tolist()) <= {0.0, 1.0}

    def test_identical_molecules_give_identical_bits(self) -> None:
        pair = [Chem.MolFromSmiles("Cc1ccccc1"), Chem.MolFromSmiles("c1ccccc1C")]
        matrix, _ = featurise.morgan_fingerprints(pair)
        assert np.array_equal(matrix[0], matrix[1])

    def test_different_molecules_differ(self, mols: list[Chem.Mol]) -> None:
        matrix, _ = featurise.morgan_fingerprints(mols)
        assert not np.array_equal(matrix[0], matrix[1])

    def test_larger_radius_sets_more_bits(self) -> None:
        big = Chem.MolFromSmiles("CC(C)Cc1ccc(C(C)C(=O)O)cc1")
        small, _ = featurise.morgan_fingerprints([big], radius=2)
        large, _ = featurise.morgan_fingerprints([big], radius=3)
        assert large.sum() > small.sum()

    def test_some_bits_are_set(self, mols: list[Chem.Mol]) -> None:
        matrix, _ = featurise.morgan_fingerprints(mols)
        assert (matrix.sum(axis=1) > 0).all()


class TestMaccsKeys:
    """The fixed, hand-curated key set."""

    def test_shape(self, mols: list[Chem.Mol]) -> None:
        matrix, names = featurise.maccs_keys(mols)
        assert matrix.shape == (len(mols), 167)
        assert len(names) == 167

    def test_padding_bit_never_set(self, mols: list[Chem.Mol]) -> None:
        matrix, _ = featurise.maccs_keys(mols)
        assert matrix[:, 0].sum() == 0.0

    def test_binary(self, mols: list[Chem.Mol]) -> None:
        matrix, _ = featurise.maccs_keys(mols)
        assert set(np.unique(matrix).tolist()) <= {0.0, 1.0}

    def test_aromatic_key_set_for_toluene(self, mols: list[Chem.Mol]) -> None:
        matrix, _ = featurise.maccs_keys([mols[3]])
        assert matrix.sum() > 0


class TestRepresentationRegistry:
    """The registry the study iterates over."""

    def test_declared_set(self) -> None:
        names = {r.name for r in featurise.REPRESENTATIONS}
        assert names == {
            "rdkit2d", "ecfp4_1024", "ecfp4_2048", "ecfp6_2048", "maccs"
        }

    def test_default_is_present(self) -> None:
        assert featurise.DEFAULT_REPRESENTATION in featurise.REPRESENTATIONS_BY_NAME

    def test_kinds_are_known(self) -> None:
        for representation in featurise.REPRESENTATIONS:
            assert representation.kind in ("descriptors", "fingerprint")

    def test_exactly_one_descriptor_block(self) -> None:
        kinds = [r.kind for r in featurise.REPRESENTATIONS]
        assert kinds.count("descriptors") == 1

    @pytest.mark.parametrize(
        "name", [r.name for r in featurise.REPRESENTATIONS]
    )
    def test_every_representation_runs(
        self, mols: list[Chem.Mol], name: str
    ) -> None:
        matrix, names = featurise.get_representation(name)(mols)
        assert matrix.shape[0] == len(mols)
        assert matrix.shape[1] == len(names)
        assert np.isfinite(matrix).all()

    def test_unknown_name(self) -> None:
        with pytest.raises(KeyError, match="unknown representation"):
            featurise.get_representation("quantum")


class TestFeaturiseFrame:
    """The dataframe entry point, including the row-dropping contract."""

    def test_rows_align(self) -> None:
        frame = pd.DataFrame(
            {"canonical_smiles": ["Cc1ccccc1", "CCO"], "pchembl_value": [6.0, 7.0]}
        )
        matrix, _, kept = featurise.featurise_frame(frame, "maccs")
        assert matrix.shape[0] == len(kept) == 2
        assert kept["pchembl_value"].tolist() == [6.0, 7.0]

    def test_bad_rows_dropped_from_both(self) -> None:
        frame = pd.DataFrame(
            {
                "canonical_smiles": ["Cc1ccccc1", "not-smiles", "CCO"],
                "pchembl_value": [6.0, 7.0, 8.0],
            }
        )
        matrix, _, kept = featurise.featurise_frame(frame, "maccs")
        assert matrix.shape[0] == len(kept) == 2
        assert kept["pchembl_value"].tolist() == [6.0, 8.0]

    def test_index_is_reset(self) -> None:
        frame = pd.DataFrame(
            {"canonical_smiles": ["bad", "Cc1ccccc1"], "pchembl_value": [1.0, 2.0]}
        )
        _, _, kept = featurise.featurise_frame(frame, "maccs")
        assert kept.index.tolist() == [0]

    def test_missing_column(self) -> None:
        with pytest.raises(KeyError, match="no column"):
            featurise.featurise_frame(pd.DataFrame({"x": [1]}), "maccs")

    def test_accepts_representation_object(self) -> None:
        frame = pd.DataFrame({"canonical_smiles": ["Cc1ccccc1"]})
        representation = featurise.get_representation("maccs")
        matrix, _, _ = featurise.featurise_frame(frame, representation)
        assert matrix.shape == (1, 167)
