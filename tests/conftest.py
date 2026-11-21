"""Shared fixtures for the test suite.

The fixtures deliberately avoid the network and the cached datasets: every
test here runs against small, hand-written inputs whose correct answers can
be worked out by hand, so a failure localises a bug rather than merely
reporting that something changed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"


@pytest.fixture
def raw_activities() -> list[dict[str, Any]]:
    """Return a hand-built batch of raw ChEMBL-shaped activity records.

    The batch is constructed so that each cleaning rule fires exactly once
    and the expected survivor count is known:

    * ``CHEMBL_A`` - two exact measurements of the same structure at 100 nM
      and 10 nM, which must merge to the median pChEMBL of 7.5.
    * ``CHEMBL_B`` - one exact measurement at 1000 nM (pChEMBL 6.0).
    * ``CHEMBL_C`` - censored (``>``), must be dropped.
    * ``CHEMBL_D`` - ChEMBL duplicate flag, must be dropped.
    * ``CHEMBL_E`` - data-validity comment, must be dropped.
    * ``CHEMBL_F`` - missing SMILES, must be dropped.
    * ``CHEMBL_G`` - potency far outside the plausible window, dropped.
    * ``CHEMBL_H`` - unparseable SMILES, must be dropped.
    * ``CHEMBL_I`` - too small (ethanol), must be dropped.
    * ``CHEMBL_J`` - the hydrochloride salt of ``CHEMBL_B``'s structure,
      which must desalt onto the same canonical SMILES and merge with it.
    """
    base = {
        "standard_type": "IC50",
        "standard_relation": "=",
        "standard_units": "nM",
        "potential_duplicate": 0,
        "data_validity_comment": None,
        "target_chembl_id": "CHEMBL_T",
        "assay_chembl_id": "CHEMBL_ASSAY1",
        "document_chembl_id": "CHEMBL_DOC1",
        "document_year": 2010,
        "pchembl_value": None,
    }
    aspirin = "CC(=O)Oc1ccccc1C(=O)O"
    toluene = "Cc1ccccc1"
    return [
        {**base, "molecule_chembl_id": "CHEMBL_A", "canonical_smiles": aspirin,
         "standard_value": "100.0"},
        {**base, "molecule_chembl_id": "CHEMBL_A", "canonical_smiles": aspirin,
         "standard_value": "10.0", "document_year": 2004,
         "assay_chembl_id": "CHEMBL_ASSAY2"},
        {**base, "molecule_chembl_id": "CHEMBL_B", "canonical_smiles": toluene,
         "standard_value": "1000.0"},
        {**base, "molecule_chembl_id": "CHEMBL_C", "canonical_smiles": "c1ccncc1",
         "standard_value": "10000.0", "standard_relation": ">"},
        {**base, "molecule_chembl_id": "CHEMBL_D", "canonical_smiles": "c1ccsc1",
         "standard_value": "50.0", "potential_duplicate": 1},
        {**base, "molecule_chembl_id": "CHEMBL_E", "canonical_smiles": "c1cc[nH]c1",
         "standard_value": "50.0", "data_validity_comment": "Outside typical range"},
        {**base, "molecule_chembl_id": "CHEMBL_F", "canonical_smiles": None,
         "standard_value": "50.0"},
        {**base, "molecule_chembl_id": "CHEMBL_G", "canonical_smiles": "c1ccoc1",
         "standard_value": "1e12"},
        {**base, "molecule_chembl_id": "CHEMBL_H",
         "canonical_smiles": "this-is-not-smiles", "standard_value": "50.0"},
        {**base, "molecule_chembl_id": "CHEMBL_I", "canonical_smiles": "CCO",
         "standard_value": "50.0"},
        {**base, "molecule_chembl_id": "CHEMBL_J",
         "canonical_smiles": f"{toluene}.Cl", "standard_value": "1000.0",
         "document_year": 2015},
    ]


@pytest.fixture
def congeneric_smiles() -> list[str]:
    """Return SMILES forming three clear Bemis-Murcko scaffold families.

    Four para-substituted benzenes, four substituted naphthalenes and four
    substituted pyridines: twelve compounds, three ring systems, so a
    scaffold split has a well-defined right answer.
    """
    benzenes = ["Cc1ccccc1", "CCc1ccccc1", "CCCc1ccccc1", "OCc1ccccc1"]
    naphthalenes = [
        "Cc1ccc2ccccc2c1",
        "CCc1ccc2ccccc2c1",
        "CCCc1ccc2ccccc2c1",
        "OCc1ccc2ccccc2c1",
    ]
    pyridines = ["Cc1ccncc1", "CCc1ccncc1", "CCCc1ccncc1", "OCc1ccncc1"]
    return benzenes + naphthalenes + pyridines


@pytest.fixture
def cached_targets() -> list[str]:
    """Return the slugs of targets that are cached in ``data/``.

    Tests that need real data skip themselves when the cache is absent, so
    that a fresh clone can still run the suite before fetching.
    """
    from ligandml.chembl import TARGETS, dataset_paths

    return [
        spec.name
        for spec in TARGETS
        if dataset_paths(spec.name, DATA_DIR)[0].exists()
    ]
