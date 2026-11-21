"""Fetch, clean and cache ChEMBL bioactivity data for several protein targets.

Everything in this module talks to the public ChEMBL REST API
(https://www.ebi.ac.uk/chembl/api/data/).  No bioactivity value is ever
invented: each row written to ``data/`` comes from a ChEMBL activity record,
and each fetch writes a sibling ``.meta.json`` recording the target, the
fetch date, the record counts and the full attrition table, so the rest of
the repository can run offline and still be audited.

Cleaning policy
---------------
The pipeline keeps only *exact* potency measurements, because a censored
value such as ``IC50 > 10000 nM`` is a bound, not a measurement, and
treating it as one silently injects false potencies into the training set.
Steps, in order, with the count removed by each recorded in the attrition
table:

1. Require a canonical SMILES and a positive ``standard_value`` in nM.
2. Require an exact relation (``=``); drop ``>``, ``<``, ``>=``, ``~`` etc.
3. Drop records ChEMBL flags as potential duplicates or annotates with a
   data-validity comment (``"Outside typical range"``, ``"Potential
   transcription error"`` and friends).
4. Require the potency to fall in a plausible assay window
   (:data:`MIN_VALUE_NM` to :data:`MAX_VALUE_NM`).
5. Standardise the structure with RDKit: parse, keep the largest covalent
   fragment (strips counter-ions), neutralise the charges a salt leaves
   behind, re-canonicalise.  Drop anything that fails to parse.  Without
   the neutralisation step a sodium carboxylate and its free acid stay two
   distinct structures with correlated labels, which is exactly the kind of
   near-duplicate pair that can straddle a split.
6. Require a drug-like size window (:data:`MIN_HEAVY_ATOMS` heavy atoms or
   more, molecular weight at most :data:`MAX_MOLWT`).
7. Collapse replicate measurements of the same standardised structure to
   their **median** pChEMBL.

Command line
------------
    python -m ligandml.chembl --list
    python -m ligandml.chembl --fetch-all --out-dir data
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Final

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

LOGGER = logging.getLogger(__name__)

# RDKit is loud about every unparseable SMILES; the attrition table counts
# them instead, so silence the C++ logger.
RDLogger.DisableLog("rdApp.*")

BASE_URL: Final[str] = "https://www.ebi.ac.uk/chembl/api/data"
"""Root of the public ChEMBL REST service."""

PAGE_SIZE: Final[int] = 1000
"""Activity records requested per page (the API caps this at 1000)."""

REQUEST_TIMEOUT: Final[float] = 180.0
"""Seconds to wait for any single HTTP response."""

MAX_RETRIES: Final[int] = 4
"""Attempts per request before giving up on the API."""

ACTIVITY_TYPES: Final[tuple[str, ...]] = ("IC50", "Ki", "EC50")
"""Potency endpoints accepted for modelling."""

MIN_COMPOUNDS: Final[int] = 800
"""A target is dropped from the study below this many unique compounds."""

MIN_VALUE_NM: Final[float] = 1e-4
"""Lower plausible bound on a reported potency, in nM (0.1 pM)."""

MAX_VALUE_NM: Final[float] = 1e8
"""Upper plausible bound on a reported potency, in nM (100 mM)."""

MIN_HEAVY_ATOMS: Final[int] = 6
"""Smallest structure kept; below this a 2D representation says nothing."""

MAX_MOLWT: Final[float] = 900.0
"""Largest molecular weight kept, in Da."""

ACTIVITY_FIELDS: Final[str] = ",".join(
    (
        "molecule_chembl_id",
        "canonical_smiles",
        "standard_type",
        "standard_relation",
        "standard_value",
        "standard_units",
        "pchembl_value",
        "assay_chembl_id",
        "assay_type",
        "document_chembl_id",
        "document_year",
        "potential_duplicate",
        "data_validity_comment",
        "target_chembl_id",
    )
)
"""Fields requested from the activity endpoint, to keep payloads small."""

OUTPUT_COLUMNS: Final[tuple[str, ...]] = (
    "molecule_chembl_id",
    "canonical_smiles",
    "standard_type",
    "standard_value_nm",
    "pchembl_value",
    "n_measurements",
    "n_assays",
    "first_year",
    "target_chembl_id",
)
"""Columns of a cleaned, cached dataset."""


@dataclass(frozen=True, slots=True)
class TargetSpec:
    """A protein target considered for the study.

    Attributes:
        chembl_id: ChEMBL target identifier, e.g. ``"CHEMBL240"``.
        name: Short machine-friendly slug used for filenames.
        label: Human-readable target name.
        rationale: Why this target is clinically interesting.
    """

    chembl_id: str
    name: str
    label: str
    rationale: str


#: The candidate targets.  All are human, well populated and clinically
#: meaningful; each is still count-checked against :data:`MIN_COMPOUNDS`
#: before it enters the study, and a target that falls short is dropped and
#: reported rather than quietly padded.
TARGETS: Final[tuple[TargetSpec, ...]] = (
    TargetSpec(
        "CHEMBL240",
        "herg",
        "hERG potassium channel (KCNH2)",
        "Cardiotoxicity liability. Predicting hERG block is a genuinely "
        "useful screening task: it kills candidates, not just papers.",
    ),
    TargetSpec(
        "CHEMBL217",
        "d2",
        "Dopamine D2 receptor (DRD2)",
        "The canonical antipsychotic target and the most-used benchmark in "
        "ligand-based virtual screening, which makes it a useful control.",
    ),
    TargetSpec(
        "CHEMBL220",
        "ache",
        "Acetylcholinesterase (ACHE)",
        "Alzheimer's disease symptomatic therapy, and a target with decades "
        "of diverse medicinal chemistry behind it.",
    ),
    TargetSpec(
        "CHEMBL284",
        "dpp4",
        "Dipeptidyl peptidase 4 (DPP4)",
        "Type 2 diabetes; the gliptins. Keeps the antidiabetic thread of the "
        "author's prior SUR1 work running through this study.",
    ),
    TargetSpec(
        "CHEMBL210",
        "adrb2",
        "Beta-2 adrenergic receptor (ADRB2)",
        "Asthma and COPD bronchodilators; a GPCR with both agonist (EC50) "
        "and antagonist (Ki) chemistry in the public record.",
    ),
)


@dataclass(slots=True)
class AttritionStep:
    """One row of the cleaning attrition table.

    Attributes:
        step: Description of what the step does.
        removed: Records discarded by this step.
        remaining: Records still in play after this step.
    """

    step: str
    removed: int
    remaining: int


@dataclass(slots=True)
class FetchReport:
    """Provenance and attrition for one completed target fetch.

    Attributes:
        target_chembl_id: ChEMBL target identifier queried.
        target_name: Short slug for the target.
        target_label: Human-readable target name.
        fetch_date: ISO date on which the API was queried.
        n_raw_records: Activity records returned by the API.
        n_exact_records: Records surviving all per-record filters.
        n_compounds: Unique standardised structures after the median merge.
        pchembl_min: Lowest pChEMBL retained.
        pchembl_max: Highest pChEMBL retained.
        pchembl_median: Median pChEMBL retained.
        n_assays: Distinct ChEMBL assays contributing.
        n_documents: Distinct ChEMBL documents contributing.
        year_min: Earliest first-publication year.
        year_max: Latest first-publication year.
        endpoint_counts: Compound counts per potency endpoint.
        attrition: The attrition table, in order.
        met_threshold: Whether the target cleared :data:`MIN_COMPOUNDS`.
    """

    target_chembl_id: str
    target_name: str
    target_label: str
    fetch_date: str
    n_raw_records: int
    n_exact_records: int
    n_compounds: int
    pchembl_min: float
    pchembl_max: float
    pchembl_median: float
    n_assays: int
    n_documents: int
    year_min: int
    year_max: int
    endpoint_counts: dict[str, int] = field(default_factory=dict)
    attrition: list[AttritionStep] = field(default_factory=list)
    met_threshold: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return the report as a JSON-serialisable dictionary."""
        payload = asdict(self)
        payload["attrition"] = [asdict(step) for step in self.attrition]
        for key in ("pchembl_min", "pchembl_max", "pchembl_median"):
            payload[key] = round(float(payload[key]), 3)
        return payload


class ChemblUnavailableError(RuntimeError):
    """Raised when the ChEMBL REST API cannot be reached or parsed."""


def _get_json(path: str, params: dict[str, Any]) -> dict[str, Any]:
    """GET a ChEMBL endpoint and return the decoded JSON body.

    Retries transient failures with linear backoff, because the study makes
    tens of thousands of records' worth of requests against a shared public
    service.

    Args:
        path: Endpoint path below :data:`BASE_URL`, e.g. ``"activity"``.
        params: Query parameters; ``format=json`` is added automatically.

    Returns:
        The decoded response body.

    Raises:
        ChemblUnavailableError: On repeated network, HTTP or decode failure.
    """
    import requests  # Imported lazily so offline runs need no network stack.

    query = dict(params)
    query["format"] = "json"
    url = f"{BASE_URL}/{path.lstrip('/')}"
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, params=query, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return dict(response.json())
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            LOGGER.warning("ChEMBL request failed (attempt %d): %s", attempt, exc)
            if attempt < MAX_RETRIES:
                time.sleep(2.0 * attempt)
    raise ChemblUnavailableError(f"ChEMBL request to {url} failed: {last_error}")


def count_activities(target_chembl_id: str) -> int:
    """Return the number of exact nM potency records held for a target.

    Args:
        target_chembl_id: Target identifier, e.g. ``"CHEMBL240"``.

    Returns:
        Total matching activity count reported by the API.

    Raises:
        ChemblUnavailableError: If the API cannot be reached.
    """
    payload = _get_json(
        "activity",
        {
            "target_chembl_id": target_chembl_id,
            "standard_type__in": ",".join(ACTIVITY_TYPES),
            "standard_units": "nM",
            "standard_relation": "=",
            "limit": 1,
        },
    )
    return int(payload.get("page_meta", {}).get("total_count", 0))


def iter_activities(
    target_chembl_id: str, page_size: int = PAGE_SIZE
) -> Iterator[dict[str, Any]]:
    """Yield every nM potency activity record for a target, page by page.

    Args:
        target_chembl_id: Target identifier to query.
        page_size: Records per request; the API caps this at 1000.

    Yields:
        Raw ChEMBL activity dictionaries.

    Raises:
        ChemblUnavailableError: If any page request fails.
    """
    offset = 0
    total: int | None = None
    while True:
        payload = _get_json(
            "activity",
            {
                "target_chembl_id": target_chembl_id,
                "standard_type__in": ",".join(ACTIVITY_TYPES),
                "standard_units": "nM",
                "limit": page_size,
                "offset": offset,
                "only": ACTIVITY_FIELDS,
            },
        )
        batch = payload.get("activities", [])
        if not batch:
            return
        yield from batch
        offset += len(batch)
        if total is None:
            total = int(payload.get("page_meta", {}).get("total_count", 0))
        LOGGER.info("%s: %d/%d records", target_chembl_id, offset, total)
        if offset >= total:
            return
        time.sleep(0.2)  # Be polite to the public service.


def pchembl_from_nm(value_nm: float) -> float:
    """Convert a potency in nanomolar to pChEMBL.

    pChEMBL is the negative base-10 logarithm of the molar potency, so
    1 nM is 9.0, 1 uM is 6.0 and 1 mM is 3.0.

    Args:
        value_nm: Potency in nM; must be strictly positive.

    Returns:
        The pChEMBL value.

    Raises:
        ValueError: If ``value_nm`` is not strictly positive.
    """
    if not value_nm > 0:
        raise ValueError(f"potency in nM must be positive, got {value_nm!r}")
    return 9.0 - math.log10(value_nm)


def nm_from_pchembl(pchembl: float) -> float:
    """Invert :func:`pchembl_from_nm`.

    Args:
        pchembl: A pChEMBL value.

    Returns:
        The equivalent potency in nM.
    """
    return 10.0 ** (9.0 - pchembl)


def standardise_smiles(smiles: str) -> str | None:
    """Parse, desalt, neutralise and re-canonicalise a SMILES string.

    Keeping only the largest covalent fragment strips counter-ions, and
    neutralising the remainder removes the charge the stripped counter-ion
    was balancing, so that a hydrochloride, a sodium salt and the free
    parent all collapse onto one structure and are then merged by the
    dedupe step rather than appearing as several compounds with correlated
    labels.

    Args:
        smiles: A SMILES string from ChEMBL.

    Returns:
        The canonical SMILES of the standardised structure, or ``None`` if
        the input cannot be parsed, has no heavy atoms, or falls outside the
        size window set by :data:`MIN_HEAVY_ATOMS` and :data:`MAX_MOLWT`.
    """
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        mol = rdMolStandardize.LargestFragmentChooser().choose(mol)
        if mol is not None:
            mol = rdMolStandardize.Uncharger().uncharge(mol)
    except Exception:  # pragma: no cover - RDKit raises bare RuntimeError
        return None
    if mol is None or mol.GetNumHeavyAtoms() < MIN_HEAVY_ATOMS:
        return None
    if Descriptors.MolWt(mol) > MAX_MOLWT:
        return None
    return Chem.MolToSmiles(mol)


def _earliest_year(years: pd.Series) -> int:
    """Return the earliest usable publication year in a group.

    Args:
        years: Document years for one compound; zero means unknown.

    Returns:
        The smallest positive year, or 0 when none is known.
    """
    known = years[years > 0]
    return int(known.min()) if len(known) else 0


def clean_activities(
    records: Iterable[dict[str, Any]],
) -> tuple[pd.DataFrame, list[AttritionStep]]:
    """Clean raw ChEMBL activities into a modelling table.

    See the module docstring for the ordered cleaning policy.

    Args:
        records: Raw activity dictionaries, e.g. from :func:`iter_activities`.

    Returns:
        A tuple of the deduplicated table (columns :data:`OUTPUT_COLUMNS`,
        sorted by descending pChEMBL) and the attrition table.
    """
    counters = {
        "missing": 0,
        "censored": 0,
        "flagged": 0,
        "out_of_range": 0,
        "unparseable": 0,
    }
    rows: list[dict[str, Any]] = []
    n_raw = 0
    smiles_cache: dict[str, str | None] = {}

    for record in records:
        n_raw += 1
        smiles = record.get("canonical_smiles")
        raw_value = record.get("standard_value")
        if not smiles or raw_value in (None, ""):
            counters["missing"] += 1
            continue
        if (record.get("standard_relation") or "").strip() != "=":
            counters["censored"] += 1
            continue
        if record.get("potential_duplicate") in (1, True) or record.get(
            "data_validity_comment"
        ):
            counters["flagged"] += 1
            continue
        try:
            value_nm = float(raw_value)
        except (TypeError, ValueError):
            counters["missing"] += 1
            continue
        if not MIN_VALUE_NM <= value_nm <= MAX_VALUE_NM:
            counters["out_of_range"] += 1
            continue
        if smiles not in smiles_cache:
            smiles_cache[smiles] = standardise_smiles(smiles)
        standard_smiles = smiles_cache[smiles]
        if standard_smiles is None:
            counters["unparseable"] += 1
            continue

        year = record.get("document_year")
        rows.append(
            {
                "molecule_chembl_id": record.get("molecule_chembl_id") or "",
                "canonical_smiles": standard_smiles,
                "standard_type": record.get("standard_type") or "",
                "standard_value_nm": value_nm,
                "pchembl_value": pchembl_from_nm(value_nm),
                "assay_chembl_id": record.get("assay_chembl_id") or "",
                "document_chembl_id": record.get("document_chembl_id") or "",
                "document_year": int(year) if year else 0,
                "target_chembl_id": record.get("target_chembl_id") or "",
            }
        )

    remaining = n_raw
    attrition = [AttritionStep("Raw activity records from the API", 0, remaining)]
    for key, label in (
        ("missing", "Dropped: missing SMILES or unparseable potency"),
        ("censored", "Dropped: censored relation (>, <, >=, <=, ~)"),
        ("flagged", "Dropped: ChEMBL duplicate flag or data-validity comment"),
        ("out_of_range", "Dropped: potency outside 0.1 pM - 100 mM"),
        ("unparseable", "Dropped: RDKit parse failure or outside size window"),
    ):
        remaining -= counters[key]
        attrition.append(AttritionStep(label, counters[key], remaining))

    if not rows:
        attrition.append(AttritionStep("Unique compounds after median merge", 0, 0))
        return pd.DataFrame(columns=list(OUTPUT_COLUMNS)), attrition

    frame = pd.DataFrame(rows)
    attrition.append(
        AttritionStep("Exact measurements retained", 0, len(frame))
    )

    # Dedupe on the standardised structure, not the ChEMBL id: salt forms and
    # re-registered molecules share a structure but not an id, and leaving
    # them separate would let near-identical rows straddle a split.
    grouped = (
        frame.groupby("canonical_smiles", as_index=False)
        .agg(
            molecule_chembl_id=("molecule_chembl_id", "min"),
            standard_type=("standard_type", lambda s: s.mode().iat[0]),
            pchembl_value=("pchembl_value", "median"),
            n_measurements=("pchembl_value", "size"),
            n_assays=("assay_chembl_id", "nunique"),
            first_year=("document_year", _earliest_year),
            target_chembl_id=("target_chembl_id", "first"),
        )
        .sort_values(["pchembl_value", "molecule_chembl_id"], ascending=[False, True])
        .reset_index(drop=True)
    )
    # Back-calculate nM from the merged median so the two potency columns of
    # a merged row always describe the same potency.
    grouped["standard_value_nm"] = grouped["pchembl_value"].map(nm_from_pchembl)
    attrition.append(
        AttritionStep(
            "Dropped: replicate measurements merged to median pChEMBL",
            len(frame) - len(grouped),
            len(grouped),
        )
    )
    attrition.append(
        AttritionStep("Unique compounds in the modelling set", 0, len(grouped))
    )
    return grouped.loc[:, list(OUTPUT_COLUMNS)], attrition


def _build_report(
    spec: TargetSpec,
    frame: pd.DataFrame,
    attrition: list[AttritionStep],
    n_raw: int,
    n_exact: int,
    n_documents: int,
) -> FetchReport:
    """Assemble a :class:`FetchReport` from a cleaned frame."""
    years = frame.loc[frame["first_year"] > 0, "first_year"]
    return FetchReport(
        target_chembl_id=spec.chembl_id,
        target_name=spec.name,
        target_label=spec.label,
        fetch_date=date.today().isoformat(),
        n_raw_records=n_raw,
        n_exact_records=n_exact,
        n_compounds=len(frame),
        pchembl_min=float(frame["pchembl_value"].min()) if len(frame) else float("nan"),
        pchembl_max=float(frame["pchembl_value"].max()) if len(frame) else float("nan"),
        pchembl_median=(
            float(frame["pchembl_value"].median()) if len(frame) else float("nan")
        ),
        n_assays=int(frame["n_assays"].sum()) if len(frame) else 0,
        n_documents=n_documents,
        year_min=int(years.min()) if len(years) else 0,
        year_max=int(years.max()) if len(years) else 0,
        endpoint_counts=(
            {str(k): int(v) for k, v in frame["standard_type"].value_counts().items()}
            if len(frame)
            else {}
        ),
        attrition=attrition,
        met_threshold=len(frame) >= MIN_COMPOUNDS,
    )


def dataset_paths(name: str, data_dir: Path) -> tuple[Path, Path]:
    """Return the CSV and metadata paths for a target slug.

    Args:
        name: Target slug, e.g. ``"herg"``.
        data_dir: Directory holding the cache.

    Returns:
        A ``(csv_path, meta_path)`` tuple.
    """
    data_dir = Path(data_dir)
    return data_dir / f"chembl_{name}.csv", data_dir / f"chembl_{name}.meta.json"


def fetch_target(spec: TargetSpec, data_dir: Path) -> FetchReport:
    """Fetch, clean and cache one target's bioactivity data.

    Args:
        spec: The target to fetch.
        data_dir: Directory for the CSV and metadata cache.

    Returns:
        A :class:`FetchReport` describing the download and its attrition.

    Raises:
        ChemblUnavailableError: If the API cannot be reached.
    """
    LOGGER.info("fetching %s (%s)", spec.chembl_id, spec.label)
    raw = list(iter_activities(spec.chembl_id))
    n_documents = len(
        {r.get("document_chembl_id") for r in raw if r.get("document_chembl_id")}
    )
    frame, attrition = clean_activities(raw)
    n_exact = next(
        (s.remaining for s in attrition if s.step == "Exact measurements retained"), 0
    )
    report = _build_report(spec, frame, attrition, len(raw), n_exact, n_documents)

    csv_path, meta_path = dataset_paths(spec.name, data_dir)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    meta_path.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
    LOGGER.info(
        "%s: %d raw -> %d compounds (threshold %d: %s)",
        spec.chembl_id,
        len(raw),
        len(frame),
        MIN_COMPOUNDS,
        "met" if report.met_threshold else "NOT met",
    )
    return report


def load_target(name: str, data_dir: Path) -> pd.DataFrame:
    """Load one cached target dataset.

    Args:
        name: Target slug, e.g. ``"herg"``.
        data_dir: Directory holding the cache.

    Returns:
        The cached modelling table.

    Raises:
        FileNotFoundError: If the cache is missing.
    """
    csv_path, _ = dataset_paths(name, data_dir)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"no cached dataset at {csv_path}; run "
            "`python -m ligandml.chembl --fetch-all` first"
        )
    return pd.read_csv(csv_path)


def load_meta(name: str, data_dir: Path) -> dict[str, Any]:
    """Load the provenance metadata for one cached target.

    Args:
        name: Target slug, e.g. ``"herg"``.
        data_dir: Directory holding the cache.

    Returns:
        The decoded metadata dictionary.

    Raises:
        FileNotFoundError: If the metadata file is missing.
    """
    _, meta_path = dataset_paths(name, data_dir)
    if not meta_path.exists():
        raise FileNotFoundError(f"no metadata at {meta_path}")
    return dict(json.loads(meta_path.read_text(encoding="utf-8")))


def available_targets(
    data_dir: Path, specs: Sequence[TargetSpec] = TARGETS
) -> list[TargetSpec]:
    """Return the cached targets that cleared :data:`MIN_COMPOUNDS`.

    Args:
        data_dir: Directory holding the cache.
        specs: Candidate targets to consider.

    Returns:
        The usable targets, in the order given.
    """
    usable: list[TargetSpec] = []
    for spec in specs:
        csv_path, meta_path = dataset_paths(spec.name, data_dir)
        if not (csv_path.exists() and meta_path.exists()):
            LOGGER.warning("%s: no cache, skipping", spec.name)
            continue
        meta = load_meta(spec.name, data_dir)
        if not meta.get("met_threshold", False):
            LOGGER.warning(
                "%s: only %d compounds, below the %d threshold; dropped",
                spec.name,
                meta.get("n_compounds", 0),
                MIN_COMPOUNDS,
            )
            continue
        usable.append(spec)
    return usable


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser for this module."""
    parser = argparse.ArgumentParser(
        description="Fetch and cache ChEMBL bioactivity data for the study targets.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data"),
        help="cache directory (default: data)",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        metavar="SLUG",
        help="subset of target slugs to fetch (default: all)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="print candidate targets and their live record counts, then exit",
    )
    parser.add_argument(
        "--fetch-all",
        action="store_true",
        help="fetch every candidate target",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``python -m ligandml.chembl``.

    Args:
        argv: Argument vector; defaults to the tail of :data:`sys.argv`.

    Returns:
        A process exit status.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    specs = TARGETS
    if args.targets:
        wanted = set(args.targets)
        specs = tuple(s for s in TARGETS if s.name in wanted)
        if not specs:
            print(f"error: no known target slugs in {sorted(wanted)}", file=sys.stderr)
            return 2

    try:
        if args.list:
            for spec in specs:
                print(
                    f"{spec.chembl_id}\t{count_activities(spec.chembl_id):>7d}\t"
                    f"{spec.name}\t{spec.label}"
                )
            return 0
        if not (args.fetch_all or args.targets):
            print("error: pass --list, --fetch-all or --targets", file=sys.stderr)
            return 2
        reports = [fetch_target(spec, args.out_dir) for spec in specs]
    except ChemblUnavailableError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                r.target_name: {
                    "target_chembl_id": r.target_chembl_id,
                    "n_raw_records": r.n_raw_records,
                    "n_compounds": r.n_compounds,
                    "met_threshold": r.met_threshold,
                }
                for r in reports
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
