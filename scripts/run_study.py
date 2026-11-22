"""Run the full study and write every result table and figure.

Stages, in order:

1. **Datasets.** Load the cached ChEMBL tables, write the dataset summary
   and the cleaning attrition table.
2. **Split comparison** - the headline. Every target, four models, two
   framings, three split strategies, on the default representation.
   Predictions are kept so that later stages reuse them rather than
   refitting.
3. **Representation comparison.** Every target, five representations, two
   models, scaffold split, classification.
4. **Applicability domain.** Error against nearest-neighbour similarity,
   and the coverage/accuracy trade-off.
5. **Screening.** Enrichment curves, the top-1%/5% screening demo, and
   classifier calibration.
6. **Null models.** Y-scrambling against the real score.
7. **Chemical space.** PCA of the fingerprint space, coloured by split.

Command line
------------
    python scripts/run_study.py                # the full default matrix
    python scripts/run_study.py --fast         # a quick subset, for smoke tests
    python scripts/run_study.py --targets herg dpp4 --models linear
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ligandml import (  # noqa: E402  # noqa: E402
    RANDOM_STATE,
    chembl,
    conformal,
    decoys,
    domain,
    evaluate,
    featurise,
    models,
    nulls,
    plots,
    splits,
    uncertainty,
)

LOGGER = logging.getLogger("run_study")

DEFAULT_DATA_DIR: Final[Path] = REPO_ROOT / "data"
DEFAULT_RESULTS_DIR: Final[Path] = REPO_ROOT / "results"
DEFAULT_FIGURES_DIR: Final[Path] = REPO_ROOT / "figures"
FEATURE_CACHE_DIR: Final[Path] = REPO_ROOT / "data" / "features"

HEADLINE_METRIC: Final[str] = "roc_auc"
"""The metric the split-strategy comparison is led with."""

REPRESENTATION_MODELS: Final[tuple[str, ...]] = ("linear", "random_forest")
"""Models used for the representation sweep: one low-capacity, one high."""

NULL_MODEL: Final[str] = "random_forest"
"""Model y-scrambled. The highest-capacity one, because it is the one that
could most plausibly be fitting noise."""

ENRICHMENT_GRID: Final[np.ndarray] = np.unique(
    np.concatenate(
        [np.logspace(-2.7, 0.0, 60), np.array([0.01, 0.05, 0.1, 0.25, 0.5, 1.0])]
    )
)
"""Fractions of the ranked list at which the enrichment curve is sampled."""

DOMAIN_SAMPLE: Final[int] = 3000
"""Training compounds sampled when deriving the in-domain threshold.

The self-similarity matrix is quadratic in the training-set size, so the
threshold - a percentile of a distribution - is estimated from a seeded
sample rather than from all of it.
"""

PCA_TARGET_LIMIT: Final[int] = 6000
"""Compounds sampled for the chemical-space PCA figure, to keep it legible."""


@dataclass(slots=True)
class CellPrediction:
    """Predictions from one fitted cell of the matrix, kept for reuse.

    Attributes:
        target: Target slug.
        representation: Representation slug.
        model: Model slug.
        strategy: Split strategy.
        task: ``"regression"`` or ``"classification"``.
        y_true: Observed test labels or values.
        y_pred: Test predictions (probabilities, or pChEMBL values).
        test_index: Positional indices of the test rows in the target table.
        train_index: Positional indices of the training rows.
    """

    target: str
    representation: str
    model: str
    strategy: str
    task: str
    y_true: np.ndarray
    y_pred: np.ndarray
    test_index: np.ndarray
    train_index: np.ndarray


@dataclass(slots=True)
class TargetData:
    """A loaded, featurised target.

    Attributes:
        spec: The target's specification.
        frame: The cleaned activity table, restricted to parseable rows.
        features: Feature matrices by representation slug.
        labels: Binary activity labels at the stated threshold.
        values: pChEMBL values.
        split_cache: Split results by strategy.
    """

    spec: chembl.TargetSpec
    frame: pd.DataFrame
    features: dict[str, np.ndarray]
    labels: np.ndarray
    values: np.ndarray
    split_cache: dict[str, splits.SplitResult] = field(default_factory=dict)

    def split(self, strategy: str, test_size: float, seed: int) -> splits.SplitResult:
        """Return (and memoise) the split for a strategy.

        Args:
            strategy: One of :data:`ligandml.splits.SPLIT_STRATEGIES`.
            test_size: Target test fraction.
            seed: Seed for the strategies that use one.

        Returns:
            The validated split.
        """
        if strategy not in self.split_cache:
            self.split_cache[strategy] = splits.make_split(
                self.frame, strategy, test_size=test_size, seed=seed
            )
        return self.split_cache[strategy]


def _feature_cache_path(target: str, representation: str) -> Path:
    """Return the npz cache path for one target/representation pair."""
    return FEATURE_CACHE_DIR / f"{target}__{representation}.npz"


def load_and_featurise(
    spec: chembl.TargetSpec,
    representations: Sequence[str],
    data_dir: Path,
    threshold: float,
    use_cache: bool = True,
) -> TargetData:
    """Load a cached target and compute every requested representation.

    Rows whose SMILES will not parse are dropped once, before any
    representation is computed, so every matrix shares one row ordering
    with the frame.

    Args:
        spec: The target to load.
        representations: Representation slugs to compute.
        data_dir: Directory holding the dataset cache.
        threshold: pChEMBL threshold for the binary labels.
        use_cache: Read and write the npz feature cache.

    Returns:
        A :class:`TargetData`.

    Raises:
        FileNotFoundError: If the dataset cache is missing.
    """
    frame = chembl.load_target(spec.name, data_dir)
    mols, keep = featurise.mols_from_smiles(frame["canonical_smiles"].tolist())
    frame = frame.loc[keep].reset_index(drop=True)
    if int((~keep).sum()):
        LOGGER.warning(
            "%s: dropped %d rows whose SMILES would not parse",
            spec.name,
            int((~keep).sum()),
        )

    matrices: dict[str, np.ndarray] = {}
    for name in representations:
        cache_path = _feature_cache_path(spec.name, name)
        if use_cache and cache_path.exists():
            with np.load(cache_path) as handle:
                cached = handle["features"]
            if cached.shape[0] == len(frame):
                matrices[name] = cached
                LOGGER.info("%s/%s: features from cache", spec.name, name)
                continue
            LOGGER.warning("%s/%s: stale cache, recomputing", spec.name, name)
        started = time.perf_counter()
        matrix, _ = featurise.get_representation(name)(mols)
        matrices[name] = matrix
        LOGGER.info(
            "%s/%s: %s in %.1fs",
            spec.name,
            name,
            matrix.shape,
            time.perf_counter() - started,
        )
        if use_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_path, features=matrix.astype(np.float32))

    return TargetData(
        spec=spec,
        frame=frame,
        features=matrices,
        labels=evaluate.binarise(frame["pchembl_value"], threshold=threshold),
        values=frame["pchembl_value"].to_numpy(dtype=float),
    )


def write_dataset_tables(
    data: Sequence[TargetData], data_dir: Path, results_dir: Path, threshold: float
) -> None:
    """Write the dataset summary and cleaning attrition tables.

    Args:
        data: The loaded targets.
        data_dir: Directory holding the dataset metadata.
        results_dir: Directory to write the CSVs into.
        threshold: Activity threshold, recorded alongside the class balance.
    """
    summary_rows: list[dict[str, Any]] = []
    attrition_rows: list[dict[str, Any]] = []
    for item in data:
        meta = chembl.load_meta(item.spec.name, data_dir)
        scaffolds = splits.scaffold_groups(item.frame["canonical_smiles"].tolist())
        years = item.frame.loc[item.frame["first_year"] > 0, "first_year"]
        summary_rows.append(
            {
                "target": item.spec.name,
                "target_chembl_id": item.spec.chembl_id,
                "target_label": item.spec.label,
                "fetch_date": meta["fetch_date"],
                "n_raw_records": meta["n_raw_records"],
                "n_exact_records": meta["n_exact_records"],
                "n_compounds": len(item.frame),
                "n_scaffolds": len(set(scaffolds.tolist())),
                "n_documents": meta["n_documents"],
                "pchembl_min": round(float(item.values.min()), 2),
                "pchembl_median": round(float(np.median(item.values)), 2),
                "pchembl_max": round(float(item.values.max()), 2),
                "pchembl_sd": round(float(item.values.std(ddof=1)), 2),
                "activity_threshold": threshold,
                "n_active": int(item.labels.sum()),
                "active_rate": round(float(item.labels.mean()), 3),
                "year_min": int(years.min()) if len(years) else 0,
                "year_max": int(years.max()) if len(years) else 0,
                "endpoint_counts": json.dumps(meta["endpoint_counts"]),
            }
        )
        for step in meta["attrition"]:
            attrition_rows.append(
                {
                    "target": item.spec.name,
                    "target_chembl_id": item.spec.chembl_id,
                    "step": step["step"],
                    "removed": step["removed"],
                    "remaining": step["remaining"],
                }
            )

    results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(results_dir / "dataset_summary.csv", index=False)
    pd.DataFrame(attrition_rows).to_csv(results_dir / "attrition.csv", index=False)
    LOGGER.info("wrote dataset_summary.csv and attrition.csv")


def run_split_comparison(
    data: Sequence[TargetData],
    representation: str,
    model_names: Sequence[str],
    strategies: Sequence[str],
    tasks: Sequence[str],
    test_size: float,
    seed: int,
    results_dir: Path,
) -> tuple[pd.DataFrame, list[CellPrediction]]:
    """Fit the headline matrix and score every cell.

    Args:
        data: The loaded targets.
        representation: Representation slug to use throughout.
        model_names: Model slugs to fit.
        strategies: Split strategies to compare.
        tasks: Framings to run.
        test_size: Target test fraction.
        seed: Seed for the models and the random split.
        results_dir: Directory for the CSV outputs.

    Returns:
        The results table and the kept predictions.
    """
    kind = featurise.get_representation(representation).kind
    rows: list[dict[str, Any]] = []
    predictions: list[CellPrediction] = []

    for item in data:
        matrix = item.features[representation]
        smiles = item.frame["canonical_smiles"].tolist()
        for strategy in strategies:
            split = item.split(strategy, test_size, seed)
            leakage = splits.scaffold_overlap(smiles, split)
            for task in tasks:
                target_values = item.labels if task == "classification" else item.values
                for model_name in model_names:
                    started = time.perf_counter()
                    prediction = models.fit_predict(
                        models.build_model(model_name, task, kind, seed=seed),
                        matrix[split.train_index],
                        target_values[split.train_index],
                        matrix[split.test_index],
                        task,
                    )
                    elapsed = time.perf_counter() - started
                    observed = target_values[split.test_index]
                    metrics = (
                        evaluate.classification_metrics(observed, prediction)
                        if task == "classification"
                        else evaluate.regression_metrics(observed, prediction)
                    )
                    row = {
                        "target": item.spec.name,
                        "target_chembl_id": item.spec.chembl_id,
                        "representation": representation,
                        "model": model_name,
                        "split": strategy,
                        "task": task,
                        "n_train": split.n_train,
                        "n_test": split.n_test,
                        "frac_test_compounds_leaked": round(
                            leakage["frac_test_compounds_seen"], 4
                        ),
                        "n_shared_scaffolds": int(leakage["n_shared_scaffolds"]),
                        "cutoff_year": split.details.get("cutoff_year", ""),
                        "fit_seconds": round(elapsed, 2),
                    }
                    row.update(
                        {k: (round(v, 4) if isinstance(v, float) else v)
                         for k, v in metrics.items()}
                    )
                    rows.append(row)
                    predictions.append(
                        CellPrediction(
                            target=item.spec.name,
                            representation=representation,
                            model=model_name,
                            strategy=strategy,
                            task=task,
                            y_true=np.asarray(observed, dtype=float),
                            y_pred=prediction,
                            test_index=split.test_index,
                            train_index=split.train_index,
                        )
                    )
                    LOGGER.info(
                        "%s | %s | %s | %s: %s in %.1fs",
                        item.spec.name,
                        strategy,
                        task,
                        model_name,
                        _headline(metrics, task),
                        elapsed,
                    )

    frame = pd.DataFrame(rows)
    results_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(results_dir / "split_comparison.csv", index=False)
    LOGGER.info("wrote split_comparison.csv (%d rows)", len(frame))
    return frame, predictions


def _headline(metrics: dict[str, float], task: str) -> str:
    """Format a one-line summary of a scored cell for the log."""
    if task == "classification":
        return (
            f"AUC {metrics['roc_auc']:.3f} EF1% {metrics['ef_1pct']:.1f} "
            f"BEDROC1% {metrics['bedroc_1pct']:.3f}"
        )
    return f"R2 {metrics['r2']:.3f} RMSE {metrics['rmse']:.3f}"


def run_decoy_screen(
    data: Sequence[TargetData],
    n_actives: int,
    ratio: int,
    test_size: float,
    seed: int,
    model_name: str,
    results_dir: Path,
) -> pd.DataFrame:
    """Score the models on a screening library instead of a measured set.

    The measured datasets are 50-83% active, which caps enrichment factor
    at 1.2-1.9 and leaves the screening section unable to say anything. A
    decoy library puts the active rate where a real screen has it, so the
    same metric has room to discriminate.

    Args:
        data: The loaded targets.
        n_actives: Actives to sample per target.
        ratio: Decoys per active to attach.
        test_size: Target test fraction for the scaffold split.
        seed: Seed for the active sample and the model.
        model_name: Model to score.
        results_dir: Directory for the CSV outputs.

    Returns:
        One row per target.
    """
    frames = {item.spec.name: item.frame for item in data}
    rng = np.random.default_rng(seed)
    construction: list[dict[str, Any]] = []
    scored: list[dict[str, Any]] = []

    for item in data:
        name = item.spec.name
        active_rows = np.flatnonzero(item.labels == 1)
        if active_rows.size == 0:
            LOGGER.warning("  %s: no actives, skipping decoy screen", name)
            continue
        keep = rng.choice(
            active_rows, size=min(n_actives, active_rows.size), replace=False
        )
        active_smiles = item.frame["canonical_smiles"].to_numpy()[keep].tolist()

        pool = decoys.cross_target_pool(frames, name)
        if pool.empty:
            LOGGER.warning("  %s: empty decoy pool, skipping", name)
            continue
        pool_smiles = pool["canonical_smiles"].tolist()

        active_mols, active_ok = featurise.mols_from_smiles(active_smiles)
        pool_mols, pool_ok = featurise.mols_from_smiles(pool_smiles)
        active_smiles = [s for s, ok in zip(active_smiles, active_ok) if ok]
        pool_smiles = [s for s, ok in zip(pool_smiles, pool_ok) if ok]

        active_props, _ = featurise.rdkit_descriptors(
            active_mols, names=decoys.MATCH_PROPERTIES
        )
        pool_props, _ = featurise.rdkit_descriptors(
            pool_mols, names=decoys.MATCH_PROPERTIES
        )
        active_fp, _ = featurise.morgan_fingerprints(active_mols)
        pool_fp, _ = featurise.morgan_fingerprints(pool_mols)
        similarity = domain.nearest_neighbour_similarity(pool_fp, active_fp)

        built = decoys.build_decoy_set(
            name,
            active_smiles,
            active_props,
            pool_smiles,
            pool_props,
            similarity,
            ratio=ratio,
        )
        construction.append(built.as_row())
        LOGGER.info(
            "  %s: %d actives + %d decoys, %.2f%% active, "
            "EF ceiling %.0f, property bias %.3f",
            name,
            built.n_actives,
            built.n_decoys,
            100.0 * built.active_rate,
            built.enrichment_ceiling_1pct,
            built.property_bias,
        )

        # Score it exactly as the measured benchmark is scored.
        mols, ok = featurise.mols_from_smiles(list(built.smiles))
        matrix, _ = featurise.morgan_fingerprints(mols)
        labels = built.labels[ok]
        frame = pd.DataFrame({"canonical_smiles": [s for s, k in zip(built.smiles, ok) if k]})
        split = splits.scaffold_split(
            frame["canonical_smiles"].tolist(), test_size=test_size
        )
        split.validate(len(frame))
        prediction = models.fit_predict(
            models.build_model(model_name, "classification", "fingerprint", seed=seed),
            matrix[split.train_index],
            labels[split.train_index],
            matrix[split.test_index],
            "classification",
        )
        observed = labels[split.test_index]
        metrics = evaluate.classification_metrics(observed, prediction)
        scored.append(
            {
                "target": name,
                "model": model_name,
                "benchmark": "decoy_library",
                "n_test": int(len(observed)),
                "active_rate": round(float(observed.mean()), 5),
                **{
                    k: (round(v, 4) if isinstance(v, float) else v)
                    for k, v in metrics.items()
                    if k
                    in (
                        "roc_auc",
                        "pr_auc",
                        "ef_1pct",
                        "ef_ceiling_1pct",
                        "bedroc_1pct",
                        "hit_rate_1pct",
                        "recall_1pct",
                        "ef_5pct",
                        "ef_ceiling_5pct",
                    )
                },
            }
        )
        LOGGER.info(
            "  %s: AUC %.3f, EF@1%% %.1f of %.0f",
            name,
            scored[-1].get("roc_auc", float("nan")),
            scored[-1].get("ef_1pct", float("nan")),
            scored[-1].get("ef_ceiling_1pct", float("nan")),
        )

    pd.DataFrame(construction).to_csv(
        results_dir / "decoy_construction.csv", index=False
    )
    frame = pd.DataFrame(scored)
    frame.to_csv(results_dir / "decoy_screen.csv", index=False)
    LOGGER.info("wrote decoy_construction.csv and decoy_screen.csv")
    return frame


def run_conformal(
    data: Sequence[TargetData],
    representation: str,
    model_name: str,
    strategies: Sequence[str],
    test_size: float,
    seed: int,
    calibration_fraction: float,
    results_dir: Path,
) -> pd.DataFrame:
    """Turn the ranking into prediction sets with a coverage guarantee.

    The training set is split again into a proper training part and a
    calibration part the model never sees, which is what buys the
    guarantee. Calibration is class-conditional, so the promise holds for
    the actives separately rather than only on average.

    Args:
        data: The loaded targets.
        representation: Representation slug.
        model_name: Model to wrap.
        strategies: Split strategies to report.
        test_size: Target test fraction.
        seed: Seed for the model and the calibration carve-out.
        calibration_fraction: Share of the training set held for
            calibration.
        results_dir: Directory for the CSV output.

    Returns:
        One row per target, strategy and significance level.
    """
    rows: list[dict[str, Any]] = []
    for item in data:
        matrix = item.features[representation]
        for strategy in strategies:
            split = item.split(strategy, test_size, seed)
            rng = np.random.default_rng(seed)
            train = np.array(split.train_index)
            rng.shuffle(train)
            n_calibration = max(1, int(round(calibration_fraction * len(train))))
            calibration_index, fit_index = train[:n_calibration], train[n_calibration:]
            if len(fit_index) < 10 or len(calibration_index) < 10:
                LOGGER.warning("  %s/%s: too few rows to calibrate", item.spec.name, strategy)
                continue

            model = models.build_model(
                model_name, "classification", "fingerprint", seed=seed
            )
            model.fit(matrix[fit_index], item.labels[fit_index])
            calibration_probs = model.predict_proba(matrix[calibration_index])
            test_probs = model.predict_proba(matrix[split.test_index])

            reports = conformal.run(
                calibration_probs,
                item.labels[calibration_index],
                test_probs,
                item.labels[split.test_index],
            )
            for report in reports:
                rows.append(
                    {
                        "target": item.spec.name,
                        "model": model_name,
                        "split": strategy,
                        "n_calibration": len(calibration_index),
                        **report.as_row(),
                    }
                )
            headline = reports[1] if len(reports) > 1 else reports[0]
            LOGGER.info(
                "  %s/%s: at %.0f%% confidence coverage %.3f "
                "(actives %.3f), %.0f%% singletons, %.0f%% empty",
                item.spec.name,
                strategy,
                100 * headline.target_coverage,
                headline.coverage,
                headline.coverage_by_class.get(1, float("nan")),
                100 * headline.singleton_rate,
                100 * headline.empty_rate,
            )

    frame = pd.DataFrame(rows)
    frame.to_csv(results_dir / "conformal.csv", index=False)
    LOGGER.info("wrote conformal.csv (%d rows)", len(frame))
    return frame


def run_time_cutoff_sensitivity(
    data: Sequence[TargetData],
    representation: str,
    model_name: str,
    results_dir: Path,
    seed: int,
) -> pd.DataFrame:
    """Does the time-split gap depend on where the line is drawn?

    A time split has no seed, so its counterpart to re-drawing is moving
    the cutoff year. If the gap only appears at one particular cutoff it
    is an artefact of that choice; if it survives three, it is not.

    Args:
        data: The loaded targets.
        representation: Representation slug.
        model_name: Model to score.
        results_dir: Directory for the CSV output.
        seed: Seed for the model and the random reference split.

    Returns:
        One row per target and cutoff.
    """
    rows: list[dict[str, Any]] = []
    for item in data:
        matrix = item.features[representation]
        years = item.frame["first_year"].tolist()
        try:
            cutoffs = uncertainty.time_cutoff_grid(years)
        except ValueError:
            LOGGER.warning("  %s: no usable years", item.spec.name)
            continue
        reference = splits.random_split(len(item.frame), seed=seed)
        reference_auc = _score_auc(
            matrix, item.labels, reference, model_name, seed
        )
        for cutoff in cutoffs:
            mask = np.asarray(
                [y == y and y is not None and float(y) >= cutoff for y in years]
            )
            test_index = np.flatnonzero(mask)
            train_index = np.flatnonzero(~mask)
            if len(test_index) < 25 or len(train_index) < 50:
                continue
            split = splits.SplitResult(
                strategy="time",
                train_index=train_index,
                test_index=test_index,
                details={"cutoff_year": cutoff},
            )
            auc = _score_auc(matrix, item.labels, split, model_name, seed)
            rows.append(
                {
                    "target": item.spec.name,
                    "model": model_name,
                    "cutoff_year": cutoff,
                    "n_train": len(train_index),
                    "n_test": len(test_index),
                    "test_fraction": round(len(test_index) / len(mask), 4),
                    "roc_auc": round(auc, 4),
                    "random_reference": round(reference_auc, 4),
                    "gap_vs_random": round(reference_auc - auc, 4),
                }
            )
            LOGGER.info(
                "  %s cutoff %d: AUC %.3f, gap %+.3f",
                item.spec.name,
                cutoff,
                auc,
                reference_auc - auc,
            )

    frame = pd.DataFrame(rows)
    frame.to_csv(results_dir / "time_cutoff_sensitivity.csv", index=False)
    LOGGER.info("wrote time_cutoff_sensitivity.csv (%d rows)", len(frame))
    return frame


def _score_auc(
    matrix: np.ndarray,
    labels: np.ndarray,
    split: splits.SplitResult,
    model_name: str,
    seed: int,
) -> float:
    """Fit one cell and return its ROC AUC."""
    prediction = models.fit_predict(
        models.build_model(model_name, "classification", "fingerprint", seed=seed),
        matrix[split.train_index],
        labels[split.train_index],
        matrix[split.test_index],
        "classification",
    )
    metrics = evaluate.classification_metrics(labels[split.test_index], prediction)
    return float(metrics["roc_auc"])


def run_repeated_splits(
    data: Sequence[TargetData],
    representation: str,
    model_name: str,
    task: str,
    test_size: float,
    seeds: Sequence[int],
    results_dir: Path,
) -> pd.DataFrame:
    """Re-draw the random and scaffold splits under several seeds.

    The rest of the study draws each split once, so its gap carries no
    seed-to-seed spread. This is the error bar a cheminformatics benchmark
    normally quotes, and it is the one a reader asks for first.

    One honest wrinkle: the study's scaffold split fills the largest
    chemotypes into training first, which is deterministic, so re-seeding
    it changes nothing. To get a distribution at all the repeats shuffle
    the scaffold groups instead. That is a slightly easier split than the
    largest-first construction, so these absolute numbers sit above the
    headline table; the spread, which is what this stage exists to
    measure, is the part to read.

    The time split has no seed at all, so it is absent here; its
    sensitivity analysis is over the cutoff year instead.

    Args:
        data: The loaded targets.
        representation: Representation slug.
        model_name: The single model to repeat.
        task: ``classification`` or ``regression``.
        test_size: Target test fraction.
        seeds: Seeds to draw.
        results_dir: Directory for the CSV outputs.

    Returns:
        The per-target summary over seeds.
    """
    kind = featurise.get_representation(representation).kind
    metric = HEADLINE_METRIC if task == "classification" else "spearman_rho"
    draws: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for item in data:
        matrix = item.features[representation]
        smiles = item.frame["canonical_smiles"].tolist()
        target_values = item.labels if task == "classification" else item.values
        scores: dict[str, list[float]] = {"random": [], "scaffold": []}
        for seed in seeds:
            for strategy in ("random", "scaffold"):
                if strategy == "random":
                    split = splits.random_split(
                        len(item.frame), test_size=test_size, seed=seed
                    )
                else:
                    split = splits.scaffold_split(
                        smiles, test_size=test_size, shuffle_groups=True, seed=seed
                    )
                split.validate(len(item.frame))
                started = time.perf_counter()
                prediction = models.fit_predict(
                    models.build_model(model_name, task, kind, seed=seed),
                    matrix[split.train_index],
                    target_values[split.train_index],
                    matrix[split.test_index],
                    task,
                )
                observed = target_values[split.test_index]
                metrics = (
                    evaluate.classification_metrics(observed, prediction)
                    if task == "classification"
                    else evaluate.regression_metrics(observed, prediction)
                )
                value = float(metrics[metric])
                scores[strategy].append(value)
                draws.append(
                    {
                        "target": item.spec.name,
                        "model": model_name,
                        "representation": representation,
                        "task": task,
                        "strategy": strategy,
                        "seed": seed,
                        "n_train": split.n_train,
                        "n_test": split.n_test,
                        "metric": metric,
                        "value": round(value, 4),
                        "fit_seconds": round(time.perf_counter() - started, 2),
                    }
                )
        for row in uncertainty.summarise_repeats(scores):
            summaries.append({"target": item.spec.name, "metric": metric, **row})
        gap = next(
            (r for r in summaries[-2:] if r["strategy"] == "scaffold"), None
        )
        if gap is not None:
            LOGGER.info(
                "  %s: gap %+0.4f +/- %0.4f over %d seeds",
                item.spec.name,
                gap.get("mean_gap_vs_reference", float("nan")),
                gap.get("sd_gap_vs_reference", float("nan")),
                len(seeds),
            )

    pd.DataFrame(draws).to_csv(results_dir / "repeated_split_draws.csv", index=False)
    frame = pd.DataFrame(summaries)
    frame.to_csv(results_dir / "repeated_splits.csv", index=False)
    LOGGER.info(
        "wrote repeated_splits.csv (%d rows) and repeated_split_draws.csv (%d rows)",
        len(frame),
        len(draws),
    )
    return frame


def write_gap_uncertainty(gaps: pd.DataFrame, results_dir: Path) -> pd.DataFrame:
    """Put a spread on the averaged split gap.

    The headline sentence averages the gap over every target and model, so
    the uncertainty that belongs beside it is the one across those cells:
    it answers whether the effect survives changing the chemistry and the
    model family, rather than how precisely one test set pins one number
    down. Both a t interval and the Wilcoxon signed-rank statistic are
    written, since twenty paired cells with a known exception is exactly
    the case where a mean and a median can disagree.

    Args:
        gaps: The per-cell gap table from :func:`write_split_gap`.
        results_dir: Directory for the CSV output.

    Returns:
        One row per task and comparison.
    """
    rows: list[dict[str, Any]] = []
    for task, group in gaps.groupby("task", sort=False):
        metric = str(group["metric"].iloc[0])
        labels = [f"{r.target}/{r.model}" for r in group.itertuples()]
        for strategy in ("scaffold", "time"):
            column = f"drop_vs_random_{strategy}"
            if column not in group.columns:
                continue
            values = group[column].tolist()
            if not any(v == v for v in values):
                continue
            summary = uncertainty.summarise_gaps(
                values, f"random to {strategy}", metric, labels=labels
            )
            rows.append({"task": task, **summary.as_row()})

    frame = pd.DataFrame(rows)
    frame.to_csv(results_dir / "split_gap_uncertainty.csv", index=False)
    for row in rows:
        LOGGER.info(
            "  %s %s: %+0.4f (95%% CI %+0.4f to %+0.4f), "
            "%d/%d cells agree, Wilcoxon p=%.4f",
            row["task"],
            row["comparison"],
            row["mean_gap"],
            row["ci95_low"],
            row["ci95_high"],
            row["n_cells_positive"],
            row["n_cells"],
            row["wilcoxon_p"],
        )
    LOGGER.info("wrote split_gap_uncertainty.csv (%d rows)", len(frame))
    return frame


def write_split_gap(
    comparison: pd.DataFrame, results_dir: Path, figures_dir: Path
) -> pd.DataFrame:
    """Summarise and plot how far the honest splits fall below the random one.

    Args:
        comparison: The headline results table.
        results_dir: Directory for the CSV output.
        figures_dir: Directory for the figures.

    Returns:
        The gap table.
    """
    gap_rows: list[dict[str, Any]] = []
    for (target, model, task), group in comparison.groupby(
        ["target", "model", "task"], sort=False
    ):
        by_split = group.set_index("split")
        metric = HEADLINE_METRIC if task == "classification" else "spearman_rho"
        secondary = "ef_1pct" if task == "classification" else "r2"
        if "random" not in by_split.index:
            continue
        random_value = float(by_split.loc["random", metric])
        row: dict[str, Any] = {
            "target": target,
            "model": model,
            "task": task,
            "metric": metric,
            "random": round(random_value, 4),
        }
        for strategy in ("scaffold", "time"):
            if strategy in by_split.index:
                value = float(by_split.loc[strategy, metric])
                row[strategy] = round(value, 4)
                row[f"drop_vs_random_{strategy}"] = round(random_value - value, 4)
                row[f"relative_drop_{strategy}"] = (
                    round((random_value - value) / random_value, 4)
                    if random_value
                    else float("nan")
                )
        row["secondary_metric"] = secondary
        for strategy in ("random", "scaffold", "time"):
            if strategy in by_split.index:
                row[f"{secondary}_{strategy}"] = round(
                    float(by_split.loc[strategy, secondary]), 4
                )
        gap_rows.append(row)

    frame = pd.DataFrame(gap_rows)
    frame.to_csv(results_dir / "split_gap.csv", index=False)
    LOGGER.info("wrote split_gap.csv (%d rows)", len(frame))
    write_gap_uncertainty(frame, results_dir)

    classification = frame[frame["task"] == "classification"]
    if not classification.empty:
        plots.plot_split_gap(
            [
                {
                    "label": f"{r.target} / {r.model}",
                    **{
                        s: getattr(r, s)
                        for s in ("random", "scaffold", "time")
                        if hasattr(r, s) and pd.notna(getattr(r, s))
                    },
                }
                for r in classification.itertuples()
            ],
            figures_dir / "split_gap.png",
            metric_label="ROC AUC",
        )
        LOGGER.info("wrote figures/split_gap.png")
    return frame


def write_headline_figures(
    comparison: pd.DataFrame,
    data: Sequence[TargetData],
    default_model: str,
    figures_dir: Path,
    results_dir: Path,
) -> None:
    """Draw the headline split-strategy figure and the leakage figure.

    Args:
        comparison: The headline results table.
        data: The loaded targets, for their labels.
        default_model: Model whose scores the headline figure shows.
        figures_dir: Directory for the figures.
        results_dir: Directory for the leakage CSV.
    """
    labels = {item.spec.name: item.spec.name for item in data}
    subset = comparison[
        (comparison["model"] == default_model)
        & (comparison["task"] == "classification")
    ]
    if not subset.empty:
        scores = {
            labels[target]: {
                str(row["split"]): float(row[HEADLINE_METRIC])
                for _, row in group.iterrows()
            }
            for target, group in subset.groupby("target", sort=False)
        }
        plots.plot_split_strategies(
            scores,
            figures_dir / "split_strategy_performance.png",
            metric_label="ROC AUC",
            chance=0.5,
            subtitle=(
                f"{default_model}, ECFP4 2048 bits, classification at "
                f"pChEMBL >= {evaluate.ACTIVITY_THRESHOLD:g}"
            ),
        )
        LOGGER.info("wrote figures/split_strategy_performance.png")

        enrichment = {
            labels[target]: {
                str(row["split"]): float(row["ef_1pct"])
                for _, row in group.iterrows()
            }
            for target, group in subset.groupby("target", sort=False)
        }
        plots.plot_split_strategies(
            enrichment,
            figures_dir / "split_strategy_enrichment.png",
            metric_label="enrichment factor at 1%",
            chance=1.0,
            subtitle=(
                "the same models judged by what a screen actually buys; "
                "the attainable ceiling differs per test set "
                "(see results/screening_demo.csv)"
            ),
        )
        LOGGER.info("wrote figures/split_strategy_enrichment.png")

    leak_rows: list[dict[str, Any]] = []
    fractions: dict[str, dict[str, float]] = {}
    for item in data:
        smiles = item.frame["canonical_smiles"].tolist()
        fractions[labels[item.spec.name]] = {}
        for strategy, split in item.split_cache.items():
            overlap = splits.scaffold_overlap(smiles, split)
            fractions[labels[item.spec.name]][strategy] = overlap[
                "frac_test_compounds_seen"
            ]
            leak_rows.append(
                {
                    "target": item.spec.name,
                    "split": strategy,
                    "n_train": split.n_train,
                    "n_test": split.n_test,
                    "n_dropped": split.n_dropped,
                    **{k: round(v, 4) for k, v in overlap.items()},
                    "details": json.dumps(split.details),
                }
            )
    pd.DataFrame(leak_rows).to_csv(results_dir / "split_leakage.csv", index=False)
    plots.plot_leakage(fractions, figures_dir / "split_leakage.png")
    LOGGER.info("wrote split_leakage.csv and figures/split_leakage.png")


def run_representation_comparison(
    data: Sequence[TargetData],
    representations: Sequence[str],
    model_names: Sequence[str],
    strategy: str,
    test_size: float,
    seed: int,
    results_dir: Path,
    figures_dir: Path,
    reuse: Sequence[CellPrediction] = (),
) -> pd.DataFrame:
    """Compare representations under one split strategy.

    Args:
        data: The loaded targets.
        representations: Representation slugs to compare.
        model_names: Model slugs to fit.
        strategy: Split strategy to hold fixed.
        test_size: Target test fraction.
        seed: Seed for the models.
        results_dir: Directory for the CSV output.
        figures_dir: Directory for the figure.
        reuse: Predictions already computed, so overlapping cells are not
            refitted.

    Returns:
        The results table.
    """
    done = {
        (p.target, p.representation, p.model, p.strategy, p.task): p for p in reuse
    }
    rows: list[dict[str, Any]] = []
    for item in data:
        split = item.split(strategy, test_size, seed)
        observed = item.labels[split.test_index]
        for name in representations:
            matrix = item.features[name]
            kind = featurise.get_representation(name).kind
            for model_name in model_names:
                key = (
                    item.spec.name, name, model_name, strategy, "classification"
                )
                cached = done.get(key)
                if cached is not None:
                    prediction, elapsed = cached.y_pred, 0.0
                else:
                    started = time.perf_counter()
                    prediction = models.fit_predict(
                        models.build_model(
                            model_name, "classification", kind, seed=seed
                        ),
                        matrix[split.train_index],
                        item.labels[split.train_index],
                        matrix[split.test_index],
                        "classification",
                    )
                    elapsed = time.perf_counter() - started
                metrics = evaluate.classification_metrics(observed, prediction)
                rows.append(
                    {
                        "target": item.spec.name,
                        "representation": name,
                        "representation_label": featurise.get_representation(
                            name
                        ).label,
                        "n_features": matrix.shape[1],
                        "model": model_name,
                        "split": strategy,
                        "task": "classification",
                        "fit_seconds": round(elapsed, 2),
                        "reused": cached is not None,
                        **{
                            k: (round(v, 4) if isinstance(v, float) else v)
                            for k, v in metrics.items()
                        },
                    }
                )
                LOGGER.info(
                    "%s | %s | %s: AUC %.3f (%.1fs)",
                    item.spec.name, name, model_name, metrics["roc_auc"], elapsed,
                )

    frame = pd.DataFrame(rows)
    frame.to_csv(results_dir / "representation_comparison.csv", index=False)
    LOGGER.info("wrote representation_comparison.csv (%d rows)", len(frame))

    best_model = (
        models.DEFAULT_MODEL
        if models.DEFAULT_MODEL in model_names
        else model_names[-1]
    )
    subset = frame[frame["model"] == best_model]
    if not subset.empty:
        scores = {
            target: {
                str(row["representation"]): float(row["roc_auc"])
                for _, row in group.iterrows()
            }
            for target, group in subset.groupby("target", sort=False)
        }
        plots.plot_representation_comparison(
            scores,
            figures_dir / "representation_comparison.png",
            metric_label=f"ROC AUC ({strategy} split, {best_model})",
        )
        LOGGER.info("wrote figures/representation_comparison.png")
    return frame


def run_applicability_domain(
    data: Sequence[TargetData],
    predictions: Sequence[CellPrediction],
    representation: str,
    model_name: str,
    strategy: str,
    seed: int,
    results_dir: Path,
    figures_dir: Path,
) -> None:
    """Measure and plot the applicability domain.

    Args:
        data: The loaded targets.
        predictions: Predictions from the headline matrix.
        representation: Representation whose bits define similarity.
        model_name: Model whose errors are examined.
        strategy: Split strategy to use.
        seed: Seed for the training-set sample used for the threshold.
        results_dir: Directory for the CSV outputs.
        figures_dir: Directory for the figures.
    """
    if featurise.get_representation(representation).kind != "fingerprint":
        LOGGER.warning(
            "applicability domain needs a binary fingerprint for Tanimoto "
            "similarity; %r is a dense descriptor block, so the stage is "
            "skipped",
            representation,
        )
        return

    by_target = {item.spec.name: item for item in data}
    bin_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    error_series: dict[str, list[tuple[float, float, int]]] = {}
    coverage_series: dict[str, list[tuple[float, float]]] = {}
    rng = np.random.default_rng(seed)

    for cell in predictions:
        if not (
            cell.representation == representation
            and cell.model == model_name
            and cell.strategy == strategy
            and cell.task == "regression"
        ):
            continue
        item = by_target[cell.target]
        matrix = item.features[representation]
        similarity = domain.nearest_neighbour_similarity(
            matrix[cell.test_index], matrix[cell.train_index]
        )
        errors = cell.y_pred - cell.y_true

        sample = cell.train_index
        if sample.size > DOMAIN_SAMPLE:
            sample = rng.choice(sample, DOMAIN_SAMPLE, replace=False)
        threshold = domain.domain_threshold(matrix[sample])

        bins = domain.error_vs_similarity(similarity, errors)
        error_series[cell.target] = [
            (b.mean_similarity, b.mean_absolute_error, b.n) for b in bins
        ]
        trend = domain.spearman_trend(
            [b.mean_similarity for b in bins],
            [b.mean_absolute_error for b in bins],
        )
        for item_bin in bins:
            bin_rows.append(
                {
                    "target": cell.target,
                    "model": model_name,
                    "split": strategy,
                    "similarity_lower": item_bin.lower,
                    "similarity_upper": item_bin.upper,
                    "n": item_bin.n,
                    "mean_similarity": round(item_bin.mean_similarity, 4),
                    "mae": round(item_bin.mean_absolute_error, 4),
                    "rmse": round(item_bin.rmse, 4),
                    "domain_threshold": round(threshold, 4),
                    "frac_out_of_domain": round(
                        float((similarity < threshold).mean()), 4
                    ),
                    "spearman_similarity_vs_mae": round(trend, 4),
                }
            )

        curve = domain.coverage_curve(
            similarity, cell.y_true, cell.y_pred, task="regression"
        )
        coverage_series[cell.target] = [(r["coverage"], r["rmse"]) for r in curve]
        for point in curve:
            coverage_rows.append(
                {
                    "target": cell.target,
                    "model": model_name,
                    "split": strategy,
                    "task": "regression",
                    **{k: round(v, 4) for k, v in point.items()},
                }
            )

    if bin_rows:
        pd.DataFrame(bin_rows).to_csv(
            results_dir / "applicability_domain.csv", index=False
        )
        plots.plot_error_vs_similarity(
            error_series, figures_dir / "error_vs_similarity.png"
        )
        LOGGER.info("wrote applicability_domain.csv and error_vs_similarity.png")
    if coverage_rows:
        pd.DataFrame(coverage_rows).to_csv(
            results_dir / "coverage_curve.csv", index=False
        )
        plots.plot_coverage_tradeoff(
            coverage_series, figures_dir / "coverage_tradeoff.png"
        )
        LOGGER.info("wrote coverage_curve.csv and coverage_tradeoff.png")


def run_screening(
    predictions: Sequence[CellPrediction],
    representation: str,
    model_name: str,
    strategy: str,
    seed: int,
    results_dir: Path,
    figures_dir: Path,
) -> None:
    """Write the enrichment curves, screening demo and calibration outputs.

    Args:
        predictions: Predictions from the headline matrix.
        representation: Representation to report on.
        model_name: Model whose ranking drives the screening demo.
        strategy: Split strategy to use.
        seed: Seed for the bootstrap intervals.
        results_dir: Directory for the CSV outputs.
        figures_dir: Directory for the figures.
    """
    curve_rows: list[dict[str, Any]] = []
    demo_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    curve_series: dict[str, list[tuple[float, float]]] = {}
    calibration_series: dict[str, tuple[list[float], list[float]]] = {}
    pooled: dict[str, tuple[list[float], list[float]]] = {}

    for cell in predictions:
        if not (
            cell.representation == representation
            and cell.strategy == strategy
            and cell.task == "classification"
        ):
            continue
        labels = cell.y_true.astype(int)

        # Pool across targets, per model, for the calibration figure.
        truths, probs = pooled.setdefault(cell.model, ([], []))
        truths.extend(labels.tolist())
        probs.extend(cell.y_pred.tolist())

        curve = evaluate.calibration_curve(labels, cell.y_pred)
        for index in range(curve.bin_centres.size):
            calibration_rows.append(
                {
                    "target": cell.target,
                    "model": cell.model,
                    "split": strategy,
                    "bin_centre": round(float(curve.bin_centres[index]), 3),
                    "mean_predicted": round(float(curve.mean_predicted[index]), 4),
                    "observed_fraction": round(
                        float(curve.observed_fraction[index]), 4
                    ),
                    "n": int(curve.counts[index]),
                    "ece": round(curve.ece, 4),
                }
            )

        if cell.model != model_name:
            continue

        for fraction in ENRICHMENT_GRID:
            curve_rows.append(
                {
                    "target": cell.target,
                    "model": model_name,
                    "split": strategy,
                    "fraction_screened": round(float(fraction), 5),
                    "recall": round(
                        evaluate.recall_at_fraction(labels, cell.y_pred, float(fraction)),
                        4,
                    ),
                    "enrichment_factor": round(
                        evaluate.enrichment_factor(
                            labels, cell.y_pred, float(fraction)
                        ),
                        3,
                    ),
                }
            )
        curve_series[cell.target] = [
            (
                float(f),
                evaluate.recall_at_fraction(labels, cell.y_pred, float(f)),
            )
            for f in ENRICHMENT_GRID
        ]

        row: dict[str, Any] = {
            "target": cell.target,
            "model": model_name,
            "representation": representation,
            "split": strategy,
            "n_test": int(labels.size),
            "n_actives": int(labels.sum()),
            "active_rate": round(float(labels.mean()), 4),
        }
        for fraction in (0.01, 0.05):
            tag = f"{fraction:.0%}".replace("%", "pct")
            row[f"n_screened_{tag}"] = max(1, int(round(fraction * labels.size)))
            row[f"hit_rate_{tag}"] = round(
                evaluate.hit_rate(labels, cell.y_pred, fraction), 4
            )
            row[f"recall_{tag}"] = round(
                evaluate.recall_at_fraction(labels, cell.y_pred, fraction), 4
            )
            point, lower, upper = evaluate.bootstrap_ci(
                labels, cell.y_pred, metric="ef", n_resamples=400,
                seed=seed, fraction=fraction,
            )
            row[f"ef_{tag}"] = round(point, 3)
            row[f"ef_{tag}_ci_low"] = round(lower, 3)
            row[f"ef_{tag}_ci_high"] = round(upper, 3)
            row[f"bedroc_{tag}"] = round(
                evaluate.bedroc(
                    labels, cell.y_pred, alpha=evaluate.bedroc_alpha(fraction)
                ),
                4,
            )
        auc, auc_low, auc_high = evaluate.bootstrap_ci(
            labels, cell.y_pred, metric="roc_auc", n_resamples=400, seed=seed
        )
        row.update(
            roc_auc=round(auc, 4),
            roc_auc_ci_low=round(auc_low, 4),
            roc_auc_ci_high=round(auc_high, 4),
            ece=round(curve.ece, 4),
        )
        demo_rows.append(row)

    if curve_rows:
        pd.DataFrame(curve_rows).to_csv(
            results_dir / "enrichment_curves.csv", index=False
        )
        plots.plot_enrichment_curves(
            curve_series, figures_dir / "enrichment_curves.png"
        )
        LOGGER.info("wrote enrichment_curves.csv and enrichment_curves.png")
    if demo_rows:
        pd.DataFrame(demo_rows).to_csv(
            results_dir / "screening_demo.csv", index=False
        )
        LOGGER.info("wrote screening_demo.csv")
    if calibration_rows:
        pd.DataFrame(calibration_rows).to_csv(
            results_dir / "calibration.csv", index=False
        )
        for model, (truths, probs) in pooled.items():
            curve = evaluate.calibration_curve(truths, probs)
            calibration_series[model] = (
                curve.mean_predicted.tolist(),
                curve.observed_fraction.tolist(),
            )
        plots.plot_calibration(calibration_series, figures_dir / "calibration.png")
        LOGGER.info("wrote calibration.csv and calibration.png")


def run_nulls(
    data: Sequence[TargetData],
    representation: str,
    model_name: str,
    strategy: str,
    test_size: float,
    n_permutations: int,
    seed: int,
    results_dir: Path,
    figures_dir: Path,
) -> pd.DataFrame:
    """Build and record y-scrambling null distributions.

    Args:
        data: The loaded targets.
        representation: Representation slug to use.
        model_name: Model to scramble.
        strategy: Split strategy to use.
        test_size: Target test fraction.
        n_permutations: Label permutations per target.
        seed: Seed for the permutations and the model.
        results_dir: Directory for the CSV output.
        figures_dir: Directory for the figure.

    Returns:
        The null-model summary table.
    """
    from sklearn.metrics import roc_auc_score

    kind = featurise.get_representation(representation).kind
    rows: list[dict[str, Any]] = []
    panels: list[tuple[str, list[float], float]] = []

    for item in data:
        matrix = item.features[representation]
        split = item.split(strategy, test_size, seed)
        started = time.perf_counter()
        result = nulls.y_scramble(
            models.build_model(model_name, "classification", kind, seed=seed),
            matrix[split.train_index],
            item.labels[split.train_index],
            matrix[split.test_index],
            item.labels[split.test_index],
            lambda truth, prediction: float(roc_auc_score(truth, prediction)),
            task="classification",
            metric="roc_auc",
            n_permutations=n_permutations,
            seed=seed,
        )
        rows.append(
            {
                "target": item.spec.name,
                "model": model_name,
                "representation": representation,
                "split": strategy,
                **{
                    k: (round(v, 4) if isinstance(v, float) else v)
                    for k, v in result.to_dict().items()
                },
                "seconds": round(time.perf_counter() - started, 1),
            }
        )
        panels.append((item.spec.name, result.null_scores.tolist(), result.observed))
        LOGGER.info(
            "%s null: observed %.3f vs null %.3f +/- %.3f, p=%.3f (%.0fs)",
            item.spec.name,
            result.observed,
            result.null_mean,
            result.null_std,
            result.p_value,
            time.perf_counter() - started,
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(results_dir / "y_scrambling.csv", index=False)
    if panels:
        plots.plot_y_scrambling(panels, figures_dir / "y_scrambling.png")
    LOGGER.info("wrote y_scrambling.csv and y_scrambling.png")
    return frame


def run_chemical_space(
    data: Sequence[TargetData],
    representation: str,
    strategies: Sequence[str],
    seed: int,
    figures_dir: Path,
) -> None:
    """Draw a PCA of chemical space coloured by each split strategy.

    Args:
        data: The loaded targets; the largest is used.
        representation: Fingerprint representation to embed.
        strategies: Strategies to show as panels.
        seed: Seed for the PCA and any subsampling.
        figures_dir: Directory for the figure.
    """
    from sklearn.decomposition import PCA

    if not data:
        return
    if featurise.get_representation(representation).kind != "fingerprint":
        LOGGER.warning(
            "chemical-space figure expects a fingerprint; skipping %r",
            representation,
        )
        return
    item = max(data, key=lambda d: len(d.frame))
    matrix = item.features[representation]
    rng = np.random.default_rng(seed)
    rows = np.arange(matrix.shape[0])
    if rows.size > PCA_TARGET_LIMIT:
        rows = np.sort(rng.choice(rows, PCA_TARGET_LIMIT, replace=False))

    pca = PCA(n_components=2, random_state=seed)
    coordinates = pca.fit_transform(matrix[rows])

    labels: dict[str, list[str]] = {}
    for strategy in strategies:
        if strategy not in item.split_cache:
            continue
        split = item.split_cache[strategy]
        membership = np.full(matrix.shape[0], "unused", dtype=object)
        membership[split.train_index] = "train"
        membership[split.test_index] = "test"
        labels[strategy] = membership[rows].tolist()

    if labels:
        plots.plot_chemical_space(
            coordinates,
            labels,
            figures_dir / "chemical_space_pca.png",
            explained=(
                float(pca.explained_variance_ratio_[0]),
                float(pca.explained_variance_ratio_[1]),
            ),
            target_label=f"{item.spec.name} ({len(rows)} compounds)",
        )
        LOGGER.info("wrote figures/chemical_space_pca.png")


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser for the study driver."""
    parser = argparse.ArgumentParser(
        description="Run the ligand-activity machine learning study.",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument(
        "--targets", nargs="+", metavar="SLUG",
        help="subset of target slugs (default: every cached target that "
             f"clears {chembl.MIN_COMPOUNDS} compounds)",
    )
    parser.add_argument(
        "--representations", nargs="+", metavar="NAME",
        default=[r.name for r in featurise.REPRESENTATIONS],
        help="representations for the representation sweep",
    )
    parser.add_argument(
        "--models", nargs="+", metavar="NAME",
        default=[m.name for m in models.MODELS],
        help="models for the headline split comparison",
    )
    parser.add_argument(
        "--splits", nargs="+", metavar="NAME",
        default=list(splits.SPLIT_STRATEGIES),
        help="split strategies to compare",
    )
    parser.add_argument(
        "--tasks", nargs="+", metavar="NAME", default=list(models.TASKS),
        help="framings to run",
    )
    parser.add_argument(
        "--threshold", type=float, default=evaluate.ACTIVITY_THRESHOLD,
        help="pChEMBL activity threshold for classification "
             f"(default: {evaluate.ACTIVITY_THRESHOLD:g}, i.e. 1 uM)",
    )
    parser.add_argument(
        "--test-size", type=float, default=splits.DEFAULT_TEST_SIZE,
        help=f"target test fraction (default: {splits.DEFAULT_TEST_SIZE})",
    )
    parser.add_argument(
        "--repeat-seeds", type=int, default=5,
        help="split re-draws for the seed-variability stage",
    )
    parser.add_argument(
        "--decoy-actives", type=int, default=150,
        help="actives sampled per target for the decoy screen",
    )
    parser.add_argument(
        "--decoy-ratio", type=int, default=decoys.DEFAULT_DECOY_RATIO,
        help="decoys attached per active",
    )
    parser.add_argument(
        "--calibration-fraction", type=float, default=0.25,
        help="share of training held out to calibrate the conformal predictor",
    )
    parser.add_argument(
        "--permutations", type=int, default=nulls.N_PERMUTATIONS,
        help=f"y-scrambling permutations (default: {nulls.N_PERMUTATIONS})",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument(
        "--fast", action="store_true",
        help="a small subset for smoke testing: two targets, two models, "
             "one representation sweep, three permutations",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="recompute features instead of using data/features/",
    )
    parser.add_argument(
        "--skip", nargs="+", default=[],
        choices=["representations", "domain", "screening", "nulls", "space", "repeats", "decoys", "conformal", "cutoffs"],
        help="stages to skip",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``python scripts/run_study.py``.

    Args:
        argv: Argument vector; defaults to the tail of :data:`sys.argv`.

    Returns:
        A process exit status.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    # LinearSVR on sparse fingerprints warns about its iteration budget on
    # some targets; the fit is still used and reported as-is.
    warnings.filterwarnings("ignore", category=UserWarning)
    from sklearn.exceptions import ConvergenceWarning

    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    specs = chembl.available_targets(args.data_dir)
    if args.targets:
        wanted = set(args.targets)
        specs = [s for s in specs if s.name in wanted]
    representations = list(dict.fromkeys(args.representations))
    model_names = list(dict.fromkeys(args.models))
    strategies = list(dict.fromkeys(args.splits))
    tasks = list(dict.fromkeys(args.tasks))
    permutations = args.permutations

    if args.fast:
        specs = specs[:2]
        model_names = ["linear", "random_forest"]
        representations = [featurise.DEFAULT_REPRESENTATION, "maccs"]
        permutations = 3

    if len(specs) < 2:
        print(
            "error: need at least 2 usable cached targets; run "
            "`python -m ligandml.chembl --fetch-all` first",
            file=sys.stderr,
        )
        return 1

    default_representation = (
        featurise.DEFAULT_REPRESENTATION
        if featurise.DEFAULT_REPRESENTATION in representations
        else representations[0]
    )
    needed = list(dict.fromkeys([default_representation, *representations]))
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    LOGGER.info(
        "targets=%s models=%s splits=%s tasks=%s",
        [s.name for s in specs], model_names, strategies, tasks,
    )

    data = [
        load_and_featurise(
            spec, needed, args.data_dir, args.threshold, use_cache=not args.no_cache
        )
        for spec in specs
    ]
    write_dataset_tables(data, args.data_dir, args.results_dir, args.threshold)

    comparison, predictions = run_split_comparison(
        data, default_representation, model_names, strategies, tasks,
        args.test_size, args.seed, args.results_dir,
    )
    default_model = (
        models.DEFAULT_MODEL if models.DEFAULT_MODEL in model_names else model_names[-1]
    )
    write_split_gap(comparison, args.results_dir, args.figures_dir)
    write_headline_figures(
        comparison, data, default_model, args.figures_dir, args.results_dir
    )

    if "decoys" not in args.skip and len(data) > 1:
        LOGGER.info("[decoys] property-matched screening library")
        run_decoy_screen(
            data, args.decoy_actives, args.decoy_ratio, args.test_size,
            args.seed, default_model, args.results_dir,
        )

    if "conformal" not in args.skip and "classification" in tasks:
        LOGGER.info("[conformal] prediction sets with a coverage guarantee")
        run_conformal(
            data, featurise.DEFAULT_REPRESENTATION, default_model,
            [s for s in strategies if s in ("random", "scaffold")] or strategies[:1],
            args.test_size, args.seed, args.calibration_fraction, args.results_dir,
        )

    if "cutoffs" not in args.skip and "time" in strategies:
        LOGGER.info("[cutoffs] does the time gap survive moving the line?")
        run_time_cutoff_sensitivity(
            data, featurise.DEFAULT_REPRESENTATION, default_model,
            args.results_dir, args.seed,
        )

    if "repeats" not in args.skip and args.repeat_seeds > 1:
        LOGGER.info("[repeats] %d split re-draws per target", args.repeat_seeds)
        run_repeated_splits(
            data,
            featurise.DEFAULT_REPRESENTATION,
            default_model,
            "classification" if "classification" in tasks else tasks[0],
            args.test_size,
            [args.seed + i for i in range(args.repeat_seeds)],
            args.results_dir,
        )

    representation_frame = pd.DataFrame()
    if "representations" not in args.skip and len(representations) > 1:
        sweep_models = [m for m in REPRESENTATION_MODELS if m in model_names]
        representation_frame = run_representation_comparison(
            data, representations, sweep_models or model_names[:1], "scaffold",
            args.test_size, args.seed, args.results_dir, args.figures_dir,
            reuse=predictions,
        )

    if "domain" not in args.skip and "regression" in tasks:
        run_applicability_domain(
            data, predictions, default_representation, default_model, "scaffold",
            args.seed, args.results_dir, args.figures_dir,
        )
    if "screening" not in args.skip and "classification" in tasks:
        run_screening(
            predictions, default_representation, default_model, "scaffold",
            args.seed, args.results_dir, args.figures_dir,
        )

    null_frame = pd.DataFrame()
    if "nulls" not in args.skip and "classification" in tasks:
        null_model = NULL_MODEL if NULL_MODEL in model_names else model_names[-1]
        null_frame = run_nulls(
            data, default_representation, null_model, "scaffold", args.test_size,
            permutations, args.seed, args.results_dir, args.figures_dir,
        )
    if "space" not in args.skip:
        run_chemical_space(
            data, default_representation, strategies, args.seed, args.figures_dir
        )

    elapsed = time.perf_counter() - started
    summary = {
        "seed": args.seed,
        "activity_threshold": args.threshold,
        "test_size": args.test_size,
        "targets": [
            {
                "name": item.spec.name,
                "chembl_id": item.spec.chembl_id,
                "n_compounds": len(item.frame),
                "active_rate": round(float(item.labels.mean()), 4),
            }
            for item in data
        ],
        "representations": representations,
        "default_representation": default_representation,
        "models": model_names,
        "default_model": default_model,
        "splits": strategies,
        "tasks": tasks,
        "n_permutations": permutations if not null_frame.empty else 0,
        "n_cells_fitted": len(comparison) + len(representation_frame),
        "wall_clock_seconds": round(elapsed, 1),
        "versions": _versions(),
    }
    (args.results_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    LOGGER.info("done in %.1f min", elapsed / 60.0)
    print(json.dumps(summary, indent=2))
    return 0


def _versions() -> dict[str, str]:
    """Return the versions of the libraries the results depend on."""
    import matplotlib
    import rdkit
    import sklearn

    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "rdkit": rdkit.__version__,
        "matplotlib": matplotlib.__version__,
    }


if __name__ == "__main__":
    raise SystemExit(main())
