"""Tests for the figure module.

Figures are checked for the things that can break silently in a headless
run: every function writes a non-trivial PNG, the colour assignment is
stable across figures, and bad input is rejected rather than producing an
empty chart that looks fine in a thumbnail.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ligandml import plots


class TestPalette:
    """The visual system the README depends on."""

    def test_split_colours_are_fixed_and_distinct(self) -> None:
        assert set(plots.SPLIT_COLOURS) == {"random", "scaffold", "time"}
        assert len(set(plots.SPLIT_COLOURS.values())) == 3

    def test_split_colours_come_from_the_categorical_order(self) -> None:
        """So that "orange means scaffold split" holds everywhere."""
        assert plots.SPLIT_COLOURS["random"] == plots.CATEGORICAL[0]
        assert plots.SPLIT_COLOURS["scaffold"] == plots.CATEGORICAL[1]
        assert plots.SPLIT_COLOURS["time"] == plots.CATEGORICAL[2]

    def test_categorical_slots_are_unique_hex(self) -> None:
        assert len(set(plots.CATEGORICAL)) == len(plots.CATEGORICAL)
        for colour in (*plots.CATEGORICAL, plots.SURFACE, plots.NULL_COLOUR):
            assert colour.startswith("#") and len(colour) == 7

    def test_surface_is_opaque_light(self) -> None:
        """An opaque surface keeps the PNGs legible on a dark README page."""
        assert plots.SURFACE.lower() == "#fcfcfb"


def _written(path: Path) -> bool:
    """True when a figure was written and is not a blank canvas."""
    return path.exists() and path.stat().st_size > 5_000


class TestFigures:
    """Each figure function, on minimal but realistic input."""

    def test_split_strategies(self, tmp_path: Path) -> None:
        path = plots.plot_split_strategies(
            {
                "herg": {"random": 0.91, "scaffold": 0.80, "time": 0.78},
                "dpp4": {"random": 0.95, "scaffold": 0.86, "time": 0.84},
            },
            tmp_path / "split.png",
            subtitle="a subtitle",
        )
        assert _written(path)

    def test_split_strategies_without_chance_line(self, tmp_path: Path) -> None:
        path = plots.plot_split_strategies(
            {"herg": {"random": 4.2, "scaffold": 2.1}},
            tmp_path / "ef.png",
            metric_label="EF at 1%",
            chance=1.0,
        )
        assert _written(path)

    def test_split_gap(self, tmp_path: Path) -> None:
        path = plots.plot_split_gap(
            [
                {"label": "herg / rf", "random": 0.91, "scaffold": 0.80, "time": 0.78},
                {"label": "dpp4 / rf", "random": 0.95, "scaffold": 0.86},
            ],
            tmp_path / "gap.png",
        )
        assert _written(path)

    def test_split_gap_sorts_without_a_time_column(self, tmp_path: Path) -> None:
        """Subsetting --splits must not break the sort key."""
        path = plots.plot_split_gap(
            [{"label": "only random+scaffold", "random": 0.9, "scaffold": 0.7}],
            tmp_path / "gap2.png",
        )
        assert _written(path)

    def test_representation_comparison(self, tmp_path: Path) -> None:
        path = plots.plot_representation_comparison(
            {
                "herg": {"rdkit2d": 0.75, "ecfp4_2048": 0.80, "maccs": 0.74},
                "dpp4": {"rdkit2d": 0.81, "ecfp4_2048": 0.86, "maccs": 0.79},
            },
            tmp_path / "reps.png",
        )
        assert _written(path)

    def test_error_vs_similarity(self, tmp_path: Path) -> None:
        path = plots.plot_error_vs_similarity(
            {
                "herg": [(0.15, 0.92, 120), (0.35, 0.78, 400), (0.65, 0.55, 300)],
                "dpp4": [(0.18, 1.05, 90), (0.40, 0.81, 300)],
            },
            tmp_path / "ad.png",
        )
        assert _written(path)

    def test_coverage_tradeoff(self, tmp_path: Path) -> None:
        path = plots.plot_coverage_tradeoff(
            {
                "herg": [(1.0, 0.86), (0.7, 0.78), (0.4, 0.69)],
                "dpp4": [(1.0, 0.92), (0.6, 0.80)],
            },
            tmp_path / "cov.png",
        )
        assert _written(path)

    def test_enrichment_curves(self, tmp_path: Path) -> None:
        fractions = np.logspace(-2.7, 0.0, 40)
        path = plots.plot_enrichment_curves(
            {
                "herg": [(float(f), float(min(1.0, f * 4))) for f in fractions],
                "dpp4": [(float(f), float(min(1.0, f * 7))) for f in fractions],
            },
            tmp_path / "enrich.png",
        )
        assert _written(path)

    def test_calibration(self, tmp_path: Path) -> None:
        path = plots.plot_calibration(
            {
                "random_forest": (
                    [0.1, 0.3, 0.5, 0.7, 0.9],
                    [0.08, 0.26, 0.5, 0.74, 0.93],
                ),
                "linear": ([0.1, 0.3, 0.5, 0.7, 0.9], [0.2, 0.35, 0.45, 0.6, 0.7]),
            },
            tmp_path / "cal.png",
        )
        assert _written(path)

    def test_y_scrambling(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(0)
        path = plots.plot_y_scrambling(
            [
                ("herg", rng.normal(0.5, 0.02, 20).tolist(), 0.80),
                ("dpp4", rng.normal(0.5, 0.02, 20).tolist(), 0.86),
            ],
            tmp_path / "null.png",
        )
        assert _written(path)

    def test_y_scrambling_single_panel(self, tmp_path: Path) -> None:
        path = plots.plot_y_scrambling(
            [("herg", [0.49, 0.51, 0.5], 0.8)], tmp_path / "null1.png"
        )
        assert _written(path)

    def test_leakage(self, tmp_path: Path) -> None:
        path = plots.plot_leakage(
            {
                "herg": {"random": 0.60, "scaffold": 0.0, "time": 0.04},
                "dpp4": {"random": 0.72, "scaffold": 0.0, "time": 0.09},
            },
            tmp_path / "leak.png",
        )
        assert _written(path)

    def test_chemical_space(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(1)
        coordinates = rng.normal(size=(600, 2))
        labels = {
            strategy: ["train"] * 480 + ["test"] * 120
            for strategy in ("random", "scaffold", "time")
        }
        path = plots.plot_chemical_space(
            coordinates,
            labels,
            tmp_path / "space.png",
            explained=(0.12, 0.07),
            target_label="herg",
        )
        assert _written(path)

    def test_chemical_space_with_unused_rows(self, tmp_path: Path) -> None:
        """Time splits drop compounds with no year; those must simply not be
        drawn rather than crashing or being mislabelled.
        """
        rng = np.random.default_rng(2)
        path = plots.plot_chemical_space(
            rng.normal(size=(300, 2)),
            {"time": ["train"] * 200 + ["test"] * 80 + ["unused"] * 20},
            tmp_path / "space2.png",
        )
        assert _written(path)


class TestFigureValidation:
    """Bad input is refused, not silently drawn as an empty chart."""

    def test_empty_inputs_rejected(self, tmp_path: Path) -> None:
        for function in (
            plots.plot_split_strategies,
            plots.plot_representation_comparison,
            plots.plot_error_vs_similarity,
            plots.plot_coverage_tradeoff,
            plots.plot_enrichment_curves,
            plots.plot_calibration,
            plots.plot_leakage,
        ):
            with pytest.raises(ValueError, match="nothing to plot"):
                function({}, tmp_path / "x.png")
        with pytest.raises(ValueError, match="nothing to plot"):
            plots.plot_split_gap([], tmp_path / "x.png")
        with pytest.raises(ValueError, match="nothing to plot"):
            plots.plot_y_scrambling([], tmp_path / "x.png")

    def test_chemical_space_shape_is_checked(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match=r"must be \(n, 2\)"):
            plots.plot_chemical_space(
                np.zeros((10, 3)), {"random": ["train"] * 10}, tmp_path / "x.png"
            )

    def test_chemical_space_label_count_is_checked(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="labels for"):
            plots.plot_chemical_space(
                np.zeros((10, 2)), {"random": ["train"] * 9}, tmp_path / "x.png"
            )

    def test_parent_directories_are_created(self, tmp_path: Path) -> None:
        path = plots.plot_leakage(
            {"herg": {"random": 0.6, "scaffold": 0.0}},
            tmp_path / "deep" / "nested" / "leak.png",
        )
        assert _written(path)
