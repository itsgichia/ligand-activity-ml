"""Tests for the split-gap uncertainty summaries.

The point of this module is to put an honest error bar on the headline
claim, so a wrong interval here would be worse than no interval at all.
Every expected value below is worked out by hand rather than compared
against another implementation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ligandml import uncertainty


class TestTCritical:
    """The inlined table replaces a SciPy dependency, so it has to be right."""

    def test_known_values(self) -> None:
        assert uncertainty.t_critical_95(1) == pytest.approx(12.706)
        assert uncertainty.t_critical_95(19) == pytest.approx(2.093)
        assert uncertainty.t_critical_95(30) == pytest.approx(2.042)

    def test_falls_back_to_normal_above_the_table(self) -> None:
        assert uncertainty.t_critical_95(500) == pytest.approx(1.959964)

    def test_rejects_zero_degrees_of_freedom(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            uncertainty.t_critical_95(0)


class TestRankWithTies:
    """Average ranks within a tie group, which the variance correction needs."""

    def test_no_ties(self) -> None:
        ranks = uncertainty._rank_with_ties(np.array([3.0, 1.0, 2.0]))
        assert list(ranks) == [3.0, 1.0, 2.0]

    def test_ties_share_the_average_rank(self) -> None:
        # Values 1, 2, 2, 4 occupy ranks 1, 2, 3, 4; the two 2s average to 2.5.
        ranks = uncertainty._rank_with_ties(np.array([1.0, 2.0, 2.0, 4.0]))
        assert list(ranks) == [1.0, 2.5, 2.5, 4.0]


class TestWilcoxon:
    """Signed-rank, checked against statistics computed by hand."""

    def test_all_positive_gives_the_minimum_statistic(self) -> None:
        """With every difference positive, the negative rank sum is zero."""
        statistic, p = uncertainty.wilcoxon_signed_rank([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        assert statistic == 0.0
        assert p < 0.05

    def test_statistic_is_the_smaller_rank_sum(self) -> None:
        """Differences -1, 2, 3, 4, 5, 6 rank 1..6; the negatives sum to 1."""
        statistic, _ = uncertainty.wilcoxon_signed_rank([-1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        assert statistic == 1.0

    def test_symmetric_differences_are_not_significant(self) -> None:
        statistic, p = uncertainty.wilcoxon_signed_rank(
            [1.0, -1.0, 2.0, -2.0, 3.0, -3.0]
        )
        assert statistic == pytest.approx(10.5)
        assert p > 0.5

    def test_zero_differences_are_dropped(self) -> None:
        """Zeros carry no sign, so they leave the test rather than bias it."""
        with_zeros = uncertainty.wilcoxon_signed_rank([0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        without = uncertainty.wilcoxon_signed_rank([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        assert with_zeros == without

    def test_too_few_values_refuses_to_claim_significance(self) -> None:
        """The normal approximation is unusable below six differences."""
        _, p = uncertainty.wilcoxon_signed_rank([0.4, 0.5, 0.6])
        assert p == 1.0

    def test_empty_input(self) -> None:
        assert uncertainty.wilcoxon_signed_rank([]) == (0.0, 1.0)


class TestSummariseGaps:
    """The headline interval."""

    def test_hand_computed_mean_and_interval(self) -> None:
        """Gaps 0.10, 0.20, 0.30, 0.40, 0.50: mean 0.30, sd 0.158114.

        se = 0.158114 / sqrt(5) = 0.0707107, and t(4) = 2.776, so the
        half-width is 0.196293 and the interval is 0.103707 to 0.496293.
        """
        summary = uncertainty.summarise_gaps(
            [0.1, 0.2, 0.3, 0.4, 0.5], "random to scaffold", "roc_auc"
        )
        assert summary.n == 5
        assert summary.mean == pytest.approx(0.30)
        assert summary.sd == pytest.approx(0.158114, abs=1e-6)
        assert summary.se == pytest.approx(0.0707107, abs=1e-6)
        assert summary.ci_low == pytest.approx(0.103707, abs=1e-5)
        assert summary.ci_high == pytest.approx(0.496293, abs=1e-5)
        assert summary.median == pytest.approx(0.3)

    def test_counts_and_names_the_contradicting_cells(self) -> None:
        summary = uncertainty.summarise_gaps(
            [0.1, 0.2, -0.05, 0.3],
            "random to time",
            "roc_auc",
            labels=["a/x", "b/x", "adrb2/rf", "c/x"],
        )
        assert summary.n_positive == 3
        assert summary.n_negative == 1
        assert summary.exceptions == ("adrb2/rf",)

    def test_drops_non_finite_gaps(self) -> None:
        summary = uncertainty.summarise_gaps(
            [0.2, float("nan"), 0.4, float("inf")], "c", "m"
        )
        assert summary.n == 2
        assert summary.mean == pytest.approx(0.3)

    def test_single_observation_has_no_width(self) -> None:
        summary = uncertainty.summarise_gaps([0.25], "c", "m")
        assert summary.sd == 0.0
        assert summary.ci_low == summary.ci_high == pytest.approx(0.25)

    def test_requires_at_least_one_finite_value(self) -> None:
        with pytest.raises(ValueError, match="at least one finite gap"):
            uncertainty.summarise_gaps([float("nan")], "c", "m")

    def test_as_row_is_flat_and_rounded(self) -> None:
        row = uncertainty.summarise_gaps([0.1, 0.2, 0.3], "c", "roc_auc").as_row()
        assert row["comparison"] == "c"
        assert row["n_cells"] == 3
        assert row["mean_gap"] == 0.2
        assert all(not isinstance(v, (list, dict)) for v in row.values())


class TestSummariseRepeats:
    """Seed-to-seed spread, the error bar a benchmark normally quotes."""

    def test_mean_sd_and_gap(self) -> None:
        rows = uncertainty.summarise_repeats(
            {"random": [0.90, 0.88, 0.92], "scaffold": [0.80, 0.78, 0.82]}
        )
        by = {r["strategy"]: r for r in rows}
        assert by["random"]["mean"] == pytest.approx(0.90)
        assert by["scaffold"]["mean"] == pytest.approx(0.80)
        # Every seed differs by exactly 0.10, so the gap has no spread.
        assert by["scaffold"]["mean_gap_vs_reference"] == pytest.approx(0.10)
        assert by["scaffold"]["sd_gap_vs_reference"] == pytest.approx(0.0)

    def test_reference_row_carries_no_gap(self) -> None:
        rows = uncertainty.summarise_repeats({"random": [0.9, 0.91]})
        assert "mean_gap_vs_reference" not in rows[0]

    def test_mismatched_seed_counts_skip_the_gap(self) -> None:
        rows = uncertainty.summarise_repeats(
            {"random": [0.9, 0.9, 0.9], "scaffold": [0.8]}
        )
        scaffold = next(r for r in rows if r["strategy"] == "scaffold")
        assert "mean_gap_vs_reference" not in scaffold

    def test_missing_reference(self) -> None:
        with pytest.raises(KeyError, match="reference strategy"):
            uncertainty.summarise_repeats({"scaffold": [0.8]}, reference="random")


class TestTimeCutoffGrid:
    """A time split has no seed, so its sensitivity is over the cutoff."""

    def test_quantile_cutoffs_are_ascending_and_distinct(self) -> None:
        years = list(range(2000, 2020))
        cutoffs = uncertainty.time_cutoff_grid(years, fractions=(0.1, 0.25, 0.5))
        assert cutoffs == sorted(set(cutoffs))
        assert all(2000 <= c <= 2020 for c in cutoffs)

    def test_a_later_cutoff_holds_out_less(self) -> None:
        years = list(range(2000, 2020))
        cutoffs = uncertainty.time_cutoff_grid(years, fractions=(0.2, 0.4))
        held_out = [sum(1 for y in years if y >= c) for c in cutoffs]
        assert held_out == sorted(held_out, reverse=True)

    def test_ignores_missing_years(self) -> None:
        cutoffs = uncertainty.time_cutoff_grid(
            [2001, float("nan"), 2005, None, 2010], fractions=(0.5,)
        )
        assert len(cutoffs) == 1
        assert math.isfinite(cutoffs[0])

    def test_requires_a_finite_year(self) -> None:
        with pytest.raises(ValueError, match="at least one finite year"):
            uncertainty.time_cutoff_grid([float("nan"), None])
