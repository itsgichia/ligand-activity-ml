"""Figures for the study.

All figures share one visual system so they can be read side by side in the
README: a single light surface that stays legible on a dark page, recessive
grid and axes, thin marks, a legend whenever more than one series is drawn,
and one fixed categorical colour order.

The colour order is fixed rather than cycled, and the first three slots -
blue, orange, aqua - carry the three split strategies in every figure that
mentions them, so "orange means scaffold split" holds across the whole
repository. The palette is colour-vision-deficiency safe on adjacent pairs,
and every figure has a CSV counterpart in ``results/`` so no reading
depends on colour alone.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import matplotlib
import numpy as np

matplotlib.use("Agg")  # Figures are written to disk, never displayed.

import matplotlib.pyplot as plt  # noqa: E402 - must follow the backend choice
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

SURFACE: Final[str] = "#fcfcfb"
"""Chart surface. Opaque, so the PNGs read on a dark README background too."""

TEXT_PRIMARY: Final[str] = "#0b0b0b"
TEXT_SECONDARY: Final[str] = "#52514e"
GRID: Final[str] = "#dedcd5"

CATEGORICAL: Final[tuple[str, ...]] = (
    "#3f6fa8",  # 1 slate blue
    "#c0612f",  # 2 rust
    "#13906c",  # 3 pine
    "#b08a22",  # 4 bronze
    "#8f5f9e",  # 5 plum
    "#60802c",  # 6 olive
    "#5f63b0",  # 7 indigo
    "#ad4038",  # 8 brick
)
"""Fixed categorical order. Never cycled; a ninth series folds into Other.

Muted earth tones rather than saturated primaries: several of these figures
put five series on one axes, and at full chroma that reads as a jumble. The
set still clears every gate on the adjacent pairlist that bars and lines
use - worst adjacent CVD dE 9.0 (protan, OKLab x100, >= 8 target), worst
adjacent normal-vision dE 16.8 (>= 15 floor), every slot above the chroma
floor and at or above 3:1 contrast with the surface. The first three slots,
which carry the split strategies, also clear the stricter all-pairs gate the
scatter figures need.
"""

#: Colour per split strategy, fixed across every figure in the study.
SPLIT_COLOURS: Final[dict[str, str]] = {
    "random": CATEGORICAL[0],
    "scaffold": CATEGORICAL[1],
    "time": CATEGORICAL[2],
}

NULL_COLOUR: Final[str] = "#9a9892"
"""Neutral grey for null distributions and reference lines."""

DPI: Final[int] = 170
"""Resolution for the written PNGs."""


def _apply_style() -> None:
    """Set the matplotlib defaults shared by every figure."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.titlecolor": TEXT_PRIMARY,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "axes.labelsize": 9.5,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "legend.labelcolor": TEXT_SECONDARY,
            "lines.linewidth": 2.0,
            "lines.markersize": 5.0,
            "font.size": 9.5,
            "figure.titlesize": 12,
            "figure.titleweight": "semibold",
        }
    )


def _tidy(axes: plt.Axes, xgrid: bool = False) -> None:
    """Strip chart junk: drop the top and right spines, recede the grid.

    Args:
        axes: The axes to tidy.
        xgrid: Whether to keep vertical grid lines as well as horizontal.
    """
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    axes.grid(axis="y", color=GRID, linewidth=0.7)
    # Matplotlib treats any styling kwarg as "turn the grid on", so the off
    # case has to be a bare call.
    if xgrid:
        axes.grid(axis="x", color=GRID, linewidth=0.7)
    else:
        axes.grid(axis="x", visible=False)


def _title(
    axes: plt.Axes,
    headline: str,
    subtitle: str = "",
    pad: float = 10.0,
    subtitle_y: float | None = 1.0,
) -> None:
    """Set a bold headline with an optional quieter subtitle beneath it.

    Args:
        axes: The axes to title.
        headline: The bold first line.
        subtitle: A smaller, secondary-ink second line.
        pad: Padding between the title block and the axes.
        subtitle_y: Axes-fraction height for the subtitle, so a caller that
            puts a legend above the axes can slot the subtitle above it.
            ``None`` places the subtitle directly under the headline, which
            is what a figure with nothing else above the axes wants.
    """
    if not subtitle:
        axes.set_title(headline, pad=pad)
        return
    if subtitle_y is None:
        axes.set_title(f"{headline}\n", pad=pad)
        axes.text(
            0.5, 1.0, subtitle, transform=axes.transAxes, ha="center",
            va="bottom", fontsize=8.6, color=TEXT_SECONDARY,
        )
        return
    axes.set_title(headline, pad=pad + 13.0)
    axes.text(
        0.5, subtitle_y, subtitle, transform=axes.transAxes, ha="center",
        va="bottom", fontsize=8.6, color=TEXT_SECONDARY,
    )


#: Height of one legend row and of the subtitle line, in inches. Used to keep
#: a title block clear of a legend whatever height that legend turns out to be.
_LEGEND_ROW_IN: Final[float] = 0.17
_LEGEND_GAP_IN: Final[float] = 0.08
_SUBTITLE_IN: Final[float] = 0.30


def _header(
    axes: plt.Axes,
    headline: str,
    subtitle: str = "",
    *,
    ncols: int = 1,
    **legend_kw: Any,
) -> None:
    """Draw a legend above the axes with the title block clear above it.

    The legend wraps to as many rows as its entries need, so the space it
    occupies is measured rather than assumed: five series at three columns
    take two rows, and a subtitle pinned at a fixed axes fraction lands on
    top of the second one. Heights are worked out in inches and converted
    to axes fractions, so a short figure gets the same visual spacing as a
    tall one.

    Args:
        axes: The axes to title and add a legend to.
        headline: The bold first line.
        subtitle: A smaller, secondary-ink second line.
        ncols: Number of legend columns.
        **legend_kw: Passed through to ``Axes.legend``.
    """
    ncols = max(int(ncols), 1)
    legend = axes.legend(
        ncols=ncols, loc="lower center", bbox_to_anchor=(0.5, 1.005), **legend_kw
    )
    rows = -(-len(legend.get_texts()) // ncols)  # ceiling division
    figure = axes.get_figure()
    axes_in = max(
        float(axes.get_position().height) * float(figure.get_size_inches()[1]), 0.1
    )
    legend_in = rows * _LEGEND_ROW_IN + _LEGEND_GAP_IN
    subtitle_y = 1.005 + legend_in / axes_in
    # _title adds 13pt of its own, so take that back out of the pad.
    pad = (legend_in + _SUBTITLE_IN) * 72.0 - 13.0
    _title(axes, headline, subtitle, pad=pad, subtitle_y=subtitle_y)


def _chance_label(
    axes: plt.Axes, chance: float, n_groups: int, text: str = "no skill"
) -> None:
    """Label a chance line in clear margin to the right of the last bar group.

    Putting the words on the line itself means either the line strikes
    through them or an opaque box behind them punches a hole in whichever
    bar happens to be underneath. Widening the axes instead gives the label
    somewhere empty to sit, with the dashed line running beneath it.

    Args:
        axes: The axes holding the bars and the chance line.
        chance: Y value of the chance line.
        n_groups: Number of bar groups, which sets the right-hand edge.
        text: The label.
    """
    right = n_groups - 0.5 + 0.78
    axes.set_xlim(-0.62, right)
    axes.text(
        right - 0.06,
        chance,
        text,
        color=TEXT_SECONDARY,
        fontsize=7.6,
        va="bottom",
        ha="right",
    )


def _suptitle(figure: Figure, headline: str, subtitle: str = "") -> None:
    """Set a bold figure headline with a quieter subtitle beneath it.

    Offsets are computed in inches and converted to figure fractions, so a
    short multi-panel figure gets the same visual spacing as a tall one.
    Using fixed figure fractions here collapses the two lines onto each
    other whenever the figure is short.

    Args:
        figure: The figure to title.
        headline: The bold first line.
        subtitle: A smaller, secondary-ink second line.
    """
    height = float(figure.get_size_inches()[1]) or 1.0
    if not subtitle:
        figure.suptitle(headline, y=1.0 + 0.16 / height)
        return
    figure.suptitle(headline, y=1.0 + 0.50 / height)
    figure.text(
        0.5, 1.0 + 0.14 / height, subtitle, ha="center", va="bottom",
        fontsize=8.6, color=TEXT_SECONDARY,
    )


def _save(figure: Figure, path: Path) -> Path:
    """Write a figure and close it.

    Args:
        figure: The figure to write.
        path: Destination PNG path; parent directories are created.

    Returns:
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_split_strategies(
    scores: Mapping[str, Mapping[str, float]],
    path: Path,
    metric_label: str = "ROC AUC",
    chance: float | None = 0.5,
    subtitle: str = "",
) -> Path:
    """The headline figure: one metric per target, grouped by split strategy.

    Args:
        scores: ``{target_label: {strategy: score}}``.
        path: Destination PNG path.
        metric_label: Y-axis label.
        chance: Value of the no-skill baseline, or ``None`` to omit it.
        subtitle: Optional line under the title.

    Returns:
        The path written.

    Raises:
        ValueError: If ``scores`` is empty.
    """
    if not scores:
        raise ValueError("nothing to plot")
    _apply_style()
    targets = list(scores)
    strategies = [s for s in SPLIT_COLOURS if any(s in v for v in scores.values())]

    figure, axes = plt.subplots(figsize=(1.3 * len(targets) + 3.0, 4.5))
    # Cap the group width so two or three targets do not produce slabs.
    group_width = min(0.78, 0.26 * len(strategies))
    width = group_width / max(len(strategies), 1)
    positions = np.arange(len(targets), dtype=float)

    for index, strategy in enumerate(strategies):
        offset = (index - (len(strategies) - 1) / 2) * width
        values = [float(scores[t].get(strategy, np.nan)) for t in targets]
        bars = axes.bar(
            positions + offset,
            values,
            width=width * 0.92,
            color=SPLIT_COLOURS[strategy],
            label=strategy,
            edgecolor=SURFACE,
            linewidth=1.4,
        )
        axes.bar_label(
            bars,
            fmt="%.2f",
            padding=2,
            fontsize=7.4,
            color=TEXT_SECONDARY,
        )

    if chance is not None:
        axes.axhline(
            chance,
            color=NULL_COLOUR,
            linewidth=1.3,
            linestyle=(0, (4, 3)),
            zorder=1,
        )
        _chance_label(axes, chance, len(targets))

    axes.set_xticks(positions)
    axes.set_xticklabels(targets, fontsize=9)
    axes.set_ylabel(metric_label)
    _header(
        axes,
        "Reported performance depends on how the data is split",
        subtitle,
        ncols=len(strategies),
        columnspacing=1.6,
        handlelength=1.4,
    )
    _tidy(axes)
    bottom = 0.0 if chance is None else min(0.0, chance - 0.1)
    peak = max(max(v.values()) for v in scores.values())
    axes.set_ylim(bottom, max(1.0, peak) * 1.06)
    return _save(figure, path)


def plot_split_gap(
    rows: Sequence[Mapping[str, Any]],
    path: Path,
    metric_label: str = "ROC AUC",
) -> Path:
    """Paired dot plot of the random-split score and its honest counterparts.

    Each row is one target/model pair; a line runs from the random-split
    score to the scaffold and time scores, so the length of the line *is*
    the overstatement.

    Args:
        rows: Dictionaries with ``label``, ``random``, ``scaffold`` and
            optionally ``time``.
        path: Destination PNG path.
        metric_label: X-axis label.

    Returns:
        The path written.

    Raises:
        ValueError: If ``rows`` is empty.
    """
    if not rows:
        raise ValueError("nothing to plot")
    _apply_style()

    def _gap(row: Mapping[str, Any]) -> float:
        """Sort key: how far the random-split score sits above the rest."""
        others = [float(row[s]) for s in ("scaffold", "time") if s in row]
        return float(row["random"]) - min(others) if others else 0.0

    ordered = sorted(rows, key=_gap)
    figure, axes = plt.subplots(figsize=(7.6, 0.33 * len(ordered) + 2.1))
    positions = np.arange(len(ordered), dtype=float)

    for position, row in zip(positions, ordered, strict=True):
        present = [s for s in ("random", "scaffold", "time") if s in row]
        values = [float(row[s]) for s in present]
        axes.plot(
            [min(values), max(values)],
            [position, position],
            color=GRID,
            linewidth=2.4,
            zorder=1,
            solid_capstyle="round",
        )
        for strategy in present:
            axes.scatter(
                float(row[strategy]),
                position,
                s=62,
                color=SPLIT_COLOURS[strategy],
                edgecolor=SURFACE,
                linewidth=1.5,
                zorder=3,
                label=strategy,
            )

    handles, labels = axes.get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=True))
    axes.set_yticks(positions)
    axes.set_yticklabels([str(r["label"]) for r in ordered], fontsize=8)
    axes.set_xlabel(metric_label)
    _title(
        axes,
        "How much a random split overstates performance",
        "each line spans one target and model; length is the overstatement",
        pad=10.0,
        subtitle_y=None,
    )
    # The plot is tall and narrow, so the legend goes underneath the axis
    # label where it cannot land on a data row however many rows there are.
    axes.legend(
        unique.values(),
        unique.keys(),
        ncols=len(unique),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.09 - 1.1 / max(len(ordered), 4)),
        columnspacing=1.8,
        handlelength=1.2,
    )
    _tidy(axes, xgrid=True)
    axes.grid(axis="y", visible=False)
    return _save(figure, path)


def plot_representation_comparison(
    scores: Mapping[str, Mapping[str, float]],
    path: Path,
    metric_label: str = "ROC AUC (scaffold split)",
) -> Path:
    """Grouped bars of representation performance, one group per target.

    Args:
        scores: ``{target_label: {representation: score}}``.
        path: Destination PNG path.
        metric_label: Y-axis label.

    Returns:
        The path written.

    Raises:
        ValueError: If ``scores`` is empty.
    """
    if not scores:
        raise ValueError("nothing to plot")
    _apply_style()
    targets = list(scores)
    representations = list(dict.fromkeys(k for v in scores.values() for k in v))

    figure, axes = plt.subplots(figsize=(1.5 * len(targets) + 3.2, 4.6))
    width = 0.8 / max(len(representations), 1)
    positions = np.arange(len(targets), dtype=float)

    for index, representation in enumerate(representations):
        offset = (index - (len(representations) - 1) / 2) * width
        values = [float(scores[t].get(representation, np.nan)) for t in targets]
        axes.bar(
            positions + offset,
            values,
            width=width * 0.9,
            color=CATEGORICAL[index % len(CATEGORICAL)],
            label=representation,
            edgecolor=SURFACE,
            linewidth=1.2,
        )

    axes.axhline(0.5, color=NULL_COLOUR, linewidth=1.3, linestyle=(0, (4, 3)), zorder=1)
    _chance_label(axes, 0.5, len(targets))
    axes.set_xticks(positions)
    axes.set_xticklabels(targets, fontsize=9)
    axes.set_ylabel(metric_label)
    _header(
        axes,
        "Representations differ far less than evaluation protocols do",
        "the spread within a target is small next to the split-strategy gap",
        ncols=len(representations),
        columnspacing=1.4,
        handlelength=1.4,
    )
    axes.set_ylim(0.0, 1.0)
    _tidy(axes)
    return _save(figure, path)


def plot_error_vs_similarity(
    series: Mapping[str, Sequence[tuple[float, float, int]]],
    path: Path,
    y_label: str = "mean absolute error (log units)",
) -> Path:
    """Error against nearest-neighbour similarity to the training set.

    Args:
        series: ``{target_label: [(mean_similarity, error, n), ...]}``.
        path: Destination PNG path.
        y_label: Y-axis label.

    Returns:
        The path written.

    Raises:
        ValueError: If ``series`` is empty.
    """
    if not series:
        raise ValueError("nothing to plot")
    _apply_style()
    figure, axes = plt.subplots(figsize=(7.4, 4.4))
    for index, (label, points) in enumerate(series.items()):
        if not points:
            continue
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        colour = CATEGORICAL[index % len(CATEGORICAL)]
        axes.plot(xs, ys, color=colour, marker="o", label=label, zorder=3)
        axes.scatter(
            xs, ys, s=46, color=colour, edgecolor=SURFACE, linewidth=1.4, zorder=4
        )
    axes.set_xlabel("Tanimoto similarity to the nearest training compound")
    axes.set_ylabel(y_label)
    _header(
        axes,
        "Predictions get worse the further a compound sits from the training set",
        "scaffold split, regression on pChEMBL",
        ncols=min(len(series), 5),
        columnspacing=1.6,
        handlelength=1.6,
    )
    _tidy(axes, xgrid=True)
    return _save(figure, path)


def plot_coverage_tradeoff(
    series: Mapping[str, Sequence[tuple[float, float]]],
    path: Path,
    y_label: str = "RMSE on retained compounds (log units)",
) -> Path:
    """The coverage/accuracy trade-off from applying a similarity cut.

    Args:
        series: ``{target_label: [(coverage, metric), ...]}``.
        path: Destination PNG path.
        y_label: Y-axis label.

    Returns:
        The path written.

    Raises:
        ValueError: If ``series`` is empty.
    """
    if not series:
        raise ValueError("nothing to plot")
    _apply_style()
    figure, axes = plt.subplots(figsize=(7.4, 4.4))
    for index, (label, points) in enumerate(series.items()):
        if not points:
            continue
        ordered = sorted(points)
        axes.plot(
            [p[0] for p in ordered],
            [p[1] for p in ordered],
            color=CATEGORICAL[index % len(CATEGORICAL)],
            label=label,
            zorder=3,
        )
    axes.set_xlabel("coverage (fraction of the test set still predicted)")
    axes.set_ylabel(y_label)
    _header(
        axes,
        "Refusing to predict outside the applicability domain buys accuracy",
        "and costs coverage; the operating point is a project decision",
        ncols=min(len(series), 5),
        columnspacing=1.6,
        handlelength=1.6,
    )
    _tidy(axes, xgrid=True)
    return _save(figure, path)


def plot_enrichment_curves(
    series: Mapping[str, Sequence[tuple[float, float]]],
    path: Path,
    marks: Sequence[float] = (0.01, 0.05),
) -> Path:
    """Cumulative recall against the fraction of the ranked list screened.

    Args:
        series: ``{target_label: [(fraction_screened, recall), ...]}``.
        path: Destination PNG path.
        marks: Fractions to mark with vertical rules.

    Returns:
        The path written.

    Raises:
        ValueError: If ``series`` is empty.
    """
    if not series:
        raise ValueError("nothing to plot")
    _apply_style()
    figure, axes = plt.subplots(figsize=(7.4, 4.6))
    left = 0.002
    # Random picking recovers actives in proportion to the fraction screened.
    # On a log x-axis that is a curve, not the usual diagonal, so it has to be
    # sampled rather than drawn as a straight segment.
    baseline = np.logspace(np.log10(left), 0.0, 200)
    axes.plot(
        baseline,
        baseline,
        color=NULL_COLOUR,
        linewidth=1.3,
        linestyle=(0, (4, 3)),
        zorder=1,
    )
    axes.text(
        0.62, 0.36, "random picking", color=TEXT_SECONDARY, fontsize=7.8,
        rotation=50, rotation_mode="anchor", ha="center", va="top",
    )
    for index, (label, points) in enumerate(series.items()):
        ordered = sorted(points)
        axes.plot(
            [p[0] for p in ordered],
            [p[1] for p in ordered],
            color=CATEGORICAL[index % len(CATEGORICAL)],
            label=label,
            zorder=3,
        )
    for mark in marks:
        axes.axvline(mark, color=TEXT_SECONDARY, linewidth=0.9, alpha=0.45, zorder=2)
        axes.text(
            mark,
            0.985,
            f" {mark:.0%}",
            color=TEXT_SECONDARY,
            fontsize=7.6,
            ha="left",
            va="top",
        )
    axes.set_xscale("log")
    axes.set_xlim(left, 1.0)
    axes.set_ylim(0.0, 1.0)
    axes.set_xlabel("fraction of the ranked list screened (log scale)")
    axes.set_ylabel("fraction of true actives recovered")
    _header(
        axes,
        "A screen only ever buys the top of the list",
        "scaffold split, random forest classifier",
        ncols=min(len(series), 5),
        columnspacing=1.6,
        handlelength=1.6,
    )
    _tidy(axes, xgrid=True)
    return _save(figure, path)


def plot_calibration(
    series: Mapping[str, tuple[Sequence[float], Sequence[float]]],
    path: Path,
) -> Path:
    """Reliability curves: predicted probability against observed frequency.

    Args:
        series: ``{model_label: (mean_predicted, observed_fraction)}``.
        path: Destination PNG path.

    Returns:
        The path written.

    Raises:
        ValueError: If ``series`` is empty.
    """
    if not series:
        raise ValueError("nothing to plot")
    _apply_style()
    figure, axes = plt.subplots(figsize=(5.6, 5.2))
    axes.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        color=NULL_COLOUR,
        linewidth=1.3,
        linestyle=(0, (4, 3)),
        zorder=1,
    )
    axes.text(
        # Low on the diagonal the curves are bunched and run straight through
        # the words; at the top the diagonal is clear of every series.
        0.66, 0.70, "perfectly calibrated", color=TEXT_SECONDARY,
        fontsize=7.8, rotation=45, rotation_mode="anchor",
    )
    for index, (label, (predicted, observed)) in enumerate(series.items()):
        colour = CATEGORICAL[index % len(CATEGORICAL)]
        axes.plot(predicted, observed, color=colour, marker="o", label=label, zorder=3)
        axes.scatter(
            predicted, observed, s=42, color=colour,
            edgecolor=SURFACE, linewidth=1.4, zorder=4,
        )
    axes.set_xlabel("mean predicted probability of activity")
    axes.set_ylabel("observed fraction active")
    _title(
        axes,
        "Are the probabilities worth their face value?",
        "scaffold split, pooled over targets",
        pad=18.0,
        subtitle_y=1.015,
    )
    axes.set_xlim(0.0, 1.0)
    axes.set_ylim(0.0, 1.0)
    axes.set_aspect("equal")
    axes.legend(title="model", title_fontsize=8.5, loc="lower right")
    _tidy(axes, xgrid=True)
    return _save(figure, path)


def plot_y_scrambling(
    panels: Sequence[tuple[str, Sequence[float], float]],
    path: Path,
    metric_label: str = "ROC AUC",
) -> Path:
    """Null distributions from label permutation, against the real score.

    Args:
        panels: ``(target_label, null_scores, observed_score)`` per target.
        path: Destination PNG path.
        metric_label: X-axis label.

    Returns:
        The path written.

    Raises:
        ValueError: If ``panels`` is empty.
    """
    if not panels:
        raise ValueError("nothing to plot")
    _apply_style()
    figure, axes_list = plt.subplots(
        1, len(panels), figsize=(2.5 * len(panels) + 0.9, 3.5), sharey=True
    )
    if len(panels) == 1:
        axes_list = [axes_list]

    for index, (axes, (label, null, observed)) in enumerate(
        zip(axes_list, panels, strict=True)
    ):
        values = np.asarray(list(null), dtype=float)
        if values.size:
            axes.hist(
                values,
                bins=min(12, max(4, values.size // 2)),
                color=NULL_COLOUR,
                edgecolor=SURFACE,
                linewidth=1.0,
                label="scrambled labels",
            )
        axes.axvline(
            observed,
            color=SPLIT_COLOURS["scaffold"],
            linewidth=2.4,
            label="real labels",
        )
        axes.set_title(label, fontsize=9.5)
        axes.set_xlabel(metric_label)
        axes.set_xlim(0.3, 1.0)
        _tidy(axes, xgrid=True)
        axes.grid(axis="y", visible=False)
        axes.yaxis.set_major_locator(MaxNLocator(integer=True))
        if index == 0:
            axes.set_ylabel("permutations")
            axes.legend(loc="upper center", fontsize=8)

    _suptitle(
        figure,
        "The signal is real: permuting the labels destroys it",
        "scaffold split, random forest classifier",
    )
    return _save(figure, path)


def plot_chemical_space(
    coordinates: np.ndarray,
    split_labels: Mapping[str, Sequence[str]],
    path: Path,
    explained: tuple[float, float] | None = None,
    target_label: str = "",
) -> Path:
    """Chemical space under each split strategy, from one shared embedding.

    The coordinates are identical in every panel; only the colouring
    changes. That makes the geometry of the three strategies directly
    comparable: a random split speckles the test set through the training
    set, while scaffold and time splits carve out whole regions.

    Args:
        coordinates: ``(n, 2)`` array of 2D embedding coordinates.
        split_labels: ``{strategy: ["train" | "test" | "unused", ...]}``,
            one label per compound per strategy.
        path: Destination PNG path.
        explained: Variance explained by each component, as fractions.
        target_label: Target name for the title.

    Returns:
        The path written.

    Raises:
        ValueError: If the inputs are mismatched or not 2D.
    """
    points = np.asarray(coordinates, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"coordinates must be (n, 2), got {points.shape}")
    if not split_labels:
        raise ValueError("nothing to plot")
    for strategy, labels in split_labels.items():
        if len(labels) != points.shape[0]:
            raise ValueError(
                f"{strategy}: {len(labels)} labels for {points.shape[0]} points"
            )

    _apply_style()
    strategies = list(split_labels)
    figure, axes_list = plt.subplots(
        1, len(strategies), figsize=(3.5 * len(strategies) + 0.6, 4.1),
        sharex=True, sharey=True,
    )
    if len(strategies) == 1:
        axes_list = [axes_list]

    for index, (axes, strategy) in enumerate(zip(axes_list, strategies, strict=True)):
        labels = np.asarray(list(split_labels[strategy]), dtype=object)
        for side, colour, alpha, size in (
            ("train", GRID, 0.9, 6),
            ("test", SPLIT_COLOURS.get(strategy, CATEGORICAL[0]), 0.7, 8),
        ):
            mask = labels == side
            if not mask.any():
                continue
            axes.scatter(
                points[mask, 0], points[mask, 1], s=size, alpha=alpha,
                color=colour, label=f"{side} (n={int(mask.sum())})", linewidth=0,
            )
        axes.set_title(f"{strategy} split", fontsize=9.5)
        axes.set_xlabel(f"PC1 ({explained[0]:.0%})" if explained else "PC1")
        if index == 0:
            axes.set_ylabel(f"PC2 ({explained[1]:.0%})" if explained else "PC2")
        axes.legend(loc="upper right", markerscale=2.4, fontsize=7.6)
        _tidy(axes, xgrid=True)

    _suptitle(
        figure,
        f"Chemical space of {target_label}, coloured by split",
        "same PCA of ECFP4 bits in every panel; only the partition changes",
    )
    return _save(figure, path)


def plot_leakage(
    fractions: Mapping[str, Mapping[str, float]], path: Path
) -> Path:
    """Fraction of test compounds whose scaffold was already in training.

    This is the mechanism behind the headline result, drawn directly.

    Args:
        fractions: ``{target_label: {strategy: fraction}}``.
        path: Destination PNG path.

    Returns:
        The path written.

    Raises:
        ValueError: If ``fractions`` is empty.
    """
    if not fractions:
        raise ValueError("nothing to plot")
    _apply_style()
    targets = list(fractions)
    strategies = [s for s in SPLIT_COLOURS if any(s in v for v in fractions.values())]
    figure, axes = plt.subplots(figsize=(1.5 * len(targets) + 3.0, 4.2))
    group_width = min(0.78, 0.26 * len(strategies))
    width = group_width / max(len(strategies), 1)
    positions = np.arange(len(targets), dtype=float)

    for index, strategy in enumerate(strategies):
        offset = (index - (len(strategies) - 1) / 2) * width
        values = [float(fractions[t].get(strategy, np.nan)) for t in targets]
        bars = axes.bar(
            positions + offset,
            values,
            width=width * 0.92,
            color=SPLIT_COLOURS[strategy],
            label=strategy,
            edgecolor=SURFACE,
            linewidth=1.4,
        )
        axes.bar_label(
            bars, fmt="%.2f", padding=2, fontsize=7.4, color=TEXT_SECONDARY
        )

    axes.set_xticks(positions)
    axes.set_xticklabels(targets, fontsize=9)
    axes.set_ylabel("fraction of test compounds\nwhose scaffold was in training")
    _header(
        axes,
        "Why random splits flatter a model",
        "a scaffold split leaks nothing by construction; "
        "a random split leaks most of the test set",
        ncols=len(strategies),
        columnspacing=1.6,
        handlelength=1.4,
    )
    axes.set_ylim(0.0, 1.08)
    _tidy(axes)
    return _save(figure, path)
