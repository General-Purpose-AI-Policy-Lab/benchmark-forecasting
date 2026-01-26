"""
plotting.py

Plotting utilities with notebook-controlled style.

Design:
- Theme + rcParams style are created in Python, but can be overridden in the notebook.
- Plotting functions are pure (no model sampling).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional

import itertools

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import arviz as az


@dataclass(frozen=True)
class Theme:
    base_color: str = "#0e294c"
    accent_color: str = "#d4af37"
    gray_color: str = "#6c757d"
    grid_color: str = "#5F86A5"
    palette: tuple[str, ...] = (
        "#1a759f",
        "#f94144",
        "#577590",
        "#43aa8b",
        "#f3722c",
        "#90be6d",
        "#277da1",
        "#f9c74f",
        "#8b7f7b",
        "#264653",
        "#6a4c93",
        "#e76f51",
        "#06aed5",
        "#f4a261",
        "#2a9d8f",
    )

    def with_overrides(self, **kwargs: Any) -> "Theme":
        return replace(self, **kwargs)


def make_mpl_style(*, scale: float = 1.0) -> dict[str, Any]:
    """Return a matplotlib rcParams dict; scale multiplies common sizes."""
    s = float(scale)
    return {
        "axes.facecolor": "none",
        "figure.facecolor": "none",
        "grid.alpha": 0.2,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10 * s,
        "axes.labelsize": 11 * s,
        "axes.titlesize": 12 * s,
        "xtick.labelsize": 9 * s,
        "ytick.labelsize": 9 * s,
        "legend.fontsize": 9 * s,
        "lines.linewidth": 2.0 * s,
    }


def apply_style(style: Mapping[str, Any], *, reset: bool = False) -> None:
    """Apply matplotlib rcParams style dict."""
    if reset:
        plt.rcdefaults()
    plt.rcParams.update(dict(style))


def plot_calibration_curve(
    idata: az.InferenceData,
    *,
    ax: Optional[plt.Axes] = None,
    n_points: int = 20,
) -> plt.Axes:
    """Plot calibration curve for posterior predictive samples."""
    y_pred = idata.predictions.stack(sample=("chain", "draw"))["y"].to_numpy()
    y_true = idata.predictions["y_true"].to_numpy()

    confidence_levels = np.linspace(0.01, 0.99, n_points)
    observed_coverage = []
    for p in confidence_levels:
        q_lo = (1 - p) / 2
        q_hi = 1 - q_lo
        lo = np.quantile(y_pred, q_lo, axis=1)
        hi = np.quantile(y_pred, q_hi, axis=1)
        observed_coverage.append(np.mean((y_true >= lo) & (y_true <= hi)))

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    ax.plot(confidence_levels, observed_coverage, "o-", label="Model calibration")
    ax.plot([0, 1], [0, 1], "k--", label="Ideal")
    ax.set_xlabel("Expected confidence level (interval width)")
    ax.set_ylabel("Observed coverage")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    return ax


def _plot_datapoints(
    ax: plt.Axes,
    obs: pd.DataFrame,
    *,
    color: str,
    size: float,
    alpha: float,
) -> None:
    ax.scatter(
        obs["release_date"],
        obs["score"],
        s=size,
        color=color,
        marker="o",
        edgecolors="white",
        alpha=alpha,
        zorder=6,
    )


def _plot_forecast_with_split_style(
    ax: plt.Axes,
    pred: pd.DataFrame,
    *,
    color: str,
    last_observed_date: pd.Timestamp,
    label: str,
    ci_alpha: float,
    mean_alpha: float,
    dash_style: Any,
) -> None:
    if pred.empty:
        return

    pred = pred.sort_values("release_date")
    x = pred["release_date"]

    # Uncertainty band across full horizon
    ax.fill_between(
        x,
        pred["mu_lower"],
        pred["mu_upper"],
        color=color,
        alpha=ci_alpha,
        linewidth=0,
        zorder=3,
    )

    # Mean split: solid until last observed, dashed after
    past = pred.loc[pred["release_date"] <= last_observed_date]
    future = pred.loc[pred["release_date"] > last_observed_date]

    if not past.empty:
        ax.plot(
            past["release_date"],
            past["mu_mean"],
            color=color,
            alpha=mean_alpha,
            linestyle="-",
            label=label,
            zorder=5,
        )

    if not future.empty:
        if not past.empty:
            future = pd.concat([past.tail(1), future], ignore_index=True)

        ax.plot(
            future["release_date"],
            future["mu_mean"],
            color=color,
            alpha=mean_alpha,
            linestyle=dash_style,
            label=None,
            zorder=5,
        )


def plot_category_forecast(
    *,
    observed: pd.DataFrame,
    forecast: pd.DataFrame,
    end_date: pd.Timestamp,
    category_label: str,
    theme: Theme,
    scale: float = 1.0,
    figsize: tuple[float, float] = (7, 4),
    ci_alpha: float = 0.18,
    mean_alpha: float = 0.75,
    scatter_alpha: float = 0.45,
    dash_style: Any = (0, (4, 2)),
) -> tuple[plt.Figure, plt.Axes]:
    """One figure for one category, multiple benchmarks.

    Legend entries are benchmark names.
    Forecast line is solid up to the last observed datapoint for that benchmark, then dashed.
    """
    fig, ax = plt.subplots(figsize=figsize)

    label_fontsize = int(13 * scale)
    tick_fontsize = int(8 * scale)
    scatter_size = 40 * scale

    benchmarks = observed["benchmark"].dropna().unique()
    color_cycle = itertools.cycle(theme.palette)

    for bench, color in zip(benchmarks, color_cycle):
        obs_b = observed.loc[observed["benchmark"] == bench]
        pred_b = forecast.loc[forecast["benchmark"] == bench]

        if obs_b.empty:
            continue

        last_date = pd.to_datetime(obs_b["release_date"].max())

        _plot_datapoints(ax, obs_b, color=color, size=scatter_size, alpha=scatter_alpha)
        _plot_forecast_with_split_style(
            ax,
            pred_b,
            color=color,
            last_observed_date=last_date,
            label=str(bench),
            ci_alpha=ci_alpha,
            mean_alpha=mean_alpha,
            dash_style=dash_style,
        )

    ax.set_xlim(right=end_date)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlabel("", fontsize=label_fontsize, fontweight="500", color=theme.base_color)

    ax.set_ylim(-0.02, 1.05)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: "{:.0%}".format(y)))
    ax.set_ylabel("Performance", fontsize=label_fontsize, fontweight="500", color=theme.base_color)

    ax.grid(True, alpha=0.1, linewidth=0.8, color=theme.grid_color)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(theme.base_color)
    ax.tick_params(axis="y", left=False, right=False)
    ax.tick_params(labelsize=tick_fontsize, colors=theme.base_color)

    ax.set_title(
        str(category_label),
        fontsize=int(13 * scale),
        fontweight="600",
        color=theme.base_color,
        pad=10,
    )

    legend = ax.legend(
        loc="lower right",
        fontsize=int(10 * scale),
        framealpha=0.95,
        facecolor="white",
        edgecolor=theme.base_color,
        fancybox=True,
        ncol=1,
        handlelength=2.0,
    )
    for text in legend.get_texts():
        text.set_color(theme.base_color)

    fig.tight_layout()
    return fig, ax
