"""
plotting.py

Matplotlib plotting utilities for the benchmark forecasting project.

Goals:
- Keep styling centralized (Theme + matplotlib rcParams helper).
- Keep plotting functions side-effect free (they only draw on provided axes).
- Match the look-and-feel of `3_Plot_forecasts.ipynb` (colors, grid, legend, line styles).
"""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import itertools

import arviz as az
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DashStyle = str | tuple[float, tuple[float, ...]]


@dataclass(frozen=True)
class Theme:
    """Visual theme constants."""

    base_color: str = "#0e294c"
    accent_color: str = "#d4af37"
    gray_color: str = "#6c757d"
    grid_color: str = "#5F86A5"

    # Elegant professional palette (copied from `3_Plot_forecasts.ipynb`).
    palette: tuple[str, ...] = (
        '#1f4788',
        '#4a7c59',
        '#457b9d',
        '#8b7f7b',
        '#264653',
        '#6a4c93',
        '#e76f51',
        '#06aed5',
        '#f4a261',
        '#2a9d8f',
    )

    def with_overrides(self, **kwargs: Any) -> "Theme":
        """Return a copy of the theme with the given fields replaced."""
        return replace(self, **kwargs)


def make_mpl_style(*, scale: float = 1.0) -> dict[str, Any]:
    """Return a minimal matplotlib rcParams dict.

    `scale` is intentionally kept (notebook API), but visual sizing is handled
    explicitly in plotting functions to match `3_Plot_forecasts.ipynb`.
    """
    _ = float(scale)  # reserved for future use
    return {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.facecolor": "none",
        "figure.facecolor": "none",
        "grid.alpha": 0.2,
    }


def apply_style(style: Mapping[str, Any], *, reset: bool = False) -> None:
    """Apply a matplotlib rcParams style dict."""
    if reset:
        plt.rcdefaults()
    plt.rcParams.update(dict(style))


def plot_calibration_curve(
    idata: az.InferenceData,
    *,
    ax: plt.Axes | None = None,
    n_points: int = 20,
) -> plt.Axes:
    """Plot a posterior predictive calibration curve.

    Expects `idata.predictions` to contain:
    - y: posterior predictive samples, shape (obs, chain, draw)
    - y_true: held-out observations, shape (obs,)
    """
    y_pred = idata.predictions.stack(sample=("chain", "draw"))["y"].to_numpy()
    y_true = idata.predictions["y_true"].to_numpy()

    confidence_levels = np.linspace(0.01, 0.99, n_points)
    observed_coverage: list[float] = []
    for p in confidence_levels:
        q_lo = (1 - p) / 2
        q_hi = 1 - q_lo
        lo = np.quantile(y_pred, q_lo, axis=1)
        hi = np.quantile(y_pred, q_hi, axis=1)
        observed_coverage.append(float(np.mean((y_true >= lo) & (y_true <= hi))))

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
        zorder=5,
    )


def _plot_forecast_with_split_style(
    ax: plt.Axes,
    pred: pd.DataFrame,
    *,
    color: str,
    last_observed_date: pd.Timestamp,
    label: str,
    ci_alpha: float,
    observed_alpha: float,
    forecast_alpha: float,
    linewidth: float,
    dash_style: DashStyle,
) -> None:
    """Plot mean + CI for a benchmark with solid-to-dashed split at last observation."""
    if pred.empty:
        return

    pred = pred.sort_values("release_date")
    x = pred["release_date"]

    ax.fill_between(
        x,
        pred["mu_lower"],
        pred["mu_upper"],
        color=color,
        alpha=ci_alpha,
        linewidth=0,
        zorder=3,
    )

    past = pred.loc[pred["release_date"] <= last_observed_date]
    future = pred.loc[pred["release_date"] >= last_observed_date]

    if not past.empty:
        ax.plot(
            past["release_date"],
            past["mu_mean"],
            color=color,
            alpha=observed_alpha,
            linestyle="-",
            linewidth=linewidth,
            label=label,
            zorder=6,
        )

    if not future.empty:
        if not past.empty:
            future = pd.concat([past.tail(1), future], ignore_index=True)

        ax.plot(
            future["release_date"],
            future["mu_mean"],
            color=color,
            alpha=forecast_alpha,
            linestyle=dash_style,
            linewidth=linewidth,
            label=None,
            zorder=6,
        )


def _benchmark_plot_order(observed: pd.DataFrame, forecast: pd.DataFrame) -> list[str]:
    """Return benchmark names ordered by posterior mean tau (left-to-right in the plot)."""
    if "benchmark" not in observed.columns:
        return []

    # Preferred: mean_tau as a datetime column (added by `generate_forecast()` patch below).
    if "mean_tau" in forecast.columns:
        tau = forecast.groupby("benchmark", dropna=False)["mean_tau"].first()
        if not np.issubdtype(tau.dtype, np.datetime64):
            starts = observed.groupby("benchmark")["release_date"].min()
            tau = starts + pd.to_timedelta(tau.astype(float), unit="D")
        return tau.sort_values().index.astype(str).tolist()

    # Fallback: stable in-input ordering.
    return observed["benchmark"].dropna().astype(str).drop_duplicates().tolist()


def plot_category_forecast(
    *,
    observed: pd.DataFrame,
    forecast: pd.DataFrame,
    end_date: pd.Timestamp,
    category_label: str,
    theme: Theme,
    scale: float = 1.0,
    figsize: tuple[float, float] = (7, 4),
    ci_alpha: float = 0.2,
    observed_line_alpha: float = 0.8,
    forecast_line_alpha: float = 0.5,
    scatter_alpha: float = 0.4,
    line_width: float = 1.5,
    dash_style: DashStyle = (5, (4, 2)),
) -> tuple[plt.Figure, plt.Axes]:
    """Plot one category: multiple benchmarks on one axis.

    Matches `3_Plot_forecasts.ipynb` styling and enforces benchmark ordering by mean_tau.
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor="none")

    label_fontsize = int(13 * scale)
    tick_fontsize = int(8 * scale)
    scatter_size = 40 * scale

    benchmarks = _benchmark_plot_order(observed, forecast)
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
            observed_alpha=observed_line_alpha,
            forecast_alpha=forecast_line_alpha,
            linewidth=line_width,
            dash_style=dash_style,
        )

    ax.set_xlim(right=end_date)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlabel("", fontsize=label_fontsize, fontweight="500", color=theme.base_color)

    ax.set_ylim(-0.02, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y*100:.0f}%"))
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
        fontsize=10,
        frameon=False,
        fancybox=False,
        ncol=1,
        handlelength=1.5,
    )
    for text in legend.get_texts():
        text.set_color(theme.base_color)
    for line in legend.get_lines():
        line.set_linewidth(2.0)
        line.set_linestyle("-")
        line.set_alpha(1.0)

    fig.tight_layout()
    return fig, ax
