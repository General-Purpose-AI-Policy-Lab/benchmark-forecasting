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
from typing import Any, Literal

import itertools

import arviz as az
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DashStyle = str | tuple[float, tuple[float, ...]]

Language = Literal["en", "fr"]

# Centralized display-name overrides (keep raw names as fallback).
BENCHMARK_NAME_OVERRIDES: dict[str, str] = {}

CATEGORY_NAME_OVERRIDES: dict[Language, dict[str, str]] = {
    "en": {},
    "fr": {
        "Domain Specific Questions": "Questions spécialisées",
        "Core AGI Progress": "Progrès AGI (cœur)",
        "General Reasoning": "Raisonnement général",
        "Autonomous SWE": "Ingénierie logicielle autonome",
        "Multimodal Understanding": "Compréhension multimodale",
        "Biology": "Biologie",
        "Agentic Computer Use": "Utilisation agentique de l’ordinateur",
        "Tier 2 Excluded": "Niveau 2 exclu",
        "Advanced Language and Writing": "Langage avancé et écriture",
        "High End Math Reasoning": "Raisonnement mathématique avancé",
        "Chemistry": "Chimie",
        "Commonsense QA": "QA de bon sens",
    },
}


def resolve_benchmark_name(raw: str) -> str:
    """Return the displayed benchmark name (central override, raw fallback)."""
    return BENCHMARK_NAME_OVERRIDES.get(raw, raw)


def resolve_category_name(raw: str, *, language: Language) -> str:
    """Return the displayed category name for `language` (override, raw fallback)."""
    return CATEGORY_NAME_OVERRIDES.get(language, {}).get(raw, raw)


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
        "axes.edgecolor": "none",
        "axes.labelcolor": "#0e294c",
        "xtick.color": "#0e294c",
        "ytick.color": "#0e294c",
        "text.color": "#0e294c",
        "axes.grid": True,
        "grid.alpha": 0.1,
        "grid.linestyle": "-",
        "grid.linewidth": 0.8,
        "legend.frameon": False,
    }


def apply_style(style: Mapping[str, Any], *, reset: bool = False) -> None:
    """Apply rcParams style dict."""
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

    ax.plot(confidence_levels, observed_coverage, color="#1f4788", linewidth=2)
    ax.plot([0, 1], [0, 1], color="#6c757d", linestyle="--", linewidth=1)
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Observed coverage")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(["Calibration", "Ideal"], loc="lower right")
    return ax


def _plot_datapoints(
    ax: plt.Axes,
    data: pd.DataFrame,
    *,
    color: str,
    size: float,
    alpha: float,
) -> None:
    """Scatter plot of observed datapoints."""
    ax.scatter(
        data["release_date"],
        data["score"],
        color=color,
        s=size,
        alpha=alpha,
        edgecolors="none",
        zorder=7,
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
    """Plot forecast mean and CI with solid (past) then dashed (future) styling."""
    pred = pred.sort_values("release_date").copy()
    past = pred.loc[pred["release_date"] <= last_observed_date]
    future = pred.loc[pred["release_date"] > last_observed_date]

    ax.fill_between(
        pred["release_date"],
        pred["mu_lower"],
        pred["mu_upper"],
        color=color,
        alpha=ci_alpha,
        linewidth=0,
        zorder=4,
    )

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
    if "mean_tau_days" in forecast.columns:
        order = (
            forecast[["benchmark", "mean_tau_days"]]
            .dropna()
            .drop_duplicates()
            .sort_values("mean_tau_days")["benchmark"]
            .astype(str)
            .tolist()
        )
        if order:
            return order

    # Fallback: stable alphabetical order.
    return sorted(set(observed["benchmark"].astype(str)))


def plot_category_forecast(
    *,
    observed: pd.DataFrame,
    forecast: pd.DataFrame,
    end_date: pd.Timestamp,
    category_label: str,
    theme: Theme,
    language: Language = "en",
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
            label=resolve_benchmark_name(str(bench)),
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

    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
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
        resolve_category_name(str(category_label), language=language),
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


def plot_asymmetry_visualization(
    idata: az.InferenceData,
    *,
    theme: Theme,
    language: str = "en",
    scale: float = 1.0,
    figsize: tuple[float, float] = (10, 6),
    n_points: int = 500,
) -> tuple[plt.Figure, plt.Axes]:
    """Illustrative figure: Harvey curve shapes vs logistic (centered at 50%).

    Uses posterior medians of alpha per benchmark.
    Plots:
      - all Harvey curves (one per benchmark), semi-transparent, same color
      - bold median-Harvey curve
      - dashed logistic reference curve
    X-axis is normalized time (k=1), with each Harvey curve shifted so y=0.5 at x=0.
    """
    if language not in {"en", "fr"}:
        raise ValueError(f"language must be 'en' or 'fr', got {language!r}")

    if "alpha" not in idata.posterior:
        raise ValueError("plot_asymmetry_visualization requires idata.posterior['alpha'].")

    # --- Colors: match the reference figure using Theme palette (no new hardcoding) ---
    # Expected palette order (from Theme): [navy, green, steel-blue, ...]
    median_color = theme.palette[0] if len(theme.palette) > 0 else theme.base_color
    logistic_color = theme.palette[1] if len(theme.palette) > 1 else theme.gray_color
    family_color = theme.palette[2] if len(theme.palette) > 2 else theme.gray_color

    # --- Helper curves (normalized to [0, 1]) ---
    def _harvey_sigmoid(z: np.ndarray, alpha: float) -> np.ndarray:
        # y = (1 - (1-a)*exp(-z))^(1/(1-a))
        # keep numerically safe for extreme z
        base = 1.0 - (1.0 - alpha) * np.exp(-z)
        base = np.maximum(base, 1e-12)
        return np.power(base, 1.0 / (1.0 - alpha))

    def _logistic_sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

    def _z_half(alpha: float) -> float:
        """Solve for z such that Harvey sigmoid equals 0.5."""
        # 0.5^(1-a) = 1 - (1-a)exp(-z)
        # z = -log((1 - 0.5^(1-a))/(1-a))
        if abs(alpha - 1.0) < 1e-8:
            return 0.0  # limit is logistic
        pow_term = np.exp((1.0 - alpha) * np.log(0.5))
        ratio = (1.0 - pow_term) / (1.0 - alpha)
        ratio = max(float(ratio), 1e-12)
        return float(-np.log(ratio))

    # --- Extract per-benchmark median alpha ---
    alpha_med_per_bench = (
        idata.posterior["alpha"].median(dim=("chain", "draw")).to_numpy()
    )
    n_benchmarks = int(alpha_med_per_bench.shape[-1])
    median_alpha = float(np.median(alpha_med_per_bench))

    # --- Time grid (normalized) ---
    t = np.linspace(-6.0, 6.0, int(n_points))

    # --- Figure ---
    fig, ax = plt.subplots(figsize=figsize, facecolor="none")

    # Plot all Harvey curves (same color, transparent)
    for a in alpha_med_per_bench:
        z0 = _z_half(float(a))
        y = _harvey_sigmoid(t + z0, float(a))  # centered at 50% crossing
        ax.plot(t, y, color=family_color, alpha=0.30, linewidth=1.0, zorder=1)

    # Bold median Harvey curve
    z0_med = _z_half(median_alpha)
    y_med = _harvey_sigmoid(t + z0_med, median_alpha)
    ax.plot(t, y_med, color=median_color, linewidth=2.5, zorder=3)

    # Logistic reference (dashed)
    y_log = _logistic_sigmoid(t)
    ax.plot(t, y_log, color=logistic_color, linewidth=2.5, linestyle="--", zorder=4)

    # --- Localized text ---
    if language == "fr":
        title = "Asymétrie des courbes de progrès : Harvey vs. Logistique"
        xlabel = "Temps (normalisé)"
        ylabel = "Performance"
        harvey_label = f"Courbes de Harvey estimées\nsur {n_benchmarks} benchmarks"
        logistic_label = "Courbe logistique\n(symétrique)"
    else:
        title = "Asymmetry of progress curves: Harvey vs. Logistic"
        xlabel = "Time (normalized)"
        ylabel = "Performance"
        harvey_label = f"Estimated Harvey curves\nacross {n_benchmarks} benchmarks"
        logistic_label = "Logistic curve\n(symmetric)"

    # --- Annotations (match reference style) ---
    bbox = dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="none", alpha=0.9)

    ax.annotate(
        harvey_label,
        xy=(-2.0, float(_harvey_sigmoid(np.array([-2.0 + z0_med]), median_alpha)[0])),
        xytext=(-3.5, 0.50),
        fontsize=int(12 * scale),
        color=median_color,
        ha="center",
        va="center",
        arrowprops=dict(arrowstyle="->", color=median_color, lw=1.5),
        bbox=bbox,
    )

    ax.annotate(
        logistic_label,
        xy=(1.0, float(_logistic_sigmoid(np.array([1.0]))[0])),
        xytext=(3.2, 0.65),
        fontsize=int(12 * scale),
        color=logistic_color,
        ha="center",
        va="center",
        arrowprops=dict(arrowstyle="->", color=logistic_color, lw=1.5),
        bbox=bbox,
    )

    # --- Formatting (match reference) ---
    ax.set_xlabel(xlabel, fontsize=int(15 * scale), fontweight="500", color=theme.base_color)
    ax.set_ylabel(ylabel, fontsize=int(15 * scale), fontweight="500", color=theme.base_color)
    ax.set_title(title, fontsize=int(17 * scale), fontweight="600", color=theme.base_color, pad=20)

    ax.grid(True, alpha=0.15, linewidth=0.8, color=theme.grid_color)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(theme.base_color)

    ax.set_xlim(-6.0, 6.0)

    # Point (4): exact y-limits and ticks
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y*100:.0f}%"))

    ax.tick_params(axis="y", left=False, right=False)
    ax.tick_params(labelsize=int(11 * scale), colors=theme.base_color)
    ax.tick_params(axis="x", labelsize=int(11 * scale), colors=theme.base_color)

    plt.tight_layout()
    return fig, ax

def plot_saturation_proportion_posterior(
    idata: az.InferenceData,
    *,
    prepared_frontier: pd.DataFrame,
    target_date: pd.Timestamp | str,
    saturation_fraction: float = 0.95,
    ci_level: float = 0.80,
    theme: Theme,
    language: str = "en",
    scale: float = 1.0,
    figsize: tuple[float, float] = (8, 5),
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes, dict[str, object]]:
    """Posterior histogram: proportion of benchmarks above saturation threshold by target_date.

    Uses only posterior parameters from `idata` and the per-benchmark first observation date
    from `prepared_frontier` (no disk IO).

    Interpretation:
      - Work on the *normalized* curve in [0, 1] (sigmoid output).
      - "Saturated" means sigmoid(t_target) > saturation_fraction, which is equivalent to
        mu(t_target) > l + saturation_fraction * (L - l) (so L/l are not needed here).

    Returns (fig, ax, summary) where summary includes mean/median/std and 80/95% CIs.
    """
    if language not in {"en", "fr"}:
        raise ValueError(f"language must be 'en' or 'fr', got {language!r}")
    if not (0.0 < saturation_fraction < 1.0):
        raise ValueError("saturation_fraction must be in (0, 1)")
    if not (0.0 < ci_level < 1.0):
        raise ValueError("ci_level must be in (0, 1)")
    if "k" not in idata.posterior or "tau" not in idata.posterior:
        raise ValueError("Requires idata.posterior['k'] and idata.posterior['tau'].")

    target_ts = pd.to_datetime(target_date)
    if pd.isna(target_ts):
        raise ValueError(f"Could not parse target_date={target_date!r}")

    # --- Colors (match previous figures, reuse Theme palette) ---
    # Base bars: steel-blue-ish (palette[2]); highlighted bars: navy (palette[0]); median line: gold accent.
    highlight_color = theme.palette[0] if len(theme.palette) > 0 else theme.base_color
    base_bar_color = theme.palette[2] if len(theme.palette) > 2 else theme.gray_color
    median_line_color = theme.accent_color

    # --- Benchmark order / alignment with posterior coords ---
    bench_coord = idata.posterior["k"].coords.get("benchmark", None)
    if bench_coord is None:
        raise ValueError("Posterior 'k' must have a 'benchmark' coordinate.")
    benchmarks = [str(b) for b in bench_coord.to_numpy().tolist()]
    n_benchmarks = len(benchmarks)

    # --- Per-benchmark start date (first observation) from prepared_frontier ---
    pf = prepared_frontier.copy()
    if "benchmark" not in pf.columns or "release_date" not in pf.columns:
        raise ValueError("prepared_frontier must contain 'benchmark' and 'release_date' columns.")
    pf["release_date"] = pd.to_datetime(pf["release_date"], errors="coerce")
    starts = pf.groupby("benchmark")["release_date"].min().reindex(benchmarks)

    if starts.isna().any():
        missing = starts[starts.isna()].index.tolist()
        raise ValueError(
            "prepared_frontier is missing benchmarks present in the posterior. "
            f"First missing examples: {missing[:10]}"
        )

    t_target = (target_ts - starts).dt.days.to_numpy(dtype=float)  # shape (B,)

    # --- Stack posterior samples ---
    posterior = idata.posterior
    k = posterior["k"].stack(sample=("chain", "draw")).transpose("benchmark", "sample").to_numpy()
    tau = posterior["tau"].stack(sample=("chain", "draw")).transpose("benchmark", "sample").to_numpy()

    # z = k*(t - tau) (broadcast t_target over samples)
    z = k * (t_target[:, None] - tau)

    # --- Sigmoid: Harvey if alpha exists, else logistic ---
    if "alpha" in posterior:
        alpha = posterior["alpha"].stack(sample=("chain", "draw")).transpose("benchmark", "sample").to_numpy()

        base = 1.0 - (1.0 - alpha) * np.exp(-z)
        base = np.maximum(base, 1e-12)
        sigmoid = np.power(base, 1.0 / (1.0 - alpha))
    else:
        sigmoid = 1.0 / (1.0 + np.exp(-z))

    above = sigmoid > float(saturation_fraction)  # shape (B, S)
    proportions = above.mean(axis=0).astype(float)  # shape (S,)

    # --- Stats ---
    mean_prop = float(np.mean(proportions))
    median_prop = float(np.median(proportions))
    std_prop = float(np.std(proportions))
    lo_q = 100.0 * (1.0 - float(ci_level)) / 2.0
    hi_q = 100.0 * (1.0 + float(ci_level)) / 2.0
    ci_prop = np.percentile(proportions, [lo_q, hi_q])
    ci80_prop = (float(ci_prop[0]), float(ci_prop[1]))
    ci95 = np.percentile(proportions, [2.5, 97.5])
    ci95_prop = (float(ci95[0]), float(ci95[1]))

    summary: dict[str, object] = {
        "n_benchmarks": n_benchmarks,
        "n_samples": int(proportions.shape[0]),
        "mean": mean_prop,
        "median": median_prop,
        "std": std_prop,
        "ci_level": float(ci_level),
        "ci": ci80_prop,
        "ci95": ci95_prop,
        "target_date": target_ts,
        "saturation_fraction": float(saturation_fraction),
    }

    # --- Plot ---
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, facecolor="none")
    else:
        fig = ax.figure

    # Bins aligned like the reference: n_benchmarks+1 bins over [0,1]
    bins = np.linspace(0.0, 1.0, n_benchmarks + 1)

    counts, bin_edges, patches = ax.hist(
        proportions,
        color=base_bar_color,
        alpha=0.7,
        edgecolor="white",
        linewidth=1.2,
        density=True,
        bins=bins,
    )

    # Highlight bars whose center lies inside the CI interval
    ci_lo, ci_hi = ci80_prop
    for i, patch in enumerate(patches):
        center = 0.5 * (bin_edges[i] + bin_edges[i + 1])
        if ci_lo <= center <= ci_hi:
            patch.set_facecolor(highlight_color)
            patch.set_alpha(0.8)

    # Vertical lines: median + CI bounds
    if language == "fr":
        legend_median = f"Médiane: {median_prop:.1%}"
        legend_ci = f"IC {int(ci_level*100)}%: [{ci_lo:.1%}, {ci_hi:.1%}]"
        title = f"Proportion de benchmarks saturés d'ici {target_ts.year}"
        subtitle = f"Saturé = performance > {saturation_fraction:.0%} de l'asymptote estimée"
        xlabel = f"Proportion de benchmarks > {saturation_fraction:.0%} de L"
        ylabel = "Densité de probabilité"
    else:
        legend_median = f"Median: {median_prop:.1%}"
        legend_ci = f"{int(ci_level*100)}% CI: [{ci_lo:.1%}, {ci_hi:.1%}]"
        title = f"Proportion of benchmarks saturated by {target_ts.year}"
        subtitle = f"Saturated = performance > {saturation_fraction:.0%} of estimated asymptote"
        xlabel = f"Proportion of benchmarks > {saturation_fraction:.0%} of L"
        ylabel = "Probability density"

    ax.axvline(
        median_prop,
        color=median_line_color,
        linewidth=2.5,
        linestyle="-",
        label=legend_median,
        zorder=5,
    )
    ax.axvline(ci_lo, color=highlight_color, linewidth=1.5, linestyle="--", alpha=0.7)
    ax.axvline(
        ci_hi,
        color=highlight_color,
        linewidth=1.5,
        linestyle="--",
        alpha=0.7,
        label=legend_ci,
    )

    # Labels / title / subtitle
    ax.set_xlabel(xlabel, fontsize=int(14 * scale), fontweight="500", color=theme.base_color)
    ax.set_ylabel(ylabel, fontsize=int(14 * scale), fontweight="500", color=theme.base_color)
    ax.set_title(title, fontsize=int(16 * scale), fontweight="600", color=theme.base_color, pad=15)

    ax.text(
        0.5,
        1.0,
        subtitle,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=int(11 * scale),
        color=theme.gray_color,
        style="italic",
    )

    # Grid & spines
    ax.grid(True, alpha=0.15, linewidth=0.8, color=theme.grid_color, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(theme.base_color)

    ax.tick_params(axis="y", left=False, right=False)
    ax.tick_params(labelsize=int(11 * scale), colors=theme.base_color)

    # X as percent, bounds like the reference
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*100:.0f}%"))
    ax.set_xlim(-0.05, 1.05)

    # Legend (force framed legend even if global rc disables it)
    legend = ax.legend(
        loc="upper left",
        fontsize=int(11 * scale),
        frameon=True,
        framealpha=0.95,
        edgecolor=theme.base_color,
        fancybox=True,
    )
    for text in legend.get_texts():
        text.set_color(theme.base_color)

    fig.tight_layout()
    return fig, ax, summary
