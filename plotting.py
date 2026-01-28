"""
plotting.py

Matplotlib plotting utilities for the benchmark forecasting project.

Goals:
- Keep styling centralized (PlotStyle + matplotlib rcParams helper).
- Keep plotting functions side-effect free (they only draw on provided axes).
- Match the look-and-feel of `3_Plot_forecasts.ipynb` (colors, grid, legend, line styles).
"""

from typing import Any, Literal

import itertools

import arviz as az

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import numpy as np
import pandas as pd


DashStyle = str | tuple[float, tuple[float, ...]]

Language = Literal["en", "fr"]
DocumentType = Literal["paper", "note"]


class PlotStyle:
    """Default plot style configuration.

    Pass the instance to plotting functions to use its colors and naming overrides.
    During instantiation, apply matplotlib rcParams.
    """

    def __init__(
        self,
        *,
        language: Language = "en",
        document_type: DocumentType = "paper",
    ) -> None:
        self.language = language
        self.document_type = document_type

        self.scale_by_document_type: dict[DocumentType, float] = {
            "paper": 1.4,
            "note": 1.0,
        }

        self.scale: float = self.scale_by_document_type[self.document_type]

        self.linewidth: float = 1.5 * self.scale

        self.figsize = (7, 4)

        self.base_color = "#0e294c"
        self.accent_color = "#d4af37"
        self.gray_color = "#6c757d"
        self.grid_color = "#5F86A5"

        self.palette = [
            "#1f4788",
            "#4a7c59",
            "#457b9d",
            "#8b7f7b",
            "#264653",
            "#6a4c93",
            "#e76f51",
            "#06aed5",
            "#f4a261",
            "#2a9d8f",
        ]

        self.benchmark_name_overrides = {}

        self.category_name_overrides = {
            "en": {},
            "fr": {
                "Domain Specific Questions": "Questions Spécialisées par Domaine",
                "Core AGI Progress": "Progrès vers l'AGI",
                "General Reasoning": "Raisonnement Général",
                "Autonomous SWE": "Ingénierie Logicielle Autonome",
                "Multimodal Understanding": "Compréhension Multimodale",
                "Biology": "Expertise en Biologie",
                "Agentic Computer Use": "Opérations Agentiques sur Ordinateur",
                "Advanced Language and Writing": "Langage et Rédaction",
                "High End Math Reasoning": "Raisonnement Mathématique Avancé",
                "Chemistry": "Expertise en Chimie",
                "Commonsense QA": "Sens Commun",
            },
        }
        self.default_rcparams= { 
            # Fonts 
            "font.family": "sans-serif", 
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], 
            # Colors 
            "axes.labelcolor": self.base_color, 
            "xtick.color": self.base_color, 
            "ytick.color": self.base_color, 
            "text.color": self.base_color, 
            # Background / spines 
            "axes.facecolor": "none", 
            "figure.facecolor": "none", 
            # Grid 
            "axes.grid": True, 
            "grid.alpha": 0.1, 
            "grid.linestyle": "-", 
            "grid.linewidth": 0.8, 
            "grid.color": self.grid_color, 
            # Legend frame 
            "legend.frameon": False, 
            # Font sizes (scaled) 
            "font.size": 12 * self.scale, 
            "axes.titlesize": 16 * self.scale, 
            "axes.labelsize": 13 * self.scale, 
            "xtick.labelsize": 8 * self.scale, 
            "ytick.labelsize": 8 * self.scale, 
            "legend.fontsize": 10, 
            "legend.title_fontsize": 10 * self.scale, 
            "figure.titlesize": 16 * self.scale, 
            # Font weights 
            "axes.labelweight": 500, 
            "axes.titleweight": 600, 
        }

        plt.rcdefaults()
        plt.rcParams.update(self.default_rcparams)

    def _benchmark_name(self, raw_name: str) -> str:
        """Return the benchmark name used for display."""
        return self.benchmark_name_overrides.get(raw_name, raw_name)

    def _category_name(self, raw_name: str) -> str:
        """Return the category name used for display."""
        return self.category_name_overrides.get(self.language, {}).get(
            raw_name, raw_name
        )


def plot_calibration_curve(
    idata: az.InferenceData,
    n_points: int = 20,
    plot_style: PlotStyle = PlotStyle(),
) -> Axes:
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

    _, ax = plt.subplots(figsize=plot_style.figsize)

    ax.plot(
        confidence_levels, observed_coverage, color=plot_style.base_color, linewidth=2
    )
    ax.plot([0, 1], [0, 1], color=plot_style.gray_color, linestyle="--", linewidth=1)
    ax.set_xlabel("Expected coverage")
    ax.set_xlim(0, 1)
    ax.set_ylabel("Observed coverage")
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.legend(["Calibration", "Ideal"], loc="lower right")
    return ax


def plot_forecasts_by_category(
    *,
    observed: pd.DataFrame,
    forecast: pd.DataFrame,
    baselines: pd.DataFrame,
    end_date: pd.Timestamp,
    category_name: str,
    plot_style: PlotStyle = PlotStyle(),
) -> tuple[Figure, Axes]:
    """Plot one category: multiple benchmarks with observed data, forecast, and baselines."""
    fig, ax = plt.subplots(figsize=plot_style.figsize)

    benchmarks_ordered = _benchmark_plot_order(observed, forecast)

    for bench, color in zip(benchmarks_ordered, itertools.cycle(plot_style.palette)):
        obs_b = observed.loc[observed["benchmark"] == bench]
        preds_b = forecast.loc[forecast["benchmark"] == bench]
        baselines_b = baselines.loc[baselines["benchmark"] == bench]

        if obs_b.empty:
            continue

        last_date = pd.to_datetime(obs_b["release_date"].max())

        _plot_datapoints(
            ax,
            obs_b,
            color=color,
            size=40 * plot_style.scale,
            alpha=0.4,
            zorder=1,
        )
        _plot_forecast_with_split_style(
            ax,
            preds_b,
            color=color,
            last_observed_date=last_date,
            label=plot_style._benchmark_name(str(bench)),
            ci_alpha=0.2,
            observed_alpha=0.8,
            forecast_alpha=0.5,
            linewidth=1.5 * plot_style.scale,
            dash_style=(5, (4, 2)),
            zorder=2,
        )
        _plot_baseline_points(
            ax,
            baselines_b,
            preds_b,
            end_date=pd.to_datetime("2030-01-01"),
            color=color,
            size=150 * plot_style.scale,
            zorder=3,
        )

    ax.set_xlim(right=end_date)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlabel("")

    ax.set_ylim(0.0, 1.05)
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y * 100:.0f}%"))
    ax.set_ylabel("Performance")

    ax.grid(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(plot_style.base_color)

    ax.tick_params(axis="y", left=False, right=False)

    if plot_style.language == "fr":
        ax.set_title(plot_style._category_name(str(category_name)), pad=20)

    bench_legend = ax.legend(
        loc="lower right",
        fancybox=False,
        ncol=1,
        handlelength=1.5,
    )
    for text in bench_legend.get_texts():
        text.set_color(plot_style.base_color)
    for line in bench_legend.get_lines():
        line.set_linewidth(2.0)
        line.set_linestyle("-")
        line.set_alpha(1.0)
    ax.add_artist(bench_legend)

    """baseline_legend = _add_baseline_legend(ax, plot_style=plot_style)
    ax.add_artist(baseline_legend)"""

    fig.tight_layout()
    return fig, ax


def plot_harvey_asymmetry(
    idata: az.InferenceData,
    plot_style: PlotStyle = PlotStyle(),
    n_points: int = 500,
) -> tuple[Figure, Axes]:
    """Illustrative figure: Harvey curve shapes vs logistic (centered at 50%).

    Uses posterior medians of alpha per benchmark.
    Plots:
      - all Harvey curves (one per benchmark), semi-transparent, same color
      - bold median-Harvey curve
      - dashed logistic reference curve
    X-axis is normalized time (k=1), with each Harvey curve shifted so y=0.5 at x=0.
    """
    if "alpha" not in idata.posterior:
        raise ValueError(
            "plot_asymmetry_visualization requires idata.posterior['alpha']."
        )

    # --- Colors: match the reference figure using PlotStyle palette (no new hardcoding) ---
    # Expected palette order (from PlotStyle): [navy, green, steel-blue, ...]
    median_color = (
        plot_style.palette[0] if len(plot_style.palette) > 0 else plot_style.base_color
    )
    logistic_color = (
        plot_style.palette[1] if len(plot_style.palette) > 1 else plot_style.gray_color
    )
    family_color = (
        plot_style.palette[2] if len(plot_style.palette) > 2 else plot_style.gray_color
    )

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
    fig, ax = plt.subplots(figsize=plot_style.figsize)

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
    if plot_style.language == "fr":
        title = "Asymétrie des courbes de progrès : Harvey vs. Logistique"
        xlabel = "Temps (normalisé)"
        ylabel = "Performance"
        harvey_label = f"Courbes de Harvey estimées\nsur {n_benchmarks} benchmarks"
        logistic_label = "Courbe logistique\n(symétrique)"
    elif plot_style.language == "en":
        xlabel = "Time (normalized)"
        ylabel = "Performance"
        harvey_label = f"Estimated Harvey curves\nacross {n_benchmarks} benchmarks"
        logistic_label = "Logistic curve\n(symmetric)"
    else:
        raise ValueError(f"language must be 'en' or 'fr', got {plot_style.language!r}")

    # --- Annotations (match reference style) ---
    bbox = dict(
        boxstyle="round,pad=0.5", facecolor="white", edgecolor="none", alpha=0.9
    )

    ax.annotate(
        harvey_label,
        xy=(-2.0, float(_harvey_sigmoid(np.array([-2.0 + z0_med]), median_alpha)[0])),
        xytext=(-2.9, 0.70),
        color=median_color,
        ha="center",
        va="center",
        arrowprops=dict(arrowstyle="->", color=median_color, lw=1.5),
        bbox=bbox,
    )

    ax.annotate(
        logistic_label,
        xy=(1.0, float(_logistic_sigmoid(np.array([1.0]))[0])),
        xytext=(3.2, 0.45),
        color=logistic_color,
        ha="center",
        va="center",
        arrowprops=dict(arrowstyle="->", color=logistic_color, lw=1.5),
        bbox=bbox,
    )

    # --- Formatting (match reference) ---
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if plot_style.language == "fr":
        ax.set_title(title, pad=20)

    ax.grid(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(plot_style.base_color)

    ax.set_xlim(-6.0, 6.0)

    # Point (4): exact y-limits and ticks
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y * 100:.0f}%"))

    ax.tick_params(axis="y", left=False, right=False)

    plt.tight_layout()
    return fig, ax


def plot_saturation_proportion_posterior(
    idata: az.InferenceData,
    *,
    prepared_frontier: pd.DataFrame,
    target_date: pd.Timestamp | str,
    saturation_fraction: float = 0.95,
    ci_level: float = 0.80,
    plot_style: PlotStyle = PlotStyle(),
) -> tuple[Figure, Axes, dict[str, object]]:
    """Posterior histogram: proportion of benchmarks above saturation threshold by target_date.

    Uses only posterior parameters from `idata` and the per-benchmark first observation date
    from `prepared_frontier` (no disk IO).

    Interpretation:
      - Work on the *normalized* curve in [0, 1] (sigmoid output).
      - "Saturated" means sigmoid(t_target) > saturation_fraction, which is equivalent to
        mu(t_target) > l + saturation_fraction * (L - l) (so L/l are not needed here).

    Returns (fig, ax, summary) where summary includes mean/median/std and 80/95% CIs.
    """
    if not (0.0 < saturation_fraction < 1.0):
        raise ValueError("saturation_fraction must be in (0, 1)")
    if not (0.0 < ci_level < 1.0):
        raise ValueError("ci_level must be in (0, 1)")
    if "k" not in idata.posterior or "tau" not in idata.posterior:
        raise ValueError("Requires idata.posterior['k'] and idata.posterior['tau'].")

    target_ts = pd.to_datetime(target_date)
    if pd.isna(target_ts):
        raise ValueError(f"Could not parse target_date={target_date!r}")

    # --- Colors (match previous figures, reuse PlotStyle palette) ---
    # Base bars: steel-blue-ish (palette[2]); highlighted bars: navy (palette[0]); median line: gold accent.
    highlight_color = (
        plot_style.palette[0] if len(plot_style.palette) > 0 else plot_style.base_color
    )
    base_bar_color = (
        plot_style.palette[2] if len(plot_style.palette) > 2 else plot_style.gray_color
    )
    median_line_color = plot_style.accent_color

    # --- Benchmark order / alignment with posterior coords ---
    bench_coord = idata.posterior["k"].coords.get("benchmark", None)
    if bench_coord is None:
        raise ValueError("Posterior 'k' must have a 'benchmark' coordinate.")
    benchmarks = [str(b) for b in bench_coord.to_numpy().tolist()]
    n_benchmarks = len(benchmarks)

    # --- Per-benchmark start date (first observation) from prepared_frontier ---
    pf = prepared_frontier.copy()
    if "benchmark" not in pf.columns or "release_date" not in pf.columns:
        raise ValueError(
            "prepared_frontier must contain 'benchmark' and 'release_date' columns."
        )
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
    k = (
        posterior["k"]
        .stack(sample=("chain", "draw"))
        .transpose("benchmark", "sample")
        .to_numpy()
    )
    tau = (
        posterior["tau"]
        .stack(sample=("chain", "draw"))
        .transpose("benchmark", "sample")
        .to_numpy()
    )

    # z = k*(t - tau) (broadcast t_target over samples)
    z = k * (t_target[:, None] - tau)

    # --- Sigmoid: Harvey if alpha exists, else logistic ---
    if "alpha" in posterior:
        alpha = (
            posterior["alpha"]
            .stack(sample=("chain", "draw"))
            .transpose("benchmark", "sample")
            .to_numpy()
        )

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
    fig, ax = plt.subplots(figsize=plot_style.figsize)

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
    if plot_style.language == "fr":
        legend_median = f"Médiane: {median_prop:.1%}"
        legend_ci = f"IC {int(ci_level * 100)}%: [{ci_lo:.1%}, {ci_hi:.1%}]"
        title = f"Proportion de benchmarks saturés d'ici {target_ts.year}"
        subtitle = (
            f"Saturé = performance > {saturation_fraction:.0%} de l'asymptote estimée"
        )
        xlabel = f"Proportion de benchmarks > {saturation_fraction:.0%} de L"
        ylabel = "Densité de probabilité"
    elif plot_style.language == "en":
        legend_median = f"Median: {median_prop:.1%}"
        legend_ci = f"{int(ci_level * 100)}% CI: [{ci_lo:.1%}, {ci_hi:.1%}]"
        xlabel = f"Proportion of benchmarks > {saturation_fraction:.0%} of L"
        ylabel = "Probability density"
    else:
        raise ValueError(f"language must be 'en' or 'fr', got {plot_style.language!r}")

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
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if plot_style.language == "fr":
        ax.set_title(title, pad=20)
        ax.text(
            0.5,
            1.0,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            color=plot_style.gray_color,
            style="italic",
        )

    # Grid & spines
    ax.grid(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(plot_style.base_color)

    ax.tick_params(axis="y", left=False, right=False)

    # X as percent, bounds like the reference
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x * 100:.0f}%"))
    ax.set_xlim(-0.05, 1.05)

    # Legend
    legend = ax.legend(
        loc="upper left",
        frameon=True,
        framealpha=0.95,
        fontsize=plot_style.scale * 10,
        edgecolor=plot_style.base_color,
        fancybox=True,
    )
    for text in legend.get_texts():
        text.set_color(plot_style.base_color)
    fig.tight_layout()
    return fig, ax, summary


def _plot_datapoints(
    ax: plt.Axes,
    data: pd.DataFrame,
    *,
    color: str,
    size: float,
    alpha: float,
    zorder: int,
) -> None:
    """Scatter plot of observed datapoints."""
    ax.scatter(
        data["release_date"],
        data["score"],
        color=color,
        s=size,
        alpha=alpha,
        edgecolors="none",
        zorder=zorder,
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
    zorder: int,
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
        zorder=zorder - 1,
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
            zorder=zorder,
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
            zorder=zorder,
        )


def _plot_baseline_points(
    ax: plt.Axes,
    baselines: pd.DataFrame,
    preds: pd.DataFrame,
    end_date: pd.Timestamp,
    *,
    color: str,
    size: float,
    zorder: int,
) -> None:
    baselines = baselines.assign(
        date=lambda df: _assign_dates_to_baselines(df, preds, end_date),
        marker=lambda df: _assign_marker_to_baselines(df),
        facecolor=lambda df: _assign_facecolor_to_baselines(df, color),
    )
    for row in baselines.itertuples(index=False):
        ax.scatter(
            row.date,
            row.score,
            edgecolors=color,
            facecolors=row.facecolor,
            s=size,
            marker=row.marker,
            zorder=zorder,
        )


def _assign_dates_to_baselines(
    baselines: pd.DataFrame,
    preds: pd.DataFrame,
    end_date: pd.Timestamp,
) -> pd.Series:
    """Assign dates to baseline points based on forecast curves."""
    out = pd.Series(index=baselines.index, dtype="datetime64[ns]")
    for bench, g in baselines.groupby("benchmark", sort=False):
        preds_bench = (
            preds.loc[preds["benchmark"] == bench]
            .sort_values("mu_mean")
            .drop_duplicates(subset=["mu_mean"])
        )
        dates = np.interp(
            g["score"].to_numpy(),
            preds_bench["mu_mean"].to_numpy(),
            preds_bench["release_date"].to_numpy().astype(np.int64),
            right=end_date.value,
        )
        out.loc[g.index] = pd.to_datetime(dates.astype(np.int64))
    return out


def _assign_marker_to_baselines(baselines: pd.DataFrame) -> pd.Series:
    """Assign marker styles to baseline points based on their name."""

    def polygon(numsides):
        return (numsides, 0, 0)

    def star(numsides):
        return (numsides, 1, 0)

    def asterisk(numsides):
        return (numsides, 2, 0)

    map_group_to_marker = {
        "Average Human": star(3),
        "Skilled Generalist": star(4),
        "Domain Expert": star(5),
        "Top Performer": star(6),
        "Committee of Average Humans": polygon(3),
        "Committee of Skilled Generalists": polygon(4),
        "Committee of Domain Experts": polygon(5),
        "Committee of Top Performers": polygon(6),
        "High School Qualifier": star(5),
        "High School Top Performer": star(6),
    }

    return baselines["group"].map(map_group_to_marker).fillna("x")

def _assign_facecolor_to_baselines(baselines: pd.DataFrame, color: str) -> pd.Series:
    """Assign marker alpha to baseline points based on their name."""

    def marker_facecolor(group: str, color: str) -> str:
        if "High School" in group:
            return "none"
        else:
            return color

    return baselines["group"].map(lambda group: marker_facecolor(group, color))


def _benchmark_plot_order(observed: pd.DataFrame, forecast: pd.DataFrame) -> list[str]:
    """Return benchmark names ordered by posterior mean tau (left-to-right in the plot)."""
    if "mean_tau" in forecast.columns:
        order = (
            forecast[["benchmark", "mean_tau"]]
            .dropna()
            .drop_duplicates()
            .sort_values("mean_tau")["benchmark"]
            .astype(str)
            .tolist()
        )
        if order:
            return order

    # Fallback: stable alphabetical order.
    return sorted(set(observed["benchmark"].astype(str)))


def _add_baseline_legend(ax: Axes, *, plot_style: PlotStyle) -> plt.Legend:
    # Keep this list in the order you want to show it.
    baseline_cols: list[tuple[str, int]] = [
        ("Average Human", 3),
        ("Skilled Generalist", 4),
        ("Domain Expert", 5),
        ("Top Performer", 6),
    ]

    def polygon(n: int):
        return (n, 0, 0)

    def star(n: int):
        return (n, 1, 0)

    handles: list[tuple[Line2D, Line2D]] = []
    labels: list[str] = []

    for label, n in baseline_cols:
        committee = Line2D(
            [],
            [],
            linestyle="None",
            marker=polygon(n),
            markersize=9,
            markerfacecolor="none",
            markeredgecolor=plot_style.gray_color,
            markeredgewidth=1.2,
        )
        individual = Line2D(
            [],
            [],
            linestyle="None",
            marker=star(n),
            markersize=9,
            markerfacecolor=plot_style.gray_color,
            markeredgecolor=plot_style.gray_color,
            markeredgewidth=1.0,
        )
        handles.append((committee, individual))
        labels.append(label)

    leg = ax.legend(
        handles,
        labels,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.4)},
        title="Baselines (committee, individual)",
        loc="lower left",
        fontsize=9,
        frameon=False,
        ncol=2,
        columnspacing=1.2,
        handletextpad=0.6,
        handlelength=2.0,
    )
    leg.get_title().set_color(plot_style.base_color)
    for t in leg.get_texts():
        t.set_color(plot_style.base_color)

    return leg
