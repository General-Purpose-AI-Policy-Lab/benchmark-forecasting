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

        # Display names only (data keeps the bare benchmark name).  Benchmarks whose
        # series comes from a specific version of the benchmark carry it in the label;
        # names that already embed a version (ARC-AGI-2, OSWorld 2.0, Blueprint-Bench 2)
        # need no override.  Versions are written without parentheses.
        # Benchmark legend corner per category (default "lower right"): Autonomous SWE
        # has its curves bunched in the lower-right corner, so the legend goes top-left.
        # Benchmark legend placement.  By default the corner is chosen automatically
        # (see _pick_legend_loc): the candidates are tried in this order and the one
        # covering the fewest plotted points and curve samples wins.  A category can
        # still be forced to a corner here, e.g. {"Autonomous SWE": "upper left"}.
        self.legend_loc_overrides = {"General Reasoning": "upper left", "Mathematics": "lower right"}
        self.legend_loc_candidates = ["lower right", "upper left", "center left", "lower center", "center right", "upper center"]
        # Legend labels are single-line except the explicit line breaks listed here,
        # used where a long name would otherwise widen the legend over the curves.
        self.legend_label_overrides = {
            "OTIS Mock AIME 2024-2025": "OTIS Mock AIME\n2024-2025",
        }
        # Human-baseline labels whose position is forced rather than chosen by the
        # occlusion search: {(benchmark, group): "below"} puts the label right under its
        # marker (used where the automatic search keeps landing on neighbouring curves).
        self.baseline_label_overrides = {
            ("FrontierMath", "Committee of Domain Experts"): "below",
        }
        # A baseline marker is dated where the forecast curve reaches its score minus this
        # offset (in score units).  The curve approaches its asymptote L <= 1 without ever
        # reaching it, so a 100% baseline dated at exact equality would sit at the end of
        # the axis; one percentage point below gives the date the curve becomes
        # indistinguishable from it.
        self.baseline_date_offset = 0.01

        self.benchmark_name_overrides = {
            "FrontierMath": "FrontierMath v2",
            "FrontierMath Tier 4": "FrontierMath Tier 4 v2",
            "FrontierSWE": "FrontierSWE v2",
            "TerminalBench": "TerminalBench v2",
            "WeirdML": "WeirdML v2",
            "PostTrainBench": "PostTrainBench v1.1",
            # Shorter label so the Domain Specific Questions legend hugs the right edge.
            "Humanity's Last Exam": "HLE",
        }

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
                "Mathematics": "Mathématiques",
                "Chemistry": "Expertise en Chimie",
                "Commonsense QA": "Sens Commun",
            },
        }
        self.default_rcparams = {
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
            "axes.edgecolor": self.base_color,
            "figure.facecolor": "none",
            # Hide x-axis tick marks
            "xtick.major.size": 0,
            "xtick.minor.size": 0,
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
            "axes.titlesize": (14 if self.document_type == "note" else 16) * self.scale,
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

    def _legend_label(self, raw_name: str) -> str:
        """Display name for the benchmark legend (see ``legend_label_overrides``)."""
        return self.legend_label_overrides.get(raw_name, self._benchmark_name(raw_name))

    def _category_name(self, raw_name: str) -> str:
        """Return the category name used for display."""
        return self.category_name_overrides.get(self.language, {}).get(
            raw_name, raw_name
        )


def plot_calibration_curve(
    idata: az.InferenceData,
    n_points: int = 20,
    plot_style: PlotStyle = PlotStyle(),
) -> tuple[Figure, Axes]:
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

    fig, ax = plt.subplots(figsize=(4, 4))

    ax.scatter(
        confidence_levels,
        observed_coverage,
        color=plot_style.base_color,
        linewidth=1.5,
        zorder=2,
    )
    ax.plot(
        [0, 1],
        [0, 1],
        color=plot_style.gray_color,
        linestyle="--",
        linewidth=plot_style.linewidth,
        zorder=1,
    )
    ax.set_xlabel("Expected coverage")
    ax.set_xlim(0, 1)
    ax.set_ylabel("Observed coverage")
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.legend(["Empirical", "Perfect"], loc="lower right", fontsize=10 * plot_style.scale)
    return fig, ax


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
    benchmark_colors = {
        bench: color
        for bench, color in zip(
            benchmarks_ordered, itertools.cycle(plot_style.palette)
        )
    }

    for bench in benchmarks_ordered:
        color = benchmark_colors[bench]
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
            label=plot_style._legend_label(str(bench)),
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
            size=(80 if plot_style.document_type == "note" else 100) * plot_style.scale,
            zorder=3,
            note_mode=plot_style.document_type == "note",
            date_offset=plot_style.baseline_date_offset,
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
        ax.set_title(plot_style._category_name(str(category_name)), pad=8)

    legend_kwargs = dict(fancybox=False, ncol=1, handlelength=1.5)
    forced = plot_style.legend_loc_overrides.get(str(category_name))
    loc = forced if forced else _pick_legend_loc(fig, ax, observed, forecast, plot_style, legend_kwargs)
    bench_legend = ax.legend(loc=loc, **legend_kwargs)
    for text in bench_legend.get_texts():
        text.set_color(plot_style.base_color)
    for line in bench_legend.get_lines():
        line.set_linewidth(2.0)
        line.set_linestyle("-")
        line.set_alpha(1.0)
    ax.add_artist(bench_legend)

    # Inline baseline labels (note and paper figures).  These come after tight_layout
    # because the label placement measures occlusion in pixels: deciding before the
    # layout is settled would optimise a geometry the reader never sees.
    fig.tight_layout()
    _add_baseline_labels(
        ax,
        baselines=baselines,
        preds=forecast,
        observed=observed,
        end_date=pd.to_datetime("2030-01-01"),
        benchmark_colors=benchmark_colors,
        plot_style=plot_style,
    )
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
        ax.plot(t, y, color=family_color, alpha=0.10, linewidth=1.0, zorder=1)

    # Bold median Harvey curve
    z0_med = _z_half(median_alpha)
    y_med = _harvey_sigmoid(t + z0_med, median_alpha)
    ax.plot(t, y_med, color=median_color, linewidth=plot_style.linewidth, zorder=3)

    # Logistic reference (dashed)
    y_log = _logistic_sigmoid(t)
    ax.plot(t, y_log, color=logistic_color, linewidth=plot_style.linewidth, linestyle="--", zorder=4)

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
        fontsize = 10 * plot_style.scale,
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
        fontsize = 10 * plot_style.scale,
    )

    # --- Formatting (match reference) ---
    ax.set_xlabel(xlabel, fontsize=11 * plot_style.scale)
    ax.set_ylabel(ylabel, fontsize=11 * plot_style.scale)
    if plot_style.language == "fr":
        ax.set_title(title, pad=8)

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

    ax.tick_params(
        axis="both", 
        which="major",
        labelsize=10 * plot_style.scale, 
        left=False, 
        right=False
    )

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
    ax.set_xlabel(xlabel, fontsize=11 * plot_style.scale)
    ax.set_ylabel(ylabel, fontsize=11 * plot_style.scale)
    # Grid & spines
    ax.grid(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(plot_style.base_color)

    ax.tick_params(
        axis="both",  
        which="major",
        labelsize=10 * plot_style.scale, 
        left=False, 
        right=False
    )

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


def plot_L_intervals(
    idata: az.InferenceData,
    *,
    prepared_frontier: pd.DataFrame | None = None,
    ci_level: float = 0.80,
    plot_style: PlotStyle = PlotStyle(),
) -> tuple[Figure, Axes]:
    """Forest plot of the per-benchmark upper asymptote $L$.

    Benchmarks are ordered by posterior median.  When `prepared_frontier` is given,
    the best score observed so far is overlaid, which shows how much of the
    asymptote is already attained (and how much is extrapolation).
    """
    if "L" not in idata.posterior:
        raise ValueError("Requires the deterministic 'L' in idata.posterior.")

    L = idata.posterior["L"].stack(sample=("chain", "draw")).transpose("benchmark", "sample")
    benchmarks = [str(b) for b in L.coords["benchmark"].to_numpy().tolist()]
    values = L.to_numpy()

    lo_q = 100.0 * (1.0 - ci_level) / 2.0
    hi_q = 100.0 * (1.0 + ci_level) / 2.0
    median = np.median(values, axis=1)
    lower = np.percentile(values, lo_q, axis=1)
    upper = np.percentile(values, hi_q, axis=1)

    order = np.argsort(median)
    y = np.arange(len(benchmarks))

    best_observed = None
    if prepared_frontier is not None:
        best_observed = (
            prepared_frontier.groupby("benchmark")["score"].max().reindex(benchmarks).to_numpy()
        )

    fig, ax = plt.subplots(figsize=(7, 0.19 * len(benchmarks) + 1.4))

    ax.hlines(
        y,
        lower[order],
        upper[order],
        color=plot_style.palette[2],
        linewidth=1.6,
        alpha=0.9,
        zorder=2,
    )
    ax.scatter(
        median[order],
        y,
        s=14,
        color=plot_style.base_color,
        zorder=3,
        label="Posterior median" if plot_style.language == "en" else "Médiane a posteriori",
    )
    if best_observed is not None:
        ax.scatter(
            best_observed[order],
            y,
            s=16,
            marker="|",
            linewidths=1.6,
            color=plot_style.accent_color,
            zorder=4,
            label="Best score observed" if plot_style.language == "en" else "Meilleur score observé",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(
        [plot_style._benchmark_name(benchmarks[i]) for i in order],
        fontsize=6.5 * plot_style.scale,
    )
    ax.set_ylim(-1, len(benchmarks))

    if plot_style.language == "fr":
        ax.set_xlabel("Asymptote supérieure $L$")
    else:
        ax.set_xlabel("Upper asymptote $L$")

    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x * 100:.0f}%"))
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower left", fontsize=8 * plot_style.scale)
    fig.tight_layout()
    return fig, ax


def plot_L_distribution(
    idata: az.InferenceData,
    *,
    L_min: float = 0.75,
    plot_style: PlotStyle = PlotStyle(),
    ci_level: float = 0.80,
) -> tuple[Figure, Axes]:
    """Population distribution of the upper asymptotes L, with one point per benchmark.

    The hierarchical prior on the rescaled asymptote is Beta(mu, sigma) on
    [L_min, 1]; the curve is that Beta at the posterior medians of ``L_raw_mu`` and
    ``L_raw_sigma``, and the points are the per-benchmark posterior medians of ``L``
    (pinned asymptotes appear as points at their fixed value).  This is the note figure
    formerly produced by the retired 3_Plot_forecasts notebook, rebuilt from the
    current model.
    """
    from scipy import stats

    posterior = idata.posterior
    L_range = 1.0 - L_min
    mu = float(np.median(posterior["L_raw_mu"].values))
    sigma = float(np.median(posterior["L_raw_sigma"].values))
    sigma = min(sigma, np.sqrt(mu * (1 - mu)) * 0.98)
    nu = mu * (1 - mu) / sigma**2 - 1
    a, b = max(mu * nu, 0.5), max((1 - mu) * nu, 0.5)
    beta = stats.beta(a, b, loc=L_min, scale=L_range)

    L_med = np.median(posterior["L"].values.reshape(-1, posterior["L"].shape[-1]), axis=0)
    n_bench = L_med.size
    mean_L = L_min + L_range * mu

    grid = np.linspace(0.5, 1.0, 600)
    pdf = beta.pdf(grid)
    pdf = pdf / pdf.max() * 8.0
    lo, hi = beta.ppf((1 - ci_level) / 2), beta.ppf(1 - (1 - ci_level) / 2)

    fr = plot_style.language == "fr"
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(grid, pdf, color=plot_style.palette[0], linewidth=3, zorder=3)
    ax.fill_between(grid, 0, pdf, color="#b8cfe0", alpha=0.3, zorder=2)
    mask = (grid >= lo) & (grid <= hi)
    ax.fill_between(grid[mask], 0, pdf[mask], color="#7fa8c9", alpha=0.5, zorder=2.5)

    rng = np.random.default_rng(42)
    jitter = rng.uniform(-0.3, 0.3, size=n_bench)
    ax.scatter(L_med, jitter, s=60, color=plot_style.palette[2], edgecolors="white",
               linewidth=1.5, alpha=0.7, zorder=4)

    ymax = pdf.max() * 1.12
    ax.set_ylim(-1, ymax)
    ax.axvline(1.0, color=plot_style.gray_color, linestyle="--", linewidth=2, alpha=0.5, zorder=1)
    ax.text(1.011, ymax, "Perfection (100%)" if fr else "Perfect score (100%)", fontsize=12,
            color=plot_style.gray_color, ha="right", va="top", rotation=90)
    ax.axvline(mean_L, color=plot_style.palette[0], linestyle=":", linewidth=2, alpha=0.7, zorder=2)
    ax.text(mean_L + 0.006, ymax, (f"Moyenne ({mean_L:.1%})" if fr else f"Mean ({mean_L:.1%})"),
            fontsize=12, color=plot_style.palette[0], ha="center", va="top", rotation=90,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.9))
    ax.annotate(f"{n_bench} benchmarks", xy=(float(np.median(L_med)), 0), xytext=(0.95, -1.7),
                fontsize=14, color=plot_style.palette[2], ha="center", va="center",
                arrowprops=dict(arrowstyle="->", color=plot_style.palette[2], lw=1.5),
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="none", alpha=0.9),
                annotation_clip=False)

    ax.set_xlabel("Performance maximale" if fr else "Upper asymptote", fontsize=17,
                  fontweight="500", color=plot_style.base_color)
    ax.set_title("Distribution estimée des performances maximales" if fr
                 else "Estimated distribution of the upper asymptotes",
                 fontsize=17, fontweight="600", color=plot_style.base_color, pad=20)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x * 100:.0f}%"))
    ax.tick_params(axis="x", labelsize=13, colors=plot_style.base_color)
    ax.set_yticks([])
    ax.tick_params(axis="y", left=False, right=False)
    ax.grid(True, alpha=0.15, linewidth=0.8, color=plot_style.grid_color, axis="x")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(plot_style.base_color)
    ax.set_xlim(0.5, 1.05)
    fig.tight_layout()
    return fig, ax


def plot_hyperparameters(
    idata: az.InferenceData,
    *,
    L_min: float = 0.75,
    plot_style: PlotStyle = PlotStyle(),
) -> tuple[Figure, np.ndarray]:
    """Posterior distributions of the population-level hyperparameters.

    Panels are drawn only for the hyperparameters present in the fit, so the same
    function works for logistic variants (no $\\alpha$) and normal-likelihood
    variants (no $s$).
    """
    posterior = idata.posterior

    def _draws(name: str) -> np.ndarray | None:
        if name not in posterior:
            return None
        return posterior[name].values.flatten()

    panels: list[tuple[str, np.ndarray, str | None]] = []  # L_min maps L_raw_mu back onto the score scale

    L_raw_mu = _draws("L_raw_mu")
    if L_raw_mu is not None:
        panels.append(
            (
                "Upper asymptote $\\mu_L$" if plot_style.language == "en" else "Asymptote $\\mu_L$",
                L_min + (1.0 - L_min) * L_raw_mu,
                None,
            )
        )

    k_mu = _draws("k_mu")
    if k_mu is not None:
        panels.append(
            (
                "Growth rate $k_\\mu$ (per day)"
                if plot_style.language == "en"
                else "Taux de croissance $k_\\mu$ (par jour)",
                k_mu,
                None,
            )
        )

    alpha_raw_mu = _draws("alpha_raw_mu")
    if alpha_raw_mu is not None:
        panels.append(
            (
                "Harvey shape $\\alpha_\\mu$" if plot_style.language == "en" else "Forme $\\alpha_\\mu$",
                alpha_raw_mu + 1.0,
                "$\\alpha = 2$ (logistic)" if plot_style.language == "en" else "$\\alpha = 2$ (logistique)",
            )
        )

    s_mu = _draws("s_mu")
    if s_mu is not None:
        panels.append(
            (
                "Skewness $s_\\mu$" if plot_style.language == "en" else "Asymétrie $s_\\mu$",
                s_mu,
                "$s = 0$ (symmetric)" if plot_style.language == "en" else "$s = 0$ (symétrique)",
            )
        )

    if not panels:
        raise ValueError("No population-level hyperparameters found in the posterior.")

    ncols = 2
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7, 2.5 * nrows))
    axes_flat = np.atleast_1d(axes).flatten()

    ylabel = "Probability density" if plot_style.language == "en" else "Densité de probabilité"

    for ax, (title, draws, ref_label) in zip(axes_flat, panels):
        ax.hist(
            draws,
            bins=50,
            density=True,
            color=plot_style.palette[2],
            alpha=0.75,
            edgecolor="white",
            linewidth=0.4,
        )
        median = float(np.median(draws))
        ax.axvline(
            median,
            color=plot_style.accent_color,
            linewidth=2.0,
            label=f"{median:.3g}",
        )
        if ref_label is not None:
            ref_value = 2.0 if "alpha" in title or "\\alpha" in title else 0.0
            ax.axvline(
                ref_value,
                color=plot_style.gray_color,
                linestyle="--",
                linewidth=1.4,
                label=ref_label,
            )
        ax.set_xlabel(title, fontsize=9 * plot_style.scale)
        ax.set_ylabel(ylabel, fontsize=8 * plot_style.scale)
        ax.set_yticks([])
        ax.grid(True, axis="x")
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.legend(fontsize=7.5 * plot_style.scale, loc="upper right")

    for ax in axes_flat[len(panels) :]:
        ax.set_visible(False)

    fig.tight_layout()
    return fig, axes_flat


def _legend_occlusion(fig, ax, observed: pd.DataFrame, forecast: pd.DataFrame, legend) -> float:
    """Share of plotted material (observed points and curve samples) under a legend box."""
    fig.canvas.draw()
    bbox = legend.get_window_extent(fig.canvas.get_renderer())
    pts = []
    if not observed.empty:
        xy = np.column_stack([mdates.date2num(pd.to_datetime(observed["release_date"])), observed["score"].to_numpy(float)])
        pts.append(ax.transData.transform(xy))
    if not forecast.empty and "mu_mean" in forecast.columns:
        xy = np.column_stack([mdates.date2num(pd.to_datetime(forecast["release_date"])), forecast["mu_mean"].to_numpy(float)])
        pts.append(ax.transData.transform(xy))
    if not pts:
        return 0.0
    P = np.vstack(pts)
    inside = (P[:, 0] >= bbox.x0) & (P[:, 0] <= bbox.x1) & (P[:, 1] >= bbox.y0) & (P[:, 1] <= bbox.y1)
    return float(inside.mean())


def _pick_legend_loc(fig, ax, observed, forecast, plot_style, legend_kwargs) -> str:
    """Try each candidate corner and keep the one hiding the least; ties go to the first."""
    best_loc, best_score = plot_style.legend_loc_candidates[0], np.inf
    for loc in plot_style.legend_loc_candidates:
        leg = ax.legend(loc=loc, **legend_kwargs)
        score = _legend_occlusion(fig, ax, observed, forecast, leg)
        leg.remove()
        if score < best_score - 1e-9:
            best_loc, best_score = loc, score
    return best_loc


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
    note_mode: bool = False,
    date_offset: float = 0.01,
) -> None:
    baselines = baselines.assign(
        date=lambda df: _assign_dates_to_baselines(df, preds, end_date, offset=date_offset),
        marker=lambda df: _assign_marker_to_baselines(df),
        facecolor=lambda df: _assign_facecolor_to_baselines(df, color),
    )
    # In note mode: uniform 4-branch stars, always filled (incl. High School).
    note_star = (4, 1, 0)
    for row in baselines.itertuples(index=False):
        ax.scatter(
            row.date,
            row.score,
            edgecolors=color,
            facecolors=color if note_mode else row.facecolor,
            s=size,
            marker=note_star if note_mode else row.marker,
            zorder=zorder,
        )


def _assign_dates_to_baselines(
    baselines: pd.DataFrame,
    preds: pd.DataFrame,
    end_date: pd.Timestamp,
    offset: float = 0.01,
) -> pd.Series:
    """Date each baseline where the forecast mean first reaches ``score - offset``.

    ``offset`` (see ``PlotStyle.baseline_date_offset``) keeps baselines at or near 100%,
    which the curve never reaches exactly, from being pushed to ``end_date``.
    """
    out = pd.Series(index=baselines.index, dtype="datetime64[ns]")
    for bench, g in baselines.groupby("benchmark", sort=False):
        preds_bench = (
            preds.loc[preds["benchmark"] == bench]
            .sort_values("mu_mean")
            .drop_duplicates(subset=["mu_mean"])
        )
        dates = np.interp(
            g["score"].to_numpy() - offset,
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

    # Only the number of branches encodes the expertise level.  Committees and the
    # high-school cohort used to carry a second distinction (polygon vs star, hollow
    # vs filled), which put up to ten marker meanings on one panel; the inline text
    # label already names the group, so that channel was redundant.
    map_group_to_marker = {
        "Average Human": star(3),
        "Skilled Generalist": star(4),
        "Domain Expert": star(5),
        "Top Performer": star(6),
        "Committee of Average Humans": star(3),
        "Committee of Skilled Generalists": star(4),
        "Committee of Domain Experts": star(5),
        "Committee of Top Performers": star(6),
        "High School Qualifier": star(5),
        "High School Top Performer": star(6),
    }

    return baselines["group"].map(map_group_to_marker).fillna("x")


def _assign_facecolor_to_baselines(baselines: pd.DataFrame, color: str) -> pd.Series:
    """Assign marker alpha to baseline points based on their name."""

    def marker_facecolor(group: str, color: str) -> str:
        # Previously the high-school cohort was drawn hollow; that distinction now
        # lives in the text label only, so every baseline marker is filled.
        return color

    return baselines["group"].map(lambda group: marker_facecolor(group, color))


# Nearest-neighbour linkage threshold for merging same-group baseline labels, as a
# fraction of each axis range.  Calibrated on the two cases that matter: closing the
# chain of seven biology expert baselines into one label needs 0.21, while the two
# ARC-AGI committee markers, which sit four years apart and must stay separate, are
# 0.47 apart.
MERGE_TOL = 0.22


def _label_box(
    ax: Axes,
    *,
    x_num: float,
    y: float,
    ha: str,
    size: tuple[float, float],
) -> tuple[float, float, float, float]:
    """Return the display-space extent a label would occupy at this anchor."""
    px, py = ax.transData.transform((x_num, y))
    w, h = size
    if ha == "left":
        x0 = px
    elif ha == "right":
        x0 = px - w
    else:
        x0 = px - w * 0.5
    return x0, x0 + w, py - h * 0.5, py + h * 0.5


def _measure_labels(
    ax: Axes,
    *,
    labels: list[str],
    fontsize: float,
) -> list[tuple[float, float]]:
    """Return each label's rendered (width, height) in pixels, box padding included.

    Falls back to a character-count estimate when the backend cannot hand back a
    renderer, so placement degrades to an approximation rather than failing.
    """
    fig = ax.figure
    pad = 0.2 * fontsize * fig.dpi / 72.0  # boxstyle="round,pad=0.2", in font units
    try:
        renderer = fig.canvas.get_renderer()
    except AttributeError:
        est = 0.58 * fontsize * fig.dpi / 72.0
        return [(len(t) * est + 2 * pad, fontsize * fig.dpi / 72.0 + 2 * pad) for t in labels]

    sizes: list[tuple[float, float]] = []
    for text in labels:
        probe = ax.text(
            0.5, 0.5, text, fontsize=fontsize, fontweight="bold", transform=ax.transAxes
        )
        extent = probe.get_window_extent(renderer)
        sizes.append((extent.width + 2 * pad, extent.height + 2 * pad))
        probe.remove()
    return sizes


def _choose_label_placements(
    ax: Axes,
    *,
    annotations: list[dict[str, Any]],
    label_ys: list[float],
    observed: pd.DataFrame,
    markers: list[dict[str, Any]],
    label_h: float,
    dx: float,
    fontsize: float,
    plot_style: PlotStyle,
) -> list[tuple[float, float, str]]:
    """Pick each label's side and height so it hides as little of the panel as possible.

    Placing a label on a fixed side of its marker is what puts it on top of the data:
    the vertical stack keeps labels apart from each other but knows nothing about where
    the score markers are, so a label with no vertical displacement at all still lands
    in the middle of a trajectory.  Here each label is scored against what it would
    actually cover -- observed scores, baseline markers, its own marker, other labels,
    the axes edges -- and the cheapest candidate wins, with displacement from the marker
    priced in so proximity is traded off rather than ignored.

    Returns ``(x in date units, y in data units, horizontal alignment)`` per annotation,
    in the order given.
    """
    x_lo_num, x_hi_num = ax.get_xlim()
    y_lo, y_hi = ax.get_ylim()
    x_range = x_hi_num - x_lo_num

    sizes = _measure_labels(ax, labels=[a["label"] for a in annotations], fontsize=fontsize)

    obs_px = np.empty((0, 2))
    if len(observed):
        obs_px = ax.transData.transform(
            np.column_stack(
                [
                    mdates.date2num(pd.to_datetime(observed["release_date"])),
                    observed["score"].to_numpy(dtype=float),
                ]
            )
        )
    # Every baseline marker, not the annotation anchors: a grouped label's anchor is a
    # centroid where no marker is drawn, and it is the drawn stars that must stay visible.
    marker_px = ax.transData.transform(
        np.array([[m["date_num"], m["score"]] for m in markers], dtype=float)
    )
    axes_box = ax.get_window_extent()

    def _count_inside(points: np.ndarray, box: tuple[float, float, float, float]) -> int:
        if not len(points):
            return 0
        x0, x1, y0, y1 = box
        return int(
            np.count_nonzero(
                (points[:, 0] >= x0)
                & (points[:, 0] <= x1)
                & (points[:, 1] >= y0)
                & (points[:, 1] <= y1)
            )
        )

    def _overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
        return not (a[1] <= b[0] or b[1] <= a[0] or a[3] <= b[2] or b[3] <= a[2])

    def _candidates(i: int) -> list[tuple[float, float, str, float]]:
        """Candidate (x, y, ha, tie-break penalty) placements for annotation ``i``."""
        ann = annotations[i]
        date_num = ann["date_num"]
        if ann.get("place") == "below":
            # Forced placement: centred right under the marker, no alternatives.
            return [(date_num, ann["score"] - 0.9 * label_h, "center", 0.0)]
        near_right = date_num > x_lo_num + 0.70 * x_range
        if ann.get("grouped", False):
            # A grouped label names a cluster, so centring it on the centroid reads best;
            # the offset variants are there for when the centre sits on the data.
            xs = [(date_num, "center", 0.0), (date_num - dx, "right", 0.3), (date_num + dx, "left", 0.3)]
        else:
            default, other = ("left", "right") if near_right else ("right", "left")
            xs = [
                (date_num - dx if default == "left" else date_num + dx,
                 "right" if default == "left" else "left", 0.0),
                (date_num - dx if other == "left" else date_num + dx,
                 "right" if other == "left" else "left", 0.3),
            ]
        out: list[tuple[float, float, str, float]] = []
        # One label height is the largest vertical move worth making: beyond that the
        # leader line has to cross a trajectory band, where it is the same colour as the
        # curve and disappears, leaving the label floating with nothing to attach it to.
        for step in (0.0, 1.0, -1.0):
            y = label_ys[i] + step * label_h
            if not (y_lo + label_h * 0.5 <= y <= y_hi - label_h * 0.4):
                continue
            for x_num, ha, x_pen in xs:
                out.append((x_num, y, ha, x_pen + abs(step) * 0.6))
        return out

    # Worst offender first: it gets the pick of the placements, and later labels work
    # around what it took.
    initial = [
        _count_inside(obs_px, _label_box(ax, x_num=c[0], y=c[1], ha=c[2], size=sizes[i]))
        for i, c in enumerate(
            [(_candidates(i) or [(annotations[i]["date_num"], label_ys[i], "left", 0.0)])[0]
             for i in range(len(annotations))]
        )
    ]
    order = sorted(range(len(annotations)), key=lambda i: -initial[i])

    chosen: list[tuple[float, float, str] | None] = [None] * len(annotations)
    placed_boxes: list[tuple[float, float, float, float]] = []

    for i in order:
        best: tuple[float, tuple[float, float, str], tuple[float, float, float, float]] | None = None
        for x_num, y, ha, tie_pen in _candidates(i):
            box = _label_box(ax, x_num=x_num, y=y, ha=ha, size=sizes[i])
            cost = (
                3.0 * _count_inside(obs_px, box)
                + 4.0 * _count_inside(marker_px, box)
                + 40.0 * sum(_overlap(box, other) for other in placed_boxes)
                # Priced so that a label only leaves its marker's own height to clear at
                # least two data points: moving is worth it, drifting is not.
                + 2.5 * abs(y - annotations[i]["score"]) / label_h
                + tie_pen
            )
            # Running off the panel is worse than any amount of occlusion.
            if box[0] < axes_box.x0 or box[1] > axes_box.x1:
                cost += 200.0
            if best is None or cost < best[0]:
                best = (cost, (x_num, y, ha), box)
        if best is None:  # no candidate fits vertically; keep the stack position
            date_num = annotations[i]["date_num"]
            chosen[i] = (date_num + dx, label_ys[i], "left")
            continue
        chosen[i] = best[1]
        placed_boxes.append(best[2])

    # The caller zips this with `annotations`, so dropping an entry would silently
    # attach every later label to the wrong marker.
    assert all(c is not None for c in chosen)
    return [c for c in chosen if c is not None]


def _add_baseline_labels(
    ax: Axes,
    *,
    baselines: pd.DataFrame,
    preds: pd.DataFrame,
    observed: pd.DataFrame,
    end_date: pd.Timestamp,
    benchmark_colors: dict[str, str],
    plot_style: PlotStyle,
) -> None:
    """Add inline text labels next to human-baseline markers (note figures only).

    When several markers share the same group label, they are merged into a
    single annotation placed at the center of mass of the markers, with a
    plural label (e.g. "Experts").
    """
    # Built inline (not from module-level dict) so %autoreload always sees it.
    _labels = {
        "fr": {
            "Average Human": ("Humain moyen", "Humains moyens"),
            "Skilled Generalist": ("Expert hors-domaine", "Experts hors-domaine"),
            "Domain Expert": ("Expert du domaine", "Experts du domaine"),
            "Top Performer": ("Top expert", "Top experts"),
            "Committee of Average Humans": ("Comité d'humains moyens", "Comités d'humains moyens"),
            "Committee of Skilled Generalists": ("Comité de généralistes", "Comités de généralistes"),
            "Committee of Domain Experts": ("Comité d'experts", "Comités d'experts"),
            "High School Qualifier": ("Lycéen qualifié", "Lycéens qualifiés"),
            "High School Top Performer": ("Top lycéen", "Top lycéens"),
        },
        "en": {
            "Average Human": ("Avg. human", "Avg. humans"),
            "Skilled Generalist": ("Skilled generalist", "Skilled generalists"),
            "Domain Expert": ("Expert", "Experts"),
            "Top Performer": ("Top performer", "Top performers"),
            "Committee of Average Humans": ("Avg. human committee", "Avg. human committees"),
            "Committee of Skilled Generalists": ("Generalist committee", "Generalist committees"),
            "Committee of Domain Experts": ("Expert\ncommittee", "Expert\ncommittees"),  # two lines: sits among steep curves
            "High School Qualifier": ("HS qualifier", "HS qualifiers"),
            "High School Top Performer": ("HS top performer", "HS top performers"),
        },
    }
    label_map = _labels.get(plot_style.language, {})

    # Collect all raw annotation data.
    raw_annotations: list[dict[str, Any]] = []
    for bench, g in baselines.groupby("benchmark", sort=False):
        color = benchmark_colors.get(str(bench))
        if color is None:
            continue  # benchmark not in this category
        preds_b = preds.loc[preds["benchmark"] == bench]
        if preds_b.empty:
            continue
        dates = _assign_dates_to_baselines(g, preds_b, end_date, offset=plot_style.baseline_date_offset)
        for idx, row in g.iterrows():
            date = dates.loc[idx]
            score = float(row["score"])
            group = str(row["group"])
            entry = label_map.get(group, (group, group))
            if isinstance(entry, str):
                singular = plural = entry
            else:
                singular, plural = entry
            raw_annotations.append(
                {
                    "date": date,
                    "date_num": mdates.date2num(date),
                    "score": score,
                    "group": group,
                    "benchmark": str(bench),
                    "singular": singular,
                    "plural": plural,
                    "color": color,
                }
            )

    if not raw_annotations:
        return

    # --- Group by label and compute center of mass ---
    by_group: dict[str, list[dict[str, Any]]] = {}
    for ann in raw_annotations:
        by_group.setdefault(ann["group"], []).append(ann)

    annotations: list[dict[str, Any]] = []
    for group, members in by_group.items():
        if len(members) == 1:
            m = members[0]
            annotations.append(
                {
                    "date_num": m["date_num"],
                    "score": m["score"],
                    "label": m["singular"],
                    "color": m["color"],
                    "place": plot_style.baseline_label_overrides.get((m["benchmark"], m["group"])),
                }
            )
        else:
            # Cluster members of the same group by proximity in both axes, so that
            # markers far apart in time are labelled separately even when they share a
            # group.  Linkage is nearest-neighbour: a marker joins a cluster when it is
            # close to *any* of its members, rather than requiring every pair to be
            # close.  That merges a spread-out family such as the biology experts into
            # one label, while keeping distant markers apart.
            x_span = max(ax.get_xlim()[1] - ax.get_xlim()[0], 1e-9)
            y_span = max(ax.get_ylim()[1] - ax.get_ylim()[0], 1e-9)

            def _gap(a: dict[str, Any], b: dict[str, Any]) -> float:
                return max(
                    abs(a["date_num"] - b["date_num"]) / x_span,
                    abs(a["score"] - b["score"]) / y_span,
                )

            members.sort(key=lambda m: m["score"])
            clusters = [[m] for m in members]
            merged = True
            while merged and len(clusters) > 1:
                merged = False
                for i in range(len(clusters)):
                    for j in range(i + 1, len(clusters)):
                        if any(_gap(a, b) <= MERGE_TOL for a in clusters[i] for b in clusters[j]):
                            clusters[i].extend(clusters.pop(j))
                            clusters[i].sort(key=lambda m: m["score"])
                            merged = True
                            break
                    if merged:
                        break

            for cluster in clusters:
                if len(cluster) == 1:
                    c = cluster[0]
                    annotations.append(
                        {
                            "date_num": c["date_num"],
                            "score": c["score"],
                            "label": c["singular"],
                            "color": c["color"],
                            "place": plot_style.baseline_label_overrides.get((c["benchmark"], c["group"])),
                        }
                    )
                else:
                    mean_date_num = float(np.mean([c["date_num"] for c in cluster]))
                    scores = [c["score"] for c in cluster]
                    mean_score = float(np.mean(scores))
                    # When all markers are tightly clustered (<5pp spread),
                    # push label above to avoid sitting on top of them.
                    if max(scores) - min(scores) < 0.05:
                        mean_score += 0.05
                    colors = {c["color"] for c in cluster}
                    label_color = colors.pop() if len(colors) == 1 else plot_style.base_color
                    annotations.append(
                        {
                            "date_num": mean_date_num,
                            "score": mean_score,
                            "label": cluster[0]["plural"],
                            "color": label_color,
                            "grouped": True,
                        }
                    )

    # Sort by score for the overlap-resolution pass.
    annotations.sort(key=lambda a: a["score"])

    # --- Overlap resolution in data-y space ---
    y_lo, y_hi = ax.get_ylim()
    label_h = (y_hi - y_lo) * (0.08 if plot_style.document_type == "paper" else 0.05)

    x_lo_num, x_hi_num = ax.get_xlim()
    x_range = x_hi_num - x_lo_num
    dx = x_range * 0.023  # small date offset to keep labels close

    def _place() -> list[float]:
        """Place labels at least ``label_h`` apart, as close to their markers as possible.

        Resolving collisions by pushing labels apart pairwise, or by hanging them from
        the top of the axes, can leave a label far below the marker it names: on the
        AGI-progress panel three baselines fall between 98% and 100%, and the third one
        ended up fifteen points below its star.  Minimising the total squared
        displacement instead is the classic isotonic-regression problem -- substituting
        ``u_r = score_r - r * label_h`` turns the spacing constraints into a
        monotonicity constraint -- so pool-adjacent-violators gives the exact optimum.
        Labels that must move end up one label-height from their neighbour, on whichever
        side of the marker costs least.
        """
        order = sorted(range(len(annotations)), key=lambda i: annotations[i]["score"])
        offsets = [annotations[i]["score"] - r * label_h for r, i in enumerate(order)]

        blocks: list[tuple[float, int]] = []  # (block mean, block size)
        for value in offsets:
            blocks.append((value, 1))
            while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
                (mean_hi, size_hi), (mean_lo, size_lo) = blocks.pop(), blocks.pop()
                size = size_lo + size_hi
                blocks.append(((mean_lo * size_lo + mean_hi * size_hi) / size, size))

        pooled: list[float] = []
        for mean, size in blocks:
            pooled.extend([mean] * size)

        ys = [0.0] * len(annotations)
        for rank, i in enumerate(order):
            ys[i] = pooled[rank] + rank * label_h
        return ys

    label_ys = _place()

    # Keep the stack inside the axes by moving or growing them, never by compressing the
    # stack: squeezing labels back together is what made them overlap in the first place.
    deficit = (y_lo + label_h * 0.5) - min(label_ys)
    if deficit > 0:
        label_ys = [y + deficit for y in label_ys]
    needed_top = max(label_ys) + label_h * 0.5
    if needed_top > y_hi:
        y_hi = needed_top
        ax.set_ylim(y_lo, y_hi)

    bbox = dict(
        boxstyle="round,pad=0.2",
        facecolor="white",
        edgecolor=plot_style.gray_color,
        linewidth=0.4,
        alpha=0.85,
    )

    fontsize = 10 if plot_style.document_type == "paper" else 7.5

    placements = _choose_label_placements(
        ax,
        annotations=annotations,
        label_ys=label_ys,
        observed=observed,
        markers=raw_annotations,
        label_h=label_h,
        dx=dx,
        fontsize=fontsize,
        plot_style=plot_style,
    )

    for ann, (label_x_num, label_y, ha) in zip(annotations, placements):
        date_num = ann["date_num"]
        is_grouped = ann.get("grouped", False)
        displaced = abs(label_y - ann["score"]) > label_h * 0.3

        label_x = mdates.num2date(label_x_num)

        # A label pushed away from its marker by the overlap pass otherwise appears to
        # point at nothing, and the reader cannot tell which marker it names.  This
        # happens whenever two baselines share a score, as on ARC-AGI where the skilled
        # generalist and the committee of average humans both sit at 98% and draw one
        # visible star between them.  Connect displaced labels to their marker -- but
        # only individual ones: a grouped label sits at the centroid of its cluster,
        # where there is no marker for a connector to land on.
        arrowprops = (
            dict(
                arrowstyle="-",
                color=ann["color"],
                linewidth=0.6,
                alpha=0.7,
                shrinkA=1.0,
                shrinkB=3.0,
            )
            if displaced and not is_grouped
            else None
        )

        ax.annotate(
            ann["label"],
            xy=(mdates.num2date(date_num), ann["score"]),
            xytext=(label_x, label_y),
            textcoords="data",
            fontsize=fontsize,
            color=ann["color"],
            fontweight="bold",
            ha=ha,
            va="center",
            bbox=bbox,
            arrowprops=arrowprops,
            zorder=10,
            clip_on=True,
        )


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


