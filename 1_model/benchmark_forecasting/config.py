"""Paths, environment, the two configuration dataclasses and the run settings shared by every step.

The two percent scripts (`2_analyses/`) read the model grid, the sampling settings and the data
cutoff from here, so the two cannot drift apart.

Importing this module first pins BLAS to one thread: with Apple Accelerate, four PyMC chain
processes each spawning a full thread pool oversubscribe the cores and slow NUTS down ~50x
(4 h instead of 3 min for the main fit, measured 2026-09-03).
"""

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_DIR = ROOT / "0_input"

# ── Output layout (the convention shared with Multiaxis_ECI) ───────────────
# Everything a run writes goes under 3_outputs/<data cutoff>/: the fit caches in fits/, the
# figures and result tables by theme (forecasts/, calibration/, sensitivity/, high_level/),
# French renders in a fr/ subfolder beside the English files. The write-ups (paper, note)
# live under 4_writeups/; the note's curated figures are refreshed there by the scripts.
OUTPUTS_DIR = ROOT / "3_outputs"
WRITEUPS_DIR = ROOT / "4_writeups"
NOTE_FIGURES_DIR = WRITEUPS_DIR / "note" / "figures"
FITS_SUBDIR = "fits"


def cutoff_dir(cutoff_tag: str | None) -> Path:
    """`3_outputs/cutoffYYYYMMDD/` for a run's cutoff tag (leading underscore or not),
    `3_outputs/no_cutoff/` when the run fits every score."""
    tag = (cutoff_tag or "").lstrip("_")
    return OUTPUTS_DIR / (tag or "no_cutoff")

# The two consumer views of benchmark-data-pipeline, copied by `python -m benchmark_forecasting
# sync` with the pipeline's build manifest so a run can be traced to the exact database it used.
SCORES_FILE = "dated_scores_flat.csv"
BASELINES_FILE = "human_baselines.csv"
MANIFEST_FILE = "build_manifest.json"
PROVENANCE_FILE = "provenance.json"
PIPELINE_DIR = ROOT.parent / "benchmark-data-pipeline"

# The pipeline decides which benchmarks are worth fitting (`0_input/metadata/benchmarks.csv` there);
# this hook exists for exclusions specific to the forecasting model and is empty on purpose.
EXCLUDED_BENCHMARKS: tuple[str, ...] = ()

SigmoidKind = Literal["logistic", "harvey"]
ErrorMetric = Literal["RMSE", "MAE"]


@dataclass(frozen=True)
class ModelConfig:
    """Model configuration.

    The ``L_*`` fields control the prior on the upper asymptote L of each benchmark, a Beta
    distribution rescaled to [L_min, 1] with a shared hyperprior on its mean. Two data-driven
    constraints refine it per benchmark (see ``data.asymptote_bounds``):

    - ``L_floor_observed``: L is at least the best performance observed on the benchmark, human
      baseline or model score in the fitted data (a frontier cannot plateau below what has been
      reached). A benchmark where a human or a model scored 1.0 is thereby pinned at 1.
    - ``L_fixed_from_ceiling``: a benchmark with a known ceiling in the pipeline's metadata gets
      L pinned at that ceiling instead of estimated.

    ``L_fixed`` pins named benchmarks by hand and wins over both. The slug names the cached fit;
    the defaults reproduce the main model.
    """

    sigmoid: SigmoidKind = "harvey"
    joint: bool = True
    top_n: int = 3
    skew: bool = True
    L_min: float = 0.75
    L_prior_mu: float = 0.96
    L_prior_sd: float = 0.02
    L_floor_observed: bool = True
    # How the floors enter the prior.  False (default): the shared Beta is simply cut off below
    # the floor, so the population parameters keep describing the asymptotes actually estimated.
    # True: proper truncated Beta, renormalised per benchmark by 1 - CDF(floor); the population
    # then describes a latent untruncated distribution and is pulled down by the floored
    # benchmarks (population mean 0.83 instead of 0.95 on the September 2026 data).
    L_floor_renormalised: bool = False
    L_fixed_from_ceiling: bool = True
    # Keep the Beta of L at b >= 1 by bounding its sd (model.L_raw_sigma_b1): below 1 the
    # density spikes at L = 1, and the hierarchy put 17 % of its prior there (P(L > 0.999) =
    # 5 %), 26 % of the joint posterior too. A benchmark whose asymptote is known to be 1 is
    # pinned instead (L_fixed_from_ceiling, or a score of 1). User decision, 2026-10-02.
    L_beta_b_min1: bool = True
    # A tuple of pairs rather than a dict so the frozen dataclass stays hashable. Benchmarks
    # absent from the fitted data (e.g. in a retrodiction subset) are ignored.
    L_fixed: tuple[tuple[str, float], ...] = ()
    # Independent model only: integrate each benchmark's own hyperpriors out (marginal.py)
    # instead of sampling them. Same model for every other quantity; the hyperparameters have
    # one member each and stay at their prior, and their geometry kept NUTS at the depth limit
    # (r-hat up to 1.75, bulk ESS 7 on the 2026-10-01 cutoff). On by default since 2026-10-02;
    # the slug carries `marg` so a cache of the sampled hierarchy is never reloaded as this.
    hyper_marginalised: bool = True

    @property
    def slug(self) -> str:
        """Short identifier for file naming."""
        parts = [self.sigmoid]
        parts.append("joint" if self.joint else "independent")
        parts.append("skew" if self.skew else "normal")
        if self.top_n != 3:
            # top_n shapes the priors on the growth rate and the skew (model.py), not only the
            # frontier: a non-default value must not reuse the default's cache.
            parts.append(f"top{self.top_n}")
        if self.L_min != 0.75:
            parts.append(f"Lmin{round(self.L_min * 100)}")
        if self.L_prior_mu != 0.96:
            parts.append(f"Lmu{round(self.L_prior_mu * 100)}")
        if self.L_prior_sd != 0.02:
            parts.append(f"Lsd{round(self.L_prior_sd * 1000)}")
        if self.L_floor_observed:
            parts.append("Lobs")
        if self.L_floor_renormalised:
            parts.append("Ltrunc")
        if self.L_fixed_from_ceiling:
            parts.append("Lceil")
        if self.L_beta_b_min1:
            parts.append("Lb1")
        if self.hyper_marginalised and not self.joint:
            parts.append("marg")
        if self.L_fixed:
            # Count plus a short digest of the (benchmark, value) pairs, so two different pinned
            # sets never share a cache file.
            digest = hashlib.md5(repr(sorted(self.L_fixed)).encode()).hexdigest()[:6]
            parts.append(f"Lfix{len(self.L_fixed)}-{digest}")
        return "_".join(parts)


@dataclass(frozen=True)
class SamplingConfig:
    """MCMC sampling configuration."""

    draws: int = 2000
    tune: int = 1000
    target_accept: float = 0.9
    seed: int = 42
    # nutpie (compiled, numba) is the default since 2026-10-01: the 16 fits of a run took 3.0 h
    # with PyMC's Python NUTS, the independent variants 640 to 2,200 s each. "pymc" restores it.
    sampler: str = "nutpie"
    init: str = "adapt_diag"      # PyMC's NUTS only; nutpie adapts its own mass matrix
    progressbar: bool = True


# ── Run settings shared by the two analysis scripts ─────────────────────────
# The data cutoff: only scores released on or before this date are fitted (inclusive). Its tag
# names the run's folder and files, so it must date the data: it is the day the synced pipeline
# run happened (`pipeline_run_date` in 0_input/provenance.json), which `checked_data_cutoff`
# enforces for the main runs. Change it when syncing a newer run and expect every fit to rerun.
DATA_CUTOFF = "2026-10-01"


def pipeline_run_date() -> str:
    """The date the synced pipeline run happened, YYYYMMDD (its `2_database/<date>/` folder), ''
    before the first sync."""
    import json
    p = INPUT_DIR / PROVENANCE_FILE
    return json.loads(p.read_text()).get("pipeline_run_date", "") if p.exists() else ""


def checked_data_cutoff(cutoff: str = DATA_CUTOFF) -> str:
    """`cutoff` once checked against the synced pipeline run; ValueError when they differ.

    The main runs (2_analyses/forecasts.py, revision_analyses.py) name their outputs after the
    cutoff, so a cutoff left behind after a sync would date a newer dataset with an older day.
    Retrospective cutoffs (`fit.retrodiction_fit`, `bounds --cutoff`) are not checked.
    """
    import pandas as pd
    run = pipeline_run_date()
    if pd.Timestamp(cutoff).strftime("%Y%m%d") != run:
        expected = pd.Timestamp(run).strftime("%Y-%m-%d") if run else "?"
        raise ValueError(
            f"DATA_CUTOFF {cutoff} does not match the synced pipeline run ({run or 'none'}, "
            f"0_input/{PROVENANCE_FILE}): set config.DATA_CUTOFF = \"{expected}\" or resync "
            "with `python -m benchmark_forecasting sync`")
    return cutoff


def cutoff_tag(cutoff) -> str:
    """`cutoffYYYYMMDD` for a date (string or Timestamp), '' for None. Names the output folder
    (`cutoff_dir`) and suffixes the fit caches and figure files."""
    if cutoff is None:
        return ""
    import pandas as pd
    return f"cutoff{pd.Timestamp(cutoff):%Y%m%d}"


MAIN_MODEL = "Harvey Joint (skew)"
# The eight variants of the sensitivity grid: sigmoid × structure × likelihood, top-3 frontier.
ALL_MODEL_CONFIGS: dict[str, ModelConfig] = {
    "Harvey Joint (skew)": ModelConfig(sigmoid="harvey", joint=True, top_n=3, skew=True),
    "Harvey Joint (normal)": ModelConfig(sigmoid="harvey", joint=True, top_n=3, skew=False),
    "Harvey Independent (skew)": ModelConfig(sigmoid="harvey", joint=False, top_n=3, skew=True),
    "Harvey Independent (normal)": ModelConfig(sigmoid="harvey", joint=False, top_n=3,
                                               skew=False),
    "Logistic Joint (skew)": ModelConfig(sigmoid="logistic", joint=True, top_n=3, skew=True),
    "Logistic Joint (normal)": ModelConfig(sigmoid="logistic", joint=True, top_n=3, skew=False),
    "Logistic Independent (skew)": ModelConfig(sigmoid="logistic", joint=False, top_n=3,
                                               skew=True),
    "Logistic Independent (normal)": ModelConfig(sigmoid="logistic", joint=False, top_n=3,
                                                 skew=False),
}

SAMPLING_CONFIG = SamplingConfig(draws=2000, tune=1000, target_accept=0.9, seed=42,
                                 progressbar=True)
# The independent variants give each benchmark its own asymptote prior. Its marginal (the
# hyperpriors integrated out) keeps some mass right against 1 for a benchmark whose data do not
# bound the asymptote from above, and NUTS diverges there on 4 to 15 % of draws. They are
# sampled at 0.95 (`_ta95` in the cache name). 0.99 is worse: on 2026-10-02 nutpie's adaptation
# drove one Harvey chain's step size to zero and the trees to the depth limit (r-hat 1.58).
SAMPLING_CONFIG_INDEPENDENT = SamplingConfig(draws=2000, tune=1000, target_accept=0.95, seed=42,
                                             progressbar=True)


def sampling_for(cfg: ModelConfig) -> SamplingConfig:
    """The sampling configuration matching a model variant."""
    return SAMPLING_CONFIG if cfg.joint else SAMPLING_CONFIG_INDEPENDENT


# Minimum pre-cutoff frontier observations for a benchmark to enter a retrodiction.
MIN_TRAIN_POINTS = 5


def variant_slug(name: str) -> str:
    """File-name slug of a variant name: 'Harvey Joint (skew)' -> 'harvey_joint_skew'."""
    return name.lower().replace(" (", "_").replace(")", "").replace(" ", "_")
