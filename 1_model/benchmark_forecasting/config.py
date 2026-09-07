"""Paths, environment and the two configuration dataclasses shared by every step.

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
FITS_DIR = ROOT / "Fits"
PLOTS_DIR = ROOT / "Plots"

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

    - ``L_floor_from_baselines``: L is at least the highest human baseline recorded for the
      benchmark (the population Beta is cut off below that value, see ``L_floor_renormalised``).
      A benchmark whose best human baseline is 1.0 is thereby pinned at 1.
    - ``L_floor_from_scores``: L is at least the best score observed on the benchmark in the fitted
      data (the frontier cannot plateau below what has been reached). A benchmark where a score of
      1.0 has been observed is thereby pinned at 1.
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
    L_floor_from_baselines: bool = True
    L_floor_from_scores: bool = True
    # How the floors enter the prior.  False (default): the shared Beta is simply cut off below
    # the floor, so the population parameters keep describing the asymptotes actually estimated.
    # True: proper truncated Beta, renormalised per benchmark by 1 - CDF(floor); the population
    # then describes a latent untruncated distribution and is pulled down by the floored
    # benchmarks (population mean 0.83 instead of 0.95 on the September 2026 data).
    L_floor_renormalised: bool = False
    L_fixed_from_ceiling: bool = True
    # A tuple of pairs rather than a dict so the frozen dataclass stays hashable. Benchmarks
    # absent from the fitted data (e.g. in a retrodiction subset) are ignored.
    L_fixed: tuple[tuple[str, float], ...] = ()

    @property
    def slug(self) -> str:
        """Short identifier for file naming."""
        parts = [self.sigmoid]
        parts.append("joint" if self.joint else "independent")
        parts.append("skew" if self.skew else "normal")
        if self.L_min != 0.75:
            parts.append(f"Lmin{round(self.L_min * 100)}")
        if self.L_prior_mu != 0.96:
            parts.append(f"Lmu{round(self.L_prior_mu * 100)}")
        if self.L_prior_sd != 0.02:
            parts.append(f"Lsd{round(self.L_prior_sd * 1000)}")
        if self.L_floor_from_baselines:
            parts.append("Lhuman")
        if self.L_floor_from_scores:
            parts.append("Lbest")
        if self.L_floor_renormalised:
            parts.append("Ltrunc")
        if self.L_fixed_from_ceiling:
            parts.append("Lceil")
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
    init: str = "adapt_diag"
    progressbar: bool = True
