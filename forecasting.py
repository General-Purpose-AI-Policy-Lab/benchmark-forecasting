"""
forecasting.py

Core modeling + validation + forecasting utilities.

Principles:
- Pure functions: do not mutate caller DataFrames.
- Separate responsibilities: prepare -> build -> fit -> validate -> forecast.
- Notebook-friendly API: small number of well-named functions.

Expected dataset columns:
Required:
  - benchmark (str)
  - release_date (datetime-like)
  - score (float)
  - lower_bound (float)

Optional:
  - category (str)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import arviz as az
import pymc as pm
from scipy.stats import energy_distance

FITS_DIR = Path("Fits")


SigmoidKind = Literal["logistic", "harvey"]
ErrorMetric = Literal["RMSE", "MAE"]


@dataclass(frozen=True)
class ModelConfig:
    """Model configuration.

    The three ``L_*`` fields control the prior on the upper asymptote.  They exist
    for the prior-sensitivity analysis; the defaults reproduce the main model, and
    the slug is left unchanged in that case so cached fits stay valid.
    """
    sigmoid: SigmoidKind = "harvey"
    joint: bool = True
    top_n: int = 3
    skew: bool = True
    L_min: float = 0.75
    L_prior_mu: float = 0.96
    L_prior_sd: float = 0.02

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


def load_dataset(path: str) -> pd.DataFrame:
    """Load dataset CSV and normalize types."""
    df = pd.read_csv(path)

    if "category" in df.columns:
        df["category"] = df["category"].astype("string")

    df["benchmark"] = df["benchmark"].astype("string")
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["lower_bound"] = pd.to_numeric(df["lower_bound"], errors="coerce")
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")

    df = df.dropna(subset=["benchmark", "release_date", "score", "lower_bound"]).reset_index(drop=True)
    return df


def select_frontier_points(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Keep points that are within top_n of the expanding best-so-far per benchmark."""
    required = {"benchmark", "release_date", "score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"select_frontier_points: missing required columns: {sorted(missing)}")

    d = df.sort_values(["benchmark", "release_date"]).copy()
    d["expanding_rank"] = (
        d.groupby("benchmark")["score"]
        .expanding()
        .rank(ascending=False, method="max")
        .reset_index(level=0, drop=True)
    )
    d = d.loc[d["expanding_rank"] <= top_n].drop(columns=["expanding_rank"]).reset_index(drop=True)
    return d


def prepare_dataset(df: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
    """Prepare dataset for modeling (frontier + time features)."""
    d = select_frontier_points(df, top_n=top_n).copy()

    first_dates = d.groupby("benchmark")["release_date"].transform("min")
    d["days"] = (d["release_date"] - first_dates).dt.days.astype(int)

    max_days = d.groupby("benchmark")["days"].transform("max").astype(float)
    d["days_mid"] = max_days / 2.0
    return d


def _ensure_constant_per_benchmark(df: pd.DataFrame, col: str) -> None:
    """Raise if a column varies within any benchmark group."""
    nunique = df.groupby("benchmark")[col].nunique(dropna=False)
    bad = nunique[nunique > 1]
    if not bad.empty:
        examples = bad.index[:10].tolist()
        raise ValueError(f"Column '{col}' must be constant within benchmark. Violations (first 10): {examples}")


def build_model(prepared: pd.DataFrame, cfg: ModelConfig) -> pm.Model:
    """Build the PyMC model."""
    required = {"benchmark", "score", "lower_bound", "days", "days_mid"}
    missing = required - set(prepared.columns)
    if missing:
        raise ValueError(f"build_model: missing required columns: {sorted(missing)}")

    _ensure_constant_per_benchmark(prepared, "lower_bound")

    bench_idx, bench_names = pd.factorize(prepared["benchmark"], sort=True)
    d = prepared.assign(benchmark_idx=bench_idx).reset_index(drop=True)

    coords = {"benchmark": bench_names, "obs": np.arange(len(d))}

    joint = cfg.joint
    top_n = cfg.top_n

    with pm.Model(coords=coords) as model:
        # Upper asymptote L: scaled Beta
        L_min, L_max = cfg.L_min, 1.0
        L_range = L_max - L_min

        # A Beta(mu, sigma) exists only for sigma < sqrt(mu * (1 - mu)).  The
        # benchmark-level sigma is clamped below, but the hyperprior's own sigma is
        # part of the model specification: silently shrinking it would misreport the
        # prior in the sensitivity analysis, so an invalid combination is an error.
        _mu_raw = (cfg.L_prior_mu - L_min) / L_range
        _sd_raw = cfg.L_prior_sd / L_range
        _sd_max = np.sqrt(_mu_raw * (1 - _mu_raw))
        if not 0 < _mu_raw < 1:
            raise ValueError(
                f"L_prior_mu={cfg.L_prior_mu} must lie strictly between L_min={cfg.L_min} and 1."
            )
        if _sd_raw >= _sd_max:
            raise ValueError(
                f"L_prior_sd={cfg.L_prior_sd} is too large for L_min={cfg.L_min}: the Beta "
                f"hyperprior on L_raw_mu requires L_prior_sd < {_sd_max * L_range:.4f}."
            )

        L_raw_mu = pm.Beta(
            "L_raw_mu",
            mu=(cfg.L_prior_mu - L_min) / L_range,
            sigma=cfg.L_prior_sd / L_range,
            dims=None if joint else "benchmark",
        )
        L_raw_sigma = pm.HalfNormal(
            "L_raw_sigma",
            sigma=cfg.L_prior_sd / L_range,
            dims=None if joint else "benchmark",
        )
        # Clamp sigma so that Beta(mu, sigma) parameters stay valid: sigma < sqrt(mu*(1-mu)).
        L_raw_sigma_safe = pm.math.minimum(
            L_raw_sigma,
            pm.math.sqrt(L_raw_mu * (1 - L_raw_mu)) - 1e-4,
        )
        L_raw = pm.Beta("L_raw", mu=L_raw_mu, sigma=L_raw_sigma_safe, dims="benchmark")
        L = pm.Deterministic("L", L_min + L_range * L_raw, dims="benchmark")

        # Lower bound l per benchmark (fixed data)
        l_per_bench = d.groupby("benchmark_idx")["lower_bound"].first().to_numpy()
        l = pm.Data("l", l_per_bench, dims="benchmark")

        # Inflection point (tau) centered at observed midpoint
        days_mid = d.groupby("benchmark_idx")["days_mid"].first().to_numpy()
        tau = pm.Gumbel("tau", mu=days_mid, beta=365 * 2, dims="benchmark")

        # Indexing / covariates
        t = pm.Data("t_obs", d["days"].to_numpy(), dims="obs")
        idx = pm.Data("idx_obs", d["benchmark_idx"].to_numpy(), dims="obs")

        # Growth rate
        k_mu = pm.Gamma("k_mu", mu=0.005, sigma=0.002, dims=None if joint else "benchmark")
        k_sigma = pm.HalfNormal("k_sigma", sigma=0.005, dims=None if joint else "benchmark")
        k = pm.Gamma("k", mu=k_mu, sigma=k_sigma, dims="benchmark")

        logits = k[idx] * (t - tau[idx])

        # Sigmoid family
        if cfg.sigmoid == "logistic":
            sigmoid = pm.math.sigmoid(logits)
        elif cfg.sigmoid == "harvey":
            alpha_raw_mu = pm.Gamma(
                "alpha_raw_mu", mu=1.5, sigma=0.5, dims=None if joint else "benchmark"
            )
            alpha_raw_sigma = pm.HalfNormal(
                "alpha_raw_sigma", sigma=0.5, dims=None if joint else "benchmark"
            )
            alpha_raw = pm.Gamma("alpha_raw", mu=alpha_raw_mu, sigma=alpha_raw_sigma, dims="benchmark")
            alpha = pm.Deterministic("alpha", alpha_raw + 1.0, dims="benchmark")

            base = pm.math.maximum(1 - (1 - alpha[idx]) * pm.math.exp(-logits), 1e-10)
            sigmoid = pm.math.exp(1 / (1 - alpha[idx]) * pm.math.log(base))
        else:
            raise ValueError(f"Unsupported sigmoid: {cfg.sigmoid}")

        mu = pm.Deterministic("mu", l[idx] + (L[idx] - l[idx]) * sigmoid, dims="obs")

        # Heteroscedastic noise: increases away from bounds
        xi_base_mu = pm.Gamma(
            "xi_base_mu",
            mu=0.05 + top_n / 50,
            sigma=0.02,
            dims=None if joint else "benchmark",
        )
        xi_base_sigma = pm.HalfNormal("xi_base_sigma", sigma=0.05, dims=None if joint else "benchmark")
        # Clamp sigma so that Gamma(mu, sigma) stays valid (alpha = (mu/sigma)^2 > 0).
        xi_base_sigma_safe = pm.math.minimum(xi_base_sigma, xi_base_mu - 1e-6)
        xi_base = pm.Gamma("xi_base", mu=xi_base_mu, sigma=xi_base_sigma_safe, dims="benchmark")

        variance_shape = pm.math.sqrt(pm.math.maximum((mu - l[idx]) * (L[idx] - mu), 0.0))
        max_variance = (L[idx] - l[idx]) / 2.0
        noise_factor = variance_shape / pm.math.maximum(max_variance, 1e-10)

        xi = pm.math.maximum(0.01 + xi_base[idx] * noise_factor, 1e-6)

        if cfg.skew:
            # Skewness (negative values = scores below latent curve)
            s_mu = pm.Normal("s_mu", mu=-2 - top_n / 2, sigma=0.5, dims=None if joint else "benchmark")
            s_sigma = pm.HalfNormal("s_sigma", sigma=1.0, dims=None if joint else "benchmark")
            s = pm.TruncatedNormal("s", mu=s_mu, sigma=s_sigma, upper=0, dims="benchmark")

            pm.SkewNormal(
                "y",
                mu=mu,
                sigma=xi,
                alpha=s[idx],
                observed=d["score"].to_numpy(),
                dims="obs",
            )
        else:
            pm.Normal(
                "y",
                mu=mu,
                sigma=xi,
                observed=d["score"].to_numpy(),
                dims="obs",
            )

    return model


def fit(
    prepared: pd.DataFrame,
    cfg: ModelConfig,
    samp: SamplingConfig,
    *,
    cache_tag: str | None = None,
    use_cache: bool = True,
) -> tuple[az.InferenceData, pm.Model]:
    """Fit the model and return (idata, model).

    Parameters
    ----------
    cache_tag : optional label appended to the slug for the NetCDF filename.
        When *None*, the filename is ``Fits/{cfg.slug}.nc``.
        Pass e.g. ``cache_tag="retro_2025"`` to get ``Fits/{cfg.slug}_retro_2025.nc``.
    use_cache : if *True* (default), load from ``Fits/`` if the file exists,
        and save there after sampling.  Set to *False* to force re-fitting.
    """
    model = build_model(prepared, cfg)

    # --- cache path ---
    fname = cfg.slug if cache_tag is None else f"{cfg.slug}_{cache_tag}"
    cache_path = FITS_DIR / f"{fname}.nc"

    if use_cache and cache_path.exists():
        print(f"  Loading cached fit: {cache_path}")
        idata = az.from_netcdf(str(cache_path))
        return idata, model

    with model:
        idata = pm.sample(
            draws=samp.draws,
            tune=samp.tune,
            return_inferencedata=True,
            random_seed=samp.seed,
            target_accept=samp.target_accept,
            init=samp.init,
            progressbar=samp.progressbar,
            idata_kwargs={"log_likelihood": True},
        )

    if use_cache:
        FITS_DIR.mkdir(exist_ok=True)
        idata.to_netcdf(str(cache_path))
        print(f"  Saved fit: {cache_path}")

    return idata, model


def temporal_holdout(
    raw: pd.DataFrame,
    *,
    cutoff_date: pd.Timestamp,
    cfg: ModelConfig,
    samp: SamplingConfig,
    min_train_points: int = 5,
    use_cache: bool = True,
) -> az.InferenceData:
    """Train on data before cutoff_date, evaluate on data >= cutoff_date."""
    prepared = prepare_dataset(raw, top_n=cfg.top_n)

    train = prepared.loc[prepared["release_date"] < cutoff_date].copy()
    train_counts = train.groupby("benchmark")["score"].size()
    keep = train_counts[train_counts >= min_train_points].index
    train = train.loc[train["benchmark"].isin(keep)].copy()

    test = prepared.loc[prepared["release_date"] >= cutoff_date].copy()
    test = test.loc[test["benchmark"].isin(train["benchmark"].unique())].copy()

    # The training set depends on min_train_points, so it belongs in the cache key.
    # The suffix is omitted at 3 for historical reasons: the fits saved under the plain
    # name predate the move to a threshold of 5, and renaming them would silently pair
    # a k=3 posterior with a k=5 label.
    cutoff_tag = cutoff_date.strftime("%Y%m%d")
    if min_train_points != 3:
        cutoff_tag = f"{cutoff_tag}_min{min_train_points}"
    idata, model = fit(train, cfg, samp, cache_tag=f"retro_{cutoff_tag}", use_cache=use_cache)

    bench_codes = pd.Categorical(
        test["benchmark"],
        categories=model.coords["benchmark"],
        ordered=True,
    ).codes
    valid = bench_codes >= 0
    test = test.loc[valid].reset_index(drop=True)
    bench_codes = bench_codes[valid]

    with model:
        pm.set_data(
            {"t_obs": test["days"].to_numpy(), "idx_obs": bench_codes},
            coords={"obs": np.arange(len(test))},
        )
        idata = pm.sample_posterior_predictive(
            idata,
            predictions=True,
            extend_inferencedata=True,
            random_seed=samp.seed,
            progressbar=samp.progressbar,
        )

    idata.predictions["y_true"] = (("obs",), test["score"].to_numpy())
    # Kept so that downstream calibration checks can group observations by benchmark
    # instead of reconstructing the mapping from row order.
    idata.predictions["benchmark_label"] = (("obs",), test["benchmark"].astype(str).to_numpy())
    return idata


def crps_score(idata: az.InferenceData) -> float:
    """Mean CRPS over the test set (energy distance based)."""
    y_pred = idata.predictions.stack(sample=("chain", "draw"))["y"].to_numpy()
    y_true = idata.predictions["y_true"].to_numpy()

    scores: list[float] = []
    for pred, true in zip(y_pred, y_true):
        scores.append(energy_distance(pred, (true,)) ** 2 / 2)
    return float(np.mean(scores))


def point_error(idata: az.InferenceData, metric: ErrorMetric = "RMSE") -> float:
    """RMSE or MAE between posterior predictive mean and the truth."""
    y_pred = idata.predictions.stack(sample=("chain", "draw"))["y"].to_numpy()
    y_true = idata.predictions["y_true"].to_numpy()

    means = np.mean(y_pred, axis=1)
    if metric == "MAE":
        return float(np.mean(np.abs(means - y_true)))
    if metric == "RMSE":
        return float(np.sqrt(np.mean((means - y_true) ** 2)))
    raise ValueError(f"Unsupported metric: {metric}")


def conformal_prediction_coverage_grouped(
    idata: az.InferenceData,
    *,
    alpha: float = 0.20,
    n_repeats: int = 100,
    seed: int = 0,
    min_calibration: int = 5,
) -> dict[str, object]:
    """CQR with calibration/test split by benchmark, repeated over random assignments.

    ``conformal_prediction_coverage`` splits the held-out observations by position.
    Because the holdout frame is ordered by benchmark, that puts one set of benchmarks
    in the calibration half and a disjoint set in the test half, which breaks the
    exchangeability CQR relies on and makes the resulting Q a transfer measurement
    with no finite-sample guarantee.

    Here whole benchmarks are assigned at random to the two halves, so exchangeability
    is required at the benchmark level -- the same assumption the hierarchical model
    already makes -- and the quantity answered is: calibrating on one set of
    benchmarks, do the intervals cover on benchmarks held out from calibration?
    Splits are repeated so the answer does not depend on one arbitrary assignment.

    Returns the median and interquartile range of Q, of the CQR coverage and of the
    raw Bayesian coverage over repeats, plus the Bayesian coverage on all held-out
    observations (which is what a coverage column should report).
    """
    y_pred = idata.predictions.stack(sample=("chain", "draw"))["y"].to_numpy()
    y_true = idata.predictions["y_true"].to_numpy()
    if "benchmark_label" not in idata.predictions:
        raise ValueError(
            "Requires 'benchmark_label' in idata.predictions; refit with temporal_holdout()."
        )
    groups = np.asarray(idata.predictions["benchmark_label"].to_numpy(), dtype=str)

    lower = np.quantile(y_pred, alpha / 2, axis=1)
    upper = np.quantile(y_pred, 1 - alpha / 2, axis=1)
    scores = np.maximum(lower - y_true, y_true - upper)

    coverage_all = float(np.mean(scores < 0))

    unique = np.unique(groups)
    if len(unique) < 4:
        raise ValueError(f"Need at least 4 benchmarks to split by group, got {len(unique)}.")

    rng = np.random.default_rng(seed)
    q_vals, cqr_cov, bayes_cov, n_cals = [], [], [], []
    for _ in range(n_repeats):
        perm = rng.permutation(unique)
        cal_groups = perm[: len(unique) // 2]
        in_cal = np.isin(groups, cal_groups)
        if in_cal.sum() < min_calibration or (~in_cal).sum() < min_calibration:
            continue
        n_cal = int(in_cal.sum())
        q_level = min(np.ceil((1 - alpha) * (n_cal + 1)) / n_cal, 1.0)
        Q = float(np.quantile(scores[in_cal], q_level))
        # y_i lies in [lower - Q, upper + Q]  <=>  max(lower - y_i, y_i - upper) <= Q,
        # so the test is against +Q: a negative Q tightens the interval and can only
        # lower the coverage relative to the unadjusted one.
        cqr_cov.append(float(np.mean(scores[~in_cal] <= Q)))
        bayes_cov.append(float(np.mean(scores[~in_cal] < 0)))
        q_vals.append(Q)
        n_cals.append(n_cal)

    if not q_vals:
        raise ValueError("No usable split: too few observations per half.")

    def _summary(v: list[float]) -> dict[str, float]:
        a = np.asarray(v)
        return {
            "median": float(np.median(a)),
            "q25": float(np.percentile(a, 25)),
            "q75": float(np.percentile(a, 75)),
        }

    return {
        "bayesian_coverage_all": coverage_all,
        "n_obs": int(len(y_true)),
        "n_benchmarks": int(len(unique)),
        "n_repeats_used": len(q_vals),
        "median_n_calibration": float(np.median(n_cals)),
        "cqr_Q": _summary(q_vals),
        "cqr_coverage": _summary(cqr_cov),
        "bayesian_coverage_test": _summary(bayes_cov),
    }


def conformal_prediction_coverage(idata: az.InferenceData, *, alpha: float = 0.20) -> dict[str, float]:
    """Conformal Quantile Regression (CQR) coverage on holdout data.

    Uses CQR (Romano et al., 2019): adjusts Bayesian credible intervals with a
    distribution-free conformal correction Q, so the final interval is
    [bayesian_lower - Q, bayesian_upper + Q].

    If the Bayesian intervals are already well-calibrated, Q ≈ 0.

    Parameters
    ----------
    idata : InferenceData with predictions group containing 'y' and 'y_true'.
    alpha : Miscoverage rate (e.g. 0.20 for 80% nominal coverage).

    Returns
    -------
    dict with keys:
        'bayesian_coverage': empirical coverage of raw Bayesian CI on test set
        'cqr_coverage': empirical coverage of CQR-adjusted intervals on test set
        'cqr_Q': conformal adjustment (Q); ~0 means Bayesian CI is well-calibrated
        'cqr_avg_width': average width of CQR intervals on test set
        'bayesian_avg_width': average width of raw Bayesian CI on test set
        'n_calibration': size of calibration set
        'n_test': size of test set
    """
    y_pred_samples = idata.predictions.stack(sample=("chain", "draw"))["y"].to_numpy()  # (n_obs, n_samples)
    y_true = idata.predictions["y_true"].to_numpy()  # (n_obs,)

    n_obs = len(y_true)

    # Bayesian credible interval bounds
    lower_q = alpha / 2
    upper_q = 1 - alpha / 2
    bayesian_lower = np.quantile(y_pred_samples, lower_q, axis=1)  # (n_obs,)
    bayesian_upper = np.quantile(y_pred_samples, upper_q, axis=1)  # (n_obs,)

    # Split: first half calibration, second half test
    n_cal = n_obs // 2
    if n_cal < 5:
        raise ValueError(f"Too few observations for conformal calibration: {n_obs}")

    # --- CQR non-conformity scores on calibration set ---
    # E_i = max(q_lower_i - y_i, y_i - q_upper_i)
    # Negative when y_i is inside the Bayesian interval
    scores_cal = np.maximum(
        bayesian_lower[:n_cal] - y_true[:n_cal],
        y_true[:n_cal] - bayesian_upper[:n_cal],
    )

    # Conformal quantile with finite-sample correction
    q_level = np.ceil((1 - alpha) * (n_cal + 1)) / n_cal
    q_level = min(q_level, 1.0)
    Q = float(np.quantile(scores_cal, q_level))

    # --- CQR-adjusted intervals on test set ---
    test_lower = bayesian_lower[n_cal:] - Q
    test_upper = bayesian_upper[n_cal:] + Q
    cqr_covered = (y_true[n_cal:] >= test_lower) & (y_true[n_cal:] <= test_upper)
    cqr_coverage = float(np.mean(cqr_covered))
    cqr_avg_width = float(np.mean(test_upper - test_lower))

    # --- Raw Bayesian coverage on test set for comparison ---
    bayesian_covered = (y_true[n_cal:] >= bayesian_lower[n_cal:]) & (y_true[n_cal:] <= bayesian_upper[n_cal:])
    bayesian_coverage = float(np.mean(bayesian_covered))
    bayesian_avg_width = float(np.mean(bayesian_upper[n_cal:] - bayesian_lower[n_cal:]))

    return {
        "bayesian_coverage": bayesian_coverage,
        "cqr_coverage": cqr_coverage,
        "cqr_Q": Q,
        "cqr_avg_width": cqr_avg_width,
        "bayesian_avg_width": bayesian_avg_width,
        "n_calibration": n_cal,
        "n_test": n_obs - n_cal,
    }


def saturation_dates(
    idata: az.InferenceData,
    *,
    prepared_frontier: pd.DataFrame,
    saturation_fraction: float = 0.95,
    ci_level: float = 0.80,
    target_date: pd.Timestamp | str = "2030-01-01",
    max_date: pd.Timestamp | str = "2100-01-01",
) -> pd.DataFrame:
    """Per-benchmark posterior distribution of the saturation date.

    Saturation is defined exactly as in ``plot_saturation_proportion_posterior``:
    the normalized curve exceeds ``saturation_fraction``.  Inverting the sigmoid
    analytically gives the crossing time, which is more informative than the
    aggregate proportion when comparing model variants.

    For the Harvey curve, :math:`\\sigma(t) = f` is reached at
    :math:`z^* = -\\log\\left[(1 - f^{1-\\alpha}) / (1 - \\alpha)\\right]`;
    for the logistic, at :math:`z^* = \\log[f / (1-f)]`.  In both cases
    :math:`t^* = \\tau + z^* / k`.

    Returns one row per benchmark with the posterior median and CI of the
    saturation date, plus the posterior probability of saturating by
    ``target_date``.
    """
    if not (0.0 < saturation_fraction < 1.0):
        raise ValueError("saturation_fraction must be in (0, 1)")

    posterior = idata.posterior
    if "k" not in posterior or "tau" not in posterior:
        raise ValueError("Requires idata.posterior['k'] and idata.posterior['tau'].")

    benchmarks = [str(b) for b in posterior["k"].coords["benchmark"].to_numpy().tolist()]

    pf = prepared_frontier.copy()
    pf["release_date"] = pd.to_datetime(pf["release_date"], errors="coerce")
    starts = pf.groupby("benchmark")["release_date"].min().reindex(benchmarks)
    if starts.isna().any():
        missing = starts[starts.isna()].index.tolist()
        raise ValueError(f"prepared_frontier is missing benchmarks: {missing[:10]}")

    def _stack(name: str) -> np.ndarray:
        return (
            posterior[name]
            .stack(sample=("chain", "draw"))
            .transpose("benchmark", "sample")
            .to_numpy()
        )

    k = _stack("k")
    tau = _stack("tau")

    f = float(saturation_fraction)
    if "alpha" in posterior:
        alpha = _stack("alpha")
        # (1 - f^(1-alpha)) / (1 - alpha) > 0 for alpha > 1, f in (0, 1)
        ratio = (1.0 - np.power(f, 1.0 - alpha)) / (1.0 - alpha)
        z_star = -np.log(np.maximum(ratio, 1e-300))
    else:
        z_star = np.full_like(k, np.log(f / (1.0 - f)))

    with np.errstate(divide="ignore", invalid="ignore"):
        t_star = tau + z_star / np.where(k > 0, k, np.nan)

    start_days = starts.to_numpy().astype("datetime64[D]").astype(float)  # days since epoch
    date_days = start_days[:, None] + t_star

    max_days = float(pd.Timestamp(max_date).to_datetime64().astype("datetime64[D]").astype(int))
    min_days = float(pd.Timestamp("1990-01-01").to_datetime64().astype("datetime64[D]").astype(int))
    date_days = np.clip(np.nan_to_num(date_days, nan=max_days), min_days, max_days)

    target_days = float(pd.Timestamp(target_date).to_datetime64().astype("datetime64[D]").astype(int))
    lo_q = 100.0 * (1.0 - ci_level) / 2.0
    hi_q = 100.0 * (1.0 + ci_level) / 2.0

    def _to_date(x: np.ndarray) -> np.ndarray:
        return np.round(x).astype("int64").astype("datetime64[D]")

    out = pd.DataFrame(
        {
            "benchmark": benchmarks,
            "sat_median": _to_date(np.median(date_days, axis=1)),
            "sat_lower": _to_date(np.percentile(date_days, lo_q, axis=1)),
            "sat_upper": _to_date(np.percentile(date_days, hi_q, axis=1)),
            "sat_median_days": np.median(date_days, axis=1),
            "p_saturated_by_target": (date_days <= target_days).mean(axis=1),
        }
    )
    if "category" in pf.columns:
        cats = pf.groupby("benchmark")["category"].first().reindex(benchmarks)
        out.insert(1, "category", cats.to_numpy())
    return out


def saturated_proportion(
    idata: az.InferenceData,
    *,
    prepared_frontier: pd.DataFrame,
    target_date: pd.Timestamp | str = "2030-01-01",
    saturation_fraction: float = 0.95,
    ci_level: float = 0.80,
    benchmarks: list[str] | None = None,
) -> dict[str, object]:
    """Posterior of the proportion of benchmarks saturated by ``target_date``.

    Same quantity as ``plotting.plot_saturation_proportion_posterior`` but without
    the figure, and restrictable to a subset of benchmarks — useful to check how
    much the headline number depends on a given group of benchmarks.
    """
    posterior = idata.posterior
    all_benchmarks = [str(b) for b in posterior["k"].coords["benchmark"].to_numpy().tolist()]
    if benchmarks is None:
        keep_idx = np.arange(len(all_benchmarks))
        kept = all_benchmarks
    else:
        wanted = set(benchmarks)
        keep_idx = np.array([i for i, b in enumerate(all_benchmarks) if b in wanted])
        kept = [all_benchmarks[i] for i in keep_idx]
        if len(kept) == 0:
            raise ValueError("None of the requested benchmarks are present in the posterior.")

    pf = prepared_frontier.copy()
    pf["release_date"] = pd.to_datetime(pf["release_date"], errors="coerce")
    starts = pf.groupby("benchmark")["release_date"].min().reindex(kept)
    t_target = (pd.to_datetime(target_date) - starts).dt.days.to_numpy(dtype=float)

    def _stack(name: str) -> np.ndarray:
        arr = (
            posterior[name]
            .stack(sample=("chain", "draw"))
            .transpose("benchmark", "sample")
            .to_numpy()
        )
        return arr[keep_idx]

    k = _stack("k")
    tau = _stack("tau")
    z = k * (t_target[:, None] - tau)

    if "alpha" in posterior:
        alpha = _stack("alpha")
        base = np.maximum(1.0 - (1.0 - alpha) * np.exp(-z), 1e-12)
        sigmoid = np.power(base, 1.0 / (1.0 - alpha))
    else:
        sigmoid = 1.0 / (1.0 + np.exp(-z))

    proportions = (sigmoid > float(saturation_fraction)).mean(axis=0).astype(float)
    lo_q = 100.0 * (1.0 - ci_level) / 2.0
    hi_q = 100.0 * (1.0 + ci_level) / 2.0
    return {
        "n_benchmarks": len(kept),
        "benchmarks": kept,
        "median": float(np.median(proportions)),
        "mean": float(np.mean(proportions)),
        "ci": (float(np.percentile(proportions, lo_q)), float(np.percentile(proportions, hi_q))),
        "ci_level": float(ci_level),
    }


def residual_diagnostics(
    idata: az.InferenceData,
    prepared: pd.DataFrame,
    *,
    lineages: dict[str, str] | None = None,
    min_shared_models: int = 5,
    min_models_per_group: int = 3,
) -> dict[str, object]:
    """Cross-benchmark dependence check on the fitted residuals.

    The hierarchical model assumes benchmark trajectories are *conditionally*
    independent given the shared hyperpriors.  If benchmarks that share a source
    dataset or a creator also share unmodeled fluctuations, residuals from the
    same model release should correlate across those benchmarks.

    Residuals are paired by ``model_version``: for each frontier observation,
    ``y - E[mu]``.  We then compute pairwise Pearson correlations between
    benchmarks over the models they have in common, and compare pairs within a
    lineage against pairs across lineages.

    ``lineages`` maps benchmark name -> lineage label; benchmarks absent from the
    mapping are each treated as their own lineage.
    """
    if "mu" not in idata.posterior:
        raise ValueError("Requires the deterministic 'mu' in idata.posterior.")
    if "model_version" not in prepared.columns:
        raise ValueError("residual_diagnostics expects a 'model_version' column.")

    # build_model consumes the rows of `prepared` in order, so the 'obs' dimension of
    # the posterior aligns with them directly.
    d = prepared.reset_index(drop=True).copy()

    mu_mean = idata.posterior["mu"].mean(dim=("chain", "draw")).to_numpy()
    if len(mu_mean) != len(d):
        raise ValueError(
            f"Posterior 'mu' has {len(mu_mean)} observations but `prepared` has {len(d)}. "
            "Pass the same frame that was used for fitting."
        )
    d["resid"] = d["score"].to_numpy() - mu_mean

    # Model x benchmark matrix of residuals (mean if a model appears twice)
    mat = d.pivot_table(index="model_version", columns="benchmark", values="resid", aggfunc="mean")

    lin = lineages or {}
    benchmarks = list(mat.columns)

    rows: list[dict[str, object]] = []
    for i, b1 in enumerate(benchmarks):
        for b2 in benchmarks[i + 1 :]:
            pair = mat[[b1, b2]].dropna()
            if len(pair) < min_shared_models:
                continue
            if pair[b1].std() == 0 or pair[b2].std() == 0:
                continue
            rows.append(
                {
                    "benchmark_1": b1,
                    "benchmark_2": b2,
                    "n_shared_models": len(pair),
                    "corr": float(pair[b1].corr(pair[b2])),
                    "same_lineage": bool(
                        b1 in lin and b2 in lin and lin[b1] == lin[b2]
                    ),
                    "lineage": lin.get(b1) if lin.get(b1) == lin.get(b2) else None,
                }
            )
    pairs = pd.DataFrame(rows)

    # Variance of residuals explained by model identity (one-way ANOVA R^2):
    # a large value would indicate a systematic "good/bad release" effect shared
    # across benchmarks, i.e. dependence the model does not represent.
    # Restricted to models evaluated on several benchmarks, since singleton groups
    # have zero within-group variance and would inflate R^2 mechanically.
    sizes = d.groupby("model_version")["resid"].transform("size")
    dm = d.loc[sizes >= min_models_per_group]
    if len(dm) > 0:
        grand = dm["resid"].mean()
        ss_total = float(((dm["resid"] - grand) ** 2).sum())
        group_means = dm.groupby("model_version")["resid"].transform("mean")
        ss_between = float(((group_means - grand) ** 2).sum())
        r2_model = ss_between / ss_total if ss_total > 0 else float("nan")
        # Bias-corrected: adjusted R^2 for a one-way layout with g groups, n rows.
        g = int(dm["model_version"].nunique())
        n = int(len(dm))
        r2_model_adj = (
            1.0 - (1.0 - r2_model) * (n - 1) / (n - g) if n > g else float("nan")
        )
    else:
        r2_model = r2_model_adj = float("nan")
        g = n = 0

    within = pairs.loc[pairs["same_lineage"], "corr"] if len(pairs) else pd.Series(dtype=float)
    across = pairs.loc[~pairs["same_lineage"], "corr"] if len(pairs) else pd.Series(dtype=float)

    return {
        "pairs": pairs,
        "residuals": d[["benchmark", "model_version", "release_date", "score", "resid"]],
        "n_pairs": int(len(pairs)),
        "n_pairs_same_lineage": int(within.size),
        "mean_corr_same_lineage": float(within.mean()) if within.size else float("nan"),
        "mean_corr_across_lineage": float(across.mean()) if across.size else float("nan"),
        "median_corr_same_lineage": float(within.median()) if within.size else float("nan"),
        "median_corr_across_lineage": float(across.median()) if across.size else float("nan"),
        "r2_model_identity": r2_model,
        "r2_model_identity_adjusted": r2_model_adj,
        "n_models_r2": g,
        "n_obs_r2": n,
        "n_models": int(d["model_version"].nunique()),
    }


def _date_grid_for_benchmark(group: pd.DataFrame, *, end_date: pd.Timestamp, n_points: int) -> pd.DataFrame:
    start_date = group["release_date"].min()
    date_range = pd.date_range(start=start_date, end=end_date, periods=n_points)

    out = pd.DataFrame(
        {
            "release_date": date_range,
            "days": (date_range - start_date).days,
            "benchmark": group["benchmark"].iloc[0],
        }
    )
    if "category" in group.columns:
        out["category"] = group["category"].iloc[0]
    return out


def generate_forecast(
    idata: az.InferenceData,
    model: pm.Model,
    *,
    prepared_frontier: pd.DataFrame,
    end_date: pd.Timestamp,
    n_points: int = 250,
    ci_level: float = 0.8,
) -> pd.DataFrame:
    """Generate a batched forecast grid for all benchmarks."""
    if "days" not in prepared_frontier.columns:
        raise ValueError("generate_forecast expects prepared_frontier from prepare_dataset() (missing 'days').")

    grid = (
        prepared_frontier.groupby("benchmark", group_keys=False)
        .apply(lambda g: _date_grid_for_benchmark(g, end_date=end_date, n_points=n_points))
        .reset_index(drop=True)
    )

    bench_codes = pd.Categorical(
        grid["benchmark"],
        categories=model.coords["benchmark"],
        ordered=True,
    ).codes
    valid = bench_codes >= 0
    grid = grid.loc[valid].reset_index(drop=True)
    bench_codes = bench_codes[valid]

    with model:
        pm.set_data(
            {"t_obs": grid["days"].to_numpy(), "idx_obs": bench_codes},
            coords={"obs": np.arange(len(grid))},
        )
        ppc = pm.sample_posterior_predictive(
            idata,
            var_names=["mu"],
            predictions=True,
            random_seed=42,
            progressbar=False,
        )

    mu_samples = ppc.predictions.stack(sample=("chain", "draw"))["mu"].to_numpy()
    alpha = (1 - ci_level) / 2

    grid["mu_mean"] = np.mean(mu_samples, axis=1)
    grid["mu_lower"] = np.quantile(mu_samples, alpha, axis=1)
    grid["mu_upper"] = np.quantile(mu_samples, 1 - alpha, axis=1)

    # Benchmark ordering helper: posterior mean inflection point (tau)
    # tau is expressed in "days since benchmark start_date". We also expose the corresponding date.
    if "tau" in idata.posterior:
        tau_days = (
            idata.posterior["tau"]
            .mean(dim=("chain", "draw"))
            .to_series()
            .astype(float)
            .rename("mean_tau_days")
        )
        start_dates = prepared_frontier.groupby("benchmark")["release_date"].min()
        tau_dates = start_dates.reindex(tau_days.index) + pd.to_timedelta(tau_days, unit="D")

        tau_df = (
            pd.DataFrame({"benchmark": tau_days.index})
            .assign(mean_tau_days=tau_days.to_numpy(), mean_tau=tau_dates.to_numpy())
        )
        grid = grid.merge(tau_df, on="benchmark", how="left")


    return grid
