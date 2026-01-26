"""
forecasting_clean.py

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

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import arviz as az
import pymc as pm
from scipy.stats import energy_distance


SigmoidKind = Literal["logistic", "harvey"]
ErrorMetric = Literal["RMSE", "MAE"]


@dataclass(frozen=True)
class ModelConfig:
    """Model configuration."""
    sigmoid: SigmoidKind = "harvey"
    joint: bool = True
    top_n: int = 3


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
        L_min, L_max = 0.75, 1.0
        L_range = L_max - L_min

        L_raw_mu = pm.Beta(
            "L_raw_mu",
            mu=(0.96 - L_min) / L_range,
            sigma=0.02 / L_range,
            dims=None if joint else "benchmark",
        )
        L_raw_sigma = pm.HalfNormal(
            "L_raw_sigma",
            sigma=0.02 / L_range,
            dims=None if joint else "benchmark",
        )
        L_raw = pm.Beta("L_raw", mu=L_raw_mu, sigma=L_raw_sigma, dims="benchmark")
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
        xi_base = pm.Gamma("xi_base", mu=xi_base_mu, sigma=xi_base_sigma, dims="benchmark")

        variance_shape = pm.math.sqrt((mu - l[idx]) * (L[idx] - mu))
        max_variance = (L[idx] - l[idx]) / 2.0
        noise_factor = variance_shape / pm.math.maximum(max_variance, 1e-10)

        xi = 0.01 + xi_base[idx] * noise_factor

        # Skewness (truncated negative)
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

    return model


def fit(prepared: pd.DataFrame, cfg: ModelConfig, samp: SamplingConfig) -> tuple[az.InferenceData, pm.Model]:
    """Fit the model and return (idata, model)."""
    model = build_model(prepared, cfg)
    with model:
        idata = pm.sample(
            draws=samp.draws,
            tune=samp.tune,
            return_inferencedata=True,
            random_seed=samp.seed,
            target_accept=samp.target_accept,
            init=samp.init,
            progressbar=samp.progressbar,
        )
    return idata, model


def temporal_holdout(
    raw: pd.DataFrame,
    *,
    cutoff_date: pd.Timestamp,
    cfg: ModelConfig,
    samp: SamplingConfig,
    min_train_points: int = 3,
) -> az.InferenceData:
    """Train on data before cutoff_date, evaluate on data >= cutoff_date."""
    prepared = prepare_dataset(raw, top_n=cfg.top_n)

    train = prepared.loc[prepared["release_date"] < cutoff_date].copy()
    train_counts = train.groupby("benchmark")["score"].size()
    keep = train_counts[train_counts >= min_train_points].index
    train = train.loc[train["benchmark"].isin(keep)].copy()

    test = prepared.loc[prepared["release_date"] >= cutoff_date].copy()
    test = test.loc[test["benchmark"].isin(train["benchmark"].unique())].copy()

    idata, model = fit(train, cfg, samp)

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

    return grid
