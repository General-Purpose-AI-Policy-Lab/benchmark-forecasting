"""Posterior forecast grid for every fitted benchmark."""

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm


def _date_grid_for_benchmark(
    group: pd.DataFrame, *, end_date: pd.Timestamp, n_points: int
) -> pd.DataFrame:
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
        raise ValueError(
            "generate_forecast expects prepared_frontier from prepare_dataset() (missing 'days')."
        )

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

        tau_df = pd.DataFrame({"benchmark": tau_days.index}).assign(
            mean_tau_days=tau_days.to_numpy(), mean_tau=tau_dates.to_numpy()
        )
        grid = grid.merge(tau_df, on="benchmark", how="left")

    return grid
