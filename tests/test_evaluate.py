"""Saturation dates invert the fitted sigmoid analytically."""

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr

from benchmark_forecasting.evaluate import saturated_proportion, saturation_dates


def _idata(k, tau, alpha=None):
    coords = {"chain": [0], "draw": np.arange(4), "benchmark": ["A", "B"]}
    data = {
        "k": (("chain", "draw", "benchmark"), np.full((1, 4, 2), k)),
        "tau": (("chain", "draw", "benchmark"), np.full((1, 4, 2), tau)),
    }
    if alpha is not None:
        data["alpha"] = (("chain", "draw", "benchmark"), np.full((1, 4, 2), alpha))
    return az.InferenceData(
        posterior=xr.Dataset(
            {n: xr.DataArray(v[1], dims=v[0]) for n, v in data.items()}, coords=coords
        )
    )


def _frontier():
    return pd.DataFrame(
        {
            "benchmark": ["A", "A", "B", "B"],
            "release_date": pd.to_datetime(
                ["2024-01-01", "2024-06-01", "2025-01-01", "2025-06-01"]
            ),
            "score": [0.1, 0.5, 0.1, 0.5],
            "category": ["x", "x", "y", "y"],
        }
    )


def test_logistic_crossing_is_tau_plus_logit_over_k():
    k, tau = 0.01, 200.0
    out = saturation_dates(_idata(k, tau), prepared_frontier=_frontier(), saturation_fraction=0.95)
    expected_days = tau + np.log(0.95 / 0.05) / k
    assert np.allclose(out["sat_median_days"] - out["sat_median_days"].min(), [0, 366])
    assert (
        out.set_index("benchmark").loc["A", "sat_median"]
        == (pd.Timestamp("2024-01-01") + pd.Timedelta(days=round(expected_days))).to_datetime64()
    )


def test_saturated_proportion_counts_benchmarks_past_the_threshold():
    k, tau = 0.01, 200.0
    prop = saturated_proportion(
        _idata(k, tau),
        prepared_frontier=_frontier(),
        target_date="2030-01-01",
        saturation_fraction=0.95,
    )
    assert prop["median"] == 1.0 and prop["n_benchmarks"] == 2
    early = saturated_proportion(
        _idata(k, tau),
        prepared_frontier=_frontier(),
        target_date="2024-12-01",
        saturation_fraction=0.95,
    )
    assert early["median"] == 0.0
    sub = saturated_proportion(
        _idata(k, tau), prepared_frontier=_frontier(), target_date="2030-01-01", benchmarks=["A"]
    )
    assert sub["benchmarks"] == ["A"]
