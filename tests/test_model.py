"""The model respects the asymptote bounds and the cache key sees the inputs that matter."""

import numpy as np
import pandas as pd
import pymc as pm

from benchmark_forecasting import config
from benchmark_forecasting.data import prepare_dataset
from benchmark_forecasting.fit import data_fingerprint, fit, temporal_holdout
from benchmark_forecasting.model import build_model, sampler_initvals


def _raw():
    dates = pd.date_range("2023-01-01", periods=8, freq="90D")
    frames = []
    for bench, hm, ceil in (
        ("Free", np.nan, np.nan),
        ("Human", 0.9, np.nan),
        ("Ceiled", np.nan, 0.95),
    ):
        frames.append(
            pd.DataFrame(
                {
                    "benchmark": bench,
                    "release_date": dates,
                    "score": np.linspace(0.1, 0.7, 8),
                    "model_version": [f"m{i}" for i in range(8)],
                    "lower_bound": 0.0,
                    "human_max": hm,
                    "ceiling": ceil,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_prior_draws_respect_the_pin_and_l_min():
    """Forward sampling ignores the truncation Potential; the floor is checked by MCMC below."""
    prepared = prepare_dataset(_raw(), top_n=3)
    model = build_model(prepared, config.ModelConfig())
    with model:
        prior = pm.sample_prior_predictive(draws=200, random_seed=1)
    L = prior.prior["L"]
    assert list(L.coords["benchmark"].values) == ["Ceiled", "Free", "Human"]
    assert np.allclose(L.sel(benchmark="Ceiled"), 0.95)
    assert float(L.sel(benchmark="Free").min()) >= 0.75
    assert "L_truncation" not in model.named_vars, "no normaliser by default"
    renorm = build_model(prepared, config.ModelConfig(L_floor_renormalised=True))
    assert "L_truncation" in renorm.named_vars


def test_without_rules_the_model_is_the_plain_scaled_beta():
    prepared = prepare_dataset(_raw(), top_n=3)
    cfg = config.ModelConfig(L_floor_observed=False, L_fixed_from_ceiling=False)
    model = build_model(prepared, cfg)
    assert "L_fixed" not in model.named_vars and "L_truncation" not in model.named_vars


def test_fingerprint_changes_with_scores_baselines_and_ceilings():
    prepared = prepare_dataset(_raw(), top_n=3)
    base = data_fingerprint(prepared)
    assert data_fingerprint(prepared.assign(score=prepared["score"] + 0.01)) != base
    assert data_fingerprint(prepared.assign(human_max=0.8)) != base
    assert data_fingerprint(prepared.assign(ceiling=1.0)) != base
    assert data_fingerprint(prepared.sample(frac=1, random_state=0)) == base, "order-free"


def test_a_short_sampling_respects_the_floor_and_keeps_the_log_likelihood():
    prepared = prepare_dataset(_raw(), top_n=3)
    cfg = config.ModelConfig()
    init = sampler_initvals(prepared, cfg)
    assert init["L_raw"][2] > (0.9 - cfg.L_min) / (1 - cfg.L_min), "Human starts above its floor"
    with build_model(prepared, cfg):
        idata = pm.sample(
            draws=5,
            tune=5,
            chains=1,
            initvals=init,
            progressbar=False,
            compute_convergence_checks=False,
            random_seed=1,
            idata_kwargs={"log_likelihood": True},
        )
    assert "log_likelihood" in idata.groups()
    assert float(idata.posterior["L"].sel(benchmark="Human").min()) >= 0.9


def test_fit_caches_under_the_given_folder_with_the_documented_name(tmp_path):
    """The cache file name carries the slug, the tag, the non-default sampling settings and the
    data fingerprint, and lands in `fits_dir`; a second call reloads it instead of sampling."""
    prepared = prepare_dataset(_raw(), top_n=3)
    cfg = config.ModelConfig()
    samp = config.SamplingConfig(draws=5, tune=5, target_accept=0.95, seed=1, progressbar=False)
    fits = tmp_path / "fits"
    fit(prepared, cfg, samp, cache_tag="cutoff20260907", fits_dir=fits)
    expected = fits / f"{cfg.slug}_cutoff20260907_ta95_n5t5_s1_d{data_fingerprint(prepared)}.nc"
    assert expected.exists(), sorted(p.name for p in fits.iterdir())
    before = expected.stat().st_mtime
    fit(prepared, cfg, samp, cache_tag="cutoff20260907", fits_dir=fits)
    assert expected.stat().st_mtime == before, "second call must reload the cache"


def test_temporal_holdout_files_its_cache_under_the_run_folder(tmp_path):
    raw = _raw()
    samp = config.SamplingConfig(draws=5, tune=5, seed=1, progressbar=False)
    fits = tmp_path / "fits"
    cutoff = pd.Timestamp(raw["release_date"].max()) - pd.Timedelta(days=200)
    idata = temporal_holdout(raw, cutoff_date=cutoff, cfg=config.ModelConfig(), samp=samp,
                             min_train_points=2, fits_dir=fits)
    names = sorted(p.name for p in fits.iterdir())
    assert names and all(f"retro_{cutoff:%Y%m%d}_min2" in n for n in names), names
    assert "predictions" in idata.groups()
