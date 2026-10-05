"""The model respects the asymptote bounds and the cache key sees the inputs that matter."""

import os

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from benchmark_forecasting import config
from benchmark_forecasting.data import prepare_dataset
from benchmark_forecasting.fit import (
    data_fingerprint,
    fit,
    n_divergent,
    prune_stale_fits,
    temporal_holdout,
    thin_cached_fits,
    thin_idata,
)
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


def test_the_nutpie_path_respects_the_floor_and_keeps_the_log_likelihood(tmp_path):
    """The default sampler gets the truncated L_raw's initial values too, and the fit keeps the
    pointwise log-likelihood the comparisons read."""
    prepared = prepare_dataset(_raw(), top_n=3)
    samp = config.SamplingConfig(draws=20, tune=50, seed=1, progressbar=False)
    assert samp.sampler == "nutpie"
    idata, _ = fit(prepared, config.ModelConfig(), samp, use_cache=False, fits_dir=tmp_path)
    assert "log_likelihood" in idata.groups() and "diverging" in idata.sample_stats
    assert float(idata.posterior["L"].sel(benchmark="Human").min()) >= 0.9


def test_fit_caches_under_the_given_folder_with_the_documented_name(tmp_path, capsys):
    """The cache file name carries the slug, the tag, the non-default sampling settings and the
    data fingerprint, and lands in `fits_dir`; a second call reloads it instead of sampling."""
    prepared = prepare_dataset(_raw(), top_n=3)
    cfg = config.ModelConfig()
    samp = config.SamplingConfig(draws=5, tune=5, target_accept=0.95, seed=1, progressbar=False)
    fits = tmp_path / "fits"
    fit(prepared, cfg, samp, cache_tag="cutoff20260907", fits_dir=fits)
    expected = fits / (f"{cfg.slug}_cutoff20260907_ta95_n5t5_s1_nutpie"
                       f"_d{data_fingerprint(prepared)}.nc")
    assert expected.exists(), sorted(p.name for p in fits.iterdir())
    capsys.readouterr()
    fit(prepared, cfg, samp, cache_tag="cutoff20260907", fits_dir=fits)
    assert "Loading cached fit" in capsys.readouterr().out, "second call must reload the cache"
    assert [p.name for p in fits.iterdir()] == [expected.name]


def test_a_new_fit_deletes_the_same_fit_on_older_data(tmp_path):
    """Saving a fit removes its caches of other fingerprints, and only those."""
    prepared = prepare_dataset(_raw(), top_n=3)
    cfg = config.ModelConfig()
    samp = config.SamplingConfig(draws=5, tune=5, seed=1, progressbar=False)
    fits = tmp_path / "fits"
    fits.mkdir()
    stem = f"{cfg.slug}_cutoff20260907_n5t5_s1_nutpie"
    (fits / f"{stem}_d000000.nc").write_bytes(b"old data")
    (fits / f"{stem}_ta95_d000000.nc").write_bytes(b"another fit")
    fit(prepared, cfg, samp, cache_tag="cutoff20260907", fits_dir=fits)
    assert sorted(p.name for p in fits.iterdir()) == sorted(
        [f"{stem}_d{data_fingerprint(prepared)}.nc", f"{stem}_ta95_d000000.nc"])


def test_prune_keeps_each_fits_last_used_cache(tmp_path):
    """Per fit the most recently used file stays; other fits and non-cache files are untouched."""
    files = {"a_s1_d111111.nc": 1, "a_s1_d222222.nc": 3, "a_s1_d333333.nc": 2,
             "b_d444444.nc": 1, "notes.nc": 1}
    for name, t in files.items():
        (tmp_path / name).write_bytes(b"x")
        os.utime(tmp_path / name, (t, t))
    assert [p.name for p in prune_stale_fits(tmp_path, dry_run=True)] == [
        "a_s1_d111111.nc", "a_s1_d333333.nc"]
    assert len(list(tmp_path.iterdir())) == 5, "a dry run deletes nothing"
    prune_stale_fits(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "a_s1_d222222.nc", "b_d444444.nc", "notes.nc"]


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


def test_marginal_priors_keep_the_hierarchical_means():
    """The independent model's marginalised priors (marginal.py) integrate to one and keep the
    mean the hierarchy implies: E[child] = E[hyper-mean] when no clamp binds."""
    from scipy import special

    from benchmark_forecasting import marginal
    from benchmark_forecasting.model import K_TABLE

    for tab, mean, logit in ((marginal.gamma_table(**K_TABLE), 0.005, False),
                             (marginal.L_raw_table(0.84, 0.08), 0.84, True)):
        y = tab.y0 + tab.h * np.arange(tab.f.size)
        x = special.expit(y) if logit else np.exp(y)
        py = np.exp(tab.f) * (x * (1 - x) if logit else x)
        mass = np.trapezoid(py, y)
        assert 0.99 < mass <= 1.001
        assert abs(np.trapezoid(py * x, y) / mass - mean) < 0.01 * mean


def test_the_marginalised_independent_model_has_no_per_benchmark_hyperpriors():
    from dataclasses import replace

    prepared = prepare_dataset(_raw(), top_n=3)
    cfg = config.ModelConfig(joint=False)
    assert cfg.slug.endswith("_marg")
    model = build_model(prepared, cfg)
    names = {v.name for v in model.free_RVs}
    assert names == {"L_raw", "tau", "k", "alpha_raw", "xi_base", "s_neg"}
    point = model.initial_point()
    init = sampler_initvals(prepared, cfg)
    assert set(init) == {"L_raw", "k", "xi_base", "alpha_raw", "s_neg"}
    assert np.isfinite(model.compile_logp()(point))
    # the joint model ignores the flag
    assert replace(config.ModelConfig(), hyper_marginalised=True).slug == config.ModelConfig().slug


def _idata(chains=2, draws=8):
    rng = np.random.default_rng(0)
    diverging = np.zeros((chains, draws), dtype=bool)
    diverging[0, 1] = diverging[1, 2] = diverging[1, 3] = True  # all on dropped draws
    return az.from_dict(posterior={"L": rng.normal(size=(chains, draws, 3))},
                        log_likelihood={"y": rng.normal(size=(chains, draws, 5))},
                        sample_stats={"diverging": diverging})


def test_thinning_keeps_one_draw_in_four_and_the_full_divergence_count():
    """Every draw-indexed group is thinned alike, once, and the divergences of the whole run stay
    readable (a count over the kept draws would miss those that were dropped)."""
    idata = _idata()
    thinned = thin_idata(idata, 4)
    for group in ("posterior", "log_likelihood", "sample_stats"):
        assert thinned[group].sizes["draw"] == 2, group
    np.testing.assert_array_equal(thinned.posterior["L"].values,
                                  idata.posterior["L"].values[:, ::4])
    assert int(thinned.sample_stats["diverging"].sum()) == 0
    assert n_divergent(thinned) == n_divergent(idata) == 3
    assert thin_idata(thinned, 4) is thinned, "a thinned fit is not thinned again"


def test_cached_fits_are_thinned_in_place_once(tmp_path):
    """The migration rewrites a cache thinned under its own name and date, and skips it after."""
    path = tmp_path / "a_s1_d123456.nc"
    _idata().to_netcdf(str(path))
    os.utime(path, (100, 100))
    assert thin_cached_fits(tmp_path, 4) == [path]
    reloaded = az.from_netcdf(str(path))
    assert reloaded.posterior.sizes["draw"] == 2 and n_divergent(reloaded) == 3
    assert path.stat().st_mtime == 100, "thinning is not a use"
    assert thin_cached_fits(tmp_path, 4) == []
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a_s1_d123456.nc"]
