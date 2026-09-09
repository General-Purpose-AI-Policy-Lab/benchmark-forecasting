"""MCMC fitting with a cache keyed on the model slug and a fingerprint of the fitted data."""

import hashlib
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from benchmark_forecasting.config import FITS_SUBDIR, ModelConfig, SamplingConfig, cutoff_dir
from benchmark_forecasting.data import prepare_dataset
from benchmark_forecasting.model import build_model, sampler_initvals


def data_fingerprint(prepared: pd.DataFrame) -> str:
    """Short hash of the fitted observations and of the asymptote inputs (baselines, ceilings)."""
    cols = ["benchmark", "release_date", "score"]
    cols += [c for c in ("lower_bound", "human_max", "ceiling") if c in prepared.columns]
    key = prepared[cols].copy()
    key["release_date"] = pd.to_datetime(key["release_date"]).dt.strftime("%Y-%m-%d")
    for c in cols[2:]:
        key[c] = pd.to_numeric(key[c], errors="coerce").astype(float).round(6)
    payload = key.sort_values(cols[:3]).to_csv(index=False)
    return hashlib.md5(payload.encode()).hexdigest()[:6]


def fit(
    prepared: pd.DataFrame,
    cfg: ModelConfig,
    samp: SamplingConfig,
    *,
    cache_tag: str | None = None,
    use_cache: bool = True,
    fits_dir: Path | None = None,
) -> tuple[az.InferenceData, pm.Model]:
    """Fit the model and return (idata, model).

    Parameters
    ----------
    cache_tag : optional label appended to the slug for the NetCDF filename.
        The name always ends with ``_d<hash>``, a fingerprint of the fitted data, and carries
        the sampling settings that differ from the defaults (``_ta95`` for the acceptance
        target, ``_n<draws>t<tune>`` and ``_s<seed>``), so a cache is only reused for the same
        data, model and sampling:
        ``{fits_dir}/{cfg.slug}[_{cache_tag}][_ta95][_n..t..][_s..]_d{hash}.nc``.
    use_cache : if *True* (default), load from ``fits_dir`` if the file exists,
        and save there after sampling.  Set to *False* to force re-fitting.
    fits_dir : the cache folder, normally the run's ``3_outputs/<cutoff>/fits/``
        (``config.cutoff_dir(tag) / config.FITS_SUBDIR``); defaults to the no-cutoff one.
    """
    fits_dir = Path(fits_dir) if fits_dir is not None else cutoff_dir(None) / FITS_SUBDIR
    model = build_model(prepared, cfg)

    # --- cache path ---
    # The filename carries a fingerprint of the fitted observations: a cache is only
    # reused for the exact same data.  Without it, a retrodiction fit computed on an
    # older dataset would be silently reloaded after a data refresh (the cutoff tag
    # alone does not change when the underlying files do).
    fname = cfg.slug if cache_tag is None else f"{cfg.slug}_{cache_tag}"
    if samp.target_accept != 0.9:
        # A tighter acceptance target changes the posterior draws, so it names the cache too.
        fname = f"{fname}_ta{round(samp.target_accept * 100)}"
    if (samp.draws, samp.tune) != (2000, 1000):
        fname = f"{fname}_n{samp.draws}t{samp.tune}"
    if samp.seed != 42:
        fname = f"{fname}_s{samp.seed}"
    fname = f"{fname}_d{data_fingerprint(prepared)}"
    cache_path = fits_dir / f"{fname}.nc"

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
            initvals=sampler_initvals(prepared, cfg),
            progressbar=samp.progressbar,
            idata_kwargs={"log_likelihood": True},
        )

    if use_cache:
        fits_dir.mkdir(parents=True, exist_ok=True)
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
    fits_dir: Path | None = None,
) -> az.InferenceData:
    """Train on data before cutoff_date, evaluate on data >= cutoff_date.

    The frontier (top-N) and the centre of the prior on the inflection date (`days_mid`) are
    computed on the whole cutoff dataset before the split; only the scores, floors and ceilings
    the model sees are the training rows. `fits_dir` is the run's cache folder, as in `fit`.
    """
    prepared = prepare_dataset(raw, top_n=cfg.top_n)

    train = prepared.loc[prepared["release_date"] < cutoff_date].copy()
    train_counts = train.groupby("benchmark")["score"].size()
    keep = train_counts[train_counts >= min_train_points].index
    train = train.loc[train["benchmark"].isin(keep)].copy()

    test = prepared.loc[prepared["release_date"] >= cutoff_date].copy()
    test = test.loc[test["benchmark"].isin(train["benchmark"].unique())].copy()

    # The training set depends on min_train_points, so it belongs in the cache key.
    cutoff_tag = f"{cutoff_date.strftime('%Y%m%d')}_min{min_train_points}"
    idata, model = fit(train, cfg, samp, cache_tag=f"retro_{cutoff_tag}", use_cache=use_cache,
                       fits_dir=fits_dir)

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
