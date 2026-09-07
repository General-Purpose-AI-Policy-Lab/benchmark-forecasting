"""The hierarchical sigmoid model: one curve per benchmark, shared hyperpriors when joint."""

import numpy as np
import pandas as pd
import pymc as pm

from benchmark_forecasting.config import ModelConfig
from benchmark_forecasting.data import _ensure_constant_per_benchmark, asymptote_bounds


def build_model(prepared: pd.DataFrame, cfg: ModelConfig) -> pm.Model:
    """Build the PyMC model for the frontier points of ``prepared`` (see ``data.prepare_dataset``).

    The upper asymptote L of each benchmark is a Beta draw rescaled to [floor, 1] around a shared
    (or, when independent, per-benchmark) mean. ``data.asymptote_bounds`` gives the floor, L_min or
    the benchmark's highest human baseline, and a pinned value where a ceiling is known. The
    population Beta therefore describes where the asymptote sits within the feasible interval of
    each benchmark, and the hyperparameter figures map it back onto [L_min, 1]. Pinned benchmarks
    keep an unused Beta draw so the coordinates stay uniform; a latent child with no likelihood
    attached carries no information about its parents, so the hyperposterior is untouched.
    """
    required = {"benchmark", "score", "lower_bound", "days", "days_mid"}
    missing = required - set(prepared.columns)
    if missing:
        raise ValueError(f"build_model: missing required columns: {sorted(missing)}")

    _ensure_constant_per_benchmark(prepared, "lower_bound")

    bench_idx, bench_names = pd.factorize(prepared["benchmark"], sort=True)
    d = prepared.assign(benchmark_idx=bench_idx).reset_index(drop=True)
    bounds = asymptote_bounds(prepared, cfg).reindex([str(b) for b in bench_names])

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

        # Beta draw shared by every benchmark, mapped onto [floor, 1] where the floor is L_min or
        # the benchmark's highest human baseline.  A truncated Beta was tried first and made NUTS
        # diverge on nearly every draw (the normaliser 1 - CDF(floor) is tiny where the floor is
        # high); rescaling keeps the geometry of the plain model.
        L_raw = pm.Beta("L_raw", mu=L_raw_mu, sigma=L_raw_sigma_safe, dims="benchmark")
        is_fixed = bounds["L_fixed"].notna().to_numpy()
        L_lo = pm.Data("L_floor", bounds["L_floor"].to_numpy(dtype=float), dims="benchmark")
        L_free = L_lo + (1.0 - L_lo) * L_raw

        if is_fixed.any():
            fixed_vals = bounds["L_fixed"].fillna(0.0).to_numpy(dtype=float)
            L_fixed_data = pm.Data("L_fixed", fixed_vals, dims="benchmark")
            L_is_fixed = pm.Data("L_is_fixed", is_fixed.astype(float), dims="benchmark")
            L = pm.Deterministic(
                "L", L_is_fixed * L_fixed_data + (1.0 - L_is_fixed) * L_free, dims="benchmark"
            )
        else:
            L = pm.Deterministic("L", L_free, dims="benchmark")

        # Lower bound l per benchmark (fixed data)
        l_per_bench = d.groupby("benchmark_idx")["lower_bound"].first().to_numpy()
        lower = pm.Data("l", l_per_bench, dims="benchmark")

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
            alpha_raw = pm.Gamma(
                "alpha_raw", mu=alpha_raw_mu, sigma=alpha_raw_sigma, dims="benchmark"
            )
            alpha = pm.Deterministic("alpha", alpha_raw + 1.0, dims="benchmark")

            base = pm.math.maximum(1 - (1 - alpha[idx]) * pm.math.exp(-logits), 1e-10)
            sigmoid = pm.math.exp(1 / (1 - alpha[idx]) * pm.math.log(base))
        else:
            raise ValueError(f"Unsupported sigmoid: {cfg.sigmoid}")

        mu = pm.Deterministic("mu", lower[idx] + (L[idx] - lower[idx]) * sigmoid, dims="obs")

        # Heteroscedastic noise: increases away from bounds
        xi_base_mu = pm.Gamma(
            "xi_base_mu",
            mu=0.05 + top_n / 50,
            sigma=0.02,
            dims=None if joint else "benchmark",
        )
        xi_base_sigma = pm.HalfNormal(
            "xi_base_sigma", sigma=0.05, dims=None if joint else "benchmark"
        )
        # Clamp sigma so that Gamma(mu, sigma) stays valid (alpha = (mu/sigma)^2 > 0).
        xi_base_sigma_safe = pm.math.minimum(xi_base_sigma, xi_base_mu - 1e-6)
        xi_base = pm.Gamma("xi_base", mu=xi_base_mu, sigma=xi_base_sigma_safe, dims="benchmark")

        variance_shape = pm.math.sqrt(pm.math.maximum((mu - lower[idx]) * (L[idx] - mu), 0.0))
        max_variance = (L[idx] - lower[idx]) / 2.0
        noise_factor = variance_shape / pm.math.maximum(max_variance, 1e-10)

        xi = pm.math.maximum(0.01 + xi_base[idx] * noise_factor, 1e-6)

        if cfg.skew:
            # Skewness (negative values = scores below latent curve)
            s_mu = pm.Normal(
                "s_mu", mu=-2 - top_n / 2, sigma=0.5, dims=None if joint else "benchmark"
            )
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
