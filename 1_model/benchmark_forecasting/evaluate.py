"""Scoring of held-out predictions, saturation timing and residual diagnostics."""

import arviz as az
import numpy as np
import pandas as pd
from scipy.stats import energy_distance

from benchmark_forecasting.config import ErrorMetric


def crps_score(idata: az.InferenceData) -> float:
    """Mean CRPS over the test set (energy distance based)."""
    y_pred = idata.predictions.stack(sample=("chain", "draw"))["y"].to_numpy()
    y_true = idata.predictions["y_true"].to_numpy()

    scores: list[float] = []
    for pred, true in zip(y_pred, y_true, strict=True):
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


def conformal_prediction_coverage(
    idata: az.InferenceData, *, alpha: float = 0.20
) -> dict[str, float]:
    """CQR coverage on holdout data with a POSITIONAL calibration/test split.

    The first submission's Table 7. Kept as an alternative for comparison; the scripts run
    `conformal_prediction_coverage_grouped` instead, because the holdout frame is ordered by
    benchmark and a positional split puts different benchmarks in the two halves, which breaks
    the exchangeability CQR relies on (no finite-sample guarantee on Q).

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
    y_pred_samples = idata.predictions.stack(sample=("chain", "draw"))[
        "y"
    ].to_numpy()  # (n_obs, n_samples)
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
    bayesian_covered = (y_true[n_cal:] >= bayesian_lower[n_cal:]) & (
        y_true[n_cal:] <= bayesian_upper[n_cal:]
    )
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

    target_days = float(
        pd.Timestamp(target_date).to_datetime64().astype("datetime64[D]").astype(int)
    )
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
                    "same_lineage": bool(b1 in lin and b2 in lin and lin[b1] == lin[b2]),
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
        r2_model_adj = 1.0 - (1.0 - r2_model) * (n - 1) / (n - g) if n > g else float("nan")
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
