# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Benchmark forecasts
#
# Fits the hierarchical Harvey model on the frontier of every benchmark, validates it by temporal
# holdout against seven variants, draws the forecast figures per category and runs the sensitivity
# analyses. Cached posteriors, figures and JSON results go under `3_outputs/<cutoff>/`; the
# French note's curated figures are refreshed in `4_writeups/note/figures/`.
#
# Percent script: run it as `python 2_analyses/forecasts.py` or open it as a notebook (Jupyter reads the `# %%`
# cells through jupytext, VS Code natively). Input: `0_input/`, synced from benchmark-data-pipeline
# with `python -m benchmark_forecasting sync`.

# %%
# Single-threaded BLAS before numpy is imported: with Apple Accelerate, four chain processes each
# spawning a full thread pool oversubscribe the cores and slow NUTS down ~50x.
import os

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
if ROOT.name == "2_analyses":            # opened as a notebook from this folder
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "1_model"))
os.chdir(ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import arviz as az  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import benchmark_forecasting as bf  # noqa: E402
from benchmark_forecasting.config import (  # noqa: E402
    ALL_MODEL_CONFIGS,
    DATA_CUTOFF,
    MAIN_MODEL,
    MIN_TRAIN_POINTS,
    NOTE_FIGURES_DIR,
    SAMPLING_CONFIG,
    checked_data_cutoff,
    cutoff_dir,
    cutoff_tag,
    sampling_for,
    variant_slug,
)

plotting = bf.plotting

# %% [markdown]
# ## Setup

# %%
# ---- Model ----
# The main model and the sampling settings come from config.py, shared with revision_analyses.py.
# The asymptote L of each benchmark is floored at the best performance observed on it (human
# baseline or model score), pinned where the pipeline records a ceiling or where a score of 1.0
# was reached (see ModelConfig and data.asymptote_bounds).
MODEL_CONFIG = ALL_MODEL_CONFIGS[MAIN_MODEL]

LANGUAGE: plotting.Language = "en"
DOCUMENT_TYPE: plotting.DocumentType = "paper"

SAVEFIGS = True

# PNG for note (Google Docs), PDF for paper (LaTeX)
IMG_EXT = "png" if DOCUMENT_TYPE == "note" else "pdf"
IMG_DPI = 300

# Also generate FR note figures
ALSO_GENERATE_FR = True

# ---- Data cutoff ----
# Only keep model results released on or before this date (inclusive) for fitting and
# forecasting; `config.DATA_CUTOFF`, the day the synced pipeline run happened, so that
# re-running later against a newer sync reproduces this run rather than silently absorbing newer
# models. None fits all available data.
DATA_CUTOFF_DATE: pd.Timestamp | None = pd.to_datetime(checked_data_cutoff(DATA_CUTOFF))

# `cutoffYYYYMMDD`: names the output folder, suffixes (with a leading underscore) the fit caches
# and the figure files.
CUTOFF_TAG = cutoff_tag(DATA_CUTOFF_DATE)
CUTOFF_SUFFIX = f"_{CUTOFF_TAG}" if CUTOFF_TAG else ""
# Everything is filed under the run's cutoff folder, 3_outputs/cutoffYYYYMMDD/: the fit
# caches in fits/, the EN paper PDFs by theme, the FR note PNGs in a fr/ subfolder beside them.
CUTOFF_DIR = cutoff_dir(CUTOFF_TAG)
FITS_DIR = CUTOFF_DIR / "fits"
FORECAST_DIR = f"{CUTOFF_DIR}/forecasts"
FORECAST_DIR_FR = f"{FORECAST_DIR}/fr"
CALIB_DIR = f"{CUTOFF_DIR}/calibration"
CALIB_DIR_FR = f"{CALIB_DIR}/fr"
SENS_DIR = f"{CUTOFF_DIR}/sensitivity"
HIGH_LEVEL_DIR = f"{CUTOFF_DIR}/high_level"
# The French note's curated figures, refreshed by every run (a write-up folder, so not dated).
NOTE_FIG_DIR = str(NOTE_FIGURES_DIR)

# ---- Output directories ----
# Created explicitly rather than relying on git having materialised them when
# checking out tracked figures: a fresh clone has none of the gitignored ones and
# the first savefig would fail.  Every folder holding FR figures also gets an svg/
# subfolder, where `plotting.save_figure(..., also_svg=True)` writes the vector copy.
for _plot_dir in (
    NOTE_FIG_DIR,
    f"{NOTE_FIG_DIR}/svg",
    HIGH_LEVEL_DIR,
    FORECAST_DIR,
    FORECAST_DIR_FR,
    f"{FORECAST_DIR_FR}/svg",
    CALIB_DIR,
    CALIB_DIR_FR,
    f"{CALIB_DIR_FR}/svg",
    SENS_DIR,
    str(FITS_DIR),
):
    os.makedirs(_plot_dir, exist_ok=True)

# %% [markdown]
# ## Load + prepare data

# %%
raw = bf.load_dataset()
provenance = json.loads((ROOT / "0_input" / "provenance.json").read_text())
print(f"Input: pipeline commit {provenance['pipeline_commit'][:8]} built {provenance['pipeline_built_at']}")

# Apply data cutoff: drop observations released after DATA_CUTOFF_DATE
if DATA_CUTOFF_DATE is not None:
    n_before = len(raw)
    # Inclusive: a model released on the freeze date belongs to the frozen dataset.
    raw = raw[raw["release_date"] <= DATA_CUTOFF_DATE].copy()
    print(f"Data cutoff {DATA_CUTOFF_DATE.date()}: {n_before} → {len(raw)} observations "
          f"({n_before - len(raw)} removed)")

# Frontier used for fitting and for the plots: one point per model family and day
# (best reasoning effort / size), then the expanding top-N.
data = bf.prepare_dataset(raw, top_n=MODEL_CONFIG.top_n)
print(f"Frontier points: {len(data)}")

baselines = bf.load_baselines()

# Where the asymptote comes from, per benchmark: estimated, floored by a human baseline or pinned.
bounds = bf.asymptote_bounds(data, MODEL_CONFIG)
print(bounds["reason"].value_counts().to_string())
print(bounds.loc[bounds["reason"] != "estimated"].to_string())

# %% [markdown]
# ## Fit + forecast

# %%
idata_forecast, model_forecast = bf.fit(
    data, MODEL_CONFIG, SAMPLING_CONFIG,
    cache_tag=CUTOFF_TAG or None,
    fits_dir=FITS_DIR,
)

# %%
END_DATE = pd.to_datetime("2030-03-01")
forecast_df = bf.generate_forecast(
    idata_forecast,
    model_forecast,
    prepared_frontier=data,
    end_date=END_DATE,
    n_points=250,
    ci_level=0.8,
)

print(forecast_df.head().to_string())

# %% [markdown]
# ### Forecast plots by category

# %%
categories = list(data["category"].dropna().unique())   # the pipeline's view always carries it

plot_style = plotting.PlotStyle(language=LANGUAGE, document_type=DOCUMENT_TYPE)
for cat in categories:
    obs_cat = data.loc[data["category"] == cat]
    pred_cat = forecast_df.loc[forecast_df["category"] == cat]

    fig, ax = plotting.plot_forecasts_by_category(
        observed=obs_cat,
        forecast=pred_cat,
        baselines=baselines,
        end_date=END_DATE,
        category_name=cat,
        plot_style=plot_style,
    )
    if SAVEFIGS:
        plotting.save_figure(
            fig,
            f"{FORECAST_DIR_FR if plot_style.language == 'fr' else FORECAST_DIR}/forecast_{cat.replace(' & ', '_').replace(' ', '_')}_{plot_style.language}_{plot_style.document_type}{CUTOFF_SUFFIX}.{IMG_EXT}",
            dpi=IMG_DPI,
            also_svg=plot_style.language == "fr",
        )
plt.show()

# %% [markdown]
# ### Posterior: proportion of benchmarks saturated by 2030

# %%
SATURATION_FRACTION = 0.95
SATURATION_TARGET_DATE = pd.Timestamp("2030-01-01")

plot_style = plotting.PlotStyle(language=LANGUAGE, document_type=DOCUMENT_TYPE)
fig, ax, sat_summary = plotting.plot_saturation_proportion_posterior(
    idata_forecast,
    prepared_frontier=data,
    target_date=SATURATION_TARGET_DATE,
    saturation_fraction=SATURATION_FRACTION,
    ci_level=0.80,
    plot_style=plot_style,
)
if SAVEFIGS:
    fig.savefig(
        f"{HIGH_LEVEL_DIR}/saturation_{plot_style.language}_{plot_style.document_type}{CUTOFF_SUFFIX}.{IMG_EXT}",
        dpi=IMG_DPI,
        bbox_inches="tight",
    )
plt.show()

print(sat_summary)

# %% [markdown]
# ### Posterior of the asymptotes and of the hyperparameters
#
# Redrawn after every main fit: the population curve of L with one point per benchmark (EN paper
# and FR note) and the hyperparameter posteriors (EN paper).

# %%
for style, folder, ext in (
    (plotting.PlotStyle(language="en", document_type="paper"), HIGH_LEVEL_DIR, "pdf"),
    (plotting.PlotStyle(language="fr", document_type="note"), NOTE_FIG_DIR, "png"),
):
    if style.language == "fr" and not ALSO_GENERATE_FR:
        continue
    fig_L, _ = plotting.plot_L_distribution(idata_forecast, L_min=MODEL_CONFIG.L_min, plot_style=style)
    if SAVEFIGS:
        plotting.save_figure(
            fig_L,
            f"{folder}/Hierarchical_L_intervals_{style.language}_{style.document_type}{CUTOFF_SUFFIX}.{ext}",
            dpi=IMG_DPI, also_svg=style.language == "fr",
        )
    plt.close(fig_L)

fig_h, _ = plotting.plot_hyperparameters(
    idata_forecast, L_min=MODEL_CONFIG.L_min, plot_style=plotting.PlotStyle(language="en", document_type="paper")
)
if SAVEFIGS:
    fig_h.savefig(f"{HIGH_LEVEL_DIR}/hyperparameters_en_paper{CUTOFF_SUFFIX}.pdf", dpi=IMG_DPI, bbox_inches="tight")
plt.close(fig_h)

# %% [markdown]
# ## Asymmetry visualization (Harvey curves vs logistic)

# %%
# The asymmetry figure reads the main fit: the Harvey curves it draws are the fitted ones.
idata_asym = idata_forecast

# %%
plot_style = plotting.PlotStyle(language=LANGUAGE, document_type=DOCUMENT_TYPE)
plotting.plot_harvey_asymmetry(idata_asym, plot_style=plot_style)
if SAVEFIGS:
    plt.savefig(
        f"{HIGH_LEVEL_DIR}/asymmetry_{plot_style.language}_{plot_style.document_type}{CUTOFF_SUFFIX}.{IMG_EXT}",
        dpi=IMG_DPI,
        bbox_inches="tight",
    )
plt.show()

# %% [markdown]
# ## FR note figures
#
# Drawn right after the main and asymmetry fits, so a run refreshes `4_writeups/note/figures/` and the
# `fr/` forecast panels before the hour of retrodiction and ablation fits below. The FR calibration
# curves need the retrodictions and follow them.

# %%
# --- Generate FR note figures ---
if ALSO_GENERATE_FR:
    print("\n=== Generating FR note figures ===")
    fr_style = plotting.PlotStyle(language="fr", document_type="note")

    # Forecasts
    for cat in categories:
        obs_cat = data.loc[data["category"] == cat]
        pred_cat = forecast_df.loc[forecast_df["category"] == cat]
        fig, ax = plotting.plot_forecasts_by_category(
            observed=obs_cat, forecast=pred_cat, baselines=baselines,
            end_date=END_DATE, category_name=cat, plot_style=fr_style,
        )
        if SAVEFIGS:
            plotting.save_figure(
                fig,
                f"{FORECAST_DIR_FR}/forecast_{cat.replace(' & ', '_').replace(' ', '_')}_fr_note{CUTOFF_SUFFIX}.png",
                dpi=IMG_DPI, also_svg=True,
            )
        plt.close(fig)
    print("  FR forecasts done")

    # Saturation
    fig, ax, _ = plotting.plot_saturation_proportion_posterior(
        idata_forecast, prepared_frontier=data,
        target_date=SATURATION_TARGET_DATE, saturation_fraction=SATURATION_FRACTION,
        ci_level=0.80, plot_style=fr_style,
    )
    if SAVEFIGS:
        plotting.save_figure(
            fig, f"{NOTE_FIG_DIR}/saturation_fr_note{CUTOFF_SUFFIX}.png", dpi=IMG_DPI, also_svg=True
        )
    plt.close(fig)
    print("  FR saturation done")

    # Asymmetry
    fig, ax = plotting.plot_harvey_asymmetry(idata_asym, plot_style=fr_style)
    if SAVEFIGS:
        plotting.save_figure(
            fig, f"{NOTE_FIG_DIR}/asymmetry_fr_note{CUTOFF_SUFFIX}.png", dpi=IMG_DPI, also_svg=True
        )

    plt.close(fig)
    print("  FR asymmetry done")

    print("=== FR note figures generated ===")

# %% [markdown]
# ## Retrodiction analysis

# %%
# --- Full model grid: Sigmoid × Structure × Likelihood = 8 variants (config.ALL_MODEL_CONFIGS) ---
cutoff_date = pd.to_datetime("2025-01-01")

# --- Retrodiction (temporal holdout) for all variants ---
retrodiction_idata = {}
for model_name, model_config in ALL_MODEL_CONFIGS.items():
    print(f"Fitting retrodiction: {model_name}")
    idata_retro = bf.temporal_holdout(
        raw,
        cutoff_date=cutoff_date,
        cfg=model_config,
        samp=sampling_for(model_config),
        min_train_points=MIN_TRAIN_POINTS,
        fits_dir=FITS_DIR,
    )
    retrodiction_idata[model_name] = idata_retro

# %%
plot_style = plotting.PlotStyle(language=LANGUAGE, document_type=DOCUMENT_TYPE)

for model_name, idata_retro in retrodiction_idata.items():
    print(f"\nEvaluating model: {model_name}")
    print("  CRPS:", bf.crps_score(idata_retro))
    print("  RMSE:", bf.point_error(idata_retro, metric="RMSE"))
    print("  MAE:", bf.point_error(idata_retro, metric="MAE"))
# The calibration figures are drawn once, in the sensitivity block below (calibration_<slug>).

# %%
# FR calibration curves, once the retrodictions exist.
if ALSO_GENERATE_FR:
    fr_style = plotting.PlotStyle(language="fr", document_type="note")
    for model_name, idata_retro in retrodiction_idata.items():
        fig, ax = plotting.plot_calibration_curve(idata_retro, n_points=20, plot_style=fr_style)
        if SAVEFIGS:
            plotting.save_figure(
                fig,
                f"{CALIB_DIR_FR}/calibration_{variant_slug(model_name)}_fr_note{CUTOFF_SUFFIX}.png",
                dpi=IMG_DPI, also_svg=True,
            )
        plt.close(fig)
    print("  FR calibration done")

# %% [markdown]
# ## Sensitivity analyses

# %%
# --- Sensitivity analyses on all 8 model variants ---

ablation_results: dict = {}
ablation_idata: dict = {}  # keep idata for LOO + CQR below

paper_style = plotting.PlotStyle(language="en", document_type="paper")

for name, cfg in ALL_MODEL_CONFIGS.items():
    print(f"\nFitting: {name}")
    idata_abl, model_abl = bf.fit(
        data, cfg, sampling_for(cfg),
        cache_tag=CUTOFF_TAG or None,
        fits_dir=FITS_DIR,
    )

    slug = variant_slug(name)   # e.g. "harvey_joint_skew"

    # --- Saturation figures at 90%, 95%, 99% thresholds ---
    sat_sum: dict = {}
    for threshold in [0.90, 0.95, 0.99]:
        fig_sat, _, sat_sum = plotting.plot_saturation_proportion_posterior(
            idata_abl,
            prepared_frontier=data,
            target_date=SATURATION_TARGET_DATE,
            saturation_fraction=threshold,
            ci_level=0.80,
            plot_style=paper_style,
        )
        if SAVEFIGS:
            fig_sat.savefig(
                f"{SENS_DIR}/saturation_{slug}_t{int(threshold*100)}_en_paper{CUTOFF_SUFFIX}.pdf",
                dpi=IMG_DPI, bbox_inches="tight",
            )
        plt.close(fig_sat)

    # Store saturation results for all thresholds
    sat_by_threshold: dict = {}
    for threshold in [0.90, 0.95, 0.99]:
        _, _, sat_t = plotting.plot_saturation_proportion_posterior(
            idata_abl, prepared_frontier=data,
            target_date=SATURATION_TARGET_DATE, saturation_fraction=threshold,
            ci_level=0.80, plot_style=paper_style,
        )
        sat_by_threshold[f"t{int(threshold*100)}"] = {"median": sat_t["median"], "ci80": sat_t["ci"]}
        plt.close("all")
    ablation_results[name] = sat_by_threshold
    sat95 = sat_by_threshold["t95"]
    print(f"  Saturation (95%): median={sat95['median']:.1%}, 80% CI=[{sat95['ci80'][0]:.1%}, {sat95['ci80'][1]:.1%}]")

    # --- Forecast figures per category ---
    forecast_abl = bf.generate_forecast(
        idata_abl, model_abl, prepared_frontier=data,
        end_date=END_DATE, n_points=250, ci_level=0.8,
    )
    for cat in data["category"].dropna().unique():
        obs_cat = data.loc[data["category"] == cat]
        pred_cat = forecast_abl.loc[forecast_abl["category"] == cat]
        fig_fc, _ = plotting.plot_forecasts_by_category(
            observed=obs_cat, forecast=pred_cat, baselines=baselines,
            end_date=END_DATE, category_name=cat, plot_style=paper_style,
        )
        if SAVEFIGS:
            fig_fc.savefig(
                f"{SENS_DIR}/forecast_{cat.replace(' & ', '_').replace(' ', '_')}_{slug}_en_paper{CUTOFF_SUFFIX}.pdf",
                dpi=IMG_DPI, bbox_inches="tight",
            )
        plt.close(fig_fc)

    # --- Calibration figure (reuse retrodiction from above) ---
    idata_retro_abl = retrodiction_idata[name]
    fig_cal, _ = plotting.plot_calibration_curve(idata_retro_abl, n_points=20, plot_style=paper_style)
    if SAVEFIGS:
        fig_cal.savefig(
            f"{CALIB_DIR}/calibration_{slug}_en_paper{CUTOFF_SUFFIX}.pdf",
            dpi=IMG_DPI, bbox_inches="tight",
        )
    plt.close(fig_cal)

    # Store full model + retrodiction idata for LOO and CQR
    ablation_idata[name] = (idata_abl, cfg, idata_retro_abl)

    # Retrodiction metrics
    crps = bf.crps_score(idata_retro_abl)
    rmse = bf.point_error(idata_retro_abl, metric="RMSE")
    ablation_results[name]["crps"] = crps
    ablation_results[name]["rmse"] = rmse
    ablation_results[name]["median"] = sat95["median"]
    ablation_results[name]["ci80"] = sat95["ci80"]
    print(f"  Retrodiction: CRPS={crps:.4f}, RMSE={rmse:.4f}")

print("\n=== Model Sensitivity Summary ===")
for name, res in ablation_results.items():
    print(f"  {name}: sat={res['median']:.1%} [{res['ci80'][0]:.1%}, {res['ci80'][1]:.1%}], CRPS={res['crps']:.4f}, RMSE={res['rmse']:.4f}")
    for tk, tv in res.items():
        if tk.startswith('t') and isinstance(tv, dict):
            print(f"    {tk}: median={tv['median']:.1%}, 80% CI=[{tv['ci80'][0]:.1%}, {tv['ci80'][1]:.1%}]")

# %%
# --- LOO-ELPD comparison + posterior inspection of α and s ---

print("=" * 60)
print("LOO-ELPD MODEL COMPARISON")
print("=" * 60)

loo_results: dict = {}
loo_summary: dict = {}  # for JSON export
for name, (idata_abl, _cfg, *_rest) in ablation_idata.items():
    try:
        loo = az.loo(idata_abl, var_name="y")
        loo_results[name] = loo
        loo_summary[name] = {
            "elpd_loo": float(loo.elpd_loo),
            "se": float(loo.se),
            "p_loo": float(loo.p_loo),
            "n_pareto_k_bad": int(np.sum(loo.pareto_k > 0.7)) if hasattr(loo, "pareto_k") else 0,
        }
        print(f"\n{name}:")
        print(f"  ELPD LOO: {loo.elpd_loo:.1f} ± {loo.se:.1f}")
        print(f"  p_loo (effective params): {loo.p_loo:.1f}")
        print(f"  Pareto k > 0.7: {loo_summary[name]['n_pareto_k_bad']} observations")
    except Exception as e:
        print(f"\n{name}: LOO failed — {e}")

comparison: pd.DataFrame = pd.DataFrame()
if len(loo_results) > 1:
    print("\n" + "-" * 60)
    print("MODEL RANKING (az.compare)")
    print("-" * 60)
    comparison = az.compare(loo_results)
    print(comparison.to_string())
    # Store ranking
    cols = comparison.columns.tolist()
    for model_name in comparison.index:
        if model_name in loo_summary:
            loo_summary[model_name]["rank"] = int(comparison.loc[model_name, "rank"]) if "rank" in cols else int(comparison.index.get_loc(model_name)) # type: ignore
            loo_summary[model_name]["elpd_diff"] = float(comparison.loc[model_name, "elpd_diff"]) if "elpd_diff" in cols else None # type: ignore
            loo_summary[model_name]["dse"] = float(comparison.loc[model_name, "dse"]) if "dse" in cols else None # type: ignore
            loo_summary[model_name]["weight"] = float(comparison.loc[model_name, "weight"]) if "weight" in cols else None # type: ignore

# --- Posterior of α_mu (Harvey shape hyperprior) ---
print("\n" + "=" * 60)
print("POSTERIOR OF α_mu (Harvey shape hyperprior)")
print("=" * 60)
print("α = 2 → standard logistic. α > 2 → asymmetric (faster saturation).\n")

posterior_summary: dict = {}
for name, (idata_abl, cfg, *_) in ablation_idata.items():
    post = idata_abl.posterior
    entry: dict = {}
    if "alpha_raw_mu" in post:
        if cfg.joint:
            alpha_mu = post["alpha_raw_mu"].values.flatten() + 1
            entry["alpha_mu_mean"] = float(np.mean(alpha_mu))
            entry["alpha_mu_median"] = float(np.median(alpha_mu))
            entry["alpha_mu_ci95"] = [float(np.percentile(alpha_mu, 2.5)), float(np.percentile(alpha_mu, 97.5))]
            entry["alpha_mu_prob_gt2"] = float((alpha_mu > 2).mean())
            print(f"{name}: mean={entry['alpha_mu_mean']:.2f}, median={entry['alpha_mu_median']:.2f}, "
                  f"95% CI=[{entry['alpha_mu_ci95'][0]:.2f}, {entry['alpha_mu_ci95'][1]:.2f}], "
                  f"P(α_mu > 2)={entry['alpha_mu_prob_gt2']:.4f}")
        else:
            alpha = post["alpha_raw"].values + 1
            alpha_flat = alpha.flatten()
            entry["alpha_mean"] = float(np.mean(alpha_flat))
            entry["alpha_median"] = float(np.median(alpha_flat))
            entry["alpha_prob_gt2"] = float((alpha_flat > 2).mean())
            print(f"{name} (per-benchmark): mean={entry['alpha_mean']:.2f}, median={entry['alpha_median']:.2f}, "
                  f"P(α > 2)={entry['alpha_prob_gt2']:.4f}")
    posterior_summary[name] = entry

# --- Posterior of s_mu (skewness hyperprior) ---
print("\n" + "=" * 60)
print("POSTERIOR OF s_mu (skewness hyperprior)")
print("=" * 60)
print("s = 0 → symmetric. s < 0 → scores below latent curve.\n")

for name, (idata_abl, cfg, *_) in ablation_idata.items():
    post = idata_abl.posterior
    if "s_mu" in post:
        if cfg.joint:
            s_mu = post["s_mu"].values.flatten()
            posterior_summary[name]["s_mu_mean"] = float(np.mean(s_mu))
            posterior_summary[name]["s_mu_median"] = float(np.median(s_mu))
            posterior_summary[name]["s_mu_ci95"] = [float(np.percentile(s_mu, 2.5)), float(np.percentile(s_mu, 97.5))]
            print(f"{name}: mean={np.mean(s_mu):.3f}, median={np.median(s_mu):.3f}, "
                  f"95% CI=[{np.percentile(s_mu, 2.5):.3f}, {np.percentile(s_mu, 97.5):.3f}]")
        else:
            s = post["s"].values.flatten()
            posterior_summary[name]["s_mean"] = float(np.mean(s))
            posterior_summary[name]["s_median"] = float(np.median(s))
            print(f"{name} (per-benchmark): mean={np.mean(s):.3f}, median={np.median(s):.3f}")
    elif "s" not in post:
        print(f"{name}: no skewness parameter (normal likelihood)")

# --- Store LOO + posterior results for JSON export ---
ablation_results["loo"] = loo_summary
ablation_results["posteriors"] = posterior_summary

# %%
# --- CQR (Conformal Quantile Regression) on all ablation variants ---
# The grouped, repeated version (whole benchmarks drawn at random into the calibration and test
# halves, 100 splits): the one the revised manuscript reports (`tables/cqr_grouped`). The
# positional split of the first submission, `bf.conformal_prediction_coverage`, is kept as an
# alternative and not run here.
cqr_results = {}
for name, (_idata_abl, _cfg, idata_retro_abl) in ablation_idata.items():
    print(f"\n=== CQR (grouped, 100 benchmark splits) for {name} ===")
    g = bf.conformal_prediction_coverage_grouped(idata_retro_abl, alpha=0.20, n_repeats=100, seed=0)
    cqr_results[f"cqr_grouped_{name}_80"] = g
    print(f"    Bayesian coverage (all held-out): {g['bayesian_coverage_all']:.1%}")
    print(f"    CQR coverage: median {g['cqr_coverage']['median']:.1%} "
          f"[{g['cqr_coverage']['q25']:.1%}, {g['cqr_coverage']['q75']:.1%}]")
    print(f"    Q: median {g['cqr_Q']['median']:+.4f} [{g['cqr_Q']['q25']:+.4f}, {g['cqr_Q']['q75']:+.4f}]")

# Save all ablation + CQR results to JSON
with open(f"{SENS_DIR}/ablation_results{CUTOFF_SUFFIX}.json", "w") as f:
    json.dump({**{k: v for k, v in ablation_results.items()}, **cqr_results}, f, indent=2, default=str)
print(f"\nResults saved to {SENS_DIR}/ablation_results{CUTOFF_SUFFIX}.json")
