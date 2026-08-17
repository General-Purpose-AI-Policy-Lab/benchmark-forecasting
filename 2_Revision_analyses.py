# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: pymc_env
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Robustness and sensitivity analyses
#
# Supplementary analyses probing how far the headline projections depend on the
# modelling choices.  Item numbers are the internal identifiers used in the
# project notes.
#
# 1. Saturation dates per benchmark and shifts across the 8 model variants
# 2. Long-horizon retrodiction (cutoffs 2022 / 2023 / 2024 / 2025)
# 5. Prior sensitivity on the upper asymptote `L`
# 6. Posterior figures: hyperparameters and per-benchmark `L`
# 7. Cross-benchmark residual dependence
# 20. Lower bounds: how many default to zero, and whether that is defensible
#
# Run stages selectively (each stage caches its fits in `Fits/`):
#
# ```bash
# uv run python 2_Revision_analyses.py cheap    # items 1, 6, 7, 20 — cached fits only
# uv run python 2_Revision_analyses.py figures  # redraw the 11 category panels — 1 cached fit
# uv run python 2_Revision_analyses.py retro    # item 2 — 3 new MCMC fits
# uv run python 2_Revision_analyses.py priors   # item 5 — 3 new MCMC fits
# uv run python 2_Revision_analyses.py retro8   # 8 variants at the 2025 cutoff — 8 new MCMC fits
# uv run python 2_Revision_analyses.py cqr      # grouped repeated CQR, 8 variants — 8 new MCMC fits
# ```
#
# Several stages can be combined in one call (`... cheap figures`).  Each stage writes
# `Plots/4-Sensitivity/revision_analyses_<stage>_<cutoff>.json`.

# %%
import json
import os
import sys

import arviz as az
from scipy.stats import energy_distance
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    _project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, _project_root)
    os.chdir(_project_root)
except NameError:
    pass

import forecasting
import plotting

STAGES = set(sys.argv[1:]) or {"cheap"}

DATA_PATH = "Data/benchmark_data_processed/all_normalized_updated_benchmarks.csv"
DATA_CUTOFF_DATE = pd.to_datetime("2026-04-01")
CUTOFF_TAG = f"cutoff{DATA_CUTOFF_DATE.strftime('%Y%m%d')}"

SATURATION_FRACTION = 0.95
SATURATION_TARGET_DATE = pd.Timestamp("2030-01-01")
END_DATE = pd.to_datetime("2030-03-01")

SAMPLING_CONFIG = forecasting.SamplingConfig(draws=2000, tune=1000, target_accept=0.9, seed=42, progressbar=True)

MAIN_MODEL = "Harvey Joint (skew)"
ALL_MODEL_CONFIGS = {
    "Harvey Joint (skew)": forecasting.ModelConfig(sigmoid="harvey", joint=True, top_n=3, skew=True),
    "Harvey Joint (normal)": forecasting.ModelConfig(sigmoid="harvey", joint=True, top_n=3, skew=False),
    "Harvey Independent (skew)": forecasting.ModelConfig(sigmoid="harvey", joint=False, top_n=3, skew=True),
    "Harvey Independent (normal)": forecasting.ModelConfig(sigmoid="harvey", joint=False, top_n=3, skew=False),
    "Logistic Joint (skew)": forecasting.ModelConfig(sigmoid="logistic", joint=True, top_n=3, skew=True),
    "Logistic Joint (normal)": forecasting.ModelConfig(sigmoid="logistic", joint=True, top_n=3, skew=False),
    "Logistic Independent (skew)": forecasting.ModelConfig(sigmoid="logistic", joint=False, top_n=3, skew=True),
    "Logistic Independent (normal)": forecasting.ModelConfig(sigmoid="logistic", joint=False, top_n=3, skew=False),
}

RESULTS_DIR = "Plots/4-Sensitivity"
# Manuscript sources live outside this repository; set TABLES_DIR to the
# manuscript's table directory to regenerate the LaTeX tables in place.
TABLES_DIR = os.environ.get("TABLES_DIR", f"{RESULTS_DIR}/tables")
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

paper_style = plotting.PlotStyle(language="en", document_type="paper")


def tex_escape(s: str) -> str:
    return str(s).replace("&", r"\&").replace("_", r"\_").replace("%", r"\%").replace("#", r"\#")


def fmt_date(ts) -> str:
    ts = pd.Timestamp(ts)
    if ts.year >= 2100:
        return ">2100"
    return ts.strftime("%Y-%m")


# %%
raw_all = forecasting.load_dataset(DATA_PATH)
raw = raw_all[raw_all["release_date"] < DATA_CUTOFF_DATE].copy()
data = forecasting.prepare_dataset(raw, top_n=3)
print(f"{data['benchmark'].nunique()} benchmarks, {len(data)} frontier observations "
      f"(cutoff {DATA_CUTOFF_DATE.date()})")

results: dict[str, object] = {}

# %% [markdown]
# ## Item 1 — Saturation dates and shifts across model variants
#
# `forecasting.saturation_dates` inverts the fitted sigmoid analytically, so the
# saturation date is available per posterior draw and per benchmark.  The quantity of
# interest is how much the skew-normal likelihood moves those dates.

# %%
if "cheap" in STAGES:
    sat_by_variant: dict[str, pd.DataFrame] = {}
    for name, cfg in ALL_MODEL_CONFIGS.items():
        idata, _model = forecasting.fit(data, cfg, SAMPLING_CONFIG, cache_tag=CUTOFF_TAG)
        sat_by_variant[name] = forecasting.saturation_dates(
            idata,
            prepared_frontier=data,
            saturation_fraction=SATURATION_FRACTION,
            target_date=SATURATION_TARGET_DATE,
            ci_level=0.80,
        ).set_index("benchmark")
        med = sat_by_variant[name]["sat_median"]
        print(f"{name:30s} median saturation date across benchmarks: {fmt_date(med.median())}")

    # --- shifts relative to the main model, in months ---
    ref = sat_by_variant[MAIN_MODEL]["sat_median_days"]
    shift_rows = []
    for name, df in sat_by_variant.items():
        shift_months = (df["sat_median_days"] - ref) / 30.44
        shift_rows.append({
            "variant": name,
            "median_sat_date": sat_by_variant[name]["sat_median"].median(),
            "mean_shift_months": float(shift_months.mean()),
            "median_shift_months": float(shift_months.median()),
            "q10_shift_months": float(shift_months.quantile(0.10)),
            "q90_shift_months": float(shift_months.quantile(0.90)),
        })
    shifts = pd.DataFrame(shift_rows).set_index("variant")
    print("\n=== Saturation-date shift vs main model (months; + = later) ===")
    print(shifts.to_string())

    # --- marginal contrasts: each modeling choice averaged over the other two ---
    def _mean_shift(from_variants: list[str], to_variants: list[str]) -> dict[str, float]:
        diffs = []
        for a, b in zip(from_variants, to_variants):
            diffs.append((sat_by_variant[b]["sat_median_days"] - sat_by_variant[a]["sat_median_days"]) / 30.44)
        d = pd.concat(diffs)
        return {"mean": float(d.mean()), "median": float(d.median()),
                "q10": float(d.quantile(0.10)), "q90": float(d.quantile(0.90))}

    contrasts = {
        "skew to normal": _mean_shift(
            ["Harvey Joint (skew)", "Harvey Independent (skew)", "Logistic Joint (skew)", "Logistic Independent (skew)"],
            ["Harvey Joint (normal)", "Harvey Independent (normal)", "Logistic Joint (normal)", "Logistic Independent (normal)"],
        ),
        "joint to independent": _mean_shift(
            ["Harvey Joint (skew)", "Harvey Joint (normal)", "Logistic Joint (skew)", "Logistic Joint (normal)"],
            ["Harvey Independent (skew)", "Harvey Independent (normal)", "Logistic Independent (skew)", "Logistic Independent (normal)"],
        ),
        "Harvey to logistic": _mean_shift(
            ["Harvey Joint (skew)", "Harvey Joint (normal)", "Harvey Independent (skew)", "Harvey Independent (normal)"],
            ["Logistic Joint (skew)", "Logistic Joint (normal)", "Logistic Independent (skew)", "Logistic Independent (normal)"],
        ),
    }
    print("\n=== Marginal contrasts (months; + = later saturation) ===")
    for k, v in contrasts.items():
        print(f"  {k:24s} mean={v['mean']:+.1f}  median={v['median']:+.1f}  "
              f"[q10={v['q10']:+.1f}, q90={v['q90']:+.1f}]")

    results["saturation_shifts"] = shifts.reset_index().to_dict(orient="records")
    results["saturation_contrasts"] = contrasts

    # --- simplified LaTeX table (goes in the sensitivity appendix) ---
    lines = [
        r"\begin{table}[ht]", r"\centering", r"\small",
        r"\begin{tabular}{lccc}", r"\toprule",
        r"Modeling choice & Mean shift & Median shift & 10--90\% range \\",
        r"\midrule",
    ]
    label = {"skew to normal": r"Skew-normal $\rightarrow$ normal likelihood",
             "joint to independent": r"Joint $\rightarrow$ independent",
             "Harvey to logistic": r"Harvey $\rightarrow$ logistic"}
    for k, v in contrasts.items():
        lines.append(
            f"{label[k]} & {v['mean']:+.1f} & {v['median']:+.1f} & "
            f"[{v['q10']:+.1f}, {v['q90']:+.1f}] \\\\"
        )
    lines += [
        r"\bottomrule", r"\end{tabular}",
        r"\caption{\revised{\textbf{Effect of each modeling choice on projected saturation dates}, in months "
        r"(positive means later saturation). Each row averages the contrast over the two remaining "
        r"factors and over all benchmarks, using the posterior median saturation date per benchmark. "
        r"Per-benchmark dates are reported in Supp.\ Table~\ref{tab:benchmark_details}.}}",
        r"\label{tab:saturation_shifts}", r"\end{table}",
    ]
    with open(f"{TABLES_DIR}/saturation_shifts.tex", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nWrote {TABLES_DIR}/saturation_shifts.tex")

# %% [markdown]
# ### Full per-benchmark table (separate file)
#
# Fills the placeholder left in the appendix: one row per benchmark with its lower
# bound, posterior asymptote, projected saturation date and probability of
# saturating by 2030.

# %%
if "cheap" in STAGES:
    idata_main, _ = forecasting.fit(data, ALL_MODEL_CONFIGS[MAIN_MODEL], SAMPLING_CONFIG, cache_tag=CUTOFF_TAG)

    L = idata_main.posterior["L"].stack(sample=("chain", "draw")).transpose("benchmark", "sample").to_numpy()
    L_bench = [str(b) for b in idata_main.posterior["L"].coords["benchmark"].to_numpy().tolist()]
    L_df = pd.DataFrame({
        "benchmark": L_bench,
        "L_median": np.median(L, axis=1),
        "L_lower": np.percentile(L, 10, axis=1),
        "L_upper": np.percentile(L, 90, axis=1),
    }).set_index("benchmark")

    detail = sat_by_variant[MAIN_MODEL].join(L_df)
    detail["lower_bound"] = data.groupby("benchmark")["lower_bound"].first()
    detail["n_obs"] = data.groupby("benchmark").size()
    detail["best_observed"] = data.groupby("benchmark")["score"].max()
    detail["source"] = data.groupby("benchmark")["source"].first()

    # Benchmarks carrying the "Tier 2 Excluded" category (assigned in notebook 0) belong
    # to none of the eleven capability categories, so they are absent from the category
    # summary but listed under "Uncategorized" in the per-benchmark table: they are in
    # the fits, hence in the aggregate proportion, and hiding them would leave the paper
    # claiming 63 benchmarks while tabulating 62.
    excluded = detail.index[detail["category"] == "Tier 2 Excluded"].tolist()
    if excluded:
        prop_all = forecasting.saturated_proportion(
            idata_main, prepared_frontier=data, target_date=SATURATION_TARGET_DATE,
            saturation_fraction=SATURATION_FRACTION, ci_level=0.80,
        )
        prop_kept = forecasting.saturated_proportion(
            idata_main, prepared_frontier=data, target_date=SATURATION_TARGET_DATE,
            saturation_fraction=SATURATION_FRACTION, ci_level=0.80,
            benchmarks=[b for b in detail.index if b not in excluded],
        )
        print(f"  uncategorized in the per-benchmark table: {excluded}")
        print(f"    saturated by 2030 with them    ({prop_all['n_benchmarks']} benchmarks): "
              f"{prop_all['median']:.1%} {prop_all['ci']}")  # type: ignore[str-format]
        print(f"    saturated by 2030 without them ({prop_kept['n_benchmarks']} benchmarks): "
              f"{prop_kept['median']:.1%} {prop_kept['ci']}")  # type: ignore[str-format]
        results["excluded_benchmarks"] = {
            "names": excluded,
            "proportion_with": {k: v for k, v in prop_all.items() if k != "benchmarks"},
            "proportion_without": {k: v for k, v in prop_kept.items() if k != "benchmarks"},
        }
        detail.loc[excluded, "category"] = "Uncategorized"

    detail = detail.sort_values(["category", "benchmark"])

    detail.to_csv(f"{RESULTS_DIR}/benchmark_details_{CUTOFF_TAG}.csv")

    # Rows are grouped under category headings rather than carrying a category column:
    # with the category spelled out on every row the table overflows the text width.
    header = (r"Benchmark & $n$ & $\ell$ & Best obs. & $L$ [80\% CI] & "
              r"Saturation date [80\% CI] \\")
    lines = [
        r"% Auto-generated by 2_Revision_analyses.py — do not edit by hand.",
        r"\begingroup\footnotesize",
        r"\begin{longtable}{@{}lccccl@{}}",
        r"\caption{\revised{\textbf{Per-benchmark summary.} Lower bound $\ell$ is random-chance performance where "
        r"it is well defined (see Appendix~\ref{app:lower_bounds}); $L$ is the posterior upper asymptote "
        r"(median and 80\% CI); the saturation date is when the posterior median trajectory reaches 95\% "
        r"of the score range, with its 80\% credible interval. $n$ counts frontier observations used for "
        r"fitting, and ``Best obs.'' is the highest score recorded up to April 2026. Rows are grouped by "
        r"capability category, with benchmarks belonging to none of the eleven categories listed last.}}"
        r"\label{tab:benchmark_details} \\",
        r"\toprule", header, r"\midrule", r"\endfirsthead",
        r"\multicolumn{6}{@{}l}{\footnotesize\itshape Table~\ref{tab:benchmark_details}, continued} \\",
        r"\toprule", header, r"\midrule", r"\endhead",
        r"\bottomrule", r"\endfoot",
    ]
    for cat, sub in detail.groupby("category", sort=True):
        lines.append(r"\addlinespace")
        lines.append(rf"\multicolumn{{6}}{{@{{}}l}}{{\itshape {tex_escape(cat)}}} \\")
        for bench, row in sub.iterrows():
            lines.append(
                f"{tex_escape(bench)} & {int(row['n_obs'])} & "
                f"{row['lower_bound']:.2f} & {row['best_observed']:.2f} & "
                f"{row['L_median']:.2f} [{row['L_lower']:.2f}, {row['L_upper']:.2f}] & "
                f"{fmt_date(row['sat_median'])} [{fmt_date(row['sat_lower'])}, {fmt_date(row['sat_upper'])}] \\\\"
            )
    lines += [r"\end{longtable}", r"\endgroup"]
    with open(f"{TABLES_DIR}/benchmark_details.tex", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {TABLES_DIR}/benchmark_details.tex ({len(detail)} benchmarks)")
    print(detail[["sat_median", "L_median", "p_saturated_by_target"]].to_string())

# %% [markdown]
# ### Category summary table
#
# Condenses the per-benchmark detail into one row per capability category; the
# per-benchmark numbers are tabulated in the appendix table instead.

# %%
if "cheap" in STAGES:
    CATEGORY_ORDER = [
        "Domain Specific Questions", "General Reasoning", "High End Math Reasoning",
        "Core AGI Progress", "Agentic Computer Use", "Autonomous SWE",
        "Biology", "Chemistry", "Commonsense QA",
        "Advanced Language and Writing", "Multimodal Understanding",
    ]
    baselines_csv = pd.read_csv("Data/human_baselines.csv")

    cat_rows = []
    for cat in CATEGORY_ORDER:
        sub = detail.loc[detail["category"] == cat]
        if sub.empty:
            print(f"  (no benchmarks in category {cat!r})")
            continue
        bl_sub = baselines_csv.loc[baselines_csv["benchmark"].isin(sub.index)]
        # Every recorded human baseline for the category, across expertise levels.
        # The ranges therefore span different populations (crowdworkers through expert
        # committees), which the caption states so they are not read as disagreement
        # between measurements of the same thing.
        human = bl_sub["score"]
        if len(human) == 0:
            human_cell = "---"
        elif human.min() == human.max():
            human_cell = f"{human.max() * 100:.0f}\\%"
        else:
            human_cell = f"{human.min() * 100:.0f}--{human.max() * 100:.0f}\\%"
        cat_rows.append({
            "category": cat,
            "n": len(sub),
            "n_with_baseline": sub.index.isin(baselines_csv["benchmark"]).sum(),
            "n_baselines": len(human),
            "groups": ", ".join(sorted(bl_sub["group"].unique())),
            "human_range": human_cell,
            "sat_median": pd.Timestamp(sub["sat_median"].median()),
            "sat_earliest": pd.Timestamp(sub["sat_median"].min()),
            "sat_latest": pd.Timestamp(sub["sat_median"].max()),
        })
    cats = pd.DataFrame(cat_rows).sort_values("sat_latest").reset_index(drop=True)
    print("\n=== Category summary (sorted by latest saturation date) ===")
    print(cats.to_string(index=False))

    lines = [
        r"\begin{table}[ht]", r"\centering", r"\small",
        r"\begin{tabular}{lcccc}", r"\toprule",
        r" & & Human & \multicolumn{2}{c}{Projected saturation} \\",
        r"\cmidrule(lr){4-5}",
        r"Category & $n$ & baseline & Earliest & Latest \\",
        r"\midrule",
    ]
    for _, r in cats.iterrows():
        lines.append(
            f"{tex_escape(r['category'])} & {int(r['n'])} & {r['human_range']} & "
            f"{fmt_date(r['sat_earliest'])} & {fmt_date(r['sat_latest'])} \\\\"
        )
    lines += [
        r"\bottomrule", r"\end{tabular}",
        r"\caption{\revised{\textbf{Projected saturation by capability category}, ordered by the latest date "
        r"within the category. "
        r"$n$ is the number of benchmarks in the category. The human-baseline column spans every human "
        r"baseline recorded for the category's benchmarks, from crowdworkers to domain experts and expert "
        r"committees, so a wide range reflects that mixture of populations rather than disagreement "
        r"between measurements of the same quantity; Appendix~\ref{app:human_baselines} lists each "
        r"baseline with its expertise level, and a dash means that no human baseline is available for any "
        r"benchmark in the category. The last two columns give the earliest and latest posterior median "
        r"saturation date among the category's benchmarks. Benchmarks are "
        r"listed individually in Supp.\ Table~\ref{tab:benchmark_details} and described in "
        r"Appendix~\ref{app:benchmarks}.}}",
        r"\label{tab:category_summary}", r"\end{table}",
    ]
    with open(f"{TABLES_DIR}/category_summary.tex", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {TABLES_DIR}/category_summary.tex")
    results["category_summary"] = cats.to_dict(orient="records")

# %% [markdown]
# ## Item 6 — Posterior figures: hyperparameters and per-benchmark asymptote

# %%
if "cheap" in STAGES:
    fig, _ = plotting.plot_hyperparameters(idata_main, plot_style=paper_style)
    fig.savefig(f"Plots/1-High_level/hyperparameters_en_paper_{CUTOFF_TAG}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, _ = plotting.plot_L_intervals(idata_main, prepared_frontier=data, plot_style=paper_style)
    fig.savefig(f"Plots/1-High_level/L_intervals_en_paper_{CUTOFF_TAG}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Wrote hyperparameter and L-interval figures to Plots/1-High_level/")

    # Numbers quoted in the appendix text
    post = idata_main.posterior
    for name, transform in [("L_raw_mu", lambda x: 0.75 + 0.25 * x), ("k_mu", lambda x: x),
                            ("alpha_raw_mu", lambda x: x + 1.0), ("s_mu", lambda x: x)]:
        if name in post:
            v = transform(post[name].values.flatten())
            print(f"  {name:12s} median={np.median(v):.4g}  80% CI=[{np.percentile(v, 10):.4g}, {np.percentile(v, 90):.4g}]")
    Lm = np.median(L, axis=1)
    print(f"  Per-benchmark L: median={np.median(Lm):.3f}, 10th-90th pct=[{np.percentile(Lm, 10):.3f}, {np.percentile(Lm, 90):.3f}], "
          f"min={Lm.min():.3f} ({L_bench[int(np.argmin(Lm))]})")
    results["L_summary"] = {
        "median": float(np.median(Lm)), "p10": float(np.percentile(Lm, 10)),
        "p90": float(np.percentile(Lm, 90)), "min": float(Lm.min()),
        "argmin": L_bench[int(np.argmin(Lm))],
    }

# %% [markdown]
# ## Item 7 — Cross-benchmark residual dependence
#
# The model treats benchmarks as conditionally independent, yet many share datasets,
# tasks, or creators.  Residuals are paired by model release, so a
# shared unmodeled component would show up as positive correlation between
# benchmarks of the same lineage.

# %%
LINEAGES = {
    "GPQA Diamond": "GPQA Diamond", "GPQA Diamond Biology": "GPQA Diamond",
    "GPQA Diamond Chemistry": "GPQA Diamond",
    "MMLU Pro Biology": "MMLU Pro", "MMLU Pro Chemistry": "MMLU Pro",
    "LAB-Bench Cloning": "LAB-Bench", "LAB-Bench LitQA2": "LAB-Bench",
    "LAB-Bench Protocol": "LAB-Bench", "LAB-Bench SeqQA": "LAB-Bench",
    "SWE-Bench Verified": "SWE-Bench", "SWE-Bench Pro": "SWE-Bench",
    "SWE-Bench Bash Only": "SWE-Bench",
    "ARC-AGI": "ARC-AGI", "ARC-AGI-2": "ARC-AGI",
    "FrontierMath": "FrontierMath", "FrontierMath Tier 4": "FrontierMath",
    "WMDP Biology": "WMDP", "WMDP Chemistry": "WMDP",
    "MultiChallenge": "MultiChallenge", "AudioMultiChallenge": "MultiChallenge",
    "PRBench Legal": "PRBench", "PRBench Finance": "PRBench",
}

# %%
if "cheap" in STAGES:
    diag = forecasting.residual_diagnostics(idata_main, data, lineages=LINEAGES, min_shared_models=5)
    print(f"Pairs with >= 5 shared model releases: {diag['n_pairs']} "
          f"(of which {diag['n_pairs_same_lineage']} share a lineage), "
          f"{diag['n_models']} distinct models")
    print(f"  mean residual correlation, same lineage : {diag['mean_corr_same_lineage']:+.3f} "
          f"(median {diag['median_corr_same_lineage']:+.3f})")
    print(f"  mean residual correlation, across       : {diag['mean_corr_across_lineage']:+.3f} "
          f"(median {diag['median_corr_across_lineage']:+.3f})")
    print(f"  residual variance explained by model identity (R^2): {diag['r2_model_identity']:.3f}")

    pairs = diag["pairs"]
    assert isinstance(pairs, pd.DataFrame)
    pairs.to_csv(f"{RESULTS_DIR}/residual_pair_correlations_{CUTOFF_TAG}.csv", index=False)
    print("\n  Same-lineage pairs:")
    print(pairs.loc[pairs["same_lineage"]].sort_values("corr", ascending=False)
          [["benchmark_1", "benchmark_2", "n_shared_models", "corr"]].to_string(index=False))

    # Does the residual correlation translate into a bias on the projections?
    # Compare the mean residual per model release with zero: a systematic
    # "strong release" effect would bias the frontier upward or downward.
    resid = diag["residuals"]
    assert isinstance(resid, pd.DataFrame)
    per_model = resid.groupby("model_version")["resid"].agg(["mean", "size"])
    per_model = per_model.loc[per_model["size"] >= 3]
    print(f"\n  Models evaluated on >= 3 benchmarks: {len(per_model)}; "
          f"mean of per-model mean residual = {per_model['mean'].mean():+.4f} "
          f"(sd {per_model['mean'].std():.4f})")

    results["residual_diagnostics"] = {
        k: v for k, v in diag.items() if not isinstance(v, pd.DataFrame)
    }
    results["residual_diagnostics"]["per_model_mean_resid_mean"] = float(per_model["mean"].mean())
    results["residual_diagnostics"]["per_model_mean_resid_sd"] = float(per_model["mean"].std())

    # --- Consequence of the dependence: collapse each lineage to one representative ---
    # If correlated benchmarks were inflating the headline proportion by being counted
    # several times, keeping one benchmark per lineage would lower it.
    all_bench = sorted(data["benchmark"].unique())
    seen_lineages: set[str] = set()
    collapsed: list[str] = []
    for b in all_bench:
        lin = LINEAGES.get(b)
        if lin is None:
            collapsed.append(b)
        elif lin not in seen_lineages:
            seen_lineages.add(lin)
            collapsed.append(b)  # first benchmark of the lineage, alphabetically

    full_prop = forecasting.saturated_proportion(
        idata_main, prepared_frontier=data, target_date=SATURATION_TARGET_DATE,
        saturation_fraction=SATURATION_FRACTION, ci_level=0.80,
    )
    coll_prop = forecasting.saturated_proportion(
        idata_main, prepared_frontier=data, target_date=SATURATION_TARGET_DATE,
        saturation_fraction=SATURATION_FRACTION, ci_level=0.80, benchmarks=collapsed,
    )
    print(f"\n  Saturated by 2030, all {full_prop['n_benchmarks']} benchmarks: "
          f"{full_prop['median']:.1%} {full_prop['ci']}")  # type: ignore[str-format]
    print(f"  Saturated by 2030, {coll_prop['n_benchmarks']} lineage representatives: "
          f"{coll_prop['median']:.1%} {coll_prop['ci']}")  # type: ignore[str-format]
    results["lineage_collapsed_saturation"] = {
        "full": {k: v for k, v in full_prop.items() if k != "benchmarks"},
        "collapsed": {k: v for k, v in coll_prop.items() if k != "benchmarks"},
        "collapsed_benchmarks": collapsed,
    }

# %% [markdown]
# ## Item 20 — Lower bounds
#
# Random chance is not 0 for many multiple-choice or structured benchmarks, so it
# matters how many benchmarks fall back on the zero default.
# We check the design of every benchmark set to zero and the weakest score ever
# observed on it.

# %%
if "cheap" in STAGES:
    lb = data.groupby("benchmark")["lower_bound"].first()
    zero_bench = lb[lb == 0].index.tolist()
    min_obs_all = raw.groupby("benchmark")["score"].min()

    print(f"lower_bound == 0: {len(zero_bench)}/{len(lb)} benchmarks")
    print(f"lower_bound  > 0: {(lb > 0).sum()}/{len(lb)} benchmarks (manually gathered)")
    print("\nNon-zero lower bounds:")
    print(lb[lb > 0].sort_values().to_string())

    zero_tab = pd.DataFrame({
        "min_observed": min_obs_all.reindex(zero_bench),
        "category": data.groupby("benchmark")["category"].first().reindex(zero_bench),
    }).sort_values("min_observed")
    print("\nBenchmarks with a zero lower bound, by weakest score ever observed:")
    print(zero_tab.to_string())

    # If the true floor were as high as the weakest observed score, the 95%-of-range
    # threshold would move by 5% of that floor.
    worst_shift = 0.05 * zero_tab["min_observed"].max()
    print(f"\nUpper bound on the induced threshold error: 5% x max(min_observed) = "
          f"{worst_shift:.3f} ({worst_shift * 100:.1f} percentage points), "
          f"for {zero_tab['min_observed'].idxmax()}")
    print(f"Benchmarks whose weakest observed score is below 10%: "
          f"{(zero_tab['min_observed'] < 0.10).sum()}/{len(zero_tab)}")

    results["lower_bounds"] = {
        "n_zero": int(len(zero_bench)),
        "n_nonzero": int((lb > 0).sum()),
        "n_zero_with_min_obs_below_10pct": int((zero_tab["min_observed"] < 0.10).sum()),
        "max_min_observed": float(zero_tab["min_observed"].max()),
        "max_min_observed_benchmark": str(zero_tab["min_observed"].idxmax()),
        "max_threshold_error": float(worst_shift),
        "nonzero_bounds": {str(k): float(v) for k, v in lb[lb > 0].items()},
        "zero_bound_min_observed": {str(k): float(v) for k, v in zero_tab["min_observed"].items()},
    }

# %% [markdown]
# ## Item 2 — Long-horizon retrodiction
#
# The submitted validation trains before 2025-01-01 and tests to 2026-04-01, i.e. a
# ~15-month horizon, while the headline claim spans four years.  We push the cutoff
# back to 2024, 2023 and 2022.  The retrospective filter (>= 3 pre-cutoff frontier
# observations) removes any benchmark that barely existed at the time, so the
# earlier cutoffs are evaluated on progressively fewer benchmarks — a limitation of
# the exercise that we report rather than hide.

# %%
RETRO_CUTOFFS = ["2022-01-01", "2023-01-01", "2024-01-01", "2025-01-01"]
MIN_TRAIN_POINTS = 5  # minimum pre-cutoff frontier observations for a benchmark to be evaluated

if "retro" in STAGES:
    retro_rows = []
    for cutoff in RETRO_CUTOFFS:
        c = pd.to_datetime(cutoff)
        prepared = forecasting.prepare_dataset(raw, top_n=3)
        train = prepared.loc[prepared["release_date"] < c]
        counts = train.groupby("benchmark")["score"].size()
        keep = counts[counts >= MIN_TRAIN_POINTS].index
        n_bench_kept = len(keep)

        print(f"\n=== Retrodiction cutoff {cutoff} — {n_bench_kept} benchmarks kept ===")
        idata_retro = forecasting.temporal_holdout(
            raw, cutoff_date=c, cfg=ALL_MODEL_CONFIGS[MAIN_MODEL],
            samp=SAMPLING_CONFIG, min_train_points=MIN_TRAIN_POINTS,
        )

        y_pred = idata_retro.predictions.stack(sample=("chain", "draw"))["y"].to_numpy()
        y_true = idata_retro.predictions["y_true"].to_numpy()
        lo = np.quantile(y_pred, 0.10, axis=1)
        hi = np.quantile(y_pred, 0.90, axis=1)
        coverage80 = float(np.mean((y_true >= lo) & (y_true <= hi)))

        crps = forecasting.crps_score(idata_retro)
        rmse = forecasting.point_error(idata_retro, metric="RMSE")
        horizon = (DATA_CUTOFF_DATE - c).days / 365.25

        # Signed error: positive means the observed score came in above the forecast,
        # i.e. progress was faster than the model predicted.  This is what decides
        # whether degraded coverage makes our projections optimistic or conservative.
        y_mean = y_pred.mean(axis=1)
        bias = float(np.mean(y_true - y_mean))
        frac_above = float(np.mean(y_true > hi))
        frac_below = float(np.mean(y_true < lo))
        rhat_max = float(np.nanmax(az.rhat(idata_retro).to_array().max().to_numpy()))

        row = {
            "cutoff": cutoff, "horizon_years": horizon, "n_benchmarks": n_bench_kept,
            "n_train": int(len(train.loc[train["benchmark"].isin(keep)])),
            "n_test": int(len(y_true)), "crps": crps, "rmse": rmse, "coverage80": coverage80,
            "bias": bias, "frac_above_interval": frac_above, "frac_below_interval": frac_below,
            "rhat_max": rhat_max,
        }
        retro_rows.append(row)
        print(f"  horizon={horizon:.1f} yr, n_test={row['n_test']}, "
              f"CRPS={crps:.4f}, RMSE={rmse:.4f}, 80% coverage={coverage80:.1%}")
        print(f"  signed error (obs - pred) = {bias:+.4f}; test points above the 80% interval: "
              f"{frac_above:.1%}, below: {frac_below:.1%}; max R-hat = {rhat_max:.3f}")

        fig, _ = plotting.plot_calibration_curve(idata_retro, n_points=20, plot_style=paper_style)
        fig.savefig(
            f"Plots/3-Calibration/calibration_harvey_joint_skew_en_paper_retro{c.strftime('%Y%m%d')}"
            f"{'' if MIN_TRAIN_POINTS == 3 else f'_min{MIN_TRAIN_POINTS}'}.pdf",
            dpi=300, bbox_inches="tight",
        )
        plt.close(fig)

    retro = pd.DataFrame(retro_rows)
    print("\n=== Long-horizon retrodiction summary ===")
    print(retro.to_string(index=False))
    retro.to_csv(f"{RESULTS_DIR}/retrodiction_horizons_min{MIN_TRAIN_POINTS}_{CUTOFF_TAG}.csv", index=False)

    lines = [
        r"\begin{table}[ht]", r"\centering", r"\small",
        r"\begin{tabular}{@{}lccccccccc@{}}", r"\toprule",
        r"Train cutoff & Horizon & Benchmarks & $n_{\text{train}}$ & $n_{\text{test}}$ & CRPS & RMSE & 80\% cov. & Above & Below \\",
        r"\midrule",
    ]
    for _, r in retro.iterrows():
        lines.append(
            f"{r['cutoff'][:7]} & {r['horizon_years']:.1f} yr & {int(r['n_benchmarks'])} & "
            f"{int(r['n_train'])} & {int(r['n_test'])} & {r['crps']:.3f} & {r['rmse']:.3f} & "
            f"{r['coverage80'] * 100:.1f}\\% & {r['frac_above_interval'] * 100:.1f}\\% & "
            f"{r['frac_below_interval'] * 100:.1f}\\% \\\\"
        )
    lines += [
        r"\bottomrule", r"\end{tabular}",
        r"\caption{\revised{\textbf{Retrodiction accuracy as a function of forecast horizon} for the main model "
        r"(Harvey joint, skew-normal). Each row trains on frontier scores released before the cutoff and "
        rf"predicts every score observed between the cutoff and April 2026. The retrospective filter keeps "
        rf"only benchmarks with at least {MIN_TRAIN_POINTS} pre-cutoff frontier observations, which is why the earlier "
        r"cutoffs cover far fewer benchmarks: at the 2022 and 2023 cutoffs only the commonsense and "
        r"early question-answering sets existed. Nominal coverage is 80\%; the last two columns split the "
        r"remaining observations into those falling above the upper bound of the 80\% credible interval and "
        r"those falling below its lower bound, so a large asymmetry means the model erred in one "
        r"direction.}}",
        r"\label{tab:retro_horizons}", r"\end{table}",
    ]
    with open(f"{TABLES_DIR}/retrodiction_horizons.tex", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {TABLES_DIR}/retrodiction_horizons.tex")
    results["retrodiction_horizons"] = retro.to_dict(orient="records")

# %% [markdown]
# ### All 8 variants at the main cutoff
#
# Table 7 of the paper (CQR, CRPS, RMSE, coverage) and the calibration curves are built
# from the 2025 holdout, so they depend on the retrospective filter too.

# %%
if "retro8" in STAGES:
    variant_rows = []
    for name, cfg in ALL_MODEL_CONFIGS.items():
        print(f"\n=== {name} (cutoff 2025-01-01, min_train_points={MIN_TRAIN_POINTS}) ===")
        idata_v = forecasting.temporal_holdout(
            raw, cutoff_date=pd.to_datetime("2025-01-01"), cfg=cfg,
            samp=SAMPLING_CONFIG, min_train_points=MIN_TRAIN_POINTS,
        )
        y_pred = idata_v.predictions.stack(sample=("chain", "draw"))["y"].to_numpy()
        y_true = idata_v.predictions["y_true"].to_numpy()
        lo, hi = np.quantile(y_pred, 0.10, axis=1), np.quantile(y_pred, 0.90, axis=1)
        cqr = forecasting.conformal_prediction_coverage(idata_v, alpha=0.20)
        row = {
            "variant": name, "n_test": int(len(y_true)),
            "coverage80": float(np.mean((y_true >= lo) & (y_true <= hi))),
            "cqr_coverage": cqr["cqr_coverage"], "cqr_Q": cqr["cqr_Q"],
            "crps": forecasting.crps_score(idata_v),
            "rmse": forecasting.point_error(idata_v, metric="RMSE"),
        }
        variant_rows.append(row)
        print(f"  n_test={row['n_test']}, coverage={row['coverage80']:.1%}, "
              f"CQR cov={row['cqr_coverage']:.1%}, Q={row['cqr_Q']:+.4f}, "
              f"CRPS={row['crps']:.4f}, RMSE={row['rmse']:.4f}")

        slug = name.lower().replace(" (", "_").replace(")", "").replace(" ", "_")
        fig, _ = plotting.plot_calibration_curve(idata_v, n_points=20, plot_style=paper_style)
        fig.savefig(f"Plots/3-Calibration/calibration_{slug}_en_paper_{CUTOFF_TAG}.pdf",
                    dpi=300, bbox_inches="tight")
        plt.close(fig)

    variants = pd.DataFrame(variant_rows)
    print("\n=== All 8 variants, 2025 holdout ===")
    print(variants.to_string(index=False))
    variants.to_csv(f"{RESULTS_DIR}/retrodiction_variants_min{MIN_TRAIN_POINTS}_{CUTOFF_TAG}.csv", index=False)
    results["retrodiction_variants"] = variants.to_dict(orient="records")

# %% [markdown]
# ### Grouped, repeated CQR at the main cutoff
#
# Replaces the positional calibration/test split of Table 7, which put one set of
# benchmarks in each half and so had no finite-sample guarantee.  Whole benchmarks are
# now assigned at random, 100 times, so Q answers a question the coverage column does
# not already answer: calibrating on some benchmarks, do the intervals cover on others?

# %%
if "cqr" in STAGES:
    cqr_rows = []
    for name, cfg in ALL_MODEL_CONFIGS.items():
        idata_c = forecasting.temporal_holdout(
            raw, cutoff_date=pd.to_datetime("2025-01-01"), cfg=cfg,
            samp=SAMPLING_CONFIG, min_train_points=MIN_TRAIN_POINTS,
        )
        g = forecasting.conformal_prediction_coverage_grouped(idata_c, alpha=0.20, n_repeats=100, seed=0)
        row = {
            "variant": name,
            "n_obs": g["n_obs"], "n_benchmarks": g["n_benchmarks"],
            "bayes_cov_all": g["bayesian_coverage_all"],
            "cqr_cov_median": g["cqr_coverage"]["median"],           # type: ignore[index]
            "cqr_cov_q25": g["cqr_coverage"]["q25"],                 # type: ignore[index]
            "cqr_cov_q75": g["cqr_coverage"]["q75"],                 # type: ignore[index]
            "Q_median": g["cqr_Q"]["median"],                        # type: ignore[index]
            "Q_q25": g["cqr_Q"]["q25"], "Q_q75": g["cqr_Q"]["q75"],  # type: ignore[index]
            "crps": forecasting.crps_score(idata_c),
            "rmse": forecasting.point_error(idata_c, metric="RMSE"),
        }
        # per-observation scores, so that "indistinguishable from the best" can be a
        # paired comparison rather than an eyeballed tolerance
        yp = idata_c.predictions.stack(sample=("chain", "draw"))["y"].to_numpy()
        yt = idata_c.predictions["y_true"].to_numpy()
        row["_crps_obs"] = np.array([energy_distance(p, (t,)) ** 2 / 2 for p, t in zip(yp, yt)])
        row["_sq_err_obs"] = (yp.mean(axis=1) - yt) ** 2
        cqr_rows.append(row)
        print(f"{name:30s} Bayes cov (all)={row['bayes_cov_all']:.1%}  "
              f"CQR cov={row['cqr_cov_median']:.1%} [{row['cqr_cov_q25']:.1%}, {row['cqr_cov_q75']:.1%}]  "
              f"Q={row['Q_median']:+.4f} [{row['Q_q25']:+.4f}, {row['Q_q75']:+.4f}]")

    cqr = pd.DataFrame(cqr_rows)
    cqr.to_csv(f"{RESULTS_DIR}/cqr_grouped_{CUTOFF_TAG}.csv", index=False)
    results["cqr_grouped"] = cqr.to_dict(orient="records")

    # bold what a paired comparison cannot separate from the best variant
    def _indistinguishable(key: str) -> list[bool]:
        obs = [r[key] for r in cqr_rows]
        means = np.array([o.mean() for o in obs])
        best = int(np.argmin(means))
        out = []
        for i, o in enumerate(obs):
            d = o - obs[best]
            se = d.std(ddof=1) / np.sqrt(len(d))
            out.append(bool(i == best or (se > 0 and abs(d.mean()) / se < 1.96)))
        return out

    crps_bold = _indistinguishable("_crps_obs")
    rmse_bold = _indistinguishable("_sq_err_obs")
    bf = lambda v, b: (r"\textbf{" + v + "}") if b else v

    lines = [
        r"\begin{table}[ht]", r"\centering", r"\small",
        r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
        r"Model variant & CRPS & RMSE & $Q$ & Bayes.\ cov. & CQR cov. \\",
        r"\midrule",
    ]
    for i, (_, r) in enumerate(cqr.iterrows()):
        label = r["variant"].replace("Independent", "indep.").replace("Joint", "joint")
        crps_s = bf(format(r["crps"], ".3f"), crps_bold[i])
        rmse_s = bf(format(r["rmse"], ".3f"), rmse_bold[i])
        q_s = "$" + format(r["Q_median"], "+.3f") + "$"
        cov_s = format(r["bayes_cov_all"] * 100, ".1f")
        cqr_s = format(r["cqr_cov_median"] * 100, ".1f")
        lines.append(
            f"{label} & {crps_s} & {rmse_s} & {q_s} & {cov_s}\\% & {cqr_s}\\% \\\\"
        )
    lines += [
        r"\bottomrule", r"\end{tabular}",
        r"\caption{\revised{\textbf{Held-out accuracy and distribution-free calibration.} CRPS and RMSE "
        r"are out-of-sample scores on the 2025 holdout (lower is better); bold marks the variants a paired "
        r"comparison cannot separate from the best. $Q$ is the Conformalized Quantile Regression "
        r"adjustment~\citep{romano_conformalized_2019} that would be needed for a finite-sample coverage "
        r"guarantee. Bayesian coverage is the fraction of all "
        r"held-out observations inside the nominal 80\% credible interval, and CQR cov.\ the coverage of the "
        r"adjusted intervals. Calibration and test halves are formed by assigning whole benchmarks at random, "
        r"so the quantity measured is whether intervals calibrated on some benchmarks cover on others; $Q$ "
        r"and CQR cov.\ are medians over 100 random assignments.}}",
        r"\label{tab:cqr_results}", r"\end{table}",
    ]
    with open(f"{TABLES_DIR}/cqr_grouped.tex", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nWrote {TABLES_DIR}/cqr_grouped.tex")

# %% [markdown]
# ## Item 5 — Prior sensitivity on the upper asymptote
#
# The saturation threshold is defined relative to the estimated asymptote $L$, whose
# prior is deliberately informative ($L_{\min} = 0.75$, $\mu_L$ centred on 0.96 with
# sd 0.02).  The question is whether that prior drives the result, especially for
# data-sparse benchmarks.  We refit with a much lower floor, a much weaker prior, and
# both at once.

# %%
# The Beta hyperprior on the asymptote only exists for sd < sqrt(mu(1-mu)) on the
# rescaled axis, which caps sd at 0.092 when L_min = 0.75 and at 0.135 when
# L_min = 0.50.  The widths below are the largest round values inside those caps.
PRIOR_VARIANTS = {
    "Main ($L_{\\min}{=}0.75$, sd 0.02)": forecasting.ModelConfig(sigmoid="harvey", joint=True, top_n=3, skew=True),
    "Low floor ($L_{\\min}{=}0.50$)": forecasting.ModelConfig(sigmoid="harvey", joint=True, top_n=3, skew=True, L_min=0.50),
    "Weak prior (sd 0.05)": forecasting.ModelConfig(sigmoid="harvey", joint=True, top_n=3, skew=True, L_prior_sd=0.05),
    "Low floor + weak prior": forecasting.ModelConfig(sigmoid="harvey", joint=True, top_n=3, skew=True, L_min=0.50, L_prior_sd=0.10),
}

if "priors" in STAGES:
    prior_rows = []
    for name, cfg in PRIOR_VARIANTS.items():
        print(f"\n=== {name} (slug: {cfg.slug}) ===")
        idata_p, _ = forecasting.fit(data, cfg, SAMPLING_CONFIG, cache_tag=CUTOFF_TAG)

        _, _, sat = plotting.plot_saturation_proportion_posterior(
            idata_p, prepared_frontier=data, target_date=SATURATION_TARGET_DATE,
            saturation_fraction=SATURATION_FRACTION, ci_level=0.80, plot_style=paper_style,
        )
        plt.close("all")

        Lp = idata_p.posterior["L"].stack(sample=("chain", "draw")).transpose("benchmark", "sample").to_numpy()
        L_med = np.median(Lp, axis=1)
        sat_dates = forecasting.saturation_dates(
            idata_p, prepared_frontier=data, saturation_fraction=SATURATION_FRACTION,
            target_date=SATURATION_TARGET_DATE, ci_level=0.80,
        )

        # A widened prior can degrade sampling, so the diagnostics belong in the table.
        rhat_max = float(np.nanmax(az.rhat(idata_p).to_array().max().to_numpy()))
        ess_min = float(np.nanmin(az.ess(idata_p).to_array().min().to_numpy()))
        n_div = int(idata_p.sample_stats["diverging"].sum()) if "diverging" in idata_p.sample_stats else -1

        row = {
            "variant": name, "L_min": cfg.L_min, "L_prior_sd": cfg.L_prior_sd,
            "sat_median": float(sat["median"]),  # type: ignore[arg-type]
            "sat_ci_low": float(sat["ci"][0]), "sat_ci_high": float(sat["ci"][1]),  # type: ignore[index]
            "L_median_across_benchmarks": float(np.median(L_med)),
            "L_p10": float(np.percentile(L_med, 10)), "L_p90": float(np.percentile(L_med, 90)),
            "median_sat_date": str(pd.Timestamp(sat_dates["sat_median"].median()).date()),
            "rhat_max": rhat_max, "ess_min": ess_min, "n_divergences": n_div,
        }
        print(f"  diagnostics: max R-hat={rhat_max:.3f}, min ESS={ess_min:.0f}, divergences={n_div}")
        prior_rows.append(row)
        print(f"  saturated by 2030: {row['sat_median']:.1%} "
              f"[{row['sat_ci_low']:.1%}, {row['sat_ci_high']:.1%}]")
        print(f"  L across benchmarks: median={row['L_median_across_benchmarks']:.3f} "
              f"[{row['L_p10']:.3f}, {row['L_p90']:.3f}], median saturation date {row['median_sat_date']}")

    priors = pd.DataFrame(prior_rows)
    print("\n=== Prior sensitivity summary ===")
    print(priors.to_string(index=False))
    priors.to_csv(f"{RESULTS_DIR}/prior_sensitivity_{CUTOFF_TAG}.csv", index=False)

    lines = [
        r"\begin{table}[ht]", r"\centering", r"\footnotesize",
        r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
        r"Prior on $L$ & Sat.\ by 2030 & 80\% CI & Median $L$ & $L$ 10--90\% & ESS / div. \\",
        r"\midrule",
    ]
    for _, r in priors.iterrows():
        lines.append(
            f"{r['variant']} & {r['sat_median'] * 100:.1f}\\% & "
            f"[{r['sat_ci_low'] * 100:.1f}, {r['sat_ci_high'] * 100:.1f}] & "
            f"{r['L_median_across_benchmarks']:.3f} & "
            f"[{r['L_p10']:.3f}, {r['L_p90']:.3f}] & "
            f"{int(r['ess_min'])} / {int(r['n_divergences'])} \\\\"
        )
    lines += [
        r"\bottomrule", r"\end{tabular}",
        r"\caption{\revised{\textbf{Sensitivity to the prior on the upper asymptote.} $L_{\min}$ is the hard floor "
        r"of the asymptote and sd is the standard deviation of the hyperprior on $\mu_L$. Relaxing the "
        r"prior does not lower the projected proportion saturated by 2030; it raises it slightly, because "
        r"weaker priors let some asymptotes fall, which narrows the score range a benchmark has to cover. "
        r"The last column reports the minimum effective sample size and the number of divergent "
        r"transitions: the most relaxed variant samples poorly and its figures should be read with that "
        r"in mind.}}",
        r"\label{tab:prior_sensitivity}", r"\end{table}",
    ]
    with open(f"{TABLES_DIR}/prior_sensitivity.tex", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {TABLES_DIR}/prior_sensitivity.tex")
    results["prior_sensitivity"] = priors.to_dict(orient="records")

# %% [markdown]
# ## Figure refresh after the ARC-AGI baseline fix
#
# `Data/human_baselines.csv` keyed the three ARC-AGI-1 baselines as `ARC-AGI-1`,
# while the processed dataset calls that benchmark `ARC-AGI`, and
# `plotting.plot_forecasts_by_category` joins baselines on the exact benchmark name.
# The 77% / 98% / 98% human reference points were therefore missing from the figure,
# although the paper's text and baseline table cite them.  The names now match, so the
# affected category figures have to be redrawn.
#
# Only the categories listed below are redrawn, with the same code path and parameters
# as `1_Forecasts.ipynb`.  Note that the notebook's own `DATA_CUTOFF_DATE` must be set
# to 2026-04-01 to reproduce the figures the paper uses.

# %%
FIGURE_CATEGORIES = sorted(c for c in [
    "Domain Specific Questions", "General Reasoning", "High End Math Reasoning", "Core AGI Progress",
    "Agentic Computer Use", "Autonomous SWE", "Biology", "Chemistry", "Commonsense QA",
    "Advanced Language and Writing", "Multimodal Understanding",
])  # every panel carries baseline markers, so all are redrawn

if "figures" in STAGES:
    idata_fig, model_fig = forecasting.fit(
        data, ALL_MODEL_CONFIGS[MAIN_MODEL], SAMPLING_CONFIG, cache_tag=CUTOFF_TAG
    )
    forecast_fig = forecasting.generate_forecast(
        idata_fig, model_fig, prepared_frontier=data,
        end_date=END_DATE, n_points=250, ci_level=0.8,
    )
    baselines = pd.read_csv("Data/human_baselines.csv")

    for style, ext in [
        (plotting.PlotStyle(language="en", document_type="paper"), "pdf"),
        (plotting.PlotStyle(language="fr", document_type="note"), "png"),
    ]:
        for cat in FIGURE_CATEGORIES:
            obs_cat = data.loc[data["category"] == cat]
            pred_cat = forecast_fig.loc[forecast_fig["category"] == cat]
            n_baselines = baselines["benchmark"].isin(obs_cat["benchmark"].unique()).sum()
            fig, _ = plotting.plot_forecasts_by_category(
                observed=obs_cat, forecast=pred_cat, baselines=baselines,
                end_date=END_DATE, category_name=cat, plot_style=style,
            )
            path = (f"Plots/2-Forecasts/forecast_{cat.replace(' ', '_')}"
                    f"_{style.language}_{style.document_type}_{CUTOFF_TAG}.{ext}")
            fig.savefig(path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"  wrote {path} ({n_baselines} baseline points in this category)")

# %%
out_path = f"{RESULTS_DIR}/revision_analyses_{'_'.join(sorted(STAGES))}_{CUTOFF_TAG}.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults written to {out_path}")
