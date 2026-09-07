# Benchmark saturation forecasting

Bayesian sigmoidal growth models for the frontier of AI benchmark scores: when does each benchmark saturate, and how much of the current set is saturated by 2030. The scores, release dates, categories, lower bounds, known ceilings and human baselines all come from [`benchmark-data-pipeline`](https://github.com/General-Purpose-AI-Policy-Lab/benchmark-data-pipeline); this repository only fits, validates and draws.

## Layout

Folders are numbered in processing order. `Plots/` and `Fits/` keep their historical names because the manuscripts in `Paper/` reference figure paths.

```
0_input/                        the pipeline's consumer views, copied by `python -m benchmark_forecasting sync`
  dated_scores_flat.csv         one row per (model, benchmark) score with a release date
  human_baselines.csv           one row per published human measurement
  build_manifest.json           the pipeline's manifest (schema, commit, feed checksums)
  provenance.json               which pipeline commit was copied, and when
1_model/benchmark_forecasting/  the package
  config.py                     paths, BLAS thread pin, ModelConfig and SamplingConfig
  data.py                       load_dataset, frontier selection, asymptote_bounds
  model.py                      build_model (PyMC)
  fit.py                        fit with the Fits/ cache, temporal_holdout
  evaluate.py                   CRPS, RMSE, conformal coverage, saturation dates, residual diagnostics
  forecast.py                   generate_forecast
  plotting.py                   every figure, EN paper and FR note styles
  sync.py                       copy the views from the pipeline checkout
1_forecasts.py                  main fit, forecasts, retrodiction, sensitivity analyses, all figures
2_revision_analyses.py          robustness analyses and LaTeX tables, in stages
Fits/                           cached posteriors (NetCDF, gitignored)
Plots/                          figures and JSON/CSV results, one subfolder per cutoff
Paper/                          bibliography and arXiv sources; other manuscript material stays local
tests/                          pytest: prior draws, synthetic posteriors, one 5-draw toy sampling
```

The two scripts are in jupytext percent format: run them with `python`, or open them as notebooks in Jupyter or VS Code.

## Data

`python -m benchmark_forecasting sync` copies `3_views/dated_scores_flat.csv`, `3_views/human_baselines.csv` and `2_database/build_manifest.json` from a `benchmark-data-pipeline` checkout (the sibling directory by default, `--pipeline PATH` otherwise) into `0_input/`, and records the pipeline commit in `provenance.json`. The copies are tracked, so a run is reproducible without the pipeline.

The pipeline decides what is fitted: which benchmarks are included and why (`0_input/metadata/benchmarks.csv` there, `excluded_benchmarks.csv` in its database), their category, lower bound (random-chance performance), known ceiling and human baselines. `config.EXCLUDED_BENCHMARKS` is a hook for exclusions specific to the forecasting model and is empty. Model names are the pipeline's `model_version`; `data.model_base_key` still folds reasoning efforts, thinking modes and parameter counts into one family per release day before the top-N frontier is taken.

`DATA_CUTOFF_DATE` in both scripts is the date the pipeline's feeds were refreshed. Scores released after it are dropped (inclusive), so a later sync does not silently change a run; the date also names the output folders and the fit caches.

## Model Details

Let $y_{i}(t)$ be the observed score for benchmark $i$ at time $t$.

The score is modeled as a sigmoidal growth curve $\mu_i(t)$ plus skewed heteroskedastic noise $\xi_i(t)$:

$$
y_{i}(t) \sim \text{SkewNormal}\big(\mu_i(t), \xi_i(t), s_i\big),
$$

where $s_i$ is the skewness parameter allowing asymmetric residuals. Negative values of $s_i$ (which the data strongly favors) mean benchmark scores fall predominantly below the latent optimal performance curve. When `skew=False` in the model configuration, a symmetric Normal likelihood is used instead.

### Sigmoidal growth curves

The sigmoidal curves model the latent mean performance $\mu_i(t)$ over time. Two families of sigmoids are implemented: a shifted logistic function, and a generalization allowing asymmetric growth, the Harvey curve.

The sigmoids are defined on the range $[\ell_i, L_i]$, where $\ell_i$ is a benchmark-specific lower bound (random-chance performance) and $L_i$ is the upper asymptote (final performance).

The lower bound $\ell_i$ is manually gathered per benchmark (or set to 0 if unknown). See `Data/benchmarks_lower_bounds.csv` for details. It is not necessarily 0, as some benchmarks may have non-zero random-chance performance (e.g. 25% for questions with 4 choices).

The upper bound $L_i$ is not necessarily 1, as benchmarks contain errors or inherent uncertainty that prevent perfect scores. It is estimated for most benchmarks, and pinned to 1 (`ModelConfig.L_fixed`) where the full range is known to be attainable: ARC-AGI, ARC-AGI-2, VPCT and EBR-bench (human baselines at or near 100%) and the two FrontierMath v2 sets (every problem has been solved at least once).

The latent mean performance on benchmark $i$ at time $t$ is then the shifted sigmoid:

$$
\mu_i(t) = \ell_i + (L_i - \ell_i) \sigma_i(t),
$$

where $\sigma_i(t) \in \left\\{ \sigma_i^{\text{log}}(t), \sigma_i^{\text{harv}}(t) \right\\}$ is the sigmoid function (Logistic or Harvey). We indicate with the exponent $\text{log}$ and $\text{harv}$ the two variants when necessary.

#### Logistic function

The logistic function is defined as:

$$
\sigma_i^{\text{log}}(t) = \frac{1}{1 + \exp\big(-k_i (t - \tau_i)\big)},
$$

where $k_i$ is the growth rate and $\tau_i$ is the inflection time.

#### Harvey function

The Harvey curve generalizes the logistic with a shape parameter $\alpha_i > 1$ that controls how sharply growth slows down (it reduces to the logistic function when $\alpha_i = 2$). It is defined as:

$$
\sigma_i^{\text{harv}}(t) = \left[1 - (1 - \alpha_i)\exp\big(-k_i (t - \tau_i)\big) \right]^{\frac{1}{1 - \alpha_i}} ,
$$

where $k_i$ is the growth-rate, $\tau_i$ is the inflection time and $\alpha_i > 1$ controls asymmetry (larger $\alpha_i$ gives earlier growth).

### Heteroskedastic noise

The observation noise $\xi_i(t)$ is heteroskedastic and approximately Beta-shaped over the interval $[\ell_i, L_i]$:

$$
\xi_i(t) = \xi_0 + \xi^{\text{base}}_i\frac{\sqrt{\big(\mu_i(t) - \ell_i\big)\big(L_i - \mu_i(t)\big)}}{(L_i - \ell_i)/2},
$$

peaking near the inflection point and shrinking near the bounds, where $\xi_0$ is a fixed parameter and $\xi^{\text{base}}_i$ is inferred per benchmark.

### Hierarchical (joint) models

The joint models define hierarchical versions where benchmarks share hyperpriors over parameters, allowing benchmarks to borrow statistical strength from each other while keeping benchmark-specific trajectories. When `joint=False`, each benchmark gets fully independent priors.

#### Upper asymptotes $L_i$:

Upper asymptotes $L_i$ are drawn from a Beta distribution shifted to $[L_{min}, 1]$:

$$
L_i = L_{min} + (1 - L_{min}) L^{\text{raw}}_i, \quad
L^{\text{raw}}_i \sim \text{Beta}(L^{\text{raw}}_{\mu}, L^{\text{raw}}_{\sigma}),
$$

where $L_{min} = 0.75$ and $L_{\mu}^{\text{raw}}, L_{\sigma}^{\text{raw}}$ are the mean and standard deviation hyperparameters, respectively (instead of the usual Beta parameters $\alpha, \beta$).

#### Growth rates $k_i$:

Growth rates $k_i$ follow a Gamma distribution:

$$
k_i \sim \text{Gamma}(k_{\mu}, k_{\sigma}),
$$

where $k_{\mu}, k_{\sigma}$ are the mean and standard deviation hyperparameters (instead of the usual Gamma shape and rate parameters $\alpha, \lambda$).

#### Inflection times $\tau_i$:

Inflection times $\tau_i$ follow a Gumbel distribution centered on empirical midpoint of each benchmark, with a scale of several years. The rationale is that for saturated benchmarks, the inflection point is roughly at the midpoint of observed data, and for unsaturated benchmarks, the inflection point is likely greater than the empirical midpoint.

#### Noise scales $\xi^{\text{base}}_i$:

Noise scales $\xi^{\text{base}}_i$ follow a Gamma distribution:

$$
\xi^{\text{base}}_i \sim \text{Gamma}(\xi^{\text{base}}_{\mu}, \xi^{\text{base}}_{\sigma}),
$$

where $\xi^{\text{base}}_ {\mu}, \xi^{\text{base}}_{\sigma}$ are the mean and standard deviation hyperparameters (instead of the usual Gamma shape and rate parameters $\alpha, \lambda$).

#### Skewness parameters $s_i$:

Skewness parameters $s_i$ follow a Normal distribution:

$$
s_i \sim \text{Normal}(s_{\mu}, s_{\sigma}),
$$

where $s_{\mu}, s_{\sigma}$ are the mean and standard deviation hyperparameters. The prior on $s_{\mu}$ is centered on negative values (reflecting the expectation that frontier scores tend to fall below latent capability), but is not truncated — the data is free to push $s_i$ toward zero or positive values. When `skew=False`, this parameter is omitted and the likelihood uses a symmetric Normal.

#### Harvey shape parameters $\alpha_i$:

Harvey shape parameters $\alpha_i$ follow a shifted Gamma distribution to enforce $\alpha_i > 1$:

$$
\alpha_i = 1 + \alpha^{\text{raw}}_i, \quad
\alpha^{\text{raw}}_i \sim \text{Gamma}(\alpha^{\text{raw}}_{\mu}, \alpha^{\text{raw}}_{\sigma}),
$$

where $\alpha^{\text{raw}}_ {\mu}, \alpha^{\text{raw}}_{\sigma}$ are the mean and standard deviation hyperparameters.


### Bounds on the upper asymptote

The asymptote $L_i$ of each benchmark has a Beta prior rescaled to $[L_{\min}, 1]$ with $L_{\min} = 0.75$ and a shared hyperprior on its mean (0.96, sd 0.02). Two per-benchmark constraints come from the data (`data.asymptote_bounds`, both on by default in `ModelConfig`):

- `L_floor_from_baselines`: $L_i$ is at least the highest human baseline recorded for the benchmark. The Beta draw is rescaled onto $[\text{baseline}, 1]$ instead of $[L_{\min}, 1]$, so the model cannot project a plateau under human performance; the shared prior then describes where the asymptote sits within each benchmark's feasible interval. (A truncated Beta was tried first and made the sampler diverge on nearly every draw.) Twenty-one benchmarks get a floor above 0.75 in the September 2026 data, from LAB-Bench SeqQA (0.78) to GSM8K (0.97).
- `L_fixed_from_ceiling`: a benchmark with a known ceiling in the pipeline's metadata has $L_i$ pinned there instead of estimated (ARC-AGI, ARC-AGI-2, EBR-bench, VPCT, Cybench and the two FrontierMath v2 sets, all at 1.0). `ModelConfig.L_fixed` pins named benchmarks by hand and wins over both rules.

`python -m benchmark_forecasting bounds` prints the resulting table. The `Lhuman` and `Lceil` tokens in a fit's file name say which rules were active, and the data fingerprint in the name covers the baselines and ceilings as well as the scores.

## Usage

```bash
uv sync                                   # dependencies, including the dev group (pytest, ruff, jupytext)
uv pip install -e .                       # the package, importable as benchmark_forecasting
uv run python -m benchmark_forecasting sync           # 0_input/ from ../benchmark-data-pipeline
uv run python 1_forecasts.py                          # about one hour: 18 MCMC fits, ~150 figures
uv run python 2_revision_analyses.py cheap figures    # stages, see below
uv run pytest && uv run ruff check .
```

`1_forecasts.py` fits the main model and an asymmetry model, runs the temporal holdout of the eight variants (sigmoid × structure × likelihood, cutoff 2025-01-01, at least 5 pre-cutoff frontier points per benchmark), the ablations, LOO and CQR, then draws the English paper figures (PDF) and the French note figures (PNG). Settings are at the top of the script: `MODEL_CONFIG`, `SAMPLING_CONFIG`, `LANGUAGE` / `DOCUMENT_TYPE`, `ALSO_GENERATE_FR`, `SAVEFIGS`, `DATA_CUTOFF_DATE`.

`2_revision_analyses.py` takes stage names as arguments and writes `Plots/4-Sensitivity/<cutoff>/revision_analyses_<stages>_<cutoff>.json`, CSV tables and LaTeX tables (in `Plots/4-Sensitivity/<cutoff>/tables/`, or in `$TABLES_DIR` to regenerate a manuscript's tables in place):

| Stage | What it does | Cost |
|---|---|---|
| `cheap` | Saturation dates and shifts across the eight variants, per-benchmark and per-category tables, posterior figures, residual dependence, lower-bound audit | cached fits only |
| `figures` | Redraws the category forecast panels | 1 cached fit |
| `retro` | Long-horizon retrodiction (cutoffs 2022 to 2025) | 4 new MCMC fits |
| `retro8` | All eight variants at the 2025 cutoff (CRPS, RMSE, coverage, calibration curves) | 8 new MCMC fits |
| `cqr` | Grouped repeated CQR, 100 random benchmark splits × 8 variants | 8 new MCMC fits |
| `priors` | Sensitivity to the prior on the asymptote | 4 new MCMC fits |

Fits are cached in `Fits/<slug>[_<tag>]_d<hash>.nc`, where the slug encodes the `ModelConfig` and the hash fingerprints the fitted data; a cache is only reused for the exact same data and configuration. Sampling needs `VECLIB_MAXIMUM_THREADS=1` and `OMP_NUM_THREADS=1` before numpy is imported (the scripts and `config.py` set them): with Apple Accelerate, four chain processes oversubscribe the cores and a 3-minute fit takes hours. Never run two samplings at once on one machine.

```python
MODEL_CONFIG = bf.ModelConfig(
    sigmoid="harvey",              # or "logistic"
    joint=True,                    # hierarchical (True) or independent (False)
    top_n=3,                       # expanding top-N frontier
    skew=True,                     # skew-normal (True) or normal (False) likelihood
    L_floor_from_baselines=True,   # L at least the highest human baseline
    L_fixed_from_ceiling=True,     # L pinned at a known ceiling
)
```

## Benchmark set (September 2026)

98 benchmarks in 12 capability categories, as included by the pipeline: Cyber (13), Autonomous SWE (12), Biology (11), General Reasoning (10), Domain Specific Questions (9), Mathematics (8), Multimodal Understanding (7), Agentic Computer Use (7), Trivia & Commonsense QA (6), Core AGI Progress (5), Advanced Language and Writing (5), Chemistry (5). The inclusion criteria and every exclusion are documented in the pipeline (`docs/decisions.md` and `2_database/excluded_benchmarks.csv` there).

Papers written before September 2026 used the 63-benchmark (April 2026) and 75-benchmark datasets built by the data-processing notebook this repository used to carry; that notebook and its `Data/` folder are in the git history up to the commit that introduced `0_input/`.
