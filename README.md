# Benchmark Progress Forecasting Pipeline

This repository forecasts AI benchmark performance over time using Bayesian sigmoidal growth models. It processes benchmark data from EpochAI, Scale AI, and RAND sources, fits hierarchical Harvey growth curves via PyMC, and generates forecast visualizations.

## Repository Structure

- **`refresh_epoch_data.py`**
  Refreshes `Data/benchmark_data/*.csv` from the Epoch AI hub exports (`uv run python refresh_epoch_data.py tmp/epoch --download`). Each file becomes exactly the current Epoch export — Epoch is trusted as the source of truth, so models it drops or re-scores (e.g. SimpleQA Verified 1.2.0, PostTrainBench v1.1) change here too. Our two FrontierMath files are mapped to the v2 task sets.

- **`0_Process_benchmarks.ipynb`**
  Cleans and standardizes benchmark data from multiple sources (EpochAI files, Scale AI leaderboards scraped live with Playwright, RAND). Includes: selecting the score column per benchmark, harmonizing identifiers, filling and correcting model release dates (`DATE_OVERRIDES`), merging sibling Scale leaderboards into one series (SWE Atlas = Codebase QnA + Test Writing + Refactoring; PRBench = Finance + Legal), applying the manual exclusions, loading per-benchmark lower bounds (random-chance baselines) and categories, and exporting cleaned outputs.

- **`1_Forecasts.ipynb`**
  Main analysis notebook. Loads the cleaned dataset and:
  1. Fits retrodiction models (Harvey/Logistic × Joint/Independent) for temporal holdout validation;
  2. Fits the main Harvey hierarchical model and generates forecasts;
  3. Produces all figures (forecasts by category, saturation proportion, Harvey asymmetry, calibration curves);
  4. Runs sensitivity analyses (skew vs normal likelihood, joint vs independent, threshold sensitivity, conformal prediction intervals);
  5. Exports numerical results to JSON.

  Almost every figure in `Plots/` comes from this notebook; the posterior diagnostics in
  `Plots/1-High_level/` and the long-horizon calibration curves come from
  `2_Revision_analyses.py` instead. The frontier counts a model family once per
  release day (best reasoning effort / size, see `forecasting.model_base_key`); the plots
  show exactly the points used for fitting.

- **`2_Revision_analyses.py`** (jupytext percent format — opens as a notebook in Jupyter, and VS Code reads its `# %%` cells natively)
  Robustness and sensitivity analyses that go beyond the ablations of notebook 1: per-benchmark
  posterior saturation dates and how they shift across the 8 model variants, long-horizon
  retrodiction (cutoffs 2022–2025), prior sensitivity on the upper asymptote $L$, cross-benchmark
  residual dependence, hyperparameter and per-benchmark $L$ posterior figures, and an audit of the
  random-chance lower bounds. Results are written to `Plots/4-Sensitivity/cutoffYYYYMMDD/` as CSV/JSON, and LaTeX
  versions of the tables to `$TABLES_DIR` (see below). Runs in selectable stages so the expensive
  refits can be skipped:

  | Stage | What it does | Cost |
  |---|---|---|
  | `cheap` | Saturation dates and variant shifts, posterior figures, residual dependence, lower-bound audit | cached fits only |
  | `figures` | Redraws the 11 category forecast panels | 1 cached fit |
  | `retro` | Long-horizon retrodiction (cutoffs 2022–2024) | 3 new MCMC fits |
  | `priors` | Asymptote-prior sensitivity | 3 new MCMC fits |
  | `retro8` | All 8 variants at the 2025 cutoff (CRPS, RMSE, coverage, calibration curves) | 8 new MCMC fits |
  | `cqr` | Grouped repeated CQR at the main cutoff, 100 random benchmark splits × 8 variants | 8 new MCMC fits |

  ```bash
  uv run python 2_Revision_analyses.py cheap           # one stage
  uv run python 2_Revision_analyses.py cheap figures   # or several at once
  ```

  Each stage writes `Plots/4-Sensitivity/<cutoff>/revision_analyses_<stage>_<cutoff>.json`. With no
  argument, `cheap` runs by default.

- **`forecasting.py`**
  Core modeling utilities: frontier selection (`select_frontier_points`, with the same-day effort/size de-duplication), model construction (`build_model`, including pinned asymptotes via `ModelConfig.L_fixed`), MCMC fitting (`fit`, cached under `Fits/` with a fingerprint of the fitted data in the filename), temporal holdout validation (`temporal_holdout`), scoring (CRPS, RMSE, MAE), forecast generation, and conformal prediction coverage (CQR). Also `saturation_dates` and `saturated_proportion` (posterior saturation timing, by analytic inversion of the fitted sigmoid) and `residual_diagnostics` (cross-benchmark residual correlation, paired by model release).

- **`plotting.py`**
  Matplotlib plotting utilities with centralized styling, supporting both English (paper/PDF) and French (note/PNG) output, including `plot_L_intervals` and `plot_hyperparameters` for posterior diagnostics.

### Directory Structure

```
.
├── 0_Process_benchmarks.ipynb  # Data loading and normalization
├── 1_Forecasts.ipynb           # Model fitting, validation, plotting, and sensitivity analyses
├── 2_Revision_analyses.py      # Robustness analyses (saturation dates, retrodiction, priors, residuals)
├── refresh_epoch_data.py       # Refresh the Epoch CSVs from the hub exports
├── forecasting.py              # Core modeling utilities
├── plotting.py                 # Matplotlib plotting utilities
├── Data/
│   ├── benchmark_data/             # Raw CSV files from EpochAI (~75 files, see refresh_epoch_data.py)
│   ├── benchmark_data_RAND/        # RAND Corporation benchmark data
│   ├── benchmark_data_processed/   # Processed/normalized data (output of notebook 0)
│   ├── benchmarks_lower_bounds.csv # Random-chance baselines per benchmark
│   └── human_baselines.csv         # Human performance baselines
├── Plots/
│   ├── 0-Note-figures/         # FR note figures (PNG)
│   ├── 1-High_level/           # Saturation proportion, Harvey asymmetry, hyperparameter and L-interval posteriors
│   ├── 2-Forecasts/            # Main forecast trajectories per category, one subfolder per cutoff
│   │   └── cutoffYYYYMMDD/     #   EN paper PDFs; FR note PNGs in fr/ underneath
│   ├── 3-Calibration/          # Calibration curves, one subfolder per cutoff (FR PNGs in fr/)
│   └── 4-Sensitivity/          # Ablation figures and CSV/JSON results, one subfolder per cutoff
│       └── tables/             # LaTeX tables (default TABLES_DIR; see 2_Revision_analyses.py)
├── Fits/                       # Saved model posteriors (NetCDF, gitignored; names end with _d<data hash>)
├── Paper/                      # Bibliography and arXiv preprint sources; other manuscript
│                               # working material is kept local and gitignored
└── tmp/                        # Jupytext conversions (gitignored)
```

Only `Paper/Benchmark_forecasting.bib`, `Paper/Arxiv/` and `Paper/.latexmkrc` are tracked.
`2_Revision_analyses.py` writes its LaTeX tables to `Plots/4-Sensitivity/<cutoff>/tables/` by default; point
`TABLES_DIR` at a manuscript directory to regenerate them in place:

```bash
TABLES_DIR=path/to/manuscript/tables uv run python 2_Revision_analyses.py cheap
```

### Data Files

- **`Data/benchmarks_lower_bounds.csv`** — Per-benchmark random-chance performance lower bounds (semicolon-delimited, European decimal notation; no `;` inside the justification column). Values are aligned on Epoch's `random_baseline` where it exists (https://epoch.ai/data/benchmark_metadata.csv).
- **`Data/human_baselines.csv`** — Human performance reference points for plotting (columns: `benchmark`, `group`, `score`, `note`, `source`)

### Benchmark set (September 2026)

75 benchmarks in 11 capability categories (Core AGI Progress, Mathematics, Autonomous SWE, Agentic Computer Use, Biology, Chemistry, Domain Specific Questions, General Reasoning, Multimodal Understanding, Advanced Language and Writing, Commonsense QA) plus one uncategorized benchmark (`Tier 2 Excluded`). Inclusion criteria: scores on a bounded 0–1 scale (Elo, speedups, dollar amounts and pool-relative "dominance" scores are excluded, see `UNNORMALIZED_BENCHMARKS`), capabilities rather than propensities, a ceiling that saturation can reach (ForecastBench's Brier index is excluded), and benchmark quality (SWE-Bench Verified/Pro, SciCode, CursorBench and HiL-Bench are excluded), a stable item set (LiveBench, whose items rotate and which is no longer evaluated, is excluded), and a frontier made of general-purpose models rather than developer-reported or task-fine-tuned entries (Adversarial NLI and ScienceQA are excluded); see `MANUALLY_EXCLUDED_BENCHMARKS` in notebook 0. Papers published before September 2026 refer to the earlier 63-benchmark dataset.

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

## Usage

### Quick start

```bash
# Install dependencies (Chromium is needed once for the Scale scraping in notebook 0)
uv sync
uv run playwright install chromium

# Optional: refresh the Epoch CSVs from the hub exports
uv run python refresh_epoch_data.py tmp/epoch --download

# Run the data processing notebook (scrapes the Scale leaderboards live)
MPLBACKEND=Agg uv run jupyter nbconvert --execute --inplace 0_Process_benchmarks.ipynb

# Run the main analysis as a Python script (recommended over nbconvert,
# which can hit IOPub timeouts on long MCMC sampling cells)
mkdir -p tmp
uv run jupytext --to py:percent 1_Forecasts.ipynb -o tmp/1_Forecasts_run.py
uv run python tmp/1_Forecasts_run.py
```

Or run interactively in Jupyter / VS Code.

### Step by step

1. Run `0_Process_benchmarks.ipynb` to generate `Data/benchmark_data_processed/all_normalized_updated_benchmarks.csv`. In `1_Forecasts.ipynb`, the `DATA_CUTOFF_DATE` parameter (default: 2026-09-04, inclusive) excludes model results released after that date, ensuring reproducibility even if new data is added to the CSVs; it also names the output subfolders and the fit caches.
2. Open `1_Forecasts.ipynb` and verify the settings at the top:
   - `LANGUAGE` / `DOCUMENT_TYPE`: controls the main figures. Default is `"en"` / `"paper"` (PDF output).
   - `ALSO_GENERATE_FR`: when `True` (default), the notebook also generates French note figures (PNG) at the end, reusing already-fitted models (no extra MCMC).
   - `SAVEFIGS`: `True` to save all figures to `Plots/`.
3. Run all cells. The notebook will:
   - Fit 4 retrodiction models and produce calibration curves → `Plots/3-Calibration/cutoffYYYYMMDD/`
   - Fit the main model and produce forecasts, saturation, and asymmetry figures → `Plots/1-High_level/` and `Plots/2-Forecasts/cutoffYYYYMMDD/`
   - Run 4 ablation models (skew/normal × joint/independent) with forecasts, saturation, and calibration → `Plots/4-Sensitivity/cutoffYYYYMMDD/`
   - Compute CQR conformal prediction intervals on all variants → `Plots/4-Sensitivity/cutoffYYYYMMDD/ablation_results_cutoffYYYYMMDD.json`
   - Generate French (note/PNG) versions of all main figures → `Plots/0-Note-figures/` and `Plots/2-Forecasts/cutoffYYYYMMDD/fr/`

**Runtime**: about one hour (18 MCMC fits of 3–4 minutes each, ~150 figures). The notebooks set `VECLIB_MAXIMUM_THREADS=1` / `OMP_NUM_THREADS=1` before importing numpy: with Apple Accelerate, four chain processes each spawning a full BLAS thread pool oversubscribe the cores and slow sampling down ~50x. Never run two samplings at the same time on one machine, for the same reason.

**Note on execution method**: `jupyter nbconvert --execute` may fail on this notebook due to IOPub timeouts during long MCMC sampling steps. The recommended approach is to convert to a Python script with jupytext and run directly (see Quick start above). For interactive use, Jupyter Lab / VS Code handles long-running cells without issue.

### Configuration

```python
MODEL_CONFIG = forecasting.ModelConfig(
    sigmoid="harvey",  # or "logistic"
    joint=True,        # hierarchical (True) or independent (False)
    top_n=3,           # track top-N frontier models
    skew=True,         # skew-normal (True) or normal (False) likelihood
    L_fixed=L_FIXED,   # asymptotes pinned to 1 for a few benchmarks (see above)
)
```

`forecasting.prepare_dataset(raw, top_n=3)` builds the frontier used for fitting and plotting (one point per model family and day, then the expanding top-N); `dedupe_effort=False` keeps every published point, for diagnostics only.

The sensitivity analyses section always runs all 4 combinations of `joint` × `skew` and produces English paper figures (PDF), regardless of the main `LANGUAGE`/`DOCUMENT_TYPE` settings.
