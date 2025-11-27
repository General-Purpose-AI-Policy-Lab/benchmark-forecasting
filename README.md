# Benchmark Progress Forecasting Pipeline

This repository contains a four-stage workflow for cleaning benchmark data, fitting two sigmoidal forecasting models (Logistic and Harvey), and generating unified plots. The implementation is structured as sequential Jupyter notebooks.

## Repository Structure

- **0_Clean_EpochAI_benchmarks.ipynb**  
  Cleans and standardizes EpochAI benchmark data. Tasks include: ingesting raw benchmark timelines, harmonizing benchmark identifiers, handling missing/irregular data, computing per-benchmark lower bounds (random-chance baselines), producing long-format tables for modeling, and exporting cleaned CSV/Parquet outputs.

- **1_Logistic_forecast.ipynb**  
  Loads the cleaned dataset and fits Bayesian shifted logistic growth models in PyMC. Provides:
  - **Independent per-benchmark models**;
  - A **joint hierarchical model** where benchmarks share hyperpriors over asymptotes, growth rates, noise scales, and skewness.
  Outputs posterior samples, asymptote distributions, and forward forecasts.

- **2_Harvey_forecast.ipynb**  
  Implements the Harvey growth model (a generalized logistic / Harvey-type sigmoid with decaying effective growth rate). As with the logistic notebook, it supports both independent and joint hierarchical variants and generates predictive distributions and long-horizon forecasts.

- **3_Plot_forecasts.ipynb**  
  Aggregates outputs from both model families and produces figures: historical data and posterior predictive envelopes of the Logistic / Harvey forecasts for each benchmark (single panels and multi-benchmark panels).

## Model Details

### Time and lower bounds

For each benchmark $i$, scores $y_{i,t}$ are modeled as functions of a time variable $t$ (e.g. days since first observation). A benchmark-specific random-chance **lower bound** $\ell_i$ is estimated from metadata (or set to 0 if unknown), and all sigmoids are defined on the shifted range $[\ell_i, 1]$.

### Shifted Logistic model

For benchmark $i$, the latent mean performance at time $t$ is

$$
\mu_i^{\text{log}}(t) = \ell_i + (L_i - \ell_i)\sigma_i(t), \quad
\sigma_i(t) = \frac{1}{1 + \exp\big(-k_i (t - t_{0,i})\big)} ,
$$

where:
- $L_i$ is the upper asymptote (final performance),
- $k_i$ is the growth rate,
- $t_{0,i}$ is the inflection time.

### Harvey model

The Harvey curve generalizes the logistic with a shape parameter $\alpha_i > 1$ that controls how sharply growth slows down. For benchmark $i$,

$$
h_i(t) = \left[1 - (1 - \alpha_i)\exp\big(-r_i (t - t_{\mathrm{i},i})\big) \right]^{\!\frac{1}{1 - \alpha_i}} ,
$$

and the shifted mean is

$$
\mu_i^{\text{harv}}(t) = \ell_i + (L_i - \ell_i)h_i(t),
$$

with:
- $L_i$ asymptote,
- $r_i$ growth-rate parameter (comparable scale to $k_i$),
- $t_{\mathrm{i},i}$ time offset (start of growth),
- $\alpha_i > 1$ controlling asymmetry (larger $\alpha_i$ gives faster early growth and stronger late slowdown).

### Observation model and heteroskedastic noise

Both model families share the same likelihood structure. For an observation at time $t$ on benchmark $i$,

$$
y_{i,t} \sim \text{SkewNormal}\big(\mu_i(t), \sigma_i(t), \alpha^{\text{skew}}_i\big),
$$

where the scale is heteroskedastic and approximately Beta-shaped over the interval $[\ell_i, L_i]$:

$$
\sigma_i(t) = \sigma_0 + \sigma^{\text{base}}_i\frac{\sqrt{\big(\mu_i(t) - \ell_i\big)\big(L_i - \mu_i(t)\big)}}{(L_i - \ell_i)/2},
$$

peaking near the inflection point and shrinking near the bounds. The skewness parameter $\alpha^{\text{skew}}_i \le 0$ allows slightly asymmetric residuals and is weakly informed by priors.

### Hierarchical (joint) models

The joint notebooks define hierarchical versions where benchmarks share hyperpriors:

- **Asymptotes:**
  - Hyperparameters $L_{\mu}, L_{\sigma}$ control the distribution of upper asymptotes close to 1.
  - Task-level $L_i$ are obtained via shifted Beta draws:

$$
L^{\text{raw}}_i \sim \text{Beta}(L_{\mu}, L_{\sigma}), \quad
L_i = \ell_i + (1 - \ell_i) L^{\text{raw}}_i.
$$

- **Growth rates (logistic and Harvey):**
  - Hyperparameters $k_{\mu}, k_{\sigma}$ (or $r_{\mu}, r_{\sigma}$ for Harvey).
  - Task-level parameters:

$$
k_i \sim \text{Gamma}(k_{\mu}, k_{\sigma}) \quad \text{or} \quad
r_i \sim \text{Gamma}(r_{\mu}, r_{\sigma}).
$$

- **Inflection / start times:**
  - Task-level $t_{0,i}$ and $t_{\mathrm{i},i}$ are centered on empirical midpoints or first-observation dates with broad priors (e.g. Gumbel/Normal with year-scale spreads).

- **Noise scales and skewness:**
  - Shared hyperpriors:
    - $\sigma^{\text{base}}_{\mu}$
    - $\sigma^{\text{base}}_{\sigma}$
    - $\alpha^{\text{skew}}_{\mu}$
    - $\alpha^{\text{skew}}_{\sigma}$
  - Task-level $\sigma^{\text{base}}_i$ and $\alpha^{\text{skew}}_i$ are drawn from these distributions.

- **Harvey shape parameter:**
  - Hyperpriors for a base parameter:
    - $\alpha^{\text{base}}_{\mu}$
    - $\alpha^{\text{base}}_{\sigma}$
  - Task-level $\alpha_i = 1 + \alpha^{\text{base}}_i$ enforce $\alpha_i > 1$.

These hierarchical models allow benchmarks to borrow statistical strength from each other while keeping benchmark-specific trajectories.

## Dependencies

Typical environment:
```text
python >= 3.10
pandas
numpy
scipy
matplotlib
pymc
arviz
datetime
typing
os
pickle
```

## Usage

Run the notebooks in order:

1. `0_Clean_EpochAI_benchmarks.ipynb`
2. `1_Logistic_forecast.ipynb`
3. `2_Harvey_forecast.ipynb`
4. `3_Plot_forecasts.ipynb`

Model fits are saved under `Fits/`, and figures are written to `Images/`.
