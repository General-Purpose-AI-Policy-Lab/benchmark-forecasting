# Benchmark Progress Forecasting Pipeline

This repository contains a four-stage workflow for cleaning benchmark data, fitting two sigmoidal forecasting models (Logistic and Harvey), and generating unified plots.

## Repository Structure

- **0_Clean_EpochAI_benchmarks.ipynb**  
  Cleans and standardizes EpochAI benchmark data. Includes: selecting and formating benchmark scores for modeling, harmonizing benchmark identifiers, handling missing/irregular data, loading per-benchmark lower bounds (random-chance baselines), and exporting cleaned outputs.

- **1_Logistic_forecast.ipynb**  
  Loads the cleaned dataset and fits Bayesian shifted logistic growth models in PyMC. Provides:
  - **Independent per-benchmark models**;
  - A **joint hierarchical model** where benchmarks share hyperpriors over asymptotes, growth rates, noise scales, and skewness.
  Outputs posterior samples, asymptote distributions, and forward forecasts.

- **2_Harvey_forecast.ipynb**  
  Implements the Harvey growth model (a generalized logistic with decaying effective growth rate). As with the logistic notebook, it supports both independent and joint hierarchical variants.

- **3_Plot_forecasts.ipynb**  
  Aggregates outputs produces figures: historical data and posterior predictive envelopes of the Logistic / Harvey forecasts for each benchmark (single panels and multi-benchmark panels).

## Model Details

Let $y_{i}(t)$ be the observed score for benchmark $i$ at time $t$. 

The score is modeled as a sigmoidal growth curve $\mu_i(t)$ plus skewed heteroskedastic noise $\xi_i(t)$:

$$
y_{i}(t) \sim \text{SkewNormal}\big(\mu_i(t), \xi_i(t), \lambda_i\big),
$$

where $\lambda_i \le 0$ is the skewness parameter allowing asymmetric residuals, i.e. benchmarks scores mostly below the latent optimal performance over time.

### Sigmoidal growth curves

The sigmoidal curves model the latent mean performance $\mu_i(t)$ over time. Two families of sigmoids are implemented: a shifted logistic function, and a generalization allowing asymmetric growth, the Harvey curve.

The sigmoids are defined on the range $[\ell_i, L_i]$, where $\ell_i$ is a benchmark-specific lower bound (random-chance performance) and $L_i$ is the upper asymptote (final performance).

The lower bound $\ell_i$ is manually gathered per benchmark (or set to 0 if unknown). See `benchmarks_lower_bounds.csv` for details. It is not necessarily 0, as some benchmarks may have non-zero random-chance performance (e.g. 25% for questions with 4 choices).

The upper bound $L_i$ is not necessarily 1, as benchmarks contain errors or inherent uncertainty that prevent perfect scores.

The latent mean performance on benchmark $i$ at time $t$ is then the shifted sigmoid:

$$
\mu_i(t) = \ell_i + (L_i - \ell_i) \sigma_i(t),
$$

where $\sigma_i(t) \in \\{\sigma_i^{\text{log}}(t), \sigma_i^{\text{harv}}(t)\\}$ is the sigmoid function (Logistic or Harvey). We indicate with the exponents $\text{log}$ and $\text{harv}$ the two variants when necessary.

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

peaking near the inflection point and shrinking near the bounds. 
[TODO explain $xi_0$ et $\xi^{base}_i$]

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
  - Time offsets $t_{0,i}$ are centered on empirical midpoints with broad priors (e.g. Gumbel - an asymetric distribution - with year-scale spreads).

- **Noise scales and skewness:**
  - Shared hyperpriors:
    - $\sigma^{\text{base}}_{\mu}$
    - $\sigma^{\text{base}}_{\sigma}$
    - $\alpha^{\text{skew}}_{\mu}$
    - $\alpha^{\text{skew}}_{\sigma}$
  - Benchmark-level parameters $\sigma^{\text{base}}_i$ and $\alpha^{\text{skew}}_i$ are drawn from the same distributions (resp. Gamma and Truncated-Normal).

- **Harvey shape parameter:**
  - Hyperpriors for the shape parameter (Gamma distribution):
    - $\alpha^{\text{harvey,base}}_{\mu}$
    - $\alpha^{\text{harvey,base}}_{\sigma}$
  - Benchmark-level parameter $\alpha^{\text{harvey}}_i = 1 + \alpha^{\text{harvey,base}}_i$, enforcing $\alpha_i > 1$.

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
