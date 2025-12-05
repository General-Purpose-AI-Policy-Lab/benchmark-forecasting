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

The joint notebooks define hierarchical versions where benchmarks share hyperpriors over parameters, allowing benchmarks to borrow statistical strength from each other while keeping benchmark-specific trajectories.

#### Upper asymptotes $L_i$:

Upper asymptotes $L_i$ are drawn from a Beta distribution shifted to $[\ell_i, 1]$:

$$
L_i = \ell_i + (1 - \ell_i) L^{\text{raw}}_i, \quad
L^{\text{raw}}_i \sim \text{Beta}(L_{\mu}, L_{\sigma}),
$$

where $L_{\mu}, L_{\sigma}$ are the mean and standard deviation hyperparameters, respectively (instead of the usual shape parameters $\alpha, \beta$).

#### Growth rates $k_i$:

Growth rates $k_i$ follow a Gamma distribution:

$$
k_i \sim \text{Gamma}(k_{\mu}, k_{\sigma}),
$$

where $k_{\mu}, k_{\sigma}$ are the mean and standard deviation hyperparameters (instead of the usual shape and rate parameters $\alpha, \lambda$).

#### Inflection times $\tau_i$:

Inflection times $\tau_i$ follow a Gumbel distribution centered on empirical midpoint of each benchmark, with a scale of several years. The rationale is that for saturated benchmarks, the inflection point is roughly at the midpoint of observed data, and for unsaturated benchmarks, the inflection point is likely greater than the empirical midpoint.

#### Noise scales $\xi^{\text{base}}_i$:

Noise scales $\xi^{\text{base}}_i$ follow a Gamma distribution:

$$
\xi^{\text{base}}_i \sim \text{Gamma}(\sigma^{\text{base}}_{\mu}, \sigma^{\text{base}}_{\sigma}),
$$

where $\sigma^{\text{base}}_ {\mu}, \sigma^{\text{base}}_{\sigma}$ are the mean and standard deviation hyperparameters (instead of the usual shape and rate parameters $\alpha, \lambda$).

#### Skewness parameters $\lambda_i$:

Skewness parameters $\lambda_i$ follow a Truncated-Normal distribution (truncated at 0):

$$
\lambda_i \sim \text{TruncatedNormal}(\alpha^{\text{skew}}_{\mu}, \alpha^{\text{skew}}_{\sigma}, 0, \infty),
$$

where $\alpha^{\text{skew}}_ {\mu}, \alpha^{\text{skew}}_{\sigma}$ are the (untruncated) mean and standard deviation hyperparameters.

#### Harvey shape parameters $\alpha_i$:

Harvey shape parameters $\alpha_i$ follow a shifted Gamma distribution to enforce $\alpha_i > 1$: 

$$
\alpha^{\text{harvey}}_i = 1 + \alpha^{\text{raw}}_i, \quad
\alpha^{\text{raw}}_i \sim \text{Gamma}(\alpha^{\text{raw}}_{\mu}, \alpha^{\text{raw}}_{\sigma}),
$$

where $\alpha^{\text{raw}}_ {\mu}, \alpha^{\text{raw}}_{\sigma}$ are the mean and standard deviation hyperparameters.

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
