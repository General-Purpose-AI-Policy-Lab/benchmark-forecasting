"""Bayesian forecasting model for benchmark performance limits."""

from typing import Literal
import arviz as az
import pymc as pm
import pandas as pd
import numpy as np

GrowthFunction = Literal["logistic", "harvey"]

class BenchmarkForecaster:
    """Hierarchical Bayesian model for forecasting performance benchmarks."""

    def __init__(
        self,
        growth_function: GrowthFunction = "logistic",
        hierarchical: bool = True,
        frontier_depth: int = 3,
        random_seed: int = 42,
    ):
        self.growth_function = growth_function
        self.hierarchical = hierarchical
        self.frontier_depth = frontier_depth
        self.random_seed = random_seed
        
        self.inference_data: az.InferenceData | None = None
        self.model: pm.Model | None = None
        self._category_map: pd.Index | None = None

    def fit(
        self,
        df: pd.DataFrame,
        draws: int = 2000,
        tune: int = 1000,
        show_progress: bool = True,
    ) -> "BenchmarkForecaster":
        """Fits the model MCMC chains to the training data."""
        bench_ids, bench_names = pd.factorize(df["benchmark"], sort=True)
        self._category_map = bench_names
        
        coords = {
            "benchmark": bench_names,
            "obs": df.index,
        }

        with pm.Model(coords=coords) as self.model:
            # Mutable data containers for later prediction swapping
            t_obs = pm.Data("t_obs", df["days"].values, dims="obs")
            idx_obs = pm.Data("idx_obs", bench_ids, dims="obs")
            
            # Static covariates per benchmark
            lower_bounds = pm.Data(
                "lower_bound", 
                df.groupby(bench_ids)["lower_bound"].first().values, 
                dims="benchmark"
            )
            midpoints = df.groupby(bench_ids)["days_midpoint"].first().values

            # --- Parameter Definitions ---
            asymptote = self._define_asymptote(dims="benchmark")
            inflection = self._define_inflection(midpoints, dims="benchmark")
            rate = self._define_growth_rate(dims="benchmark")
            
            # --- Latent Mean Function ---
            sigmoid = self._calculate_sigmoid(
                t_obs, idx_obs, inflection, rate
            )
            
            # Linear interpolation between lower bound and estimated asymptote
            latent_performance = pm.Deterministic(
                "latent_performance", 
                lower_bounds[idx_obs] + (asymptote[idx_obs] - lower_bounds[idx_obs]) * sigmoid, 
                dims="obs"
            )

            # --- Likelihood ---
            noise_scale, skewness = self._define_noise_process(
                latent_performance, asymptote, lower_bounds, idx_obs, dims="benchmark"
            )

            pm.SkewNormal(
                "score_obs",
                mu=latent_performance,
                sigma=noise_scale,
                alpha=skewness[idx_obs],
                observed=df["score"].values,
                dims="obs",
            )

            self.inference_data = pm.sample(
                draws,
                tune=tune,
                random_seed=self.random_seed,
                target_accept=0.9,
                init="adapt_diag",
                progressbar=show_progress,
            )

        return self

    def predict(
        self, 
        df_future: pd.DataFrame, 
        ci_width: float = 0.8
    ) -> pd.DataFrame:
        """Generates posterior predictions on new time points."""
        if not self.model or not self.inference_data:
            raise RuntimeError("Model must be fit before prediction.")

        # Map string benchmarks to existing integer IDs
        bench_ids = pd.Categorical(
            df_future["benchmark"],
            categories=self._category_map,
            ordered=True,
        ).codes

        if (bench_ids == -1).any():
            raise ValueError("Prediction data contains unknown benchmarks.")

        with self.model:
            pm.set_data(
                {
                    "t_obs": df_future["days"].values,
                    "idx_obs": bench_ids,
                },
                coords={"obs": df_future.index},
            )
            
            pp_samples = pm.sample_posterior_predictive(
                self.inference_data,
                var_names=["latent_performance"],
                predictions=True,
                random_seed=self.random_seed,
                progressbar=False,
            )

        # Extract latent mean statistics
        mu_matrix = pp_samples.predictions.stack(sample=("chain", "draw"))["latent_performance"].values
        
        result = df_future.copy()
        result["mean_pred"] = np.mean(mu_matrix, axis=1)
        result["lower_ci"] = np.quantile(mu_matrix, (1 - ci_width) / 2, axis=1)
        result["upper_ci"] = np.quantile(mu_matrix, 1 - (1 - ci_width) / 2, axis=1)
        
        return result

    # --- Internal Parameter Builders ---

    def _define_asymptote(self, dims: str):
        """Priors for the theoretical maximum score (L)."""
        # Scaled Beta prior between 0.75 and 1.0
        L_MIN, L_MAX = 0.75, 1.0
        L_RANGE = L_MAX - L_MIN
        
        dims_hyper = None if self.hierarchical else dims

        # Raw Beta parameters
        mu_raw = pm.Beta("L_raw_mu", mu=(0.96 - L_MIN) / L_RANGE, sigma=0.02/L_RANGE, dims=dims_hyper)
        sigma_raw = pm.HalfNormal("L_raw_sigma", sigma=0.02/L_RANGE, dims=dims_hyper)
        
        raw_val = pm.Beta("L_raw", mu=mu_raw, sigma=sigma_raw, dims=dims)
        return pm.Deterministic("asymptote", L_MIN + L_RANGE * raw_val, dims=dims)

    def _define_inflection(self, midpoints, dims: str):
        """Priors for the time of maximum growth (tau)."""
        return pm.Gumbel("inflection_point", mu=midpoints, beta=365 * 2, dims=dims)

    def _define_growth_rate(self, dims: str):
        """Priors for the steepness of the curve (k)."""
        dims_hyper = None if self.hierarchical else dims
        mu = pm.Gamma("rate_mu", mu=0.005, sigma=0.002, dims=dims_hyper)
        sigma = pm.HalfNormal("rate_sigma", sigma=0.005, dims=dims_hyper)
        return pm.Gamma("growth_rate", mu=mu, sigma=sigma, dims=dims)

    def _calculate_sigmoid(self, t, idx, tau, k):
        """Computes the growth curve values based on selected function type."""
        logits = k[idx] * (t - tau[idx])
        
        if self.growth_function == "logistic":
            return pm.math.sigmoid(logits)
            
        elif self.growth_function == "harvey":
            # Harvey (Generalized Logistic) allows asymmetry
            dims_hyper = None if self.hierarchical else "benchmark"
            alpha_mu = pm.Gamma("alpha_mu", mu=1.5, sigma=0.5, dims=dims_hyper)
            alpha_sigma = pm.HalfNormal("alpha_sigma", sigma=0.5, dims=dims_hyper)
            
            alpha_raw = pm.Gamma("alpha_raw", mu=alpha_mu, sigma=alpha_sigma, dims="benchmark")
            alpha = pm.Deterministic("asymmetry_factor", alpha_raw + 1.0, dims="benchmark")
            
            base = pm.math.maximum(1 - (1 - alpha[idx]) * pm.math.exp(-logits), 1e-10)
            return pm.math.exp(1 / (1 - alpha[idx]) * pm.math.log(base))
        
        raise ValueError(f"Unknown growth function: {self.growth_function}")

    def _define_noise_process(self, mu, L, l_bound, idx, dims: str):
        """Constructs heteroscedastic noise scaling and skewness."""
        dims_hyper = None if self.hierarchical else dims
        
        # Base noise level
        base_mu = pm.Gamma("noise_base_mu", mu=0.05 + self.frontier_depth / 50, sigma=0.02, dims=dims_hyper)
        base_sigma = pm.HalfNormal("noise_base_sigma", sigma=0.05, dims=dims_hyper)
        noise_base = pm.Gamma("noise_base", mu=base_mu, sigma=base_sigma, dims=dims)
        
        # Heteroscedasticity: variance peaks when performance is changing most rapidly
        # We approximate this using the geometric mean of distances to bounds
        variance_shape = pm.math.sqrt((mu - l_bound[idx]) * (L[idx] - mu))
        max_theoretical_var = (L[idx] - l_bound[idx]) / 2.0
        scaling_factor = variance_shape / pm.math.maximum(max_theoretical_var, 1e-10)
        
        noise_scale = 0.01 + noise_base[idx] * scaling_factor

        # Skewness parameters
        skew_mu = pm.Normal("skew_mu", mu=-2 - self.frontier_depth / 2, sigma=0.5, dims=dims_hyper)
        skew_sigma = pm.HalfNormal("skew_sigma", sigma=1, dims=dims_hyper)
        skewness = pm.TruncatedNormal("skewness", mu=skew_mu, sigma=skew_sigma, upper=0, dims=dims)
        
        return noise_scale, skewness