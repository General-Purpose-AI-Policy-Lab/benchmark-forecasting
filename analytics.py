"""Evaluation metrics and visualization for model diagnostics."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
from scipy.stats import energy_distance

from data_pipeline import preprocess_for_modeling
from forecasting import BenchmarkForecaster

def temporal_cross_validate(
    df: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    model_config: dict | None = None,
) -> tuple[az.InferenceData, pd.DataFrame]:
    """Trains on past data and predicts future data (holdout)."""
    config = model_config or {}
    frontier_depth = config.get("frontier_depth", 3)
    
    df_proc = preprocess_for_modeling(df, top_n_frontier=frontier_depth)

    # Time-based split
    train_mask = df_proc["release_date"] < cutoff_date
    
    # Filter benchmarks that exist in training set to ensure valid referencing
    valid_benchmarks = set(df_proc[train_mask]["benchmark"].unique())
    
    df_train = df_proc[train_mask & df_proc["benchmark"].isin(valid_benchmarks)]
    df_test = df_proc[~train_mask & df_proc["benchmark"].isin(valid_benchmarks)]

    forecaster = BenchmarkForecaster(**config)
    forecaster.fit(df_train, show_progress=True)

    # Prepare test set for posterior predictive sampling
    test_bench_ids = pd.Categorical(
        df_test["benchmark"],
        categories=forecaster._category_map,
        ordered=True
    ).codes

    with forecaster.model:
        import pymc as pm
        pm.set_data(
            {"t_obs": df_test["days"].values, "idx_obs": test_bench_ids},
            coords={"obs": df_test.index}
        )
        idata_preds = pm.sample_posterior_predictive(
            forecaster.inference_data,
            predictions=True,
            extend_inferencedata=False,
            random_seed=42,
            progressbar=True
        )
    
    # Inject ground truth for metric calculation
    idata_preds.predictions["score_true"] = (("obs"), df_test["score"].values)
    
    return idata_preds, df_test


def calculate_accuracy_metrics(idata: az.InferenceData) -> dict[str, float]:
    """Computes CRPS, RMSE, and MAE from posterior predictions."""
    y_pred = idata.predictions.stack(sample=("chain", "draw"))["score_obs"].values
    y_true = idata.predictions["score_true"].values

    # Continuous Ranked Probability Score via Energy Distance
    crps_values = [energy_distance(p, (t,)) ** 2 / 2 for p, t in zip(y_pred, y_true)]
    
    point_predictions = np.mean(y_pred, axis=1)
    
    return {
        "crps": np.mean(crps_values),
        "rmse": np.sqrt(np.mean((point_predictions - y_true) ** 2)),
        "mae": np.mean(np.abs(point_predictions - y_true))
    }


def plot_calibration_curve(idata: az.InferenceData) -> plt.Figure:
    """Plots observed coverage vs expected confidence intervals."""
    y_pred = idata.predictions.stack(sample=("chain", "draw"))["score_obs"].values
    y_true = idata.predictions["score_true"].values

    expected_confidence = np.linspace(0.01, 0.99, 20)
    observed_coverage = []

    for p in expected_confidence:
        lower = np.quantile(y_pred, (1 - p) / 2, axis=1)
        upper = np.quantile(y_pred, 1 - (1 - p) / 2, axis=1)
        is_covered = (y_true >= lower) & (y_true <= upper)
        observed_coverage.append(np.mean(is_covered))

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(expected_confidence, observed_coverage, "o-", label="Model Calibration")
    ax.plot([0, 1], [0, 1], "k--", label="Ideal")
    ax.set_xlabel("Expected Confidence Level")
    ax.set_ylabel("Observed Coverage")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig