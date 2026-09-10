"""Bayesian sigmoidal growth models for benchmark saturation, fed by benchmark-data-pipeline.

The public API is re-exported here so scripts can write ``bf.fit(...)`` after
``import benchmark_forecasting as bf``.
"""

from benchmark_forecasting import config, plotting, pytensor_compat
from benchmark_forecasting.config import ModelConfig, SamplingConfig
from benchmark_forecasting.data import (
    asymptote_bounds,
    load_baselines,
    load_dataset,
    model_base_key,
    prepare_dataset,
    select_frontier_points,
)
from benchmark_forecasting.evaluate import (
    conformal_prediction_coverage,
    conformal_prediction_coverage_grouped,
    crps_score,
    point_error,
    residual_diagnostics,
    saturated_proportion,
    saturation_dates,
)
from benchmark_forecasting.fit import data_fingerprint, fit, temporal_holdout
from benchmark_forecasting.forecast import generate_forecast
from benchmark_forecasting.model import build_model, sampler_initvals
from benchmark_forecasting.sync import sync

__all__ = [
    "ModelConfig",
    "SamplingConfig",
    "asymptote_bounds",
    "build_model",
    "config",
    "conformal_prediction_coverage",
    "conformal_prediction_coverage_grouped",
    "crps_score",
    "data_fingerprint",
    "fit",
    "generate_forecast",
    "load_baselines",
    "load_dataset",
    "model_base_key",
    "plotting",
    "point_error",
    "prepare_dataset",
    "residual_diagnostics",
    "sampler_initvals",
    "saturated_proportion",
    "saturation_dates",
    "select_frontier_points",
    "sync",
    "temporal_holdout",
]

# Xcode 27's linker rejects the `-ld64` flag PyTensor adds on macOS; strip it when refused.
# Models are built at call time, so applying after the imports is early enough.
pytensor_compat.apply()
