"""Data ingestion and preprocessing pipeline for benchmark forecasting."""

import pandas as pd

def load_benchmarks(
    path: str = "benchmark_data_processed/all_normalized_updated_benchmarks.csv",
) -> pd.DataFrame:
    """Loads and strictly types the raw benchmark dataset."""
    df = pd.read_csv(path).astype(
        {
            "benchmark": "string",
            "release_date": "datetime64[ns]",
            "score": "float64",
            "lower_bound": "float64",
        }
    )
    return df.dropna(subset=["benchmark", "release_date", "score", "lower_bound"])


def _compute_days_since_inception(df: pd.DataFrame) -> pd.DataFrame:
    """Adds a relative time column 'days' starting from 0 for each benchmark."""
    df = df.copy()
    inception_dates = df.groupby("benchmark")["release_date"].transform("min")
    df["days"] = (df["release_date"] - inception_dates).dt.days
    return df


def filter_to_frontier(df: pd.DataFrame, top_n: int = 1) -> pd.DataFrame:
    """Retains only the top N scores per benchmark over time (the 'frontier')."""
    df_sorted = df.sort_values(["benchmark", "release_date"])
    
    # Calculate rank within an expanding window to identify historical SOTA
    expanding_rank = (
        df_sorted.groupby("benchmark")["score"]
        .expanding()
        .rank(ascending=False, method="max")
        .reset_index(level=0, drop=True)
    )
    
    return df_sorted[expanding_rank <= top_n].reset_index(drop=True)


def preprocess_for_modeling(df: pd.DataFrame, top_n_frontier: int = 3) -> pd.DataFrame:
    """Prepares the raw data for the Bayesian model.
    
    Filters for the frontier, calculates relative days, and computes the 
    midpoint for sigmoid priors.
    """
    df_clean = filter_to_frontier(df, top_n=top_n_frontier)
    df_clean = _compute_days_since_inception(df_clean)

    # Midpoint of the observed time range, used to center the inflection point prior
    max_days = df_clean.groupby("benchmark")["days"].transform("max")
    df_clean["days_midpoint"] = max_days / 2.0

    return df_clean


def create_forecast_grid(
    history_df: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    periods: int = 100
) -> pd.DataFrame:
    """Generates a future time grid for each benchmark in the history."""
    
    def _generate_benchmark_grid(group: pd.DataFrame) -> pd.DataFrame:
        start_date = group["release_date"].min()
        future_dates = pd.date_range(start=start_date, end=cutoff_date, periods=periods)
        
        return pd.DataFrame({
            "release_date": future_dates,
            "days": (future_dates - start_date).days,
            "benchmark": group["benchmark"].iloc[0],
            # Propagate static metadata if present
            "category": group["category"].iloc[0] if "category" in group else None,
        })

    return (
        history_df.groupby("benchmark", group_keys=False)
        .apply(_generate_benchmark_grid)
        .reset_index(drop=True)
    )