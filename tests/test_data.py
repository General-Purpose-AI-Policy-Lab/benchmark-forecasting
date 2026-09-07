"""Frontier selection and the per-benchmark asymptote bounds."""

import numpy as np
import pandas as pd
import pytest

from benchmark_forecasting import config
from benchmark_forecasting.data import (
    asymptote_bounds,
    load_dataset,
    model_base_key,
    prepare_dataset,
    select_frontier_points,
)


@pytest.mark.parametrize(
    "names",
    [
        ("gpt-5.2_high", "gpt-5.2_xhigh", "gpt-5.2_low"),
        ("Claude Sonnet 4.5 (Thinking)", "Claude Sonnet 4.5 (Non-Thinking)"),
        ("LLaMA-7B", "LLaMA-65B"),
        ("claude-opus-4-5-20251101", "claude-opus-4-5"),
    ],
)
def test_model_base_key_merges_efforts_sizes_and_dates(names):
    assert len({model_base_key(n) for n in names}) == 1


def test_model_base_key_keeps_product_tiers_apart():
    assert model_base_key("gpt-4.1") != model_base_key("gpt-4")
    assert model_base_key("gemini-2.5-pro") != model_base_key("gemini-2.5-flash")


def _frame():
    return pd.DataFrame(
        {
            "benchmark": ["B"] * 5,
            "release_date": pd.to_datetime(
                ["2024-01-01", "2024-02-01", "2024-02-01", "2024-03-01", "2024-04-01"]
            ),
            "score": [0.2, 0.5, 0.4, 0.3, 0.6],
            "model_version": ["a", "b_high", "b_low", "c", "d"],
            "lower_bound": [0.0] * 5,
        }
    )


def test_select_frontier_points_keeps_the_expanding_top_n_once_per_family():
    top1 = select_frontier_points(_frame(), top_n=1)
    assert top1["model_version"].tolist() == ["a", "b_high", "d"], "b_low merged into b, c below"
    all_points = select_frontier_points(_frame(), top_n=1, dedupe_effort=False)
    assert "b_low" not in all_points["model_version"].tolist(), "b_low is not top-1 either way"
    assert len(select_frontier_points(_frame(), top_n=3)) == 4, "five rows, b counted once"


def test_prepare_dataset_adds_days_from_the_first_point():
    prepared = prepare_dataset(_frame(), top_n=3)
    assert prepared["days"].iloc[0] == 0 and prepared["days"].max() == 91
    assert (prepared["days_mid"] == 45.5).all()


def _bounds_frame():
    return pd.DataFrame(
        {
            "benchmark": ["Free", "Human", "Perfect", "Ceiled", "Low", "Solved", "Beaten"],
            "human_max": [np.nan, 0.9, 1.0, 0.8, 0.5, np.nan, 0.85],
            "ceiling": [np.nan, np.nan, np.nan, 0.95, np.nan, np.nan, np.nan],
            "score": [0.3, 0.5, 0.4, 0.6, 0.2, 1.0, 0.92],
        }
    )


def test_asymptote_bounds_apply_the_three_rules():
    bounds = asymptote_bounds(_bounds_frame(), config.ModelConfig())
    assert bounds.loc["Free", "L_floor"] == 0.75 and np.isnan(bounds.loc["Free", "L_fixed"])
    assert bounds.loc["Human", "L_floor"] == 0.9 and np.isnan(bounds.loc["Human", "L_fixed"])
    assert bounds.loc["Perfect", "L_fixed"] == 1.0
    assert bounds.loc["Ceiled", "L_fixed"] == 0.95 and bounds.loc["Ceiled", "L_floor"] == 0.8
    assert bounds.loc["Low", "L_floor"] == 0.75, "a baseline below L_min changes nothing"
    assert bounds.loc["Solved", "L_fixed"] == 1.0, "a score of 1.0 pins the asymptote"
    assert bounds.loc["Beaten", "L_floor"] == 0.92, "the best score beats the human baseline"
    assert bounds.loc["Beaten", "reason"] == "floor: best model score"


def test_asymptote_bounds_switches_and_manual_pins():
    off = config.ModelConfig(L_floor_observed=False, L_fixed_from_ceiling=False)
    bounds = asymptote_bounds(_bounds_frame(), off)
    assert (bounds["L_floor"] == 0.75).all() and bounds["L_fixed"].isna().all()
    manual = asymptote_bounds(_bounds_frame(), config.ModelConfig(L_fixed=(("Ceiled", 0.9),)))
    assert manual.loc["Ceiled", "L_fixed"] == 0.9, "ModelConfig.L_fixed wins over the ceiling"
    with pytest.raises(ValueError, match="above the ceiling"):
        asymptote_bounds(
            _bounds_frame().assign(ceiling=[np.nan, 0.85, np.nan, 0.95, np.nan, np.nan, np.nan]),
            config.ModelConfig(),
        )


def test_slug_names_the_asymptote_rules():
    assert config.ModelConfig().slug == "harvey_joint_skew_Lobs_Lceil"
    plain = config.ModelConfig(L_floor_observed=False, L_fixed_from_ceiling=False)
    assert plain.slug == "harvey_joint_skew"


@pytest.mark.skipif(
    not (config.INPUT_DIR / config.SCORES_FILE).exists(), reason="0_input not synced"
)
def test_shipped_input_loads_with_human_baselines_attached():
    raw = load_dataset()
    assert {
        "benchmark",
        "release_date",
        "score",
        "lower_bound",
        "category",
        "human_max",
        "ceiling",
    } <= set(raw.columns)
    assert raw[["release_date", "score", "lower_bound"]].notna().all().all()
    assert raw["human_max"].notna().any() and raw["ceiling"].notna().any()
