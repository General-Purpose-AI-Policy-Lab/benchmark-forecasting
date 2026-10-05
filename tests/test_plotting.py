"""Locks on the figure conventions: one baseline-marker scheme for every figure, and a
vector copy beside every French raster."""

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from benchmark_forecasting import config, plotting  # noqa: E402
from benchmark_forecasting.data import load_baselines  # noqa: E402

# Branches per star, by expertise level: a committee takes the count of the humans it is
# made of, so the reader reads the level off the marker and the group off the inline label.
EXPECTED_BRANCHES = {
    "Average Human": 3,
    "Skilled Generalist": 4,
    "Domain Expert": 5,
    "Top Performer": 6,
    "Committee of Average Humans": 3,
    "Committee of Skilled Generalists": 4,
    "Committee of Domain Experts": 5,
    "Committee of Top Performers": 6,
    "High School Qualifier": 5,
    "High School Top Performer": 6,
}


def test_baseline_markers_are_stars_numbered_by_expertise():
    markers = plotting._assign_marker_to_baselines(
        pd.DataFrame({"group": list(EXPECTED_BRANCHES)})
    )
    assert list(markers) == [(n, 1, 0) for n in EXPECTED_BRANCHES.values()]


@pytest.mark.parametrize("document_type", ["paper", "note"])
def test_the_marker_scheme_does_not_depend_on_the_document(document_type):
    """The note used to flatten every baseline to one star; both documents now agree."""
    style = plotting.PlotStyle(language="fr", document_type=document_type)
    observed = pd.DataFrame(
        {
            "benchmark": ["B"] * 3,
            "category": ["C"] * 3,
            "release_date": pd.to_datetime(["2024-01-01", "2024-07-01", "2025-01-01"]),
            "score": [0.2, 0.5, 0.7],
        }
    )
    forecast = pd.DataFrame(
        {
            "benchmark": ["B"] * 3,
            "category": ["C"] * 3,
            "release_date": pd.to_datetime(["2024-01-01", "2025-01-01", "2026-01-01"]),
            "mu_mean": [0.2, 0.7, 0.9],
            "mu_lower": [0.1, 0.6, 0.8],
            "mu_upper": [0.3, 0.8, 1.0],
        }
    )
    baselines = pd.DataFrame(
        {"benchmark": ["B", "B"], "group": ["Average Human", "Domain Expert"], "score": [0.4, 0.8]}
    )
    fig, ax = plotting.plot_forecasts_by_category(
        observed=observed, forecast=forecast, baselines=baselines,
        end_date=pd.Timestamp("2026-01-01"), category_name="C", plot_style=style,
    )
    # A star with n branches is a path of 2n + 1 vertices; the observed scores are circles.
    drawn = {len(c.get_paths()[0].vertices) for c in ax.collections if len(c.get_offsets())}
    plt.close(fig)
    expected = {2 * EXPECTED_BRANCHES[g] + 1 for g in baselines["group"]}
    assert expected <= drawn, (expected, drawn)


@pytest.mark.skipif(
    not (config.INPUT_DIR / config.BASELINES_FILE).exists(), reason="0_input not synced"
)
def test_every_group_in_the_synced_baselines_has_a_marker_and_a_label():
    groups = set(load_baselines()["group"].dropna())
    assert groups <= set(EXPECTED_BRANCHES), groups - set(EXPECTED_BRANCHES)
    markers = plotting._assign_marker_to_baselines(pd.DataFrame({"group": sorted(groups)}))
    assert "x" not in set(markers), "a group fell back to the unlabelled cross marker"


def test_save_figure_writes_the_vector_copy_in_an_svg_subfolder(tmp_path):
    fig = plt.figure()
    png = tmp_path / "fr" / "forecast_Biology_fr_note_cutoff20260907.png"
    plotting.save_figure(fig, png, dpi=50, also_svg=True)
    plt.close(fig)
    assert png.exists()
    svg = png.parent / "svg" / f"{png.stem}.svg"
    assert svg.exists() and svg.read_text(encoding="utf-8").lstrip().startswith("<?xml")


def test_save_figure_writes_no_svg_by_default(tmp_path):
    fig = plt.figure()
    png = tmp_path / "forecast_en_paper.pdf"
    plotting.save_figure(fig, png, dpi=50)
    plt.close(fig)
    assert png.exists() and not (png.parent / "svg").exists()


def test_asymptote_panels_split_the_ranking_top_to_bottom_then_left_to_right():
    """Two panels hold every benchmark once, the first one the highest asymptotes."""
    import arviz as az
    import numpy as np

    names = [f"B{i}" for i in range(5)]
    # Benchmark i has its asymptote near 0.8 + 0.04 i, so the ranking is B4, B3, ..., B0.
    draws = 0.8 + 0.04 * np.arange(5)[None, None, :] + np.zeros((1, 10, 5))
    idata = az.from_dict(posterior={"L": draws}, coords={"benchmark": names},
                         dims={"L": ["benchmark"]})
    fig, axes = plotting.plot_L_intervals(idata, n_columns=2)

    def top_to_bottom(ax):
        return [t.get_text() for t in ax.get_yticklabels()][::-1]

    assert len(axes) == 2
    assert top_to_bottom(axes[0]) + top_to_bottom(axes[1]) == ["B4", "B3", "B2", "B1", "B0"]
    # Same row pitch in both panels despite their different lengths.
    assert np.ptp(axes[0].get_ylim()) == np.ptp(axes[1].get_ylim())
    plt.close(fig)
