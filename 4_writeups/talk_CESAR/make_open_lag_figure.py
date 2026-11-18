"""Open-weights lag behind the closed frontier, public vs private benchmarks only (slide 16).

Same figure as Multiaxis_ECI/5_outputs/open_closed_frontier/frontier_gap_openness_lag.png
(data generation 2026-10-19, today line at 1 October 2026, the data cutoff, where the original draws it), drawn with two of its four curves.
plotly / pymc are only imported, never used by the print figure: stubbed so the system python runs it.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock
for m in ["pymc", "pytensor", "pytensor.tensor", "nutpie", "plotly", "plotly.graph_objects",
          "plotly.subplots", "plotly.io", "kaleido"]:
    sys.modules.setdefault(m, MagicMock())
ECI = Path(__file__).resolve().parents[3] / "Multiaxis_ECI"
sys.path.insert(0, str(ECI / "2_model"))
from multiaxis_eci import config  # noqa: E402
from multiaxis_eci.analysis.frontier_gap import load_scope  # noqa: E402
from multiaxis_eci.viz import frontier_gap_print as fgp  # noqa: E402
from multiaxis_eci.viz.frontier_gap_print import lag_fig_print, save_print  # noqa: E402
# Helvetica Neue is not installed outside macOS: its metric clones first
fgp.STYLE["font.sans-serif"] = ["Helvetica Neue", "Nimbus Sans", "Nimbus Sans L", "Liberation Sans", "Arial", "DejaVu Sans"]

scopes = ["public", "private"]
results = {s: load_scope(config.COMPARISONS_DIR, "openness", s) for s in scopes}
fig = lag_fig_print(results, scopes, today="2026-10-01", label_scope="public", group="openness",
                    lang="en", counts={"public": 61, "private": 19})
out = Path(__file__).parent / "figures"
save_print(fig, out / "open_lag_public_private.png", out / "open_lag_public_private.svg")
print("wrote", out / "open_lag_public_private.png")
