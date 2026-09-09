"""Locks on the numbered layout: outputs per cutoff under 3_outputs/, nothing pointing at the
old Plots/ and Fits/ folders, no personal path in a tracked file."""

import re
import subprocess
from pathlib import Path

from benchmark_forecasting import config

ROOT = Path(__file__).resolve().parents[1]


def _tracked(suffixes):
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, text=True,
                         check=True).stdout
    files = [ROOT / f for f in out.split("\0") if f and f.endswith(tuple(suffixes))
             and not f.startswith(("3_outputs/", "4_writeups/", "archive/"))
             and not f.endswith("test_layout.py")]
    assert files, "git ls-files returned nothing: the locks would pass on an empty set"
    return files


def test_cutoff_dir_is_dated_first():
    assert config.cutoff_dir("cutoff20260907") == ROOT / "3_outputs" / "cutoff20260907"
    assert config.cutoff_dir("_cutoff20260907") == config.cutoff_dir("cutoff20260907")
    assert config.cutoff_dir(None) == ROOT / "3_outputs" / "no_cutoff"
    assert config.NOTE_FIGURES_DIR == ROOT / "4_writeups" / "note" / "figures"


def test_no_stale_output_paths():
    stale = re.compile(r"(?<![\w/])(Plots|Fits|Paper)/|[\"']Fits[\"']"
                       r"|1_forecasts\.py|2_revision_analyses\.py|3_Plot_forecasts|0-Note-figures"
                       r"|benchmarks_lower_bounds\.csv")
    hits = [f"{p.relative_to(ROOT)}:{i}" for p in _tracked((".py", ".md", ".toml"))
            for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
            if stale.search(line)]
    assert not hits, hits


def test_no_personal_paths_in_tracked_files():
    home = re.compile(r"/Users/[a-z]")
    hits = [str(p.relative_to(ROOT))
            for p in _tracked((".py", ".md", ".toml", ".json", ".bib", ".csv"))
            if home.search(p.read_text(encoding="utf-8", errors="ignore"))]
    assert not hits, hits
