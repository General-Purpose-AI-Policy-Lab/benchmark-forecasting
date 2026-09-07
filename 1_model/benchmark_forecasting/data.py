"""Loading the pipeline's views, the frontier selection and the per-benchmark asymptote bounds.

Input columns (``3_views/dated_scores_flat.csv`` of benchmark-data-pipeline): ``benchmark``,
``release_date``, ``score``, ``lower_bound`` are required; ``category``, ``model_version``,
``source`` and ``ceiling`` are carried along. ``load_dataset`` adds ``human_max``, the highest
human baseline of the benchmark, from ``human_baselines.csv``.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark_forecasting.config import (
    BASELINES_FILE,
    EXCLUDED_BENCHMARKS,
    INPUT_DIR,
    SCORES_FILE,
    ModelConfig,
)


def load_baselines(path: str | Path = INPUT_DIR / BASELINES_FILE) -> pd.DataFrame:
    """Human baselines, one row per published measurement: benchmark, group, score, note, source."""
    baselines = pd.read_csv(path)
    baselines["score"] = pd.to_numeric(baselines["score"], errors="coerce")
    return baselines.dropna(subset=["benchmark", "score"]).reset_index(drop=True)


def load_dataset(
    path: str | Path = INPUT_DIR / SCORES_FILE,
    baselines_path: str | Path | None = INPUT_DIR / BASELINES_FILE,
    *,
    excluded: tuple[str, ...] = EXCLUDED_BENCHMARKS,
) -> pd.DataFrame:
    """Load the dated scores view, normalise types and attach ``human_max`` per benchmark."""
    df = pd.read_csv(path)

    if "category" in df.columns:
        df["category"] = df["category"].astype("string")
    df["benchmark"] = df["benchmark"].astype("string")
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["lower_bound"] = pd.to_numeric(df["lower_bound"], errors="coerce")
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    if "ceiling" in df.columns:
        df["ceiling"] = pd.to_numeric(df["ceiling"], errors="coerce")

    df = df.dropna(subset=["benchmark", "release_date", "score", "lower_bound"])
    df = df.loc[~df["benchmark"].isin(excluded)].reset_index(drop=True)

    if baselines_path is not None and Path(baselines_path).exists():
        human_max = load_baselines(baselines_path).groupby("benchmark")["score"].max()
        df["human_max"] = df["benchmark"].astype(str).map(human_max).astype(float)
    return df


def asymptote_bounds(frame: pd.DataFrame, cfg: ModelConfig) -> pd.DataFrame:
    """Per-benchmark floor and pinned value of the upper asymptote L, one row per benchmark.

    ``L_floor`` is ``cfg.L_min`` raised, when ``cfg.L_floor_observed``, to the best performance
    observed on the benchmark: its highest human baseline or its best model score in ``frame``.
    ``L_fixed`` is NaN for an estimated asymptote; otherwise the pinned value, from ``cfg.L_fixed``
    first, then the pipeline's ``ceiling`` when ``cfg.L_fixed_from_ceiling``, then 1.0 when the
    floor already reaches 1.
    ``reason`` says which rule applied.
    """
    benchmarks = sorted(frame["benchmark"].astype(str).unique())
    per_bench = frame.assign(benchmark=frame["benchmark"].astype(str)).groupby("benchmark")
    human_max = (
        per_bench["human_max"].max() if "human_max" in frame.columns else pd.Series(dtype=float)
    )
    ceiling = per_bench["ceiling"].max() if "ceiling" in frame.columns else pd.Series(dtype=float)
    best = per_bench["score"].max() if "score" in frame.columns else pd.Series(dtype=float)
    fixed_map = dict(cfg.L_fixed)

    rows = []
    for bench in benchmarks:
        floor, fixed, reason = cfg.L_min, np.nan, "estimated"
        hm = human_max.get(bench, np.nan)
        bs = best.get(bench, np.nan)
        if cfg.L_floor_observed:
            if pd.notna(hm) and hm > floor:
                floor, reason = float(hm), "floor: highest human baseline"
            if pd.notna(bs) and bs > floor:
                floor, reason = float(bs), "floor: best model score"
        ceil = ceiling.get(bench, np.nan)
        if bench in fixed_map:
            fixed, reason = float(fixed_map[bench]), "pinned: ModelConfig.L_fixed"
        elif cfg.L_fixed_from_ceiling and pd.notna(ceil):
            fixed, reason = float(ceil), "pinned: known ceiling"
        elif floor >= 1.0 - 1e-9:
            fixed, reason = 1.0, f"pinned: {reason.removeprefix('floor: ')} at 1"
        if pd.notna(fixed):
            if not cfg.L_min < fixed <= 1.0:
                raise ValueError(f"L_fixed[{bench!r}]={fixed} must lie in (L_min={cfg.L_min}, 1].")
            if floor > fixed + 1e-9:
                raise ValueError(f"{bench!r}: human baseline {floor} above the ceiling {fixed}.")
        rows.append({"benchmark": bench, "L_floor": floor, "L_fixed": fixed, "reason": reason})
    return pd.DataFrame(rows).set_index("benchmark")


_EFFORT_TOKENS = (
    r"(?:xhigh|x-high|high|medium|med|low|minimal|promax|none|default|unknown|instant|"
    r"thinking|non-thinking|no-thinking|nothinking|non-reasoning|reasoning|adaptive\s+thinking|"
    r"thinking\s*\d+k?|reasoning\s*\d+k?|budget\s*\d+k?|effort\s*=\s*\w+|\d+k)"
)


def model_base_key(name: str) -> str:
    """Identity of a model family once effort, thinking mode and size are removed.

    Leaderboards publish the same model at several reasoning efforts on the same day
    (``gpt-5.2_high`` / ``_xhigh`` / ``_low``, ``claude-sonnet-4-5_16K`` / ``_32K``,
    ``(Thinking)`` / ``(Non-Thinking)``, ``-instant`` / ``-thinking``) and open
    families at several parameter counts (``LLaMA-7B`` … ``65B``, ``Qwen3-235B-A22B``,
    ``PaLM 2-S/M/L``).  Counted separately they fill the top-N frontier with one
    release.  The key strips those markers, dated id fragments (the release date is
    matched separately), ``pre-release`` / ``preview`` qualifiers and source tags,
    then lower-cases and drops punctuation except dots (``gpt-4.1`` vs ``gpt-4``).
    Product tiers (``-pro``, ``-mini``, ``-flash``, ``-air``, Qwen ``-max``) are kept.
    Audited against the September 2026 dataset (same-day groups per benchmark).
    """
    s = str(name).strip().rstrip("†*").strip()
    s = re.sub(r"_anthropic$", "", s, flags=re.I)  # source tag on some Scale rows
    s = re.sub(r"[-_ ]?(?:20\d{2}-?\d{2}-?\d{2})(?=[_\s(-]|$)", "", s)  # dated ids
    s = re.sub(r"\s*\(\d{1,2}/\d{1,2}\)", "", s)  # "Gemini 3 Deep Think (2/26)"
    s = re.sub(r"[-_ ]pre-?release", "", s, flags=re.I)
    s = re.sub(r"[-_ ]preview(?=[-_\s(]|$)", "", s, flags=re.I)
    # Parameter counts: "-7B", " 0.5B", "(7B)", "-235B-A22B", "(64B/64E)", "PaLM 2-L",
    # "-small" / "-large".
    s = re.sub(r"\s*\(\d+(?:\.\d+)?[bB](?:/\d+[eE])?\)", "", s)
    s = re.sub(r"[-_ ]?\d+(?:\.\d+)?[bB](?:[-_ ]?[aA]\d+(?:\.\d+)?[bB])?(?=[-_\s(]|$)", "", s)
    s = re.sub(r"(?<=PaLM 2)-[SML]\b", "", s)
    s = re.sub(r"[-_ ](?:small|large)(?=[-_\s(]|$)", "", s, flags=re.I)
    prev = None
    while prev != s:
        prev = s
        s = re.sub(
            r"[_\s-]*\(\s*(?:reasoning[\s_-]*)?" + _EFFORT_TOKENS + r"(?:\s+thinking)?\s*\)\s*$",
            "",
            s,
            flags=re.I,
        ).strip()
        s = re.sub(
            r"[_\s-]+(?:reasoning[\s_-]*)?" + _EFFORT_TOKENS + r"\s*$", "", s, flags=re.I
        ).strip()
        # "max" only in the Epoch/Scale effort spellings ("_max", " max", "(max)"),
        # never "-max", which is a Qwen product tier.
        s = re.sub(r"(?:[_\s]+|\(\s*)max\s*\)?\s*$", "", s, flags=re.I).strip()
        s = re.sub(r"-thinking-max$", "", s, flags=re.I).strip()
    return re.sub(r"[^a-z0-9.]", "", s.lower())


def select_frontier_points(
    df: pd.DataFrame, top_n: int, *, dedupe_effort: bool = True
) -> pd.DataFrame:
    """Keep points that are within top_n of the expanding best-so-far per benchmark.

    With ``dedupe_effort`` (the default for fitting), a model family published at
    several efforts or sizes on the same day counts once, through its best score,
    before the top-N selection.  Pass ``dedupe_effort=False`` for the display
    frontier, which keeps every published point.
    """
    required = {"benchmark", "release_date", "score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"select_frontier_points: missing required columns: {sorted(missing)}")

    d = df.sort_values(["benchmark", "release_date"]).copy()
    if dedupe_effort and "model_version" in d.columns:
        d["_base"] = d["model_version"].map(model_base_key)
        d = (
            d.sort_values("score", ascending=False, kind="stable")
            .drop_duplicates(["benchmark", "release_date", "_base"], keep="first")
            .drop(columns="_base")
            .sort_values(["benchmark", "release_date"])
        )
    d["expanding_rank"] = (
        d.groupby("benchmark")["score"]
        .expanding()
        .rank(ascending=False, method="max")
        .reset_index(level=0, drop=True)
    )
    d = d.loc[d["expanding_rank"] <= top_n].drop(columns=["expanding_rank"]).reset_index(drop=True)
    return d


def prepare_dataset(df: pd.DataFrame, *, top_n: int, dedupe_effort: bool = True) -> pd.DataFrame:
    """Prepare dataset for modeling (frontier + time features).

    ``dedupe_effort=False`` gives the display variant (all same-day effort and size
    variants kept); see :func:`select_frontier_points`.
    """
    d = select_frontier_points(df, top_n=top_n, dedupe_effort=dedupe_effort).copy()

    first_dates = d.groupby("benchmark")["release_date"].transform("min")
    d["days"] = (d["release_date"] - first_dates).dt.days.astype(int)

    max_days = d.groupby("benchmark")["days"].transform("max").astype(float)
    d["days_mid"] = max_days / 2.0
    return d


def _ensure_constant_per_benchmark(df: pd.DataFrame, col: str) -> None:
    """Raise if a column varies within any benchmark group."""
    nunique = df.groupby("benchmark")[col].nunique(dropna=False)
    bad = nunique[nunique > 1]
    if not bad.empty:
        examples = bad.index[:10].tolist()
        raise ValueError(
            f"Column '{col}' must be constant within benchmark. Violations (first 10): {examples}"
        )
