"""Refresh Data/benchmark_data/*.csv from the Epoch hub exports.

Default mode is REPLACE: each file becomes exactly the current Epoch export (Epoch is
trusted as the source of truth; models it drops are dropped here too).  Pass --union
to keep rows Epoch no longer publishes (rows of the same model are still updated).

External leaderboards: https://epoch.ai/data/external_benchmarks/<slug>.csv
Internal runs:        https://epoch.ai/data/benchmarks.csv (the current runs of each
                      task; Epoch re-runs its whole suite when a task version changes).
Usage:
    uv run python refresh_epoch_data.py <export_dir> [--download] [--union]

With --download the exports are fetched into <export_dir> first (benchmark_metadata.csv,
benchmarks.csv and one <slug>.csv per external benchmark mapped in notebook 0);
otherwise <export_dir> must already hold them.  Run 0_Process_benchmarks.ipynb afterwards
(its Scale scraping needs `uv run playwright install chromium` once).
"""
import json, os, re, sys
import numpy as np, pandas as pd

S = sys.argv[1]
MODE = "union" if "--union" in sys.argv[2:] else "replace"
DOWNLOAD = "--download" in sys.argv[2:]
nb = json.load(open("0_Process_benchmarks.ipynb"))
src = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
i = src.find("BENCHMARK_NAME_MAP = {"); j = src.find("}", i)
files = re.findall(r'"([a-z0-9_]+\.csv)"\s*:\s*"([^"]+)"', src[i:j])
i = src.find("BENCHMARK_SCORE_COLUMN_MAP = {"); j = src.find("\n}", i)
scoremap = dict(re.findall(r'"([a-z0-9_]+)"\s*:\s*"([^"]+)"', src[i:j]))
if DOWNLOAD:
    import urllib.request
    os.makedirs(S, exist_ok=True)
    for url, dest in [("https://epoch.ai/data/benchmark_metadata.csv", "benchmark_metadata.csv"),
                      ("https://epoch.ai/data/benchmarks.csv", "benchmarks.csv")]:
        urllib.request.urlretrieve(url, f"{S}/{dest}")
    for fn, _ in files:
        if fn.endswith("_external.csv"):
            slug = fn[: -len("_external.csv")]
            try:
                urllib.request.urlretrieve(f"https://epoch.ai/data/external_benchmarks/{slug}.csv", f"{S}/{fn}")
            except Exception as exc:  # noqa: BLE001 - report and move on, the file is then left untouched
                print(f"{fn}: pas d'export Epoch ({exc})")
meta = pd.read_csv(f"{S}/benchmark_metadata.csv")
runs = pd.read_csv(f"{S}/benchmarks.csv", low_memory=False)
# Our FrontierMath files hold the v2 series (switched 2026-09-03); the metadata still
# points their historical names at the superseded v1 tasks.
TASK_OVERRIDE = {"frontiermath.csv": "FrontierMath-Tiers-1-3-v2-Private",
                 "frontiermath_tier_4.csv": "FrontierMath-Tier-4-v2-Private"}
VARIANT_COLS = ["Harness", "Scaffold", "Agent", "Setting", "Reasoning effort", "Reasoning", "Tool setting", "Step budget", "Aggregation"]

def se(x):
    m = re.search(r"±\s*([0-9.]+)", str(x)); return float(m.group(1)) if m else np.nan

def union_update(old, new, keys):
    """Rows of `new` replace same-key rows of `old`; other old rows are kept."""
    keys = [k for k in keys if k in old.columns and k in new.columns]
    o = old.copy(); n = new.copy()
    for k in keys:
        o[k] = o[k].astype(str); n[k] = n[k].astype(str)
    o["_k"] = o[keys].agg("|".join, axis=1); n["_k"] = n[keys].agg("|".join, axis=1)
    kept = old[~o["_k"].isin(set(n["_k"]))]
    out = pd.concat([kept, new[old.columns]], ignore_index=True)
    return out, len(old) - len(kept), len(new) - (len(new) - (len(new) - len(set(n["_k"]) - set(o["_k"]))))

report = []
for fn, name in files:
    path = f"Data/benchmark_data/{fn}"
    if not os.path.exists(path) or fn == "epoch_capabilities_index.csv":
        continue
    old = pd.read_csv(path, low_memory=False)
    scol = scoremap.get(fn[:-4].replace("_external", ""))
    if fn.endswith("_external.csv"):
        if not os.path.exists(f"{S}/{fn}"):
            report.append((fn, "export absent, inchangé")); continue
        ex = pd.read_csv(f"{S}/{fn}", low_memory=False)
        ren = {"Version release date": "Release date", "Country (of organization)": "Country", "Link": "Source link"}
        ex = ex.rename(columns={k: v for k, v in ren.items() if k in ex.columns and v not in ex.columns})
        if scol and scol not in ex.columns:
            report.append((fn, f"COLONNE SCORE '{scol}' absente de l'export, inchangé")); continue
        new = pd.DataFrame({c: (ex[c] if c in ex.columns else np.nan) for c in old.columns})
        keys = ["Model version", "Name"] + [c for c in VARIANT_COLS if c in old.columns]
        if "Name" not in old.columns: keys = ["Model version"] + [c for c in VARIANT_COLS if c in old.columns]
    else:
        task = TASK_OVERRIDE.get(fn)
        if task is None:
            m = meta[meta.source_file == fn]
            if not len(m): report.append((fn, "pas dans la metadata, inchangé")); continue
            task = m.benchmark.iloc[0]
        e = runs[(runs.task == task) & (runs.Status == "Success")]
        new = pd.DataFrame({"Model version": e["model"], "mean_score": e["Best score (across scorers)"],
                            "Best score (across scorers)": e["Best score (across scorers)"],
                            "Release date": e["Version release date"], "Organization": e["Organization"],
                            "Country": e.get("Country (of organization)", np.nan),
                            "Training compute (FLOP)": e["Training compute (FLOP)"],
                            "Training compute notes": e["Training compute notes"],
                            "stderr": e["Scores"].map(se), "Log viewer": e["log viewer"], "Logs": e["logs"],
                            "Started at": e["started_at"], "id": e["id_runs"]})
        new = pd.DataFrame({c: (new[c] if c in new.columns else np.nan) for c in old.columns})
        keys = ["Model version"]
    new = new.dropna(subset=["Model version"]) if not fn.endswith("_external.csv") else new  # exports keep id-less rows (Name only)
    if MODE == "union":
        merged, replaced, _ = union_update(old, new, keys)
    else:
        merged, replaced = new.reset_index(drop=True), len(old)
    added = len(merged) - len(old)
    lostmods = sorted(set(old["Model version"].dropna().astype(str)) - set(merged["Model version"].dropna().astype(str)))
    newmods = sorted(set(merged["Model version"].dropna().astype(str)) - set(old["Model version"].dropna().astype(str)))
    chg = 0
    if scol and scol in old.columns:
        a = old.dropna(subset=["Model version"]).groupby("Model version")[scol].first()
        b = merged.dropna(subset=["Model version"]).groupby("Model version")[scol].first()
        common = a.index.intersection(b.index)
        chg = int((pd.to_numeric(a[common], errors="coerce").round(4) != pd.to_numeric(b[common], errors="coerce").round(4)).sum())
    merged.to_csv(path, index=False)
    report.append((fn, f"{len(old)}->{len(merged)} (+{added}) | nouveaux modèles: {len(newmods)} {newmods[:3]}{'…' if len(newmods) > 3 else ''} | perdus: {len(lostmods)} {lostmods[:3]}{'…' if len(lostmods) > 3 else ''} | scores modifiés: {chg}"))
for fn, r in report:
    print(f"{fn:<40} {r}")
