"""Timeline of projected saturation dates by capability category (C&ESAR 2026 talk).

Draws Table 1 of the paper as a picture: one row per category, ordered by the latest
median saturation date, one dot per benchmark (posterior median), a bar from the
earliest to the latest benchmark of the category.
"""
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "3_outputs/cutoff20261001/sensitivity/benchmark_details_cutoff20261001.csv"
OUT = Path(__file__).resolve().parent / "figures"

NAVY, GOLD, GRAY, LIGHT = "#0e294c", "#d4af37", "#9aa3ad", "#e9eef4"
SECURITY = {"Cyber", "Biology", "Chemistry", "Autonomous SWE"}
TODAY = pd.Timestamp("2026-11-18")
SHORT = {"Trivia & Commonsense QA": "Trivia & commonsense QA", "Core AGI Progress": "AGI progress",
         "Advanced Language and Writing": "Language & writing", "Domain Specific Questions": "Domain knowledge",
         "Multimodal Understanding": "Multimodal", "Agentic Computer Use": "Agentic computer use",
         "General Reasoning": "General reasoning", "Autonomous SWE": "Autonomous SWE"}

plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 20})

df = pd.read_csv(CSV, parse_dates=["sat_median"])
order = df.groupby("category")["sat_median"].max().sort_values().index.tolist()

fig, ax = plt.subplots(figsize=(16, 8.4), dpi=200)
ax.axvspan(pd.Timestamp("2021-07-01"), TODAY, color=LIGHT, zorder=0)
for i, cat in enumerate(order):
    sub = df[df.category == cat]
    sec = cat in SECURITY
    col = NAVY if sec else GRAY
    ax.plot([sub.sat_median.min(), sub.sat_median.max()], [i, i], lw=9, color=col,
            alpha=0.25, solid_capstyle="round", zorder=1)
    ax.scatter(sub.sat_median, [i] * len(sub), s=95, color=col, edgecolor="white",
               linewidth=1.2, zorder=3)
    label = f"{SHORT.get(cat, cat)}  ({len(sub)})"
    ax.text(pd.Timestamp("2021-06-01"), i, label, ha="right", va="center", fontsize=19,
            color=NAVY if sec else "#4a5560", fontweight="bold" if sec else "normal")

ax.axvline(TODAY, color=GOLD, lw=3, zorder=2)
ax.text(TODAY, -0.95, "Today", color="#a8861f", ha="center", va="bottom", fontsize=19,
        fontweight="bold")
ax.axvline(pd.Timestamp("2030-01-01"), color=NAVY, lw=2, ls=(0, (5, 4)), zorder=2)
ax.text(pd.Timestamp("2030-01-01"), -0.95, "2030", color=NAVY, ha="center", va="bottom",
        fontsize=19, fontweight="bold")

n_before = int((df.sat_median < TODAY).sum())
ax.text(pd.Timestamp("2021-09-01"), len(order) - 0.35,
        f"{n_before} / {len(df)} benchmarks\nalready saturated", ha="left", va="bottom",
        fontsize=18, color="#4a5560")

ax.set_ylim(len(order) - 0.2 + 1.0, -1.1)
ax.set_xlim(pd.Timestamp("2021-07-01"), pd.Timestamp("2030-09-01"))
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.tick_params(axis="x", labelsize=19, colors="#4a5560", length=0, pad=8)
ax.set_yticks([])
for s in ["top", "right", "left"]:
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color("#c5ccd4")
ax.grid(axis="x", color="#dde3ea", lw=1, zorder=0)
fig.subplots_adjust(left=0.25, right=0.985, top=0.97, bottom=0.08)
OUT.mkdir(exist_ok=True)
fig.savefig(OUT / "frise_saturation_by_category.png")
print("saved", n_before, "before today;", order)
