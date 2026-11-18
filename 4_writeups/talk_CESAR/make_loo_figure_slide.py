"""LOO-ELPD comparison of the 8 model variants (paper Table 2), deck style."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
fam = "Lato" if any("Lato" in f.name for f in fm.fontManager.ttflist) else "DejaVu Sans"
plt.rcParams.update({"font.family": fam, "font.size": 17})
NAVY, GRID, MUTED = "#0D2B45", "#E3E8EE", "#6C757D"
# variant (curve, structure, likelihood), ΔELPD vs best, SE of the difference  -- Table 2 of the paper
rows = [("Harvey · joint · skew", 0.0, 0.0), ("Harvey · independent · skew", -56.2, 7.4),
        ("Harvey · joint · normal", -81.1, 12.2), ("Logistic · joint · skew", -91.5, 12.2),
        ("Logistic · independent · skew", -123.6, 14.4), ("Harvey · independent · normal", -125.1, 12.3),
        ("Logistic · joint · normal", -152.9, 15.5), ("Logistic · independent · normal", -174.7, 15.1)]
fig, ax = plt.subplots(figsize=(8.9, 4.98), dpi=300)
for i, (lab, d, se) in enumerate(rows):
    y = len(rows) - 1 - i
    if se: ax.errorbar(d, y, xerr=2 * se, fmt="none", ecolor=NAVY, elinewidth=2.5, capsize=0, alpha=0.45)
    main = i == 0
    ax.plot(d, y, "o", ms=13 if main else 11, color=NAVY, mec="white", mew=2, zorder=3)
    ax.text(d - (2 * se if se else 0) - (9 if main else 5), y, f"{d:.0f}".replace("-", "−") if d else "main model", ha="right", va="center",
            fontsize=16, color=NAVY if main else MUTED, fontweight="bold" if main else "normal")
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0] for r in rows][::-1])
for t in ax.get_yticklabels():
    if t.get_text() == rows[0][0]: t.set_fontweight("bold")
    t.set_color(NAVY)
ax.set_xlim(-245, 14)
ax.axvline(0, color=NAVY, lw=1, alpha=0.6)
ax.set_xlabel("ΔELPD vs. main model (higher is better, bars = ±2 SE)", color=NAVY)
ax.grid(axis="x", color=GRID, lw=1); ax.set_axisbelow(True)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(MUTED); ax.tick_params(axis="x", colors=MUTED); ax.tick_params(axis="y", length=0)
fig.tight_layout()
fig.savefig("figures/loo_comparison_slide.png", dpi=300, facecolor="white")
print("ok", fam)
