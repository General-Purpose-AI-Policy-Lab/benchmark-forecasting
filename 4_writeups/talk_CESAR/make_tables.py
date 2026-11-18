"""Render the backup tables of the C&ESAR talk as PNGs sized to their slide placeholders."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent
NAVY = "#0D2B45"
plt.rcParams.update({"font.family": "Liberation Sans"})
for name, t in json.load(open(HERE / "tables.json")).items():
    W, H = t["box"]["w"], t["box"]["h"]
    rows, n = t["rows"], len(t["rows"])
    fig = plt.figure(figsize=(W, H), dpi=200); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    scale = (W - 0.4) / sum(t["widths"]); xs = [0.2]
    for w in t["widths"]: xs.append(xs[-1] + w * scale)
    rh = (H - 0.2) / n
    for i, r in enumerate(rows):
        y = 0.1 + rh * (i + 0.5)
        for j, cell in enumerate(r):
            left = t["align"][j] == "START"
            ax.text(xs[j] + 0.1 if left else xs[j + 1] - 0.1, y, cell, ha="left" if left else "right", va="center",
                    fontsize=t["size"], color=NAVY, fontweight="bold" if i == 0 else "normal")
    for yy, lw in [(0.1, 2), (0.1 + rh, 1.2), (0.1 + rh * n, 2)]:
        ax.plot([0.2, W - 0.2], [yy, yy], color=NAVY, lw=lw)
    fig.savefig(HERE / "figures" / f"{name}.png", facecolor="white"); plt.close(fig)
    print(name, W, H)
