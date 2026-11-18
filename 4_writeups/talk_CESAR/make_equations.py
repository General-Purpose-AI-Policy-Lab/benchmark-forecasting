"""Equation images for the C&ESAR slides (matplotlib mathtext, navy, transparent)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["mathtext.fontset"] = "cm"
NAVY = "#0D2B45"
EQS = {
    "eq_harvey": r"$\mu_i(t) = \ell_i + (L_i - \ell_i)\,\left[1 - (1 - \alpha_i)\,e^{-k_i(t - \tau_i)}\right]^{\frac{1}{1-\alpha_i}}$",
    "eq_likelihood": r"$y_i(t) \sim \mathrm{SkewNormal}\left(\mu_i(t),\ \xi_i(t),\ s_i\right)$",
}
for name, tex in EQS.items():
    fig = plt.figure(figsize=(10, 1.2))
    fig.text(0.5, 0.5, tex, fontsize=30, color=NAVY, ha="center", va="center")
    fig.savefig(f"figures/{name}.png", dpi=250, transparent=True, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(name)
