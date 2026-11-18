"""Slide-ready PNGs of the paper figures for the C&ESAR 2026 talk."""
import subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUTS = ROOT / "3_outputs/cutoff20261001"
FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True)
DPI = 250
FONT = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
NAVY = (14, 41, 76)

def render(pdf, name):
    out = FIG / name
    subprocess.run(["pdftoppm", "-png", "-r", str(DPI), "-singlefile", str(pdf), str(out)], check=True)
    return Image.open(str(out) + ".png").convert("RGB")

cat = lambda c: OUTS / f"forecasts/forecast_{c}_en_paper_cutoff20261001.pdf"
single = {
    "fig1a_cyber": cat("Cyber"), "fig1b_autonomous_swe": cat("Autonomous_SWE"),
    "fig1c_biology": cat("Biology"), "fig1d_chemistry": cat("Chemistry"),
    "fig3a_domain_knowledge": cat("Domain_Specific_Questions"), "fig3b_general_reasoning": cat("General_Reasoning"),
    "fig3c_mathematics": cat("Mathematics"), "fig3d_agi_progress": cat("Core_AGI_Progress"),
    "fig5a_agentic": cat("Agentic_Computer_Use"), "fig5b_trivia": cat("Trivia_Commonsense_QA"),
    "fig5c_language": cat("Advanced_Language_and_Writing"), "fig5d_multimodal": cat("Multimodal_Understanding"),
    "fig2a_harvey": OUTS / "high_level/asymmetry_en_paper_cutoff20261001.pdf",
    "fig2b_saturation_2030": OUTS / "high_level/saturation_en_paper_cutoff20261001.pdf",
    "fig6_hyperparameters": OUTS / "high_level/hyperparameters_en_paper_cutoff20261001.pdf",
    "fig7_asymptotes": OUTS / "high_level/L_intervals_en_paper_cutoff20261001.pdf",
}
for v in ["harvey_joint_skew", "harvey_joint_normal", "harvey_independent_skew", "harvey_independent_normal",
          "logistic_joint_skew", "logistic_joint_normal", "logistic_independent_skew", "logistic_independent_normal"]:
    single[f"fig4_calib_{v}"] = OUTS / f"calibration/calibration_{v}_en_paper_cutoff20261001.pdf"
imgs = {k: render(p, k) for k, p in single.items()}

def grid(keys, titles, ncol, name, cell_w=1600, title_px=70, pad=30):
    cells = []
    for k in keys:
        im = imgs[k]; im = im.resize((cell_w, round(im.height * cell_w / im.width)), Image.LANCZOS); cells.append(im)
    ch = max(c.height for c in cells) + (title_px if titles else 0)
    nrow = -(-len(cells) // ncol)
    W, H = ncol * cell_w + (ncol - 1) * pad, nrow * ch + (nrow - 1) * pad
    canvas = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(canvas)
    f = ImageFont.truetype(FONT, int(title_px * 0.62))
    for i, c in enumerate(cells):
        x, y = (i % ncol) * (cell_w + pad), (i // ncol) * (ch + pad)
        if titles:
            d.text((x + cell_w / 2, y + title_px / 2), titles[i], fill=NAVY, font=f, anchor="mm")
        canvas.paste(c, (x, y + (title_px if titles else 0)))
    canvas.save(FIG / f"{name}.png", optimize=True)
    return canvas.size

print(grid(["fig1c_biology", "fig1d_chemistry"], ["Biology (dual-use)", "Chemistry (dual-use)"], 2, "fig1cd_biochem"))
print(grid(["fig3a_domain_knowledge", "fig3b_general_reasoning", "fig3c_mathematics", "fig3d_agi_progress"],
           ["Domain-specific knowledge", "General reasoning", "Mathematics", "AGI progress benchmarks"], 2, "fig3_general"))
print(grid(["fig5a_agentic", "fig5b_trivia", "fig5c_language", "fig5d_multimodal"],
           ["Agentic computer use", "Trivia & commonsense QA", "Language understanding", "Multimodal understanding"], 2, "fig5_additional"))
print(grid([k for k in imgs if k.startswith("fig4_calib")],
           ["Harvey joint (skew)", "Harvey joint (normal)", "Harvey indep. (skew)", "Harvey indep. (normal)",
            "Logistic joint (skew)", "Logistic joint (normal)", "Logistic indep. (skew)", "Logistic indep. (normal)"],
           4, "fig4_calibration_all", cell_w=900, title_px=60))
mosaic = ["fig1a_cyber", "fig1b_autonomous_swe", "fig1c_biology", "fig1d_chemistry", "fig3a_domain_knowledge",
          "fig3b_general_reasoning", "fig3c_mathematics", "fig3d_agi_progress", "fig5a_agentic", "fig5b_trivia",
          "fig5c_language", "fig5d_multimodal"]
print(grid(mosaic, ["Cyber", "Autonomous SWE", "Biology", "Chemistry", "Domain knowledge", "General reasoning",
                    "Mathematics", "AGI progress", "Agentic computer use", "Trivia & commonsense", "Language", "Multimodal"],
           4, "mosaic_12_domains", cell_w=1000, title_px=80))
for k, im in imgs.items():
    print(k, im.size)
