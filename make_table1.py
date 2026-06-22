#!/usr/bin/env python
"""Render Table 1 (qualitative steering examples) as a booktabs-style PNG."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

# ---- Font: register Times New Roman explicitly -------------------------------
TNR = "/System/Library/Fonts/Supplemental/Times New Roman.ttf"
TNR_B = "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf"
TNR_I = "/System/Library/Fonts/Supplemental/Times New Roman Italic.ttf"
TNR_BI = "/System/Library/Fonts/Supplemental/Times New Roman Bold Italic.ttf"
for p in (TNR, TNR_B, TNR_I, TNR_BI):
    fm.fontManager.addfont(p)
FAMILY = fm.FontProperties(fname=TNR).get_name()  # "Times New Roman"
plt.rcParams["font.family"] = FAMILY
plt.rcParams["font.serif"] = [FAMILY, "DejaVu Serif"]
plt.rcParams["axes.unicode_minus"] = False

# ---- Colors ------------------------------------------------------------------
ACCENT = {"neg": "#1F77B4", "rand": "#FF7F0E", "base": "#6C757D"}
TINT = {"neg": "#D6E6F4", "rand": "#FDE7D1", "base": "#EDEDF0"}
HDR_BG = "#2F3B52"
BODY = "#1A1A1A"
RULE = "#000000"

# ---- Content -----------------------------------------------------------------
# Each block: feature, subtitle, prompt, then 3 (cond_kind, cond_label, cont)
BLOCKS = [
    ("f4191", "pepper", "“…salt and ground black ___”", [
        ("base", "Baseline", "“pepper, then added fresh basil leaves…”"),
        ("neg", "Negative (c=2)", "“rice, then basil and a few drops of olive oil…”"),
        ("rand", "Random (c=4)", "“pepper, then a cup of water to simmer…”"),
    ]),
    ("f8639", "egg", "“…a plate of scrambled ___”", [
        ("base", "Baseline", "“eggs, toast and bacon.”"),
        ("neg", "Negative (c=0.5)", "“rice, two pieces of fried fish, a plate of rice…”"),
        ("rand", "Random (c=1)", "“eggs, toast and a cup of coffee…”"),
    ]),
    ("f5568", "goal, inert", "“…scored the winning ___”", [
        ("base", "Baseline", "“goal for the United States, a 1–0 victory…”"),
        ("neg", "Negative (c=4)", "“goal … a 1–0 victory…”  (unchanged)"),
        ("rand", "Random (c=4)", "“goal … at Tim Howard.”"),
    ]),
]

COLS = ["Feature", "Prompt (concept-tail)", "Condition", "Continuation"]
# left edges + widths (fractions of axis), sum of widths = 1.0
WIDTHS = [0.11, 0.27, 0.14, 0.48]
XPAD = 0.008  # left text padding inside cell (axis fraction)
LEFTS = []
acc = 0.0
for w in WIDTHS:
    LEFTS.append(acc)
    acc += w

# ---- Geometry ----------------------------------------------------------------
N_BODY = 9
ROW_H = 1.0          # row height in data units
HDR_H = 1.15
TOP = N_BODY * ROW_H + HDR_H   # y of very top
FIG_W = 12.5
# height: rows + header + title space, proportional
FIG_H = 0.46 * (N_BODY + 1) + 1.0

fig = plt.figure(figsize=(FIG_W, FIG_H))
ax = fig.add_axes([0.012, 0.02, 0.976, 0.88])
ax.set_xlim(0, 1)
ax.set_ylim(0, TOP)
ax.axis("off")

FS_BODY = 12.5
FS_HDR = 13.0
FS_TITLE = 14.5

def cell_text(x_left, y_center, s, color=BODY, weight="normal",
              style="normal", size=FS_BODY):
    ax.text(x_left + XPAD, y_center, s, ha="left", va="center",
            color=color, fontsize=size, fontweight=weight, fontstyle=style,
            family=FAMILY, clip_on=False)

# y of top of header row
y = TOP
# ---- Header background --------------------------------------------------------
ax.add_patch(plt.Rectangle((0, y - HDR_H), 1.0, HDR_H, facecolor=HDR_BG,
                           edgecolor="none", zorder=0))
hdr_yc = y - HDR_H / 2
for i, c in enumerate(COLS):
    cell_text(LEFTS[i], hdr_yc, c, color="white", weight="bold", size=FS_HDR)

# ---- Body rows ----------------------------------------------------------------
def draw_row(y_top, kind, feat, sub, prompt, cond_label, cont):
    yc = y_top - ROW_H / 2
    # row tint background across all 4 cells
    ax.add_patch(plt.Rectangle((0, y_top - ROW_H), 1.0, ROW_H,
                               facecolor=TINT[kind], edgecolor="none", zorder=0))
    # Feature cell (name bold + italic subtitle below) only when feat given
    if feat is not None:
        ax.text(LEFTS[0] + XPAD, yc + 0.13, feat, ha="left", va="center",
                color=BODY, fontsize=FS_BODY, fontweight="bold",
                family=FAMILY, clip_on=False)
        ax.text(LEFTS[0] + XPAD, yc - 0.20, sub, ha="left", va="center",
                color=BODY, fontsize=FS_BODY - 1.5, fontstyle="italic",
                family=FAMILY, clip_on=False)
    if prompt is not None:
        cell_text(LEFTS[1], yc, prompt)
    # Condition (accent color, bold)
    cell_text(LEFTS[2], yc, cond_label, color=ACCENT[kind], weight="bold")
    # Continuation
    cell_text(LEFTS[3], yc, cont)

y = TOP - HDR_H
for bi, (feat, sub, prompt, rows) in enumerate(BLOCKS):
    for ri, (kind, cond, cont) in enumerate(rows):
        show_feat = feat if ri == 1 else None       # middle row only
        show_sub = sub if ri == 1 else None
        show_prompt = prompt if ri == 1 else None
        draw_row(y, kind, show_feat, show_sub, show_prompt, cond, cont)
        y -= ROW_H

# ---- Booktabs rules -----------------------------------------------------------
LW_THICK = 1.8
LW_THIN = 0.8
def rule(yv, lw):
    ax.plot([0, 1.0], [yv, yv], color=RULE, lw=lw, zorder=5,
            solid_capstyle="butt")

rule(TOP, LW_THICK)                 # top rule (above header)
rule(TOP - HDR_H, LW_THICK)         # under header
# thin rule between feature groups (after rows 3 and 6)
rule(TOP - HDR_H - 3 * ROW_H, LW_THIN)
rule(TOP - HDR_H - 6 * ROW_H, LW_THIN)
rule(0, LW_THICK)                   # bottom rule

# ---- Title --------------------------------------------------------------------
title = ("Negative steering swaps the concept out fluently; a matched random push "
         "does not (f5568 is causally inert)")
fig.suptitle(title, y=0.99, fontsize=FS_TITLE, family=FAMILY,
             fontweight="bold")

OUT = "results_v2/table1_qualitative.png"
fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
print("saved", OUT, "family=", FAMILY, "fig", FIG_W, FIG_H)
