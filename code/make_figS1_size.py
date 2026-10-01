# NOTE: input paths were machine-specific; the prefix has been replaced with '.'.
# Edit the path constants near the top if your layout differs from the repo layout.
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_figS1_size.py — review Minor 11：两数据集缺陷尺度分布直方图（SI Fig. S1）

折算口径与 code/scale_profile.py 完全一致：letterbox 到 640 输入后的等效目标边长
= sqrt((w_px*s)*(h_px*s))，s = 640/max(W,H)。输出到投稿包 figures/figS1_size_distribution.{png,pdf}
"""
import collections
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from matplotlib import rcParams

rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 9,
                 "axes.linewidth": 0.8, "axes.labelsize": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
                 "xtick.direction": "in", "ytick.direction": "in", "legend.fontsize": 8,
                 "legend.frameon": False, "svg.fonttype": "none", "pdf.fonttype": 42,
                 "savefig.dpi": 300, "savefig.bbox": "tight", "figure.dpi": 150})

DATA = "./data"
OUT = "./投稿材料_SIVP_20261001/01_稿件_LaTeX/figures"
REF = "./_verify/scale_profile.json"
ref = json.load(open(REF, encoding="utf-8"))

SETS = [
    ("GC10-DET (cold-rolled)", os.path.join(DATA, "GC10-DET-yolo"), (2048, 1000),
     json.load(open(os.path.join(DATA, "GC10-DET-yolo", "split_stats.json"), encoding="utf-8"))["names"]),
    ("NEU-DET (hot-rolled)", os.path.join(DATA, "NEU-DET-split"), (200, 200),
     ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]),
]

fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9))
for ax, (title, root, (W, H), names) in zip(axes, SETS):
    sizes = []
    for split in ("train", "test"):
        ldir = os.path.join(root, split, "labels")
        if not os.path.isdir(ldir):
            continue
        for fn in os.listdir(ldir):
            if not fn.endswith(".txt"):
                continue
            for line in open(os.path.join(ldir, fn), encoding="utf-8"):
                p = line.split()
                if len(p) != 5:
                    continue
                w_px, h_px = float(p[3]) * W, float(p[4]) * H
                s = 640.0 / max(W, H)
                sizes.append(math.sqrt((w_px * s) * (h_px * s)))
    sizes = np.array(sizes)
    bins = np.logspace(math.log10(max(3, sizes.min())), math.log10(sizes.max()), 45)
    ax.hist(sizes, bins=bins, color="#4C72B0" if "GC10" in title else "#C44E52", alpha=0.85, edgecolor="none")
    med = float(np.median(sizes))
    ax.axvline(med, color="k", lw=0.9, ls="-")
    ax.axvline(32, color="#777777", lw=0.9, ls="--")
    ax.set_xscale("log")
    ax.set_xticks([10, 32, 100, 300])
    ax.set_xticklabels(["10", "32", "100", "300"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(6, 520)
    ax.set_xlabel("object side at 640 px input (px, log)")
    ax.set_ylabel("instances")
    frac32 = 100.0 * float((sizes < 32).mean())
    ax.set_title("(%s) %s" % ("a" if "GC10" in title else "b", title), fontsize=9, loc="left")
    ax.text(0.98, 0.94, "n = %d instances\nmedian %.0f px\n%.0f%% below 32 px" % (len(sizes), med, frac32),
            transform=ax.transAxes, ha="right", va="top", fontsize=7.4)
    ax.margins(x=0.05, y=0.14)
fig.legend([plt.Line2D([], [], color="k", lw=0.9), plt.Line2D([], [], color="#777777", lw=0.9, ls="--")],
           ["median", "32 px"], loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=8)
os.makedirs(OUT, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, "figS1_size_distribution.%s" % ext))
print("写出:", os.path.join(OUT, "figS1_size_distribution.png"))

# 与既有 scale_profile.json 交叉核对（train+test 合并口径需自己算，这里只核对 test 单独口径）
for (tag, root, (W, H), names) in SETS:
    sizes = []
    ldir = os.path.join(root, "test", "labels")
    for fn in os.listdir(ldir):
        if not fn.endswith(".txt"):
            continue
        for line in open(os.path.join(ldir, fn), encoding="utf-8"):
            p = line.split()
            if len(p) != 5:
                continue
            w_px, h_px = float(p[3]) * W, float(p[4]) * H
            s = 640.0 / max(W, H)
            sizes.append(math.sqrt((w_px * s) * (h_px * s)))
    sizes.sort()
    key = [k for k in ref if k.startswith("NEU-DET" if "NEU" in tag else "GC10-DET") and k.endswith("test")][0]
    r = ref[key]
    print("%-24s 本次 test 中位 %.1f / json %.1f | <32px %.3f / json %.3f"
          % (tag, sizes[len(sizes) // 2], r["eq640_median"], sum(1 for v in sizes if v < 32) / len(sizes), r["frac_below_32px"]))
