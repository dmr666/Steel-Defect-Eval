# NOTE: input paths were machine-specific; the prefix has been replaced with '.'.
# Edit the path constants near the top if your layout differs from the repo layout.
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_figs_paper.py — paper3 论文用图（英文标签，与 make_figs_gc10.py 同规范）

输出到 _paper/figs_paper/（png + svg + pdf，300dpi，Times New Roman，刻度朝内，SVG 真文本）
    fig1_framework.*      协议与配置矩阵示意（无数据）
    fig2_capacity.*       容量饱和：参数量 vs 两项指标（2 数据集 × 2 尺度 × 5 配置）
    fig3_transfer.*       跨尺度迁移性：nano−s 的差值与合并标准差（2 数据集）
    fig4_imbalance.*      不平衡五臂：总体 mAP50 与稀有类 AP50（2 数据集）
    fig5_pareto.*         代价-精度 Pareto：ONNX-CPU 延迟 vs val mAP50-95（统一批次）

数据源（只读）：
    _results/gc10_20260929/tables/table_main.tsv
    _results/neu_20260929/tables/table_main.tsv
    _results/gc10_nano_20260929/tables_nano/table_main.tsv
    _results/neu_nano_20260930/small/tables_nano/table_main.tsv
    _results/{gc10_20260929,neu_20260929}/tables/table_imbalance.tsv
    _results/deploy_20260930/deploy_20260930_{gc10,neu}_{s,nano}/*.json   （统一批次）
"""
import json
import math
import os
import statistics as st
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import NullFormatter

rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.linewidth": 0.8,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "figure.dpi": 150,
})

ROOT = "."
R = os.path.join(ROOT, "_results")
OUT = os.environ.get("FIGOUT") or os.path.join(ROOT, "_paper", "figs_paper")

CELLS5 = ["v_base", "v_p2", "v_dcsa", "v_wiou", "v_full"]
NICE = {"v_base": "baseline", "v_p2": "+P2", "v_dcsa": "+DCSA", "v_wiou": "+WIoU v3", "v_full": "+stack+$w_c$"}
SRC_MAIN = {
    ("GC10", "s"): f"{R}/gc10_20260929/tables/table_main.tsv",
    ("GC10", "n"): f"{R}/gc10_nano_20260929/tables_nano/table_main.tsv",
    ("NEU", "s"): f"{R}/neu_20260929/tables/table_main.tsv",
    ("NEU", "n"): f"{R}/neu_nano_20260930/small/tables_nano/table_main.tsv",
}
DEP = {
    ("GC10", "s"): f"{R}/deploy_20260930/deploy_20260930_gc10_s",
    ("GC10", "n"): f"{R}/deploy_20260930/deploy_20260930_gc10_nano",
    ("NEU", "s"): f"{R}/deploy_20260930/deploy_20260930_neu_s",
    ("NEU", "n"): f"{R}/deploy_20260930/deploy_20260930_neu_nano",
}
PARAMS = {  # best.pt 实测参数量（M）
    ("GC10", "s"): {"v_base": 11.139, "v_wiou": 11.139, "v_dcsa": 11.154, "v_p2": 10.640, "v_full": 10.654},
    ("NEU", "s"): {"v_base": 11.138, "v_wiou": 11.138, "v_dcsa": 11.153, "v_p2": 10.639, "v_full": 10.653},
    ("GC10", "n"): {"v_base": 3.013, "v_wiou": 3.013, "v_dcsa": 3.018, "v_p2": 2.928, "v_full": 2.933},
    ("NEU", "n"): {"v_base": 3.012, "v_wiou": 3.012, "v_dcsa": 3.018, "v_p2": 2.927, "v_full": 2.933},
}
C_GC, C_NEU = "#4C72B0", "#C44E52"


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(os.path.join(OUT, "%s.%s" % (name, ext)))
    plt.close(fig)
    print("  写出", name + ".{png,svg,pdf}")


def num(s):
    s = (s or "").strip()
    if not s or s == "—":
        return (None, 0.0)
    if "±" in s:
        a, b = s.split("±")
        return (float(a), float(b))
    return (float(s), 0.0)


def read_main(path):
    out, hdr = {}, None
    for line in open(path, encoding="utf-8"):
        f = line.rstrip("\n").rstrip("\r").split("\t")
        if not line.strip():
            continue
        if hdr is None:
            hdr = f
            continue
        if len(f) < 9:
            continue
        d = dict(zip(hdr, f))
        if int(d["n"]) == 0:
            continue
        out[d["cell"]] = {"val50": num(d["val mAP50"]), "val95": num(d["val mAP50-95"]),
                          "test95": num(d.get("test mAP50-95", "")), "n": int(d["n"])}
    return out


def read_deploy(d):
    per = defaultdict(lambda: defaultdict(list))
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".json"):
            continue
        j = json.load(open(os.path.join(d, fn), encoding="utf-8"))
        key = (j.get("cell") or fn[:-5]).rsplit("_seed", 1)[0]
        per[key]["cpu"].append(j["onnx_cpu"]["p50"])
        per[key]["gf"].append(j["gflops"])
    return {k: {m: (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0) for m, v in dd.items()} for k, dd in per.items()}


def read_imbal(path):
    out, hdr = {}, None
    for line in open(path, encoding="utf-8"):
        f = line.rstrip("\n").rstrip("\r").split("\t")
        if not line.strip():
            continue
        if hdr is None:
            hdr = f
            continue
        if len(f) < 7:
            continue
        out[f[0].strip()] = {"val50": num(f[4]), "val95": num(f[3]), "rare": num(f[5])[0]}
    return out


M = {k: read_main(v) for k, v in SRC_MAIN.items()}
# NEU ext（v_full@320 / @960，2026-09-30 第二批训练）合入 NEU s 表，供 fig5(c) 用
EXT_TABLES = f"{R}/neu_ext_20260930/small/tables/table_main_ext.tsv"
EXT_DEPDIR = f"{R}/neu_ext_20260930/ext_deploy/runs_deploy_ext"
if os.path.isfile(EXT_TABLES):
    M[("NEU", "s")].update(read_main(EXT_TABLES))
D = {k: read_deploy(v) for k, v in DEP.items()}
DEXT = read_deploy(EXT_DEPDIR) if os.path.isdir(EXT_DEPDIR) else {}


def gf_of(cell):
    """GFLOPs（确定性量）：ext cell 取 ext 部署 json，其余取统一批"""
    a = DEXT.get(cell) or D[("NEU", "s")].get(cell)
    return a["gf"][0] if a else None


IM = {"GC10": read_imbal(f"{R}/gc10_20260929/tables/table_imbalance.tsv"),
      "NEU": read_imbal(f"{R}/neu_20260929/tables/table_imbalance.tsv")}
print("载入:", {k: len(v) for k, v in M.items()}, {k: len(v) for k, v in D.items()})

# ============ fig1 framework ============
fig, ax = plt.subplots(figsize=(7.1, 2.9))
ax.set_xlim(0, 100)
ax.set_ylim(0, 46)
ax.axis("off")


def box(x, y, w, h, text, fc="#eef2f7", ec="#4C72B0", fs=7.9, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.2",
                                fc=fc, ec=ec, lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight=weight)


def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8,
                                 lw=0.8, color="#555555", shrinkA=1, shrinkB=1))


box(1, 30, 21, 12, "GC10-DET\n10 classes $\\cdot$ 11.4:1\n2,292 images", fc="#f2f7f2")
box(1, 11, 21, 12, "NEU-DET\n6 classes $\\cdot$ 2.4:1\n1,800 images", fc="#fdf3f3")
box(27, 28, 24, 14, "backbone scale\nYOLOv8s 11.1 M\nYOLOv8n 3.0 M", fc="#f7f7f7")
box(27, 9, 24, 15, "treatment arms\n+shallow head (P1/P2)\n+attention (DCSA)\n+WIoU v3 $\\cdot$ $w_c$ $\\cdot$ Focal")
box(56, 28, 20, 14, "fixed protocol\nsplit seed 0 $\\cdot$ 640 px\n200 ep $\\cdot$ SGD $\\cdot$ 3 seeds")
box(56, 9, 20, 15, "evaluation\nval + held-out test\nper-class AP $\\pm$ SD")
box(80, 28, 19, 14, "deployment battery\nparams $\\cdot$ GFLOPs\nFP32/FP16 $\\cdot$ ONNX-CPU\nVRAM $\\cdot$ size")
box(80, 9, 19, 15, "cross-dataset\n$\\times$ cross-scale\nreproducibility\nverdict")
arrow(22.5, 36, 26.5, 35)
arrow(22.5, 17, 26.5, 16)
arrow(51.5, 35, 55.5, 35)
arrow(51.5, 16.5, 55.5, 16.5)
arrow(76.5, 35, 79.5, 35)
arrow(76.5, 16.5, 79.5, 16.5)
arrow(89.5, 27.5, 89.5, 24.5)
ax.text(50, 44, "120 training runs = 114 main matrix + 6 resolution runs",
        ha="center", fontsize=8.4, style="italic")
save(fig, "fig1_framework")

# ============ fig2 capacity ============
fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0))
for ax, key, title in zip(axes, ("val50", "val95"), ("val mAP50", "val mAP50-95")):
    for ds, marker, color in (("GC10", "o", C_GC), ("NEU", "s", C_NEU)):
        for sc, filled in (("s", True), ("n", False)):
            xs, ys, es = [], [], []
            for c in CELLS5:
                r = M[(ds, sc)].get(c, {}).get(key)
                if not r or r[0] is None:
                    continue
                xs.append(PARAMS[(ds, sc)][c])
                ys.append(r[0])
                es.append(r[1] or 0.0)
            ax.errorbar(xs, ys, yerr=es, marker=marker, ls="none", ms=4.6, capsize=1.6,
                        mfc=(color if filled else "white"), mec=color, ecolor=color, mew=0.9,
                        label="%s %s" % (ds, "s (11.1 M)" if sc == "s" else "n (3.0 M)"))
    ax.set_xscale("log")
    ax.set_xticks([3.0, 11.0])
    ax.set_xticklabels(["3.0", "11.1"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("parameters (M)")
    ax.set_ylabel(title)
    ax.set_xlim(2.6, 13.5)
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0 - 0.08 * (y1 - y0), y1 + 0.04 * (y1 - y0))
axes[0].set_title("(a)", fontsize=9, loc="left")
axes[1].set_title("(b)", fontsize=9, loc="left")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=7.8)
save(fig, "fig2_capacity")

# ============ fig3 transferability ============
fig, ax = plt.subplots(figsize=(7.1, 3.1))
w = 0.36
x = np.arange(len(CELLS5))
for i, (ds, color) in enumerate((("GC10", C_GC), ("NEU", C_NEU))):
    dels, errs, sigs = [], [], []
    for c in CELLS5:
        a = M[(ds, "s")].get(c, {}).get("val95")
        b = M[(ds, "n")].get(c, {}).get("val95")
        d = b[0] - a[0]
        pooled = math.sqrt(((a[1] or 0) ** 2 + (b[1] or 0) ** 2) / 2)
        dels.append(d)
        errs.append(pooled)
        sigs.append(abs(d) > 2 * pooled)
    pos = x + (i - 0.5) * w
    bars = ax.bar(pos, dels, w, color=color, alpha=0.85, label=ds, edgecolor="black", lw=0.4)
    for b, s in zip(bars, sigs):
        if s:
            b.set_hatch("///")
    ax.errorbar(pos, dels, yerr=errs, fmt="none", ecolor="black", elinewidth=0.7, capsize=1.8)
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels([NICE[c] for c in CELLS5])
ax.set_ylabel("$\\Delta$ val mAP50-95  (nano $-$ s)")
ax.margins(y=0.16)
ax.legend(loc="upper left")
ax.text(0.99, 0.97, "hatched = $|\\Delta|>2\\,\\sigma_{pooled}$", ha="right", va="top",
        transform=ax.transAxes, fontsize=7.4, style="italic")
save(fig, "fig3_transfer")

# ============ fig4 imbalance ============
arms = list(IM["GC10"].keys())
# 稀有类 AP50 的逐种子离散度（code/review_stats.py 由 perclass_ap.json 计算）
_RARE_JSON = f"{R}/review_stats/rare_ap_perseed.json"
RARE_PD = json.load(open(_RARE_JSON, encoding="utf-8")) if os.path.isfile(_RARE_JSON) else {}
RARE_KEY = {ds: dict(zip(["无处理", "指数加权 0.5", "指数加权 1.0", "重采样", "Focal"],
                         ["%s|%s" % (ds, c) for c in ("v_p2dcsawiou", "v_full_cpw0p5", "v_full",
                                                      "v_resample", "v_focal")]))
            for ds in ("GC10", "NEU")}
ARM_EN = {"无处理": "none", "指数加权 0.5": "$w_c$=0.5", "指数加权 1.0": "$w_c$=1.0",
          "重采样": "resample", "Focal": "Focal"}
fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9), sharex=True)
x = np.arange(len(arms))
for ax, field, title in zip(axes, ("val50", "rare"), ("aggregate val mAP50", "rare-class AP50")):
    w = 0.36
    for i, (ds, color) in enumerate((("GC10", C_GC), ("NEU", C_NEU))):
        vals, errs = [], []
        for a in arms:
            r = IM[ds].get(a)
            raw = r.get(field) if r else None
            if raw is None:
                vals.append(0.0)
                errs.append(0.0)
            elif isinstance(raw, tuple):
                vals.append(raw[0])
                errs.append(raw[1] or 0.0)
            else:
                vals.append(raw)
                errs.append(0.0)
        if field == "rare":
            for j, a in enumerate(arms):
                k = RARE_KEY.get(ds, {}).get(a)
                if k and k in RARE_PD:
                    vals[j] = RARE_PD[k]["mean"]
                    errs[j] = RARE_PD[k]["sd"]
        ax.bar(x + (i - 0.5) * w, vals, w, color=color, alpha=0.85, edgecolor="black", lw=0.4, label=ds)
        ax.errorbar(x + (i - 0.5) * w, vals, yerr=errs, fmt="none", ecolor="black", elinewidth=0.7, capsize=1.8)
    ax.set_xticks(x)
    ax.set_xticklabels([ARM_EN.get(a, a) for a in arms], fontsize=7.8)
    ax.set_ylabel(title)
    y0, y1 = ax.get_ylim()
    ax.set_ylim(0, y1 * 1.18)
    if field == "rare":
        ax.text(0.5, 0.985, "rare-class AP50, mean $\\pm$ SD over three seeds (both datasets)",
                transform=ax.transAxes, fontsize=6.6, va="top", ha="center", style="italic")
axes[0].set_title("(a)", fontsize=9, loc="left")
axes[1].set_title("(b)", fontsize=9, loc="left")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=7.8)
save(fig, "fig4_imbalance")

# ============ fig5 pareto ============
CFG5 = [("v_base", "o", "#4C72B0"), ("v_p2", "s", "#DD8452"), ("v_dcsa", "^", "#55A868"),
        ("v_wiou", "D", "#8172B3"), ("v_full", "v", "#C44E52")]
fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.2))
for ax, ds in zip(axes[:2], ("GC10", "NEU")):
    for c, marker, color in CFG5:
        xs, ys, xe = [], [], []
        for sc in ("s", "n"):
            d = D[(ds, sc)].get(c)
            m = M[(ds, sc)].get(c, {}).get("val95")
            if not d or not m or m[0] is None:
                continue
            xs.append(d["cpu"][0])
            xe.append(d["cpu"][1] or 0.0)
            ys.append(m[0])
        if len(xs) < 2:
            continue
        ax.plot(xs, ys, marker=marker, ls="-", lw=0.9, ms=4.6, color=color, mfc="white", mew=0.9,
                clip_on=False, label=NICE[c])
        ax.plot(xs[1], ys[1], marker=marker, ls="none", ms=4.6, color=color, clip_on=False)
        ax.errorbar(xs, ys, xerr=xe, fmt="none", ecolor=color, elinewidth=0.6, capsize=1.3, alpha=0.85)
    ax.set_xscale("log")
    ax.set_xlabel("ONNX Runtime CPU latency, batch 1 (ms, log)")
    ax.set_ylabel("val mAP50-95")
    ax.set_title("(%s) %s-DET" % ("a" if ds == "GC10" else "b", ds), fontsize=9, loc="left")
    ax.margins(x=0.30, y=0.22)
axes[0].text(0.03, 0.05, "open marker: s scale\nfilled marker: nano scale", transform=axes[0].transAxes,
             fontsize=7.2, style="italic")
axes[1].text(0.03, 0.05, "open marker: s scale\nfilled marker: nano scale", transform=axes[1].transAxes,
             fontsize=7.2, style="italic")

# ---- panel (c): NEU-DET 分辨率扫描（v_base / v_full @320/640/960）----
ax = axes[2]
RES5 = [("baseline", "o", "#4C72B0", [("v_base_s320_b8", 320), ("v_base_b8", 640), ("v_base_s960_b8", 960)]),
        ("full stack", "v", "#C44E52", [("v_full_s320_b8", 320), ("v_full", 640), ("v_full_s960_b8", 960)])]
tNEU = M[("NEU", "s")]
for nm, mk, col, items in RES5:
    off_y = 6 if nm == "baseline" else -14          # 两线在 640 处很贴近，标签错开上下
    ax_off = (0, off_y)
    xs, ys, es = [], [], []
    for cell, px in items:
        r = tNEU.get(cell, {}).get("val95")
        if not r or r[0] is None:
            continue
        xs.append(px)
        ys.append(r[0])
        es.append(r[1] or 0.0)
    if not xs:
        continue
    ax.errorbar(xs, ys, yerr=es, marker=mk, ms=4.6, lw=0.9, color=col, mfc="white", mew=0.9,
                capsize=1.3, elinewidth=0.6, clip_on=False, label=nm)
    for (cell, x), y in zip([(c, p) for c, p in items if c in tNEU], ys):
        g = gf_of(cell)
        if g:
            ax.annotate("%.1f" % g, (x, y), textcoords="offset points", xytext=ax_off,
                        ha="center", fontsize=6.4, color=col)
ax.set_xticks([320, 640, 960])
ax.set_xticklabels(["320", "640", "960"])
ax.set_xlabel("Input resolution (px)")
ax.set_ylabel("val mAP50-95")
ax.set_title("(c) NEU-DET resolution sweep", fontsize=9, loc="left")
ax.margins(x=0.20, y=0.24)
ax.legend(loc="upper right", fontsize=7.4)
ax.text(0.03, 0.04, "numbers below markers: GFLOPs\n320/960 full stack: second training batch",
        transform=ax.transAxes, fontsize=6.5, style="italic")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=5, fontsize=7.8)
save(fig, "fig5_pareto")

# ============ fig6 qualitative defect crops ============
import glob
from PIL import Image
from matplotlib.patches import Rectangle

DATASETS6 = [
    ("GC10-DET (cold-rolled)", f"{ROOT}/data/GC10-DET-yolo/test",
     ["chongkong", "hanfeng", "yueyawan", "shuiban", "youban", "siban", "yiwu", "yahen", "zhehen", "yaozhe"]),
    ("NEU-DET (hot-rolled)", f"{ROOT}/data/NEU-DET-split/test",
     ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]),
]


def small_examples(split_dir, k=3, max_boxes=5):
    """挑"框最小"的样本：体现小目标难点；返回 [(短边px, 图路径, 框, 宽, 高)]"""
    cands = []
    for lp in glob.glob(os.path.join(split_dir, "labels", "*.txt")):
        ip = os.path.join(split_dir, "images", os.path.basename(lp)[:-4] + ".jpg")
        if not os.path.exists(ip):
            continue
        boxes = []
        for line in open(lp):
            p = line.split()
            if len(p) < 5:
                continue
            boxes.append((int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])))
        if not boxes or len(boxes) > max_boxes:
            continue
        try:
            w, h = Image.open(ip).size
        except Exception:
            continue
        mn = min(min(bw * w, bh * h) for _, _, _, bw, bh in boxes)
        if mn < 4:
            continue
        cands.append((mn, ip, boxes, w, h))
    cands.sort(key=lambda t: t[0])
    return cands[:k]


fig, axes = plt.subplots(2, 3, figsize=(7.1, 3.4))
for r, (name, split, names) in enumerate(DATASETS6):
    ex = small_examples(split)
    for c in range(3):
        ax = axes[r][c]
        ax.axis("off")
        if c >= len(ex):
            continue
        mn, ip, boxes, w, h = ex[c]
        # 以"最小框"为中心（要展示的正是小目标难点），而不是文件里的第一个框
        b_min = min(boxes, key=lambda b: b[3] * w * b[4] * h)
        print("  %s: %s  最小框 %s %.0fpx（共 %d 个框）" % (
            name, os.path.basename(ip), names[b_min[0]] if b_min[0] < len(names) else b_min[0],
            min(b_min[3] * w, b_min[4] * h), len(boxes)))
        cx = min(max(int(b_min[1] * w), 0), w - 1)
        cy = min(max(int(b_min[2] * h), 0), h - 1)
        cw = ch = 96
        x0 = min(max(cx - cw // 2, 0), max(w - cw, 0))
        y0 = min(max(cy - ch // 2, 0), max(h - ch, 0))
        im = Image.open(ip).convert("L").crop((x0, y0, x0 + cw, y0 + ch))
        ax.imshow(im, cmap="gray")
        for cls, bx, by, bw, bh in boxes:
            px, py = bx * w - x0, by * h - y0
            pw, ph = bw * w, bh * h
            if px + pw / 2 < 0 or py + ph / 2 < 0 or px - pw / 2 > cw or py - ph / 2 > ch:
                continue
            ax.add_patch(Rectangle((px - pw / 2, py - ph / 2), pw, ph, fill=False, ec="#C44E52",
                                   lw=0.9, clip_on=True))
            tx = min(max(px - pw / 2, 1.0), cw - 34.0)      # 夹到裁剪范围内，避免撑爆 tight bbox
            ty = min(max(py - ph / 2 - 2, 5.0), ch - 2.0)
            ax.text(tx, ty, "%s %.0fpx" % (names[cls] if cls < len(names) else cls, min(pw, ph)),
                    color="#C44E52", fontsize=5.6, va="bottom", clip_on=True)
        ax.plot([6, 38], [ch - 7, ch - 7], color="black", lw=1.6, solid_capstyle="butt")
        ax.text(6, ch - 10, "32 px", fontsize=5.8, va="bottom")
        ax.set_xlim(0, cw)
        ax.set_ylim(ch, 0)
    axes[r][0].text(-0.10, 0.5, name, transform=axes[r][0].transAxes, rotation=90,
                    ha="center", va="center", fontsize=7.6)
fig.text(0.5, 0.005, "crops at native pixel scale (96$\\times$96); red = ground-truth boxes; scale bar = 32 px",
         ha="center", fontsize=7.0, style="italic")
save(fig, "fig6_defects")

print("\n全部图已写入:", OUT)
