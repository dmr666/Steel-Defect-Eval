# NOTE: machine-specific paths were replaced with '.'/placeholders.
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""reviewB_analyze.py —— B 档结果汇总（from-scratch 对照 / 加种子效应量 / e300 / IoU 曲线）

输入：本地解包后的 reviewB 归档目录（默认 _results/reviewB_20261001/extract）
输出：_results/reviewB_20261001/tables/*.md + 控制台摘要
口径：mAP50-95 为主指标；显著性沿用论文的 pooled-2σ 规则，另附 Welch + Holm（同一家族内校正）。
      不手改任何数字：全部由 perclass_ap.json 现算。
"""
import glob
import json
import os
import re
import statistics as st
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if "__file__" in dir() else "."
EXTRACT = os.environ.get("REVIEWB_DIR", "./_results/reviewB_20261001/extract")
OUT = "./_results/reviewB_20261001/tables"
SEEDS_MAIN = ["42", "7", "2024"]
SEEDS_EXTRA = ["13", "99"]
M = {}


# ---------- 统计工具（纯 Python，Welch + Holm） ----------
def betacf(a, b, x):
    MAXIT, EPS, FPMIN = 200, 3e-16, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < EPS:
            break
    return h


def betai(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = _lgamma(a + b) - _lgamma(a) - _lgamma(b) + a * __import__("math").log(x) + b * __import__("math").log(1 - x)
    import math
    if x < (a + 1) / (a + b + 2):
        return math.exp(lbeta) * betacf(a, b, x) / a
    return 1.0 - math.exp(lbeta) * betacf(b, a, 1 - x) / b


def _lgamma(z):
    import math
    return math.lgamma(z)


def welch(x, y):
    """返回 (t, p, df)"""
    import math
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float("nan"), float("nan"), float("nan")
    mx, my = st.mean(x), st.mean(y)
    vx, vy = st.variance(x), st.variance(y)
    se2 = vx / nx + vy / ny
    if se2 <= 0:
        return 0.0, 1.0, float("nan")
    t = (mx - my) / math.sqrt(se2)
    df = se2 ** 2 / ((vx / nx) ** 2 / (nx - 1) + (vy / ny) ** 2 / (ny - 1))
    p = betai(df / 2.0, 0.5, df / (df + t * t))
    return t, p, df


def holm(pvals):
    """返回与输入同序的 Holm 校正 p 值"""
    idx = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals)
    adj = [0.0] * m
    run = 0.0
    for rank, i in enumerate(idx):
        v = min(1.0, (m - rank) * pvals[i])
        run = max(run, v)
        adj[i] = run
    return adj


# ---------- 数据装载 ----------
def key_of(path):
    """从路径里解析出 (组, run 名)。组 = 归档顶层目录名"""
    parts = path.replace("\\", "/").split("/")
    run = parts[-2]
    # 组：找 runs_scratch_s / runs_scratch_n / runs_e300n / runs / runs_nano + 上级工作区
    for i, p in enumerate(parts):
        if p.startswith("runs"):
            ws = parts[i - 1] if i else ""
            ws = "gc10" if "gc10" in ws or "GC10" in ws else "neu"
            return ws, p, run
    return "?", "?", run


def load(dirpath):
    data = {}
    for f in glob.glob(os.path.join(dirpath, "**", "perclass_ap.json"), recursive=True):
        try:
            j = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        ws, proj, run = key_of(f)
        cell = re.sub(r"_seed\d+$", "", run)
        seed = (re.search(r"_seed(\d+)$", run) or [None, "?"])[1]
        t = j.get("test") or {}
        data.setdefault((ws, proj), {}).setdefault(cell, {})[seed] = {
            "val50": j.get("mAP50"), "val5095": j.get("mAP5095"),
            "test50": t.get("mAP50"), "test5095": t.get("mAP5095"),
            "params": j.get("params"), "epochs": (j.get("epochs") or 200),
        }
    return data


def ms(vals):
    vals = [v for v in vals if isinstance(v, (int, float))]
    if not vals:
        return float("nan"), float("nan")
    return st.mean(vals), (st.stdev(vals) if len(vals) > 1 else 0.0)


def line(name, cellmap, seeds):
    v50 = [cellmap[s]["val50"] for s in seeds if s in cellmap]
    v95 = [cellmap[s]["val5095"] for s in seeds if s in cellmap]
    t50 = [cellmap[s]["test50"] for s in seeds if s in cellmap]
    t95 = [cellmap[s]["test5095"] for s in seeds if s in cellmap]
    n = len(v95)
    out = {"n": n}
    for k, arr in (("v50", v50), ("v95", v95), ("t50", t50), ("t95", t95)):
        m, s = ms(arr)
        out[k], out[k + "_sd"] = m, s
    return out


def main():
    if not os.path.isdir(EXTRACT):
        print("!! 找不到解包目录:", EXTRACT)
        return 1
    data = load(EXTRACT)
    os.makedirs(OUT, exist_ok=True)
    print("归档里的组:")
    for k, cells in sorted(data.items()):
        print("  %-22s cells=%d" % (str(k), len(cells)))

    D = {}
    for (ws, proj), cells in data.items():
        D[(ws, proj)] = cells

    def g(ws, proj):
        return D.get((ws, proj), {})

    # ---------- 1) from-scratch 对照（M6） ----------
    lines = ["# Table B1. From-scratch vs COCO-pretrained baselines (test mAP50-95)", "",
             "| dataset | scale | pretrained (mean±SD, n) | from-scratch (mean±SD, n) | Δ(scratch−pretrained) | Δ(s−n) pretrained | Δ(s−n) scratch |",
             "|---|---|---|---|---|---|---|"]
    for ws, dn in (("gc10", "GC10-DET"), ("neu", "NEU-DET")):
        pre = g(ws, "runs")
        scr = g(ws, "runs_scratch_s")
        pren = g(ws, "runs_nano")
        scrn = g(ws, "runs_scratch_n")
        b = line("base", pre.get("v_base", {}), SEEDS_MAIN + SEEDS_EXTRA)
        bn = line("base", pren.get("v_base", {}), SEEDS_MAIN)
        sb = line("scratch-s", scr.get("v_base", {}), SEEDS_MAIN)
        sn = line("scratch-n", scrn.get("v_base", {}), SEEDS_MAIN)
        lines.append("| %s | s | %.4f±%.4f (%d) | %.4f±%.4f (%d) | %+.4f | %+.4f | %+.4f |" % (
            dn, b["t95"], b["t95_sd"], b["n"], sb["t95"], sb["t95_sd"], sb["n"],
            sb["t95"] - b["t95"], b["t95"] - bn["t95"], sb["t95"] - sn["t95"]))
        lines.append("| %s | nano | %.4f±%.4f (%d) | %.4f±%.4f (%d) | %+.4f | — | — |" % (
            dn, bn["t95"], bn["t95_sd"], bn["n"], sn["t95"], sn["t95_sd"], sn["n"],
            sn["t95"] - bn["t95"]))
        print("\n[from-scratch] %s: pretrained s=%.4f n=%.4f | scratch s=%.4f n=%.4f" % (
            dn, b["t95"], bn["t95"], sb["t95"], sn["t95"]))
    open(os.path.join(OUT, "B1_scratch_vs_pretrained.md"), "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")

    # ---------- 2) 加种子：关键格子的 5 种子效应量（M1） ----------
    rows, ps = [], []
    for ws, dn in (("gc10", "GC10-DET"), ("neu", "NEU-DET")):
        cells = g(ws, "runs")
        base = cells.get("v_base", {})
        for cell in ("v_full", "v_focal", "v_wiou", "v_p2", "v_dcsa"):
            if cell not in cells:
                continue
            a3 = [base[s]["test5095"] for s in SEEDS_MAIN if s in base]
            b3 = [cells[cell][s]["test5095"] for s in SEEDS_MAIN if s in cells[cell]]
            a5 = [base[s]["test5095"] for s in SEEDS_MAIN + SEEDS_EXTRA if s in base]
            b5 = [cells[cell][s]["test5095"] for s in SEEDS_MAIN + SEEDS_EXTRA if s in cells[cell]]
            d3 = st.mean(b3) - st.mean(a3) if len(a3) > 1 and len(b3) > 1 else float("nan")
            d5 = st.mean(b5) - st.mean(a5) if len(a5) > 1 and len(b5) > 1 else float("nan")
            pooled3 = ((st.stdev(a3) ** 2 + st.stdev(b3) ** 2) / 2) ** 0.5 if len(a3) > 1 and len(b3) > 1 else float("nan")
            pooled5 = ((st.stdev(a5) ** 2 + st.stdev(b5) ** 2) / 2) ** 0.5 if len(a5) > 1 and len(b5) > 1 else float("nan")
            t, p, df = welch(b5, a5)
            ps.append(p)
            rows.append((dn, cell, d3, pooled3, d5, pooled5, p, len(a5), len(b5)))
    if rows:
        adj = holm(ps)
        lines = ["# Table B2. Key cells with five seeds (test mAP50-95)", "",
                 "| dataset | cell | Δ (n=3) | 2σ_pooled (n=3) | Δ (n=5) | 2σ_pooled (n=5) | Welch p | Holm p | verdict (pooled / Holm) |",
                 "|---|---|---|---|---|---|---|---|---|"]
        for (dn, cell, d3, p3, d5, p5, p, na, nb), pa in zip(rows, adj):
            v1 = "sig." if abs(d5) > 2 * p5 else "n.s."
            v2 = "sig." if pa < 0.05 else "n.s."
            lines.append("| %s | %s | %+.4f | %.4f | %+.4f | %.4f | %.4f | %.4f | %s / %s |" % (
                dn, cell, d3, p3, d5, p5, p, pa, v1, v2))
        open(os.path.join(OUT, "B2_five_seeds.md"), "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
        print("\n[五种子] 写出 B2，共 %d 行" % len(rows))

    # ---------- 3) NEU e300（Minor 9） ----------
    e300 = D.get(("neu", "runs_e300n"), {})
    if e300:
        lines = ["# Table B3. NEU-DET at 300 epochs vs 200 epochs", "",
                 "| cell | 200 ep (mean±SD, n) | 300 ep (mean±SD, n) | Δ(300−200) | 2σ_pooled |",
                 "|---|---|---|---|---|"]
        for cell in sorted(e300):
            a = line(cell, g("neu", "runs").get(cell, {}), SEEDS_MAIN)
            b = line(cell, e300[cell], SEEDS_MAIN)
            pooled = ((a["t95_sd"] ** 2 + b["t95_sd"] ** 2) / 2) ** 0.5
            lines.append("| %s | %.4f±%.4f (%d) | %.4f±%.4f (%d) | %+.4f | %.4f |" % (
                cell, a["t95"], a["t95_sd"], a["n"], b["t95"], b["t95_sd"], b["n"], b["t95"] - a["t95"], 2 * pooled))
            print("[e300 NEU] %s: 200ep=%.4f 300ep=%.4f Δ=%+.4f" % (cell, a["t95"], b["t95"], b["t95"] - a["t95"]))
        open(os.path.join(OUT, "B3_neu_e300.md"), "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")

    # ---------- 4) IoU 曲线 ----------
    for ws in ("neu", "gc10"):
        for f in glob.glob(os.path.join(EXTRACT, "**", "iou_sweep.json"), recursive=True):
            low = f.replace("\\", "/").lower()
            # NEU 工作区目录名是 defect（不含 "neu" 字样），按 gc10 是否为真值来配对
            if (ws == "gc10") != ("gc10" in low):
                continue
            j = json.load(open(f, encoding="utf-8"))
            lines = ["# Table B4. AP vs IoU threshold (%s)" % j.get("split", "test"), "",
                     "| cell | " + " | ".join("%.2f" % t for t in j["thresholds"]) + " | mAP50-95 |",
                     "|---" * (len(j["thresholds"]) + 2) + "|"]
            for cell, d2 in j.get("agg", {}).items():
                vals = [(d2["iou%.2f" % t][0] if isinstance(d2["iou%.2f" % t], list) else d2["iou%.2f" % t]) for t in j["thresholds"]]
                lines.append("| %s | " % cell + " | ".join("%.4f" % v for v in vals) + " | %.4f |" % st.mean(vals))
            open(os.path.join(OUT, "B4_iou_curve_%s.md" % ws), "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
            print("[IoU] 写出 B4_%s" % ws)
    print("\n完成，表在:", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
