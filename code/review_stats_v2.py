# NOTE: machine-specific paths were replaced with '.'/placeholders.
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""review_stats_v2.py — 重算 SI Table S9（Holm 多重比较校正）

与 review_stats.py 的差别（review返修 B 档）：
  1) 数据源换成已合并的本地归档 _results/reviewB_20261001/extract_all
     （= 原主矩阵 + 本次补的 seed 13/99），不再逐个读 tar；
  2) 关键格子允许 n=5（v_base / v_full / v_focal 及同等补了种子的臂），
     其余仍是 n=3；表里加一列 n（cfg/base）如实标出；
  3) 其余（Welch + Holm、行的排序、rare_ap_perseed.json）与原脚本一致。

输出（覆盖）:
  _results/review_stats/holm_test_partition.md   → SI Table S9
  _results/review_stats/rare_ap_perseed.json     → Fig.4(b) 逐种子（仍只用主 3 种子）
  _results/review_stats/summary.txt
原 n=3 版本先备份成 holm_test_partition.n3_backup.md
"""
import glob
import json
import os
import shutil
import statistics as st
import sys

ROOT = "."
RES = os.path.join(ROOT, "_results")
OUT = os.path.join(RES, "review_stats")
EX = os.path.join(RES, "reviewB_20261001", "extract_all")

# Pure-stdlib independent-samples t-test, inlined so this file runs standalone.
def _t_cdf(t, df):
    def pdf(x):
        c = math.gamma((df + 1) / 2) / (math.sqrt(df * math.pi) * math.gamma(df / 2))
        return c * (1 + x * x / df) ** (-(df + 1) / 2)
    a, b, n = -20.0, t, 40000
    h = (b - a) / n
    s = pdf(a) + pdf(b)
    for i in range(1, n):
        x = a + i * h
        s += pdf(x) * (4 if i % 2 else 2)
    return s * h / 3


def ttest_ind(a, b):
    """Independent-samples t-test: returns (t, p, mean_a, mean_b)."""
    na, nb = len(a), len(b)
    ma, mb = sum(a) / na, sum(b) / nb
    va = sum((x - ma) ** 2 for x in a) / (na - 1)
    vb = sum((x - mb) ** 2 for x in b) / (nb - 1)
    t = (ma - mb) / math.sqrt(va / na + vb / nb)
    return t, 2 * (1 - _t_cdf(abs(t), na + nb - 2)), ma, mb


WS = {"GC10": "defect_gc10", "NEU": "defect"}
SCALES = {"s": "runs", "nano": "runs_nano"}
SEEDS_MAIN = ["42", "7", "2024"]
RARE = {"GC10": ["zhehen", "yahen"], "NEU": ["pitted_surface", "scratches"]}
IMBAL = ["v_p2dcsawiou", "v_full_cpw0p5", "v_full", "v_resample", "v_focal"]


def load(ds, scale):
    d = {}
    pat = os.path.join(EX, WS[ds], SCALES[scale], "**", "perclass_ap.json")
    for f in glob.glob(pat, recursive=True):
        run = os.path.basename(os.path.dirname(f))
        if "_seed" not in run:
            continue
        cell, seed = run.rsplit("_seed", 1)
        if cell.endswith("_e300") and cell not in ("v_base_e300",):
            pass  # 保留 e300 变体（原表里就有 v_base_e300 / v_full_e300）
        try:
            j = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        d.setdefault(cell, {})[seed] = j
    return d


def t95(j):
    return float(j["test"]["mAP5095"])


def vals(per, cell):
    d = per.get(cell) or {}
    return [t95(d[s]) for s in SEEDS_MAIN if s in d] + \
           [t95(d[s]) for s in sorted(d) if s not in SEEDS_MAIN]


ALL_SEEDS = ["42", "7", "2024", "13", "99"]


def paired_vals(per, cell):
    """cfg 与 v_base 配对的种子集合：只有两边都补了 13/99 才用 5 种子，
    否则退回主 3 种子 —— 保证每个比较内部 cfg 与 base 的 n 相同。"""
    d = per.get(cell) or {}
    bd = per.get("v_base") or {}
    extra = all(s in d for s in ("13", "99")) and all(s in bd for s in ("13", "99"))
    seeds = ALL_SEEDS if extra else SEEDS_MAIN
    v = [t95(d[s]) for s in seeds if s in d]
    b = [t95(bd[s]) for s in seeds if s in bd]
    return v, b


def rare50(j, ds):
    cl = j["classes"]
    return sum(cl[c]["ap50"] for c in RARE[ds] if c in cl) / len(RARE[ds])


def holm(pvals):
    idx = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals)
    adj = [0.0] * m
    prev = 0.0
    for k, i in enumerate(idx):
        v = min(1.0, (m - k) * pvals[i])
        prev = max(prev, v)
        adj[i] = prev
    return adj


def main():
    old = os.path.join(OUT, "holm_test_partition.md")
    if os.path.isfile(old) and not os.path.isfile(os.path.join(OUT, "holm_test_partition.n3_backup.md")):
        shutil.copy2(old, os.path.join(OUT, "holm_test_partition.n3_backup.md"))
        print("原 n=3 表已备份: holm_test_partition.n3_backup.md")

    lines_md = ["## Table S9. Multiple-comparison control for the held-out test comparisons", ""]
    summary = []
    rare_out = {}
    n5_cells = []

    for ds in ("GC10", "NEU"):
        for scale in ("s", "nano"):
            per = load(ds, scale)
            base = vals(per, "v_base")
            if len(base) < 3:
                print("!! %s %s: v_base 只有 %d 个种子，跳过" % (ds, scale, len(base)))
                continue
            fam, info = [], []
            for c in sorted(per):
                if c == "v_base":
                    continue
                v, b = paired_vals(per, c)
                if len(v) < 3 or len(b) < 3:
                    continue
                t, p, m1, m2 = ttest_ind(v, b)
                fam.append(p)
                info.append((c, len(v), len(b), m1, m2, t, p))
            adj = holm(fam)
            order = sorted(range(len(fam)), key=lambda i: fam[i])
            for i in order:
                c, nv, nb, m, mb, t, p = info[i]
                verdict = "significant" if adj[i] < 0.05 else "not significant"
                lines_md.append("| %s-DET | %s | %s | %d / %d | %+.4f | %.2f | %.4f | %.4f | %s |"
                                % (ds, scale, c, nv, nb, m - mb, t, p, adj[i], verdict))
                if adj[i] < 0.05:
                    summary.append("%s %s %-16s n=%d/%d Δ=%+.4f raw_p=%.4f Holm_p=%.4f" %
                                   (ds, scale, c, nv, nb, m - mb, p, adj[i]))
                if nv > 3:
                    n5_cells.append("%s %s %s (n=%d)" % (ds, scale, c, nv))
            if scale == "s":
                for arm in IMBAL:
                    d = per.get(arm) or {}
                    r = [rare50(d[s], ds) for s in SEEDS_MAIN if s in d]
                    if len(r) == 3:
                        rare_out["%s|%s" % (ds, arm)] = {"seeds": r, "mean": st.mean(r), "sd": st.stdev(r)}
            print("%s %s: 家族 %d 个比较，Holm 后显著 %d 个" %
                  (ds, scale, len(fam), sum(1 for x in adj if x < 0.05)))

    lines_md.append("")
    lines_md.append("Note: Welch two-sample t-test on the per-seed test mAP50-95 of each configuration against "
                    "the unmodified baseline of the same dataset and scale; the n column gives the number of "
                    "seeds behind the configuration and the baseline. The key configurations (baseline, full "
                    "stack, Focal) were extended from 3 to 5 seeds in revision, so their rows carry a larger n "
                    "than the rest of the family; *p* values are Holm-adjusted across all comparisons of that "
                    "family (s scale per dataset; shared configurations at the nano scale). The verdict column "
                    "uses α = 0.05 after adjustment. The main text reports the two-standard-deviation pooled "
                    "rule (|Δ| > 2σ_pooled) as its primary criterion; this table is the stricter, "
                    "multiplicity-controlled check on the same per-seed data.")

    open(old, "w", encoding="utf-8", newline="\n").write("\n".join(lines_md) + "\n")
    json.dump(rare_out, open(os.path.join(OUT, "rare_ap_perseed.json"), "w", encoding="utf-8"), indent=1)
    open(os.path.join(OUT, "summary.txt"), "w", encoding="utf-8", newline="\n").write(
        "Holm 调整后仍显著的比较（α=0.05）:\n" + ("\n".join(summary) if summary else "(无)") + "\n")
    print("\nHolm 后仍显著：")
    print("\n".join(summary) if summary else "(无)")
    print("\n扩到 n>3 的格子：", ", ".join(n5_cells) if n5_cells else "(无)")
    print("\n写出:", old)


main()
