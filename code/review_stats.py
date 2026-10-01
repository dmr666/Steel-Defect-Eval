# NOTE: input paths were machine-specific; the prefix has been replaced with '.'.
# Edit the path constants near the top if your layout differs from the repo layout.
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""review_stats.py — review A 档：多重比较校正（Welch + Holm）与逐种子稀有类 AP（供 Fig.4(b) 误差棒）

只读本地归档（各 small.tar.gz 里的 perclass_ap.json），输出：
  _results/review_stats/holm_test_partition.md   → SI Table S9 内容
  _results/review_stats/rare_ap_perseed.json     → Fig.4(b) 逐种子数据
  _results/review_stats/summary.txt              → 控制台摘要（哪些结论在 Holm 后变化）
"""
import json
import math
import os
import statistics as st
import sys
import tarfile

ROOT = "."
RES = os.path.join(ROOT, "_results")
OUT = os.path.join(RES, "review_stats")
os.makedirs(OUT, exist_ok=True)

sys.path.insert(0, "C:/Users/69416/AppData/Local/hermes/skills/research/sci-paper-workflow/scripts")
from t_test_pure_python import ttest_ind  # noqa: E402

SRC = {
    ("GC10", "s"): (f"{RES}/gc10_20260929/defect_gc10_runs_small.tar.gz", "runs"),
    ("GC10", "nano"): (f"{RES}/gc10_nano_20260929/defect_gc10_runs_nano_small.tar.gz", "runs_nano"),
    ("NEU", "s"): (f"{RES}/neu_ext_20260930/defect_runs_small.tar.gz", "runs"),
    ("NEU", "nano"): (f"{RES}/neu_nano_20260930/defect_runs_nano_small.tar.gz", "runs_nano"),
}
RARE = {"GC10": ["zhehen", "yahen"], "NEU": ["pitted_surface", "scratches"]}
IMBAL = ["v_p2dcsawiou", "v_full_cpw0p5", "v_full", "v_resample", "v_focal"]
SEEDS = ["42", "7", "2024"]


def load(ds, scale):
    path, sub = SRC[(ds, scale)]
    per = {}
    with tarfile.open(path, "r:gz") as tf:
        for m in tf.getmembers():
            if not m.name.endswith("perclass_ap.json"):
                continue
            cell_seed = m.name.split("/")[-2]
            if "_seed" not in cell_seed:
                continue
            cell, seed = cell_seed.rsplit("_seed", 1)
            if seed not in SEEDS:
                continue
            j = json.loads(tf.extractfile(m).read().decode("utf-8"))
            per.setdefault(cell, {})[seed] = j
    return per


def vals(per, cell, key="test95"):
    d = per.get(cell) or {}
    out = []
    for s in SEEDS:
        if s not in d:
            continue
        j = d[s]
        if key == "test95":
            out.append(float(j["test"]["mAP5095"]))
        elif key == "rare_val50":
            cl = j["classes"]
            out.append(sum(cl[c]["ap50"] for c in RARE[DS_CUR] if c in cl) / len(RARE[DS_CUR]))
    return out


def holm(pvals):
    """返回 Holm 校正后的 p 值（保序）"""
    idx = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals)
    adj = [0.0] * m
    prev = 0.0
    for k, i in enumerate(idx):
        v = min(1.0, (m - k) * pvals[i])
        prev = max(prev, v)
        adj[i] = prev
    return adj


lines_md = ["## Table S9. Multiple-comparison control for the held-out test comparisons", ""]
lines_md.append("Welch two-sample t-test (n = 3 versus 3 seeds) on the per-seed test mAP50-95 of each "
                "configuration against the unmodified baseline of the same dataset and scale; "
                "*p* values are Holm-adjusted across all comparisons of that family "
                "(14 configurations on GC10-DET and 14 on NEU-DET at the s scale; 4 shared configurations "
                "at the nano scale). The verdict column uses α = 0.05 after adjustment.")
lines_md.append("")
lines_md.append("| dataset | scale | configuration | mean Δ test mAP50-95 | Welch t | raw p | Holm p | verdict (α=0.05) |")
lines_md.append("|---|---|---|---|---|---|---|---|")

summary = []
rare_out = {}

for (ds, scale), _ in SRC.items():
    global DS_CUR
    DS_CUR = ds
    per = load(ds, scale)
    cells = sorted(per)
    base = vals(per, "v_base")
    fam, fam_names = [], []
    for c in sorted(per):
        if c == "v_base":
            continue
        v = vals(per, c)
        if len(v) < 3 or len(base) < 3:
            continue
        t, p, m1, m2 = ttest_ind(v, base)
        fam.append(p)
        fam_names.append((c, v, base, t, p, m1, m2))
    adj = holm(fam)
    order = sorted(range(len(fam)), key=lambda i: fam[i])
    for rank, i in enumerate(order):
        c, v, b, t, p, m, mb = fam_names[i]
        verdict = "significant" if adj[i] < 0.05 else "not significant"
        lines_md.append("| %s-DET | %s | %s | %+.4f | %.2f | %.4f | %.4f | %s |"
                        % (ds, scale, c, m - mb, t, p, adj[i], verdict))
        if adj[i] < 0.05:
            summary.append("%s %s %-16s Δ=%+.4f raw_p=%.4f Holm_p=%.4f → 显著" % (ds, scale, c, m - mb, p, adj[i]))
    # 稀有类 AP50 逐种子（不平衡臂）
    if scale == "s":
        for arm in IMBAL:
            r = vals(per, arm, key="rare_val50")
            if len(r) == 3:
                rare_out["%s|%s" % (ds, arm)] = {"seeds": r, "mean": st.mean(r), "sd": st.stdev(r)}
    print("%s %s: 家族 %d 个比较，Holm 后显著 %d 个"
          % (ds, scale, len(fam), sum(1 for x in adj if x < 0.05)))

lines_md.append("")
lines_md.append("Note: the main text reports the two-standard-deviation pooled rule (|Δ| > 2σ_pooled) as "
                "its primary criterion; the table above is a stricter, multiplicity-controlled check on the "
                "same per-seed data. Comparisons that are significant under the pooled rule but not after "
                "Holm adjustment are flagged as borderline in §5.")

open(os.path.join(OUT, "holm_test_partition.md"), "w", encoding="utf-8", newline="\n").write("\n".join(lines_md) + "\n")
json.dump(rare_out, open(os.path.join(OUT, "rare_ap_perseed.json"), "w", encoding="utf-8"), indent=1)
open(os.path.join(OUT, "summary.txt"), "w", encoding="utf-8", newline="\n").write(
    "Holm 调整后仍显著的比较（α=0.05）:\n" + ("\n".join(summary) if summary else "(无)") + "\n")
print("\nHolm 后仍显著：")
print("\n".join(summary) if summary else "(无)")
print("\n写出:", os.path.join(OUT, "holm_test_partition.md"), "|", os.path.join(OUT, "rare_ap_perseed.json"))
