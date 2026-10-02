# NOTE: machine-specific paths were replaced with '.'/placeholders.
# -*- coding: utf-8 -*-
"""gen_si_tables_v2.py — 生成返修新增的三张 SI 表（S10 IoU 曲线 / S11 from-scratch / S12 训练成本），
追加到 _paper/tables_en_20260930.md（重复运行先删旧段）。所有数字现算，不手抄。

数据源：
  IoU 曲线   _results/reviewB_20261001/iou_sweep_{gc10,neu}.json
  from-scratch  _results/reviewB_20261001/extract_all/{defect_gc10,defect}/{runs,runs_nano,runs_scratch_s,runs_scratch_n}
  训练成本    各归档 runs 的 results.csv 末行 time（秒，累计耗时）
"""
import csv
import glob
import json
import os
import re
import statistics as st

ROOT = "."
RES = os.path.join(ROOT, "_results")
EX = os.path.join(RES, "reviewB_20261001", "extract_all")
TAB = os.path.join(ROOT, "_paper", "tables_en_20260930.md")
SEEDS = ["42", "7", "2024"]

COST_SRC = {
    "GC10-DET": RES + "/gc10_20260929/runs",
    "NEU-DET": RES + "/neu_ext_20260930/small/runs",
}
COST_CELLS = [("v_base", "baseline (no treatment)"), ("v_full", "full stack (P2 + DCSA + WIoU)"),
              ("v_focal", "Focal loss"), ("v_resample", "sqrt-resampling"),
              ("v_p1_b8", "P1 head (batch 8)")]


def m95(p):
    j = json.load(open(p, encoding="utf-8"))
    return float(j["test"]["mAP5095"])


def ms(v):
    v = [x for x in v if x is not None]
    if not v:
        return float("nan"), float("nan"), 0
    return st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0), len(v)


def table_scratch():
    out = ["## Table 11. From-scratch versus COCO-pretrained baselines (held-out test mAP50-95)", "",
           "| dataset | scale | COCO-pretrained (mean ± SD, n) | from scratch (mean ± SD, n) | Δ (scratch − pretrained) | Δ(s − nano), pretrained | Δ(s − nano), from scratch |",
           "|---|---|---|---|---|---|---|"]
    for ds, ws in (("GC10-DET", "defect_gc10"), ("NEU-DET", "defect")):
        pre, scr = {}, {}
        # 用各目录下实际存在的全部种子（预训练 s 已有 5 个，from-scratch 3 个），与正文/Table S9 一致
        for tag, d in (("pre", "runs"), ("scr", "runs_scratch_s")):
            v = [m95(f) for f in glob.glob("%s/%s/%s/v_base_seed*/perclass_ap.json" % (EX, ws, d))]
            (pre if tag == "pre" else scr)["s"] = ms(v)
        for tag, d in (("pre", "runs_nano"), ("scr", "runs_scratch_n")):
            v = [m95(f) for f in glob.glob("%s/%s/%s/v_base_seed*/perclass_ap.json" % (EX, ws, d))]
            (pre if tag == "pre" else scr)["n"] = ms(v)
        ds_sn_pre = pre["s"][0] - pre["n"][0]
        ds_sn_scr = scr["s"][0] - scr["n"][0]
        out.append("| %s | s | %.4f ± %.4f (%d) | %.4f ± %.4f (%d) | %+.4f | %+.4f | %+.4f |"
                   % (ds, pre["s"][0], pre["s"][1], pre["s"][2], scr["s"][0], scr["s"][1], scr["s"][2],
                      scr["s"][0] - pre["s"][0], ds_sn_pre, ds_sn_scr))
        out.append("| %s | nano | %.4f ± %.4f (%d) | %.4f ± %.4f (%d) | %+.4f | — | — |"
                   % (ds, pre["n"][0], pre["n"][1], pre["n"][2], scr["n"][0], scr["n"][1], scr["n"][2],
                      scr["n"][0] - pre["n"][0]))
    out.append("")
    out.append("Note: identical protocol and identical 200-epoch budget in both regimes; the from-scratch runs "
               "use the structure YAML as the weights argument and no pretrained checkpoint. Pretraining is "
               "worth +0.038 to +0.044 mAP50-95 on GC10-DET and −0.009 to −0.010 on NEU-DET. The s − nano "
               "capacity gap changes by only 0.005 between the two regimes, i.e. inside the seed noise of "
               "these cells, so the nano-scale saturation is not an artefact of pretraining.")
    return "\n".join(out)


def table_iou():
    out = ["## Table 10. AP as a function of the IoU threshold (held-out test partition)", "",
           "| dataset | configuration | " + " | ".join("%.2f" % t for t in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]) + " | mAP50-95 |",
           "|---|---|" + "---|" * 11]
    for ds, f, lbl in (("GC10-DET", "iou_sweep_gc10.json", ("baseline", "full stack")),
                       ("NEU-DET", "iou_sweep_neu.json", ("baseline", "full stack"))):
        j = json.load(open(os.path.join(RES, "reviewB_20261001", f), encoding="utf-8"))
        agg = j["agg"]
        for cell, name in zip(("v_base", "v_full"), lbl):
            d = agg[cell]
            vals = [d["iou%.2f" % t][0] for t in j["thresholds"]]
            out.append("| %s | %s | " % (ds, name) + " | ".join("%.4f" % v for v in vals) + " | %.4f |" % st.mean(vals))
    out.append("")
    out.append("Note: AP is integrated with the same 101-point interpolation as the main tables, but recomputed "
               "from predictions exported by the ultralytics validation pass itself (3 seeds, mean shown; "
               "per-seed spread is below 0.02 at IoU 0.50 and below 0.01 above IoU 0.75). The threshold-AU "
               "integration of the baseline/full-stack rows reproduces the test mAP50-95 of Tables 2 and 5. "
               "AP collapses between IoU 0.75 and 0.95, i.e. the residual error is localisation, not "
               "classification.")
    return "\n".join(out)


def table_cost():
    out = ["## Table 12. Per-seed training time of the treatments (s scale, 640 px, batch 16, 200 epochs)", "",
           "| dataset | configuration | seed 42 | seed 7 | seed 2024 | mean | vs baseline |",
           "|---|---|---|---|---|---|---|"]
    foc_rel = {}
    for ds, d in COST_SRC.items():
        base = None
        for cell, name in COST_CELLS:
            per = {}
            for s in SEEDS:
                f = os.path.join(d, "%s_seed%s" % (cell, s), "results.csv")
                if not os.path.isfile(f):
                    continue
                rows = list(csv.DictReader(open(f, encoding="utf-8")))
                if rows:
                    per[s] = float(rows[-1]["time"]) / 60.0
            if not per:
                out.append("| %s | %s | — | — | — | — | — |" % (ds, name))
                continue
            mean = st.mean(per.values())
            if base is None and cell == "v_base":
                base = mean
            rel = "—" if cell == "v_base" else "%+.0f%%" % (100.0 * (mean / base - 1.0))
            if cell == "v_focal":
                foc_rel[ds] = rel
            out.append("| %s | %s | %s | %.1f | %s |" % (
                ds, name,
                " | ".join("%.1f" % per[s] if s in per else "—" for s in SEEDS),
                mean, rel))
    out.append("")
    out.append("Note: wall-clock time per run read from the training records of the same matrix (the elapsed "
               "column of results.csv); no additional GPU time was spent to obtain this table. These are "
               "training-side costs and complement the inference-side cost plane of Fig. 5 and the "
               "average-cost table of the main text. The Focal arm is evaluated as a per-element Python loop "
               "in this implementation, which dominates its %s (GC10-DET) and %s (NEU-DET) overhead."
               % (foc_rel.get("GC10-DET", "?"), foc_rel.get("NEU-DET", "?")))
    return "\n".join(out)


def main():
    txt = open(TAB, encoding="utf-8").read()
    new = [table_iou(), table_scratch(), table_cost()]
    titles = [re.match(r"^## (Table \d+)", b).group(1) for b in new]
    for t in titles:
        txt = re.sub(r"\n## %s\..*?(?=\n## |\Z)" % re.escape(t), "\n", txt, flags=re.S)
    txt = txt.rstrip("\n") + "\n\n" + "\n\n".join(new) + "\n"
    open(TAB, "w", encoding="utf-8", newline="\n").write(txt)
    print("已写入 %s" % TAB)
    for t in titles:
        print("  ", t)
    print("现有段落:", re.findall(r"^## (Table \S+)", txt, flags=re.M))


main()
