# NOTE: input paths were machine-specific; the prefix has been replaced with '.'.
# Edit the path constants near the top if your layout differs from the repo layout.
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""analyze_matrix.py - 把 runs/ 下的 perclass_ap.json 汇总成论文用表（纯标准库）

用法：
    python analyze_matrix.py runs                 # 就地打印 + 写 tables/ 目录
    python analyze_matrix.py runs --out tables

产出（写在 <out>/ 下，TSV 便于直接粘进 Excel/Word）：
    table_main.tsv        主表：每个 cell 的 val/test mAP（种子均值±标准差）
    table_imbalance.tsv   不平衡四臂（含 cls_pw 0.5 档）：总体 + 稀有类/多数类 AP
    table_resolution.tsv  分辨率扫描 + 浅层头一行
    perclass_matrix.tsv   每类 AP（种子均值）× 各 cell
    integrity.txt         完整性/有效性报告：缺哪些 run、哪些必须作废

纪律：只认 json 里的数（种子矩阵），缺 run 就报缺，不用任何推算值填充。
"""

import argparse
import json
import os
import re
import statistics as st
import sys

# 与 run_gc10.sh / run_gc10_ext.sh 的 cell 列表一致（GC10 主线，13 cell × 3 种子）
EXPECTED = [
    ("v_base", 640, 16, 0.0), ("v_p2", 640, 16, 0.0), ("v_dcsa", 640, 16, 0.0),
    ("v_wiou", 640, 16, 0.0), ("v_p2dcsawiou", 640, 16, 0.0), ("v_full", 640, 16, 1.0),
    ("v_full", 640, 16, 0.5), ("v_p1", 640, 8, 0.0),
    ("v_base", 320, 8, 0.0), ("v_base", 640, 8, 0.0), ("v_base", 960, 8, 0.0),
    ("v_resample", 640, 16, 0.0), ("v_focal", 640, 16, 0.0),
]
SEEDS = (42, 7, 2024)
# 可选加固 cell（300 轮同算力对照）：跑过才进表，没跑不算"缺失"
OPTIONAL = [
    ("v_base", 640, 16, 0.0, 300),
    ("v_full", 640, 16, 1.0, 300),
]
NAME_RE = re.compile(r"^(?P<variant>v_[a-z0-9]+?)(?:_s(?P<imgsz>\d+))?(?:_b(?P<batch>\d+))?"
                     r"(?:_cpw(?P<cpw>\d+p\d+))?(?:_e(?P<epochs>\d+))?_seed(?P<seed>\d+)$")


def cell_name(variant, imgsz, batch, cls_pw, epochs=200):
    """与 defect_train.run_name 同规则：非默认值才进名字（默认 imgsz=640 / batch=16 / epochs=200）。"""
    tag = variant
    if imgsz != 640:
        tag += "_s%d" % imgsz
    if batch != 16:
        tag += "_b%d" % batch
    if cls_pw != 0.0 and not (variant == "v_full" and cls_pw == 1.0):
        tag += "_cpw%s" % ("%.1f" % cls_pw).replace(".", "p")
    if epochs != 200:
        tag += "_e%d" % epochs
    return tag


def load(runs_dir):
    recs, bad = [], []
    for n in sorted(os.listdir(runs_dir)):
        f = os.path.join(runs_dir, n, "perclass_ap.json")
        if not os.path.isfile(f):
            continue
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception as e:
            bad.append((n, "json 读取失败: %s" % e))
            continue
        m = NAME_RE.match(n)
        d["_dir"] = n
        d["_batch"] = int(m.group("batch")) if (m and m.group("batch")) else 16
        d["_epochs"] = int(m.group("epochs")) if (m and m.group("epochs")) else 200
        d["_cell"] = cell_name(d.get("variant", "?"), d.get("imgsz", 640),
                               d["_batch"], float(d.get("cls_pw", 0.0)), d["_epochs"])
        recs.append(d)
    return recs, bad


def ms(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "—"
    if len(vals) == 1:
        return "%.4f" % vals[0]
    return "%.4f±%.4f" % (st.mean(vals), st.stdev(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="?", default="runs")
    ap.add_argument("--out", default="tables")
    ap.add_argument("--classes", default="", help="逗号分隔的稀有类名（不平衡表用），默认取实例最少的两个")
    ap.add_argument("--stats", default="", help="split_stats.json 路径（不给则自动在 runs/../data 下找）")
    a = ap.parse_args()

    recs, bad = load(a.runs)
    os.makedirs(a.out, exist_ok=True)
    L = []

    def out(s=""):
        print(s)
        L.append(s)

    # ---------- 完整性 ----------
    have = {}
    for d in recs:
        have.setdefault(d["_cell"], {})[d["seed"]] = d
    expected_names = {cell_name(v, i, b, c) for v, i, b, c in EXPECTED}
    missing = []
    for v, i, b, c in EXPECTED:
        n = cell_name(v, i, b, c)
        got = sorted(have.get(n, {}).keys())
        if sorted(got) != sorted(SEEDS):
            missing.append((n, [s for s in SEEDS if s not in got]))
    opt_names = {cell_name(v, i, b, c, e) for v, i, b, c, e in OPTIONAL}
    extra = sorted(set(have) - expected_names - opt_names)
    opt_done = {n: sorted(have[n].keys()) for n in sorted(opt_names) if n in have}
    bad_dcsa = [d["_dir"] for d in recs
                if d.get("variant") in ("v_dcsa", "v_p2dcsawiou", "v_full", "v_resample", "v_focal")
                and d.get("ckpt_has_dcsa") is not True]

    out("=" * 78)
    out("矩阵完整性报告  (%s)" % os.path.abspath(a.runs))
    out("=" * 78)
    out("已收到 run: %d / %d（%d 个 cell × %d 种子）" % (len(recs), len(EXPECTED) * len(SEEDS),
                                                    len(EXPECTED), len(SEEDS)))
    if missing:
        out("未完成 cell（%d 个）:" % len(missing))
        for n, ss in missing:
            out("   %-26s 缺种子 %s" % (n, ss))
    else:
        out("全部 cell 齐 ✓")
    if opt_done:
        out("可选加固 cell（300 轮）已完成：")
        for n, ss in opt_done.items():
            out("   %-26s 种子 %s" % (n, ss))
    else:
        out("可选加固 cell（300 轮）：未跑（不影响主表）")
    if extra:
        out("警告：出现预期外的 run 目录（%s）——检查是否命名漂移或重复配置" % ", ".join(extra))
    if bad_dcsa:
        out("!! 必须作废的 run（带 DCSA 变体但 ckpt_has_dcsa 非 True）：%s" % ", ".join(bad_dcsa))
    for n, why in bad:
        out("!! %s: %s" % (n, why))
    if not missing and not extra and not bad_dcsa and not bad:
        out("结构检查全部通过。")

    # ---------- 主表 ----------
    out("")
    out("=" * 78)
    out("表 A 主表（种子均值±标准差，离散度=样本标准差 n-1）  val = 验证划分, test = 有 test 字段的 run（训练时 --final-test 或 eval_test.py 补测）")
    out("=" * 78)
    hdr = ("cell", "variant", "imgsz", "batch", "cls_pw", "n", "val mAP50-95", "val mAP50",
           "test mAP50-95", "params", "不平衡臂", "ckpt_has_dcsa")
    rows = []
    for v, i, b, c in EXPECTED:
        n = cell_name(v, i, b, c)
        ss = have.get(n)
        if not ss:
            rows.append((n, v, i, b, "%.1f" % c, 0, "—", "—", "—", "—", "—", "—"))
            continue
        ds = [ss[s] for s in sorted(ss)]
        vals = [d.get("mAP5095") for d in ds]
        v50 = [d.get("mAP50") for d in ds]
        tst = [d.get("test", {}).get("mAP5095") for d in ds if isinstance(d.get("test"), dict)]
        prm = ds[0].get("params", "—")
        arm = ds[0].get("imbalance_arm", "—")
        ck = all(d.get("ckpt_has_dcsa") is True for d in ds) if v in (
            "v_dcsa", "v_p2dcsawiou", "v_full", "v_resample", "v_focal") else "—"
        rows.append((n, v, i, b, "%.1f" % c, len(ds), ms(vals), ms(v50),
                     ms([x for x in tst if x is not None]) if tst else "—",
                     (str("%d" % prm)) if isinstance(prm, int) else str(prm), str(arm), str(ck)))
    # 可选加固 cell（300 轮）：跑过才进表
    for v, i, b, c, ep in OPTIONAL:
        n = cell_name(v, i, b, c, ep)
        ss = have.get(n)
        if not ss:
            continue
        ds = [ss[s] for s in sorted(ss)]
        tst = [d.get("test", {}).get("mAP5095") for d in ds if isinstance(d.get("test"), dict)]
        prm = ds[0].get("params", "—")
        rows.append((n, v, i, b, "%.1f" % c, len(ds),
                     ms([d.get("mAP5095") for d in ds]), ms([d.get("mAP50") for d in ds]),
                     ms([x for x in tst if x is not None]) if tst else "—",
                     (str("%d" % prm)) if isinstance(prm, int) else str(prm),
                     str(ds[0].get("imbalance_arm", "—")), "可选"))
    with open(os.path.join(a.out, "table_main.tsv"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(hdr) + "\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")
    _dicts = [dict(zip(hdr, x)) for x in rows]
    w = [max([len(hdr[k])] + [len(str(d[hdr[k]])) for d in _dicts]) for k in range(len(hdr))]
    out("  ".join(h.ljust(w[k]) for k, h in enumerate(hdr)))
    for r in rows:
        out("  ".join(str(x).ljust(w[k]) for k, x in enumerate(r)))

    # ---------- 不平衡表 ----------
    out("")
    out("=" * 78)
    out("表 B 不平衡处理四臂（同架构：P2 + DCSA + WIoU；只差不平衡处理方式）")
    out("=" * 78)
    arms = [("无处理", cell_name("v_p2dcsawiou", 640, 16, 0.0)),
            ("指数加权 0.5", cell_name("v_full", 640, 16, 0.5)),
            ("指数加权 1.0", cell_name("v_full", 640, 16, 1.0)),
            ("重采样", cell_name("v_resample", 640, 16, 0.0)),
            ("Focal", cell_name("v_focal", 640, 16, 0.0))]
    cls_names = None
    for _, n in arms:
        for d in have.get(n, {}).values():
            cls_names = list(d.get("classes", {}).keys())
            break
        if cls_names:
            break
    inst_hint = ""
    if cls_names:
        if a.classes:
            rare = [c for c in a.classes.split(",") if c in cls_names]
        else:
            # 稀有类判据应当是「实例数最少」，不是「AP 最低」——优先读 split_stats.json
            cands = [a.stats] if a.stats else []
            cands += [os.path.join(a.runs, "..", "data", d, "split_stats.json")
                      for d in ("GC10-DET-yolo", "NEU-DET-split")]
            rare = []
            for p in cands:
                if p and os.path.isfile(p):
                    try:
                        s = json.load(open(p, encoding="utf-8"))
                        inst = s["stats"]["train"]["instances"]
                        ns = s.get("names") or list(cls_names)
                        order = sorted(range(len(inst)), key=lambda k: inst[k])
                        rare = [ns[k] for k in order[:2] if ns[k] in cls_names]
                        print("稀有类来自 %s：%s" % (os.path.basename(os.path.dirname(p)), rare))
                    except Exception as e:
                        print("读 split_stats 失败(%s): %s" % (p, e))
                    if rare:
                        break
            if not rare:   # 兜底：退回 AP 最低的两类，并明确打印警告
                ref = None
                for _, n in arms:
                    for d in have.get(n, {}).values():
                        ref = d.get("classes", {})
                        break
                    if ref:
                        break
                rare = sorted(ref, key=lambda c: ref[c]["ap50"])[:2] if ref else []
                print("!! 没找到 split_stats.json，稀有类退回「AP 最低两类」口径：%s" % rare)
        inst_hint = ",".join(rare)
        hdr2 = ("臂", "cell", "n", "mAP50-95", "mAP50", "稀有类(%s)" % inst_hint, "其余类均值")
        with open(os.path.join(a.out, "table_imbalance.tsv"), "w",
                  encoding="utf-8", newline="\n") as f:
            f.write("\t".join(hdr2) + "\n")
            out("  ".join(hdr2))
            for label, n in arms:
                ss = have.get(n)
                if not ss:
                    line = (label, n, "0", "—", "—", "—", "—")
                else:
                    ds = list(ss.values())
                    per = [[d["classes"][c]["ap50"] for c in rare] for d in ds if "classes" in d]
                    rmean = st.mean([st.mean(x) for x in per]) if per else None
                    others = [st.mean([v2["ap50"] for c2, v2 in d["classes"].items() if c2 not in rare])
                              for d in ds if "classes" in d]
                    line = (label, n, str(len(ds)), ms([d.get("mAP5095") for d in ds]),
                            ms([d.get("mAP50") for d in ds]),
                            ("%.4f" % rmean) if rmean is not None else "—",
                            ms(others))
                out("  ".join(str(x) for x in line))
                f.write("\t".join(str(x) for x in line) + "\n")
        out("注：稀有类按该数据集实例数最少的两类（当前判定：%s）；报告时要给出处置前/后的实例分布。" % inst_hint)

    # ---------- 分辨率 / 浅层头 ----------
    out("")
    out("=" * 78)
    out("表 C 输入分辨率扫描（统一 batch=8）与浅层头（P1）")
    out("=" * 78)
    res = [("320", cell_name("v_base", 320, 8, 0.0)), ("640", cell_name("v_base", 640, 8, 0.0)),
           ("960", cell_name("v_base", 960, 8, 0.0)), ("P1 头 640", cell_name("v_p1", 640, 8, 0.0)),
           ("P2 头 640 (b16)", cell_name("v_p2", 640, 16, 0.0))]
    hdr3 = ("配置", "cell", "n", "val mAP50-95", "val mAP50", "params")
    with open(os.path.join(a.out, "table_resolution.tsv"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(hdr3) + "\n")
        out("  ".join(hdr3))
        for label, n in res:
            ss = have.get(n)
            if not ss:
                line = (label, n, "0", "—", "—", "—")
            else:
                ds = list(ss.values())
                line = (label, n, str(len(ds)), ms([d.get("mAP5095") for d in ds]),
                        ms([d.get("mAP50") for d in ds]), str(ds[0].get("params", "—")))
            out("  ".join(str(x) for x in line))
            f.write("\t".join(str(x) for x in line) + "\n")

    # ---------- 每类 AP 矩阵 ----------
    if cls_names:
        with open(os.path.join(a.out, "perclass_matrix.tsv"), "w",
                  encoding="utf-8", newline="\n") as f:
            hdr4 = ["cell", "n"] + cls_names + ["mAP50-95"]
            f.write("\t".join(hdr4) + "\n")
            for v, i, b, c in EXPECTED:
                n = cell_name(v, i, b, c)
                ss = have.get(n)
                if not ss:
                    continue
                ds = list(ss.values())
                row = [n, str(len(ds))]
                for cn in cls_names:
                    row.append(ms([d["classes"][cn]["ap50"] for d in ds if cn in d.get("classes", {})]))
                row.append(ms([d.get("mAP5095") for d in ds]))
                f.write("\t".join(row) + "\n")
        out("")
        out("每类 AP 矩阵已写 perclass_matrix.tsv（%d 类 × %d cell）" % (len(cls_names), len(EXPECTED)))

    with open(os.path.join(a.out, "integrity.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L) + "\n")
    print("\n表已写入: %s" % os.path.abspath(a.out))
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
