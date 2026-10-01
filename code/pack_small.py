# NOTE: input paths were machine-specific; the prefix has been replaced with '.'.
# Edit the path constants near the top if your layout differs from the repo layout.
"""09_pack_small.py - 核表小包：把 runs/ 的核表件 + tables/ + runs_deploy/ 打成一个几 MB 的 tar.gz（不含权重）

用法:
    python 09_pack_small.py [ROOT] [OUTDIR]
      ROOT   结果根目录（默认当前目录），例如 ${DATA_ROOT}/neu
      OUTDIR 输出目录（默认 = ROOT）

产物:
    <OUTDIR>/<根目录名>_small.tar.gz   含 runs/<cell>/{results.csv,args.yaml,perclass_ap.json}、
                                     tables/、runs_deploy/{summary.tsv,env.txt,<cell>.json}
    tables/table_deploy.tsv            部署测量按 cell 聚合（种子均值±标准差），论文表 7 用
最后打印一段 COPY_BEGIN/COPY_END，只有十几到几十行，可直接贴回来核对；完整数据在 tar.gz 里。
"""
import json
import os
import re
import statistics as st
import sys
import tarfile
import time

NAME_RE = re.compile(r"^(?P<variant>v_[a-z0-9]+?)(?:_s(?P<imgsz>\d+))?(?:_b(?P<batch>\d+))?"
                     r"(?:_cpw(?P<cpw>\d+p\d+))?(?:_e(?P<epochs>\d+))?_seed(?P<seed>\d+)$")


def cellkey(name):
    m = NAME_RE.match(name)
    if not m:
        return None
    return name[:name.rfind("_seed")]


def ms(x):
    return "%.2f" % x


def mean_std(vals):
    vals = [v for v in vals if isinstance(v, (int, float))]
    if not vals:
        return "NA"
    if len(vals) == 1:
        return ms(vals[0])
    # 离散度口径必须与 analyze_matrix.py 一致（样本标准差 n-1），否则同一份产物里两张表的 ± 不可比
    return "%s±%s" % (ms(st.mean(vals)), ms(st.stdev(vals)))


def scan_runs(root, runsdir="runs"):
    runs = os.path.join(root, runsdir)
    info = {"dirs": 0, "cells": 0, "bestpt": [], "json": [], "test": [], "csv": [],
            "data_yaml": set(), "nclass": set()}
    if not os.path.isdir(runs):
        return info
    for n in sorted(os.listdir(runs)):
        d = os.path.join(runs, n)
        if not (os.path.isdir(d) and NAME_RE.match(n)):
            continue
        info["dirs"] += 1
        if os.path.isfile(os.path.join(d, "weights", "best.pt")):
            info["bestpt"].append(n)
        js = os.path.join(d, "perclass_ap.json")
        if os.path.isfile(js):
            info["json"].append(n)
            try:
                rec = json.load(open(js, encoding="utf-8"))
            except Exception:
                rec = {}
            # 数据集身份：防止把另一个数据集的结果当成这次的目标收进来（历史上真发生过）
            if rec.get("data_yaml"):
                info["data_yaml"].add(str(rec["data_yaml"]))
            if isinstance(rec.get("classes"), dict):
                info["nclass"].add(len(rec["classes"]))
            if isinstance(rec.get("test"), dict) and rec["test"].get("mAP50") is not None:
                info["test"].append(n)
        if os.path.isfile(os.path.join(d, "results.csv")):
            info["csv"].append(n)
    info["cells"] = len(set(cellkey(n) for n in info["json"]))
    return info


def build_deploy_table(root, tables_dir):
    """把 runs_deploy/<cell>.json 按 cell（去掉 _seedN）聚合成论文表 7 用的一行一个 cell。"""
    dd = os.path.join(root, "runs_deploy")
    if not os.path.isdir(dd):
        return None
    groups = {}
    for n in sorted(os.listdir(dd)):
        if not n.endswith(".json"):
            continue
        name = n[:-5]
        k = cellkey(name)
        if k is None:
            continue
        try:
            rec = json.load(open(os.path.join(dd, n), encoding="utf-8"))
        except Exception:
            continue
        g = groups.setdefault(k, {"seeds": [], "imgsz": rec.get("imgsz"), "params": rec.get("params"),
                                  "gflops": rec.get("gflops"), "mb": rec.get("model_MB"),
                                  "fp32": [], "fp16": [], "cpu": [], "mem": []})
        g["seeds"].append(name)
        for src, dst in (("pt_fp32", "fp32"), ("pt_fp16", "fp16"), ("onnx_cpu", "cpu")):
            v = (rec.get(src) or {}).get("p50")
            if isinstance(v, (int, float)):
                g[dst].append(v)
        if isinstance(rec.get("gpu_mem_MB"), (int, float)):
            g["mem"].append(rec["gpu_mem_MB"])
    if not groups:
        return None
    if not os.path.isdir(tables_dir):
        os.makedirs(tables_dir)
    out = os.path.join(tables_dir, "table_deploy.tsv")
    hdr = ["cell", "seeds", "imgsz", "params", "GFLOPs", "pt_fp32_p50_ms", "pt_fp16_p50_ms",
           "onnx_cpu_p50_ms", "gpu_mem_MB", "model_MB"]
    lines = ["\t".join(hdr)]
    for k in sorted(groups):
        g = groups[k]
        lines.append("\t".join([
            k, str(len(g["seeds"])), str(g["imgsz"]), str(g["params"]), str(g["gflops"]),
            mean_std(g["fp32"]), mean_std(g["fp16"]), mean_std(g["cpu"]),
            mean_std(g["mem"]), str(g["mb"])]))
    open(out, "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
    return out


def pack(root, outdir, runsdir="runs", tabledir="tables"):
    runs = os.path.join(root, runsdir)
    tar_path = os.path.join(outdir, os.path.basename(os.path.abspath(root)) + "_" + runsdir + "_small.tar.gz")
    n_csv = n_args = n_json = 0
    with tarfile.open(tar_path, "w:gz") as t:
        if os.path.isdir(runs):
            for cell in sorted(os.listdir(runs)):
                d = os.path.join(runs, cell)
                if not (os.path.isdir(d) and NAME_RE.match(cell)):
                    continue
                for f in ("results.csv", "args.yaml", "perclass_ap.json"):
                    fp = os.path.join(d, f)
                    if os.path.isfile(fp):
                        t.add(fp, arcname=os.path.join(runsdir, cell, f))
                        n_csv += (f == "results.csv")
                        n_args += (f == "args.yaml")
                        n_json += (f == "perclass_ap.json")
        for sub in (tabledir, "runs_deploy"):
            sd = os.path.join(root, sub)
            if not os.path.isdir(sd):
                continue
            for f in sorted(os.listdir(sd)):
                fp = os.path.join(sd, f)
                if os.path.isfile(fp) and f.endswith((".tsv", ".txt", ".json")):
                    t.add(fp, arcname=os.path.join(sub, f))
    return tar_path, {"results.csv": n_csv, "args.yaml": n_args, "perclass_ap.json": n_json}


def show_tail(path, n):
    if not path or not os.path.isfile(path):
        return ["(没有这个文件: %s)" % (path or "?")]
    ls = [x.rstrip() for x in open(path, encoding="utf-8", errors="replace") if x.strip()]
    if len(ls) <= n:
        return ls
    return ls[:n] + ["...（共 %d 行，完整见 tar.gz）" % len(ls)]


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    outdir = sys.argv[2] if len(sys.argv) > 2 else root
    runsdir = sys.argv[3] if len(sys.argv) > 3 else "runs"   # 主矩阵 runs；nano 对照 runs_nano
    # 表格目录与结果目录绑定：收 nano 不许覆盖主矩阵的 tables/
    if len(sys.argv) > 4:
        tabledir = sys.argv[4]
    elif runsdir == "runs":
        tabledir = "tables"
    else:
        tabledir = "tables_" + (runsdir[5:] if runsdir.startswith("runs_") else runsdir)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    info = scan_runs(root, runsdir)
    tables_dir = os.path.join(root, tabledir)
    dep = build_deploy_table(root, tables_dir)
    tar_path, counts = pack(root, outdir, runsdir, tabledir)

    print("=".ljust(74, "="))
    print("核表小包  " + time.strftime("%Y-%m-%d %H:%M:%S") + "   根目录: " + os.path.abspath(root)
          + "   结果子目录: " + runsdir + "   表目录: " + tabledir)
    print("=".ljust(74, "="))
    print("[1] runs/: 有效 run 目录 %d 个（%d 个 cell）| best.pt %d | perclass_ap.json %d | 带 test 数 %d"
          % (info["dirs"], info["cells"], len(info["bestpt"]), len(info["json"]), len(info["test"])))
    print("    数据集: %s | 类别数: %s   <-- 先核对是不是这次要收的那个数据集"
          % (", ".join(sorted(info["data_yaml"])) or "?", ", ".join(str(x) for x in sorted(info["nclass"])) or "?"))
    miss_t = [n for n in info["json"] if n not in info["test"]]
    miss_b = [n for n in info["json"] if n not in info["bestpt"]]
    print("    缺 test 的 run: " + ("(无)" if not miss_t else (" ".join(miss_t) if len(miss_t) <= 12
                                          else "%d 个（列前 12：%s ...）" % (len(miss_t), " ".join(miss_t[:12])))))
    print("    缺 best.pt 的 run: " + ("(无)" if not miss_b else " ".join(miss_b)))
    print("    缺 results.csv 的 run: " + ("(无)" if not info["csv"] or len(info["csv"]) == info["dirs"]
                                       else " ".join(n for n in info["json"] if n not in info["csv"])))
    print("[2] 小包: %s  (%.2f MB)" % (tar_path, os.path.getsize(tar_path) / 1048576.0))
    print("    包内: results.csv %d / args.yaml %d / perclass_ap.json %d + tables/ + runs_deploy/"
          % (counts["results.csv"], counts["args.yaml"], counts["perclass_ap.json"]))
    if dep is None:
        print("    (runs_deploy/ 里没有可聚合的 json，部署表未生成)")

    print()
    print("COPY_BEGIN")
    print("数据集: %s（%s 类）" % (", ".join(sorted(info["data_yaml"])) or "?",
          ", ".join(str(x) for x in sorted(info["nclass"])) or "?"))
    print("runs 概况: 有效目录 %d | best.pt %d | json %d | 带 test %d | 缺 test %d"
          % (info["dirs"], len(info["bestpt"]), len(info["json"]), len(info["test"]), len(miss_t)))
    print("---- tables/table_main.tsv ----")
    for x in show_tail(os.path.join(tables_dir, "table_main.tsv"), 20):
        print(x)
    print("---- tables/integrity.txt（只看前 6 行：完整性结论）----")
    for x in show_tail(os.path.join(tables_dir, "integrity.txt"), 6):
        print(x)
    print("---- tables/table_deploy.tsv ----")
    dep_lines = show_tail(dep, 20) if dep else ["(没有 runs_deploy/ 下的 json，部署表未生成)"]
    for x in dep_lines:
        print(x)
    print("COPY_END")
    print("=".ljust(74, "="))


main()
